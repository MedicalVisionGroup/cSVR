import os
import gc
import math
import os.path as path
from xml.parsers.expat import model
import torch 
import os
torch.cuda.synchronize()
import interpol
import models
import datasets
from torchvision.utils import save_image 
from torch.utils.data import DataLoader
import cornucopia as cc
from matplotlib import pyplot as plt
import pdb
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D
import numpy as np
from pytorch_lightning import seed_everything
import sys
import cornucopia 
from cornucopia.utils.warps import affine_flow
from grid_utils import og_slice_pos_pre, make_grid_one
from grid_utils import og_slice_pos_pre, make_grid_one, divide_into_stacks
import sys
import sys 
import slice_saver
from nifti_utils import make_init_stacks, normalize_by_second_mode
import argparse


# The models are loaded once and reused across all subjects. The first forward pass
# triggers one-time GPU init (CUDA context, cuDNN/cuSOLVER, custom CUDA extension JIT)
# that costs ~15 s. We warm up only once so only one subject pays it; every subsequent
# call skips the warmup and runs the real forward directly.
_WARMED_UP = False

# Pose-estimation time for the subject that ran most recently, in seconds: MLP ordering
# + SVR net forward + the pose post-processing (splat_thin, project = flow -> slice
# affines, splat_thick). Counting the post-forward steps mirrors SVoRT's POSE ESTIMATION
# span, which also includes its post-forward correction/scoring/selection. Set by every
# run_svr_inference call, so a caller can read it back instead of scraping the
# "POSE ESTIMATION:" line out of stdout.
LAST_POSE_ESTIMATION_S = None


def warmup(model, model_mlp, init_stacks_input, downsampled_input_tensor):
    """Trigger the one-time GPU init once, up front, so its ~15 s cost is not counted in
    any subject's timed total. Safe to call before every subject: it self-gates on
    _WARMED_UP and returns immediately once warmed. Runs zero-input forwards on the given
    subject's tensors and discards the output, so it has no effect on results."""
    global _WARMED_UP
    if _WARMED_UP:
        return
    try:
        torch.cuda.synchronize()
        _t0 = time.time()
        with torch.no_grad():
            init_stacks = init_stacks_input.cuda()
            downsampled_input = downsampled_input_tensor.cuda()
            if model_mlp is not None:
                for _ in range(3):
                    _ = model_mlp((torch.zeros_like(downsampled_input, device=init_stacks.device), init_stacks))
            for _ in range(3):
                _ = model((torch.zeros_like(downsampled_input), init_stacks))
        torch.cuda.synchronize()
        _WARMED_UP = True
        print(f"[TIMING] warmup (pre-subject, one-time): {time.time() - _t0:.2f} s")
    except Exception as e:
        # Leave _WARMED_UP False so the gated warmup inside run_svr_inference still runs.
        print(f"Pre-subject warmup skipped ({e}); will warm up during first inference instead.")


def run_svr_inference(model, model_mlp, args=None, init_stacks_input=None, downsampled_input_tensor=None, output_dir=None, downsampled_input_tensor_hres=None):
    # SET RUNNING PARAMETERS
    global _WARMED_UP, LAST_POSE_ESTIMATION_S

    # Debug volumes (original_slices / slices_reoriented / splat / splat_proj) written
    # next to the subject. args.save_debug_images=False turns them off for timing runs,
    # where the write is cost that has nothing to do with the pipeline being measured.
    save_images = getattr(args, "save_debug_images", True)
    slice_size = 128
    imgnum = 0
    output_slices = None


    with torch.no_grad():
        torch.cuda.synchronize()
        _t0 = time.time()
        if init_stacks_input is not None and downsampled_input_tensor is not None:
                init_stacks = init_stacks_input.cuda()
                downsampled_input = downsampled_input_tensor.cuda()
        else:
                init_stacks = torch.load(args.init_stack_template ).cuda() #[:,:-1]
                downsampled_input = torch.load(args.input_template ).cuda() #[:,:,:-1]


        stack1, stack2 ,stack3, allstacks1, allstacks2, allstacks3 = divide_into_stacks(init_stacks, downsampled_input, stacks_split=True)
        rot_90_pred = False
        torch.cuda.synchronize(); print(f"[TIMING]   setup_load: {time.time()-_t0:.2f} s"); _t0 = time.time()

        if model_mlp is not None:
            print("Running MLP...")
            print(downsampled_input.shape,init_stacks.shape )
            if not _WARMED_UP:
                for i in range (3):
                    _ = model_mlp((torch.zeros_like(downsampled_input,device=init_stacks.device), init_stacks))

            torch.cuda.synchronize(); print(f"[TIMING]   mlp_warmup(3x): {time.time()-_t0:.2f} s"); _t0 = time.time()
            torch.cuda.synchronize()
            start = time.time()
            # The ordering MLP was trained on per-stack (second-mode) normalized intensities,
            # but the SVR network and the reconstruction consume the stacks as preprocessed.
            # So normalize a COPY of each stack for the MLP call only; downsampled_input /
            # stack1..3 stay untouched downstream.
            mlp_input = downsampled_input.clone()
            _b0 = 0
            for _st in (stack1, stack2, stack3):
                _n = _st.shape[2]
                _img = mlp_input[0, 0, _b0:_b0+_n].cpu().numpy()
                _msk = (mlp_input[0, 1, _b0:_b0+_n] > 0.5).cpu().numpy()
                if _msk.any():
                    _img = normalize_by_second_mode(_img, _msk, verbose=False)
                    mlp_input[0, 0, _b0:_b0+_n] = torch.as_tensor(
                        _img, dtype=mlp_input.dtype, device=mlp_input.device)
                _b0 += _n
            stack_mlp = model_mlp((mlp_input,init_stacks))


            torch.cuda.synchronize()
            end = time.time()
            time_mlp = end - start
            print(f"MLP Inference time opt: {time_mlp:.6f} seconds")

            stack_pred = stack_mlp[:,0:6]

            order_reverse = stack_pred.argmax(dim=1)  % 2
            stack_order = stack_pred.argmax(dim=1)  //2
            print("MLP output:")
            print(stack_pred)

            print(order_reverse)
            print(stack_order)
            print(f"STACK ORDER argmax (unconstrained): {[int(c) for c in stack_order]}")

            # Assign stacks from the MLP's probabilities, constrained to a permutation: the
            # per-stack argmax alone can give two stacks the same orientation. Classes are
            # [sag, sag_rev, cor, cor_rev, axi, axi_rev]; per stack take each orientation's
            # better reverse variant, then pick the permutation with the highest total
            # log-probability (3! = 6, brute force). The input order of the stacks (i.e.
            # their filenames) therefore never decides the assignment.
            probs = torch.softmax(stack_pred, dim=1).view(3, 3, 2)  # [stack, orient, rev]
            P = probs.max(dim=2).values                              # [3, 3]
            perms = [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]
            best_perm = max(perms, key=lambda pm: sum(float(torch.log(P[i, pm[i]] + 1e-9)) for i in range(3)))
            stack_order = torch.tensor(best_perm, device=stack_pred.device)
            order_reverse = torch.tensor([int(probs[i, best_perm[i], 1] > probs[i, best_perm[i], 0]) for i in range(3)],
                                         device=stack_pred.device)

            if getattr(args, "no_mlp", False):
                # --no-mlp: the stacks arrive as (sag, cor, axi) from their filenames, so
                # input stack i has orientation i. Keep that order and take only the 180°
                # in-plane decision from the MLP, read off for the known orientation.
                print(f"--no-mlp: stack order taken from the filenames (sag, cor, axi); "
                      f"MLP proposal {list(best_perm)} ignored, MLP used for the 180° flips only.")
                stack_order = torch.tensor([0, 1, 2], device=stack_pred.device)
                order_reverse = torch.tensor([int(probs[i, i, 1] > probs[i, i, 0]) for i in range(3)],
                                             device=stack_pred.device)

            if(rot_90_pred):
                rot_pred = stack_mlp[:,6:]
                rot90_amount = rot_pred.argmax(dim=1)  % 4
        else:
            print("No MLP model given: identity stack order and default flips.")
            time_mlp = 0.0
            order_reverse = torch.tensor([0, 1, 0], device=stack1.device)  # Default order (no reversal)
            stack_order = torch.tensor([0, 1, 2], device=stack1.device)    # identity permutation


        stacks = [stack1, stack2, stack3]
        all_stacks = [allstacks1, allstacks2, allstacks3]

        for i in range(3):
    #  stacks[i] = stacks[i].flip(dims=[2]) # make this 18-
            if(rot_90_pred):
                if rot90_amount[i] == 1:
                    stacks[i] = torch.rot90(stacks[i], k=-1, dims=(3,4))
                if rot90_amount[i] == 2:
                    stacks[i] = torch.rot90(stacks[i], k=1, dims=(3,4))
                if rot90_amount[i] == 3:
                    stacks[i] = torch.rot90(stacks[i], k=2, dims=(3,4))

            if order_reverse[i] == 1:
                stacks[i] = torch.rot90(stacks[i], k=2, dims=(2,3))


        
        print(f"ORDER REVERSE: {[int(r) for r in order_reverse]}")

        # reshuffle so that position k is filled by input stack i for which order[i] == k.
        # e.g. order=[0,1,2] -> [stack1, stack2, stack3]; order=[2,0,1] -> [stack2, stack3, stack1]
        order = [int(stack_order[i].item()) for i in range(3)]
        print(f"STACK ORDER: {order}")
        # Safety net only: the constrained decode above always yields a permutation. A
        # non-permutation (e.g. [2,2,0]) would leave an unfilled (None) slot in out_stacks
        # and crash torch.cat, so fall back to identity order in that case.
        if sorted(order) != [0, 1, 2]:
            print(f"WARNING: MLP predicted a non-permutation stack order {order}; "
                  f"falling back to identity order [0, 1, 2] (stacks assumed already sag/cor/axi).")
            order = [0, 1, 2]
        out_stacks = [None, None, None]
        for i in range(3):
            out_stacks[order[i]] = stacks[i]
        downsampled_input2 = torch.cat(out_stacks, dim=2)
        if order != [0, 1, 2]:
            print(f"Reshuffling init_stacks to match order {order}")

            # Rebuild init_stacks for the reshuffled order via make_init_stacks.
            # Input stack i (slice dir i, dims[i] slices) moves to position order[i], so
            # position k holds slice dir inv[k] with dims[inv[k]] slices, where inv = order^{-1}.
            dims = [all_stacks[i].shape[1] for i in range(3)]
            inv = [0, 0, 0]
            for i in range(3):
                inv[order[i]] = i
            reordered_dims = [dims[inv[k]] for k in range(3)]
         #   init_stacks = make_init_stacks(reordered_dims, init_stacks.device, stack_order=tuple(inv))
            init_stacks = make_init_stacks(reordered_dims, init_stacks.device)

        # warmp up (only on the first subject; see _WARMED_UP note above)
        torch.cuda.synchronize(); _t0 = time.time()
        if not _WARMED_UP:
            for i in range (3):
                _ = model((torch.zeros_like(downsampled_input2),init_stacks))
        # #  pdb.set_trace()
        torch.cuda.synchronize(); print(f"[TIMING]   model_warmup(3x): {time.time()-_t0:.2f} s")
        _WARMED_UP = True
        torch.cuda.synchronize()
        start_model = time.time()

        stack = model((downsampled_input2,init_stacks))
    #    pdb.set_trace() #models.losses.classification_onehot_loss(stack, torch.tensor([[0,1,0]]).cuda())
        torch.cuda.synchronize()
        end_model = time.time()
        time_unet = end_model - start_model
        print(f"Inference time opt {time_unet:.6f} seconds")
        torch.cuda.synchronize()
        _t_step = time.time()

        ALL_STACKS =  init_stacks[1]
        ALL_STACKS_no_ot =  init_stacks[0]
    


        if(type(stack) == list):
            stack = stack[-1]
        splat = model.unet3.splat.apply_flow_thin( downsampled_input2[:,:1], stack, ALL_STACKS, mask= downsampled_input2[:,1:],volume_shape= [slice_size,slice_size,slice_size], slice_dim = [1,1,1], vol_dim=1, flow_dim=1) #item[0][None].shape[-3:])
        splat = splat[:,:-1] / (splat[:,-1:] + 1e-12 * splat[:,-1:].max().item()) # normalize
        torch.cuda.synchronize(); time_splat_thin = time.time()-_t_step; print(f"[TIMING]   splat_thin: {time_splat_thin:.2f} s"); _t_step = time.time()


        psf_vals = torch.ones((1,2))*0.5

        psf_vals = psf_vals.cuda()
        psf_coords = torch.zeros((3,2))
        psf_coords[0,0] = 0
        psf_coords[0,1] = 1
        psf_coords = psf_coords.cuda()

        psf=(psf_vals, psf_coords)



        motion_2_3, aff4 = model.project_new_feb4_crop(stack[:,0:3],downsampled_input2[:,1:], ALL_STACKS, slice_in=0, spacing=1, shape=[stack.shape[2],slice_size,slice_size])
        torch.cuda.synchronize(); time_project = time.time()-_t_step; print(f"[TIMING]   project: {time_project:.2f} s"); _t_step = time.time()

        splat_thick = model.unet3.splat.apply_flow_thick(downsampled_input2[:,:1], aff4, ALL_STACKS, mask= downsampled_input[:,1:],volume_shape= [128,128,128], slice_dim = [1,1,1], vol_dim=1, flow_dim=1, psf=(psf_vals, psf_coords)) #item[0][None].shape[-3:])
        splat_thick = splat_thick[:,:-1] / (splat_thick[:,-1:] + 1e-12 * splat_thick[:,-1:].max().item()) # normalize
        torch.cuda.synchronize(); time_splat_thick = time.time()-_t_step; _t_step = time.time()
        print(f"[TIMING]   splat_thick: {time_splat_thick:.2f} s")
        LAST_POSE_ESTIMATION_S = time_mlp + time_unet + time_splat_thin + time_project + time_splat_thick
        print("POSE ESTIMATION:{:.3f}".format(LAST_POSE_ESTIMATION_S))



        if (save_images==True ) :
                imgnames = ['original_slices','slices_reoriented','splat','splat_proj']
                #imgnames = ['input','input_ds','splat','splat_gt','target_before','target_up','target_all','splat_inter']
                imgs = [downsampled_input[0][0].detach(),downsampled_input2[0][0].detach(), splat[0,0].detach(),  splat_thick[0,0].detach()]

            #  imgs = [item[0][0][None,:,:,:,:][0][0].detach(),item[0][0][None,:,:,::2,::2][0][0].detach(), splat[0,0].detach(), splat_gt[0,0].detach(),  target[0,2].detach(), target_up[0,2].detach(),target_all[0,2].detach(),splat_inter[0,0].detach()]



        if save_images:
            imgs = [img.cpu() for img in imgs]

            if args.input_template is not None:
                    current_output_dir = path.dirname(args.input_template)
                    input_basename = path.basename(args.input_template).replace('.pt', '')
            else:
                if output_dir is not None:
                    current_output_dir = output_dir
                else:
                    current_output_dir = args.save_folder if args.save_folder else "."
                input_basename = args.suffix if hasattr(args, 'suffix') and args.suffix else "direct_input"

            if not path.exists(current_output_dir):
                os.makedirs(current_output_dir, exist_ok=True)

            for i in range(len(imgs)):
                initial_np = imgs[i].numpy()
                # nii_image = nib.Nifti1Image(initial_np, affine=flip_row_12*0.8)  # You might need to specify the affine transformation matrix
                I = np.eye(4)
                I[0:3,0:3] = I[0:3,0:3]*1.406
                nii_image = nib.Nifti1Image(initial_np, affine=I)
                #  nii_image = nib.Nifti1Image(initial_np, affine=np.eye(4))  # You might need to specify the affine transformation matrix
                
                # Ensure cSVR_files subdirectory exists
                csvr_files_dir = path.join(current_output_dir, 'cSVR_files')
                if not path.exists(csvr_files_dir):
                    os.makedirs(csvr_files_dir, exist_ok=True)
                    
                nib.save(nii_image, path.join(csvr_files_dir, '%s_%s.nii.gz' % (input_basename, imgnames[i])))
                

        torch.cuda.synchronize(); print(f"[TIMING]   save_debug_images: {time.time()-_t_step:.2f} s"); _t_step = time.time()
        if args.save_slices:
            print("Running slice_saver...")
            # affs expected format: stack of [og_aff, matrix_]
            # ALL_STACKS (init_stacks) seems to be the 'og_aff' equivalent (initial positions)
            # aff4 is the computed affine array

            # slice_saver expects: og_aff, matrix_ = affs.tensor_split(2,0)
            # So we need to concat them on dim 0.
            # ALL_STACKS shape: likely (N, 4, 4) or similar?
            # In project_new_feb4: grid = make_grid_one(ALL_STACKS, ...)
            # ALL_STACKS was init_stacks[1] earlier.

            # Need to ensure dimensions match for concatenation.
            # Assuming ALL_STACKS and aff4 are compatible (N, 4, 4).
            # motion_2_3, aff4 = model.project_new_feb4_crop(stack[:,0:3],downsampled_input2[:,1:], ALL_STACKS, slice_in=0, spacing=1, shape=[stack.shape[2],slice_size,slice_size])
            # motion_2_3, aff_all = model.project_new_feb4_crop(stack[:,0:3],item[0][0][None,:,:,::2,::2][:,1:], ALL_STACKS, slice_in=0, spacing=1, shape=[item[0][0].shape[1],item[0][0].shape[2],item[0][0].shape[3]])
            # aff4[:,0:3,3] = aff4[:,0:3,3]*2
            # ALL_STACKS[:,0:3,3] = ALL_STACKS[:,0:3,3]*2


            if(args.synth):
                ALL_STACKS[:,0:3,3] = ALL_STACKS[:,0:3,3]*2
                aff4[:,0:3,3] = aff4[:,0:3,3]*2
                print("multiplied affs")
                print(aff4.shape, ALL_STACKS.shape)
            affs = torch.cat([ALL_STACKS.unsqueeze(0), aff4.unsqueeze(0)], dim=0)

            # downsampled_input shape: (1, 1, H, W, D) ?
            # In loop: downsampled_input = item[0][0][None,:,:,::2,::2] -> (1, 1, ...)
            # slice_saver refactored code usage: imgs[0,0][n, :, :]
            # So passing downsampled_input as 'imgs' is correct if it has matching structure.

            if args.input_template is not None:
                current_output_dir = path.dirname(args.input_template)
                input_basename = path.basename(args.input_template).replace('.pt', '')
            else:
                if output_dir is not None:
                    current_output_dir = output_dir
                else:
                    current_output_dir = args.save_folder if args.save_folder else "."
                input_basename = "direct_input"

            if not path.exists(current_output_dir):
                os.makedirs(current_output_dir, exist_ok=True)

    
            print("Is hres input provided?", downsampled_input_tensor_hres is not None)
            imgs_for_saver = downsampled_input_tensor_hres.cuda() if downsampled_input_tensor_hres is not None else downsampled_input2
            output_slices = slice_saver.save_slices(imgs_for_saver, affs,args.save_folder, current_output_dir,  args.clin, imgnum, args.slice_res, save_to_disk=getattr(args, 'save_slices_to_disk', True))
            print(f"[TIMING]   slice_saver: {time.time()-_t_step:.2f} s"); _t_step = time.time()
            print("slice_saver done.")

    return output_slices

def main(args=None, init_stacks_input=None, downsampled_input_tensor=None, output_dir=None):
    # TO DO
    # make it use actual image dimension

    if args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--init_stack_template', type=str, default='/data/vision/polina/users/mfirenze/svr_my_train_2024/data_sampled/init_stack_%s_auto2.pt')
        parser.add_argument('--input_template', type=str, default='/data/vision/polina/users/mfirenze/svr_my_train_2024/data_sampled/%s_auto2.pt')
        parser.add_argument('--save-slices', action='store_true', help='run slice_saver')
        parser.add_argument('--no_save_slices_to_disk', dest='save_slices_to_disk', action='store_false', help='keep slices in memory instead of writing .nii.gz to disk')
        parser.set_defaults(save_slices_to_disk=True)
        parser.add_argument('--clin', action='store_true', help='clin spacing flag for slice_saver')
        parser.add_argument('--save_folder', type=str, default="saved_slices", help='Directory to save output slices')
        parser.add_argument('--slice_res', type=tuple, default=(1,1,1), help='slice resolution for slice_saver')
        parser.add_argument('--suffix', type=str, default=None, help='Suffix for input_basename')
        args = parser.parse_args()

    # CHECK GPU
    if torch.cuda.is_available():
        print("CUDA AVAILABLE")
    else:
        print("no CUDA")


    # SET RUNNING PARAMETERS
    seed_num = 1 #2
    #seed_everything(seed_num, workers=True) # Moved to run_svr_inference
    # ... other parameters moved to run_svr_inference ...

    # LOAD MODEL
    folder = 'run5'
    model1024 = True
    model256 = False
    model512 = False
    mlp_test = False
    slice_size = 128
    ckpt_path_recon = "feta3d0_multi_stack_svr_final_sb2_crop_flow_SNet3d2_1024_multi_crop_l22_loss_grid_multiscale_rot40_nodes100_bigger_model_v2_out_plane12_augs_v2_"
    ckpt_path_recon = "feta3d0_multi_stack_svr_final_sb2_crop_flow_SNet3d2_1024_multi_crop_l22_loss_grid_multiscale_40_0.03_0.1_70_12_lr0.0001__test_new_config_fix_lr"
    #ckpt_path_recon = "feta3d0_mlp_multi_stack_svr_final_sb2_crop_flow_SNet3d2_1024_MLP_classification_onehot_loss_40_0.03_0.1_70_12_argparse_test"
    root_path_mine = '/data/vision/polina/users/mfirenze/svr_my_train_2024/checkpoints/'
    root_path_mine = '/data/vision/polina/users/mfirenze/cSVR/checkpoints/'

    root_path_new = '/data/vision/polina/users/mfirenze/cSVR/checkpoints/'
    ckpt_path_mlp = "feta3d0_mlp_multi_stack_svr_final_sb2_crop_flow_SNet3d2_1024_MLP_classification_multihot_loss2_40_0.03_0.1_180_12_lr1e-05_mlp_norm_64size_vec_rots_loss_smooth_huge_mlp4_drop_0.2_augs"

    model_name = ckpt_path_recon[-15:]
    #(model_name)#
    trainee = models.segment(model=models.flow_SNet3d2_512_multi_crop())
    if model1024:
       start = time.time()
       trainee = models.segment(model=models.flow_SNet3d2_1024_multi_crop())
       end = time.time()
       print(f"Loading svr segment time {end - start:.6f} seconds")
       if(mlp_test):

         trainee = models.segment(model=models.flow_SNet3d2_1024_MLP())

    elif model512:
      trainee = models.segment(model=models.flow_SNet3d2_512_multi_crop())
    elif model256:
       trainee = models.segment(model=models.flow_SNet3d2_256_multi_crop())

    print("no load!")
    start = time.time()
    trainee.load_state_dict(torch.load(path.join(root_path_mine, ckpt_path_recon, 'best.ckpt'),map_location='cuda')['state_dict'])
    end = time.time()
    print(f"Loading SVR checkpoint time {end - start:.6f} seconds")
    model = trainee.model.cuda()

    start = time.time()
    trainee_mlp = models.segment(model=models.flow_SNet3d2_1024_MLP())
    end = time.time()
    print(f"Loading MLP segment time {end - start:.6f} seconds")

    start = time.time()
    trainee_mlp.load_state_dict(torch.load(path.join(root_path_new, ckpt_path_mlp, 'best.ckpt'), map_location='cuda')['state_dict'], strict = False)
    end = time.time()
    print(f"Loading MLP checkpoint time {end - start:.6f} seconds")
    model_mlp = trainee_mlp.model.cuda()


    #sets = datasets.feta3d0_multi_stack_svr_final_sb2_crop(subsample=subsample, zooms=0.3)

    #sets = datasets.feta3d0_mlp_multi_stack_svr_final_sb2_crop(subsample=subsample, zooms=0.3, mlp_training= False)

    run_svr_inference(model, model_mlp, args, init_stacks_input, downsampled_input_tensor, output_dir)

if __name__ == "__main__":
    main()
