import os
import gc
import math
import os.path as path
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


import argparse


def run_svr_inference(model, model_mlp, args=None, init_stacks_input=None, downsampled_input_tensor=None, output_dir=None):
    # SET RUNNING PARAMETERS
    seed_num = 1 #2
    seed_everything(seed_num, workers=True)
    save_images = True #False #True #True
    subsample=1
    img_num = 3 # 0 # which image in dataset want to test
    tot_test = 1 # CHANGE THIS TO 3 when doing actual tests
    slice_size = 128
    mlp_test = False

    # 0 # which image in dataset want to test
    imgnum_test = 0

    sets = datasets.feta3d0_mlp_multi_stack_svr_final_sb2_crop(subsample=subsample, zooms=0.3, mlp_training= False)

    # SET initial PARAMETERS
    avg_loss = 0
    flip_row_12 = np.array([[-1, 0, 0, 0], # SAVE IN SAME COORDINATE SYSTEM AS
                            [0, 0, 1, 0],
                            [0, -1, 0, 0],
                            [0,  0, 0, 1]])


    for imgnum in range(img_num, img_num+tot_test): #range(len(sets[1])):
        with torch.no_grad():
            # # Get original dataset before transforms
            # true, segs, _ = sets[1].__getitem__(imgnum, gpu=False)  # sets[1] gets validation set
            # true = true.cuda()  # torch.Size([1, 256, 256, 256]), true.min = 0  true.max = 1490.2251
            # segs = segs.cuda() #torch.Size([1, 256, 256, 256])
            # item = sets[1].transforms(true, segs, cpu=False, gpu=True) # this is where the augementation happens
            # if not mlp_test:
            #   mask = item[1][None][:,-1:]

            #   target = item[1][None,:,:,::2,::2]

            #   target[:,0:3] = target[:,0:3]*0.5
            # else :
            #   target = item[1]

            # init_stacks = item[0][1].clone()
            # init_stacks[:,:,0:3,3] = init_stacks[:,:,0:3,3]/2


            # downsampled_input = item[0][0][None,:,:,::2,::2]


            # print("IMG NUM test")
            # print(imgnum_test)
            # imgnum_test = imgnum_test + 1

            if init_stacks_input is not None and downsampled_input_tensor is not None:
                 init_stacks = init_stacks_input.cuda()
                 downsampled_input = downsampled_input_tensor.cuda()
            else:
                 init_stacks = torch.load(args.init_stack_template ).cuda() #[:,:-1]
                 downsampled_input = torch.load(args.input_template ).cuda() #[:,:,:-1]





         #   stack = model((downsampled_input,init_stacks))
            print("Running MLP...")

            print(downsampled_input.shape,init_stacks.shape )
            for i in range (3):
                _ = model_mlp((torch.zeros_like(downsampled_input,device=init_stacks.device), init_stacks))
                
            torch.cuda.synchronize()
            start = time.time()
            stack_mlp = model_mlp((downsampled_input,init_stacks))

            torch.cuda.synchronize()
            end = time.time()
            time_mlp = end - start
            print(f"MLP Inference time opt: {time_mlp:.6f} seconds")


            stack1, stack2 ,stack3 = divide_into_stacks(init_stacks, downsampled_input)
          #  stack = model_mlp((downsampled_input,init_stacks))

            rot_90_pred = False
            stack_pred = stack_mlp[:,0:6]
            order_reverse = stack_pred.argmax(dim=1)  % 2

            if(rot_90_pred):
                rot_pred = stack_mlp[:,6:]
                rot90_amount = rot_pred.argmax(dim=1)  % 4



            stacks = [stack1, stack2, stack3]

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


            stack1, stack2, stack3 = stacks
            downsampled_input2 = torch.cat([stack1, stack2, stack3], dim=2)


            # warmp up 
            for i in range (3):
                _ = model((torch.zeros_like(downsampled_input2),init_stacks))
          # #  pdb.set_trace()
            torch.cuda.synchronize()
            start_model = time.time()
            stack = model((downsampled_input2,init_stacks))
        #    pdb.set_trace() #models.losses.classification_onehot_loss(stack, torch.tensor([[0,1,0]]).cuda())
            torch.cuda.synchronize()
            end_model = time.time()
            time_unet = end_model - start_model
            print(f"Inference time opt {time_unet:.6f} seconds")
            print("POSE ESTIMATION:{:.3f}".format(time_unet   + time_mlp))

            ALL_STACKS =  init_stacks[1]
            ALL_STACKS_no_ot =  init_stacks[0]

            splat = model.unet3.splat.apply_flow_thin( downsampled_input2[:,:1], stack, ALL_STACKS, mask= downsampled_input[:,1:],volume_shape= [slice_size,slice_size,slice_size], slice_dim = [1,1,1], vol_dim=1, flow_dim=1) #item[0][None].shape[-3:])
            splat = splat[:,:-1] / (splat[:,-1:] + 1e-12 * splat[:,-1:].max().item()) # normalize


            psf_vals = torch.ones((1,2))*0.5

            psf_vals = psf_vals.cuda()
            psf_coords = torch.zeros((3,2))
            psf_coords[0,0] = 0
            psf_coords[0,1] = 1
            psf_coords = psf_coords.cuda()
  
            psf=(psf_vals, psf_coords)



            motion_2_3, aff4 = model.project_new_feb4_crop(stack[:,0:3],downsampled_input2[:,1:], ALL_STACKS, slice_in=0, spacing=1, shape=[stack.shape[2],slice_size,slice_size])

            splat_thick = model.unet3.splat.apply_flow_thick(downsampled_input2[:,:1], aff4, ALL_STACKS, mask= downsampled_input[:,1:],volume_shape= [128,128,128], slice_dim = [1,1,1], vol_dim=1, flow_dim=1, psf=(psf_vals, psf_coords)) #item[0][None].shape[-3:])
            splat_thick = splat_thick[:,:-1] / (splat_thick[:,-1:] + 1e-12 * splat_thick[:,-1:].max().item()) # normalize



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

        
                slice_saver.save_slices(downsampled_input2, affs,args.save_folder, current_output_dir,  args.clin, imgnum, args.slice_res)
                print("slice_saver done.")

def main(args=None, init_stacks_input=None, downsampled_input_tensor=None, output_dir=None):
    # TO DO
    # make it use actual image dimension

    # python process_nifti.py /data/vision/polina/users/mfirenze/svr_my_train_2024/data_sampled/validation/MAP-C436 --suffix "_auto5" --output-dir /data/vision/polina/users/mfirenze/cSVR/data_preprocess --run-cSVR --clin "True" --save-slices
    # nesvor svr --input-slices --output-volume MAP_C436_auto3_svr1.nii.gz

    if args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--init_stack_template', type=str, default='/data/vision/polina/users/mfirenze/svr_my_train_2024/data_sampled/init_stack_%s_auto2.pt')
        parser.add_argument('--input_template', type=str, default='/data/vision/polina/users/mfirenze/svr_my_train_2024/data_sampled/%s_auto2.pt')
        parser.add_argument('--save-slices', action='store_true', help='run slice_saver')
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
