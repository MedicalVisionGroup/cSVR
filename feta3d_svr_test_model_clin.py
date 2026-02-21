import os
import gc
import math
import os.path as path
import torch 
import os
torch.cuda.synchronize()
import interpol
import models
import models.losses
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
import sys


# CHECK GPU
if torch.cuda.is_available():
    print("CUDA AVAILABLE")
else:
    print("no CUDA")


# SET RUNNING PARAMETERS
seed_num = 1 #2
seed_everything(seed_num, workers=True)
save_images = True #False #True #True
subsample=1
img_num = 3 # 0 # which image in dataset want to test
tot_test = 1 # CHANGE THIS TO 3 when doing actual tests


# 0 # which image in dataset want to test
imgnum_test = 0

# LOAD MODEL
folder = 'run5'
model1024 = True
model256 = False
model512 = False
mlp_test = False
slice_size = 128
ckpt_path_recon = "feta3d0_multi_stack_svr_final_sb2_crop_flow_SNet3d2_1024_multi_crop_l22_loss_grid_multiscale_rot40_nodes100_bigger_model_v2_out_plane12_augs_v2_"
#ckpt_path_recon = "feta3d0_mlp_multi_stack_svr_final_sb2_crop_flow_SNet3d2_1024_MLP_classification_onehot_loss_40_0.03_0.1_70_12_argparse_test"
root_path_mine = '/data/vision/polina/users/mfirenze/svr_my_train_2024/checkpoints/'
#root_path_mine = '/data/vision/polina/users/mfirenze/cSVR/checkpoints/'

model_name = ckpt_path_recon[-15:]
print(model_name)#
trainee = models.segment(model=models.flow_SNet3d2_512_multi_crop())
if model1024:
   trainee = models.segment(model=models.flow_SNet3d2_1024_multi_crop())
   if(mlp_test):
     trainee = models.segment(model=models.flow_SNet3d2_1024_MLP())
     print("loaded mlp test model")
elif model512:
  trainee = models.segment(model=models.flow_SNet3d2_512_multi_crop())
elif model256:
   trainee = models.segment(model=models.flow_SNet3d2_256_multi_crop())

print("")
trainee.load_state_dict(torch.load(path.join(root_path_mine, ckpt_path_recon, 'best.ckpt'))['state_dict'])
model = trainee.model.cuda()
#sets = datasets.feta3d0_multi_stack_svr_final_sb2_crop(subsample=subsample, zooms=0.3)

sets = datasets.feta3d0_mlp_multi_stack_svr_final_sb2_crop(subsample=subsample, zooms=0.3, mlp_training= False)

# SET initial PARAMETERS
avg_loss = 0
flip_row_12 = np.array([[-1, 0, 0, 0], # SAVE IN SAME COORDINATE SYSTEM AS 
                        [0, 0, 1, 0],
                        [0, -1, 0, 0],
                        [0,  0, 0, 1]])
loss_2 = 0


for imgnum in range(img_num, img_num+tot_test): #range(len(sets[1])):
    with torch.no_grad():
        # Get original dataset before transforms
        true, segs, _ = sets[1].__getitem__(imgnum, gpu=False)  # sets[1] gets validation set
        true = true.cuda()  # torch.Size([1, 256, 256, 256]), true.min = 0  true.max = 1490.2251
        segs = segs.cuda() #torch.Size([1, 256, 256, 256])
        item = sets[1].transforms(true, segs, cpu=False, gpu=True) # this is where the augementation happens
        if not mlp_test:
          mask = item[1][None][:,-1:]
          
          target = item[1][None,:,:,::2,::2]
          
          target[:,0:3] = target[:,0:3]*0.5 
        else :  
          target = item[1]

        init_stacks = item[0][1].clone()
        init_stacks[:,:,0:3,3] = init_stacks[:,:,0:3,3]/2

        start = time.time()
        downsampled_input = item[0][0][None,:,:,::2,::2]


        if(imgnum >= 5):
           line = open("/data/vision/polina/users/mfirenze/svr_my_train_2024/data_sampled/test/test.txt").read().splitlines()[imgnum_test]
           downsampled_input = torch.load('/data/vision/polina/users/mfirenze/svr_my_train_2024/data_sampled/%s_rot_cm_t.pt' % line).to(item[0][0].device) #[:,:,:-1]
           init_stacks = torch.load('/data/vision/polina/users/mfirenze/svr_my_train_2024/data_sampled/init_stack_%s_rot_cm_t.pt' % line).to(item[0][0].device) #[:,:-1]
           
         #  init_stacks = torch.load('/data/vision/polina/users/mfirenze/svr_my_train_2024/data_sampled/init_stack_%s_no_rot.pt' % line).to(item[0][0].device) #[:,:-1]
         #  downsampled_input = torch.load('/data/vision/polina/users/mfirenze/svr_my_train_2024/data_sampled/%s_no_rot.pt' % line).to(item[0][0].device) #[:,:,:-1]
           
           
           imgnum_test = imgnum_test + 1


        else:
          line = open("/data/vision/polina/users/mfirenze/svr_my_train_2024/data_sampled/validation/val.txt").read().splitlines()[imgnum]
          init_stacks = torch.load('/data/vision/polina/users/mfirenze/svr_my_train_2024/data_sampled/init_stack_%s_rot_cm.pt' % line).to(item[0][0].device) #[:,:-1]
          downsampled_input = torch.load('/data/vision/polina/users/mfirenze/svr_my_train_2024/data_sampled/%s_rot_cm.pt' % line).to(item[0][0].device) #[:,:,:-1]

        print("IMG NUM test")
        print(imgnum_test)
        imgnum_test = imgnum_test + 1
        init_stacks = torch.load('/data/vision/polina/users/mfirenze/svr_my_train_2024/data_sampled/init_stack_%s_auto2.pt' % line).to(item[0][0].device) #[:,:-1]
        downsampled_input = torch.load('/data/vision/polina/users/mfirenze/svr_my_train_2024/data_sampled/%s_auto2.pt' % line).to(item[0][0].device) #[:,:,:-1]


        stack = model((downsampled_input,init_stacks))
        end = time.time()
        print(f"Inference time 1: {end - start:.6f} seconds")

        torch.cuda.synchronize()  
        start = time.time()
        stack = model((downsampled_input,init_stacks))
        torch.cuda.synchronize()  
        end = time.time()
        print(f"Inference time 2: {end - start:.6f} seconds")

        start = time.time()
        print("Model gets all black")
        stack = model((downsampled_input,init_stacks))
        end = time.time()

        print(f"Inference time 3: {end - start:.6f} seconds")
        start = time.time()
        print("Model gets all black")





      # #  pdb.set_trace()
        stack = model((downsampled_input,init_stacks))
    #    pdb.set_trace() #models.losses.classification_onehot_loss(stack, torch.tensor([[0,1,0]]).cuda())
        end = time.time()
        print(f"Inference time 4 {end - start:.6f} seconds")


        ALL_STACKS =  init_stacks[1]
        ALL_STACKS_no_ot =  init_stacks[0]

        splat = model.unet3.splat.forward_retooled_one( downsampled_input[:,:1], stack, ALL_STACKS, mask= downsampled_input[:,1:],volume_shape= [slice_size,slice_size,slice_size], slice_dim = [1,1,1], vol_dim=1, flow_dim=1) #item[0][None].shape[-3:])
        splat = splat[:,:-1] / (splat[:,-1:] + 1e-12 * splat[:,-1:].max().item()) # normalize

      
        psf_vals = torch.ones((1,2))*0.5

        psf_vals = psf_vals.to(device=target.device)
        psf_coords = torch.zeros((3,2))
        psf_coords[0,0] = 0
        psf_coords[0,1] = 1
        psf_coords = psf_coords.to(device=target.device)
        print("new psf!")
        psf=(psf_vals, psf_coords)
          
                      

        motion_2_3, aff4 = model.project_new_feb4_crop(stack[:,0:3],downsampled_input[:,1:], ALL_STACKS, slice_in=0, spacing=1, shape=[stack.shape[2],slice_size,slice_size])
      
        splat_thick = model.unet3.splat.apply_flow_thick(downsampled_input[:,:1], aff4, ALL_STACKS, mask= downsampled_input[:,1:],volume_shape= [128,128,128], slice_dim = [1,1,1], vol_dim=1, flow_dim=1, psf=(psf_vals, psf_coords)) #item[0][None].shape[-3:])
        splat_thick = splat_thick[:,:-1] / (splat_thick[:,-1:] + 1e-12 * splat_thick[:,-1:].max().item()) # normalize
       
        loss_2 = models.losses.l22_loss_grid(stack, target, eps=0).item()
        print("overall loss")
        print(loss_2)

        if (save_images==True ) :
              imgnames = ['input','input_ds','splat','splat_thick']
              #imgnames = ['input','input_ds','splat','splat_gt','target_before','target_up','target_all','splat_inter']
              imgs = [item[0][0][None,:,:,:,:][0][0].detach(),item[0][0][None,:,:,::2,::2][0][0].detach(), splat[0,0].detach(),  splat_thick[0,0].detach()]

            #  imgs = [item[0][0][None,:,:,:,:][0][0].detach(),item[0][0][None,:,:,::2,::2][0][0].detach(), splat[0,0].detach(), splat_gt[0,0].detach(),  target[0,2].detach(), target_up[0,2].detach(),target_all[0,2].detach(),splat_inter[0,0].detach()]



        if save_images:
            imgs = [img.cpu() for img in imgs]
            for i in range(len(imgs)):
                initial_np = imgs[i].numpy()                
               # nii_image = nib.Nifti1Image(initial_np, affine=flip_row_12*0.8)  # You might need to specify the affine transformation matrix
                I = np.eye(4)
                I[0:3,0:3] = I[0:3,0:3]*1.406
                nii_image = nib.Nifti1Image(initial_np, affine=I)  
              #  nii_image = nib.Nifti1Image(initial_np, affine=np.eye(4))  # You might need to specify the affine transformation matrix
                nib.save(nii_image, '/data/vision/polina/users/mfirenze/cSVR/outputs/vol_%d_seed_%d_model_%s_%s_no_rot.nii.gz' % (imgnum, seed_num, model_name, imgnames[i]))
                print("DONE 1:D")
