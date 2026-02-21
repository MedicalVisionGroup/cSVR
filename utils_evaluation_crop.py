#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import gc
#import ants
import math
import os.path as path
# import utils
import torch
import models
import models.losses
import datasets
from torchvision.utils import save_image 
from torch.utils.data import DataLoader
import cornucopia as cc
from matplotlib import pyplot as plt
import pdb

import matplotlib.pyplot as plt
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D
import numpy as np
import argparse

from datetime import datetime
import torch
import models
import models.losses
import cornucopia as cc
#get_ipython().run_line_magic('matplotlib', 'inline')

from datasets.transforms import BoundingBox3d
from pytorch_lightning import seed_everything

import cornucopia 
from cornucopia.utils.warps import affine_flow

from grid_utils import og_slice_pos_pre, make_grid_one

if torch.cuda.is_available():
    print("CUDA AVAILABLE")
else:
    print("no CUDA")

import torch.nn.functional as F
import time
import torch
torch.cuda.empty_cache()




def run_model(full_path, img_num, gt, save_gt, folder_save="", subsample=1, hres=False, clin=True, **motion_params):


    print("GT TRUE")
    # Load dataset
    print("CLIN?")
    print(clin)
    print("CHANGED THE CODE DOES IT AFFECT ANYTHING????? ")
    if  hres and not clin:
        print("HRES")
        sets = datasets.feta3d0_multi_stack_svr_final_sb2_crop_hr(subsample=subsample)
        trainee = models.segment(model=models.flow_SNet3d2_512_multi_crop_hr_low_features())

    else:
        print("LOW RES")
        print("MOTION PARAMS")
        print(motion_params["rotations"],type(motion_params["rotations"]))
        sets = datasets.feta3d0_multi_stack_svr_final_sb2_crop(subsample=subsample, rotations = motion_params["rotations"], translations= motion_params["translations"], noise= motion_params["noise"], bulk_rotations_plane=70)
       # trainee = models.segment(model=models.flow_SNet3d2_128_multi_crop())
      #  trainee = models.segment(model=models.flow_SNet3d2_256_multi_crop())
      #  trainee = models.segment(model=models.flow_SNet3d2_512_multi_crop())
        trainee = models.segment(model=models.flow_SNet3d2_1024_multi_crop())

     #  trainee = models.segment(model=models.flow_SNet3d2_512_multi_crop())
    true, segs, _ = sets[1].__getitem__(img_num, gpu=False)  # sets[1] gets validation set
    true = true.cuda()  # torch.Size([1, 256, 256, 256]), true.min = 0  true.max = 1490.2251
    segs = segs.cuda() #torch.Size([1, 256, 256, 256])
    scales = datasets.transforms.ScaleZeroOne()

    if(gt and not clin):

        true_scaled, true_mask = scales(true, segs)

        # flip_row_12 = np.array([[-1, 0, 0, 0],
        #                 [0, 0, 1, 0],
        #                 [0, -1, 0, 0],
        #                 [0,  0, 0, 1]])
        
        matrix = np.diag([0.8, 0.8, 0.8, 1.0]) #@ flip_row_12

        
        save_mask = save_gt[0:-7]+"_mask.nii.gz"
        save_brain = save_gt[0:-7]+"_brain.nii.gz"
        nii_image = nib.Nifti1Image(true_scaled[0].detach().cpu().numpy(),matrix)  
        nib.save(nii_image , save_brain)

       # binary_mask = (true_mask[0] > 0).to(dtype=true_mask.dtype) 
        binary_mask = (true_scaled[0] > 0).to(dtype=true_mask.dtype) 
        nii_image = nib.Nifti1Image(binary_mask.detach().cpu().numpy(),matrix)  
        nib.save(nii_image , save_mask)
        




    # Dataset after transforms (with motion + repeated)
    if (not clin):
        item = sets[1].transforms(true, segs, cpu=False, gpu=True) 
        mask = item[1][None][:,-1:]

        slice_size =  int(item[0][0][None][:,:1].shape[3]/2)
        if(not hres):
            slice_size =  int(slice_size / 2)
    
 
    trainee.load_state_dict(torch.load(path.join(full_path,'last.ckpt'))['state_dict'])
    model = trainee.model.cuda()
    with torch.no_grad():
        if(not gt or clin):
            if hres:
                if not clin:
                    print("evaluate not clin")
                    torch.cuda.synchronize() 
                    start = time.time()
                    stack = model((item[0][0][None,:,:,:,:], item[0][1]))
                    torch.cuda.synchronize() 
                    end = time.time()
    
                    print("POSE ESTIMATION:{:.3f}".format(end - start))
                else:
                    #line = open("/data/vision/polina/users/mfirenze/svr_my_train_2024/data_sampled/validation/val.txt").read().splitlines()[img_num]
                    line = open("/data/vision/polina/users/mfirenze/svr_my_train_2024/data_sampled/validation/val_test.txt").read().splitlines()[img_num]

                    #downsampled_input = torch.load('./data_sampled/%s_rot.pt' % line).to(true.device)[0]
                    #downsampled_input = torch.load('./data_sampled/%s_rot.pt' % line).to(true.device)[0]
                   # downsampled_input = torch.load('./data_sampled/%s_rot_norm2.pt' % line).to(true.device)[0]
                   # downsampled_input = torch.load('./data_sampled/%s_rot_c.pt' % line).to(true.device)[0]

                    # ORIGINAL
                  #  downsampled_input = torch.load('./data_sampled/%s_rot_cm.pt' % line).to(true.device)[0]

                    path1 = f'./data_sampled/{line}_rot_cm.pt'
                    path2 = f'./data_sampled/{line}_rot_cm_t.pt'
                  #  path1 = "hard_case"
                    if os.path.exists(path1):
                        downsampled_input = torch.load(path1).to(true.device)[0]
                        init_stacks = torch.load('./data_sampled/init_stack_%s_rot_cm.pt' % line).to(true.device)
                    elif os.path.exists(path2):
                        downsampled_input = torch.load(path2).to(true.device)[0]
                        init_stacks = torch.load('./data_sampled/init_stack_%s_rot_cm_t.pt' % line).to(true.device)
                    elif img_num==0:
                        print("HARD CASE!!! :-)")
                        downsampled_input = torch.load('./data_sampled/sub5_sub_clin_h.pt' ).to(true.device)[0] #[:,:,:-1]
                        init_stacks = torch.load('./data_sampled/init_stack_sub5_sub_clin_h.pt' ).to(true.device) #[:,:-1]
                    else:
                        print("path doesn't exist")



                  # downsampled_input = torch.load('./data_sampled/%s_rot_full_.pt' % line).to(true.device)[0]
                 #   downsampled_input = torch.load('./data_sampled/%s_rot_full_dt.pt' % line).to(true.device)[0][:,0:64]

                    print("LOADING FILE!!")
                    print('./data_sampled/init_stack_%s_masked_rot.pt' % line) 
                    #init_stacks = torch.load('./data_sampled/init_stack_%s_rot.pt' % line).to(true.device)
                    #init_stacks = torch.load('./data_sampled/init_stack_%s_rot.pt' % line).to(true.device)
                   # init_stacks = torch.load('./data_sampled/init_stack_%s_rot_norm2.pt' % line).to(true.device)
                #    init_stacks = torch.load('./data_sampled/init_stack_%s_rot_c.pt' % line).to(true.device)
                   
                   # this one
                   #  init_stacks = torch.load('./data_sampled/init_stack_%s_rot_cm.pt' % line).to(true.device)
                    


                #  init_stacks = torch.load('./data_sampled/init_stack_%s_rot_full_.pt' % line).to(true.device)
#                    init_stacks = torch.load('./data_sampled/init_stack_%s_rot_full_dt.pt' % line).to(true.device)[:,0:64]

                    print("NEW file")
                    print(downsampled_input.shape, init_stacks.shape)
                    item = (downsampled_input,init_stacks),None

                    for n in range(3):
                        out_warp_up = model((torch.ones_like(item[0][0][None]), torch.zeros_like(item[0][1])))
             
                    torch.cuda.synchronize() 
                    start = time.time()
                    stack = model((item[0][0][None], item[0][1]))
                    torch.cuda.synchronize() 
                    end = time.time()
                    print("POSE ESTIMATION:{:.3f}".format(end - start))
            else:

            
                init_stacks = item[0][1].clone()
                init_stacks[:,:,0:3,3] = init_stacks[:,:,0:3,3]/2
                start = time.time()
     
                model_not_ot = True

                stack = model((item[0][0][None,:,:,::2,::2],init_stacks))
                if(model_not_ot):
                    grid_start = make_grid_one(init_stacks[0], [128, 128],  1,  [1,1,1], device=true.device)
                    grid_end = make_grid_one(init_stacks[1], [128, 128],  1,  [1,1,1], device=true.device)
                    new_grid = grid_start - grid_end
                    stack = stack - new_grid[None]
                stack_up  = F.interpolate(stack[0], size=[256, 256], mode='bicubic', align_corners=True)[None]*2
                grid_start = make_grid_one(item[0][1][0], [256,256,256],  1,  [1,1,1], device=stack.device)
                grid_end = make_grid_one(item[0][1][1], [ 256,256,256],  1,  [1,1,1], device=stack.device)
                new_grid = grid_start - grid_end
                
                stack = stack_up + new_grid[None]
                hres = True
               
                


                end = time.time()
                print("POSE ESTIMATION:{:.3f}".format(end - start))
   
      
        else:
            print("GROUND TRUTH")
            if hres or gt:
                stack = item[1][None,:,:,:,:]
            
            else:
                stack = item[1][None,:,:,::2,::2]*0.5
            
            

        os.makedirs(folder_save, exist_ok=True)

        project_and_save(stack, item, folder_save, img_num, model, hres, gt, clin)
    
    return stack, item[0]


def project_and_save(stack, item, folder_save, img_num, model, hres, gt, clin=False):
    
    with torch.no_grad():
        ALL_STACKS =   item[0][1][1]
       # line = open("/data/vision/polina/users/mfirenze/svr_my_train_2024/data_sampled/validation/val.txt").read().splitlines()[img_num]
      #  ALL_STACKS = torch.load('./data_sampled/init_stack_%s_rot_full.pt' % line).to(item[0][1].device)[1,0:64]

        print("ALL STACKS!")
        print(ALL_STACKS)
        save_splat = True

        if ((hres or gt) or clin==True):
            
            motion_2_3, aff_all = model.project_new_feb4_crop(stack[:,0:3],item[0][0][None,:,:,:,:][:,1:], ALL_STACKS, slice_in=0, spacing=1, shape=[item[0][0].shape[1],item[0][0].shape[2],item[0][0].shape[3]])
         #   aff_all[:,0:3,3] = aff_all[:,0:3,3]*(1/0.875)
            psf_vals = torch.ones((1,4))*0.25
            psf_vals = psf_vals.to(device=stack.device)
            psf_coords = torch.zeros((3,4))
            psf_coords[0,0] = 0
            psf_coords[0,1] = 1
            psf_coords[0,2] = 2
            psf_coords[0,3] = 3
            psf_coords = psf_coords.to(device=stack.device)

            if clin:
                psf_vals = torch.ones((1,2))*0.5
                psf_vals = psf_vals.to(device=stack.device)
                psf_coords = torch.zeros((3,2))
                psf_coords[0,0] = 0
                psf_coords[0,1] = 1
                psf_coords = psf_coords.to(device=stack.device)

            if save_splat:
                sl_size = item[0][0][None,:,:,:,:].shape[3]
                splat_thick = model.unet3.splat.forward_retooled_one_thick(item[0][0][None,:,:,:,:][:,:1], aff_all, ALL_STACKS, mask= item[0][0][None,:,:,:,:][:,1:],volume_shape= [sl_size,sl_size,sl_size], slice_dim = [1,1,1], vol_dim=1, flow_dim=1, psf=(psf_vals, psf_coords)) #item[0][None].shape[-3:])
                splat_thick = splat_thick[:,:-1] / (splat_thick[:,-1:] + 1e-12 * splat_thick[:,-1:].max().item()) # normalize

                print("SAVE SPLAT THICK")
                nii_image = nib.Nifti1Image(splat_thick[0,0].detach().cpu().numpy(), np.eye(4))  
                nib.save(nii_image , '/data/vision/polina/users/mfirenze/svr_my_train_2024/splat_thick_new.nii.gz')

                splat_thick = model.unet3.splat.forward_retooled_one(item[0][0][None,:,:,:,:][:,:1], stack, ALL_STACKS, mask= item[0][0][None,:,:,:,:][:,1:],volume_shape= [sl_size,sl_size,sl_size], slice_dim = [1,1,1], vol_dim=1, flow_dim=1) #item[0][None].shape[-3:])
                splat_thick = splat_thick[:,:-1] / (splat_thick[:,-1:] + 1e-12 * splat_thick[:,-1:].max().item()) # normalize

                print("SAVE SPLAT THICK")
                nii_image = nib.Nifti1Image(splat_thick[0,0].detach().cpu().numpy(), np.eye(4))  
                base_path = f"/data/vision/polina/users/mfirenze/svr_my_train_2024/splat_256_{img_num}.nii.gz"

                # If it already exists, keep adding "_" until it's unique
                save_path = base_path
                while os.path.exists(save_path):
                    parts = os.path.splitext(save_path)          # ('.../file', '.gz') for .gz
                    if parts[1] == ".gz":                        # handle .nii.gz separately
                        stem, ext1 = os.path.splitext(parts[0])  # split into '.../file' and '.nii'
                        save_path = f"{stem}_{ext1}{parts[1]}"   # add underscore before .nii.gz
                    else:
                        save_path = f"{parts[0]}_{parts[1]}"

                # Save NIfTI
              #  nib.save(nii_image, save_path)
                print(f"Saved to {save_path}")
                    
                    
        else:
            init_stacks = item[0][1].clone()
            init_stacks[:,:,0:3,3] = init_stacks[:,:,0:3,3]/2
            ALL_STACKS = init_stacks[1]
           
            motion_2_3, aff_all = model.project_new_feb4_crop(stack[:,0:3],item[0][0][None,:,:,::2,::2][:,1:], ALL_STACKS, slice_in=0, spacing=1, shape=[item[0][0].shape[1],item[0][0].shape[2],item[0][0].shape[3]])
            aff_all[:,0:3,3] = aff_all[:,0:3,3]*2
            

            # # TEST ADDITION
            ALL_STACKS =  item[0][1][1]

            if save_splat:
                psf_vals = torch.ones((1,4))*0.25
                psf_vals = psf_vals.to(device=stack.device)
                psf_coords = torch.zeros((3,4))
                psf_coords[0,0] = 0
                psf_coords[0,1] = 1
                psf_coords[0,2] = 2
                psf_coords[0,3] = 3
                psf_coords = psf_coords.to(device=stack.device)
                print("new psf!")
                psf=(psf_vals, psf_coords)

                splat_thick = model.unet3.splat.forward_retooled_one_thick(item[0][0][None,:,:,:,:][:,:1], aff_all, ALL_STACKS, mask= item[0][0][None,:,:,:,:][:,1:],volume_shape= [256,256,256], slice_dim = [1,1,1], vol_dim=1, flow_dim=1, psf=(psf_vals, psf_coords)) #item[0][None].shape[-3:])
                splat_thick = splat_thick[:,:-1] / (splat_thick[:,-1:] + 1e-12 * splat_thick[:,-1:].max().item()) # normalize
                
                print("SAVE SPLAT THICK")
                nii_image = nib.Nifti1Image(splat_thick[0,0].detach().cpu().numpy(), np.eye(4))  
                base_path = f"/data/vision/polina/users/mfirenze/svr_my_train_2024/splat_thin_{img_num}.nii.gz"

                # If it already exists, keep adding "_" until it's unique
                save_path = base_path
                while os.path.exists(save_path):
                    parts = os.path.splitext(save_path)          # ('.../file', '.gz') for .gz
                    if parts[1] == ".gz":                        # handle .nii.gz separately
                        stem, ext1 = os.path.splitext(parts[0])  # split into '.../file' and '.nii'
                        save_path = f"{stem}_{ext1}{parts[1]}"   # add underscore before .nii.gz
                    else:
                        save_path = f"{parts[0]}_{parts[1]}"

                # Save NIfTI
                nib.save(nii_image, save_path)
                print(f"Saved to {save_path}")
                    
                    
          
 


        all_slices = item[0][0][None,:,:,:,:][:,:]
       
        idx_stack1 = (item[0][1][0][:, 0, 0] == 1).nonzero(as_tuple=False)[-1].item()+1
        idx_stack2 = (item[0][1][0][:, 1, 0] == -1).nonzero(as_tuple=False)[-1].item()+1


      #  idx_stack1 = (item[0][1][0][:, 0, 0] == 0.875).nonzero(as_tuple=False)[-1].item()+1
      #  idx_stack2 = (item[0][1][0][:, 1, 0] == -0.875).nonzero(as_tuple=False)[-1].item()+1
        if(not hres):
        #    item[0][1][0][:,0:3,3] = item[0][1][0][:,0:3,3]*2
            print("REMOVED TIMES TWO")
        

        stack1, aff1_p, aff1_og = all_slices[:,:,0:idx_stack1], aff_all[0:idx_stack1], item[0][1][1][0:idx_stack1]
        stack2, aff2_p, aff2_og=  all_slices[:,:,idx_stack1:idx_stack2], aff_all[idx_stack1:idx_stack2],  item[0][1][1][idx_stack1:idx_stack2]
        stack3, aff3_p, aff3_og = all_slices[:,:,idx_stack2:], aff_all[idx_stack2:],  item[0][1][1][idx_stack2:]
        

        aff1 = torch.stack([aff1_og, aff1_p], dim=0)
        aff2 = torch.stack([aff2_og, aff2_p], dim=0)
        aff3 = torch.stack([aff3_og, aff3_p], dim=0)


        torch.save(stack1,  '%s/%s_slices_stack1_256.pth' % (folder_save,img_num ))
        torch.save(aff1, '%s/%s_aff_stack1_256.pth' % (folder_save,img_num ))

        torch.save(stack2, '%s/%s_slices_stack2_256.pth' % (folder_save,img_num ))
        torch.save(aff2, '%s/%s_aff_stack2_256.pth' % (folder_save,img_num ))

        torch.save(stack3, '%s/%s_slices_stack3_256.pth' % (folder_save,img_num ))
        torch.save(aff3, '%s/%s_aff_stack3_256.pth' % (folder_save,img_num ))

        stack_num = 0
        if(clin and img_num!=4 ):
            voxel_size = (1.406, 1.406, 3)
         #   voxel_size = (1.406, 3, 1.406)

            #print("CHANGE VOXEL ORDER")
          #  voxel_size = (3, 1.406, 1.406)
        if (clin and (img_num==4 or img_num==6)):
            voxel_size = (1.289, 1.289, 3)
          #  voxel_size = (1.289, 3, 1.289 )
        if not clin:
            voxel_size = (0.8, 0.8, 3.2)
          #  voxel_size = (1.6, 1.6, 3.2)
          #  voxel_size = (0.8, 3.2, 0.8)

        REORIENT = False
        if(REORIENT):
            for stack in [stack1, stack2, stack3]:

                affine = np.diag([*voxel_size, 1])
                d1, w1,h1 = stack[0,0].shape
        
                T1c = affine[0:3, 3]
                T1c[0] = -(w1 - 1) / 2 * voxel_size[0]
                T1c[1] = -(h1 - 1) / 2 * voxel_size[1]
                T1c[2] = -(d1 - 1) / 2 * voxel_size[2]

                T1 = np.eye(4)
                if(stack_num==0):
                    I1 =  np.array([
                                [ 1,  0,  0],
                                [0,  1,  0.0000],
                                [ 0,  0,  1],
                            ], dtype=np.float32)
                    
                    R_xz_swap = np.array([
                        [0.0, 0.0, 1.0],  # New X comes from Old Z
                        [0, 1.0, 0.0],  # New Y comes from Old Y
                        [1.0, 0, 0.0],  # New Z comes from Old X
                    ], dtype=np.float32)
                elif(stack_num==1):
                    I1 = np.array([
                            [ 0.0000,  1.0000,  0.0000],
                            [-1.0000,  0.0000,  0.0000],
                            [ 0.0000,  0.0000,  1.0000],
                        ], dtype=np.float32)
                    


                    R_xz_swap = np.array([
                        [-1, 0, 0],  # New X comes from Old Z
                        [0, 0,   -1],  # New Y comes from Old Y
                        [0, -1, 0],  # New Z comes from Old X
                    ], dtype=np.float32)


                else:
                    I1 = np.array([
                            [ 0.,  1.,  0.],
                            [ 0.,  0.,  1.],
                            [ 1.,  0.,  0.],
                        ], dtype=np.float32)
                    
                    R_xz_swap = np.array([
                        [1.0, 0, 0],  # New X comes from Old Z
                        [0, 0,   1],  # New Y comes from Old Y
                        [0,1, 0],  # New Z comes from Old X
                    ], dtype=np.float32).T
                T1[0:3, 0:3] = I1[0:3, 0:3]@ affine[0:3, 0:3] 
                T1[0:3, 3] = I1 @ T1c 

                perm = np.eye(4)


                perm[0:3, 0:3] = R_xz_swap
    





            #   nifti_img = nib.Nifti1Image(stack[0,0].detach().cpu().numpy().transpose(1,2,0), affine)
                nifti_img = nib.Nifti1Image(stack[0,0].detach().cpu().numpy().transpose(2,1,0), perm @ T1)
            #  nifti_img = nib.Nifti1Image(stack[0,0].detach().cpu().numpy(), affine)
                nib.save(nifti_img, '%s/%s_stack%d.nii.gz' % (folder_save, img_num, stack_num ))

            #  nifti_img = nib.Nifti1Image(stack[0,1].detach().cpu().numpy().transpose(1,2,0), affine)
            #    nifti_img = nib.Nifti1Image(stack[0,1].detach().cpu().numpy().transpose(2,1,0), perm @ T1)

            # TO MAKE IT RECON ALL THE SLICES
               # nifti_img = nib.Nifti1Image(np.ones_like(stack[0,1].detach().cpu().numpy()).transpose(2,1,0), perm @ T1)
                nifti_img = nib.Nifti1Image(stack[0,1].detach().cpu().numpy().transpose(2,1,0), perm @ T1)
            #  nifti_img = nib.Nifti1Image(stack[0,1].detach().cpu().numpy(), affine)
                nib.save(nifti_img, '%s/%s_stack%d_mask.nii.gz' % (folder_save, img_num, stack_num ))
                stack_num+=1
        else:
            for stack in [stack1, stack2, stack3]:

                
                affine = np.diag([*voxel_size, 1])  # Identity affine with voxel spacing

                nifti_img = nib.Nifti1Image(stack[0,0,:,:,:].detach().cpu().numpy().transpose(1,2,0), affine)
                nib.save(nifti_img, '%s/%s_stack%d.nii.gz' % (folder_save, img_num,stack_num ))

                nifti_img = nib.Nifti1Image(stack[0,1,:,:,:].detach().cpu().numpy().transpose(1,2,0), affine)
                nib.save(nifti_img, '%s/%s_stack%d_mask.nii.gz' % (folder_save, img_num,stack_num ))
                stack_num+=1









def plot_3D_vol(volume, slice_x=128, idx=128):
    """
    Plot axial, coronal, and sagittal slices from a 3D volume.

    Parameters:
        volume (torch.Tensor or np.ndarray): A 3D volume of shape (D, H, W)
        slice_x (int): Index for slicing in the sagittal plane
        idx (int): Index for slicing in the coronal and axial planes
    """
    im1 = volume[slice_x, :, :]  # Axial slice
    im2 = volume[:, idx, :]      # Coronal slice
    im3 = volume[ :, :, idx]      # Sagittal slice

    fig, axs = plt.subplots(1, 3, figsize=(12, 5)) 
    axs[0].imshow(im1, cmap='gray', origin='lower')
    axs[0].set_title('Axial Slice')

    axs[1].imshow(im2, cmap='gray', origin='lower')
    axs[1].set_title('Coronal Slice')

    axs[2].imshow(im3, cmap='gray', origin='lower')
    axs[2].set_title('Sagittal Slice')

    plt.tight_layout()
    plt.show()



flip_row_12 = np.array([[-1, 0, 0, 0],
                        [0, 0, 1, 0],
                        [0, -1, 0, 0],
                        [0,  0, 0, 1]])


#root_path_mine = '/data/vision/polina/users/mfirenze/svr_my_train_2024/checkpoints/'


#ckpt_path_recon = "feta3d0_multi_stack_svr_final_sb2_flow_SNet3d2_512_multi_l22_loss_affine_invariant_3stacks_grid_no_flipbulkrot_0_no_crop_v9_sb4_no_flip_loss_reorient_clean_fix_ortho3_offset_rtx"


parser = argparse.ArgumentParser(description='Get testing info')


parser.add_argument('--hres', type=str, required=True, help='whether to use ground truth transform')
parser.add_argument('--clin', type=str, required=True, help='whether it is testing clinical case')

parser.add_argument('--gt', type=str, required=True, help='whether to use ground truth transform')
parser.add_argument('--folder_save', type=str, required=True, help='where to save params')
parser.add_argument('--model_folder', type=str, required=True, help='model to test')
parser.add_argument('--img_num', type=int, required=True, help='which volume to evaluate')
parser.add_argument('--save_gt', type=str, required=False, help='where to save params')
parser.add_argument('--rot', type=int, required=False, default=0, help='which volume to evaluate')
parser.add_argument('--translation', type=float, required=False, default=0, help='which volume to evaluate')
parser.add_argument('--noise', type=float, required=False, default=0, help='which volume to evaluate')


args = parser.parse_args()

folder_save = args.folder_save
trans_param = args.translation

img_num = args.img_num
rot = args.rot
trans = args.translation
noise = args.noise
save_gt = args.save_gt
clin = args.clin
if(args.gt=="True"):
    gt = True
else:
    gt = False

if(args.hres=="True"):
    hres = True
else:
    hres = False

if(args.clin=="True"):
    clin = True
else:
    clin = False
ckpt_path_recon = args.model_folder

print("params")
print(trans, rot,noise)
motion_params = {
    "translations": trans,
    "rotations": rot,
    "noise": noise
}

#seed_num = 1
seed_everything(img_num, workers=True)


print("GT OR NOT")
print(gt)
#folder_save_ =  "/data/vision/polina/users/mfirenze/svr_my_train_2024/saved_outputs/val12"
start_all = time.time()
run_model(full_path=ckpt_path_recon, img_num=img_num , folder_save=folder_save, subsample=1, gt = gt,hres=hres, save_gt=save_gt,clin=clin,**motion_params)
end_all = time.time()
print("RAN ALL MODEL in {:.3f} seconds".format(end_all - start_all))


    

