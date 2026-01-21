import os
import gc
#import ants
import math
import os.path as path
# import utils
import torch
import models
import models.losses

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



from pytorch_lightning import seed_everything

def save_slices(imgs, affs, subfolder_name,save_folder, clin, img_num, slice_res):
    # args:
    # imgs: tensor of shape (N, H, W) presumably? Or (N, H, W) matching original load
    # affs: tensor containing affines, will be split into og_aff and matrix_
    # save_folder: string path
    # subfolder_name: string name of subfolder to create inside save_folder
    # clin: boolean or string (handled below)
    # img_num: string or int, used for conditional logic
    
    # Ensure save directory exists
    
    full_save_path = os.path.join(save_folder, subfolder_name)
    os.makedirs(full_save_path, exist_ok=True)
    print("Slice save path:")
    print(full_save_path)
    
    # Logic from original script to handle 'clin' and spacing
    # Note: original code checks string "True" for args.clin

            
    if not clin:
        spacing = 1 #9
        slice_dim = 0.8 #1.46 #0.8 #0.8 #1.46 #.46
        spacing_r = 4 #4#3.95

    
    # img_num comes as string in args but might be passed as int, convert to str for comparison
    img_num_str = str(img_num)
    
    if clin and img_num_str != "4":

        spacing =   1 #1 #9
        slice_dim = 1.406 #1.289 #1.406 #1.46 #0.8 #0.8 #1.46 #.46
        spacing_r = 2.134 #4#3.95
    
    if clin and img_num_str == "4":

        spacing =   1 #1 #9
        slice_dim = 1.289 #1.289 #1.406 #1.46 #0.8 #0.8 #1.46 #.46
        spacing_r = 2.327 #4#3.95

    slice_dim = slice_res[0]
    spacing_r = slice_res[2]/slice_res[0]



    # Original code:
    # output based on the size settings...
    torch.set_grad_enabled(False)

    flip_col_02 = np.array([
                [0, 0, 1, 0],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0,0,0,1]
            ])
    flip_row_12 = np.array([[-1, 0, 0, 0],
                            [0, 0, 1, 0],
                            [0, -1, 0, 0],
                            [0,  0, 0, 1]])

    # Split affs as in original code:
    # og_aff, matrix_ = torch.load(...).tensor_split(2,0)
    # We assume 'affs' passed in is the loaded tensor before split
    og_aff_tensor, matrix_tensor = affs.tensor_split(2, 0)
    og_aff = og_aff_tensor[0].detach().cpu().numpy()
    matrix_ = matrix_tensor[0].detach().cpu().numpy()
    
    # user said "imgs ... torch tensors". Original code: imgs = torch.load(...) [0,0]
    # We assume 'imgs' passed in is already the tensor of interest (slices)
    
    input_size = matrix_.shape[0]

    with torch.no_grad():
        count = -1
        disp = -1 # Matches original initialization somewhat, though original had outer loop.
                  # In original, disp starts at -1 and increments.

        for n in range(0, input_size, spacing):
            count = count + 1
            disp = disp + 1
          
            og_mat = matrix_[n,:,:,]
            og_mat[0:3,3] = og_mat[0:3,3] * slice_dim
            mat = og_mat.copy()            

            og_aff[n,0,3] = og_aff[n,0,3]+(spacing_r*slice_dim)*0.5 
            disp_vector = og_aff[n,0:3,3] * slice_dim

            disp_vector = (( mat)[:3,:3] @ disp_vector)

            # scale d dimension
            scale_z = np.eye(4)
            scale_z[0,0] = slice_dim
            scale_z[1,1] = slice_dim
            scale_z[2,2] = spacing_r*slice_dim
            mat2 = (  mat ) @ flip_col_02 @ scale_z

            # add displacements
            mat2[0, 3] = mat2[0, 3]+disp_vector[0]
            mat2[1, 3] = mat2[1, 3]+disp_vector[1] 
            mat2[2, 3] = mat2[2, 3]+disp_vector[2] 


            mat2 = np.round(mat2, decimals=6)
            np.set_printoptions(suppress=True)
            
            # imgs[n, :, :] -> shape assumes (N, H, W)
            nii_image = nib.Nifti1Image(imgs[0,0][n, :, :].detach().cpu().numpy()[ :, :, np.newaxis], affine=mat2) 
            slices_2 = (imgs[0,0][n, :, :]>0).float()
            
            nii_image2 = nib.Nifti1Image(slices_2[:, :, np.newaxis].detach().cpu().numpy(), affine=mat2)
            
            # Save files

            nib.save(nii_image, '%s/%d.nii.gz' % (full_save_path, disp))
            nib.save(nii_image2, '%s/mask_%d.nii.gz' % (full_save_path, disp))

