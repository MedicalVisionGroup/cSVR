import os
import nibabel as nib
#import pandas as pd
import numpy as np
from nilearn import plotting, datasets, image
import matplotlib.pyplot as plt

import sys
import matplotlib.colors as mcolors
# Add the directory containing the module to the path


#import IPython.display as display
from nibabel.viewers import OrthoSlicer3D
import sys
import math
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as FF

import random
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import interpol
import cornucopia as cc
from PIL import Image
from torchvision.transforms.functional import resize
import pdb

import matplotlib
#matplotlib.use('TKAgg')  # Use the TkAgg backend which works well with X11 forwarding
import matplotlib.pyplot as plt
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D
from matplotlib.colors import Normalize
#from cornucopia as cc
from cornucopia.utils.warps import affine_flow

import os
import gc
#import ants
import math
import os.path as path
# import utils
import torch
import models

import datasets
from torchvision.utils import save_image 
from torch.utils.data import DataLoader
import cornucopia as cc
from matplotlib import pyplot as plt
import pdb
import time

import matplotlib.pyplot as plt
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D
import numpy as np


import models

#from pytorch_lightning import seed_everything

#seed_everything(1, workers=True)
import torch.nn.functional as F



def og_slice_pos(num_slices, slice_dims, slice_spacing, device):
    aff_slices = (torch.eye(4, device=device))
    aff_slices[0,0],aff_slices[1,1],aff_slices[2,2] = slice_dims[0], slice_dims[1], slice_dims[2]
    aff_slices = aff_slices.expand(num_slices, -1, -1).clone()
    
    indices = torch.arange(num_slices, device=aff_slices.device, dtype=aff_slices.dtype)
    aff_slices[:, 0, 3] = indices * slice_spacing
    return aff_slices


def identity(shape, **backend):
    """Returns an identity transformation field.

    Parameters
    ----------
    shape : (dim,) sequence of int
        Spatial dimension of the field.

    Returns
    -------
    grid : (*shape, dim) tensor
        Transformation field

    """
    backend.setdefault('dtype', torch.get_default_dtype())
    return torch.stack(cartesian_grid(shape, **backend), dim=-1)


def grid_from_affine(affine, shape):
    """Generate an affine flow field

    Parameters
    ----------
    affine : ([B], D+1, D+1) tensor
        Affine matrix
    shape : (D,) list[int]
        Lattice size

    Returns
    -------
    flow : ([B], *shape, D) tensor, Affine flow

    """
    ndim = len(shape)
    backend = dict(dtype=affine.dtype, device=affine.device)

    # add spatial dimensions so that we can use batch matmul
    for _ in range(ndim):
        affine = affine.unsqueeze(-3)
    lin, trl = affine[..., :ndim, :ndim], affine[..., :ndim, -1]

    # create affine transform
    grid = identity(shape, **backend)
   
    grid = lin.matmul(grid.unsqueeze(-1)).squeeze(-1)
    grid = grid.add_(trl)

    # subtract identity to get a flow
  #  flow = sub_identity_(flow)

    return grid.movedim(-1, 0)



def cartesian_grid(shape, **backend):
    """Wrapper for meshgrid(arange(...))

    Parameters
    ----------
    shape : list[int]

    Returns
    -------
    list[Tensor]

    """
    return meshgrid_ij(*(torch.arange(s, **backend) for s in shape))


def meshgrid_ij(*x):
        return torch.meshgrid(*x)




def make_grid(og_affs, slice_dir, slice_shape, voxel_size, volume_shape, device):
    """
    Constructs a 3D sampling grid from per-slice affine transformations.

    Parameters:
    ----------
    og_affs : torch.Tensor
        Tensor of shape (N, 4, 4) containing original affine matrices for each slice in real space (mm).
    slice_dir : int
        Index (0, 1, or 2) specifying the slice direction (e.g., axial, coronal, sagittal), determines R used.
    slice_shape : tuple of int
        Shape of each slice as (n, H, W), n is number of slices
    voxel_size : float
        Isotropic voxel size in millimeters.
    volume_shape : tuple of int
        Shape of the target volume 

    Returns:
    -------
    all_grids : torch.Tensor
        Tensor of shape (3, D, H, W) representing the sampling grid for the volume,
        constructed by transforming each slice's grid using its corresponding affine.
    """
    
    M2_arr = M2(slice_dir, device, grid_scale_f = volume_shape)
    S_arr = torch.diag(torch.tensor([1/voxel_size, 1/voxel_size, 1/voxel_size, 1.0], device=device))
    
    
    all_grids = torch.ones((3, slice_shape[0], slice_shape[1], slice_shape[2]))
    
    for i in range(128):
        aff_in =    M2_arr @ S_arr @ og_affs[i] 
        one_grid = grid_from_affine(aff_in, [1,slice_shape[1], slice_shape[2]])
        all_grids[:,i:i+1] = one_grid
    
    
    return all_grids

def make_grid_one(og_affs, slice_shape, voxel_size, slice_dims, device):
    """
    Constructs a 3D sampling grid from per-slice affine transformations.

    Parameters:
    ----------
    og_affs : torch.Tensor
        Tensor of shape (N, 4, 4) containing original affine matrices for each slice in real space (mm).
    slice_dir : int
        Index (0, 1, or 2) specifying the slice direction (e.g., axial, coronal, sagittal), determines R used.
    slice_shape : tuple of int
        Shape of each slice as (n, H, W), n is number of slices
    voxel_size : float
        Isotropic voxel size in millimeters.
    volume_shape : tuple of int
        Shape of the target volume 

    Returns:
    -------
    all_grids : torch.Tensor
        Tensor of shape (3, D, H, W) representing the sampling grid for the volume,
        constructed by transforming each slice's grid using its corresponding affine.
    """
    
    
    scale_arr = (torch.eye(4, device=device))
    scale_arr[0,0],scale_arr[1,1],scale_arr[2,2] = slice_dims[0], slice_dims[1], slice_dims[2]
    
    S_arr = torch.diag(torch.tensor([1/voxel_size, 1/voxel_size, 1/voxel_size, 1.0], device=device))
    
    
    all_grids = torch.ones((3, og_affs.shape[0], slice_shape[0], slice_shape[1]), device=device)
    
    for i in range(len(og_affs)):
   
        aff_in =   S_arr @ og_affs[i] @ scale_arr
        one_grid = grid_from_affine(aff_in, [1,slice_shape[0], slice_shape[1]])
        all_grids[:,i:i+1] = one_grid
    
    
    return all_grids

    
    

def M2(slice_dir, device, grid_scale_f = [1,1,1]):

        
        sl  = slice_dir

        if (sl == 0):
            affine = torch.eye(4)

        if (sl == 2): 
            affine = torch.eye(4)
            affine[0,0] = 0
            affine[0,2] = -1
            affine[2,0] = 1 #1
            affine[2,2]= 0 #-1
    
            rot_90 = torch.eye(4)
            rot_90[1,2] = 1
            rot_90[2,1] = -1
            rot_90[2,2] = 0
            rot_90[1,1] = 0
            affine = affine @ rot_90

        if (sl == 1):
            affine = torch.eye(4)
            affine[0,0] =   0
            affine[1,1]=    0
            affine[0,1] =   1 #1
            affine[1,0] =  -1 #-1
        
        affine[0,3] = (grid_scale_f[0]-1)/2
        affine[1,3]= (grid_scale_f[1]-1)/2 
        affine[2,3]= (grid_scale_f[2]-1)/2 
        

        # print("NO MINUS 1")
        # affine[0,3] = (grid_scale_f[0])/2
        # affine[1,3]= (grid_scale_f[1])/2 
        # affine[2,3]= (grid_scale_f[2])/2 

        
        t = affine[0:3,3]    
        aff_m = affine[:3,:3]
        d = aff_m @ t
        affine[0:3,3] = t-d
        
       # ans = grid_from_affine(affine, shape).to(x.device) #.flip(0)
       # ans = ans[None]

        return affine.to(device)
    
def og_slice_pos_pre(num_slices, slice_dims, slice_spacing, slice_dir, volume_shape, device):
    aff_slices = (torch.eye(4, device=device))
    aff_slices[0,0],aff_slices[1,1],aff_slices[2,2] = slice_dims[0], slice_dims[1], slice_dims[2]
    aff_slices = aff_slices.expand(num_slices, -1, -1).clone()
    
    indices = torch.arange(num_slices, device=aff_slices.device, dtype=aff_slices.dtype)
    aff_slices[:, 0, 3] = indices * slice_spacing
    
    M2_arr = M2(slice_dir, device, grid_scale_f = volume_shape)

    
    aff_slices = torch.einsum('ij,bjk->bik', M2_arr, aff_slices) 
    
    return aff_slices

def divide_into_stacks(ALL_STACKS, xs):

    idx_stack1 = (ALL_STACKS[0][:, 0, 0] == 1).nonzero(as_tuple=False)[-1].item()+1
    idx_stack2 = (ALL_STACKS[0][:, 1, 0] == -1).nonzero(as_tuple=False)[-1].item()+1
    idx_stack3 = (ALL_STACKS[0][:, 2, 0] == 1).nonzero(as_tuple=False)[-1].item()+1

    idx_stack1, idx_stack2, idx_stack3 = sorted([idx_stack1, idx_stack2, idx_stack3])
    
    
    xs_stack1 = xs[:, :, :idx_stack1]
    xs_stack2 = xs[:, :, idx_stack1:idx_stack2]
    xs_stack3 = xs[:, :, idx_stack2:]

    return (xs_stack1, xs_stack2, xs_stack3)