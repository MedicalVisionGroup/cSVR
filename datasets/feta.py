
import os
import sys
import glob
import random
import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
from . import pairset
from . import transforms
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torchvision.transforms import ToPILImage, InterpolationMode
from torchvision.transforms.functional import to_tensor, resize
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import verify_str_arg
import torch.nn.functional as FF
from scipy.ndimage import gaussian_filter
import math
import pdb
#import cv2
from scipy.ndimage import zoom

class FeTA(VisionDataset):
    def __init__(
            self,
            root: str = '../feta_2.1_reg',
            image_set: str = 'train',
            split: str = '',
            stride: int = 1,
            out_shape: list = [256,256,256],
            numinput: int = 1,
            numclass: int = 1,
            multiply: int = 1,
            weights = 1,
            transforms: Optional[Callable] = None,
            **kwargs
    ):
        super().__init__(root, transforms)
        image_sets = ['train', 'test', 'val']
        self.stride = stride
        self.multiply  = multiply
        self.image_set = image_set
        self.image_file = '%s_rec-%s_T2w_norm_reg.nii'
   


        self.image_file_edges = '%s_rec-%s_T2w_reg.nii'

      #  self.image_file_edges = '%s_rec-%s_T2w_norm_reg.nii'
        self.label_file = '%s_rec-%s_dseg_reg.nii'
        self.numinput = numinput
        self.numclass = numclass

        with open(os.path.join('./datasets/feta_2.1', image_set), 'r') as f:
            path_names = [p.strip() for p in f.readlines()]
        include_edges = True
        only_edges = False
        if(only_edges):
            self.image_file = self.image_file_edges

        path_names = [path_names[i] for i in self.split] if isinstance(split, list) else path_names

        self.images = [os.path.join(self.root, p, 'anat', self.image_file % (p, 'mial' if p < 'sub-041' else 'irtk' if p < 'sub-081' else 'nmic')) for p in path_names]

   

        self.labels = [os.path.join(self.root, p, 'anat', self.label_file % (p, 'mial' if p < 'sub-041' else 'irtk' if p < 'sub-081' else 'nmic')) for p in path_names]
        if(include_edges):
            self.images_edges = [os.path.join(self.root, p, 'anat', self.image_file_edges % (p, 'mial' if p < 'sub-041' else 'irtk' if p < 'sub-081' else 'nmic')) for p in path_names]
            self.images = self.images + self.images_edges
            self.labels = self.labels + self.labels

       

    def __getitem__(self, index: int, cpu=True, gpu=False) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        image = self.images[self.stride*index % len(self.images)]
        label = self.labels[self.stride*index % len(self.images)]

        nifti_img = nib.load(image)
        nifti_img = nib.as_closest_canonical(nifti_img)
        img = np.array(nifti_img.get_fdata(),  dtype=np.float32)[None] #.transpose(2,1,0)

        original_voxel_size = np.array( nib.load(image).header.get_zooms())

        target = nib.load(label)
        target = nib.as_closest_canonical(target)
        target = np.asarray(target.dataobj,  dtype=np.int8)




        if self.transforms is not None:

  
            img, target = self.transforms(img, target, cpu=cpu, gpu=gpu)

        
        return img, target, index

    def __len__(self) -> int:
        return len(self.images) * self.multiply

    def __outshape__(self) -> list:
        return self.out_shape

    def __numinput__(self) -> int:
        return self.numinput

    def __weights__(self):
        return self.weights

    def __numclass__(self) -> int:
        return self.numclass

class CRL(VisionDataset):
    def __init__(
            self,
            root: str = '../CRL_FetalBrainAtlas_2017v3_lia',
            image_set: str = 'train',
            split: str = '',
            stride: int = 1,
            out_shape: list = [256,256,256],
            numinput: int = 1,
            numclass: int = 8,
            multiply: int = 1,
            weights = 1,
            transforms: Optional[Callable] = None,
            **kwargs
    ):
        super().__init__(root, transforms)
        image_sets = ['train', 'test', 'val']
        self.stride = stride
        self.multiply  = multiply
        self.image_set = image_set
        self.image_file = '%s_rec-%s_T2w.nii'
        self.label_file = '%s_rec-%s_dseg.nii'
        self.numinput = numinput
        self.numclass = numclass

        with open(os.path.join('./datasets/crl', image_set), 'r') as f:
            path_names = [p.strip() for p in f.readlines()]

        path_names = [path_names[i] for i in self.split] if isinstance(split, list) else path_names

        self.images = [os.path.join(self.root, '%s.nii.gz' % p) for p in path_names]
        self.labels = [os.path.join(self.root, '%s_regional.nii.gz' % p) for p in path_names]


    def __getitem__(self, index: int, cpu=True, gpu=False) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        image = self.images[self.stride*index % len(self.images)]
        label = self.labels[self.stride*index % len(self.images)]
        # label = label if os.path.isfile(label) else self.labels[self.stride*index]

        img = np.asarray(nib.load(image).dataobj, dtype=np.float32)[None] #fs.Volume.read(image).data[None]
        target = np.asarray(nib.load(label).dataobj, dtype=np.int8)[None] #fs.Volume.read(label).data[None]



        if self.transforms is not None:
            img, target = self.transforms(img, target, cpu=cpu, gpu=gpu)

        return img, target, index

    def __len__(self) -> int:
        return len(self.images) * self.multiply

    def __outshape__(self) -> list:
        return self.out_shape

    def __numinput__(self) -> int:
        return self.numinput

    def __weights__(self):
        return self.weights

    def __numclass__(self) -> int:
        return self.numclass

class MIAL(FeTA):
    def __init__(
            self,
            root: str = '../MIAL/lia', #'../MIAL/lia',
            stride: int = 1,
            out_shape: list = [256,256,256],
            numinput: int = 1,
            numclass: int = 1,
            multiply: int = 1,
            weights = 1,
            slice = 1,
            transforms: Optional[Callable] = None,
            **kwargs
    ):
        super().__init__(root=root, transforms=transforms)
        self.stride = stride
        self.multiply  = multiply
        self.image_set = 'test'
        self.image_file = '%s_run-%d_T2w_norm.nii.gz'
        self.label_file = '%s_run-%d_T2w_mask.nii.gz'
        self.numinput = numinput
        self.numclass = numclass

        runs = [5, 6] if slice == 0 else [1, 2] if slice == 1 else [3, 4]
        path_names = ['sub-01']

        self.images = [os.path.join(self.root, p, 'anat', self.image_file % (p, r)) for r in runs for p in path_names]
        self.labels = [os.path.join(self.root, p, 'anat', self.label_file % (p, r)) for r in runs for p in path_names]

class Clin(VisionDataset):
    def __init__(
            self,
            root: str = '../feta_2.1_reg',
            image_set: str = 'train-clin',
            split: str = '',
            stride: int = 1,
            out_shape: list = [256,256,256],
            numinput: int = 1,
            numclass: int = 1,
            multiply: int = 1,
            weights = 1,
            transforms: Optional[Callable] = None,
            **kwargs
    ):
        super().__init__(root, transforms)
        image_sets = ['train', 'test', 'val']
        self.stride = stride
        self.multiply  = multiply
        self.image_set = image_set
        self.image_file = 'mask4.nii'
        self.label_file = 'stack4.nii'
        self.numinput = numinput
        self.numclass = numclass

        with open(os.path.join('../datasets/feta_2.1', image_set), 'r') as f:
            path_names = [p.strip() for p in f.readlines()]

        path_names = [path_names[i] for i in self.split] if isinstance(split, list) else path_names


    def __getitem__(self, index: int, cpu=True, gpu=False) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        image = self.images[self.stride*index % len(self.images)]
        label = self.labels[self.stride*index % len(self.images)]
        # label = label if os.path.isfile(label) else self.labels[self.stride*index]

        img = np.asarray(nib.load(image).dataobj, dtype=np.float32)[None] #fs.Volume.read(image).data[None]
        target = np.asarray(nib.load(label).dataobj, dtype=np.int8)[None] #fs.Volume.read(label).data[None]
      #  pdb.set_trace()
       # img = np.repeat(img,8,axis=3)
       # pdb.set_trace()

       # pdb.set_trace()

        # Calculate scaling factors for each dimension
        original_voxel_size = np.array( nib.load(image).header.get_zooms())
        target_voxel_size = np.array([0.75, 0.75, 3])
        scaling_factors = original_voxel_size / target_voxel_size
        scaling_factors = np.insert(scaling_factors, 0, 1)
        img = zoom(img, scaling_factors, mode='nearest')
        target = zoom(target, scaling_factors, mode='nearest')
        x_curr_shape = int(img.shape[1])
        y_curr_shape = int(img.shape[2]) #412


       

        #CONVERT PHYSICAL SZME SIZE
        # Calculate scaling factors for each dimension
        # original_voxel_size = np.array( nib.load(image).header.get_zooms())
        # target_voxel_size = np.array([0.8, 0.8, 0.8])
        # scaling_factors = original_voxel_size / target_voxel_size
        # scaling_factors = np.insert(scaling_factors, 0, 1)
        # img = zoom(img, scaling_factors, mode='nearest')
        # target = zoom(target, scaling_factors, mode='nearest')
        # x_curr_shape = int(img.shape[1])
        # y_curr_shape = int(img.shape[2]) #412
      
      
       # pdb.set_trace()
        if(x_curr_shape >256):
            x_start = int((x_curr_shape-256)/2)
            y_start = int((y_curr_shape-256)/2)
            img = img[0,x_start:x_curr_shape-x_start,y_start:y_curr_shape-y_start,:]
            target = target[0,x_start:x_curr_shape-x_start,y_start:y_curr_shape-y_start,:]
        

        img = img[None,:,:,:]
        target = target[None,:,:,:]
        img = np.repeat(img,4,axis=3)
        target = np.repeat(target,4,axis=3)

        x_dim, y_dim = img.shape[-3], img.shape[-2]
        pad_num = int((256 - img.shape[-1])/2)

        pad_zeros = np.zeros((x_dim, y_dim,pad_num))[None,:,:,:]
       # img = img[None,:,:,:]
       # target = target[None,:,:,:]


        #ADDED THIS


      #  target = np.repeat(target,8, axis=3)
      #  pdb.set_trace()

        concatenated_img = np.concatenate((pad_zeros, img, pad_zeros), axis=3)
        target_img = np.concatenate((pad_zeros, target, pad_zeros), axis=3)

        img = concatenated_img
        target = target_img

        target = target.astype(np.int32)


        if self.transforms is not None:
            img, target = self.transforms(img, target, cpu=cpu, gpu=gpu)

        return img, target, index

    def __len__(self) -> int:
        return len(self.images) * self.multiply

    def __outshape__(self) -> list:
        return self.out_shape

    def __numinput__(self) -> int:
        return self.numinput

    def __weights__(self):
        return self.weights

    def __numclass__(self) -> int:
        return self.numclass


def feta3d_svr(root='../feta_2.1_mial', slice=1, spacing=2, subsample=2, **kwargs):

    trainformer = transforms.Compose([transforms.ToTensor3d(), transforms.ScaleZeroOne(), transforms.RandAffine3dSlice(spacing=spacing, zooms=(-0.1,0.1), subsample=subsample, slice=slice)], gpuindex=1)
    transformer = transforms.Compose([transforms.ToTensor3d(), transforms.ScaleZeroOne(), transforms.RandAffine3dSlice(spacing=spacing, zooms=(-0.1,0.1), subsample=subsample, slice=slice, augment=False)], gpuindex=1)
    testsformer = transforms.Compose([transforms.ToTensor3d(), transforms.ScaleZeroOne()], gpuindex=1)
   
    # atlasformer = transforms.Compose([transforms.ToTensor3d(), transforms.ScaleZeroOne(), transforms.Subsample3d()], gpuindex=1)
    train = FeTA(root, image_set='train', multiply=5, transforms=trainformer, **kwargs)
    valid = FeTA(root, image_set='val',   multiply=8, transforms=transformer, **kwargs)
    # tests = FeTA(root, image_set='val',   transforms=testsformer, **kwargs)
    tests = MIAL(slice=slice, transforms=testsformer, **kwargs)
    extra = CRL(root='../CRL_FetalBrainAtlas_2017v3_lia', image_set='train', multiply=30, transforms=trainformer, **kwargs)
    # atlas = CRL(root='../CRL_FetalBrainAtlas_2017v3_reg', image_set='atlas')
    # _train, _, _ = brain3d_svr(train_set='train-500', slice=slice, spacing=spacing, subsample=subsample, **kwargs)
    
    return pairset.Sumset(train,extra), valid, tests




def feta3d_svr_3stacks(root='../feta_2.1_mial', spacing=2, subsample=2,  slice=[0,1,2], flow_final = False, crop=False, rotations=20, translations=0, noise=0, bulk_rotations_plane=0, bulk_rotations_tr_plane=0, mlp_training=False, **kwargs):


    trainformer = transforms.Compose([transforms.ToTensor3d(), transforms.ScaleZeroOne(), transforms.GenerateMotionTrajectory(spacing=spacing, subsample=subsample, slice=slice, flow_final = flow_final, crop=crop, augment= True, rotations=rotations, translations=translations, noise=noise, bulk_rotations_plane=bulk_rotations_plane, bulk_rotations_tr_plane=bulk_rotations_tr_plane, mlp_training=mlp_training, **kwargs )], gpuindex=1)
    transformer = transforms.Compose([transforms.ToTensor3d(), transforms.ScaleZeroOne(), transforms.GenerateMotionTrajectory(spacing=spacing,  subsample=subsample, slice=slice, augment=False, flow_final=flow_final, crop=crop, normalize_img = False, rotations=rotations, noise = noise, translations=translations, bulk_rotations_plane=bulk_rotations_plane, bulk_rotations_tr_plane=bulk_rotations_tr_plane, mlp_training=mlp_training, **kwargs)], gpuindex=1)
    testsformer = transforms.Compose([transforms.ToTensor3d(), transforms.ScaleZeroOne()], gpuindex=1)
   
    # atlasformer = transforms.Compose([transforms.ToTensor3d(), transforms.ScaleZeroOne(), transforms.Subsample3d()], gpuindex=1)


    train = FeTA(root, image_set='train', multiply=5, transforms=trainformer, **kwargs)
    valid = FeTA(root, image_set='val',   multiply=8, transforms=transformer, **kwargs)
    # tests = FeTA(root, image_set='val',   transforms=testsformer, **kwargs)
    tests = MIAL(slice=slice, transforms=testsformer, **kwargs)
    extra = CRL(root='../CRL_FetalBrainAtlas_2017v3_lia', image_set='train', multiply=30, transforms=trainformer, **kwargs)
    # atlas = CRL(root='../CRL_FetalBrainAtlas_2017v3_reg', image_set='atlas')

    # _train, _, _ = brain3d_svr(train_set='train-500', slice=slice, spacing=spacing, subsample=subsample, **kwargs)

    return pairset.Sumset(train,extra), valid, tests

def feta3d_svr_no_gpu(root='../feta_2.1_mial', slice=1, spacing=2, subsample=2, **kwargs):
    trainformer = transforms.Compose([transforms.ToTensor3d(), transforms.ScaleZeroOne(), transforms.RandAffine3dSlice(spacing=spacing, zooms=(-0.1,0.1), subsample=subsample, slice=slice)])
    transformer = transforms.Compose([transforms.ToTensor3d(), transforms.ScaleZeroOne(), transforms.RandAffine3dSlice(spacing=spacing, zooms=(-0.1,0.1), subsample=subsample, slice=slice, augment=False)])
    testsformer = transforms.Compose([transforms.ToTensor3d(), transforms.ScaleZeroOne()], gpuindex=1)
   
    # atlasformer = transforms.Compose([transforms.ToTensor3d(), transforms.ScaleZeroOne(), transforms.Subsample3d()], gpuindex=1)


    train = FeTA(root, image_set='train', multiply=5, transforms=trainformer, **kwargs)
    valid = FeTA(root, image_set='val',   multiply=8, transforms=transformer, **kwargs)
    # tests = FeTA(root, image_set='val',   transforms=testsformer, **kwargs)
    tests = MIAL(slice=slice, transforms=testsformer, **kwargs)
    extra = CRL(root='../CRL_FetalBrainAtlas_2017v3_lia', image_set='train', multiply=30, transforms=trainformer, **kwargs)
    # atlas = CRL(root='../CRL_FetalBrainAtlas_2017v3_reg', image_set='atlas')

    # _train, _, _ = brain3d_svr(train_set='train-500', slice=slice, spacing=spacing, subsample=subsample, **kwargs)

    return pairset.Sumset(train,extra), valid, tests


def feta3d0_svr(**kwargs):
    return feta3d_svr(slice=0)


def feta3d1_svr(**kwargs):
    return feta3d_svr(slice=1)

def feta3d2_svr(**kwargs):
    return feta3d_svr(slice=2)

def feta3d0_4_svr(subsample=2, **kwargs):
    return feta3d_svr(slice=0, spacing=4, subsample=subsample)

def feta3d0_multi_stack_svr(subsample=4, **kwargs):
    return feta3d_svr_3stacks(spacing=4, subsample=subsample)

def feta3d0_multi_stack_svr_sb2(subsample=2, **kwargs):
    return feta3d_svr_3stacks(spacing=4, subsample=subsample)


def feta3d0_multi_stack_svr_final(subsample=4, slice=[0,1,2], **kwargs):
    return feta3d_svr_3stacks(spacing=4, subsample=subsample, flow_final=True, slice=slice)


def feta3d0_multi_stack_svr_final_sb2_v2(spacing=4, subsample=2, slice=[0,1,2],  translations=0.1, rotations=20, bulk_translations=0, bulk_rotations=0, zooms=0, **kwargs):
    return feta3d_svr_3stacks(spacing=spacing, subsample=subsample, slice=slice,  translations=translations, rotations=rotations, bulk_translations=bulk_translations, bulk_rotations=bulk_rotations, zooms=zooms, flow_final=True)



def feta3d0_multi_stack_svr_final_sb2(subsample=2, slice=[0,1,2], **kwargs):
    return feta3d_svr_3stacks(spacing=4, subsample=subsample, flow_final=True,slice=slice,  **kwargs)


def feta3d0_multi_stack_svr_final_sb2_crop(subsample=2, slice=[0,1,2],crop=True,rotations=20, translations=0, noise=0, bulk_rotations_plane=0, bulk_rotations_tr_plane=0,mlp_training = False, **kwargs):

    return feta3d_svr_3stacks(spacing=4, subsample=subsample, flow_final=True,slice=slice, crop=crop, rotations=rotations, translations=translations,noise=noise,bulk_rotations_plane=bulk_rotations_plane,bulk_rotations_tr_plane=bulk_rotations_tr_plane,mlp_training = False,**kwargs)

def feta3d0_mlp_multi_stack_svr_final_sb2_crop(subsample=2, slice=[0,1,2],crop=True,rotations=20, translations=0, noise=0, bulk_rotations_plane=0, bulk_rotations_tr_plane=0, mlp_training = True, **kwargs):

    return feta3d_svr_3stacks(spacing=4, subsample=subsample, flow_final=True,slice=slice, crop=crop, rotations=rotations, translations=translations,noise=noise,bulk_rotations_plane=bulk_rotations_plane,bulk_rotations_tr_plane=bulk_rotations_tr_plane, mlp_training = mlp_training, **kwargs)



def feta3d0_multi_stack_svr_final_sb2_crop_6(subsample=2, slice=[0,0,1,1,2,2],crop=True, bulk_rotations_plane=0, bulk_rotations_tr_plane=0,**kwargs):
    return feta3d_svr_3stacks(spacing=4, subsample=subsample, flow_final=True,slice=slice, crop=crop,bulk_rotations_plane=bulk_rotations_plane,bulk_rotations_tr_plane=bulk_rotations_tr_plane,**kwargs)


def feta3d0_multi_stack_svr_final_sb2_crop_hr(subsample=1, slice=[0,1,2],crop=True, **kwargs):
    return feta3d_svr_3stacks(spacing=4, subsample=subsample, flow_final=True,slice=slice, crop=crop, **kwargs)


def feta3d0_multi_stack_svr_final_sb2_og(subsample=2, slice=[0,1,2], **kwargs):
    return feta3d_svr_3stacks(spacing=4, subsample=subsample, flow_final=True,slice=slice)

def feta3d0_multi_stack_svr_final_sb1(subsample=1, slice=[0,1,2], **kwargs):
    return feta3d_svr_3stacks(spacing=4, subsample=subsample, flow_final=True,slice=slice)

def feta3d0_multi_stack_svr_final_sb2_s0(subsample=2, slice=[0,0,0], **kwargs):
    return feta3d_svr_3stacks(spacing=4, subsample=subsample, flow_final=True, slice=slice)



def feta3d0_4_svr_3stacks(subsample=2, **kwargs):
    return feta3d_svr_3stacks(slice=0, spacing=1, subsample=subsample)

def feta3d0_4_svr1(subsample=2, **kwargs):
    return feta3d_svr(slice=0, spacing=1, subsample=subsample)


def feta3d0_4_svr_no_gpu(subsample=2, **kwargs):
    return feta3d_svr_no_gpu(slice=0, spacing=4, subsample=subsample)

def feta3d0_4_svr_no_rot(subsample=2, **kwargs):
    return feta3d_svr_no_rot(slice=0, spacing=4, subsample=subsample)

def feta3d0_transform(subsample=2, **kwargs):
    return feta3d_svr_translation(slice=0, spacing=4, subsample=subsample)

def feta3d0_4_svr_clin(subsample=2, **kwargs):
    return feta3d_svr_clin(slice=0, spacing=4, subsample=subsample)

def feta3d0_4_svr_clin2(subsample=2, **kwargs):
    return feta3d_svr_clin2(slice=0, spacing=4, subsample=subsample)

def feta3d0_4_svr_clin_recon(subsample=2, **kwargs):
    return feta3d_svr_clin_recon(slice=0, spacing=4, subsample=subsample)

def feta3d0_4_svr_clin_stack(subsample=2, **kwargs):

    return feta3d_svr_clin_stack(slice=0, spacing=4, subsample=subsample)

def feta3d0_4_svr_clin_stack_no_gpu(subsample=2, **kwargs):

    return feta3d_svr_clin_stack_no_gpu(slice=0, spacing=4, subsample=subsample)

def feta3d1_4_svr(subsample=2, **kwargs):
    return feta3d_svr(slice=1, spacing=4, subsample=subsample)

def feta3d2_4_svr(subsample=2, **kwargs):
    return feta3d_svr(slice=2, spacing=4, subsample=subsample)

def feta3d_inpaint(root='/data/vision/polina/scratch/siyoung/Developer/feta_2.1_mial', spacing=4, subsample=1, slice=1, train_set='train', valid_set='val', tests_set='val', **kwargs):
    trainformer = transforms.Compose([transforms.ToTensor3d(), transforms.ScaleZeroOne(), transforms.RandAffine3dSliceSplat(spacing=spacing, slice=slice, zooms=(-0.1,0.1))], gpuindex=1)
    transformer = transforms.Compose([transforms.ToTensor3d(), transforms.ScaleZeroOne(), transforms.RandAffine3dSliceSplat(spacing=spacing, slice=slice, zooms=(-0.1,0.1))], gpuindex=1)

    train = FeTA(root, image_set=train_set, multiply=5, transforms=trainformer, **kwargs)
    valid = FeTA(root, image_set=valid_set, transforms=transformer, **kwargs)
    tests = FeTA(root, image_set=tests_set, transforms=transformer, **kwargs)
    extra = CRL(root='../CRL_FetalBrainAtlas_2017v3_lia', image_set='train', multiply=30, transforms=trainformer, **kwargs)

    return pairset.Sumset(train,extra), valid, tests

def feta3d_4_inpaint(**kwargs):
    return feta3d_inpaint(spacing=4, slice=None)

def feta3d0_inpaint(**kwargs):
    return feta3d_inpaint(slice=0)

def feta3d1_inpaint(**kwargs):
    return feta3d_inpaint(slice=1)

def feta3d2_inpaint(**kwargs):
    return feta3d_inpaint(slice=2)