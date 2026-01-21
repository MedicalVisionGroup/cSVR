import sys
import math
import torch
import torch.nn as nn
import random
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import interpol
import sys




from PIL import Image
import pdb
import inspect
from cornucopia.utils.warps import affine_flow
from scipy import ndimage
from cornucopia.random import Sampler, Normal, Uniform
#from cornucopia.random import Sampler, Uniform, RandInt, Fixed, make_range, RandKFrom, UniformWithFlip
#in order to save image

from nilearn.datasets import load_mni152_template
from nilearn.plotting import plot_img
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn.functional as FF
from pathlib import Path
import importlib.util
import cornucopia as cc



project_root = "./models"
# Add it to sys.path if not already there:
if project_root not in sys.path:
    sys.path.insert(0, project_root)



from grid_utils import og_slice_pos_pre


class Compose(transforms.Compose):

    def __init__(self, transforms, gpuindex=1):
        super().__init__(transforms)
        self.gpuindex = gpuindex

    def __call__(self, *args, cpu=True, gpu=True, **kwargs):
        if cpu:
            for t in self.transforms[:self.gpuindex]:
                args = t(*args)
        if gpu:
            for t in self.transforms[self.gpuindex:]:
                args = t(*args)

        return args


class Pad2D():
    def __init__(self, padding=16, padding_mode=['circular','replicate','replicate']):
        super().__init__()
        self.padding = padding
        self.padding_dims = (padding * torch.eye(2, dtype=torch.int).repeat_interleave(2,1)).tolist()
        self.padding_mode = padding_mode

    def forward(self, features):
        for d in range(2):
            features = F.pad(features[None], pad=self.padding_dims[d], mode=self.padding_mode[d])[0]
        
        return features

def negate_list(lst):
    return [-x for x in lst]

def bucket_angles(t):
    if(t<-180).any()or(t>180).any():
        raise ValueError("All angles must be between -180 and 180")
    buckets = torch.zeros_like(t, dtype=torch.long, device=t.device)
    buckets[(t >= -45)  & (t < 45)]   = 0
    buckets[(t >= 45)   & (t < 135)]  = 1
    buckets[(t >= -135) & (t < -45)]  = 2
    buckets[(t >= 135)  | (t < -135)] = 3
    return buckets

class GenerateMotionTrajectory:
    #zooms=(-0.35,0.02) d
  #  def __init__(self, spacing=1, subsample=1, translations=0.1, rotations= 20, bulk_translations=0, bulk_rotations=0, zooms=0, slice=1, nodes=(1,16), shots=2, augment=False, noise=False, X=3, flow_final=False, crop=False, normalize_img = True):
  #  def __init__(self, spacing=1, subsample=1, translations=0.1, rotations=20, bulk_translations=0, bulk_rotations=0, zooms=0, slice=1, nodes=(8,16), shots=2, augment=False, noise=False, X=3, flow_final=False, crop=False,  normalize_img = True):
  #  def __init__(self, spacing=1, subsample=1, translations=0.03, rotations= 20, bulk_translations=0, bulk_rotations=70, zooms=(-0.35,-0.2), slice=1, nodes=(1,16), shots=2, augment=False, noise=False, X=3, flow_final=False, crop=False, normalize_img = True):
   # def __init__(self, spacing=1, subsample=1, translations=0, rotations= 0, bulk_translations=0, bulk_rotations=0, zooms=0, slice=1, nodes=(1,16), shots=2, augment=False, noise=False, X=3, flow_final=False, crop=False, normalize_img = True):
   # def __init__(self, spacing=1, subsample=1, translations=0.03, rotations= 20, bulk_translations=0, bulk_rotations=70, zooms=0, slice=1, nodes=(1,16), shots=2, augment=False, noise=False, X=3, flow_final=False, crop=False, normalize_img = True):
  #  def __init__(self, spacing=1, subsample=1, translations=0.03, rotations= 20, bulk_translations=0, bulk_rotations=70, zooms=(-0.35,-0.2), slice=1, nodes=(1,16), shots=2, augment=False, noise=False, X=3, flow_final=False, crop=False, normalize_img = True):
  #  def __init__(self, spacing=1, subsample=1, translations=0.03, rotations= 20, bulk_translations=0, bulk_rotations=70, zooms=(-0.141,-0.140), slice=1, nodes=(2,4), shots=2, augment=False, noise=False, X=3, flow_final=False, crop=False, normalize_img = True):
    # last train

    #last train mod
 #   def __init__(self, spacing=1, subsample=1, translations=0.03, rotations= 40, bulk_translations=0,  bulk_rotations_plane=0, bulk_rotations_tr_plane=0, zooms=(-0.35,-0.2), slice=1, nodes=(1,100), shots=2, augment=False, noise=0, X=3, flow_final=False, crop=False, normalize_img = True):
  #  def __init__(self, spacing=1, subsample=1, translations=0.03, rotations= 40, bulk_translations=0,  bulk_rotations_plane=0, bulk_rotations_tr_plane=0, zooms=(-0.45,0), slice=1, nodes=(1,100), shots=2, augment=False, noise=0, X=3, flow_final=False, crop=False, normalize_img = True):

   # def __init__(self, spacing=1, subsample=1, translations=0.03, rotations= 40, bulk_translations=0, bulk_rotations=70, zooms=(-0.35,-0.2), slice=1, nodes=(1,100), shots=2, augment=False, noise=True, X=3, flow_final=False, crop=False, normalize_img = True):
 #   def __init__(self, spacing=1, subsample=1, translations=0.03, rotations= 40, bulk_translations=0, bulk_rotations=70, zooms=(-0.45,0), slice=1, nodes=(1,100), shots=2, augment=False, noise=True, X=3, flow_final=False, crop=False, normalize_img = True):

   # def __init__(self, spacing=1, subsample=1, translations=0.03, rotations= 10, bulk_translations=0, bulk_rotations=0, zooms=(-0.35,-0.2), slice=1, nodes=(1,16), shots=2, augment=False, noise=True, X=3, flow_final=False, crop=False, normalize_img = True):

    # eval
  #  def __init__(self, spacing=1, subsample=1, translations=0.03, rotations= 30, bulk_translations=0, bulk_rotations_plane=180, bulk_rotations_tr_plane=10, zooms=0, slice=1, nodes=(1,100), shots=2, augment=False, noise=0, X=3, flow_final=False, crop=False, normalize_img = True):
  #  def __init__(self, spacing=1, subsample=1, translations=0.03, rotations= 30, bulk_translations=0, bulk_rotations_plane=70, bulk_rotations_tr_plane=0, zooms=0, slice=1, nodes=(1,100), shots=2, augment=False, noise=0, X=3, flow_final=False, crop=False, normalize_img = True):

  # eval hard
  #  def __init__(self, spacing=1, subsample=1, translations=0.05, rotations= 30, bulk_translations=0, bulk_rotations=70, zooms=0, slice=1, nodes=(1,16), shots=2, augment=False, noise=False, X=3, flow_final=False, crop=False, normalize_img = True):
    def __init__(self, spacing=1, subsample=1, translations=0.03, rotations= 40, bulk_translations=0,  bulk_rotations_plane=0, bulk_rotations_tr_plane=0, zooms=0, slice=1, nodes=(1,100), shots=2, augment=False, noise=0, X=3, flow_final=False, crop=False, normalize_img = True, mlp_training=False, verbose=False, **kwargs):


        # DEFINE MOTION PARAMETERS
        self.verbose = verbose
        if(self. verbose):
            print("MOTION PARAMS:")
            print(f"spacing: {spacing}")
            print(f"subsampling: {subsample}")
            print(f"translations: {translations}")
            print(f"rotations: {rotations}")
            print(f"bulk rotation plane: {bulk_rotations_plane}")
            print(f"bulk rotation through plane: {bulk_rotations_tr_plane}")
            print(f"bulk translations: {bulk_translations}")
            print(f"slice: {slice}")
            print(f"crop: {crop}")
            print(f"augment: {augment}")
            print(f"zooms: {zooms}")
            print(f"noise: {noise}")
            print(f"augment: {augment}")
            print(f"mlp training: {mlp_training}")


        self.subsample = subsample
        self.spacing = spacing 
        
        self.slice = slice if isinstance(slice, (tuple, list)) else [slice]
     #   self.flip = cc.fov.RandomFlipTransform(axes=[-3]) #+ cc.fov.PatchTransform(192) # Turns flips off
        self.flip_ax1 = cc.Rot180Transform(axis=1)
        self.flip_ax2 = cc.Rot180Transform(axis=2)
        self.mlp_training = mlp_training
       # self.flip = cc.Rot180Transform(axis=2)
     #   self.flip = cc.Rot180Transform(axis=0)
        if(self.mlp_training):

            self.stack_order_pred = True
            self.no_rotation = False
            self.rot_180_axis = True
            self.predict_vec = True
        else:
            self.stack_order_pred = False
            self.no_rotation = False
            self.rot_180_axis = False
            self.predict_vec = False
        self.zoom = cc.RandomAffineTransform(translations=0, rotations=0, shears=0, zooms=zooms, iso=True)
        self.augment = augment #add_noise
        self.noise = noise
        self.crop = crop
        self.flow_final = flow_final
        self.X = X # nu,ber of stacks
        self.normalize_img = normalize_img
        self.bias_field = False
        self.gamma_field = False
        self.slice_bias_apply = False
        self.smooth = False
        self.slice_drop_out = 0.2
        
        
        if(self.augment==True): # do not do this for validation set

            self.bias_field = True
            self.gamma_field = True
            self.slice_bias_apply = True
            self.smooth = False

        else:
            self.slice_drop_out = 0


        # SPECIFY BULK ROTATIONS 
        random_motion = True # only True for debugging purposes
        one_plane_bulk = False # only have in-plane bulk rotations
        if(random_motion):
            if(one_plane_bulk):
                bulk1 = [0,0,bulk_rotations_plane]
                bulk2 = [0,bulk_rotations_plane,0]
                bulk3 = [  bulk_rotations_plane,0,0]

            else:
                bulk1 = [bulk_rotations_tr_plane,bulk_rotations_tr_plane,bulk_rotations_plane]
                bulk2 = [bulk_rotations_tr_plane,bulk_rotations_plane,bulk_rotations_tr_plane]
                bulk3 = [bulk_rotations_plane,bulk_rotations_tr_plane,bulk_rotations_tr_plane]
    
            bulk_rotations_list = []
            for i in slice:
                if(i==0):
                    bulk_rotations_list.append(bulk1)
                elif(i==1):
                    bulk_rotations_list.append(bulk2)
                elif(i==2):
                    bulk_rotations_list.append(bulk3)


               # bulk_rotations_list  = [[bulk_rotations,bulk_rotations,bulk_rotations],[bulk_rotations,bulk_rotations,bulk_rotations],[bulk_rotations,bulk_rotations,bulk_rotations]]
           # pdb.set_trace()
            
            # self.base = [cc.RandomSlicewiseAffineTransform(nodes=nodes, shots=shots, spacing=spacing, subsample=subsample, slice=s, translations=translations, rotations=rotations,
            #                                           bulk_translations=bulk_translations, bulk_rotations=bulk_rotations_list[s], shears=0, zooms=0) for s in self.slice]
        
            # self.base = [cc.RandomSlicewiseAffineTransform(nodes=nodes, shots=shots, spacing=spacing, subsample=subsample, slice=s, translations=translations, rotations=rotations,
            #                                             bulk_translations=bulk_translations, bulk_rotations=UniformWithFlip(bulk_rotations_list[s],prob=0.5) , shears=0, zooms=0) for s in self.slice]
            # self.base = [cc.RandomSlicewiseAffineTransform(nodes=nodes, shots=shots, spacing=spacing, subsample=subsample, slice=s, translations=Normal(0,translations/2), rotations=Normal(0,rotations/2),
            #                                             bulk_translations=bulk_translations, bulk_rotations=UniformWithFlip(bulk_rotations_list[s],prob=0.5) , shears=0, zooms=0) for s in self.slice]
            # self.base = [cc.RandomSlicewiseAffineTransform(nodes=nodes, shots=shots, spacing=spacing, subsample=subsample, slice=s, translations=Normal(0,translations), rotations=Normal(0,rotations/2),
            #                                               bulk_translations=bulk_translations, bulk_rotations=bulk_rotations_list[s] , shears=0, zooms=0) for s in self.slice]
            # self.base = [cc.RandomSlicewiseAffineTransform(nodes=nodes, shots=shots, spacing=spacing, subsample=subsample, slice=s, translations=Normal(0,translations), rotations=Normal(0,rotations/2),
            #                                               bulk_translations=bulk_translations, bulk_rotations=bulk_rotations_list[s] , shears=0, zooms=0) for s in self.slice]


            self.base = [cc.RandomSlicewiseAffineTransform(nodes=nodes, shots=shots, spacing=spacing, subsample=subsample, slice=s, translations=Normal(0,translations), rotations=Normal(0,rotations/2),
                                                         bulk_translations=bulk_translations, bulk_rotations=(negate_list(bulk_rotations_list[s]), bulk_rotations_list[s]) , shears=0, zooms=0) for s in self.slice]

            # self.base = [cc.RandomSlicewiseAffineTransform(nodes=nodes, shots=shots, spacing=spacing, subsample=subsample, slice=s, translations=Normal(0,translations), rotations=Normal(0,rotations/2),
            #                                              bulk_translations=bulk_translations, bulk_rotations=bulk_rotations_list[s] , shears=0, zooms=0) for s in self.slice]
            # self.base = [cc.RandomSlicewiseAffineTransform(nodes=nodes, shots=shots, spacing=spacing, subsample=subsample, slice=s, translations=Normal(0,translations), rotations=Normal(0,rotations/2),
            #                                               bulk_translations=bulk_translations, bulk_rotations=Normal(0,bulk_rotations_list[s]) , shears=0, zooms=0) for s in self.slice]

            # self.base = [cc.RandomSlicewiseAffineTransform(nodes=nodes, shots=shots, spacing=spacing, subsample=subsample, slice=s, translations=Normal(0,translations), rotations=rotations,
            #                                                bulk_translations=bulk_translations, bulk_rotations=(negate_list(bulk_rotations_list[s]), bulk_rotations_list[s]), shears=0, zooms=0) for s in self.slice]
        else: # only for debugging purposes
            print("NOT RANDOM MOTION")
            trans_new = torch.tensor([[0, 0, 0] for i in range(64)])
            trans_new = trans_new.view(64,3).tolist()
            rots_new = torch.tensor([[0, 45, 0] for i in range(64)])
            rots_new = rots_new.view(64,3).tolist()
            self.base = [cc.SlicewiseAffineTransform(spacing=spacing, subsample=subsample, slice=s,rotations=rots_new , shears=torch.tensor([0,0,0]),  #30*torch.ones((3,128)
                                                            translations= trans_new, zooms = torch.tensor([0,0,0]), unit='vox')  for s in self.slice]# CHANGED ZOOMS #torch.tensor([0,0.5,0.0])

        # SPECIFY AUGMENTATIONS
        prob_aug = 0.3
        self.mult = cc.MaybeTransform(cc.RandomGaussianNoiseTransform(sigma=0.1), prob_aug) 
     #   self.mult = cc.MaybeTransform(cc.RandomGaussianNoiseTransform(sigma=noise), 1)  # always do noise during validation
        self.bias = cc.MaybeTransform(cc.RandomMulFieldTransform(vmax=1), prob_aug) 
        self.gamma = cc.MaybeTransform(cc.RandomGammaTransform(gamma=(0.5, 2)), prob_aug) 
        self.slice_bias = cc.MaybeTransform(cc.RandomSlicewiseMulFieldTransform(slice=0, thickness=2, vmax=1), prob_aug)
        self.smoother = cc.MaybeTransform(cc.RandomSmoothTransform(fwhm=6), prob_aug)

    # HELPER FUNCTIONS
    def stack_in_single_plane(self, og): # turn slices to correct plane
        len_s = len(self.slice)
        ss = og.shape[1]//len_s
        new_vol = torch.zeros_like(og)
        for i, stack_num in enumerate(self.slice):
            
            if(stack_num == 0):
                print("stack {}, i {}, true stack 0".format(stack_num, i))
                stack_n = og[:,ss*i:ss*(i+1),:,:]
    
            if(stack_num == 1):
                print("stack {}, i {}, true stack 1".format(stack_num, i))
                stack = og[:,ss*i:ss*(i+1),:,:]
                stack_n = torch.rot90(stack, k=1, dims=(1, 2))

            if(stack_num == 2): 
                print("stack {}, i {}, true stack 2".format(stack_num, i))
                stack = og[:,ss*i:ss*(i+1),:,:]
                stack_n = torch.rot90(stack, k=-1, dims=(1, 3)) # for slice1
                stack_n = torch.rot90(stack_n, k=1, dims=(2, 3))
            new_vol[:,ss*i:ss*(i+1),:,:] = stack_n
        return new_vol

      
    
    def generate_initial_flow(self, vol_n): # add correct offset to account for planar slices
      
        len_s = len(self.slice)
        ss = vol_n.shape[2]//len_s  
        flow_new = torch.zeros((3,vol_n.shape[2],vol_n.shape[3],vol_n.shape[4])).to(vol_n.device)
        shape  = [vol_n.shape[2]//len_s,vol_n.shape[3],vol_n.shape[4]]

        for i in range(len_s):
            sl = self.slice[i]

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
                print("slice 2 rot")
                print(affine)
    
            affine[0,3] = (shape[0]-1)/2
            affine[1,3] = (shape[1]-1)/2
            affine[2,3] =(shape[2]-1)/2
        
            t = affine[0:3,3]    
            aff_m = affine[:3,:3]
            d = aff_m @ t
            affine[0:3,3] = t-d
            ans = affine_flow(affine, shape).movedim(-1, 0).to(vol_n.device) #.flip(0)
            flow_new[:,ss*i:ss*(i+1),:,:] = ans

        return flow_new

    def __call__(self, img1, seg1):
        
        if img1.ndim == 5 :
            img1, seg1 = zip(*[self(img1[i], seg1[i]) for i in range(img1.shape[0])])
            if(self.crop):
                return (img1[0][0][None], img1[0][1]), torch.stack(seg1, 0)
            else:
                return torch.stack(img1, 0), torch.stack(seg1, 0)

        numstacks = len(self.slice) #torch.randint(1, len(self.slice) + 1, [1]).item()img2
        # normalize image - remove if want to avoid shifts in brightness
        if self.normalize_img:
            print("Clamping values")
            img1 = (img1.clamp(min=0.1) - 0.1) * (1 / 0.9)

        else:
            print("No clamping")

        # MASK AUGMENTATIONS - USE IMPERFECT MASK AS AUGEMENTATION
        correct_mask = True
        rand_slice = False
        rand_mask = False
        if not correct_mask:
            seg1 = ((img1 > 0) | (seg1 > 0)).float()
        else:
            if(not rand_slice):
                print("Use correct mask!")
                seg1 = (seg1 > 0).float()
                while seg1.ndim < 4:
                    seg1 = seg1.unsqueeze(0)
            else:
    
                choices = (torch.rand(1, seg1.shape[0], 1, 1, device=seg1.device) > 0.5)
                img1 = torch.where(choices, img1, (seg1 > 0) * img1)
                if(not rand_mask):
                    seg1 = (seg1 > 0).float()
                    while seg1.ndim < 4:
                        seg1 = seg1.unsqueeze(0)
                else:
                    seg1 = ((img1 > 0) | (seg1 > 0)).float()

        

        # APPLY AUGMENTATIONS
        img1 = self.smoother(img1) if self.smooth else img1
        img1 = self.mult(img1) if self.noise > 0 else img1        
        img1 = self.bias(img1) if self.bias_field else img1
        img1 = self.gamma(img1) if self.gamma_field else img1


        # REVERSE ORDER
        if(self.stack_order_pred):
            rot180_stack = [random.randint(0, 1) for _ in range(3)]
        else:
            rot180_stack = [0,0,0]

        # APPLY ZOOM AUGMENTATION
        xform = self.zoom.make_final(img1)
        img1 = xform(img1)
        seg1 = xform(seg1)


        # SHUFFLE ORDER OF STACKS FOR MLP TRAINING
        if(self.mlp_training):
            self.slice = random.sample(self.slice, k=len(self.slice))
            print(f"Shuffled slices for MLP training: {self.slice}")        
        xform = [self.base[sl].make_final(img1) for sl in self.slice]


        #  GENERATE MOTION TRAJECTORIES
        if(self.stack_order_pred and not self.rot_180_axis):
            img0 = torch.cat([
                self.flip(xform[i](img1)) if rot180_stack[i] == 1 else xform[i](img1)
                for i in range(len(self.slice))
            ], dim=1)

            seg0 = torch.cat([
                self.flip(xform[i](seg1).gt(0).float()) if rot180_stack[i] == 1 else xform[i](seg1).gt(0).float()
                for i in range(len(self.slice))
            ], dim=1)
        elif self.stack_order_pred and self.rot_180_axis:
            imgs = []
            segs = []  #flip_ax2

            for i in range(len(self.slice)):
                # apply transform once

                if rot180_stack[i] == 1:
                    flip_fn = self.flip_ax1 if self.slice[i] == 2 else self.flip_ax2
                    img1_f = flip_fn(img1)
                    seg1_f = flip_fn(seg1)

                    img_i = xform[i](img1_f)
                    seg_i = xform[i](seg1_f).gt(0).float()
                else:

                    img_i = xform[i](img1)
                    seg_i = xform[i](seg1).gt(0).float()


                imgs.append(img_i)
                segs.append(seg_i)

            img0 = torch.cat(imgs, dim=1)
            seg0 = torch.cat(segs, dim=1)

        else: # ORIGINAL WITH NO CHANGE IN STACK ORDER OR 180
            img0 = torch.cat([xform[i](img1) for i in range(len(self.slice))], dim=1)
            seg0 = torch.cat([xform[i](seg1).gt(0).float() for i in range(len(self.slice))], 1)

        # EXTRACT FLOW FROM GENERATED MOTION TRAJECTORY
        flow = torch.cat([xform[i].flow for i in range(len(self.slice))], 1) # 
        
        # EXTRACT BULK ROTATION VALUES FOR MLP PREDICTION
        idx_slice = [2,1,0]
        bulk_rot_plane = torch.tensor([xform[i].bulk_rotations[idx_slice[self.slice[i]]] for i in range(len(self.slice))]).to(img1.device)
        rot_buckets = bucket_angles(bulk_rot_plane)

        # CONCATE MASK AND IMAGE TENSOR
        img0, flow = torch.cat([img0, seg0]), torch.cat([flow, seg0])

        # COMBINE ALL SLICES
        all_slices = img0[0][:,:,:][None][None] 

        # TURN SLICES RIGHT SIDE UP
        img0 = self.stack_in_single_plane(img0) # TEMPORARY
        flow = self.stack_in_single_plane(flow)
        img0[:1] = self.slice_bias(img0[:1]) if self.slice_bias_apply else img0[:1]

        # GENERATE ORTHOGONAL FLOW
        ONE_LAYER_ONLY = False
        ot = False # to train make this false

        if(ONE_LAYER_ONLY):
            ot = False
        if(ot):
            flow_ot = self.generate_initial_flow(flow[None][:,1:4])
        else:
            flow_ot = self.generate_initial_flow(flow[None][:,1:4])
            flow_ot = torch.zeros_like(flow_ot)

        # MAKE TOTAL FLOW, FLOW GENERATED FROM MOTION + RESIDUAL FROM THE STACK
        new_flow = torch.zeros_like(flow).to(flow.device)
        new_flow[0:3] = (flow_ot+ flow[None][:,:3,:,:,:])
        new_flow[3] = flow[None][0,3,:,:,:] # keep mask in last dimension
        flow = new_flow

    
        if (self.crop==False):
            return img0, flow
        else: # remove duplicates and black slices

            if(self.slice==[0,1,2]):
                sl_shape = img0.shape[2]

                #slice_dim 
                STACK_OG = og_slice_pos_pre(sl_shape, [1,1,1], 1, 0, [sl_shape,sl_shape,sl_shape], device=flow.device)

                STACK1 = og_slice_pos_pre(sl_shape, [1,1,1], 1, self.slice[0], [sl_shape,sl_shape,sl_shape], device=flow.device)
                STACK2 = og_slice_pos_pre(sl_shape, [1,1,1], 1, self.slice[1], [sl_shape,sl_shape,sl_shape], device=flow.device)
                STACK3 = og_slice_pos_pre(sl_shape, [1,1,1], 1, self.slice[2], [sl_shape,sl_shape,sl_shape], device=flow.device)

                ALL_STACKS = torch.zeros((2, sl_shape*3,4,4), device=flow.device)
                ALL_STACKS[0,0:sl_shape] =   STACK1
                ALL_STACKS[0, sl_shape:sl_shape*2] = STACK2
                ALL_STACKS[0, sl_shape*2:sl_shape*3] = STACK3
                no_stack_info = False # do not give fact that slices are orthogonal up!

                if(no_stack_info):
                    print("No stack information")
                    ALL_STACKS[0, sl_shape:sl_shape*2] = STACK1
                    ALL_STACKS[0, sl_shape*2:sl_shape*3] = STACK1
                
                # save slice height
                ALL_STACKS[1,   0:sl_shape] =   STACK1
                ALL_STACKS[1, sl_shape:sl_shape*2] = STACK1
                ALL_STACKS[1, sl_shape*2:sl_shape*3] = STACK1

                actually_crop = True
                remove_repeat = True


            else:

                len_s = len(self.slice)
                sl_shape = img0.shape[1]//len_s

                print("here :))))")
            
                #slice_dim 
                STACK_OG = og_slice_pos_pre(sl_shape, [1,1,1], 1, 0, [sl_shape,sl_shape,sl_shape], device=flow.device)

                STACK1 = og_slice_pos_pre(sl_shape, [1,1,1], 1, self.slice[0], [sl_shape,sl_shape,sl_shape], device=flow.device)
                STACK2 = og_slice_pos_pre(sl_shape, [1,1,1], 1, self.slice[1], [sl_shape,sl_shape,sl_shape], device=flow.device)
                STACK3 = og_slice_pos_pre(sl_shape, [1,1,1], 1, self.slice[2], [sl_shape,sl_shape,sl_shape], device=flow.device)

                ALL_STACKS = torch.zeros((2, img0.shape[1],4,4), device=flow.device)

                for i in range(len_s):
                    stack_num = self.slice[i]
                    if(stack_num == 0):
                        ALL_STACKS[0,sl_shape*i:sl_shape*(i+1)] =   STACK1
                    if(stack_num == 1):
                        ALL_STACKS[0,sl_shape*i:sl_shape*(i+1)] =   STACK2
                    if(stack_num == 2):
                        ALL_STACKS[0,sl_shape*i:sl_shape*(i+1)] =   STACK3

            no_stack_info = False # do not give fact that slices are orthogonal up!

            if(no_stack_info):
                print("NO STACK INFORMATION!!!")
                ALL_STACKS[0, 0:sl_shape] = STACK_OG
                ALL_STACKS[0, sl_shape:sl_shape*2] = STACK_OG
                ALL_STACKS[0, sl_shape*2:sl_shape*3] = STACK_OG
                
            # save slice height
            if(self.slice==[0,1,2]):
                ALL_STACKS[1,   0:sl_shape] =   STACK_OG
                ALL_STACKS[1, sl_shape:sl_shape*2] = STACK_OG
                ALL_STACKS[1, sl_shape*2:sl_shape*3] = STACK_OG

            else:
                for i in range(len_s):
                    stack_num = self.slice[i]
                    ALL_STACKS[1,sl_shape*i:sl_shape*(i+1)] =   STACK_OG
        
         
            remove_repeat = True
            print("REMOVE REPEAT?")
            print(remove_repeat)
            
            if (remove_repeat==True):
                
                rep = int(self.spacing / self.subsample)# number of times slice is repeated

                # remove duplicates
                img0 = img0[:,::rep]
                flow = flow[:,::rep]
                ALL_STACKS = ALL_STACKS[:,::rep]

                # find all the slices that have non-zero masks
                keep = img0[1:].reshape(img0[1:].shape[1], -1).any(dim=1)

                print("Number of slices")
                print(keep.sum())
                # ensure even number of slices
                if(keep.sum()%2==1):
                    idx_last = torch.nonzero(keep, as_tuple=True)[0][-1]
                    idx_first =  torch.nonzero(keep, as_tuple=True)[0][0]
                    if(idx_last<keep.shape[0]-1):
                        keep[idx_last+1] = 1
                    else:

                        idx_first = torch.nonzero(keep, as_tuple=True)[0][0]
                        if(idx_first>0):
                            keep[idx_first-1] = 1
                        else:
                            keep[idx_first] = 0
                            print("ODD NUMBER OF SLICES")

                # filter out slices with non-zero masks
                img0 = img0[:,keep]   
     
                ALL_STACKS = ALL_STACKS[:,keep]
                flow = flow[:,keep]

            # 
            # if(ONE_LAYER_ONLY):
            #  #   flow = flow[:,:,::16,::16]*(1/16)
            #     flow = flow[:,:,:,:] * (1/16)
            
            if self.mlp_training:
                print("IN MLP TRAINING")
                
                num_classes = 3
                num_rotations = 4 
                one_hot_stack = FF.one_hot(torch.tensor(self.slice), num_classes=num_classes).float().to(flow.device)
                one_hot_rots = FF.one_hot(rot_buckets, num_classes=num_rotations).float().to(flow.device)
                one_hot = torch.cat([one_hot_stack, one_hot_rots], dim=1)
         
                print(img0.shape, ALL_STACKS.shape,one_hot.shape )

                if self.predict_vec and not self.no_rotation:
                    stack_orientation_6 = torch.tensor(self.slice) * 2
                    stack_neg = torch.tensor(rot180_stack) + stack_orientation_6
                    one_hot_stack_dir = FF.one_hot(stack_neg, num_classes=6).float().to(flow.device)
                    one_hot = torch.cat([one_hot_stack_dir, one_hot_rots], dim=1)

                    if(self.slice_drop_out!=0 and self.augment==True):
                        drop_out = torch.rand(1).item() * self.slice_drop_out  # random float between 0 and 0.6
            
                        print("drop out")
                        print(drop_out)
                        img0 = img0.contiguous()

                        # 2. get shape
                        B, S, H, W = img0.shape
                        print("B,S,H,W:", B, S, H, W)

                        # 3. create idx SAFELY
                        keep = max(1, int((1-drop_out) * S))
                        idx = torch.arange(S, device=img0.device, dtype=torch.long)

                        # 4. random drop WITHOUT index_select first
                        perm = torch.randperm(S, device=img0.device)
                        idx = idx[perm][:keep]
                        idx, _ = torch.sort(idx)

                        # 6. index
                        
                        img0 = img0.index_select(1, idx)
                        ALL_STACKS = ALL_STACKS.index_select(1, idx)

                        return (img0,ALL_STACKS), one_hot
                    else:
                        return (img0,ALL_STACKS), one_hot
                

                if self.predict_vec and self.no_rotation:
                    stack_orientation_6 = torch.tensor(self.slice) * 2

                    
                    stack_neg = torch.tensor(rot180_stack) + stack_orientation_6
         
                    one_hot_stack_dir = FF.one_hot(stack_neg, num_classes=6).float().to(flow.device)
                #    one_hot = torch.cat([one_hot_stack_dir, one_hot_rots], dim=1)
                    return (img0,ALL_STACKS), one_hot_stack_dir

                if self.no_rotation:
                    one_hot_order = FF.one_hot(torch.tensor(rot180_stack), num_classes=2).float().to(flow.device)
                    one_hot = torch.cat([one_hot_stack, one_hot_order], dim=1)
                    return (img0,ALL_STACKS), one_hot
                
                if self.stack_order_pred:
                    one_hot_order = FF.one_hot(torch.tensor(rot180_stack), num_classes=2).float().to(flow.device)
                    one_hot = torch.cat([one_hot_stack, one_hot_rots, one_hot_order], dim=1)
                    return (img0,ALL_STACKS), one_hot
                return (img0,ALL_STACKS), one_hot


            print("GT DIMS:", flow.shape)
            return (img0,ALL_STACKS), flow
        
        

 

class Pad2d(torch.nn.Module):
    
    def __init__(self, padding, fill=0, padding_mode="constant"):
        super().__init__()

        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def forward(self, img, seg):
        """
        Args:
            img (PIL Image or Tensor): Image to be padded.

        Returns:
            PIL Image or Tensor: Padded image.
        """
        return F.pad(img, self.padding, self.fill, self.padding_mode),\
               F.pad(seg, self.padding, self.fill, self.padding_mode)






class Normalize():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, seg):
        mean = torch.as_tensor(self.mean).reshape([-1] + [1] * (img.ndim - 1))
        std = 1 / torch.as_tensor(self.std).reshape([-1] + [1] * (img.ndim - 1))

        return std * (img - mean), seg


class ToTensor(transforms.ToTensor):
    def __init__(self, numclass=1, imgtype='img'):
        super().__init__()
        self.numclass = numclass
        self.imgtype = imgtype

    def __call__(self, img, seg):
        if self.imgtype == 'img':
            img = F.to_tensor(img)
            # img = torch.as_tensor(img)
        elif self.imgtype == 'label':
            img = torch.as_tensor(np.array(img), dtype=torch.int64)
    
        seg = torch.as_tensor(np.array(seg), dtype=torch.int64)

        return img, seg




class ScaleZeroOne():
    def __init__(self, sig_gamma_sq=0.0):
        self.sig_gamma_sq = sig_gamma_sq

    def __call__(self, img, seg):
        if img.ndim == 5:
            img, seg = zip(*[self(img[i], seg[i]) for i in range(img.shape[0])])
            return torch.stack(img, 0), torch.stack(seg, 0)

        gamma = torch.empty(1).normal_(std=math.sqrt(self.sig_gamma_sq)).item() if self.sig_gamma_sq > 0 else 0

        
        img = (img - img.min()) * (1 / (img.max() - img.min())) ** math.exp(gamma)
    

        return img, seg



class ToTensor3d(transforms.ToTensor):
    def __init__(self, numclass=1):
        super().__init__()
        self.numclass = numclass
        self.is_cuda = False

    def __call__(self, img, seg):
        img = torch.as_tensor(img)# * (1/255)
        seg = torch.as_tensor(seg)

        return img, seg