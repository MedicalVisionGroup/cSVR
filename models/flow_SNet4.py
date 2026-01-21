import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .flow_UNetS import Flow_UNet, Flow_UNet_3stacks, Pool4MLP, MLP, Pool4MLP_attention, MLP_drop_out, MLP_norm, Flow_UNet_3stacks_encoder_only
import pdb
from cornucopia.utils.warps import affine_flow
from cornucopia.utils.warps import *
from .grid_utils import make_grid, og_slice_pos, make_grid_one
import nibabel as nib
import numpy as np
import os

class Flow_SNet_MLP(nn.Module):
    def __init__(self, *args, X=3, spacing=1, num_conv_per_flow=4, slice=[0,1,2],drop=0, rigid=False, crop=False, drop_out = 0, **kwargs):
        super().__init__()

        conv_kernel_sizes = [[1,3,3],[3,1,3],[3,3,1]]
        pool_kernel_sizes = [[1,2,2],[2,1,2],[2,2,1]]
        self.rigid = rigid # originally False
        self.crop = crop
        self.spacing = spacing
        
        slab_stride_sizes = [[spacing,1,1],[1,spacing,1],[1,1,spacing]] # with no repetition should be 1?
        slab_kernel_sizes = [[spacing,3,3],[3,spacing,3],[3,3,spacing]] # with no repetition should be 1?

        self.slice = slice
        slice = 0
        self.larger_image = True
        self.only_input_mask = False
        # defines 2D U-net (encoder), put all slices in the first dimension so use slice = 0
        self.unets = Flow_UNet_3stacks_encoder_only(*args, conv_kernel_sizes=conv_kernel_sizes[slice], pool_kernel_sizes=pool_kernel_sizes[slice],
                               slab_kernel_sizes=slab_kernel_sizes[slice], slab_stride_sizes=slab_stride_sizes[slice], 
                               mask=True, dropout_p=drop, num_conv_per_flow=num_conv_per_flow, X=X, **kwargs)
        
        # defines 3D U-net (decoder), use 3D convolution
        # self.unet3 = Flow_UNet_3stacks(*args, conv_kernel_sizes=3, pool_kernel_sizes=2, mask=True, dropout_p=drop, 
        #                        num_conv_per_flow=num_conv_per_flow, normalize_splat=False, X=X, **kwargs)
      #  self.strides = [self.unet3.enc_blocks[d].pool_stride for d in range(len(self.unet3.enc_blocks))]
        self.strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
  #      pdb.set_trace()
      #  self.unet3.flo_blocks = self.unet3.enc_blocks = None # unsure what this is
        self.count = 0 # if want to start applying projection after some iteration
        self.drop_out = drop_out

       # self.pool4mlp = Pool4MLP(pool_feature=3, pool_slices=8)
        #self.mlp = MLP(hidden_sizes=[128, 64], out_features=3)
       # self.mlp = MLP(hidden_sizes=[128, 64], out_features=7)
        #self.mlp = MLP(hidden_sizes=[128, 64], out_features=8)
      #  self.mlp = MLP(hidden_sizes=[128, 64], out_features=9)
        
        # #more slice feats 
      #  self.pool4mlp = Pool4MLP(pool_feature=3, pool_slices=15)
      #  self.mlp = MLP(hidden_sizes=[128, 64], in_features = 15, out_features=9)
        
        # self.pool4mlp = Pool4MLP(pool_feature=3, pool_slices=15)
        # self.mlp = MLP(hidden_sizes=[128, 64], in_channel_features = 3, in_slice_features = 15, out_features=9)

    
        # #more slice feats  + bigger MLP
        # self.pool4mlp = Pool4MLP(pool_feature=3, pool_slices=15)
        # self.mlp = MLP(hidden_sizes=[256, 128], in_features = 15, out_features=9)

        #more slice feats  + bigger MLP
        # self.pool4mlp = Pool4MLP(pool_feature=3, pool_slices=15)
        # self.mlp = MLP(hidden_sizes=[256, 128], in_channel_features = 3, in_slice_features = 15, out_features=9)

        # self.pool4mlp = Pool4MLP(pool_feature=30, pool_slices=8)
        # self.mlp = MLP(hidden_sizes=[256, 128], in_channel_features = 30, in_slice_features = 8, out_features=9)


        #attention mechanism
        # self.pool4mlp = Pool4MLP_attention(pool_feature=3, pool_slices=8)
        # self.mlp = MLP(hidden_sizes=[256, 128], in_features = 8, out_features=9)

        # more slice features and 16x16
        # self.pool4mlp = Pool4MLP(pool_feature=30, pool_slices=8)
        # self.mlp = MLP(hidden_sizes=[256, 128], in_channel_features = 30, in_slice_features = 8, out_features=9, plane_features=16)
       
        # self.pool4mlp = Pool4MLP(pool_feature=30, pool_slices=15)
        # self.mlp = MLP(hidden_sizes=[256, 128], in_channel_features = 30, in_slice_features = 15, out_features=9, plane_features=16)

    #     self.pool4mlp = Pool4MLP(pool_feature=30, pool_slices=15)
    #     self.mlp = MLP(hidden_sizes=[256, 128], in_channel_features = 30, in_slice_features = 15, out_features=5, plane_features=32)

        # self.pool4mlp = Pool4MLP(pool_feature=30, pool_slices=15)
      # # self.mlp = MLP(hidden_sizes=[256, 128], in_channel_features = 30, in_slice_features = 15, out_features=9, plane_features=32)
        # self.mlp = MLP(hidden_sizes=[256, 128], in_channel_features = 30, in_slice_features = 15, out_features=6, plane_features=32)

        self.drop_out = 0
        
    #     self.pool4mlp = Pool4MLP(pool_feature=30, pool_slices=15)
    #    # self.mlp = MLP(hidden_sizes=[256, 128], in_channel_features = 30, in_slice_features = 15, out_features=9, plane_features=32)
    #     self.mlp = MLP_drop_out(hidden_sizes=[256, 128], in_channel_features = 30, in_slice_features = 15, out_features=6, plane_features=32, dropout=self.drop_out)


      #  self.pool4mlp = Pool4MLP(pool_feature=30, pool_slices=15)
    #    # self.mlp = MLP(hidden_sizes=[256, 128], in_channel_features = 30, in_slice_features = 15, out_features=9, plane_features=32)
      #  self.mlp = MLP_norm(hidden_sizes=[256, 128], in_channel_features = 30, in_slice_features = 15, out_features=6, plane_features=32, dropout=self.drop_out)


        # self.pool4mlp = Pool4MLP(pool_feature=30, pool_slices=15)

        # self.mlp = MLP_norm(hidden_sizes=[512, 512, 256, 128, 64], in_channel_features = 30, in_slice_features = 15, out_features=10, plane_features=64, dropout=self.drop_out)
      #  self.pool4mlp = Pool4MLP(pool_feature=30, pool_slices=15)
    #    # self.mlp = MLP(hidden_sizes=[256, 128], in_channel_features = 30, in_slice_features = 15, out_features=9, plane_features=32)
       # self.mlp = MLP_norm(hidden_sizes=[256, 128], in_channel_features = 30, in_slice_features = 15, out_features=10, plane_features=64, dropout=self.drop_out)
      #  self.mlp = MLP_norm(hidden_sizes=[512, 256, 128], in_channel_features = 30, in_slice_features = 15, out_features=10, plane_features=64, dropout=self.drop_out)
       # self.mlp = MLP_norm(hidden_sizes=[512, 256, 128, 64], in_channel_features = 30, in_slice_features = 15, out_features=10, plane_features=64, dropout=self.drop_out)
     #   self.mlp = MLP_norm(hidden_sizes=[512, 512, 128], in_channel_features = 30, in_slice_features = 15, out_features=10, plane_features=64, dropout=self.drop_out)
      #  self.mlp = MLP_norm(hidden_sizes=[512, 512, 256, 128], in_channel_features = 30, in_slice_features = 15, out_features=10, plane_features=64, dropout=self.drop_out)
       # self.mlp = MLP_norm(hidden_sizes=[512, 512, 256, 128, 64], in_channel_features = 30, in_slice_features = 15, out_features=10, plane_features=64, dropout=self.drop_out)

        # # LARGE
        # self.pool4mlp = Pool4MLP(pool_feature=30, pool_slices=15)
        # self.mlp = MLP_norm(hidden_sizes=[512, 512, 256, 128, 64], in_channel_features = 30, in_slice_features = 15, out_features=10, plane_features=64, dropout=self.drop_out)


        # # SMALL
        self.pool4mlp = Pool4MLP(pool_feature=30, pool_slices=15)
        self.mlp = MLP_norm(hidden_sizes=[256, 128], in_channel_features = 30, in_slice_features = 15, out_features=10, plane_features=64, dropout=self.drop_out)

        # MEDIUM 
        # self.pool4mlp = Pool4MLP(pool_feature=30, pool_slices=15)
        # self.mlp = MLP_norm(hidden_sizes=[256, 128, 64], in_channel_features = 30, in_slice_features = 15, out_features=10, plane_features=64, dropout=self.drop_out)

    def forward(self, x):
            
            # DEFINE SIZES
            if(self.crop):
                slices, ALL_STACKS = x[0],x[1]
                slice_input_size = int(x[0].shape[3])
                xs, mask = slices.tensor_split(2,1) # split into two across dimension 1 to get image, mask
            else:
                xs, mask = x.tensor_split(2,1) # split into two across dimension 1 to get image, mask
            if(self.only_input_mask):
                xs = mask

            skips = [None] * len(self.unets.enc_blocks)
            masks = [mask] * len(self.unets.enc_blocks)
            sizes = [[xs.shape[3], xs.shape[3], xs.shape[3]]] * len(self.unets.enc_blocks)
            
            # DOWN SAMPLING
            
            if (self.larger_image):
             #   self.unets.enc_blocks = self.unets.enc_blocks[0:3]
                self.unets.enc_blocks = self.unets.enc_blocks[0:2]
            for d in range(len(self.unets.enc_blocks)):
            
                skips[d] = xs = self.unets.enc_blocks[d](xs) # downsampling arm
                sizes[d] = [sizes[d - 1][i] // self.strides[d][i] for i in range(len(self.strides[d]))]
                masks[d] = F.interpolate(masks[d - 1], size=skips[d].shape[2:], mode='trilinear', align_corners=True).gt(0).float()
                
                # indices = torch.linspace(0, xs.shape[1] - 1, steps=10).long()
                # out_dir = "nifti_with_rots"
                # os.makedirs(out_dir, exist_ok=True)

                # for i, idx in enumerate(indices):
                #     vol = xs[0, idx]            # [192, 8, 8]
                #     vol_np = vol.cpu().numpy()

                #     nii = nib.Nifti1Image(vol_np, affine=np.eye(4))
                #     nib.save(nii, f"{out_dir}/vol_{idx.item():04d}_{d}.nii.gz")
        


            # ENCODER + INITIALIZE SHAPES
            flow = torch.zeros([xs.shape[0], xs.ndim - 2] + list(xs.shape[2:]), device=xs.device)
            mask = torch.ones([xs.shape[0], 1] + list(xs.shape[2:]), device=xs.device)
            x3 = torch.zeros([xs.shape[0], xs.shape[1]] + sizes[-1], device=xs.device)

            
            idx_stack1 = (ALL_STACKS[0][:, 0, 0] == 1).nonzero(as_tuple=False)[-1].item()+1
            idx_stack2 = (ALL_STACKS[0][:, 1, 0] == -1).nonzero(as_tuple=False)[-1].item()+1
            idx_stack3 = (ALL_STACKS[0][:, 2, 0] == 1).nonzero(as_tuple=False)[-1].item()+1

            idx_stack1, idx_stack2, idx_stack3 = sorted([idx_stack1, idx_stack2, idx_stack3])
            
            
            xs_stack1 = xs[:, :, :idx_stack1]
            xs_stack2 = xs[:, :, idx_stack1:idx_stack2]
            xs_stack3 = xs[:, :, idx_stack2:]



 
            x_pool1 = self.pool4mlp(xs_stack1)
            x_pool2 = self.pool4mlp(xs_stack2)
            x_pool3 = self.pool4mlp(xs_stack3)

            x_pool = torch.cat((x_pool1, x_pool2, x_pool3), dim=0)
            
            out_rot = self.mlp(x_pool)

            
            return out_rot
            


class Flow_SNet_multi(nn.Module):
    def __init__(self, *args, X=3, spacing=1, num_conv_per_flow=4, slice=[0,1,2],drop=0, rigid=False, crop=False,**kwargs):
        super().__init__()
        conv_kernel_sizes = [[1,3,3],[3,1,3],[3,3,1]]
        pool_kernel_sizes = [[1,2,2],[2,1,2],[2,2,1]]
        self.rigid = rigid # originally False
        self.crop = crop
        self.spacing = spacing
        slab_stride_sizes = [[spacing,1,1],[1,spacing,1],[1,1,spacing]] # with no repetition should be 1?
        slab_kernel_sizes = [[spacing,3,3],[3,spacing,3],[3,3,spacing]] # with no repetition should be 1?

        self.slice = slice
        slice = 0

        # defines 2D U-net (encoder), put all slices in the first dimension so use slice = 0
        self.unets = Flow_UNet_3stacks(*args, conv_kernel_sizes=conv_kernel_sizes[slice], pool_kernel_sizes=pool_kernel_sizes[slice],
                               slab_kernel_sizes=slab_kernel_sizes[slice], slab_stride_sizes=slab_stride_sizes[slice], 
                               mask=True, dropout_p=drop, num_conv_per_flow=num_conv_per_flow, X=X, **kwargs)
        
        # defines 3D U-net (decoder), use 3D convolution
        self.unet3 = Flow_UNet_3stacks(*args, conv_kernel_sizes=3, pool_kernel_sizes=2, mask=True, dropout_p=drop, 
                               num_conv_per_flow=num_conv_per_flow, normalize_splat=False, X=X, **kwargs)
        self.strides = [self.unet3.enc_blocks[d].pool_stride for d in range(len(self.unet3.enc_blocks))]

        self.unet3.flo_blocks = self.unet3.enc_blocks = None # unsure what this is
        self.count = 0 # if want to start applying projection after some iteration

    def forward(self, x):
            
            # DEFINE SIZES
            
            slices, ALL_STACKS = x[0],x[1]
            slice_input_size = int(x[0].shape[3])
            xs, mask = slices.tensor_split(2,1) # split into two across dimension 1 to get image, mask


            skips = [None] * len(self.unets.enc_blocks)
            masks = [mask] * len(self.unets.enc_blocks)
            sizes = [[xs.shape[3], xs.shape[3], xs.shape[3]]] * len(self.unets.enc_blocks)
            
            # ENCODER OF U-NET
            for d in range(len(self.unets.enc_blocks)):
                skips[d] = xs = self.unets.enc_blocks[d](xs) # downsampling arm
                sizes[d] = [sizes[d - 1][i] // self.strides[d][i] for i in range(len(self.strides[d]))]
                masks[d] = F.interpolate(masks[d - 1], size=skips[d].shape[2:], mode='trilinear', align_corners=True).gt(0).float()


            # ENCODER + INITIALIZE SHAPES
            flow = torch.zeros([xs.shape[0], xs.ndim - 2] + list(xs.shape[2:]), device=xs.device)
            mask = torch.ones([xs.shape[0], 1] + list(xs.shape[2:]), device=xs.device)
            x3 = torch.zeros([xs.shape[0], xs.shape[1]] + sizes[-1], device=xs.device)

            # DEFINE PSF
            if(not self.crop):
                slice_input_size = 128
    
            if( self.crop and slice_input_size == 128):
                psf_vals = torch.ones((1,2))*0.5
                psf_vals = psf_vals.to(device=xs.device)
                psf_coords = torch.zeros((3,2))
                psf_coords[0,0] = 0 #1 
                psf_coords[0,1] = 1 #2
                psf_coords = psf_coords.to(device=xs.device)
                psf=(psf_vals, psf_coords)

    
            if( self.crop and slice_input_size == 256):
                psf_vals = torch.ones((1,4))*0.25
                psf_vals = psf_vals.to(device=xs.device)
                psf_coords = torch.zeros((3,4))
                psf_coords[0,0] = 0 #1 
                psf_coords[0,1] = 1 #2
                psf_coords[0,2] = 2 #2
                psf_coords[0,3] = 3 #2
                psf_coords = psf_coords.to(device=xs.device)
                psf=(psf_vals, psf_coords)

            
            # DIFFERENT EXPERIMENTAL SETTINGS
            range_loop = range(len(self.unet3.dec_blocks))
            ONE_LAYER_ONLY = False
            TWO_LAYER_ONLY = False
            THREE_LAYER_ONLY = False
            OT_FALSE = False
            AFFINE_LOSS = False
            SLICE_LOSS = False
            MULTI_SCALE_LOSS = True
            MULTI_SCALE_LOSS_REAL = False
            
            simplified_architecture = True # True for last run 
            simplified_architecture2 = False
            extra_bottleneck = True
            correct_mask = False 
            MIDDLE_PROJECT_ONLY = True # True for last run
         
            SAVE_INTER = False
            STOP_LAYER = 0
       
            # CHANGE NUMBER OF LAYERS
            all_flows = []
            if(extra_bottleneck):
                range_loop = range(len([0,1,2,3,4,5]))
                range_loop = [0,1,2,3,4,4]
            else:
                range_loop = range(len(self.unet3.dec_blocks))
            if(ONE_LAYER_ONLY):
                range_loop = range(4,5)
            if(TWO_LAYER_ONLY):
                range_loop = range(2,4)
            if(THREE_LAYER_ONLY):
                range_loop = range(1,4)

            
            
            l = 5 # to index when extra_bottleneck is used
            for u in reversed(range_loop): 
                if(not simplified_architecture2 and not simplified_architecture):
                    xs = self.unets.dec_blocks[u](xs, skips[u])
                shape_sp = [skips[u].shape[-1],skips[u].shape[-1],skips[u].shape[-1]]
                flow_dim = int(slice_input_size/flow.shape[3]) # dimension of flow 
                x_ratio =  int(slice_input_size/shape_sp[0]) # 
                vol_dim = int(slice_input_size/shape_sp[0]) # dimension of splatted vol

                # VOLUME RECON
                if(u==0 and MIDDLE_PROJECT_ONLY): # PROJECT LAST LAYER                    
                    slices = x[0][:,:1]
                    flow= flow[:,0:3] * 2
                    flow = F.interpolate(flow, size=slices.shape[2:], mode='trilinear', align_corners=True) if flow.shape[2:] != slices.shape[2:] else flow  
                    grid_start = make_grid_one(ALL_STACKS[0], [skips[u].shape[3],skips[u].shape[4]],  vol_dim,  [1,x_ratio,x_ratio], device=flow.device)
                    grid_end = make_grid_one(ALL_STACKS[1], [skips[u].shape[3],skips[u].shape[4]],  vol_dim,  [1,x_ratio,x_ratio], device=flow.device)
                    new_grid = grid_start - grid_end
                    flow_tot = flow + new_grid[None]
                    motion_2_3, aff3 = self.project_new_feb4_crop(flow_tot, x[0][:,1:], ALL_STACKS[1], slice_in=0, spacing=1, shape=[flow_tot.shape[2],flow_tot.shape[3],flow_tot.shape[4]], vol_dim=vol_dim,  slice_dim=[1,x_ratio,x_ratio], detach_=True)
                  
                    
                    if correct_mask == True:
                        mask_correct = x[0][:,1:]
                    else:
                        mask_correct = torch.ones_like(mask)
                    splat = self.unet3.splat.apply_flow_thick(skips[u], aff3,ALL_STACKS[1], flow_dim, vol_dim, shape_sp, [1,x_ratio,x_ratio], mask=mask_correct,  mode='bilinear', psf=psf) #item[0][None].shape[-3:])
                    splat = splat[:,:-1] / splat[:,-1:].detach().flatten(2).max(axis=2)[0][...,None,None,None].expand(splat[:,:-1].shape) # normalize #splat[:,-1:].max().item() 

                # VOLUME RECON
                else:
                    if correct_mask == True:
                        mask_correct = x[0][:,1:][:,:,:,::x_ratio,::x_ratio]
                    else:
                        mask_correct = torch.ones_like(mask)
                    #    def apply_flow_thin(self,x, flow, ALL_STACKS, flow_dim, vol_dim, volume_shape, slice_dim, mask=None, mode='trilinear',ot=True):

                    splat = self.unet3.splat.apply_flow_thin(skips[u], flow, ALL_STACKS[0], flow_dim, vol_dim, shape_sp, [1,x_ratio,x_ratio], mask=mask_correct,  mode='bilinear') #item[0][None].shape[-3:])
                    splat = splat[:,:-1] / splat[:,-1:].detach().flatten(2).max(axis=2)[0][...,None,None,None].expand(splat[:,:-1].shape) # normalize #splat[:,-1:].max().item() 

                # SAVE INPUT FOR DEBUGGING PURPOSES
                if(SAVE_INTER and x_ratio==16 and u==STOP_LAYER): 
                    return self.save_inter_debug(x, flow, flow_dim, x_ratio, ALL_STACKS, psf, 'aff_init_img2_v1_B_my_model.pth', 'splat_init_img2_v1_B_my_model_aug12.nii.gz')
                    


                # APPLY BOTTLENECK STEP
                if(simplified_architecture):
                   # x3 = splat
                    if not extra_bottleneck:
                        x3 = self.unet3.dec_blocks[u](splat, splat)
                    if extra_bottleneck:
                       x3 = self.unet3.dec_blocks[l](splat, splat)

                elif(simplified_architecture2):
                    x3 = splat

                else:
                    x3 = self.unet3.dec_blocks[u](x3, splat)
                
                del splat
                slice_shape = shape_sp.copy()
                slice_shape[0] = slice_shape[0]*3
    
               # SLICING OPERATION
                if(u==0 and  MIDDLE_PROJECT_ONLY):
                    # mask_correct = torch.ones_like(mask)
                    if correct_mask == True:
                        mask_correct = x[0][:,1:][:,:,:,::x_ratio,::x_ratio]
                    else:
                        mask_correct = torch.ones_like(mask)
                    xw = self.unet3.warp.apply_flow_thick(x3, aff3, ALL_STACKS[1], flow_dim, vol_dim, shape_sp, [1,x_ratio,x_ratio], mask=mask_correct,  mode='bilinear',  psf=psf) #item[0][None].shape[-3:])

                else:
                    if correct_mask == True:
                        mask_correct = x[0][:,1:][:,:,:,::x_ratio,::x_ratio]
                    else:
                        mask_correct = torch.ones_like(mask)
                    xw = self.unet3.warp.apply_flow_thin(x3, flow, ALL_STACKS[0], flow_dim, vol_dim, shape_sp, [1,x_ratio,x_ratio], mask=mask_correct,  mode='bilinear') #item[0][None].shape[-3:])
                
                del x3



                # REFINE FLOW
                if(simplified_architecture):
                    if not(extra_bottleneck):
                        
                        skips_new = self.unets.dec_blocks[u](skips[u], skips[u])
                        flow, mask = self.unets.flow_add(flow, self.unets.flo_blocks[u](torch.cat([skips_new, xw], 1)))
                        del skips_new

                    if extra_bottleneck:
                        skips_new = self.unets.dec_blocks[l](skips[u], skips[u])  
                        flow, mask = self.unets.flow_add(flow, self.unets.flo_blocks[l](torch.cat([skips_new, xw], 1)))
                        del skips_new

                        l = l - 1

                elif(simplified_architecture2):
                    skips_new = skips[u]
          
                    flow, mask = self.unets.flow_add(flow, self.unets.flo_blocks[u](torch.cat([skips_new, xw], 1)))
                else:

                    flow2, mask = self.unets.flow_add(flow, self.unets.flo_blocks[u](torch.cat([xs, xw], 1)))                 
                    flow = flow2

                del xw
            


                # MULTISCALE LOSS
                if (MULTI_SCALE_LOSS or MULTI_SCALE_LOSS_REAL):
                    out_f = 1
                    if(x_ratio!=16):
                        flow_dim_ = flow_dim/2
                    else:
                        flow_dim_ = flow_dim
                    if (x_ratio!=1):
                        scale_num = int(flow_dim_/out_f )
                        flow_ = flow[:,0:3] * scale_num
                    else:
                        scale_num = 1
                        flow_ = flow[:,0:3]
                    if(MULTI_SCALE_LOSS):
                        flow_  = F.interpolate(flow_, size=[ flow.shape[2],flow.shape[3]*scale_num,flow.shape[4]*scale_num], mode='trilinear', align_corners=True)
                    if(x_ratio<16):
                        all_flows.append(flow_)




                if(SAVE_INTER and u==STOP_LAYER):
                     return self.save_inter_debug(x, flow, flow_dim, x_ratio, ALL_STACKS, psf, f'aff_{x_ratio}_img2_v1_B.pth', f'splat_{x_ratio}_img2_v1_B__aug12.nii.gz')
              #  pdb.set_trace()

            if(self.rigid == False): #  if(v9 and self.rigid == False and ortho3==True):
                project_try = False
                if(ONE_LAYER_ONLY or OT_FALSE):
       
                    if(SLICE_LOSS):
       
                        x_f_ratio = 1
                        out_f = 1
                       # x[0] = skips[u]
                     
                        grid_start = make_grid_one(ALL_STACKS[0], [ x[0][:,1:].shape[3], x[0][:,1:].shape[4]],  out_f,  [1,x_f_ratio,x_f_ratio], device=flow.device)
                        grid_end = make_grid_one(ALL_STACKS[1], [ x[0][:,1:].shape[3], x[0][:,1:].shape[4]],  out_f,  [1,x_f_ratio,x_f_ratio], device=flow.device)
                        new_grid = grid_start - grid_end
     
                        flow_tot = flow + new_grid[None]
                        motion_save, aff_save = self.project_new_feb4_crop(flow_tot, x[0][:,1:][:,:,:,:,:], ALL_STACKS[1], slice_in=0, spacing=1, shape=[flow_tot.shape[2],flow_tot.shape[3],flow_tot.shape[4]], vol_dim=out_f,  slice_dim=[1,x_f_ratio,x_f_ratio], detach_=True)
                        splat_inter = self.unet3.splat.apply_flow_thick(x[0][:,:1], aff_save,ALL_STACKS[1], 1, 1, [x[0][:,1:].shape[3],x[0][:,1:].shape[3],x[0][:,1:].shape[3]], [1,x_f_ratio,x_f_ratio], mask=x[0][:,1:][:,:,:,:,:],  mode='bilinear', psf=psf) #item[0][None].shape[-3:])
                        splat_inter = splat_inter[:,:-1] / (splat_inter[:,-1:] + 1e-12 * splat_inter[:,-1:].max().item())
                        flow_dim = 1
                        xw = self.unet3.warp.apply_flow_thin(splat_inter, flow, ALL_STACKS[0], flow_dim, vol_dim, shape_sp, [1,x_ratio,x_ratio], mask=x[0][:,1:][:,:,:,:,:],  mode='bilinear') #item[0][None].shape[-3:])
                      

                        weight = 10000
                        slice_diff = torch.mean((xw*x[0][0,1]-x[0][0,0])**2)*weight

                        return (flow, slice_diff)
    
                    elif(AFFINE_LOSS):

                        return (flow, ALL_STACKS[0])
                    elif(MULTI_SCALE_LOSS or MULTI_SCALE_LOSS_REAL):

                        return all_flows
                    else:

                        return flow
            
          
                flow_to_ortho = self.generate_initial_flow(flow)
                flow_tot = flow + flow_to_ortho[None]

                grid_start = make_grid_one(ALL_STACKS[0], [skips[u].shape[3],skips[u].shape[4]],  vol_dim,  [1,x_ratio,x_ratio], device=flow.device)
                grid_end = make_grid_one(ALL_STACKS[1], [skips[u].shape[3],skips[u].shape[4]],  vol_dim,  [1,x_ratio,x_ratio], device=flow.device)
                new_grid = grid_start - grid_end

                
                flow_tot = flow + new_grid[None]


                tr_count = 10
                if(project_try and self.count>tr_count):                    
                    flow_tot2, aff3 = self.project_new_feb4_crop(flow_tot, x[0][:,1:], ALL_STACKS[1], slice_in=0, spacing=1, shape=[flow_tot.shape[2],flow_tot.shape[3],flow_tot.shape[4]], vol_dim=vol_dim,  slice_dim=[1,x_ratio,x_ratio])

                    l2_dif = torch.mean((flow_tot2 - flow_tot) ** 2)
     
                    if(l2_dif<200):

                        flow_tot = flow_tot2
                    else:
                        flow_tot = flow + new_grid[None]

                self.count = self.count + 1
                return flow_tot
              #  return flow_tot #.flip(1) #.flip(1) # og no flip
            
            if( self.rigid == True ):
          #  if(v9):

                self.count = self.count + 1

                flow_to_ortho = self.generate_initial_flow(flow)
                flow_tot = flow + flow_to_ortho[None]
              #  tr_project = 12000 #6000 #3000 # orig 300, 3000'
                tr_project = 300



                if self.count>=tr_project:   # USED WRONG MASK

                    input_flow = flow_tot[:,0:3,0:128,:,:]
                    mask1 = x[:,:1,0:128]
                    motion_2_1, _ = self.project_new_feb4(input_flow,mask1, slice_in=0, spacing=1,shape=[128,128,128])
                    
                    input_flow = flow_tot[:,0:3,128:256,:,:]
                    mask1 = x[:,:1,128:256]
                    motion_2_2, _ = self.project_new_feb4(input_flow,mask1, slice_in=0, spacing=1, shape=[128,128,128])
                    
                    input_flow = flow_tot[:,0:3,256:384,:,:]
                    mask1 = x[:,:1,256:384]
                    motion_2_3, _ = self.project_new_feb4(input_flow,mask1, slice_in=0, spacing=1, shape=[128,128,128])
           
                    motion_2 = torch.cat((motion_2_1,motion_2_2,motion_2_3),dim=2)
                    flow_tot = motion_2
                return flow_tot
            


    def flow_to_orthogonal2(self, vol_n):


      #  pdb.set_trace()
        if(self.slice == [0,1,2]):
           
            ss = vol_n.shape[2]//3

            
            flow_new = torch.zeros((3,vol_n.shape[2],vol_n.shape[3],vol_n.shape[4])).to(vol_n.device)
           
            shape  = [vol_n.shape[2]//3,vol_n.shape[3],vol_n.shape[4]]
            for sl in range(3):

                if (sl == 0):
                    affine = torch.eye(4)

                if (sl == 1): 
                    affine = torch.eye(4)
                    affine[0,0] = 0
                    affine[1,1] = 0
                    affine[0,1] = -1 #1
                    affine[1,0] = 1 #-1
            
                    rot_90 = torch.eye(4)
                    rot_90[1,2] = 1
                    rot_90[2,1] = -1
                    rot_90[2,2] = 0
                    rot_90[1,1] = 0
            
                    affine = affine @ rot_90

                if (sl == 2):
                    affine = torch.eye(4)
                    #affine[0,0] = 0
                    #affine[0,2] = -1
                # affine[2,0] = 1
                # affine[2,2] = 0
                    affine[0,0] = 0
                    affine[0,2] = -1
                    affine[2,0] = 1 #1
                    affine[2,2]= 0 #-1
                
                affine[0,3] = (shape[0]-1)/2
                affine[1,3] = (shape[1]-1)/2
                affine[2,3] =(shape[2]-1)/2
            
                t = affine[0:3,3]    
                aff_m = affine[:3,:3]
                d = aff_m @ t
                affine[0:3,3] = t-d
            
                ans = affine_flow(affine, shape).movedim(-1, 0).to(vol_n.device) #.flip(0)
                flow_new[:,ss*sl:ss*(sl+1),:,:] = ans

            #flow_new_p2 = torch.cat((flow_new,torch.ones_like(flow_new[0][None])), axis=0)
            return flow_new
        if(self.slice == [0,0,0]):
            flow_new = torch.zeros((3,vol_n.shape[2],vol_n.shape[3],vol_n.shape[4])).to(vol_n.device)
            return flow_new
        
    def generate_initial_flow(self, vol_n):

      #  pdb.set_trace()
        if(self.slice == [0,1,2]):
           
            ss = vol_n.shape[2]//3

            
            flow_new = torch.zeros((3,vol_n.shape[2],vol_n.shape[3],vol_n.shape[4])).to(vol_n.device)
           
            shape  = [vol_n.shape[2]//3,vol_n.shape[3],vol_n.shape[4]]
            for sl in range(3):

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


                affine[0,3] = (shape[0]-1)/2
                affine[1,3] = (shape[1]-1)/2
                affine[2,3] =(shape[2]-1)/2
            
                t = affine[0:3,3]    
                aff_m = affine[:3,:3]
                d = aff_m @ t
                affine[0:3,3] = t-d
            
                ans = affine_flow(affine, shape).movedim(-1, 0).to(vol_n.device) #.flip(0)
                flow_new[:,ss*sl:ss*(sl+1),:,:] = ans

            #flow_new_p2 = torch.cat((flow_new,torch.ones_like(flow_new[0][None])), axis=0)
            return flow_new
        if(self.slice == [0,0,0]):
            flow_new = torch.zeros((3,vol_n.shape[2],vol_n.shape[3],vol_n.shape[4])).to(vol_n.device)
            return flow_new              
            # ORIGINAL

           # return flow.flip(1) #.unsqueeze(2) #torch.stack(flow.flip(1).split(1,0), 2)
    

    def undo_flow_turn(self, flow_og): #input is flow 768x256x256
        ss = flow_og.shape[2]//3 #1x3x768x256x256
        new_flow = torch.zeros_like(flow_og)
       # print(new_flow.shape)
        flow0 = flow_og[:,:,0:ss,:,:]
        flow1 = flow_og[:,:,ss:ss*2,:,:]
        flow2 = flow_og[:,:,ss*2:ss*3,:,:]
    
    
        new_flow[:,:,0:ss,:,:] = flow0
        new_flow[:,:,ss:ss*2,:,:] =  torch.rot90(flow1, k=2, dims=(2, 3)) 
       # new_flow[:,:,ss:ss*2,:,:] =  torch.rot90(flow1, k=-1, dims=(1, 2)) 
    
        flow_2_init  = torch.rot90(flow2, k=2, dims=(2, 4))
        flow_2_f = torch.rot90(flow_2_init, k=-2, dims=(2, 3))
        
        #flow_2_init = torch.rot90(flow2, k=1, dims=(1, 2))
       # flow_2_f  = torch.rot90(flow_2_init, k=-1, dims=(1, 3))
        
        new_flow[:,:,ss*2:ss*3,:,:] = flow_2_f
    
        return new_flow
    def flow_turn(self, flow_og): #input is flow 768x256x256
        ss = flow_og.shape[2]//3 #1x3x768x256x256
        new_flow = torch.zeros_like(flow_og)
       # print(new_flow.shape)
        flow0 = flow_og[:,:,0:ss,:,:]
        flow1 = flow_og[:,:,ss:ss*2,:,:]
        flow2 = flow_og[:,:,ss*2:ss*3,:,:]
    
    
        new_flow[:,:,0:ss,:,:] = flow0
        new_flow[:,:,ss:ss*2,:,:] =  torch.rot90(flow1, k=-2, dims=(2, 3)) 
       # new_flow[:,:,ss:ss*2,:,:] =  torch.rot90(flow1, k=-1, dims=(1, 2)) 
    
    
        flow_2_init = torch.rot90(flow2, k=2, dims=(2, 3))
        flow_2_f  = torch.rot90(flow_2_init, k=-2, dims=(2, 4))
        #flow_2_init = torch.rot90(flow2, k=1, dims=(1, 2))
       # flow_2_f  = torch.rot90(flow_2_init, k=-1, dims=(1, 3))
        
        new_flow[:,:,ss*2:ss*3,:,:] = flow_2_f

        return new_flow

    def upsample_flow_old(self, stack, spacing=2):
        self.slice = 0
        stack = stack.squeeze(0) #[0,:,0] # 3
        stack = stack.movedim(1 + self.slice, 0).unflatten(0, [-1, spacing]).movedim(1,-1) 

        shape = [spacing * s + 1 for s in stack.shape[-3:]] #[257, 257, 5]
        stack = torch.cat([stack, stack[...,:1].lerp(stack[...,1:], 2.0)], -1) # torch.Size([64, 3, 128, 128, 3])
        stack = torch.nn.functional.pad(stack, (0,0,0,1,0,1)) #torch.Size([64, 3, 129, 129, 3])

        stack = torch.nn.functional.interpolate(stack, size=shape, mode='trilinear', align_corners=True) 
        stack = stack[...,:-1,:-1,:-1].movedim(-1,1).flatten(0,1).movedim(0, 1 + self.slice)
        stack = stack.unsqueeze(0)
        
        return spacing * stack

    def upsample_flow_new(self, stack, spacing):
            # stack [1, 3, 192, 64, 64
       # pdb.set_trace()
        self.slice = 0
        stack = stack.squeeze(0) #[0,:,0] # 3
        stack = stack.movedim(1 + self.slice, 0).unflatten(0, [-1, spacing]).movedim(1,-1) 
            # stack 192, 3, 64, 64, 1]
     #   pdb.set_trace()

        shape = [spacing * s + 1 for s in stack.shape[-3:]] #[65, 65, 2] # add one to start and end
       # pdb.set_trace()
        stack = torch.cat([stack, stack[...,:1].lerp(stack[...,1:], 2.0)], -1) # torch.Size([192, 3, 64, 64, 1])
        stack = torch.nn.functional.pad(stack, (0,0,0,1,0,1)) #orch.Size([192, 3, 65, 65, 1])
       # pdb.set_trace()

        stack = torch.nn.functional.interpolate(stack, size=shape, mode='trilinear', align_corners=True) 
        stack = stack[...,:-1,:-1,:-1].movedim(-1,1).flatten(0,1).movedim(0, 1 + self.slice)
        stack = stack.unsqueeze(0)
        
        return spacing * stack


    def save_inter_debug(self, x, flow, flow_dim, x_ratio, ALL_STACKS, psf, aff_name, nii_name):
        x_f_ratio = 1
        out_f = 1
        if(x_ratio!=16):
            flow_dim_ = flow_dim/2
        else:
            flow_dim_ = flow_dim
        grid_start = make_grid_one(ALL_STACKS[0], [ x[0][:,1:].shape[3], x[0][:,1:].shape[4]],  out_f,  [1,x_f_ratio,x_f_ratio], device=flow.device)
        grid_end = make_grid_one(ALL_STACKS[1], [ x[0][:,1:].shape[3], x[0][:,1:].shape[4]],  out_f,  [1,x_f_ratio,x_f_ratio], device=flow.device)
        new_grid = grid_start - grid_end
        if(x_ratio!=1):
            scale_num = int(flow_dim_/out_f )
            flow_ = flow[:,0:3] * scale_num
            flow_  = F.interpolate(flow_, size=[ flow.shape[2],flow.shape[3]*scale_num,flow.shape[4]*scale_num], mode='trilinear', align_corners=True)
        else:
            flow_ = flow.clone()
        flow_tot = flow_ + new_grid[None]
        
        motion_save, aff_save = self.project_new_feb4_crop(flow_tot, x[0][:,1:][:,:,:,:,:], ALL_STACKS[1], slice_in=0, spacing=1, shape=[flow_tot.shape[2],flow_tot.shape[3],flow_tot.shape[4]], vol_dim=out_f,  slice_dim=[1,x_f_ratio,x_f_ratio])
        torch.save(aff_save.detach().cpu(), aff_name)
        splat_inter = self.unet3.splat.apply_flow_thick(x[0][:,:1][:,:,:,:,:], aff_save,ALL_STACKS[1], 1, 1, [x[0][:,1:].shape[3],x[0][:,1:].shape[3],x[0][:,1:].shape[3]], [1,x_f_ratio,x_f_ratio], mask=x[0][:,1:][:,:,:,:,:],  mode='bilinear', psf=psf) #item[0][None].shape[-3:])
        splat_inter = splat_inter[:,:-1] / (splat_inter[:,-1:] + 1e-12 * splat_inter[:,-1:].max().item())
        nii_image = nib.Nifti1Image(splat_inter[0][0].detach().cpu().numpy(), np.eye(4))  
        nib.save(nii_image , nii_name)

        return flow_tot

    def upsample_flow(self, stack, spacing=2):
      #  pdb.set_trace()
        stack = stack.squeeze(0) #[0,:,0]
        if(stack.shape[1]==384):
            stack0 = stack[:,0:128,:,:]
            stack1 = stack[:,128:256,:,:]
            stack2 = stack[:,256:384,:,:]

        else:
            ss = int(stack.shape[1]//3)
            stack0 = stack[:,0:ss,:,:]
            stack1 = stack[:,ss:ss*2,:,:]
            stack2 = stack[:,ss*2:ss*3,:,:]


         #   stack0 = stack[:,0:64,:,:]
          #  stack1 = stack[:,64:128,:,:]
          #  stack2 = stack[:,128:192,:,:]

        stack0_ = stack0.movedim(1 + 0, 0).unflatten(0, [-1, spacing]).movedim(1,-1)
       # pdb.set_trace()
        stack1_ = stack1.movedim(1 + 1, 0).unflatten(0, [-1, spacing]).movedim(1,-1)
        stack2_ = stack2.movedim(1 + 2, 0).unflatten(0, [-1, spacing]).movedim(1,-1)
       # pdb.set_trace()
        shape = [spacing * s + 1 for s in stack0_.shape[-3:]]
      #  pdb.set_trace()

       # stack = stack.squeeze(0) #[0,:,0]
       # stack = stack.movedim(1 + self.slice, 0).unflatten(0, [-1, spacing]).movedim(1,-1)
       # shape = [spacing * s + 1 for s in stack.shape[-3:]]
        
      #  stack0__ = torch.cat([stack0_, stack0_[...,:1].lerp(stack0_[...,1:], 4)], -1)
      #  stack1__ = torch.cat([stack1_, stack1_[...,:1].lerp(stack1_[...,1:], 4)], -1)
      #  stack2__ = torch.cat([stack2_, stack2_[...,:1].lerp(stack2_[...,1:], 4)], -1)

        stack0__ = torch.cat([stack0_, stack0_[...,:1].lerp(stack0_[...,1:], 2)], -1)
        stack1__ = torch.cat([stack1_,stack1_[...,:1].lerp(stack1_[...,1:], 2)], -1)
        stack2__ = torch.cat([stack2_,stack2_[...,:1].lerp(stack2_[...,1:], 2)], -1)
       # pdb.set_trace()
        stack = torch.cat((stack0__,stack1__,stack2__),dim=0)
      #  pdb.set_trace()


        stack0__ = torch.nn.functional.pad(stack0__, (0,0,0,1,0,1))
        stack1__ = torch.nn.functional.pad(stack1__, (0,0,0,1,0,1))
        stack2__ = torch.nn.functional.pad(stack2__, (0,0,0,1,0,1))
        
       # stack = torch.nn.functional.interpolate(stack, size=shape, mode='trilinear', align_corners=True)
       # pdb.set_trace()
       # stack = stack[...,:-1,:-1,:-1].movedim(-1,1).flatten(0,1).movedim(0, 1 + self.slice)
       # stack = stack.unsqueeze(0)

        stack0 = torch.nn.functional.interpolate(stack0__, size=shape, mode='trilinear', align_corners=True)
        stack0 = stack0[...,:-1,:-1,:-1].movedim(-1,1).flatten(0,1).movedim(0, 1 +0)
        stack0 = stack0.unsqueeze(0)

        stack1 = torch.nn.functional.interpolate(stack1__, size=shape, mode='trilinear', align_corners=True)
        stack1 = stack1[...,:-1,:-1,:-1].movedim(-1,1).flatten(0,1).movedim(0, 1 + 1)
        stack1 = stack1.unsqueeze(0)

        stack2 = torch.nn.functional.interpolate(stack2__, size=shape, mode='trilinear', align_corners=True)
        stack2 = stack2[...,:-1,:-1,:-1].movedim(-1,1).flatten(0,1).movedim(0, 1 + 2)
        stack2 = stack2.unsqueeze(0)

        stack_all = torch.cat((stack0,stack1,stack2),dim=2)

    #    pdb.set_trace()
        
        return spacing * stack_all
    def compensate(self, out, tar):
        batch, chans, *size = out.shape
        grid = torch.stack(torch.meshgrid([torch.arange(1., s + 1., device=out.device) for s in size], indexing='ij'))
        mask = tar[:,chans:] # if masked else torch.ones_like(tar[:,chans:])
        
        B = (out[:,:chans].flip(1) + grid).masked_select(mask.bool()).reshape(batch, chans, -1) #.transpose(1,2)
        A = (tar[:,:chans].flip(1) + grid).masked_select(mask.bool()).reshape(batch, chans, -1) #.transpose(1,2)
        
        mean_B = B.mean(-1, keepdim=True).detach()
        mean_A = A.mean(-1, keepdim=True).detach()
        X = torch.linalg.svd(torch.linalg.lstsq((B - mean_B).transpose(1,2), (A - mean_A).transpose(1,2)).solution.detach())
        R = (X.U @ X.S.sign().diag_embed() @ X.Vh).transpose(1,2)

        out = out.flip(1) + grid - mean_B.unflatten(-1,[1,1,1])
        out = (R @ out.flatten(2)).unflatten(2, out.shape[2:])
        out = (out - grid + mean_A.unflatten(-1,[1,1,1])).flip(1)
     #   pdb.set_trace()
        return out
    def compensate_3stacks(self, out, tar):
        batch, chans, *size = out.shape
        size[0] = size[0]//3
        #grid2 = torch.stack(torch.meshgrid([torch.arange(1., s + 1.) for s in size], indexing='ij'))

        grid = torch.stack(torch.meshgrid([torch.arange(1., s + 1., device=out.device) for s in size], indexing='ij'))
        grid = torch.cat((grid, grid, grid), dim=1)
        mask = tar[:,chans:] # if masked else torch.ones_like(tar[:,chans:])
        
       # B = (out[:,:chans].flip(1) + grid).masked_select(mask.bool()).reshape(batch, chans, -1) #.transpose(1,2)
        #A = (tar[:,:chans].flip(1) + grid).masked_select(mask.bool()).reshape(batch, chans, -1) #.transpose(1,2)
        
        B = (out[:,:chans].flip(1) + grid).masked_select(mask.bool()).reshape(batch, chans, -1) #.transpose(1,2)
        A = (tar[:,:chans].flip(1) + grid).masked_select(mask.bool()).reshape(batch, chans, -1) #.transpose(1,2)
        
        mean_B = B.mean(-1, keepdim=True).detach()
        mean_A = A.mean(-1, keepdim=True).detach()
        X = torch.linalg.svd(torch.linalg.lstsq((B - mean_B).transpose(1,2), (A - mean_A).transpose(1,2)).solution.detach())
        R = (X.U @ X.S.sign().diag_embed() @ X.Vh).transpose(1,2)

        out = out.flip(1) + grid - mean_B.unflatten(-1,[1,1,1])
        out = (R @ out.flatten(2)).unflatten(2, out.shape[2:])
        out = (out - grid + mean_A.unflatten(-1,[1,1,1])).flip(1)
     #   pdb.set_trace()
        return out

    def project_3stacks(self, pred, mask):
     #   print(self.slice, self.spacing)
        batch, chans, *size = pred.shape
        
        ones = torch.ones([1] + size, dtype=pred.dtype, device=pred.device) #_like(grid[0])])
        size[0] = size[0]//3
        #grid2 = torch.stack(torch.meshgrid([torch.arange(1., s + 1.) for s in size], indexing='ij'))

        grid = torch.stack(torch.meshgrid([torch.arange(1., s + 1., device=pred.device) for s in size], indexing='ij'))
        grid = torch.cat((grid, grid, grid), dim=1)
        grid = torch.cat([grid, ones])
        warp = pred + grid[:pred.shape[1]] # torch.cat([pred, 0 * ones.unsqueeze(0)], 1) + grid

        
        tot_size = size.copy()
        tot_size[0] = size[0]*3
     #  pdb.set_trace()
        self.slice = 0
        M = mask.expand([batch, 1] + tot_size).unflatten(2 + self.slice, [-1, self.spacing]).movedim(2 + self.slice, 1).flatten(3).transpose(2,3)
        A = grid.expand([batch, chans + 1] + tot_size).unflatten(2 + self.slice, [-1, self.spacing]).movedim(2 + self.slice, 1).flatten(3).transpose(2,3)
        B = warp.unflatten(2 + self.slice, [-1, self.spacing]).movedim(2 + self.slice, 1).flatten(3).transpose(2,3)
        
        mean_A = torch.sum(A * M, 2, keepdim=True) / torch.sum(M, 2, keepdim=True)
        mean_A[mean_A.isnan()] = 0
        mean_A[...,-1] = 0

        X = torch.linalg.lstsq(M * (A - mean_A), M * (B - mean_A[...,:-1])).solution
        pdb.set_trace()
        R = torch.linalg.svd(X[...,:-1,:]) # 3x3



        ####
        t = X[...,-1:,:] # tx, ty, tz
        P = torch.cat([R.U @ R.S.sign().diag_embed() @ R.Vh, t], 2) #.detach() # 47 x 4 x 3 # 0 0 0 1
     #   pdb.set_trace()
        num_slices = P.shape[1]
        aff_arr = P.transpose(2, 3)
      #  pdb.set_trace()
        affine = torch.zeros((1, num_slices, 4, 4))
        affine[:, :, :3, :] = aff_arr

        affine[:, :, 3, 0:3] = torch.zeros((1, num_slices, 3))
        affine[:, :, 3, 3] = torch.ones((1, num_slices))


        torch.set_printoptions(sci_mode=False)
       # print(affine)
        
        size[self.slice] = self.spacing
        flow = (A - mean_A) @ P - (A - mean_A)[...,:-1]
        flow = flow.transpose(3,2).unflatten(3, size).movedim(1, 2 + self.slice).flatten(2 + self.slice, 3 + self.slice)
      #  pdb.set_trace()
        return flow
    

    def project_new_feb4(self, pred, mask, slice_in=0, spacing=4, shape=[128,128,128], detach_=False):
        
        if True== True:
     #   print("PROJECT THIS ONE")
            spacing = spacing 
            self_slice=slice_in
            batch, chans, *size = pred.shape #size = [192, 64, 64]

            # Define grid and ones to make homog. coords
            ones = torch.ones([1] + size, dtype=pred.dtype, device=pred.device) 
            grid = torch.stack(torch.meshgrid([torch.arange(1., s + 1., device=pred.device) for s in size], indexing='ij')) - 1 # grid shape torch.Size([3, 192, 64, 64])
            
            # Combine grid with homog ones and add to flow
            grid_ones = torch.cat([grid, ones]) # grida shape - torch.Size([4, 192, 64, 64])
        #    print("CLAMPED")
         #   pred = torch.ones_like(pred)
            warp = pred + grid[:pred.shape[1]] # warp torch.Size([1, 3, 192, 64, 64])

            # Flatten mask, grid, and flow
            M = mask.expand([batch, chans] + size).unflatten(2 + self_slice, [-1, spacing]).movedim(2 + self_slice, 1).flatten(3).transpose(2,3)
            A = grid.expand([batch, chans] + size).unflatten(2 + self_slice, [-1, spacing]).movedim(2 + self_slice, 1).flatten(3).transpose(2,3)
            A_ones = grid_ones.expand([batch, chans + 1] + size).unflatten(2 + self_slice, [-1, spacing]).movedim(2 + self_slice, 1).flatten(3).transpose(2,3)
            B = warp.unflatten(2 + self_slice, [-1, spacing]).movedim(2 + self_slice, 1).flatten(3).transpose(2,3) # B shape torch.Size([1, 192, 4096, 3])    
            

            affine_all = torch.zeros((3,shape[0],shape[1],shape[2]), device=pred.device)
            aff_arr = torch.zeros((shape[0],4,4), device=pred.device)
            yes_grad = True
            detach_ = True

            for n in range(shape[0]):
                if(torch.sum(M[0,n].bool())>0):

                    B_filter = B[0,n,:,:].masked_select(M[0,n].bool()).view(1,1,-1,3)
                    A_filter = A[0,n,:,:].masked_select(M[0,n].bool()).view(1,1,-1,3)
                    
                    mean_A_filt = torch.mean(A_filter, 2, keepdim=True)
                    mean_B_filt = torch.mean(B_filter, 2, keepdim=True)

                    B_eq = (B_filter-mean_B_filt)[0,0].T
                    A_eq = (A_filter-mean_A_filt)[0,0].T
                  #  H_eq = A_eq @ B_eq.T
                    H_eq = A_eq.double() @ B_eq.double().T
                 #   with torch.no_grad():
                    if yes_grad:
                        try:
                            eps = 1e-6
                            if torch.isinf(H_eq).any():
                                print("Warning infinite")
                
                            H_eq[torch.isinf(H_eq)] = 1000
                            #print(H_eq)
                         #   H_eq_reg = H_eq + eps * torch.eye(3, device=H_eq.device)
                            U, S, Vh = torch.linalg.svd(H_eq.to(dtype=torch.float32))

                            if( detach_):
                                U = U.detach()
                                S = S.detach()
                                Vh = Vh.detach() # 3x3 R -result of SVD contains U, S, Vh
                            X = Vh.T @ U.T

                            if(torch.det(X.float())<0 ):
                
                                V_new = Vh.T.clone()
                                V_new[0:3,2] = -Vh.T[0:3,2]
                                X = V_new @ U.T

                            t_solve =  (mean_B_filt[0,0].T - X @ mean_A_filt[0,0].T).T

                        except RuntimeError as e:
                            print("Run time error: ")
                            print(e)
                            torch.save(pred.detach().cpu(),'pred_error.pth')
                            X = torch.eye(3, device=pred.device)
                            t_solve = torch.tensor([[0,0,0]],device=pred.device)

                else:
                    X = torch.eye(3, device=pred.device)
                    t_solve = torch.tensor([[0,0,0]],device=pred.device)

                g = identity([1,shape[1],shape[1]])[0].unsqueeze(-1).reshape(shape[1]*shape[1],3).to(pred.device)
                g[:,0] = g[:,0] + n
                
                affine_all[:,n] = ((X @ g.T  + t_solve.T) - g.T).reshape(3,shape[1],shape[1])

                aff_arr[n] = torch.eye(4, device=pred.device)
                aff_arr[n,0:3,0:3] = X            
                aff_arr[n,0:3,3] = t_solve.T[:,0]

        return affine_all[None], aff_arr
    
    def project_new_feb4_crop(self, pred, mask, ALL_STACKS, slice_in=0, spacing=4, shape=[128,128,128], vol_dim=1,  slice_dim=[1,1,1], detach_=False):

        if True== True:

            spacing = spacing 
            self_slice=slice_in
            batch, chans, *size = pred.shape #size = [192, 64, 64]

            # Define grid and ones to make homog. coords
            ones = torch.ones([1] + size, dtype=pred.dtype, device=pred.device) 
            grid = make_grid_one(ALL_STACKS, [pred.shape[3],pred.shape[4]],  vol_dim,  slice_dim, device=ALL_STACKS.device)
    

            # Combine grid with homog ones and add to flow
            grid_ones = torch.cat([grid, ones]) # grida shape - torch.Size([4, 192, 64, 64])
            warp = pred + grid[:pred.shape[1]] # warp torch.Size([1, 3, 192, 64, 64])

            # Flatten mask, grid, and flow
            M = mask.expand([batch, chans] + size).unflatten(2 + self_slice, [-1, spacing]).movedim(2 + self_slice, 1).flatten(3).transpose(2,3)
            A = grid.expand([batch, chans] + size).unflatten(2 + self_slice, [-1, spacing]).movedim(2 + self_slice, 1).flatten(3).transpose(2,3)
            A_ones = grid_ones.expand([batch, chans + 1] + size).unflatten(2 + self_slice, [-1, spacing]).movedim(2 + self_slice, 1).flatten(3).transpose(2,3)
            B = warp.unflatten(2 + self_slice, [-1, spacing]).movedim(2 + self_slice, 1).flatten(3).transpose(2,3) # B shape torch.Size([1, 192, 4096, 3])    


            affine_all = torch.zeros((3,shape[0],shape[1],shape[2]), device=pred.device)
            aff_arr = torch.zeros((shape[0],4,4), device=pred.device)
            yes_grad = True
 

            for n in range(shape[0]):
                if(torch.sum(M[0,n].bool())>0):

                    B_filter = B[0,n,:,:].masked_select(M[0,n].bool()).view(1,1,-1,3)
                    A_filter = A[0,n,:,:].masked_select(M[0,n].bool()).view(1,1,-1,3)

                    mean_A_filt = torch.mean(A_filter, 2, keepdim=True)
                    mean_B_filt = torch.mean(B_filter, 2, keepdim=True)

                    B_eq = (B_filter-mean_B_filt)[0,0].T
                    A_eq = (A_filter-mean_A_filt)[0,0].T


                    #### DEBUG CHECK
                    if torch.isnan(A_filter).any() or torch.isinf(A_filter).any():
                        print(f"NaN/Inf in A_filter at n={n}")
                    if torch.isnan(B_filter).any() or torch.isinf(B_filter).any():
                        print(f"NaN/Inf in B_filter at n={n}")

                    if A_filter.shape[2] == 0 or B_filter.shape[2] == 0:
                        print(f"After masking: empty filter at n={n}")




                #  H_eq = A_eq @ B_eq.T
                    H_eq = A_eq.double() @ B_eq.double().T
                #   with torch.no_grad():
                    if yes_grad:
                        try:
                            eps = 1e-6
                            if torch.isinf(H_eq).any():
                                print("INFINITE")

                            H_eq[torch.isinf(H_eq)] = 1000
                            #print(H_eq)

                            # REGULARIZE
                            H_eq = H_eq + eps * torch.eye(3, device=H_eq.device)
                            # print("New H_eq")
                            # print(H_eq)
                            U, S, Vh = torch.linalg.svd(H_eq.to(dtype=torch.float32))
                            # print("S values")
                            # print(S)
                            if( detach_):
                                U = U.detach()
                                S = S.detach()
                                Vh = Vh.detach() # 3x3 R -result of SVD contains U, S, Vh
                            X = Vh.T @ U.T

                            if(torch.det(X.float())<0 ):

                                V_new = Vh.T.clone()
                                V_new[0:3,2] = -Vh.T[0:3,2]
                                X = V_new @ U.T

                            t_solve =  (mean_B_filt[0,0].T - X @ mean_A_filt[0,0].T).T

                        except RuntimeError as e:
                            print("run time error ")
                            print(e)
                            print(H_eq)
                            torch.save(pred.detach().cpu(),'pred_error.pth')
                            X = torch.eye(3, device=pred.device)
                            t_solve = torch.tensor([[0,0,0]],device=pred.device)

                else:
                    X = torch.eye(3, device=pred.device)
                    t_solve = torch.tensor([[0,0,0]],device=pred.device)

                g = identity([1,shape[1],shape[1]])[0].unsqueeze(-1).reshape(shape[1]*shape[1],3).to(pred.device)
               
                g[:,0] = g[:,0] + ALL_STACKS[n][0,3].item()

                affine_all[:,n] = ((X @ g.T  + t_solve.T) - g.T).reshape(3,shape[1],shape[1])

                aff_arr[n] = torch.eye(4, device=pred.device)
                aff_arr[n,0:3,0:3] = X            
                aff_arr[n,0:3,3] = t_solve.T[:,0]

        return affine_all[None], aff_arr

    def project_two_flows(self, pred_est, pred_gt, mask, ALL_STACKS, slice_in=0, spacing=4, shape=[128,128,128], vol_dim=1,  slice_dim=[1,1,1], detach_=False):

    

        spacing = spacing 
        self_slice=slice_in
        batch, chans, *size = pred_est.shape #size = [192, 64, 64]


        # Define grid and ones to make homog. coords
        ones = torch.ones([1] + size, dtype=pred_est.dtype, device=pred_est.device) 
        grid = make_grid_one(ALL_STACKS, [pred_est.shape[3],pred_est.shape[4]],  vol_dim,  slice_dim, device=ALL_STACKS.device)
        warp = pred_gt[:]+ grid[:pred_est.shape[1]]

        # Combine grid with homog ones and add to flow
        grid_ones = torch.cat([grid, ones]) # grida shape - torch.Size([4, 192, 64, 64])
        grid_gt = pred_est + grid[:pred_est.shape[1]] # warp torch.Size([1, 3, 192, 64, 64])
        
        # Flatten mask, grid, and flow
        M = mask.expand([batch, chans] + size).unflatten(2 + self_slice, [-1, spacing]).movedim(2 + self_slice, 1).flatten(3).transpose(2,3)
        A = grid_gt.expand([batch, chans] + size).unflatten(2 + self_slice, [-1, spacing]).movedim(2 + self_slice, 1).flatten(3).transpose(2,3)
        A_ones = grid_ones.expand([batch, chans + 1] + size).unflatten(2 + self_slice, [-1, spacing]).movedim(2 + self_slice, 1).flatten(3).transpose(2,3)
        B = warp.expand([batch, chans] + size).unflatten(2 + self_slice, [-1, spacing]).movedim(2 + self_slice, 1).flatten(3).transpose(2,3)

        # set up variables
        aff_arr = torch.zeros((4,4), device=pred_est.device)
        yes_grad = True

        # compute SVD
        B_filter = B[0,:,:,:].masked_select(M[0,:].bool()).view(1,1,-1,3)
        A_filter = A[0,:,:,:].masked_select(M[0,:].bool()).view(1,1,-1,3)
        

        mean_A_filt = torch.mean(A_filter, 2, keepdim=True)
        mean_B_filt = torch.mean(B_filter, 2, keepdim=True)

        B_eq = (B_filter-mean_B_filt)[0,0].T
        A_eq = (A_filter-mean_A_filt)[0,0].T

        if torch.isnan(A_filter).any() or torch.isinf(A_filter).any():
            print(f"NaN/Inf in A_filter at n={n}")
        if torch.isnan(B_filter).any() or torch.isinf(B_filter).any():
            print(f"NaN/Inf in B_filter at n={n}")

        if A_filter.shape[2] == 0 or B_filter.shape[2] == 0:
            print(f"After masking: empty filter at n={n}")


    #  H_eq = A_eq @ B_eq.T
        H_eq = A_eq.double() @ B_eq.double().T
    #   with torch.no_grad():
        if yes_grad:
            try:
                eps = 1e-6
                if torch.isinf(H_eq).any():
                    print("INFINITE")

                H_eq[torch.isinf(H_eq)] = 1000
                #print(H_eq)

                # REGULARIZE
                H_eq = H_eq + eps * torch.eye(3, device=H_eq.device)

                U, S, Vh = torch.linalg.svd(H_eq.to(dtype=torch.float32))

                if( detach_):
                    U = U.detach()
                    S = S.detach()
                    Vh = Vh.detach() # 3x3 R -result of SVD contains U, S, Vh
                X = Vh.T @ U.T
                print(X)

                if(torch.det(X.float())<0 ):

                    V_new = Vh.T.clone()
                    V_new[0:3,2] = -Vh.T[0:3,2]
                    X = V_new @ U.T
                t_solve =  (mean_B_filt[0,0].T - X @ mean_A_filt[0,0].T).T

                print("Difference")
                print(torch.sum((mean_B_filt[0,0].T - (X @ mean_A_filt[0,0].T + t_solve.T))**2))

            except RuntimeError as e:
                print("run time error ")
                print(e)
                print(H_eq)
                torch.save(pred_est.detach().cpu(),'pred_est_error.pth')
                X = torch.eye(3, device=pred_est.device)
                t_solve = torch.tensor([[0,0,0]],device=pred_est.device)

        grid_unrolled = grid_gt.reshape(3,shape[0]*shape[1]*shape[2]).to(pred_est.device)
        grid_og = grid[:pred_est.shape[1]].reshape(3,shape[0]*shape[1]*shape[2]).to(pred_est.device)

        affine_all = ((X @ grid_unrolled  + t_solve.T) - grid_og).reshape(3,shape[0],shape[1], shape[2])

        aff_arr = torch.eye(4, device=pred_est.device)
        aff_arr[0:3,0:3] = X            
        aff_arr[0:3,3] = t_solve.T[:,0]

        return affine_all[None], aff_arr

    def project_new_feb4_256_og(self, pred, mask, slice_in=0, spacing=4, shape=[128,128,128]):
        torch.set_printoptions(sci_mode=False)
       # print("PROJECT THIS ONE")
        spacing = spacing
        #set slice 0 since all the slices are turned}
        self_slice=slice_in
        batch, chans, *size = pred.shape #size = [192, 64, 64]
      #  print(batch, chans, *size)

        # Define grid and add it to the warp 
        ones = torch.ones([1] + size, dtype=pred.dtype, device=pred.device) # ones.shape torch.Size([1, 192, 64, 64])
        grid = torch.stack(torch.meshgrid([torch.arange(1., s + 1., device=pred.device) for s in size], indexing='ij')) - 1 # grid shape torch.Size([3, 192, 64, 64])

        grid_ones = torch.cat([grid, ones]) # grida shape - torch.Size([4, 192, 64, 64])
        warp = pred + grid[:pred.shape[1]] # warp torch.Size([1, 3, 192, 64, 64])

        # flatten mask, grid, warp, pred
        #M = mask.expand([batch, chans + 1] + size).unflatten(2 + self_slice, [-1, spacing]).movedim(2 + self_slice, 1).flatten(3).transpose(2,3)
        M = mask.expand([batch, chans] + size).unflatten(2 + self_slice, [-1, spacing]).movedim(2 + self_slice, 1).flatten(3).transpose(2,3)

        A = grid.expand([batch, chans ] + size).unflatten(2 + self_slice, [-1, spacing]).movedim(2 + self_slice, 1).flatten(3).transpose(2,3)
        A_ones = grid_ones.expand([batch, chans + 1] + size).unflatten(2 + self_slice, [-1, spacing]).movedim(2 + self_slice, 1).flatten(3).transpose(2,3)
        B = warp.unflatten(2 + self_slice, [-1, spacing]).movedim(2 + self_slice, 1).flatten(3).transpose(2,3) # B shape torch.Size([1, 192, 4096, 3])    
        W = pred.unflatten(2 + self_slice, [-1, spacing]).movedim(2 + self_slice, 1).flatten(3).transpose(2,3)
    #   pdb.set_trace()

        # Center grid around 0, remove average translation
        mean_A = torch.mean(A, 2, keepdim=True) #/ torch.sum(W, 2, keepdim=True) # mean_A shape torch.Size([1, 192, 1, 4])
        mean_A_ones = torch.mean(A_ones, 2, keepdim=True) #/ torch.sum(W, 2, keepdim=True) # mean_A shape torch.Size([1, 192, 1, 4])
        mean_B = torch.mean(B, 2, keepdim=True) #/ torch.sum(W, 2, keepdim=True) # mean_A shape torch.Size([1, 192, 1, 4])
        #shape = [256,256,256]
        affine_all = torch.zeros((3,shape[0],shape[1],shape[2]), device=pred.device)
        aff_arr = torch.zeros((shape[0],4,4), device=pred.device)
        for n in range(shape[0]):
            if(torch.sum(M[0,n].bool())>0):
            # print("mask")
                #print("mask in use")
                B_filter = B[0,n,:,:].masked_select(M[0,n].bool()).view(1,1,-1,3)
                A_filter = A[0,n,:,:].masked_select(M[0,n].bool()).view(1,1,-1,3)
        
                mean_A_filt = torch.mean(A_filter, 2, keepdim=True)
                mean_B_filt = torch.mean(B_filter, 2, keepdim=True)
        
                B_eq = (B_filter-mean_B_filt)[0,0].T
                A_eq = (A_filter-mean_A_filt)[0,0].T
                H_eq = A_eq @ B_eq.T
        
                SVD_eq = torch.linalg.svd(H_eq) # 3x3 R -result of SVD contains U, S, Vh	
                X = SVD_eq.Vh.T @ SVD_eq.U.T          
                X[torch.abs(X) < 1e-5] = 0


                if(torch.det(X)<0 ):
                    V_new = SVD_eq.Vh.T.clone()
                    V_new[0:3,2] = -SVD_eq.Vh.T[0:3,2]
                    X = V_new @ SVD_eq.U.T
                    X[torch.abs(X) < 1e-5] = 0

                t_solve =  (mean_B_filt[0,0].T - X @ mean_A_filt[0,0].T).T
                t_solve = t_solve*2


            else:
        
                X = torch.eye(3, device=pred.device)
                t_solve = torch.tensor([[0,0,0]],device=pred.device)

            g = identity([1,shape[0],shape[0]])[0].unsqueeze(-1).reshape(shape[0]*shape[0],3).to(pred.device)
            g[:,0] = g[:,0] + n
            affine_all[:,n] = ((X @ g.T  + t_solve.T) - g.T).reshape(3,shape[0],shape[0])
            aff_arr[n] = torch.eye(4, device=pred.device)
            aff_arr[n,0:3,0:3] = X   
            aff_arr[n,0:3,3] = t_solve.T[:,0]
            aff_arr[torch.abs(aff_arr) < 1e-5] = 0

        return affine_all[None], aff_arr
    

    def project_new_feb4_256(self, pred, mask, slice_in=0, spacing=4, shape=[128,128,128], center=False):
        torch.set_printoptions(sci_mode=False)
       # print("PROJECT THIS ONE")

        spacing = spacing
        #set slice 0 since all the slices are turned}
        self_slice=slice_in
        
        
        batch, chans, *size = pred.shape #size = [192, 64, 64]
      #  print(batch, chans, *size)

        # Define grid and add it to the warp 
        ones = torch.ones([1] + size, dtype=pred.dtype, device=pred.device) # ones.shape torch.Size([1, 192, 64, 64])
        grid = torch.stack(torch.meshgrid([torch.arange(1., s + 1., device=pred.device) for s in size], indexing='ij')) - 1 # grid shape torch.Size([3, 192, 64, 64])

        # pdb.set_trace()
        if(center==True):
      
            g_mean0 = grid[0].mean()
            g_mean1 = grid[1].mean()
            g_mean2 =  grid[2].mean()
            grid[0] = grid[0]  - g_mean0
            grid[1] = grid[1] - g_mean1
            grid[2] = grid[2] - g_mean2


        grid_ones = torch.cat([grid, ones]) # grida shape - torch.Size([4, 192, 64, 64])
        warp = pred + grid[:pred.shape[1]] # warp torch.Size([1, 3, 192, 64, 64])

        # flatten mask, grid, warp, pred
        #M = mask.expand([batch, chans + 1] + size).unflatten(2 + self_slice, [-1, spacing]).movedim(2 + self_slice, 1).flatten(3).transpose(2,3)
        M = mask.expand([batch, chans] + size).unflatten(2 + self_slice, [-1, spacing]).movedim(2 + self_slice, 1).flatten(3).transpose(2,3)

        A = grid.expand([batch, chans ] + size).unflatten(2 + self_slice, [-1, spacing]).movedim(2 + self_slice, 1).flatten(3).transpose(2,3)
        A_ones = grid_ones.expand([batch, chans + 1] + size).unflatten(2 + self_slice, [-1, spacing]).movedim(2 + self_slice, 1).flatten(3).transpose(2,3)
        B = warp.unflatten(2 + self_slice, [-1, spacing]).movedim(2 + self_slice, 1).flatten(3).transpose(2,3) # B shape torch.Size([1, 192, 4096, 3])    
        W = pred.unflatten(2 + self_slice, [-1, spacing]).movedim(2 + self_slice, 1).flatten(3).transpose(2,3)
    #   pdb.set_trace()

        # Center grid around 0, remove average translation
        mean_A = torch.mean(A, 2, keepdim=True) #/ torch.sum(W, 2, keepdim=True) # mean_A shape torch.Size([1, 192, 1, 4])
        mean_A_ones = torch.mean(A_ones, 2, keepdim=True) #/ torch.sum(W, 2, keepdim=True) # mean_A shape torch.Size([1, 192, 1, 4])
        mean_B = torch.mean(B, 2, keepdim=True) #/ torch.sum(W, 2, keepdim=True) # mean_A shape torch.Size([1, 192, 1, 4])
        #shape = [256,256,256]
        

        affine_all = torch.zeros((3,shape[0],shape[1],shape[2]), device=pred.device)
        aff_arr = torch.zeros((shape[0],4,4), device=pred.device)
        for n in range(shape[0]):
            if(torch.sum(M[0,n].bool())>0):
            # print("mask")
                #print("mask in use")
                
                B_filter = B[0,n,:,:].masked_select(M[0,n].bool()).view(1,1,-1,3)
                A_filter = A[0,n,:,:].masked_select(M[0,n].bool()).view(1,1,-1,3)
        
                mean_A_filt = torch.mean(A_filter, 2, keepdim=True)
                mean_B_filt = torch.mean(B_filter, 2, keepdim=True)
        
                B_eq = (B_filter-mean_B_filt)[0,0].T
                A_eq = (A_filter-mean_A_filt)[0,0].T
                H_eq = A_eq @ B_eq.T
        
                # B_eq = (B-mean_B)[0,n].T
                # A_eq = (A-mean_A)[0,n].T
                # H_eq = A_eq @ B_eq.T
               # print("H_eq")
                
                SVD_eq = torch.linalg.svd(H_eq) # 3x3 R -result of SVD contains U, S, Vh	
                X = SVD_eq.Vh.T @ SVD_eq.U.T
            
          
                X[torch.abs(X) < 1e-5] = 0

      

            # pdb.set_trace()

                if(torch.det(X)<0 ):
                    V_new = SVD_eq.Vh.T.clone()
                    V_new[0:3,2] = -SVD_eq.Vh.T[0:3,2]
                    X = V_new @ SVD_eq.U.T
                    X[torch.abs(X) < 1e-5] = 0
               # pdb.set_trace()
                # print("mean B shape")
                # print(mean_B_filt[0,0].T.shape)
                t_solve =  (mean_B_filt[0,0].T - X @ mean_A_filt[0,0].T).T
                t_solve = t_solve*2
               # t_solve =  (mean_B[0,n].T - X @ mean_A[0,n].T).T
                
            # affine = torch.eye(4)
            # affine[0:3,0:3] = X
            #  affine[0:3,3] = t_solve.T
            #     X = torch.tensor([[ 1.0000,  0.0000,  0.0000],
            # [ 0.0000,  0.7071,  0.7071],
            # [ 0.0000, -0.7071,  0.7071]])
            # pdb.set_trace()

            else:
        
                X = torch.eye(3, device=pred.device)
                t_solve = torch.tensor([[0,0,0]],device=pred.device)
              # print("in else!")
        
        # if(grid_slice==0):
            
            g = identity([1,shape[0],shape[0]])[0].unsqueeze(-1).reshape(shape[0]*shape[0],3).to(pred.device)
            g[:,0] = g[:,0] + n
            # elif(grid_slice==1):
            #   #  pdb.set_trace()
            #     g = identity([shape[0],1,shape[0]]).unsqueeze(-1).reshape(shape[0]*shape[0],3)
            #     g[:,1] = g[:,1]+n
            
            affine_all[:,n] = ((X @ g.T  + t_solve.T) - g.T).reshape(3,shape[0],shape[0])


            aff_arr[n] = torch.eye(4, device=pred.device)

        
            aff_arr[n,0:3,0:3] = X

            
            aff_arr[n,0:3,3] = t_solve.T[:,0]
            aff_arr[torch.abs(aff_arr) < 1e-5] = 0

        return affine_all[None], aff_arr
    
    def project(self, pred, mask):
       # print("PROJECT THIS ONE")
       # print(self.spacing)

        self.spacing = 2 #1
       
       # print(self.spacing)
        #set slice 0 since all the slices are turned
        self_slice=0 
        batch, chans, *size = pred.shape #size = [192, 64, 64]

        ones = torch.ones([1] + size, dtype=pred.dtype, device=pred.device) # ones.shape torch.Size([1, 192, 64, 64])
        grid = torch.stack(torch.meshgrid([torch.arange(1., s + 1., device=pred.device) for s in size], indexing='ij')) # grid shape torch.Size([3, 192, 64, 64])
        grid = torch.cat([grid, ones]) # grida shape - torch.Size([4, 192, 64, 64])
        warp = pred + grid[:pred.shape[1]] # warp torch.Size([1, 3, 192, 64, 64])
 

        M = mask.expand([batch, 1] + size).unflatten(2 + self_slice, [-1, self.spacing]).movedim(2 + self_slice, 1).flatten(3).transpose(2,3) # mask shape torch.Size([1, 192, 4096, 1])
        A = grid.expand([batch, chans + 1] + size).unflatten(2 + self_slice, [-1, self.spacing]).movedim(2 + self_slice, 1).flatten(3).transpose(2,3) # A shape torch.Size([1, 192, 4096, 4])
        B = warp.unflatten(2 + self_slice, [-1, self.spacing]).movedim(2 + self_slice, 1).flatten(3).transpose(2,3) # B shape torch.Size([1, 192, 4096, 3])
        
        
        mean_A = torch.sum(A * M, 2, keepdim=True) / torch.sum(M, 2, keepdim=True) # mean_A shape torch.Size([1, 192, 1, 4])

        mean_A[mean_A.isnan()] = 0
        mean_A[...,-1] = 0 # last dimension zeros
    
      #  pdb.set_trace()

      #  print("in project!!")
        
        
        with torch.no_grad():
                X = torch.linalg.lstsq(M * (A - mean_A), M * (B - mean_A[...,:-1])).solution # X.shape torch.Size([1, 192, 4, 3])
                # epsilon = 1e-6  # Small regularization term
                # identity = torch.eye(M.shape[-1], device=M.device).unsqueeze(0)  # Create identity matrix
                # regularized_matrix = M * (A - mean_A) + epsilon * identity

                # X = torch.linalg.lstsq(regularized_matrix, M * (B - mean_A[..., :-1])).solution
            #  lambda_reg = 1000000  # Adjust regularization strength

            #   X = torch.linalg.lstsq(M * (A - mean_A) + lambda_reg * torch.eye(M.size(0),device=M.device ), M * (B - mean_A[...,:-1])).solution

                R = torch.linalg.svd(X[...,:-1,:]) # 3x3 R -result of SVD contains U, S, Vh

        ####
        

        t = X[...,-1:,:] # tx, ty, tz, torch.Size([1, 192, 1, 3])
        P = torch.cat([R.U @ R.S.sign().diag_embed() @ R.Vh, t], 2) #.detach() # torch.Size([1, 192, 4, 3])

        # previous size - [192, 64, 64]
        size[self_slice] = self.spacing
         # new size[1, 64, 64]
 
        flow = (A - mean_A) @ P - (A - mean_A)[...,:-1] # flow size torch.Size([1, 192, 4096, 3])
    
        flow = flow.transpose(3,2).unflatten(3, size).movedim(1, 2 + self_slice).flatten(2 + self_slice, 3 + self_slice) # flow size torch.Size([1, 3, 192, 64, 64])
   
        flow[flow.isnan()] = 0 
 

        return flow
    


    
class Flow_SNet(nn.Module):
    def __init__(self, *args, X=3, spacing=1, num_conv_per_flow=4, drop=0, **kwargs):
        
        super().__init__()
        conv_kernel_sizes = [[1,3,3],[3,1,3],[3,3,1]]
        pool_kernel_sizes = [[1,2,2],[2,1,2],[2,2,1]]
        slab_stride_sizes = [[spacing,1,1],[1,spacing,1],[1,1,spacing]]
        slab_kernel_sizes = [[spacing,3,3],[3,spacing,3],[3,3,spacing]]
        self.rigid = True
        slice = 0
        self.spacing = spacing
        
        self.unets = Flow_UNet(*args, conv_kernel_sizes=conv_kernel_sizes[slice], pool_kernel_sizes=pool_kernel_sizes[slice],
                               slab_kernel_sizes=slab_kernel_sizes[slice], slab_stride_sizes=slab_stride_sizes[slice], 
                               mask=True, dropout_p=drop, num_conv_per_flow=num_conv_per_flow, X=X, **kwargs)
        self.unet3 = Flow_UNet(*args, conv_kernel_sizes=3, pool_kernel_sizes=2, mask=True, dropout_p=drop, 
                               num_conv_per_flow=num_conv_per_flow, normalize_splat=False, X=X, **kwargs)
        self.strides = [self.unet3.enc_blocks[d].pool_stride for d in range(len(self.unet3.enc_blocks))]

        self.unet3.flo_blocks = self.unet3.enc_blocks = None
    

    def forward(self, x):
        xs, mask = x.tensor_split(2,1) # torch.squeeze(2) #cat(x.unbind(2), 0)
        # mask = x[:,1:]

        skips = [None] * len(self.unets.enc_blocks)
        masks = [mask] * len(self.unets.enc_blocks)
        sizes = [list(xs.shape[2:])] * len(self.unets.enc_blocks)

        for d in range(len(self.unets.enc_blocks)):
            skips[d] = xs = self.unets.enc_blocks[d](xs)
            sizes[d] = [sizes[d - 1][i] // self.strides[d][i] for i in range(len(self.strides[d]))]
            masks[d] = F.interpolate(masks[d - 1], size=skips[d].shape[2:], mode='trilinear', align_corners=True).gt(0).float()

        flow = torch.zeros([xs.shape[0], xs.ndim - 2] + list(xs.shape[2:]), device=xs.device)
        mask = torch.ones([xs.shape[0], 1] + list(xs.shape[2:]), device=xs.device)
        x3 = torch.zeros([xs.shape[0], xs.shape[1]] + sizes[-1], device=xs.device)

        # 3D u-net in decoder

        
        for u in reversed(range(len(self.unet3.dec_blocks))):
            
            xs = self.unets.dec_blocks[u](xs, skips[u])
            


          #  turn stacks back orthogonal
            splat = self.unet3.splat(skips[u], flow, mask=torch.ones_like(mask), shape=sizes[u]) #.sum(axis=0, keepdim=True)
            # splat = splat[:,:-1] / (splat[:,-1:] + 1e-4 * splat[:,-1].detach().max().item())
            splat = splat[:,:-1] / splat[:,-1:].detach().flatten(2).max(axis=2)[0][...,None,None,None].expand(splat[:,:-1].shape) # normalize #splat[:,-1:].max().item() 
            
            x3 = self.unet3.dec_blocks[u](x3, splat)
          #  print("slicing")
            xw = self.unet3.warp(self.unet3.interp(torch.cat([x3], 1).expand([xs.shape[0]] + [-1] * 4), list(xs.shape[2:])), flow)


            # turn slices parallel
          
            flow, mask = self.unets.flow_add(flow, self.unets.flo_blocks[u](torch.cat([xs, xw], 1)))


        if self.rigid:
            print("rigid!!!")
            
           # all_x = x.clone()
           
            mask2 = x[:, 1:].clone().detach().requires_grad_(True) + mask*0.0
            
          
          #  pdb.set_trace()
            #flow = self.project(flow, all_x[:,1:])
            
            flow = self.project(flow, mask2)

          #  flow = self.project_mask(flow, mask)

        return flow.flip(1) #.unsqueeze(2) #torch.stack(flow.flip(1).split(1,0), 2)
    

    def project(self, pred, mask):
        

        self.slice = 0
        batch, chans, *size = pred.shape
        ones = torch.ones([1] + size, dtype=pred.dtype, device=pred.device) #_like(grid[0])])
        grid = torch.stack(torch.meshgrid([torch.arange(1., s + 1., device=pred.device) for s in size], indexing='ij'))
        grid = torch.cat([grid, ones])
        warp = pred + grid[:pred.shape[1]] # torch.cat([pred, 0 * ones.unsqueeze(0)], 1) + grid

        M = mask.expand([batch, 1] + size).unflatten(2 + self.slice, [-1, self.spacing]).movedim(2 + self.slice, 1).flatten(3).transpose(2,3)
        A = grid.expand([batch, chans + 1] + size).unflatten(2 + self.slice, [-1, self.spacing]).movedim(2 + self.slice, 1).flatten(3).transpose(2,3)
        B = warp.unflatten(2 + self.slice, [-1, self.spacing]).movedim(2 + self.slice, 1).flatten(3).transpose(2,3)
        
        mean_A = torch.sum(A * M, 2, keepdim=True) / torch.sum(M, 2, keepdim=True)
        mean_A[mean_A.isnan()] = 0
        mean_A[...,-1] = 0

        no_grad = True
        detach_only = True

    
        if no_grad and not detach_only:
            
            with torch.no_grad():
                X = torch.linalg.lstsq(M * (A - mean_A), M * (B - mean_A[...,:-1])).solution
                X = X.detach()
                X = torch.nan_to_num(X, nan=0.0)
                U, S, Vh = torch.linalg.svd(X[...,:-1,:]) # 3x3

        elif detach_only:

                X = torch.linalg.lstsq(M * (A - mean_A), M * (B - mean_A[...,:-1])).solution
                X = X.detach()
                X = torch.nan_to_num(X, nan=0.0)
                U, S, Vh = torch.linalg.svd(X[...,:-1,:]) # 3x3
                U = U.detach()
                S = S.detach()
                Vh = Vh.detach()
                



        else:
            X = torch.linalg.lstsq(M * (A - mean_A), M * (B - mean_A[...,:-1])).solution
            U, S, Vh = torch.linalg.svd(X[...,:-1,:]) # 3x3

        t = X[...,-1:,:] # tx, ty, tz
        P = torch.cat([U @ S.sign().diag_embed() @ Vh, t], 2) #.detach() # 47 x 4 x 3 # 0 0 0 1
        
        num_slices = P.shape[1]
        aff_arr = P.transpose(2, 3)
      #  pdb.set_trace()
        affine = torch.zeros((1, num_slices, 4, 4))
        affine[:, :, :3, :] = aff_arr

        affine[:, :, 3, 0:3] = torch.zeros((1, num_slices, 3))
        affine[:, :, 3, 3] = torch.ones((1, num_slices))


        torch.set_printoptions(sci_mode=False)
       # print(affine)
        
        size[self.slice] = self.spacing
        flow = (A - mean_A) @ P - (A - mean_A)[...,:-1]
        flow = flow.transpose(3,2).unflatten(3, size).movedim(1, 2 + self.slice).flatten(2 + self.slice, 3 + self.slice)
      #  pdb.set_trace()
      #  print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        
        return flow
    
    def compensate(self, out, tar):
        batch, chans, *size = out.shape
        grid = torch.stack(torch.meshgrid([torch.arange(1., s + 1., device=out.device) for s in size], indexing='ij'))
        mask = tar[:,chans:] # if masked else torch.ones_like(tar[:,chans:])
        
        B = (out[:,:chans].flip(1) + grid).masked_select(mask.bool()).reshape(batch, chans, -1) #.transpose(1,2)
        A = (tar[:,:chans].flip(1) + grid).masked_select(mask.bool()).reshape(batch, chans, -1) #.transpose(1,2)
        
        mean_B = B.mean(-1, keepdim=True).detach()
        mean_A = A.mean(-1, keepdim=True).detach()
        X = torch.linalg.svd(torch.linalg.lstsq((B - mean_B).transpose(1,2), (A - mean_A).transpose(1,2)).solution.detach())
        R = (X.U @ X.S.sign().diag_embed() @ X.Vh).transpose(1,2)

        out = out.flip(1) + grid - mean_B.unflatten(-1,[1,1,1])
        out = (R @ out.flatten(2)).unflatten(2, out.shape[2:])
        out = (out - grid + mean_A.unflatten(-1,[1,1,1])).flip(1)
     #   pdb.set_trace()
        return out
    


    def upsample_flow(self, stack, spacing=2):
        self.slice = 0
        stack = stack.squeeze(0) #[0,:,0]
        #divide flow into 64 segments of 2 slices

        stack = stack.movedim(1 + self.slice, 0).unflatten(0, [-1, spacing]).movedim(1,-1) 
       # torch.Size([64, 3, 128, 128, 2])

      #  pdb.set_trace()
        # why is there a +1?
        shape = [spacing * s + 1 for s in stack.shape[-3:]] #[257, 257, 5]
        
        # weight second slice more and interpolate between those two plus concat with the original
        stack = torch.cat([stack, stack[...,:1].lerp(stack[...,1:], 2.0)], -1) # torch.Size([64, 3, 128, 128, 3])
      #  pdb.set_trace()
        # why is there an added dimension at the end?
        stack = torch.nn.functional.pad(stack, (0,0,0,1,0,1)) #torch.Size([64, 3, 129, 129, 3])
      #  pdb.set_trace()
        stack = torch.nn.functional.interpolate(stack, size=shape, mode='trilinear', align_corners=True) 
        # torch.Size([64, 3, 257, 257, 5])
      #  pdb.set_trace()
        stack = stack[...,:-1,:-1,:-1].movedim(-1,1).flatten(0,1).movedim(0, 1 + self.slice)
        stack = stack.unsqueeze(0)
        
        return spacing * stack

    def upsample_flow2(self, stack, spacing=2):
        self.slice = 0
       # stack = stack.squeeze(0) #[0,:,0]
        #divide flow into 64 segments of 2 slices
       # stack = stack.movedim(1 + self.slice, 0).unflatten(0, [-1, spacing]).movedim(1,-1)
        shape = [spacing * s + 1 for s in stack.shape[-3:]]
        shape = [ 256, 256, 256]
        
        # weight second slice more and interpolate between those two plus concat with the original
       # stack = torch.cat([stack, stack[...,:1].lerp(stack[...,1:], 2.0)], -1)
     #   stack = torch.nn.functional.pad(stack, (0,0,0,1,0,1))
      #  pdb.set_trace()
        stack = torch.nn.functional.interpolate(stack, size=shape, mode='trilinear', align_corners=True)
       # pdb.set_trace()
      #  stack = stack[...,:-1,:-1,:-1].movedim(-1,1).flatten(0,1).movedim(0, 1 + self.slice)
       # pdb.set_trace()
       # stack = stack.unsqueeze(0)
        
        return spacing * stack
    

def flow_SNet2d(*args, slice=1, norm=True, **kwargs):
    return Flow_SNet(slice=slice, input_channels=1, base_num_features=[16, 24, 32, 48, 64, 96, 128, 192, 256, 320], num_classes=8, num_pool=6, norm=norm, X=2)


#def flow_SNet3d(*args, slice=1, spacing=1, norm=True, num_conv_per_flow=4, num_classes=8, base_num_features=[24, 32, 48, 64, 96, 128, 192], num_pool=5, **kwargs):
  #  return Flow_SNet(slice=slice, spacing=spacing, input_channels=1, base_num_features=base_num_features, num_conv_per_flow=num_conv_per_flow, num_classes=num_classes, num_pool=num_pool, norm=norm, X=3)

def flow_SNet3d(*args, spacing=1, norm=True, num_conv_per_flow=4, num_classes=8, base_num_features=[24, 32, 48, 64, 96, 128, 192], num_pool=5, **kwargs):
    return Flow_SNet(spacing=spacing, input_channels=1, base_num_features=base_num_features, num_conv_per_flow=num_conv_per_flow, num_classes=num_classes, num_pool=num_pool, norm=norm, X=3)


#def flow_SNet3d(*args, slice=1, spacing=1, norm=True, base_num_features=[24, 32, 48, 64, 96, 128, 192], num_pool=5, **kwargs):
    #return Flow_SNet(slice=slice, spacing=spacing, input_channels=1, base_num_features=base_num_features, num_classes=8, num_pool=num_pool, norm=norm, X=3)

def flow_SNet3d_multi(*args, spacing=1, slice = [0,1,2], norm=True, base_num_features=[24, 32, 48, 64, 96, 128, 192], num_pool=5, rigid=False,crop=False,**kwargs):
    return Flow_SNet_multi(spacing=spacing, slice=slice, input_channels=1, base_num_features=base_num_features, num_classes=8, num_pool=num_pool, norm=norm, X=3, rigid=rigid, crop=crop)


def flow_SNet3d_MLP(*args, spacing=1, slice = [0,1,2], norm=True, base_num_features=[24, 32, 48, 64, 96, 128, 192], num_pool=5, rigid=False,crop=False,**kwargs):
    return Flow_SNet_MLP(spacing=spacing, slice=slice, input_channels=1, base_num_features=base_num_features, num_classes=8, num_pool=num_pool, norm=norm, X=3, rigid=rigid, crop=crop)


def flow_SNet3d_multi_s0(*args, spacing=1, slice=[0,0,0], norm=True, base_num_features=[24, 32, 48, 64, 96, 128, 192], num_pool=5, **kwargs):
    return Flow_SNet_multi(spacing=spacing, slice=slice, input_channels=1, base_num_features=base_num_features, num_classes=8, num_pool=num_pool, norm=norm, X=3)


def flow_SNet3d0(*args, **kwargs):
    return flow_SNet3d(slice=0, spacing=2)

def flow_SNet3d1(*args, **kwargs):
    return flow_SNet3d(slice=1, spacing=2)

def flow_SNet3d1_32(*args, **kwargs):
    return flow_SNet3d(slice=1, spacing=2, base_num_features=[32, 32, 32, 32, 32, 32, 32], num_pool=4, norm=True)

def flow_SNet3d0_192(*args, **kwargs):
    return flow_SNet3d(slice=0, spacing=2, base_num_features=[24, 32, 48, 64, 96, 128, 192], num_pool=4, norm=True)

def flow_SNet3d1_192(*args, **kwargs):
    return flow_SNet3d(slice=1, spacing=2, base_num_features=[24, 32, 48, 64, 96, 128, 192], num_pool=4, norm=True)

def flow_SNet3d2_192(*args, **kwargs):
    return flow_SNet3d(slice=2, spacing=2, base_num_features=[24, 32, 48, 64, 96, 128, 192], num_pool=4, norm=True)

def flow_SNet3d1_192_4(*args, **kwargs):
    return flow_SNet3d(slice=1, spacing=4, base_num_features=[24, 32, 48, 64, 96, 128, 192], num_pool=5, norm=True)

def flow_SNet3d0_256(*args, **kwargs):
    return flow_SNet3d(slice=0, spacing=2, base_num_features=[32, 48, 64, 96, 128, 192, 256], num_pool=4, norm=True)

def flow_SNet3d1_256(*args, **kwargs):
    return flow_SNet3d(slice=1, spacing=2, base_num_features=[32, 48, 64, 96, 128, 192, 256], num_pool=4, norm=True)

def flow_SNet3d2_256(*args, **kwargs):
    return flow_SNet3d(slice=2, spacing=2, base_num_features=[32, 48, 64, 96, 128, 192, 256], num_pool=4, norm=True)

def flow_SNet3d0_384(*args, **kwargs):
    return flow_SNet3d(slice=0, spacing=2, base_num_features=[48, 64, 96, 128, 192, 256, 384], num_pool=4, norm=True)

def flow_SNet3d1_384(*args, **kwargs):
    return flow_SNet3d(slice=1, spacing=2, base_num_features=[48, 64, 96, 128, 192, 256, 384], num_pool=4, norm=True)

def flow_SNet3d2_384(*args, **kwargs):
    return flow_SNet3d(slice=2, spacing=2, base_num_features=[48, 64, 96, 128, 192, 256, 384], num_pool=4, norm=True)

def flow_SNet3d0_512(*args, **kwargs):
    return flow_SNet3d(slice=0, spacing=2, base_num_features=[32, 64, 128, 256, 512, 1024], num_pool=4, norm=True)

#def flow_SNet3d0_1024(*args, **kwargs):
  #  return flow_SNet3d(slice=0, spacing=2, base_num_features=[64, 128, 256, 512, 1024, 2048], num_pool=4, norm=True)


def flow_SNet3d0_1024(*args, **kwargs):
    return flow_SNet3d(spacing=2, base_num_features=[64, 128, 256, 512, 1024, 2048], num_pool=4, norm=True)


def flow_SNet3d1_32_clin(*args, **kwargs):
    return flow_SNet3d(slice=1, spacing=2, base_num_features=[32, 32, 32], num_pool=2, norm=True)



def flow_SNet3d1_32_mutli(*args, **kwargs):
    print("32 multi")
    return flow_SNet3d_multi(slice=0, spacing=2, base_num_features=[32, 32, 32, 32, 32, 32, 32], num_pool=4, norm=True)



def flow_SNet3d2_64_multi(*args, **kwargs):
    return flow_SNet3d_multi(slice=[0,1,2], spacing=2, base_num_features=[32, 48, 64, 64, 64, 64], num_pool=4, norm=True)


def flow_SNet3d2_256_multi(*args, **kwargs):
    return flow_SNet3d_multi(slice=[0,1,2], spacing=2, base_num_features=[32, 48, 64, 96, 128, 192, 256], num_pool=4, norm=True)


def flow_SNet3d2_512_multi(*args, **kwargs):
    print("IN flow_SNet3d2_512_multi")
    return flow_SNet3d_multi(slice=[0,1,2], spacing=2, base_num_features=[32, 64, 128, 256, 512, 1024], num_pool=4, norm=True)


def flow_SNet3d2_512_multi_rigid(*args, **kwargs):
    print("IN flow_SNet3d2_512_multi rigid")
    return flow_SNet3d_multi(slice=[0,1,2], spacing=2, base_num_features=[32, 64, 128, 256, 512, 1024], num_pool=4, norm=True, rigid=True)

def flow_SNet3d2_64_multi_crop(*args, **kwargs):

    return flow_SNet3d_multi(slice=[0,1,2], spacing=2, base_num_features=[4, 8, 16, 32, 64, 128], num_pool=4, norm=True, rigid=False, crop=True)


def flow_SNet3d2_128_multi_crop(*args, **kwargs):
    print("IN flow_SNet3d2_512 crop")
    # 48, 96 
    return flow_SNet3d_multi(slice=[0,1,2], spacing=2, base_num_features=[8, 16, 32, 64, 128, 256], num_pool=4, norm=True, rigid=False, crop=True)

def flow_SNet3d2_256_multi_crop(*args, **kwargs):
    print("IN flow_SNet3d2_512 crop")
    # 48, 96 
    return flow_SNet3d_multi(slice=[0,1,2], spacing=2, base_num_features=[16, 32, 64, 128, 256, 512], num_pool=4, norm=True, rigid=False, crop=True)



def flow_SNet3d2_512_multi_crop(*args, **kwargs):
    print("IN flow_SNet3d2_512 crop")
    # 48, 96 
    return flow_SNet3d_multi(slice=[0,1,2], spacing=2, base_num_features=[32, 64, 128, 256, 512, 1024], num_pool=4, norm=True, rigid=False, crop=True)


def flow_SNet3d2_768_multi_crop(*args, **kwargs):
    print("IN flow_SNet3d2_768 crop")
    # 48, 96 
    return flow_SNet3d_multi(slice=[0,1,2], spacing=2, base_num_features=[48, 96, 192, 384, 768, 1536], num_pool=4, norm=True, rigid=False, crop=True)


def flow_SNet3d2_256_multi_crop(*args, **kwargs):
    print("IN flow_SNet3d2_256 crop")
    # 48, 96 
    return flow_SNet3d_multi(slice=[0,1,2], spacing=2, base_num_features=[32, 48, 64, 96, 128, 192, 256], num_pool=4, norm=True, rigid=False, crop=True)



def flow_SNet3d2_512_multi_crop_multi_slice(*args, **kwargs):
    print("IN flow_SNet3d2_512 crop multi slice")
    # 48, 96 
    return flow_SNet3d_multi(slice=[0,1,2,0,1,2], spacing=2, base_num_features=[32, 64, 128, 256, 512, 1024], num_pool=4, norm=True, rigid=False, crop=True)

def flow_SNet3d2_256_multi_crop_redistribute(*args, **kwargs):

    return flow_SNet3d_multi(slice=[0,1,2], spacing=2, base_num_features=[32, 64, 128, 512, 1024, 1024], num_pool=4, norm=True, rigid=False, crop=True)


def flow_SNet3d2_256_multi_crop_redistribute2(*args, **kwargs):

    return flow_SNet3d_multi(slice=[0,1,2], spacing=2, base_num_features=[32, 64, 128, 1024, 1024, 1024], num_pool=4, norm=True, rigid=False, crop=True)

def flow_SNet3d2_512_multi_crop_simp2(*args, **kwargs):

    return flow_SNet3d_multi(slice=[0,1,2], spacing=2, base_num_features=[64, 128, 256, 512, 1024, 2048], num_pool=4, norm=True, rigid=False, crop=True)


def flow_SNet3d2_1024_multi_crop(*args, **kwargs):

    return flow_SNet3d_multi(slice=[0,1,2], spacing=2, base_num_features=[64, 128, 256, 512, 1024, 2048], num_pool=4, norm=True, rigid=False, crop=True)

def flow_SNet3d2_1024_MLP(*args, **kwargs):

    return flow_SNet3d_MLP(slice=[0,1,2], spacing=2, base_num_features=[64, 128, 256, 512, 1024, 2048], num_pool=4, norm=True, rigid=False, crop=True, drop_out=0)



def flow_SNet3d2_256_MLP(*args, **kwargs):

    return flow_SNet3d_MLP(slice=[0,1,2], spacing=2, base_num_features=[16, 32, 64, 128, 256, 512], num_pool=4, norm=True, rigid=False, crop=True)




def flow_SNet3d2_2056_MLP(*args, **kwargs):
    print("IN flow_SNet3d2_1024 crop")
    # 48, 96 
    return flow_SNet3d_MLP(slice=[0,1,2], spacing=2, base_num_features=[128, 256, 512, 1024, 2048, 4096], num_pool=4, norm=True, rigid=False, crop=True)



def flow_SNet3d2_4096_MLP(*args, **kwargs):
    print("IN flow_SNet3d2_1024 crop")
    # 48, 96 
    return flow_SNet3d_MLP(slice=[0,1,2], spacing=2, base_num_features=[256, 512, 1024, 2048, 4096, 8192], num_pool=4, norm=True, rigid=False, crop=True)




def flow_SNet3d2_1024_multi_crop_small_inner(*args, **kwargs):
    print("IN flow_SNet3d2_1024 crop")
    # 48, 96 
    return flow_SNet3d_multi(slice=[0,1,2], spacing=2, base_num_features=[16, 32, 64, 128, 1024, 2048], num_pool=4, norm=True, rigid=False, crop=True)

def flow_SNet3d2_1024_multi_crop_medium_inner(*args, **kwargs):
    print("IN flow_SNet3d2_1024 crop")
    # 48, 96 
    return flow_SNet3d_multi(slice=[0,1,2], spacing=2, base_num_features=[16, 32, 64, 256, 1024, 2048], num_pool=4, norm=True, rigid=False, crop=True)

def flow_SNet3d2_1024_multi_crop_small_outer(*args, **kwargs):
    print("IN flow_SNet3d2_1024 crop")
    # 48, 96 
    return flow_SNet3d_multi(slice=[0,1,2], spacing=2, base_num_features=[16, 32, 64, 128, 512, 2048], num_pool=4, norm=True, rigid=False, crop=True)


def flow_SNet3d2_1024_dist_multi_crop(*args, **kwargs):
    print("IN flow_SNet3d2_1024 crop")
    # 48, 96 
    return flow_SNet3d_multi(slice=[0,1,2], spacing=2, base_num_features=[64, 128, 192, 256, 512, 1024], num_pool=4, norm=True, rigid=False, crop=True)



def flow_SNet3d2_1024_dist2_multi_crop(*args, **kwargs):
    print("IN flow_SNet3d2_1024 crop")
    # 48, 96 
    return flow_SNet3d_multi(slice=[0,1,2], spacing=2, base_num_features=[128, 64, 64, 64, 64, 64], num_pool=4, norm=True, rigid=False, crop=True)


def flow_SNet3d2_512_multi_crop_sp1(*args, **kwargs):
    print("IN flow_SNet3d2_512 crop")
    
    return flow_SNet3d_multi(slice=[0,1,2], spacing=1, base_num_features=[32, 64, 128, 256, 512, 1024], num_pool=4, norm=True, rigid=False, crop=True)


def flow_SNet3d2_512_multi_crop_1layer(*args, **kwargs):
    print("IN flow_SNet3d2_512 crop")
    return flow_SNet3d_multi(slice=[0,1,2], spacing=2, base_num_features=[32, 64], num_pool=1, norm=True, rigid=False, crop=True)


def flow_SNet3d2_512_multi_crop_hr(*args, **kwargs):
    print("IN flow_SNet3d2_512 crop high res")
    return flow_SNet3d_multi(slice=[0,1,2], spacing=1, base_num_features=[32, 64, 128, 256, 512, 1024], num_pool=4, norm=True, rigid=False, crop=True)


def flow_SNet3d2_512_multi_crop_hr_low_features(*args, **kwargs):
    print("IN flow_SNet3d2_512 crop high res")
    return flow_SNet3d_multi(slice=[0,1,2], spacing=1, base_num_features=[8, 16, 32, 64, 256, 512], num_pool=4, norm=True, rigid=False, crop=True)

def flow_SNet3d2_512_multi_crop_hr_new_features(*args, **kwargs):
    print("IN flow_SNet3d2_512 crop high res")
    return flow_SNet3d_multi(slice=[0,1,2], spacing=1, base_num_features=[16, 32, 64, 128, 256, 1024], num_pool=4, norm=True, rigid=False, crop=True)


def flow_SNet3d2_512_multi_crop_hr_new_features2(*args, **kwargs):
    print("IN flow_SNet3d2_512 crop high res")
    return flow_SNet3d_multi(slice=[0,1,2], spacing=1, base_num_features=[8, 16, 32, 64, 128, 1024], num_pool=4, norm=True, rigid=False, crop=True)


def flow_SNet3d2_256_multi_crop(*args, **kwargs):
    print("IN flow_SNet3d2_256 crop")
    return flow_SNet3d_multi(slice=[0,1,2], spacing=2, base_num_features=[32, 48, 64, 96, 128, 192, 256], num_pool=4, norm=True, rigid=False, crop=True)


def flow_SNet3d2_512_multi_s0(*args, **kwargs):
    return flow_SNet3d_multi(slice=[0,0,0], spacing=2, base_num_features=[32, 64, 128, 256, 512, 1024], num_pool=4, norm=True)


#def flow_SNet3d2_512_multi_rigid(*args, **kwargs):
  #  return flow_SNet3d_multi(slice=0, spacing=2, base_num_features=[32, 64, 128, 256, 512, 1024], num_pool=4, norm=True, rigid=True)



def flow_SNet3d0_1024_multi(*args, **kwargs):
    return flow_SNet3d_multi(slice=0, spacing=2, base_num_features=[64, 128, 256, 512, 1024, 2048], num_pool=4, norm=True)

# def flow_SNet3d0_4_1_256(*args, **kwargs):
#     return flow_SNet3d(slice=0, spacing=4, base_num_features=[24, 32, 48, 64, 96, 128, 192], num_pool=4, norm=True)

# def flow_SNet3d0_4_1_512(*args, **kwargs):
#     return flow_SNet3d(slice=0, spacing=4, base_num_features=[32, 64, 128, 256, 512, 1024], num_pool=5, norm=True)