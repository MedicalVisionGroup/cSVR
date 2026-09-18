import os
import sys
import math
import torch
import random
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torch.nn.functional as FF

from cornucopia.utils.warps import affine_flow
from cornucopia.random import Normal, Uniform
import cornucopia as cc
print(cc.__file__)

#import cornucopia as cc
import pdb


project_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
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


class RandomSignalVoidTransform:
    """Simulates signal voids on 2D slices via random oriented elliptical dropouts.

    Args:
        prob: per-slice probability of applying a void.
    """
    def __init__(self, prob=0.3):
        self.prob = prob

    def __call__(self, x):
        # x: [C, D, H, W] — apply independently to each 2D slice along D
        slices = x.permute(1, 0, 2, 3).clone()  # [D, C, H, W]

        idx = torch.rand(slices.shape[0], device=slices.device) < self.prob
        n = idx.sum()
        if n > 0:
            h, w = slices.shape[-2:]
            gy = torch.linspace(-(h - 1) / 2, (h - 1) / 2, h, device=slices.device)
            gx = torch.linspace(-(w - 1) / 2, (w - 1) / 2, w, device=slices.device)
            yc = (torch.rand(n, device=slices.device) - 0.5) * (h - 1)
            xc = (torch.rand(n, device=slices.device) - 0.5) * (w - 1)

            gy = gy.view(1, -1, 1) - yc.view(-1, 1, 1)
            gx = gx.view(1, 1, -1) - xc.view(-1, 1, 1)

            theta = 2 * math.pi * torch.rand((n, 1, 1), device=slices.device)
            c, s = torch.cos(theta), torch.sin(theta)
            gx, gy = c * gx - s * gy, s * gx + c * gy

            a = 5 + torch.rand_like(theta) * 10
            A = torch.rand_like(theta) * 0.5 + 0.5
            sx = torch.rand_like(theta) * 15 + 1
            sy = a**2 / sx

            sx = -0.5 / sx**2
            sy = -0.5 / sy**2

            mask = 1 - A * torch.exp(sx * gx**2 + sy * gy**2)
            slices[idx, 0] = slices[idx, 0] * mask

        return slices.permute(1, 0, 2, 3)  # back to [C, D, H, W]


class RandomRicianNoiseTransform:
    """Simulates Rician noise on 2D slices, matching MRI magnitude image physics.

    Noise is only added within-signal (above threshold) to leave zero background
    untouched. Sigma is sampled uniformly from [sigma[0], sigma[1]] each call.

    Args:
        sigma: (min, max) noise standard deviation range.
        threshold: voxel intensity threshold defining the signal mask.
        prob: per-slice probability of applying noise.
    """
    def __init__(self, sigma=(0.01, 0.1), threshold=0.05, prob=0.5):
        self.sigma = sigma
        self.threshold = threshold
        self.prob = prob

    def __call__(self, x):
        # x: [C, D, H, W] — apply independently to each 2D slice along D
        D = x.shape[1]
        idx = torch.rand(D, device=x.device) < self.prob
        if idx.sum() == 0:
            return x

        sigma = self.sigma[0] + torch.rand(1).item() * (self.sigma[1] - self.sigma[0])

        x = x.clone()
        sel = x[0, idx]  # [n, H, W]
        mask = sel > self.threshold
        noise1 = torch.randn_like(sel) * sigma
        noise2 = torch.randn_like(sel) * sigma
        noisy = torch.sqrt((sel + noise1) ** 2 + noise2 ** 2)
        x[0, idx] = torch.where(mask, noisy, sel)

        return x


class GenerateMotionTrajectory:
    def __init__(self, spacing=1, subsample=1, translations=0.03, rotations=40, bulk_translations=0, bulk_rotations_plane=0, bulk_rotations_tr_plane=0, zooms=(-0.35,-0.2), slice=1, nodes=(1,5), shots=2, augment=False, noise=0, X=3, flow_final=False, crop=False, normalize_img=True, mlp_training=False, verbose=False, cut=0, gaussian_psf = False, spacing_range=None, **kwargs):

        # DEFINE MOTION PARAMETERS
        self.verbose = verbose
        if self.verbose:
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
            print(f"gaussian_psf: {gaussian_psf}")

        self.subsample = subsample
        self.spacing = spacing
        self.spacing_range = spacing_range
        self.gaussian_psf = gaussian_psf

        self.slice = slice if isinstance(slice, (tuple, list)) else [slice]
        self.flip_ax1 = cc.Rot180Transform(axis=1)
        self.flip_ax2 = cc.Rot180Transform(axis=2)
        self.mlp_training = mlp_training
        if self.mlp_training:
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
        self.augment = augment
        self.noise = noise
        self.crop = crop
        self.flow_final = flow_final
        self.X = X
        self.normalize_img = normalize_img
        self.cut = cut
        self.bias_field = False
        self.gamma_field = False
        self.slice_bias_apply = False
        self.signal_void_apply = False
        self.rician_noise_apply = False
        self.smooth = False
        self.slice_drop_out = 0.2

        if self.augment:
            print("Adding augmentations!!")
            self.bias_field = False
            self.gamma_field = False
            self.slice_bias_apply = False
            self.signal_void_apply = True
            self.rician_noise_apply = True
            self.smooth = False
        else:
            print("No augmentations.")
            self.slice_drop_out = 0

        # SPECIFY BULK ROTATIONS
        bulk1 = [bulk_rotations_tr_plane, bulk_rotations_tr_plane, bulk_rotations_plane]
        bulk2 = [bulk_rotations_tr_plane, bulk_rotations_plane, bulk_rotations_tr_plane]
        bulk3 = [bulk_rotations_plane, bulk_rotations_tr_plane, bulk_rotations_tr_plane]

        bulk_rotations_list = []
        for i in slice:
            if i == 0:
                bulk_rotations_list.append(bulk1)
            elif i == 1:
                bulk_rotations_list.append(bulk2)
            elif i == 2:
                bulk_rotations_list.append(bulk3)

        #pdb.set_trace()
        # bulk_rotations_list_1 = [[12, 179, 180], [12, 180, 12], [180, 12, 12]]
        # bulk_rotations_list_2 = [[12, 181, 180], [12, 180, 12], [180, 12, 12]]
        # # self.base = [cc.RandomSlicewiseAffineTransform(nodes=nodes, shots=shots, spacing=spacing, subsample=subsample, slice=s, translations=Normal(0, translations), rotations=Normal(0, rotations/2),
        #                                              bulk_translations=bulk_translations, bulk_rotations=(negate_list(bulk_rotations_list[s]), bulk_rotations_list[s]), shears=0, zooms=0) for s in self.slice]
        # RAND BULK ROT 180
        # self.base = [cc.RandomSlicewiseAffineTransform(nodes=nodes, shots=shots, spacing=spacing, subsample=subsample, slice=s, translations=Normal(0, translations), rotations=Normal(0, rotations/2),bulk_translations=bulk_translations, bulk_rotations=(negate_list(bulk_rotations_list[s]), bulk_rotations_list[s]), shears=0, zooms=0, rand_rot_bulk180=True) for s in self.slice]




        self.base = [cc.RandomSlicewiseAffineTransform(nodes=nodes, shots=shots, spacing=spacing, subsample=subsample, slice=s, translations=Normal(0, translations), rotations=Normal(0, rotations/2),
                                                        bulk_translations=bulk_translations, bulk_rotations=(negate_list(bulk_rotations_list[s]), bulk_rotations_list[s]), shears=0, zooms=0, gaussian_psf=self.gaussian_psf) for s in self.slice]


        # spacing
        # spacing=Uniform(1, 7.5)
        # pdb.set_trace()
        # self.base = [cc.RandomSlicewiseAffineTransform(nodes=nodes, shots=shots, spacing=spacing, subsample=subsample, slice=s, translations=Normal(0, translations), rotations=Normal(0, rotations/2),
        #                                                 bulk_translations=bulk_translations, bulk_rotations=(negate_list(bulk_rotations_list[s]), bulk_rotations_list[s]), shears=0, zooms=0, gaussian_psf=self.gaussian_psf) for s in self.slice]


        
        # pdb.set_trace()
        # self.base = [cc.RandomSlicewiseAffineTransform(nodes=nodes, shots=shots, spacing=spacing, subsample=subsample, slice=s, translations=Normal(0, translations), rotations=Normal(0, rotations/2),
        #                                              bulk_translations=bulk_translations, bulk_rotations=(bulk_rotations_list_1[s], bulk_rotations_list_2[s]), shears=0, zooms=0) for s in self.slice]

        # sample trajectories!
        # self.base = [cc.RandomSlicewiseAffineTransform(nodes=nodes, shots=shots, spacing=spacing, subsample=subsample, slice=s, trajectory_mode=True,trajectory_path='/data/vision/polina/users/mfirenze/SVoRT/dataset/traj.npy',
        #                                              bulk_rotations=(negate_list(bulk_rotations_list[s]), bulk_rotations_list[s]), shears=0, zooms=0,  trajectory_time_step=(1,2),trajectory_relative=True) for s in self.slice]

        # SPECIFY AUGMENTATIONS
        prob_aug = 0.3
        self.mult = cc.MaybeTransform(cc.RandomGaussianNoiseTransform(sigma=0.1), prob_aug)
        self.bias = cc.MaybeTransform(cc.RandomMulFieldTransform(vmax=1), prob_aug)
        self.gamma = cc.MaybeTransform(cc.RandomGammaTransform(gamma=(0.5, 2)), prob_aug)
        self.slice_bias = cc.MaybeTransform(cc.RandomSlicewiseMulFieldTransform(slice=0, thickness=2, vmax=1), prob_aug)
        self.smoother = cc.MaybeTransform(cc.RandomSmoothTransform(fwhm=6), prob_aug)
        self.signal_void = RandomSignalVoidTransform(prob=0.5)
        self.rician_noise = RandomRicianNoiseTransform(sigma=(0.01, 0.1), threshold=0.05, prob=0.5)

    # HELPER FUNCTIONS
    def stack_in_single_plane(self, og): # turn slices to correct plane
        len_s = len(self.slice)
        ss = og.shape[1]//len_s
        new_vol = torch.zeros_like(og)
        for i, stack_num in enumerate(self.slice):
            if stack_num == 0:
                stack_n = og[:,ss*i:ss*(i+1),:,:]
            if stack_num == 1:
                stack = og[:,ss*i:ss*(i+1),:,:]
                stack_n = torch.rot90(stack, k=1, dims=(1, 2))
            if stack_num == 2:
                stack = og[:,ss*i:ss*(i+1),:,:]
                stack_n = torch.rot90(stack, k=-1, dims=(1, 3))
                stack_n = torch.rot90(stack_n, k=1, dims=(2, 3))
            new_vol[:,ss*i:ss*(i+1),:,:] = stack_n
        return new_vol

    def stack_in_single_plane_stack(self, og, stack_num): # turn slices to correct plane

        
        if stack_num == 0:
            return og
        if stack_num == 1:
            og = torch.rot90(og, k=1, dims=(1, 2))
        if stack_num == 2:
            og = torch.rot90(og, k=-1, dims=(1, 3))
            og = torch.rot90(og, k=1, dims=(2, 3))
  
        return og


    def generate_initial_flow(self, vol_n): # add correct offset to account for planar slices

        len_s = len(self.slice)
        ss = vol_n.shape[2]//len_s
        flow_new = torch.zeros((3,vol_n.shape[2],vol_n.shape[3],vol_n.shape[4])).to(vol_n.device)
        shape = [vol_n.shape[2]//len_s, vol_n.shape[3], vol_n.shape[4]]

        for i in range(len_s):
            sl = self.slice[i]

            if sl == 0:
                affine = torch.eye(4)

            if sl == 2:
                affine = torch.eye(4)
                affine[0,0] = 0
                affine[0,2] = -1
                affine[2,0] = 1
                affine[2,2] = 0

                rot_90 = torch.eye(4)
                rot_90[1,2] = 1
                rot_90[2,1] = -1
                rot_90[2,2] = 0
                rot_90[1,1] = 0

                affine = affine @ rot_90

            if sl == 1:
                affine = torch.eye(4)
                affine[0,0] = 0
                affine[1,1] = 0
                affine[0,1] = 1
                affine[1,0] = -1

            affine[0,3] = (shape[0]-1)/2
            affine[1,3] = (shape[1]-1)/2
            affine[2,3] = (shape[2]-1)/2

            t = affine[0:3,3]
            aff_m = affine[:3,:3]
            d = aff_m @ t
            affine[0:3,3] = t-d
            ans = affine_flow(affine, shape).movedim(-1, 0).to(vol_n.device)
            flow_new[:,ss*i:ss*(i+1),:,:] = ans

        return flow_new

    def _preprocess(self, img1, seg1):
        """Normalize image and prepare binary mask."""
        if self.normalize_img:
            img1 = (img1.clamp(min=0.1) - 0.1) * (1 / 0.9)
        seg1 = (seg1 > 0).float()
        while seg1.ndim < 4:
            seg1 = seg1.unsqueeze(0)
        
        if self.cut != 0:
            img1 = img1[:,self.cut:-self.cut,self.cut:-self.cut,self.cut:-self.cut]
            seg1 = seg1[:,self.cut:-self.cut,self.cut:-self.cut,self.cut:-self.cut]
        
        return img1, seg1

    def _apply_pre_motion_augmentations(self, img1):
        """Apply smoothing, noise, bias field, and gamma augmentations."""
        if self.smooth:
            img1 = self.smoother(img1)
        if self.noise > 0:
            img1 = self.mult(img1)
        if self.bias_field:
            img1 = self.bias(img1)
        if self.gamma_field:
            img1 = self.gamma(img1)
        return img1

    def _apply_zoom(self, img1, seg1):
        """Apply random zoom augmentation to image and mask."""
        xform = self.zoom.make_final(img1)
        return xform(img1), xform(seg1)

    def _sample_rot180_flags(self):
        """Sample random 180-degree rotation flags per stack."""
        if self.stack_order_pred:
            return [random.randint(0, 1) for _ in range(3)]
        return [0, 0, 0]

    def _build_motion_transforms(self, img1):
        """Create per-stack slicewise affine transforms."""
        if self.mlp_training:
            self.slice = random.sample(self.slice, k=len(self.slice))
        
        if self.spacing_range is not None:
            shared_spacing = float(Uniform(*self.spacing_range)())
            self.spacing = shared_spacing
        else:
            shared_spacing = self.spacing
        for b in self.base:
            b.spacing = shared_spacing
        print("B SPACING")
        print(b.spacing)
        return [self.base[sl].make_final(img1) for sl in self.slice]

    def _apply_motion_transforms(self, img1, seg1, xform, rot180_stack):
        """Apply motion transforms with optional 180-degree flips."""
        if self.stack_order_pred and not self.rot_180_axis:
            img0 = torch.cat([
                self.flip(xform[i](img1)) if rot180_stack[i] == 1 else xform[i](img1)
                for i in range(len(self.slice))
            ], dim=1)
            seg0 = torch.cat([
                self.flip(xform[i](seg1).gt(0).float()) if rot180_stack[i] == 1 else xform[i](seg1).gt(0).float()
                for i in range(len(self.slice))
            ], dim=1)

        elif self.stack_order_pred and self.rot_180_axis:
            imgs, segs = [], []
            for i in range(len(self.slice)):
                if rot180_stack[i] == 1:
                    flip_fn = self.flip_ax1 if self.slice[i] == 2 else self.flip_ax2
                    img_i = xform[i](flip_fn(img1))
                    seg_i = xform[i](flip_fn(seg1)).gt(0).float()
                else:
                    img_i = xform[i](img1)
                    seg_i = xform[i](seg1).gt(0).float()
                imgs.append(img_i)
                segs.append(seg_i)
            img0 = torch.cat(imgs, dim=1)
            seg0 = torch.cat(segs, dim=1)

        else:
            
            if not self.gaussian_psf:
                img0 = torch.cat([xform[i](img1) for i in range(len(self.slice))], dim=1)
                seg0 = torch.cat([xform[i](seg1).gt(0).float() for i in range(len(self.slice))], 1)
                flow = torch.cat([xform[i].flow for i in range(len(self.slice))], 1)
                ALL_STACKS = None
            if(self.gaussian_psf):
              #  pdb.set_trace()
              #  img0 = torch.cat([self.stack_in_single_plane_stack(xform[i](img1),self.slice[i]) for i in range(len(self.slice))], dim=1)
                
                img_all = []
                stack_all = []
                for i in range(len(self.slice)):
                    img0_ = self.stack_in_single_plane_stack(xform[i](img1),self.slice[i])
                    ALL_STACKS_ =  self._build_stack_positions_stack(img0_, self.slice[i])
               
                    if(i==1): #i==1
                        ALL_STACKS_[0,:,self.slice[i],3] = ALL_STACKS_[0,:,self.slice[i],3] - (self.spacing/self.subsample-1)/2  
                    else:
                        ALL_STACKS_[0,:,self.slice[i],3] = ALL_STACKS_[0,:,self.slice[i],3] + (self.spacing/self.subsample-1)/2  
                    ALL_STACKS_[1,:,0,3] = ALL_STACKS_[1,:,0,3] + (self.spacing/self.subsample-1)/2  

                    img_all.append(img0_)
                    stack_all.append(ALL_STACKS_)
                img0 = torch.cat(img_all, dim=1)
                ALL_STACKS = torch.cat(stack_all, dim=1)
               
                seg0 = torch.cat([self.stack_in_single_plane_stack(xform[i](seg1).gt(0).float(),self.slice[i]) for i in range(len(self.slice))], dim=1)
                flow = torch.cat([self.stack_in_single_plane_stack(xform[i].flow, self.slice[i]) for i in range(len(self.slice))], 1)
                print("check spacing is saved")
                print(xform[0].spacing, xform[1].spacing, xform[2].spacing)
                print(type(xform[0].spacing), type(xform[1].spacing), type(xform[2].spacing))
                self.spacing = xform[0].spacing
              #  seg0 = torch.cat([xform[i](seg1).gt(0).float() for i in range(len(self.slice))], 1)
                

        
        return img0, seg0, flow, ALL_STACKS




    def _combine_img_flow_and_reorient(self, img0, seg0, flow):
        """Concatenate mask channels, reorient to single plane, apply post-motion augmentations."""
        img0 = torch.cat([img0, seg0])
        flow = torch.cat([flow, seg0])

        img0 = self.stack_in_single_plane(img0)
        flow = self.stack_in_single_plane(flow)

        if self.slice_bias_apply:
            img0[:1] = self.slice_bias(img0[:1])
        if self.signal_void_apply:
            img0[:1] = self.signal_void(img0[:1])
        if self.rician_noise_apply:
            img0[:1] = self.rician_noise(img0[:1])

        return img0, flow


    def _combine_img_flow_and_reorient_stack(self, img0, seg0, flow):
        """Concatenate mask channels, reorient to single plane, apply post-motion augmentations."""
        img0 = torch.cat([img0, seg0])
        flow = torch.cat([flow, seg0])

        # img0 = self.stack_in_single_plane(img0)
        # flow = self.stack_in_single_plane(flow)

        if self.slice_bias_apply:
            img0[:1] = self.slice_bias(img0[:1])
        if self.signal_void_apply:
            img0[:1] = self.signal_void(img0[:1])
        if self.rician_noise_apply:
            img0[:1] = self.rician_noise(img0[:1])

        return img0, flow
    def _build_total_flow(self, flow):
        """Combine motion flow with zero orthogonal residual."""
        new_flow = torch.zeros_like(flow)
        new_flow[0:3] = flow[None][:, :3, :, :, :]
        new_flow[3] = flow[None][0, 3, :, :, :]  # keep mask in last dimension
        return new_flow

    def _build_stack_positions(self, img0, flow):
        """Build ALL_STACKS position matrices for each slice stack."""
        if self.slice == [0, 1, 2]:
            sl_shape = img0.shape[2]
            shape_list = [sl_shape, sl_shape, sl_shape]

            STACK_OG = og_slice_pos_pre(sl_shape, [1,1,1], 1, 0, shape_list, device=flow.device)
            STACK1 = og_slice_pos_pre(sl_shape, [1,1,1], 1, self.slice[0], shape_list, device=flow.device)
            STACK2 = og_slice_pos_pre(sl_shape, [1,1,1], 1, self.slice[1], shape_list, device=flow.device)
            STACK3 = og_slice_pos_pre(sl_shape, [1,1,1], 1, self.slice[2], shape_list, device=flow.device)

            ALL_STACKS = torch.zeros((2, sl_shape*3, 4, 4), device=flow.device)
            ALL_STACKS[0, 0:sl_shape] = STACK1
            ALL_STACKS[0, sl_shape:sl_shape*2] = STACK2
            ALL_STACKS[0, sl_shape*2:sl_shape*3] = STACK3

            ALL_STACKS[1, 0:sl_shape] = STACK1
            ALL_STACKS[1, sl_shape:sl_shape*2] = STACK1
            ALL_STACKS[1, sl_shape*2:sl_shape*3] = STACK1

        else:
            len_s = len(self.slice)
            sl_shape = img0.shape[1] // len_s
            shape_list = [sl_shape, sl_shape, sl_shape]

            STACK_OG = og_slice_pos_pre(sl_shape, [1,1,1], 1, 0, shape_list, device=flow.device)
            stacks = [og_slice_pos_pre(sl_shape, [1,1,1], 1, self.slice[i], shape_list, device=flow.device) for i in range(len_s)]

            ALL_STACKS = torch.zeros((2, img0.shape[1], 4, 4), device=flow.device)
            for i in range(len_s):
                ALL_STACKS[0, sl_shape*i:sl_shape*(i+1)] = stacks[i]
                ALL_STACKS[1, sl_shape*i:sl_shape*(i+1)] = STACK_OG

        return ALL_STACKS

    def _build_stack_positions_stack(self, img0, slice):
        """Build ALL_STACKS position matrices for each slice stack."""

        print("IN BUILD ALL STACKS")
        num_slices = img0.shape[1]
        ALL_STACKS = torch.zeros((2, num_slices, 4, 4), device=img0.device)
        ALL_STACKS[0] = og_slice_pos_pre(num_slices, [1,1,1], self.spacing/self.subsample,slice, [256/self.subsample,256/self.subsample,256/self.subsample], device=img0.device)
        ALL_STACKS[1] = og_slice_pos_pre(num_slices, [1,1,1], self.spacing/self.subsample,0, [256/self.subsample,256/self.subsample,256/self.subsample], device=img0.device)


        return ALL_STACKS


    def _remove_duplicates_and_empty(self, img0, flow, ALL_STACKS):
        """Subsample repeated slices and remove slices with empty masks."""
        rep = int(self.spacing / self.subsample)

        
        if self.gaussian_psf:
            rep = 1
        img0 = img0[:, ::rep]
        flow = flow[:, ::rep]
        ALL_STACKS = ALL_STACKS[:, ::rep]
        

        # find slices with non-zero masks
        keep = img0[1:].reshape(img0[1:].shape[1], -1).any(dim=1)

        # ensure even number of slices
        if keep.sum() % 2 == 1:
            idx_last = torch.nonzero(keep, as_tuple=True)[0][-1]
            idx_first = torch.nonzero(keep, as_tuple=True)[0][0]
            if idx_last < keep.shape[0] - 1:
                keep[idx_last + 1] = 1
            elif idx_first > 0:
                keep[idx_first - 1] = 1
            else:
                keep[idx_first] = 0
                print("ODD NUMBER OF SLICES")

        img0 = img0[:, keep]
        ALL_STACKS = ALL_STACKS[:, keep]
        flow = flow[:, keep]
        return img0, flow, ALL_STACKS

    def _apply_slice_dropout(self, img0, ALL_STACKS):
        """Randomly drop slices during MLP training augmentation."""
        drop_out = torch.rand(1).item() * self.slice_drop_out
        img0 = img0.contiguous()
        B, S, H, W = img0.shape

        keep = max(1, int((1 - drop_out) * S))
        idx = torch.arange(S, device=img0.device, dtype=torch.long)
        perm = torch.randperm(S, device=img0.device)
        idx = idx[perm][:keep]
        idx, _ = torch.sort(idx)

        return img0.index_select(1, idx), ALL_STACKS.index_select(1, idx)

    def _build_mlp_labels(self, rot180_stack, rot_buckets, device):
        """Build one-hot encoded labels for MLP training."""
        num_classes = 3
        num_rotations = 4
        one_hot_stack = FF.one_hot(torch.tensor(self.slice), num_classes=num_classes).float().to(device)
        one_hot_rots = FF.one_hot(rot_buckets, num_classes=num_rotations).float().to(device)
        one_hot = torch.cat([one_hot_stack, one_hot_rots], dim=1)

        if self.predict_vec and not self.no_rotation:
            stack_orientation_6 = torch.tensor(self.slice) * 2
            stack_neg = torch.tensor(rot180_stack) + stack_orientation_6
            one_hot_stack_dir = FF.one_hot(stack_neg, num_classes=6).float().to(device)
            return torch.cat([one_hot_stack_dir, one_hot_rots], dim=1)

        if self.predict_vec and self.no_rotation:
            stack_orientation_6 = torch.tensor(self.slice) * 2
            stack_neg = torch.tensor(rot180_stack) + stack_orientation_6
            return FF.one_hot(stack_neg, num_classes=6).float().to(device)

        if self.no_rotation:
            one_hot_order = FF.one_hot(torch.tensor(rot180_stack), num_classes=2).float().to(device)
            return torch.cat([one_hot_stack, one_hot_order], dim=1)

        if self.stack_order_pred:
            one_hot_order = FF.one_hot(torch.tensor(rot180_stack), num_classes=2).float().to(device)
            return torch.cat([one_hot_stack, one_hot_rots, one_hot_order], dim=1)

        return one_hot

    def __call__(self, img1, seg1):
        # Handle batched 5D input
        if img1.ndim == 5:
            img1, seg1 = zip(*[self(img1[i], seg1[i]) for i in range(img1.shape[0])])
            if self.crop:
                return (img1[0][0][None], img1[0][1]), torch.stack(seg1, 0)
            else:
                return torch.stack(img1, 0), torch.stack(seg1, 0)

        # 1. Preprocess image and mask
        img1, seg1 = self._preprocess(img1, seg1)

        # 2. Apply pre-motion augmentations
        img1 = self._apply_pre_motion_augmentations(img1)

        # 3. Sample rotation flags and apply zoom
        rot180_stack = self._sample_rot180_flags()
        img1, seg1 = self._apply_zoom(img1, seg1)

        # 4. Build and apply per-stack motion transforms
        xform = self._build_motion_transforms(img1)


        img0, seg0, flow, ALL_STACKS = self._apply_motion_transforms(img1, seg1, xform, rot180_stack)

        # 5. Combine channels, reorient slices, apply post-motion augmentations
        if(not self.gaussian_psf):
            img0, flow = self._combine_img_flow_and_reorient(img0, seg0, flow)
        else:
            img0, flow = self._combine_img_flow_and_reorient_stack(img0, seg0, flow)

        # 6. Build total flow (motion + zero orthogonal residual)
        flow = self._build_total_flow(flow)

        if not self.crop:
            return img0, flow

        # 7. Build stack position matrices, remove duplicates and empty slices
        if(self.gaussian_psf == False):
            ALL_STACKS = self._build_stack_positions(img0, flow)
      #  pdb.set_trace()
        img0, flow, ALL_STACKS = self._remove_duplicates_and_empty(img0, flow, ALL_STACKS)

        # 8. MLP training: build labels and optionally apply slice dropout
        if self.mlp_training:
            one_hot = self._build_mlp_labels(rot180_stack, rot_buckets, flow.device)

            if self.predict_vec and not self.no_rotation and self.slice_drop_out != 0 and self.augment:
                img0, ALL_STACKS = self._apply_slice_dropout(img0, ALL_STACKS)

            return (img0, ALL_STACKS), one_hot

        return (img0, ALL_STACKS), flow



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
        img = torch.as_tensor(img)
        seg = torch.as_tensor(seg)

        return img, seg