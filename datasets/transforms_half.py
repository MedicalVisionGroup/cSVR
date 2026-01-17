import sys
import math
import torch
import torch.nn as nn
import random
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import interpol
import cornucopia as cc
from PIL import Image
import pdb
import inspect
from cornucopia.utils.warps import affine_flow
from scipy import ndimage
from cornucopia.random import Sampler, Normal



#in order to save image
from nilearn.datasets import load_mni152_template
from nilearn.plotting import plot_img
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


project_root = "/data/vision/polina/users/mfirenze/cSVR/models"

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

class ToGPU():
    def __call__(self, img, seg):
        return img, seg

class NLLNormalize():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, seg):
        return (1/self.std) * (img.log_softmax(dim=0) - self.mean), seg

class OneHotLabels():
    def __init__(self, num_classes, index=0):
        self.num_classes = num_classes
        self.index = index

    def __call__(self, img, seg):
        if seg == None:
            return img, seg

        if seg.ndim == 5:

            img, seg = zip(*[self(img[i], seg[i]) for i in range(img.shape[0])])
            return torch.stack(img, 0), torch.stack(seg, 0)

        hot = nn.functional.one_hot(seg[self.index].long(), num_classes=self.num_classes).movedim(-1,0)
        seg = torch.cat([seg[:self.index], hot, seg[self.index+1:]], 0)

        return img, seg

class OneHotTwoLabels(OneHotLabels):
    def __call__(self, img, seg):
        if seg == None:
            return img, seg

        _, seg0 = super().__call__(None, seg[0:1])
        _, seg1 = super().__call__(None, seg[1:2])

        return img, torch.cat([seg0, seg1], 0)

class Quantize():
    def __init__(self, quantiles=torch.arange(0.5,1.0,0.05)):
        self.quantiles = torch.as_tensor(quantiles)

    def __call__(self, img, seg):
        seg = torch.bucketize(seg.contiguous(), torch.cat([torch.tensor([-np.inf]), seg.quantile(self.quantiles), torch.tensor([np.inf])],0))
        # seg = torch.bucketize(seg, torch.cat([seg.quantile(self.quantiles)],0))
        return img, seg - 1

class SigmoidLabels():
    def __call__(self, img, seg):
        return img, torch.sigmoid(seg)

class GaussLabels():
    def __init__(self, num_classes=25, sigma=0.5):
        self.num_classes = num_classes
        self.gauss = GaussianFilter(1, (sigma))
        self.sigma = sigma
        
    def __call__(self, img, seg):
        if seg == None:
            return img, seg

        delta = (seg.max() - seg.min()) / self.num_classes
        seg = torch.floor((seg - seg.min()) / delta).clamp(max=self.num_classes - 1)
        seg = torch.nn.functional.one_hot(seg[0].long(), num_classes=self.num_classes).permute([2,0,1])

        if self.sigma > 0:
            seg = self.gauss(seg.float())
            seg = (seg / seg.sum(0, keepdims=True)) #(seg / seg.sum(-1, keepdims=True)).permute([2, 0, 1])
        
        return img, seg

class OneHotImages():
    def __init__(self, num_classes, exclude_ignore_index=True):
        self.num_classes = num_classes
        self.exclude_ignore_index = exclude_ignore_index

    def __call__(self, img, seg):
        img = img[0].clamp(max=self.num_classes)
        dims = list(range(img.ndim + 1))

        img = torch.nn.functional.one_hot(img.long(), num_classes=self.num_classes + 1)\
                                 .permute(dims[-1:] + dims[:-1]).float()#.log()

        if self.exclude_ignore_index:
            img = img[:self.num_classes]

        return img, seg

class Laplacian3D():
    def __init__(self, kernel_size=3):
        self.kernel_size = kernel_size

    def __call__(self, img, seg):
        img = img - nn.functional.avg_pool3d(img[None], kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2)[0]

        return torch.abs(img), seg

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

class Rand2DElastic():
    def __init__(
            self,
            spacing = [16, 16],
            magnitude_range = [-4, 4],
            spatial_size = [256, 512],
            intermode = 'nearest',
    ):
        self.spacing = spacing
        self.magnitude_range = magnitude_range
        self.spatial_size = spatial_size
        self.mode = intermode

    def __call__(self, img, seg):
        img = torch.nn.functional.pad(img[None], (0,1,0,0), mode='circular') # pad horizontally
        seg = torch.nn.functional.pad(seg[None], (0,1,0,0), mode='circular') # pad horizontally
        rand = torch.zeros(1, 2, self.spatial_size[0] // self.spacing[0] + 0, self.spatial_size[1] // self.spacing[1] + 1)
        rand[:,:,1:-1,1:-1].uniform_(self.magnitude_range[0], self.magnitude_range[1])
        rand = torch.nn.functional.interpolate(rand.float(), (self.spatial_size[0] + 0, self.spatial_size[1] + 1), mode='bilinear', align_corners=True)
        grid = torch.stack(torch.meshgrid(torch.arange(0, self.spatial_size[0] + 0), torch.arange(0, self.spatial_size[1] + 1))).unsqueeze(0)

        grid = (grid + rand) - (0.5 * torch.tensor([self.spatial_size[0] - 1, self.spatial_size[1] - 0]).reshape(1,2,1,1))
        grid = (grid * (2.0 / torch.tensor([self.spatial_size[0] - 1, self.spatial_size[1] - 0]).reshape(1,2,1,1)))[:,[1,0],:,:-1]

        img = torch.nn.functional.grid_sample(img, grid.permute(0,2,3,1), align_corners=True, mode='bicubic')[0]
        seg = torch.nn.functional.grid_sample(seg, grid.permute(0,2,3,1), align_corners=True, mode=self.mode)[0]

        return img, seg

class RandGaussianSmooth():
    def __init__(self, sigma=(0.5, 1), prob=0.2, approx='erf'):
        self.sigma = sigma
        self.prob = prob
        self.approx = approx
    
    def __call__(self, img, seg):
        if torch.rand(1) > self.prob:
            return img, seg

        sigma = torch.empty(1).uniform_(self.sigma[0], self.sigma[1]).item()
        filter = GaussianFilter(3, sigma, approx=self.approx)

        return filter(img), seg

class RandScaleIntensity():
    def __init__(self, factors = (-0.25, 0.25), prob = 0.15):
        self.prob = prob
        self.factors = factors

    def __call__(self, img, seg):
        if torch.rand(1) > self.prob:
            return img, seg

        factor = torch.empty(1).uniform_(self.factors[0], self.factors[1]).item()
        return img * (1 + factor), seg

class RandAdjustContrast():
    def __init__(self, prob = 0.15, factors = (0.75, 1.25)):
        self.prob = prob
        self.factors = factors

    def __call__(self, img, seg):
        if torch.rand(1) > self.prob:
            return img, seg

        factor = torch.empty(1).uniform_(self.factors[0], self.factors[1]).item()
        return ((img - img.mean()) * factor + img.mean()).clamp(img.min(), img.max()), seg

class RandAdjustGamma():
    def __init__(self, prob = 0.3, factors = (0.7, 1.5), invert=False):
        self.prob = prob
        self.factors = factors
        self.invert = invert

    def __call__(self, img, seg):
        if torch.rand(1) > self.prob:
            return img, seg

        img = -img if self.invert else img

        factor = torch.empty(1).uniform_(self.factors[0], self.factors[1]).item()
        newimg = torch.pow((img - img.min()) / (img.max() - img.min() + 1e-8), factor) 
        
        return (newimg - newimg.mean()) / (newimg.std() + 1e-8) * img.std() + img.mean(), seg

class DropoutNoise():
    def __init__(self, prob=(0, 0.1), scales=(1,5), **kwargs):
        self.prob = prob
        self.scales = scales
        # self.avgfilter = torch.nn.AvgPool3d(3, stride=1, padding=1)

    def __call__(self, image, target):
        prob = torch.empty(1).uniform_(self.prob[0], self.prob[1]).item()
        # scale = 2 ** torch.randint(0,2, (1,)).item()
        scale = torch.randint(self.scales[0], self.scales[1], (1,)).item()
        scale = torch.tensor([1, 1, scale, scale, scale])
        shape = ((torch.as_tensor(image[None].shape) / scale).int()).tolist()
        noise = torch.rand(shape) > self.prob   # self.avgfilter((torch.randn(shape)))
        noise = nn.functional.interpolate(noise, image[0].shape, mode='nearest', align_corners=False)
        image = image * noise[0]

        return image, target

class Resample3dSlice:
    def __init__(self, spacing=1, slice=1, size=None):
        self.spacing = spacing
        self.slice = slice - 3

    def __call__(self, img1, seg1):
        if img1.ndim == 5:
            img1, seg1 = zip(*[self(img1[i], seg1[i]) for i in range(img1.shape[0])])
            return torch.stack(img1, 0), torch.stack(seg1, 0)

        img0 = img1.repeat_interleave(self.spacing, self.slice)
        seg0 = seg1.repeat_interleave(self.spacing, self.slice)
        flow = torch.zeros([3] + list(img0.shape[1:]))

        return img0 * seg0, seg0 #torch.cat([flow.flip(0), seg0], 0)

class BoundingBox3d:
    def __init__(self, spacing=1, subsample=1):
        self.spacing = spacing / subsample

    def __call__(self, img1, seg1, mask):
        nonz = mask.nonzero() # seg1[self.index][None].nonzero()
        mins = [(ind - 1 * self.spacing).div(self.spacing).int().mul(self.spacing).int() for ind in nonz.min(0).values[-3:]]
        mins = torch.tensor(mins).clamp(torch.tensor([0,0,0]), torch.tensor(img1.shape[-3:]))

        maxs = [(ind + 1 * self.spacing).div(self.spacing).int().mul(self.spacing).int() for ind in nonz.max(0).values[-3:]]
        maxs = torch.tensor(maxs).clamp(torch.tensor([0,0,0]), torch.tensor(img1.shape[-3:]))

        img1 = img1[..., mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]
        seg1 = seg1[..., mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]

        return img1, seg1

class RandDilate3dSlice:
    def __init__(self, slice=1, spacing=1, size=(3,9), p=0.5):
        self.slice = slice
        self.spacing = spacing
        self.size = size
        self.p = p

    def __call__(self, img1, seg1):
        if img1.ndim == 5:
            img1, seg1 = zip(*[self(img1[i], seg1[i]) for i in range(img1.shape[0])])
            return torch.stack(img1, 0), torch.stack(seg1, 0)

        fore = (seg1 != 0).movedim(self.slice + 1, 1)[:, ::self.spacing].contiguous()
        back = (img1 == 0).movedim(self.slice + 1, 1)[:, ::self.spacing].contiguous()

        for i in range(back.shape[1]):
            if torch.rand(1).item() > self.p:
                continue

            size = (torch.randint(self.size[0], self.size[1] + 1, [2]) - 1) // 2 * 2 + 1 #.tolist()
            padd = size // 2
            fore[:,i] = nn.functional.max_pool2d(fore[:,i].float(), kernel_size=size.tolist(), stride=1, padding=padd.tolist())

            size = (torch.randint(self.size[0], self.size[1] + 1, [2]) - 1) // 2 * 2 + 1 #.tolist()
            padd = size // 2
            back[:,i] = nn.functional.max_pool2d(back[:,i].float(), kernel_size=size.tolist(), stride=1, padding=padd.tolist())

        mask = (~back | fore).repeat_interleave(self.spacing, 1).movedim(1, self.slice + 1).contiguous()

        return img1 * mask, seg1

class RandAffine3dSlice:
    # Firs tline is original
    # changed bulk translations to 0
    def __init__(self, spacing=1, translations=0.1, rotations=20, bulk_translations=0, bulk_rotations=45, zooms=0, subsample=1, slice=1, nodes=(8,16), shots=2, augment=True, noise=False, X=3):
 #   def __init__(self, spacing=1, translations=0, rotations=0, bulk_translations=0, bulk_rotations=180, zooms=0, subsample=1, slice=1, nodes=(8,16), shots=2, augment=True, noise=False, X=3):

        self.slice = slice if isinstance(slice, (tuple, list)) else [slice]
        # self.crop = cc.fov.PatchTransform(192)
        self.flip = cc.fov.RandomFlipTransform(axes=[-3]) #+ cc.fov.PatchTransform(192)
        self.zoom = cc.MaybeTransform(cc.RandomAffineTransform(translations=0, rotations=0, shears=0, zooms=zooms, iso=True), 0.9)
        self.base = [cc.RandomSlicewiseAffineTransform(nodes=nodes, shots=shots, spacing=spacing, subsample=subsample, slice=s, translations=translations, rotations=rotations,
                                                       bulk_translations=bulk_translations, bulk_rotations=bulk_rotations, shears=0, zooms=0) for s in self.slice]
        self.mult = cc.MaybeTransform(cc.RandomGaussianNoiseTransform(sigma=0.01), 0.5) \
            # + cc.MaybeTransform(cc.RandomGammaTransform(value=(0.7, 1.5)), 0.5) + cc.MaybeTransform(cc.RandomMultFieldTransform(), 0.5)
        self.bound = BoundingBox3d(spacing=spacing, subsample=subsample)
        self.augment = augment #add_noise
        self.noise = noise
        print("MOTION PARAMS")
        print("rot:")
        print(rotations)
        print("translation:")
        print(translations)

        
        self.X = X
    def __call__(self, img1, seg1):
        
     #   print("in call")
     #   print(f"Original shape of img1 {img1.shape}")
     #   pdb.set_trace()
        if img1.ndim == 5:
         #   print("hello")
            img1, seg1 = zip(*[self(img1[i], seg1[i]) for i in range(img1.shape[0])])
            return torch.stack(img1, 0), torch.stack(seg1, 0)
        
      #  pdb.set_trace()
        numstacks = 1 #torch.randint(1, len(self.slice) + 1, [1]).item()
        img1 = (img1.clamp(min=0.1) - 0.1) * (1 / 0.9)
        seg1 = ((img1 > 0) | (seg1 > 0)).float()
        # img1, seg1 = self.crop(img1, seg1)
      #  print(f"Next shape of img1 {img1.shape}")
        img1, seg1 = self.flip(img1, seg1) if self.augment else (img1, seg1)
      #  pdb.set_trace()
        xform = self.zoom.make_final(img1)
       # pdb.set_trace()
        img1, seg1 = xform(img1, seg1) if self.augment else (img1, seg1)
       # print(f"Next shape of img1 {img1.shape}")
       # pdb.set_trace()
      #  pdb.set_trace()
        xform = [self.base[i].make_final(img1) for i in range(numstacks)]
        img0 = torch.cat([xform[i](img1) for i in range(numstacks)], 1)
       # pdb.set_trace()
        seg0 = torch.cat([xform[i](seg1).gt(0).float() for i in range(numstacks)], 1)
        # img0 = self.mult(img0) if self.noise else img0
        flow = torch.cat([xform[i].flow.flip(0) for i in range(numstacks)], 1) # 
        # img0, flow = self.bound(torch.cat([img0, seg0]), torch.cat([flow, seg0]), mask=seg0)
        img0, flow = torch.cat([img0, seg0]), torch.cat([flow, seg0])
      #  print(f"Final shape of img1 {img0.shape}")
      #  pdb.set_trace()


        return img0, flow


class GenerateMotionTrajectory:
    #zooms=(-0.35,0.02) d
    def __init__(self, spacing=1, subsample=1, translations=0.1, rotations=20, bulk_translations=0, bulk_rotations=0, zooms=0, slice=1, nodes=(8,16), shots=2, augment=False, noise=False, X=3, flow_final=False, crop=False):


        print("MOTION PARAMS:")
        print(f"spacing: {spacing}")
        print(f"subsampling: {subsample}")
        print(f"translations: {translations}")
        print(f"rotations: {rotations}")
        print(f"bulk rotation: {bulk_rotations}")
        print(f"bulk translations: {bulk_translations}")
        print(f"slice: {slice}")
        print(f"crop: {crop}")
        print(f"augment: {augment}")
        print(f"zooms: {zooms}")
        print(f"noise: {noise}")

        self.subsample = subsample
        self.spacing = spacing 
        self.slice = slice if isinstance(slice, (tuple, list)) else [slice]
       # self.flip = cc.fov.RandomFlipTransform(axes=[-3]) #+ cc.fov.PatchTransform(192) # Turns flips off
        self.zoom = cc.RandomAffineTransform(translations=0, rotations=0, shears=0, zooms=zooms, iso=True)
        self.augment = augment #add_noise
        self.noise = noise
        self.crop = crop
        self.flow_final = flow_final
        self.X = X # nu,ber of stacks

        random_motion = True # only True for debugging purposes
        if(random_motion):
            self.base = [cc.RandomSlicewiseAffineTransform(nodes=nodes, shots=shots, spacing=spacing, subsample=subsample, slice=s, translations=translations, rotations=rotations,
                                                        bulk_translations=bulk_translations, bulk_rotations=bulk_rotations, shears=0, zooms=0) for s in self.slice]
        else: # only for debugging purposes
            print("NOT RANDOM MOTION")
            trans_new = torch.tensor([[0, 0, 0] for i in range(64)])
            trans_new = trans_new.view(64,3).tolist()
            rots_new = torch.tensor([[0, 45, 0] for i in range(64)])
            rots_new = rots_new.view(64,3).tolist()
            self.base = [cc.SlicewiseAffineTransform(spacing=spacing, subsample=subsample, slice=s,rotations=rots_new , shears=torch.tensor([0,0,0]),  #30*torch.ones((3,128)
                                                            translations= trans_new, zooms = torch.tensor([0,0,0]), unit='vox')  for s in self.slice]# CHANGED ZOOMS #torch.tensor([0,0.5,0.0])
      
        # Define transfomrs
      #  self.mult = cc.MaybeTransform(cc.RandomGaussianNoiseTransform(sigma=0.1), 1) 
        self.mult = cc.MaybeTransform(cc.RandomGaussianNoiseTransform(sigma=0.08), 0.5) \
            # + cc.MaybeTransform(cc.RandomGammaTransform(value=(0.7, 1.5)), 0.5) + cc.MaybeTransform(cc.RandomMultFieldTransform(), 0.5)
      #  self.bound = BoundingBox3d_3stacks(spacing=spacing, subsample=subsample)



    def stack_in_single_plane(self, og): # turn slices to correct plane
            if(self.slice == [0,1,2]):
                ss = og.shape[1]//3
            #   ss = og.shape[2]
                new_vol = og.clone()
                stack0 = og[:,0:ss,:,:]
                stack1 = og[:,ss:ss*2,:,:]
                stack2 = og[:,ss*2:ss*3,:,:]

                
                stack0_n = stack0.clone()
                stack1_n = torch.rot90(stack1, k=1, dims=(1, 2))

                stack2_n = torch.rot90(stack2, k=-1, dims=(1, 3)) # for slice1
                stack2_n = torch.rot90(stack2_n, k=1, dims=(2, 3))

                new_vol[:,0:ss,:,:] = stack0_n
                new_vol[:,ss:ss*2,:,:] = stack1_n
                new_vol[:,ss*2:ss*3,:,:] = stack2_n
                return new_vol
            if(self.slice == [0,0,0]):
                return og

    
    def generate_initial_flow(self, vol_n): # add correct offset to account for planar slices
      
        
        if(self.slice == [0,1,2]):
          #  ss = vol_n.shape[2]//3
            ss = vol_n.shape[3]
            new_vol = vol_n.clone()
            stack0 = vol_n[:,:,0:ss,:,:]
            stack1 = vol_n[:,:,ss:ss*2,:,:]
            stack2 = vol_n[:,:,ss*2:ss*3,:,:]

            
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
                    print("slice 1 rot")
                    print(affine)

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

                flow_new[:,ss*sl:ss*(sl+1),:,:] = ans

                
            #flow_new_p2 = torch.cat((flow_new,torch.ones_like(flow_new[0][None])), axis=0)
            return flow_new
        if(self.slice == [0,0,0]):
            flow_new = torch.zeros((3,vol_n.shape[2],vol_n.shape[3],vol_n.shape[4])).to(vol_n.device)
            return flow_new

    
    def __call__(self, img1, seg1):

        if img1.ndim == 5 :
            img1, seg1 = zip(*[self(img1[i], seg1[i]) for i in range(img1.shape[0])])
            if(self.crop):
                return (img1[0][0][None], img1[0][1]), torch.stack(seg1, 0)
            else:
                return torch.stack(img1, 0), torch.stack(seg1, 0)
       
        numstacks = 3 #torch.randint(1, len(self.slice) + 1, [1]).item()img2
        
        # normalize image - remove if want to avoid shifts in brightness
        img1 = (img1.clamp(min=0.1) - 0.1) * (1 / 0.9)
        seg1 = ((img1 > 0) | (seg1 > 0)).float()

        # add noise and zooms

        img1 = self.mult(img1) if self.noise else img1
        xform = self.zoom.make_final(img1)
        img1 = xform(img1)
        seg1 = xform(seg1)

        # applies RandomSliceWiseAffine transform to make each stack seperately 
        
        xform = [self.base[i].make_final(img1) for i in range(numstacks)]
            
        #concatenate image intensities, masks, and flow into one tensor for all stacks
        img0 = torch.cat([xform[i](img1) for i in range(numstacks)], 1)
        seg0 = torch.cat([xform[i](seg1).gt(0).float() for i in range(numstacks)], 1)
        flow = torch.cat([xform[i].flow for i in range(numstacks)], 1) # 
 

        # concatenate mask to image tensor and flow tensor
        img0, flow = torch.cat([img0, seg0]), torch.cat([flow, seg0])

        # get all slice intensity values only to save   
        all_slices = img0[0][:,:,:][None][None] 
        # torch.save(all_slices[:,:,0:256:2], "slices_stack1_256_id.pth")
        # torch.save(all_slices[:,:,256:512,::2], "slices_stack2_256_id.pth")
        # torch.save(all_slices[:,:,512:768,:,::2], "slices_stack3_256_id.pth")
        
        # put all the images in plane and change flow to also be in plane
        img0 = self.stack_in_single_plane(img0) # TEMPORARY
        flow = self.stack_in_single_plane(flow)

        # create new flow that corresponds to the original plane the slices were in
        ONE_LAYER_ONLY = False
        ot = True
        if(ONE_LAYER_ONLY):
            ot = False
         # for debugging turn off
       # ot = True
        if(ot):
            print("FAKE FLOW TO TEST SAMPLING")
            flow = flow[:,::2,::2,::2]
            flow_ot = self.generate_initial_flow(flow[None][:,1:4])
        #    flow_ot = self.generate_initial_flow(flow[None][:,1:4,::2,::2,::2])
        else:
            print("NOT INCLUDING OT!!!")

            flow_ot = self.generate_initial_flow(flow[None][:,1:4])
            flow_ot = torch.zeros_like(flow_ot)
        #  flow_ot = torch.zeros_like(flow_ot)
        
        new_flow = torch.zeros_like(flow).to(flow.device)
       # new_flow[0:3] = (flow_ot+ flow[None][:,:3,:,:,:])
        print("CHANGED")
        new_flow[0:3] = (flow_ot+ flow[None][:,:3,:,:,:]*0.5)
        new_flow[3] = flow[None][0,3,:,:,:] # keep mask in last dimension
        flow = new_flow

        if (self.crop==False):
            print("hola 8 no flip!")
            return img0, flow
        else: # remove duplicates and black slices
            print("CROP!!!!!!!!")
    
            sl_shape = img0.shape[2]
            print("here :))))")
            #slice_dim 
            STACK1 = og_slice_pos_pre(sl_shape, [1,1,1], 1, 0, [sl_shape,sl_shape,sl_shape], device=flow.device)
            STACK2 = og_slice_pos_pre(sl_shape, [1,1,1], 1, 1, [sl_shape,sl_shape,sl_shape], device=flow.device)
            STACK3 = og_slice_pos_pre(sl_shape, [1,1,1], 1, 2, [sl_shape,sl_shape,sl_shape], device=flow.device)

            ALL_STACKS = torch.zeros((2, sl_shape*3,4,4), device=flow.device)
            ALL_STACKS[0,0:sl_shape] =   STACK1
            ALL_STACKS[0, sl_shape:sl_shape*2] = STACK2
            ALL_STACKS[0, sl_shape*2:sl_shape*3] = STACK3
            no_stack_info = False # do not give fact that slices are orthogonal up!

            if(no_stack_info):
                print("NO STACK INFORMATION!!!")
                ALL_STACKS[0, sl_shape:sl_shape*2] = STACK1
                ALL_STACKS[0, sl_shape*2:sl_shape*3] = STACK1
            
            # save slice height
            ALL_STACKS[1,   0:sl_shape] =   STACK1
            ALL_STACKS[1, sl_shape:sl_shape*2] = STACK1
            ALL_STACKS[1, sl_shape*2:sl_shape*3] = STACK1

            actually_crop = True
            remove_repeat = True

            print("ACTUALLY CROP?")
            print(actually_crop)
            print("REMOVE REPEAT?")
            print(remove_repeat)
            
            if (actually_crop and remove_repeat==False):
                keep = img0[1:].reshape(img0[1:].shape[1], -1).any(dim=1)
                img0 = img0[:,keep]   
               # ALL_STACKS = ALL_STACKS[:,keep[::2]] # 
                # when run
                ALL_STACKS = ALL_STACKS[:,keep]

        
                flow = flow[:,keep]
            
            if (actually_crop and remove_repeat==True):
                
                
                rep = int(self.spacing / self.subsample)# number of times slice is repeated

                # remove duplicates
                img0 = img0[:,::rep]
                print("CHANGED")
              #  flow = flow[:,::rep]
                flow = flow[:,::2]
                ALL_STACKS = ALL_STACKS[:,::rep]

                # find all the slices that have non-zero masks
                keep = img0[1:].reshape(img0[1:].shape[1], -1).any(dim=1)

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
                pdb.set_trace()

                flow = flow[:,keep]

            # 
            # if(ONE_LAYER_ONLY):
            #  #   flow = flow[:,:,::16,::16]*(1/16)
            #     flow = flow[:,:,:,:] #*(1/16)
                
            return (img0,ALL_STACKS), flow
        
        

 

class RandAffine3dSlice2_no_gpu:
    def __init__(self, spacing=1, translations=0, rotations=0, bulk_translations=0, bulk_rotations=0, zooms=0, subsample=1, slice=1, nodes=(8,16), shots=2, augment=True, noise=False, X=3):
        self.slice = slice if isinstance(slice, (tuple, list)) else [slice]
        # self.crop = cc.fov.PatchTransform(192)
        self.flip = cc.fov.RandomFlipTransform(axes=[-3]) #+ cc.fov.PatchTransform(192)
        self.zoom = cc.MaybeTransform(cc.RandomAffineTransform(translations=0, rotations=0, shears=0, zooms=zooms, iso=True), 0.9)
        self.base = [cc.RandomSlicewiseAffineTransform(nodes=nodes, shots=shots, spacing=spacing, subsample=subsample, slice=s, translations=translations, rotations=rotations, 
                                                       bulk_translations=bulk_translations, bulk_rotations=bulk_rotations, shears=0, zooms=0) for s in self.slice]
        self.mult = cc.MaybeTransform(cc.RandomGaussianNoiseTransform(sigma=0.01), 0.5) \
            # + cc.MaybeTransform(cc.RandomGammaTransform(value=(0.7, 1.5)), 0.5) + cc.MaybeTransform(cc.RandomMultFieldTransform(), 0.5)
        self.bound = BoundingBox3d(spacing=spacing, subsample=subsample)
        self.augment = augment #add_noise
        self.noise = noise
        self.X = X
        print("in init")
    def __call__(self, img1, seg1):
        #print("CALLED")
        if img1.ndim == 5:
            print("in 5 dim loop")
            img1, seg1 = zip(*[self(img1[i], seg1[i]) for i in range(img1.shape[0])])
            return torch.stack(img1, 0), torch.stack(seg1, 0)

      #  print(self.augment)
        numstacks = 1 #torch.randint(1, len(self.slice) + 1, [1]).item()
        img1 = (img1.clamp(min=0.1) - 0.1) * (1 / 0.9)
       # seg1 = ((img1 > 0) | (seg1 > 0)).float()
        seg1 = ((img1 > 0) | (seg1 > 0)).double()
        

        img0 = torch.cat([ torch.as_tensor(img1).float(), torch.as_tensor(seg1).float()], axis=0)
       # pdb.set_trace()
       # img0 = torch.cat([ torch.as_tensor(img1), torch.as_tensor(seg1)], axis=0)
        flow_3 = torch.zeros((3, 128, 128, 128)) #.cuda()

        flow = torch.cat([ flow_3,img0[1,:,:,:][None]], axis=0)
       # pdb.set_trace()
       # print(f"Final shape of img1 {img0.shape}")
        return img0, flow

        # # img1, seg1 = self.crop(img1, seg1)
        # img1, seg1 = self.flip(img1, seg1) if self.augment else (img1, seg1)
        # xform = self.zoom.make_final(img1)
        # img1, seg1 = xform(img1, seg1) if self.augment else (img1, seg1)
        # pdb.set_trace()

        # xform = [self.base[i].make_final(img1) for i in range(numstacks)]
        # img0 = torch.cat([xform[i](img1) for i in range(numstacks)], 1)
        # seg0 = torch.cat([xform[i](seg1).gt(0).float() for i in range(numstacks)], 1)
        # # img0 = self.mult(img0) if self.noise else img0
        # flow = torch.cat([xform[i].flow.flip(0) for i in range(numstacks)], 1) # 
        # # img0, flow = self.bound(torch.cat([img0, seg0]), torch.cat([flow, seg0]), mask=seg0)
        # img0, flow = torch.cat([img0, seg0]), torch.cat([flow, seg0])
        # print("finish")
        # img0 = img0.float()
        # flow = flow.float()
        
class BoundingBox3d_3stacks:
    def __init__(self, spacing=1, subsample=1, crop_size = [32,32,32]): #[30,30,30]
        self.spacing = spacing / subsample
        self.crop_size = [int(x // spacing) for x in crop_size] 
       

    def __call__(self, img1, seg1, mask):
        # ss = img1.shape[1]//3
        # new_img = img1.clone()
        # new_seg0 = seg0.clone()
        # stack0 = img1[:,0:ss,:,:]
        # stack1 = img1[:,ss:ss*2,:,:]
        # stack2 = img1[:,ss*2:ss*3,:,:]

        # sstack0 = seg0[:,0:ss,:,:]
        # sstack1 = seg0[:,ss:ss*2,:,:]
        # sstack2 = seg0[:,ss*2:ss*3,:,:]
    


        nonz = mask.nonzero()
        
        print(8 * self.spacing)
        border = 8 * self.spacing
       # border = 8
       # border = 2

        

        print("border size")
        print(border)
        mins = [(ind - border).div(self.spacing).int().mul(self.spacing).int() for ind in nonz.min(0).values[-3:]]
        mins = torch.tensor(mins).clamp(torch.tensor([0,0,0]), torch.tensor(img1.shape[-3:]))

        maxs = [(ind + border).div(self.spacing).int().mul(self.spacing).int() for ind in nonz.max(0).values[-3:]]
        maxs = torch.tensor(maxs).clamp(torch.tensor([0,0,0]), torch.tensor(img1.shape[-3:]))

        max_s = img1.shape[2]
        dif_xy = (maxs-mins)[2] - (maxs-mins)[1]
        

        # print(dif_xy)
        if(dif_xy>0): # dimension 2 larger
            mins[1] = mins[1] - int(dif_xy//2)
            maxs[1] = maxs[1] + int(dif_xy//2)
            #if(mins[1]%2==1):
            #     mins[1] = mins[1] + 1
            # if(maxs[1]%2==1):
            #     maxs[1] = maxs[1] - 1
        else:
            mins[2] = mins[2] + int(dif_xy//2)
            maxs[2] = maxs[2] - int(dif_xy//2)
            # if(mins[2]%2==1):
            #     mins[2] = mins[2] + 1
            # if(maxs[2]%2==1):
            #     maxs[2] = maxs[2] - 1
        # print(mins, maxs)
        # if((maxs[2]-mins[2])%2==1):
        #     mins[2] = mins[2] - 1
        # if((maxs[1]-mins[1])%2==1):
        #     mins[1] = mins[1] - 1
        
        if (maxs[2]>max_s):
            over_step = max_s - maxs[2]
            maxs[2] = max_s
            mins[2] = mins[2] - over_step
        if (maxs[1]>max_s):
            over_step = max_s - maxs[1]
            maxs[1] = max_s
            mins[1] = mins[1] - over_step
        if (mins[1]<0):
            over_step = -mins[1]
            mins[1] = 0
            maxs[1] = maxs[1] + over_step
        if (mins[2]<0):
            over_step = -mins[2]
            mins[2] = 0
            maxs[2] = maxs[2] + over_step



        # print("clipped")
        # print(mins, maxs)
        if(mins[2]==0 and maxs[2]==max_s):
            mins[1] = 0
            maxs[1] = max_s
        if(mins[1]==0 and maxs[1]==max_s):
            mins[2] = 0
            maxs[2] = max_s
        
        dif_xy = (maxs-mins)[2] - (maxs-mins)[1]
        if(dif_xy!=0):
            mins[1] = 0
            mins[2] = 0
            maxs[1] = max_s
            maxs[2] = max_s

            
        print("clipped")
        print(mins, maxs)
        img1 = img1[..., :, mins[1]:maxs[1], mins[2]:maxs[2]]
        seg1 = seg1[..., :, mins[1]:maxs[1], mins[2]:maxs[2]]


       # pdb.set_trace()

      #  img1 = img1[..., mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]
      #  seg1 = seg1[..., mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]


       # print("IN BOUNDING BOX")
       # print("og shape")
       # print(img1.shape)
     #  pdb.set_trace()
       # img1 = img1[..., mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]
       # seg1 = seg1[..., mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]

        
        return img1, seg1   

class RandAffine3dSliceSplat(RandAffine3dSlice):
    def __init__(self, spacing=4, subsample=1, slice=1, nodes=None, size=None):
        super().__init__(spacing=spacing, subsample=subsample, slice=slice)
        self.mult = cc.MaybeTransform(cc.RandomMultFieldTransform(), 0.5) #making this bigger makes transformations smaller
        self.crop = cc.fov.RandomPatchTransform(patch_size=size)
        self.base = cc.RandomSlicewiseAffineTransform(nodes=nodes, spacing=spacing, subsample=subsample, slice=slice, shears=0, zooms=0)

    def __call__(self, img1, seg1):
        if img1.ndim == 5:
            img1, seg1 = zip(*[self(img1[i], seg1[i]) for i in range(img1.shape[0])])
            return torch.stack(img1, 0), torch.stack(seg1, 0)

        xform = self.base.get_parameters(img1)
        matrix, flow, warp = xform.get_parameters(img1)

        img0 = xform.forward_with_parameters(img1, (matrix, flow, warp))
        img0 = self.mult(img0)

        img0 = interpol.grid_push(img0, warp, bound=1, extrapolate=True)
        ones = interpol.grid_count(warp, bound=1, extrapolate=True)[None] + 1e-4
        img0, img1, ones = self.crop(img0, img1, ones)

        return torch.cat([img0 / ones], 0), img1 # - img0



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


class CompactLabels(object):
    
    def __init__(self, oldlabels, direction=+1):
        self.oldlabels = oldlabels
        self.newlabels = list(range(len(oldlabels)))
        self.direction = direction

    def __call__(self, img ,seg):
        if seg == None:
            return None
        if self.direction == +1:
            seg[seg < self.oldlabels[ 0]] = self.oldlabels[ 0]
            seg[seg > self.oldlabels[-1]] = self.oldlabels[-1]
            for l in range(len(self.oldlabels)):
                seg[seg == self.oldlabels[l]] = self.newlabels[l]

        if self.direction == -1:
            seg[seg < self.newlabels[ 0]] = self.newlabels[ 0]
            seg[seg > self.newlabels[-1]] = self.newlabels[-1]
            for l in reversed(range(len(self.newlabels))):
                seg[seg == self.newlabels[l]] = self.oldlabels[l]

        return img, seg

class ReplaceLabels(object):
    def __init__(self, newlabels, gpu=True):
        self.newlabels = newlabels
        self.gpu = gpu

    def __call__(self, img ,seg):
        if seg == None:
            img, seg = img, None

        for l in range(len(self.newlabels)):
            seg[seg == l] = self.newlabels[l]
        seg[seg >= len(self.newlabels)] = 0 #self.newlabels[0]

        return img, seg


class GaussNoise(torch.nn.Module):
    def __init__(self, std=(0,0.1), gpu=True):
        super().__init__()
        self.std = std
        self.gpu = gpu
        
    def forward(self, img, seg):
        std = torch.empty(1).uniform_(self.std[0], self.std[1]).item()
        img = img + std * torch.randn(img.shape, device=img.device)

        return img, seg

class Subsample3d(object):
    def __init__(self, factors=[2,2,2]):
        super().__init__()
        self.factors = factors

    def __call__(self, img, seg):
        if img.ndim == 5:
            img, seg = zip(*[self(img[i], seg[i]) for i in range(img.shape[0])])
            return torch.stack(img, 0), torch.stack(seg, 0)

        img, seg = img[:,::self.factors[0],::self.factors[1],::self.factors[2]], \
            seg[:,::self.factors[0],::self.factors[1],::self.factors[2]]
        
        return img, seg

class RandSkullNoise3d(object):
    def __init__(self, shape, depth):
        self.shape = shape
        self.depth = depth
    
    def __call__(self, img, seg):
        depth = 2 * torch.randint(self.depth,[1]).item() + 1
        strel = torch.ones(1,1,depth,1,1)

        mask = (seg[None].transpose(0,1) != 0).float()
        for d in range(2, 5):
            mask = nn.functional.conv3d(mask, strel.transpose(2, d), padding='same')
        mask = ((mask != 0) * (seg[None].transpose(0,1) == 0)).float()

        noise = mask.transpose(0,1)[0] * nn.functional.interpolate(torch.randn([1, 2] + self.shape), img[0].shape,\
                                                                   mode='trilinear', align_corners=True)[0]
        return img + noise, seg

class RandomNoise(object):
    def __init__(self, std, num_classes, downscale=1):
        self.std = std
        self.num_classes = num_classes
        self.downscale = downscale

    def __call__(self, image, target):
        image = image.clamp(max=self.num_classes)
        shape = torch.as_tensor(image.shape) // torch.as_tensor([self.downscale, self.downscale])
        noise = 255 * torch.ones(list(shape), dtype=torch.long)
        # max_n = self.num_classes + 1 #image.max() + 1
        randp = (torch.rand(list(shape)) < self.std) #should flip?
        randv = (self.num_classes * torch.rand(list(shape))).floor()
        coord = [0,0]
        
        for i in range(10): #take 5 random steps
            magnitude = math.ceil(torch.rand(1) / 0.5)
            direction = 1 if torch.rand(1) < 0.5 else -1
            dimension = 1 if torch.rand(1) < 0.5 else 0
            coord[dimension] += magnitude * direction
            index = randp.roll(coord[0],0).roll(coord[1],1)
            noise[index] = randv.roll(coord[0],0).roll(coord[1],1)[index].long()

        noise = F.resize(noise[None], (image.shape[0], image.shape[1]), Image.NEAREST)[0]
        image[noise != 255] = noise[noise != 255]

        image = torch.nn.functional.one_hot(image, num_classes=self.num_classes).permute(2,0,1).float()

        return image[:self.num_classes], target

class RandomNoise3d(object):
    def __init__(self, std=(2,8), scales=(1,3), **kwargs):
        self.std = std
        self.scales = scales
        self.avgfilter = torch.nn.AvgPool3d(3, stride=1, padding=1)

    def __call__(self, image, target):
        std = torch.empty(1).uniform_(self.std[0], self.std[1]).item()
        # scale = 2 ** torch.randint(0,2, (1,)).item()
        scale = torch.randint(self.scales[0], self.scales[1], (1,)).item()
        scale = torch.tensor([1, 1, scale, scale, scale])
        shape = ((torch.as_tensor(image[None].shape) / scale).int()).tolist()
        noise = self.avgfilter((torch.randn(shape)))
        noise = nn.functional.interpolate(noise, image[0].shape, mode='trilinear', align_corners=False)
        image = ((image.log() + 3).clamp(min=0) + std * noise[0]).softmax(0)

        return image, target

class RandomNoise2d(object):
    def __init__(self, std=(2,8), scales=(1,5), **kwargs):
    # def __init__(self, std=(2,2), scales=(3,4), **kwargs):
        self.std = std
        self.scales = scales
        self.avgfilter = torch.nn.AvgPool2d(3, stride=1, padding=1)

    def __call__(self, image, target):
        std = torch.empty(1).uniform_(self.std[0], self.std[1]).item()
        scale = torch.randint(self.scales[0], self.scales[1], (1,)).item()
        scale = torch.tensor([1, 1, scale, scale])
        shape = ((torch.as_tensor(image[None].shape) / scale).int()).tolist()
        noise = self.avgfilter((torch.randn(shape)))
        noise = nn.functional.interpolate(noise, image[0].shape, mode='bilinear', align_corners=False)
        image = ((image.log() + 3).clamp(min=0) + std * noise[0]).softmax(0)

        return image, target

class LabelToImage(object):
    def __call__(self, image, target):

        return target, target

class ClampPercentile:
    def __init__(self, percentile=0.02):
        self.percentile = percentile

    def __call__(self, img, seg):
        minvals = torch.kthvalue(img.flatten(1),round(img.flatten(1).shape[1]*(self.percentile-0)),dim=1)[0][:,None,None]
        maxvals = torch.kthvalue(img.flatten(1),round(img.flatten(1).shape[1]*(1-self.percentile)),dim=1)[0][:,None,None]
        
        return torch.min(torch.max(img,minvals),maxvals), seg
    

class Normalize():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, seg):
        mean = torch.as_tensor(self.mean).reshape([-1] + [1] * (img.ndim - 1))
        std = 1 / torch.as_tensor(self.std).reshape([-1] + [1] * (img.ndim - 1))

        return std * (img - mean), seg

class CatImages():
    def __call__(self, img0, img1, flo):
        return torch.cat([img0, img1], 0), flo

class NormalizeFlow():
    def __call__(self, img, flo, mul=1e+2):
        flo = torch.stack([flo[i] / flo.shape[flo.ndim - 1 - i] for i in range(flo.shape[0])], 0)

        return img, mul * flo

class ToFloTensor(transforms.ToTensor):
    def __call__(self, img0, img1, flo, mask=None, *args):
        img0 = F.to_tensor(img0)
        img1 = F.to_tensor(img1)
        mask = torch.empty([1] + list(img0.shape[1:])) if mask is None else mask
        flo = torch.as_tensor(flo)

        return torch.cat([img0, img1], 0), flo #torch.cat([flo, mask], 0)

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

class ToImagePair(transforms.ToTensor):
    def __call__(self, img, seg):

        return torch.cat([img, img], 0), torch.cat([seg, seg], 0)

class Window():
    def __init__(self, minval=0, maxval=None):
        super().__init__()
        self.minval = minval
        self.maxval = maxval

    def __call__(self, img, seg):
        img[img < self.minval] = self.minval
        img[img > self.maxval] = self.minval
        
        return img, seg


class ClampMin():
    def __init__(self, minval=0, maxval=None):
        super().__init__()
        self.minval = minval
        self.maxval = maxval

    def __call__(self, img, seg):
        img[img < self.minval] = self.minval
        img[img > self.maxval] = self.maxval
        
        return img, seg

class MultiplicativeNoise():
    def __init__(self, sig_bias_max=0.5, size=(4,4,4), mode='trilinear', gpu=True):
        self.sig_bias_max = sig_bias_max
        self.size = size
        self.mode = mode
        self.gpu = gpu

    def __call__(self, img, seg):
        sig_bias = torch.empty(1).uniform_(to=self.sig_bias_max).item()
        bias = torch.empty(self.size, device=img.device).normal_(std=sig_bias) #torch.randn(size, 
        bias = nn.functional.interpolate(bias[None, None], img[0].shape, mode=self.mode, align_corners=False)[0]
        img[0] = img[0] * torch.exp(bias)

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

class ScaleBrightness():
    def __init__(self, sig_gamma_sq=0.4):
        self.sig_gamma_sq = sig_gamma_sq

    def __call__(self, img, seg):
        self.sig_gamma_sq=0.4
        gamma = torch.empty(1).normal_(std=math.sqrt(self.sig_gamma_sq)).item() if self.sig_gamma_sq > 0 else 0
        img[0] = ((img[0] - img[0].min()) / (img[0].max() - img[0].min())) ** math.exp(gamma)

        return img, seg

class FlipBrightness():
    def __call__(self, img, seg):
        if torch.rand(1) < 0.5:
            img[0] = 1 - img[0]

        return img, seg

class Positional2d():
    def __init__(self, order):
        self.order = order

    def __call__(self, img, seg):
        if self.order > 0:
            _, x__, y__ = torch.meshgrid(torch.arange( 0, 1, dtype=torch.float),
                                         torch.arange(-1, 1, 2/img.shape[-2]), 
                                         torch.arange(-1, 1, 2/img.shape[-1]))
            img = [img]
            for f in range(0, self.order):
                s = math.pi * (2 ** f)
                img = img + [torch.sin(s*x__), torch.cos(s*x__), torch.sin(s*y__), torch.cos(s*y__)]
            img = torch.cat(img, 0)

        return img, seg

class Positional3d():
    def __init__(self, order):
        self.order = order

    def __call__(self, img, seg):
        if self.order > 0:
            _, dx, dy, dz = torch.meshgrid(torch.arange(0, 1, dtype=torch.float),\
                                           2 * torch.arange(0, img.shape[-3]) / (img.shape[-3] - 1) - 1,\
                                           2 * torch.arange(0, img.shape[-2]) / (img.shape[-2] - 1) - 1,\
                                           2 * torch.arange(0, img.shape[-1]) / (img.shape[-1] - 1) - 1)
            return torch.cat([img, dx, dy, dz]), seg

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

class Pad3d():
    def __init__(self, margin=[32,64,64], multiple=[16,32,32], channels=[0], crop=True):
        self.margin = torch.as_tensor(margin)
        self.multiple = torch.as_tensor(multiple)
        self.channels = channels
        self.crop = crop

    def __call__(self, img, seg):
        bbox_lb = torch.as_tensor([0, 0, 0])
        bbox_ub = torch.as_tensor(seg.shape[1:])

        if self.crop:
            bbox_lb, bbox_ub = generate_spatial_bounding_box(np.asarray(seg[self.channels]))
            bbox_lb = torch.as_tensor(bbox_lb)
            bbox_ub = torch.as_tensor(bbox_ub)

        need_to_pad = self.margin - ((bbox_ub - bbox_lb) % self.multiple)

        lb = (need_to_pad / 2.).int()
        ub = (need_to_pad - lb).int() #(box_ub - box_lb) + need_to_pad - lb - self.patch_size

        img_shape = torch.as_tensor(seg.shape[1:])
        img_zeros = 0 * img_shape

        cbox_lb = torch.clamp(bbox_lb - lb, min=img_zeros, max=None)
        cbox_ub = torch.clamp(bbox_ub + ub, min=None, max=img_shape)
        
        pad_lb = torch.clamp(bbox_lb - lb, min=None, max=img_zeros)
        pad_ub = torch.clamp(bbox_ub + ub, min=img_shape, max=None) - img_shape

        img = img[:, cbox_lb[0]:cbox_ub[0], cbox_lb[1]:cbox_ub[1], cbox_lb[2]:cbox_ub[2]]
        seg = seg[:, cbox_lb[0]:cbox_ub[0], cbox_lb[1]:cbox_ub[1], cbox_lb[2]:cbox_ub[2]]

        img = torch.nn.functional.pad(img, (-pad_lb[2], pad_ub[2], -pad_lb[1], pad_ub[1], -pad_lb[0], pad_ub[0]))
        seg = torch.nn.functional.pad(seg, (-pad_lb[2], pad_ub[2], -pad_lb[1], pad_ub[1], -pad_lb[0], pad_ub[0]))

        return img, seg

class Crop3d(torch.nn.Module):
    def __init__(self, margin=0, multiple=32, random=False):
        super().__init__()
        self.margin = margin
        self.multiple = multiple
        self.random = random

    def forward(self, img, seg):
        a, b = generate_spatial_bounding_box(np.asarray(seg), margin=self.margin)

        a = np.array(a)
        b = np.array(b)
        r = (b - a) % self.multiple
        c = np.random.randint(r + 1) if self.random else r // 2
        d = r - c
        a = a + c
        b = b - d

        img = img[:,a[0]:b[0],a[1]:b[1],a[2]:b[2]]
        seg = seg[:,a[0]:b[0],a[1]:b[1],a[2]:b[2]]

        return img, seg

class Crop2d(torch.nn.Module):
    def __init__(self, margin=0):
        super().__init__()
        self.margin = margin

    def forward(self, img, seg):
        a, b = generate_spatial_bounding_box(np.asarray(seg), margin=self.margin)

        img = img[:,a[0]:b[0],a[1]:b[1]]#,a[2]:b[2]]
        seg = seg[:,a[0]:b[0],a[1]:b[1]]#,a[2]:b[2]]

        return img, seg

class RandomHorizontalFlip(transforms.RandomHorizontalFlip):

    def __init__(self, p=0.5):
        super().__init__()

    def forward(self, img, seg):

        if torch.rand(1) < self.p:
            return F.hflip(img), F.hflip(seg)

        return img, seg

class RandomFlipIntensity(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img, seg):
        if torch.rand(1) < self.p:
            img = 1 - img
        
        return img, seg

class RandomFlip3d(nn.Module):
    def __init__(self, p=0.5, dim=0):
        super().__init__()
        self.p = p
        self.dim = dim if isinstance(dim, list) else [dim]

    def forward(self, img, seg):
        for dim in self.dim:
            if torch.rand(1) < self.p:
                img, seg = img.flip(dim + 1), seg.flip(dim + 1)

        return img, seg

class Resize(transforms.Resize):

    def __init__(self, size, interpolation=Image.LANCZOS):
        super().__init__(size, interpolation)

    def forward(self, img, seg):
        return F.resize(img, self.size, self.interpolation), F.resize(seg, self.size, Image.NEAREST)
    

class RandomResize(transforms.Resize):

    def __init__(self, scale=(0.5, 2.0, 0.25), interpolation=2):
        self.scale = self.size = np.arange(scale[0],scale[1]+scale[2],scale[2])
        self.interpolation = interpolation

    def get_params(self, img):
        scale = self.scale[random.randint(0,len(self.scale)-1)]
        width, height = F._get_image_size(img)
        return (int(round(scale*height)), int(round(scale*width)))

    def __call__(self, img, seg):
        size = self.get_params(img)
        return F.resize(img, size, self.interpolation), F.resize(seg, size, 0)

class RandomPaddedCrop(transforms.RandomCrop):

    def __init__(self, size, padding=None, pad_if_needed=False, fill=255, padding_mode=["replicate", "circular"],\
                 segment_mode=["constant", "constant"]):
        super().__init__(size, padding, pad_if_needed, fill, "constant")
        self.padding_mode = padding_mode
        self.segment_mode = segment_mode
        
    def __call__(self, img, seg):
        if self.padding is not None:
            padding = [self.padding[1], self.padding[1], 0, 0]
            img = torch.nn.functional.pad(img[None], padding, self.padding_mode[1])[0]
            seg = torch.nn.functional.pad(seg[None], padding, self.segment_mode[1], **{'value': self.fill})[0]

            padding = [0, 0, self.padding[0], self.padding[0]]
            img = torch.nn.functional.pad(img[None], padding, self.padding_mode[0])[0]
            seg = torch.nn.functional.pad(seg[None], padding, self.segment_mode[0], **{'value': self.fill})[0]

        width, height = F._get_image_size(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0, 0, 0]
            img = torch.nn.functional.pad(img[None], padding, self.padding_mode[1])[0]
            seg = torch.nn.functional.pad(seg[None], padding, self.segment_mode[1], **{'value': self.fill})[0]
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, 0, 0, self.size[0] - height]
            img = torch.nn.functional.pad(img[None], padding, self.padding_mode[0])[0]
            seg = torch.nn.functional.pad(seg[None], padding, self.segment_mode[0], **{'value': self.fill})[0]

        i, j, h, w = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w), F.crop(seg, i, j, h, w)

class RandomResizedCrop(transforms.RandomResizedCrop):

    def __init__(self, size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2):
        super().__init__(size, scale, ratio, interpolation)

    def forward(self, img, seg):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)

        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation), F.resized_crop(seg, i, j, h, w, self.size, 0)

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
        width, height = F._get_image_size(img)
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            side = torch.arange(1,4.2,0.2)[torch.randint(0,16,(1,))].item() #torch.empty(1).uniform_(1./math.sqrt(scale[1]), 1./math.sqrt(scale[0])).item()
            target_area = area / (side ** 2)
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

class Resize3d(transforms.Resize):

    def __init__(self, size, interpolation=1):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, seg):
        return img[::2,::2,::2], seg[::2,::2,::2]

class CenterCrop(transforms.CenterCrop):
    def __init__(self, size):
        super().__init__(size)

    def forward(self, img, seg):
        return F.center_crop(img, self.size), F.center_crop(seg, self.size)

class RandomCrop(transforms.RandomCrop):

    def __init__(self, size, stride=[1,1,1], padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__(size, padding, pad_if_needed, fill, padding_mode)
        
    def __call__(self, img, seg):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)
            seg = F.pad(seg, self.padding, self.fill, "constant")

        width, height = F._get_image_size(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
            seg = F.pad(seg, padding, self.fill, "constant")
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = F.pad(img, padding, self.fill, self.padding_mode)
            seg = F.pad(seg, padding, self.fill, "constant")

        i, j, h, w = self.get_params(img, self.size)


        return F.crop(img, i, j, h, w), F.crop(seg, i, j, h, w)