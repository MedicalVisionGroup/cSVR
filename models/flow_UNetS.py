import interpol
import math
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import pdb
from cornucopia.utils.warps import affine_flow

from .grid_utils import make_grid, og_slice_pos, make_grid_one
#from cSVR.utils.grid_utils import make_grid, og_slice_pos, make_grid_one
import sys
import os


__all__ = ['Flow_UNet', 'flow_UNet2d', 'flow_UNet2d_postwarp', 'flow_UNet2d_nowarp', 'flow_UNet3d', 'flow_UNet3d_nowarp', 'flow_UNet3d_postwarp']

class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            if module.weight.requires_grad:
                module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None and module.bias.requires_grad:
                module.bias = nn.init.constant_(module.bias, 0)

class Pool4MLP_attention(nn.Module):
    def __init__(self, pool_feature, pool_slices):
        super().__init__()
        self.pool_feature = pool_feature

        # Feature pooling (unchanged)
        self.pool_feat = nn.AdaptiveAvgPool1d(pool_feature)

        # Slice attention: shared across all N
        self.slice_score = nn.Linear(pool_feature, 1)

    def forward(self, x):
        # x: (B, C, N, H, W)
        B, C, N, H, W = x.shape
        print("X original shape:", x.shape)

        # ---------- FEATURE POOLING ----------
        x = x.permute(0, 2, 3, 4, 1)             # (B, N, H, W, C)
        x = x.reshape(-1, C).unsqueeze(1)        # (B*N*H*W, 1, C)
        x = self.pool_feat(x)                    # (B*N*H*W, 1, pool_feature)
        x = x.squeeze(1).reshape(B, N, H, W, self.pool_feature)
        x = x.permute(0, 4, 1, 2, 3)             # (B, pool_feature, N, H, W)

        # ---------- SLICE ATTENTION (VARIABLE N) ----------
        # Aggregate spatial info per slice
        x_slice = x.mean(dim=(3, 4))             # (B, pool_feature, N)
        x_slice = x_slice.permute(0, 2, 1)       # (B, N, pool_feature)


        # V1
        # # Compute attention weights per slice
        # scores = self.slice_score(x_slice)       # (B, N, 1)
        # weights = torch.softmax(scores, dim=1)   # normalize over slices

        # # Weighted sum over slices torch.Size([1, 3, 8, 8])
        # x = (x * weights.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)).sum(dim=2)
        # # x: (B, pool_feature, H, W)

        scores = self.slice_score(x_slice)       # (B, N, 1)
        weights = torch.softmax(scores, dim=1)   # normalize over slices
        k = 8

        
        # get top-k slice indices per batch
        topk_vals, topk_idx = torch.topk(weights.squeeze(-1), k=k, dim=1)  # [B, k]
        
        # gather corresponding slices
        x_topk = torch.gather(
            x,
            dim=2,
            index=topk_idx[:, None, :, None, None].expand(
                -1, x.size(1), -1, x.size(3), x.size(4)
            )
        )  # [B, C, k, H, W]
        
        # apply corresponding weights
        w_topk = topk_vals[:, None, :, None, None]  # [B, 1, k, 1, 1]
        x = (x_topk * w_topk) #.sum(dim=2)             # [B, C, H, W]
        
        return x
    
class Pool4MLP(nn.Module):
    def __init__(self, pool_feature, pool_slices):
        super().__init__()
        self.pool_feature = pool_feature
        self.pool_slices = pool_slices
        self.pool_feat = nn.AdaptiveAvgPool1d(pool_feature)
        self.pool_sl = nn.AdaptiveAvgPool1d(pool_slices)
    
    def forward(self, x):
        # x: (B, C, N, H, W)
        B, C, N, H, W = x.shape
        
        # Step 1: move C to last
        x = x.permute(0, 2, 3, 4, 1)             # (B, N, H, W, C)
        
        # Step 2: flatten everything except C for pooling
        x = x.reshape(-1, C).unsqueeze(1)        # (B*N*H*W, 1, C)
        
        # Step 3: pool C → pool_feature
        x = self.pool_feat(x)                         # (B*N*H*W, 1, pool_feature)
        
        # Step 4: reshape back
        x = x.squeeze(1).reshape(B, N, H, W, self.pool_feature)
        
        # Step 5: move pooled dim to channel position
        x = x.permute(0, 4, 1, 2, 3)             # (B, pool_feature, N, H, W)
   

        
        B, C, N, H, W = x.shape
        # Step 1: bring N to the last dimension
        x = x.permute(0, 1, 3, 4, 2)         # (B, C, H, W, N)


        # Step 2: flatten the rest for 1D pooling
        x = x.reshape(-1, N).unsqueeze(1)    # (B*C*H*W, 1, N)

        # Step 3: apply 1D adaptive pooling
        x = self.pool_sl(x)                          # (B*C*H*W, 1, 10)

        # Step 4: reshape back
        x = x.squeeze(1).reshape(B, C, H, W, self.pool_slices)  # (B, C, H, W, 10)


        # Step 5: move pooled dimension back to channel ordering
        x = x.permute(0, 1, 4, 2, 3)         # (B, C, 10, H, W)

        return x
    
class MLP(nn.Module):
    def __init__(self, hidden_sizes=[128, 64],  in_channel_features = 3, in_slice_features = 8, out_features=3, plane_features=8):
        """
        hidden_sizes: list of hidden layer sizes
        out_features: dimension of final output
        """
        super().__init__()

        in_features = in_channel_features * in_slice_features * plane_features * plane_features  # fixed input: (3,8,8,8)

        # Build a list of Linear layers
        layers = []
        prev = in_features

        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            prev = h

        # Output layer
        layers.append(nn.Linear(prev, out_features))

        # Register as a ModuleList
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        # flatten input
        x = x.reshape(x.size(0), -1)

        # apply all layers except last with ReLU
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))

        # last layer without activation
        x = self.layers[-1](x)

        return x
    
class MLP_norm(nn.Module):
    def __init__(
        self,
        hidden_sizes=[128, 64],
        in_channel_features=3,
        in_slice_features=8,
        out_features=3,
        plane_features=8,
        dropout=0.05,
    ):
        super().__init__()

        # Flattened input features
        in_features = in_channel_features * in_slice_features * plane_features * plane_features

        layers = []

        # Optional input LayerNorm
        layers.append(nn.LayerNorm(in_features))

        prev = in_features
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LayerNorm(h))      # stable activations
            layers.append(nn.GELU())            # smoother than ReLU
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            prev = h

        # Output layer (no activation, no dropout)
        layers.append(nn.Linear(prev, out_features))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # Flatten input
        x = x.reshape(x.size(0), -1)
        return self.net(x)
    
class MLP_drop_out(nn.Module):
    def __init__(
        self,
        hidden_sizes=[128, 64],
        in_channel_features=3,
        in_slice_features=8,
        out_features=3,
        plane_features=8,
        dropout=0,
    ):
        super().__init__()

        in_features = (
            in_channel_features
            * in_slice_features
            * plane_features
            * plane_features
        )

        layers = []
        prev = in_features

        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout))
            prev = h

        # Output layer (no dropout, no activation)
        layers.append(nn.Linear(prev, out_features))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        return self.net(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=True, norm=True, drop=0.0, relu=True, X=3):
        super().__init__()

        #3D if not specified
        kernel_size = [kernel_size] * X if not isinstance(kernel_size, (list, tuple)) else kernel_size
        stride = [stride] * X if not isinstance(stride, (list, tuple)) else stride
        padding = [(kernel_size[i] - 1) // 2 if stride[i] == 1 else 0 for i in range(len(kernel_size))]
        
        #This set of lines sets convolutional operations to true depending on input parameters
        # EX: Instance norm if norm is True otherwise nn.Identity
        self.norm = eval("nn.InstanceNorm%dd" % X)(out_channels, affine=True) if norm else nn.Identity()
        self.relu = eval("nn.LeakyReLU")() if relu else nn.Identity()
        self.conv = eval("nn.Conv%dd" % X)(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias)
        self.drop = eval("nn.Dropout%dd" % X)(drop) if drop else nn.Identity()

    def forward(self, x):
        # return self.drop(self.conv(self.relu(self.norm(x))))
        return self.relu(self.norm(self.drop(self.conv(x))))


# NEVER CALLED AND THE SAME??
class ConvTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=True, norm=True, drop=0.0, relu=True, X=3):
        super().__init__()
        kernel_size = [kernel_size] * X if not isinstance(kernel_size, (list, tuple)) else kernel_size
        stride = [stride] * X if not isinstance(stride, (list, tuple)) else stride
        padding = [(kernel_size[i] - 1) // 2 if stride[i] == 1 else 0 for i in range(len(kernel_size))]

        self.norm = eval("nn.InstanceNorm%dd" % X)(out_channels, affine=True) if norm else nn.Identity()
        self.relu = eval("nn.LeakyReLU")() if relu else nn.Identity()
        self.conv = eval("nn.ConvTranspose%dd" % X)(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias)
        self.drop = eval("nn.Dropout%dd" % X)(drop) if drop else nn.Identity()

    def forward(self, x):
        # return self.drop(self.conv(self.relu(self.norm(x))))
        return self.relu(self.norm(self.drop(self.conv(x))))


# makes sure stride is slab amount
class SlabbedConvLayers(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs, slab_kernel_sizes=3, conv_kernel_sizes=3,
                 slab_stride_sizes=1, conv_stride_sizes=1, pool_stride=1,  mask=False, bias=True, norm=True, drop=0.0, relu=True, X=3):
        super().__init__()
        self.input_channels = in_channels
        self.output_channels = out_channels + mask
        self.slab_stride_sizes = slab_stride_sizes if isinstance(slab_stride_sizes, (list, tuple)) else [slab_stride_sizes] * X
        self.pool_stride = pool_stride if isinstance(pool_stride, (list, tuple)) else [pool_stride] * X

        self.blocks = []
        for i in range(max(0, num_convs - 1)):
            stride = slab_stride_sizes if i == 0 else conv_stride_sizes
            kernel = slab_kernel_sizes if i == 0 else conv_kernel_sizes
            in_channels = self.input_channels #if i == 0 else self.output_channels
            out_channels = self.input_channels # self.spacing * self.output_channels if i == num_convs - 1 else self.input_channels
            self.blocks.append(ConvBlock(in_channels, out_channels, kernel_size=kernel, stride=stride, bias=bias, norm=norm, drop=drop, relu=relu, X=X))

        for i in range(max(0, num_convs - 1), num_convs):
            stride = slab_stride_sizes
            kernel = slab_kernel_sizes
            in_channels = self.input_channels # if i == 0 else self.output_channels
            out_channels = self.output_channels # if i == num_convs - 1 else self.input_channels
            self.blocks.append(ConvTransposeBlock(in_channels, out_channels, kernel_size=kernel, stride=stride, bias=bias, norm=norm, drop=drop, relu=False, X=X))

        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x):
        for i in range(len(self.blocks)):
            x = self.blocks[i](torch.cat([x], 1))

        return torch.cat([x[:,:3], x[:,3:].sigmoid()], 1)


class StackedConvLayers(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs, kernel_size=3, pool_stride=1, bias=True, norm=True, drop=0.0, relu=True, skip=False, X=3):
        super().__init__()
        self.mode = 'bilinear' if X == 2 else 'trilinear' if X == 3 else 'nearest'
        self.input_channels = in_channels
        self.output_channels = out_channels
        self.pool_stride = pool_stride if isinstance(pool_stride, (list, tuple)) else [pool_stride] * X

        self.blocks = []
        for i in range(num_convs):
            in_channels = self.input_channels if i == 0 else self.output_channels
            out_channels = self.output_channels
            kernel = kernel_size
            self.blocks.append(ConvBlock(in_channels, out_channels, kernel_size=kernel, stride=1, bias=bias, norm=norm, drop=drop, relu=relu, X=X))
        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x):
        size = [x.shape[d] // self.pool_stride[d - 2] for d in range(2, x.ndim)]
        for i in range(len(self.blocks)):
            x = F.interpolate(x, size=size, mode=self.mode, align_corners=True) if i == 0 else x
            x = self.blocks[i](torch.cat([x], 1))
        return x

class StackedConvTransposeLayers(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs, kernel_size=3, pool_stride=1, bias=True, norm=True, drop=0.0, relu=True, skip=True, X=3):
        super().__init__()
        self.mode = 'bilinear' if X == 2 else 'trilinear' if X == 3 else 'nearest'
        self.input_channels = in_channels
        self.output_channels = out_channels
        self.pool_stride = pool_stride if isinstance(pool_stride, (list, tuple)) else [pool_stride] * X

        self.blocks = []
        for i in range(num_convs):
            in_channels = 2 * self.input_channels if skip and i == 0 else self.input_channels
            out_channels = self.input_channels if i < num_convs - 1 else self.output_channels
            kernel = kernel_size
            self.blocks.append(ConvBlock(in_channels, out_channels, kernel_size=kernel, stride=1, bias=bias, norm=norm, drop=drop, relu=relu, X=X))
        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x, skip):
        size = [skip.shape[d] for d in range(2, skip.ndim)]
        for i in range(len(self.blocks)):
            x = F.interpolate(x, size=size, mode=self.mode, align_corners=True) if i == 0 else x
            x = self.blocks[i](torch.cat([x, skip], 1) if i == 0 else x)#, pad)
        return x #F.interpolate(x, size=size, mode=self.mode, align_corners=True)

class UpsampleLayer(nn.Module):
    def __init__(self, X=3):
        super().__init__()
        self.mode = 'linear' if X == 1 else 'bilinear' if X == 2 else 'trilinear' if X == 3 else 'nearest'

    def forward(self, x, shape):
        x = F.interpolate(x, size=shape, mode=self.mode, align_corners=True)

        return x

class WarpingLayer(nn.Module):
    def __init__(self, warp=True, X=3, transpose=False, normalize=True):
        super().__init__()
        self.mode = 'linear' if X == 1 else 'bilinear' if X == 2 else 'trilinear' if X == 3 else 'nearest'
        self.warp = warp
        self.transpose = transpose
        self.normalize = normalize
      #  print(self.mode)

    def forward(self, x, flow, mask=None, mode='bilinear', shape=None):

        flow = torch.stack([flow[:, d - 2] * ((x.shape[d] - 1)/(flow.shape[d] - 1)) for d in range(2, x.ndim)], 1)
        # interpolate to make flow same size as input
        flow = F.interpolate(flow, size=x.shape[2:], mode=self.mode, align_corners=True) if flow.shape[2:] != x.shape[2:] else flow
        grid = [torch.arange(0, x.shape[d], dtype=torch.float, device=x.device) for d in range(2, x.ndim)]
        grid = self.warp * flow + torch.stack(torch.meshgrid(grid, indexing='ij'), 0)
        shape = shape if shape is not None else list(x.shape[2:])

        if not self.transpose:
            grid = 2.0 / (torch.tensor(grid.shape[2:], device=x.device).reshape([1,-1] + [1] * (grid.ndim - 2)) - 1) * grid - 1.0
            grid = torch.permute(grid.flip(1), [0] + list(range(2, grid.ndim)) + [1])
            x1 = F.grid_sample(x, grid, mode=mode, padding_mode='border', align_corners=True)

        else:
            grid = torch.permute(grid, [0] + list(range(2, grid.ndim)) + [1]) * ((torch.tensor(shape, device=x.device) - 1)/(torch.tensor(grid.shape[2:], device=x.device) - 1))
           # pdb.set_trace()
            mask = F.interpolate(mask, size=x.shape[2:], mode=self.mode, align_corners=True) if mask.shape[2:] != x.shape[2:] else mask
            x1 = interpol.grid_push(torch.cat([x * mask, mask], 1), grid, shape=shape, bound=1, extrapolate=True) #
        return x1 #torch.cat([x0, x1], 0)
    
    def forward_th(self, x, flow, thickness = 2, mask=None, mode='bilinear', shape=None):
        if not self.transpose:
            thickness = 0.5


        flow = torch.stack([flow[:, d - 2] * ((x.shape[d] - 1)/(flow.shape[d] - 1)) for d in range(2, x.ndim)], 1)
        flow = F.interpolate(flow, size=x.shape[2:], mode=self.mode, align_corners=True) if flow.shape[2:] != x.shape[2:] else flow
      #  print("this!")
        # change grid to account for thickness
        grid_ndim = 5
        grid = [torch.arange(0, x.shape[d], dtype=torch.float, device=x.device) for d in range(2, x.ndim)]
        grid =  torch.stack(torch.meshgrid(grid, indexing='ij'), 0)
        grid[0,:,:,:]=grid[0,:,:,:]*thickness
        grid_shape = list(grid.shape[1:])
        grid_shape[0] = grid_shape[0]*thickness #add spacing 
        grid = grid[None]
        flow[:,0,:,:,:]=flow[:,0,:,:,:]*thickness


        grid = True * flow + grid
        shape = shape if shape is not None else list(x.shape[2:])


        if not self.transpose: #slice
            grid = 2.0 / (torch.tensor(grid_shape, device=x.device).reshape([1,-1] + [1] * (grid_ndim - 2)) - 1) * grid - 1.0
            grid = torch.permute(grid.flip(1), [0] + list(range(2, grid_ndim)) + [1])
            x1 = F.grid_sample(x, grid, mode=mode, padding_mode='border', align_corners=True)
        else: #splat
            grid = torch.permute(grid, [0] + list(range(2, grid.ndim)) + [1]) * ((torch.tensor(shape, device=x.device) - 1)/(torch.tensor(grid.shape[2:], device=x.device) - 1))
            mask = F.interpolate(mask, size=x.shape[2:], mode=self.mode, align_corners=True) if mask.shape[2:] != x.shape[2:] else mask
            x1 = interpol.grid_push(torch.cat([x * mask, mask], 1), grid, shape=shape, bound=1, extrapolate=True) #
        return x1 #torch.cat([x0, x1], 0)

class WarpingLayer_3stacks(nn.Module):
    def __init__(self, warp=True, X=3, transpose=False, normalize=True):
        super().__init__()
        self.mode = 'linear' if X == 1 else 'bilinear' if X == 2 else 'trilinear' if X == 3 else 'nearest'
        self.warp = warp
        self.transpose = transpose
        self.normalize = normalize
      #  print(self.mode)
    
    def forward_3stacks(self, x, flow, mask=None, mode='bilinear', shape=None, transpose=True, num_stacks=3):
       # print("here!")
        mode2 = 'trilinear'
        
        flow = torch.stack([flow[:, d - 2] * ((x.shape[d] - 1)/(flow.shape[d] - 1)) for d in range(2, x.ndim)], 1)    
        flow = F.interpolate(flow, size=x.shape[2:], mode=mode2, align_corners=True) if flow.shape[2:] != x.shape[2:] else flow

        #make grid
        if not transpose:
            grid = [torch.arange(0, shape[d], dtype=torch.float, device=x.device) for d in range(len(shape))]
            grid = torch.stack(torch.meshgrid(grid, indexing='ij'), 0)
            grid_list = [grid] * num_stacks
            grid = torch.cat((grid_list), dim=1)
            grid = True * flow + grid
           # print("ELSE 1")
        else:
            grid = [torch.arange(0, x.shape[d], dtype=torch.float, device=x.device) for d in range(2, x.ndim)]
            #print("ELSE 2")
            grid = self.warp * flow + torch.stack(torch.meshgrid(grid, indexing='ij'), 0)
        shape = shape if shape is not None else list(x.shape[2:])
       # print("done3")
        if not transpose: #slice
            grid = 2.0 / (torch.tensor(grid.shape[2:], device=x.device).reshape([1,-1] + [1] * (grid.ndim - 2)) - 1) * grid - 1.0
            grid = torch.permute(grid.flip(1), [0] + list(range(2, grid.ndim)) + [1])
            x1 = F.grid_sample(x, grid, mode=mode, padding_mode='border', align_corners=True)
        else: #splat
            
            grid = torch.permute(grid, [0] + list(range(2, grid.ndim)) + [1]) # * ((torch.tensor(shape, device=x.device) - 1)/(torch.tensor(grid.shape[2:], device=x.device) - 1))
          #  print("done4")
            mask = F.interpolate(mask, size=x.shape[2:], mode=mode2, align_corners=True) if mask.shape[2:] != x.shape[2:] else mask
    
            x1 = interpol.grid_push(torch.cat([x * mask, mask], 1), grid, shape=shape, bound=1, extrapolate=True) #
        return x1 #torch.cat([x0, x1], 0)
    
    
  
    def apply_flow_thin(self,x, flow, ALL_STACKS, flow_dim, vol_dim, volume_shape, slice_dim, mask=None, mode='trilinear',ot=True):
        """
        Applies slice-wise spatial transformation to the input volume using a flow field and grid sampling.

        Parameters:
            x (torch.Tensor): Input volume tensor of shape (B, C, D, H, W).
            flow (torch.Tensor): Flow field of shape (1, 3, D', H', W') representing displacements.
            flow_dim (int or float): Original physical dimension of flow in mm.
            out_dim (int or float): Target physical output dimension in mm.
            volume_shape (tuple): Shape of the output 3D volume (D, H, W).
            slice_dim (tuple): Dimensions of individual slices (slice_thickness, height, width).
            mask (torch.Tensor, optional): Mask tensor for weighted interpolation.
            mode (str): Interpolation mode for resizing and grid sampling (default: 'trilinear').
            shape (tuple, optional): Desired output shape for the warped volume.
            self_transpose (bool): If True, uses push interpolation; if False, uses grid sampling.

        Returns:
            torch.Tensor: Warped or reconstructed volume based on the flow field and transformation mode.
        """
        # print("call to og affs")
        # print(128, slice_dim, slice_dim[0])
        # print("call to make grid")
        # print( 0, [128,x.shape[3],x.shape[4]],  vol_dim, volume_shape)

        grid_start = make_grid_one(ALL_STACKS, [x.shape[3],x.shape[4]],  vol_dim,  slice_dim, device=flow.device)


        flow= flow[:,0:3] * flow_dim/vol_dim
   

        interp_shape = list(flow.shape)
        interp_shape[3:] = list(x.shape[3:])
        flow = F.interpolate(flow, size=interp_shape[2:], mode='trilinear', align_corners=True) if flow.shape[2:] != x.shape[2:] else flow        
    

        grid =  flow + grid_start
    # shape = shape if shape is not None else list(x.shape[2:])

        
        if not self.transpose: # slice
            grid1 = 2.0 / (torch.tensor(volume_shape, device=x.device).reshape([1,-1] + [1] * (grid[:,:,0:128].ndim - 2)) - 1) * grid[:,:,0:128] - 1.0
            grid2 = 2.0 / (torch.tensor(volume_shape, device=x.device).reshape([1,-1] + [1] * (grid[:,:,0:128].ndim - 2)) - 1) * grid[:,:,128:256] - 1.0
            grid3 = 2.0 / (torch.tensor(volume_shape, device=x.device).reshape([1,-1] + [1] * (grid[:,:,0:128].ndim - 2)) - 1) * grid[:,:,256:384] - 1.0
            grid =  torch.cat([grid1, grid2, grid3], dim=2)
            grid = torch.permute(grid.flip(1), [0] + list(range(2, grid.ndim)) + [1])
            x1 = F.grid_sample(x, grid, mode=mode, padding_mode='border', align_corners=True)

            
            
        else:

            grid = torch.permute(grid, [0] + list(range(2, grid.ndim)) + [1]) 
            mask = F.interpolate(mask, size=x.shape[2:], mode=self.mode, align_corners=True) if mask.shape[2:] != x.shape[2:] else mask
            intensities = torch.cat([x * mask, mask], 1)
            x1 = interpol.grid_push(intensities, grid, shape=volume_shape, bound=1, extrapolate=True) #
        return x1 #torch.cat([x0, x1], 0)        
      

    def apply_flow_thick(self,
                                x,
                                affs,
                                ALL_STACKS,
                                flow_dim,
                                vol_dim,
                                volume_shape,
                                slice_dim,
                                psf,
                                mask=None,
                                mode='trilinear'):
        """
        Memory-optimized, cleaned version of your forward.
        Notes:
        - avoid building long-lived intermediate tensors
        - delete temporaries ASAP
        - if running inference, call model under `with torch.inference_mode():` for extra savings
        """

        device = x.device
        psf_vals, psf_coords = psf[0], psf[1]
        n_psf = int(psf_vals.shape[1])

        # --- if splatting (transpose) we need to repeat slice intensities ---
        if self.transpose:
            x = x.repeat_interleave(n_psf, dim=2)
            if mask is not None:
                mask = mask.repeat_interleave(n_psf, dim=2)

        # --- build base grid and repeat for PSF support ---
        grid_start = make_grid_one(ALL_STACKS, [x.shape[3], x.shape[4]], vol_dim, slice_dim, device=device)
        grid_repeat = grid_start.repeat_interleave(n_psf, dim=1)
        affs = affs.repeat_interleave(n_psf, dim=0)

        # expand PSF coords to match repeated grid and device
        psf_coords = psf_coords.repeat(1, grid_start.shape[1]).unsqueeze(-1).unsqueeze(-1).to(device)

        # build final per-psf-grid and clean temporaries immediately
        new_grid = psf_coords + grid_repeat
        del psf_coords, grid_repeat, grid_start

        grid_add_combined = new_grid[None]                          # shape (1, coords, H, W, ...)
        ones = torch.ones(1, 1, grid_add_combined.shape[2], grid_add_combined.shape[3], grid_add_combined.shape[4], device=device)
        grid_ones = torch.cat([grid_add_combined, ones], dim=1)     # homogeneous coords
        del grid_add_combined, ones, new_grid

        # apply affine transforms: einsum same as original
        grid_transformed = torch.einsum('nij,jnlm->inlm', affs, grid_ones[0])
        del affs, grid_ones
        grid_transformed_ = grid_transformed[:3][None]              # keep x,y,z rows
        del grid_transformed

        # --- non-transpose (sampling / slicing) branch ---
        if not self.transpose:
            # normalize coords into [-1, 1] for grid_sample
            vol_shape_tensor = torch.tensor(volume_shape, device=device)
            # build scale reshape without extra allocation where possible
            scale = 2.0 / (vol_shape_tensor.reshape([1, -1] + [1] * (grid_transformed_.ndim - 2)) - 1.0)
            grid = scale * grid_transformed_ - 1.0
            grid = torch.permute(grid.flip(1), [0] + list(range(2, grid.ndim)) + [1])

            # sample
            x1 = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=True)
            del grid, scale, vol_shape_tensor

            # apply PSF mixing: reshape -> einsum -> reshape-back
            B, C, Z, H, W = x1.shape
            x1_unfold = x1.reshape(B, C, int(Z / n_psf), n_psf, H, W).movedim(3, 2)
            # psf_vals: (k, n_psf) expected; einsum accumulates PSF along psf dim
            x1 = torch.einsum('ij,bcjklm->bciklm', psf_vals, x1_unfold)
            # final shape: (B, C, out_z, H, W)
            x1 = x1.reshape(B, C, x1.shape[3], x1.shape[4], x1.shape[5])

            # cleanup
            del x1_unfold, psf_vals

        # --- transpose (splat / push) branch ---
        else:
            grid = torch.permute(grid_transformed_, [0] + list(range(2, grid_transformed_.ndim)) + [1])

            if mask is None:
                raise ValueError("mask is required when self.transpose is True")

            # ensure mask has same per-slice resolution as x
            if mask.shape[2:] != x.shape[2:]:
                mask = F.interpolate(mask, size=x.shape[2:], mode='trilinear', align_corners=True)

            # combine PSF with mask (reduce temporary allocations, free quickly)
            mask_unfold = mask.reshape(mask.shape[0], mask.shape[1], int(mask.shape[2] / n_psf), n_psf, mask.shape[3], mask.shape[4])
            mask_unfold = torch.einsum('ij,bckjlm->bcjklm', psf_vals, mask_unfold)
            mask_unfold = mask_unfold.movedim(3, 2).reshape(mask.shape)

            # build intensities (signal * mask, and mask channel)
            intensities = torch.cat([x * mask_unfold, mask_unfold], dim=1)

            # free what we can before expensive push
            del mask_unfold, x, psf_vals

            # call push (keeps same interface as your original)
            x1 = interpol.grid_push(intensities, grid, shape=volume_shape, bound=1, extrapolate=True)

            # cleanup
            del intensities, grid

        # final memory reclaim and return
        torch.cuda.empty_cache()
        return x1


    def forward_all_stacks(self, x, flow,mask=None, mode='bilinear', shape=None, num_stacks=3, x_ratio=1, out_sz=1, flow_dim=1, max_size=128, ot =True): # FORMERLY CHANGED THIS ONE DOES IT RIGHT JAN 12 SPLAT
        """
        Does splatting (if self.transpose is True) and slicing (if self.transpose is False)

        This function takes in three orthogonal stacks (axial, coronal, sagittal),
        scales each stack using self.scale_flow, and then either:
        - slices: warps the input `x` using F.grid_sample (if self.transpose is False), or
        - splats: pushes masked input data into a 3D volume using interpol.grid_push.

        Args:
            x (Tensor): Input tensor of shape (B, C, D, H, W) representing slice data. Slices are along D dimension
            flow (Tensor): Flow field tensor of shape (B, 3, num_stacks * D, H, W).
            mask (Tensor, optional): Binary mask tensor of shape (B, 1, D, H, W). Used only if self.transpose is True.
            mode (str): Interpolation mode for grid_sample ('bilinear' or 'nearest'). Default: 'bilinear'.
            shape (tuple, optional): Output shape used by grid_push when self.transpose is True.
            num_stacks (int): Number of orthogonal stacks to split the flow into (default: 3).
            x_ratio (float): In-plane scaling ratio for input pixels (e.g., anisotropy correction).
            out_sz (int): Target voxel size used in output volume (in mm).
            flow_dim (int): Scale of the flow field in each spatial dimension (e.g., 1 for voxel units). (in mm)
            max_size (int): Spatial size for output grid generation.
            ot (bool): Whether to apply orthogonal flow to place in correct orientation

        Returns:
            Tensor:  splatted volume or set of slices, depending on self.transpose.
        """
        
        # x -> slices in
        # 
        sss = int(flow.shape[2]/num_stacks)

        
        flow1 = flow[:,0:3,0:sss,:,:]
        flow2 = flow[:,0:3,sss:sss*2,:,:]
        flow3 = flow[:,0:3,sss*2:sss*3,:,:]


        if(self.transpose):
            x_shape_in =list(x.shape)
            x_shape_in[2] = int(x_shape_in[2]/num_stacks)
        else:
            x_shape_in =list(x.shape)


        grid1 = self.scale_flow(flow=flow1, flow_vox_dim=[flow_dim,flow_dim,flow_dim], x_shape= x_shape_in, pix_dim=[1,x_ratio,x_ratio],  out_vox_dim=[out_sz,out_sz,out_sz], out_shape=[max_size/1,max_size/1,max_size/1], slice_dir=0, ot=ot)
        grid2 = self.scale_flow(flow=flow2, flow_vox_dim=[flow_dim,flow_dim,flow_dim], x_shape= x_shape_in, pix_dim=[x_ratio,1,x_ratio],  out_vox_dim=[out_sz,out_sz,out_sz], out_shape=[max_size/1,max_size/1,max_size/1], slice_dir=1, ot=ot)
        grid3 = self.scale_flow(flow=flow3, flow_vox_dim=[flow_dim,flow_dim,flow_dim], x_shape= x_shape_in, pix_dim=[x_ratio,x_ratio,1],  out_vox_dim=[out_sz,out_sz,out_sz], out_shape=[max_size/1,max_size/1,max_size/1], slice_dir=2, ot=ot)
     #   print("")

        if(self.transpose):
            grid = torch.cat((grid1,grid2,grid3),dim=1)
            print("splat shape")
            print(grid.shape)
        else:
            grid = torch.cat((grid1,grid2,grid3),dim=2)
            print("slice shape")
            print(grid.shape)


        if self.transpose: #splat
            mask = F.interpolate(mask, size=x.shape[2:], mode='trilinear', align_corners=True) if mask.shape[2:] != x.shape[2:] else mask
          #  pdb.set_trace()
            x1 = interpol.grid_push(torch.cat([x * mask, mask], 1), grid, shape=shape, bound=1, extrapolate=True) #
        
        if not self.transpose: # slice
           # pdb.set_trace()
           # grid = 2.0 / (torch.tensor(grid.shape[2:], device=x.device).reshape([1,-1] + [1] * (grid.ndim - 2)) - 1) * grid - 1.0
            grid = torch.permute(grid.flip(1), [0] + list(range(2, grid.ndim)) + [1])
            x1 = F.grid_sample(x, grid, mode=mode, padding_mode='border', align_corners=True)


        return x1 #torch.cat([x0, x1], 0)


        
        

        
    
    def forward(self, x, flow, mask=None, mode='bilinear', shape=None, num_stacks=3): # FORMERLY CHANGED THIS ONE DOES IT RIGHT JAN 12 SPLAT
      #  print("in this forward!!!!")
       # print("orginal forward")
        mode2 = 'trilinear'
        og= True
        shape = shape if shape is not None else list(x.shape[2:])
        if(og==True):
            flow_shape = x.shape
        else:
            flow_shape =  list(x.shape)
            flow_shape[2] = shape[0]*int(flow.shape[2]/shape[0])

       
        if not self.transpose:
            flow_shape = list(flow_shape)
            flow_shape[2] = ((x.shape[2])/(flow.shape[3]*num_stacks))*(x.shape[2])

        flow = torch.stack([flow[:, d - 2] * ((flow_shape[d] - 1)/(flow.shape[d] - 1)) for d in range(2, x.ndim)], 1)  
        flow_shape =  list(x.shape)
        flow = F.interpolate(flow, size=flow_shape[2:], mode=mode2, align_corners=True) if flow.shape[2:] != flow_shape[2:] else flow
        
        shape = shape if shape is not None else list(flow_shape[2:])

      #  pdb.set_trace()
        
        flow_to_ot = self.flow_to_orthogonal2(flow)[None]
       # flow = flow + self.flow_to_orthogonal2(flow)[None]
 
        if self.transpose: # splat

            new_shape = list(flow_shape[2:])
            if(shape !=new_shape):  
                new_shape[0] = int(new_shape[0]/num_stacks)
            else: # for splatting into seperate staccks
               new_shape = shape
           # pdb.set_trace()
            
          #  new_shape = [128*6,128,128]
            grid = [torch.arange(0, new_shape[d], dtype=torch.float, device=x.device) for d in range(len(new_shape))]
            grid = torch.stack(torch.meshgrid(grid, indexing='ij'), 0)
           # pdb.set_trace()     
            

            grid_list = [grid] * num_stacks
            grid = torch.cat((grid_list), dim=1)
            del grid_list
            

           
            grid = True * flow_to_ot + grid
           
   

        else:
          #  print("ELSE 2")
           # flow[:,0] = flow[:,0]*2
            flow = flow + flow_to_ot
            
        
            grid = [torch.arange(0, flow_shape[d], dtype=torch.float, device=x.device) for d in range(2, x.ndim)]
            grid = self.warp * flow + torch.stack(torch.meshgrid(grid, indexing='ij'), 0)

        shape = shape if shape is not None else list(flow_shape[2:])



        if not self.transpose: #slice
            grid = 2.0 / (torch.tensor(grid.shape[2:], device=x.device).reshape([1,-1] + [1] * (grid.ndim - 2)) - 1) * grid - 1.0
            grid = torch.permute(grid.flip(1), [0] + list(range(2, grid.ndim)) + [1])
            x1 = F.grid_sample(x, grid, mode=mode, padding_mode='border', align_corners=True)

        else: #splat
         #   pdb.set_trace()
            
            #* ((torch.tensor(shape, device=x.device) - 1)/(torch.tensor(new_shape, device=x.device) - 1))
           # grid = grid[None]
            new_grid =  torch.permute(grid, [0] + list(range(2, grid.ndim)) + [1])
           # pdb.set_trace()
            
            height = new_shape[0]
            crop = new_shape[1]
            ratio = num_stacks/int(flow.shape[2]/shape[0])
            center1 = (height-crop)/2 #(height-1)/2 before was 64 + 10 - found these numbers empiracally 
            center2 = center1*ratio # before was 32
            # print("CENTER 1, CENTER2")
            # print(center1, center2)
     
            
            
          #  which axis gets scaled by half changes depending on the orientation of the stack
          #  pdb.set_trace()
           # pdb.set_trace() 
            new_grid[:,0:height,:,:,:] = (torch.permute(grid, [0] + list(range(2, grid.ndim)) + [1])  * torch.tensor([ratio,1,1],device=x.device))[:,0:height,:,:,:] #- torch.tensor([10,0,0],device=x.device)
            new_grid[:,height:height*2,:,:,:] = (torch.permute(grid, [0] + list(range(2, grid.ndim)) + [1])  * torch.tensor([1,ratio,1],device=x.device) )[:,height:height*2,:,:,:] + torch.tensor([-center1,center2,0],device=x.device) # was 32 for 3 stacks
            new_grid[:,height*2:,:,:,:] = (torch.permute(grid, [0] + list(range(2, grid.ndim)) + [1])  * torch.tensor([1,1,ratio],device=x.device))[:,height*2:,:,:,:] + torch.tensor([-center1,0,center2],device=x.device)

            new_flow =  torch.permute(flow, [0] + list(range(2, grid.ndim)) + [1]) #* torch.tensor([1,1,1],device=x.device)
            
            new_grid = True * new_flow + new_grid
            grid = new_grid
       

            if(og==False):
                x = F.interpolate(x, size=flow_shape[2:], mode=mode2, align_corners=True) 

            mask = F.interpolate(mask, size=flow_shape[2:], mode=mode2, align_corners=True) if mask.shape[2:] != flow_shape[2:] else mask
          #  pdb.set_trace()
            x1 = interpol.grid_push(torch.cat([x * mask, mask], 1), grid, shape=shape, bound=1, extrapolate=True) #



        return x1 #torch.cat([x0, x1], 0)
    

   
class Flow_UNet(nn.Module):
    def __init__(self, input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage=4, num_conv_per_flow=4,
                 featmul=2, mask=False, norm=False, dropout_p=0.0, weightInitializer=InitWeights_He(1e-2), pool_kernel_sizes=2,
                 slab_kernel_sizes=3, conv_kernel_sizes=3, slab_stride_sizes=1, conv_stride_sizes=1, convolutional_pooling=True,
                 normalize_splat=True, convolutional_upsampling=True, max_num_features=None, bias=False, shortcut=True, warp=True, X=3):
        print("defining U-nets")
        super().__init__()

        num_features = [24, 32, 48, 64, 96, 128, 192, 256, 320] if isinstance(base_num_features, int) else base_num_features

        self.num_features = num_features
        self.enc_blocks = [None] * (num_pool + 1)
        self.dec_blocks = [None] * (num_pool + 1)
        self.flo_blocks = [None] * (num_pool + 1)

        self.mode = 'linear' if X == 1 else 'bilinear' if X == 2 else 'trilinear' if X == 3 else 'nearest'
        self.mask = mask
        self.seg_outputs = []
        self.warp = WarpingLayer(warp=warp, X=X)
        self.splat = WarpingLayer(warp=warp, X=X, normalize=normalize_splat, transpose=True)
        self.interp = UpsampleLayer(X=X)

        # StackedConvLayers is same implementation as StackedConvTransposeLayers
        # add convolutions
        self.enc_blocks[0] = StackedConvLayers(input_channels, num_features[0], num_conv_per_stage, conv_kernel_sizes, 1, bias=bias, norm=norm, drop=dropout_p, relu=True, X=X)
        for d in range(num_pool): #encoder
            # add convolutions
            in_channels = num_features[d + 0] #min(base_num_features * (featmul ** d), 384)
            out_channels = num_features[d + 1] #min(base_num_features * (featmul ** (d + 1)), 384)
            self.enc_blocks[d + 1] = StackedConvLayers(in_channels, out_channels, num_conv_per_stage, 
                                                       conv_kernel_sizes, pool_kernel_sizes,  bias=bias, norm=norm, drop=dropout_p, relu=True, X=X)
        for u in reversed(range(num_pool)): #decoder
            # add convolution
            in_channels = num_features[u + 1] #min(base_num_features * (featmul ** (u + 1)), 384) * 2
            out_channels = num_features[u + 0] #min(base_num_features * (featmul ** u), 384) * 2
            self.dec_blocks[u + 1] = StackedConvTransposeLayers(in_channels, out_channels, num_conv_per_stage, 
                                                                conv_kernel_sizes, pool_kernel_sizes, bias=bias, norm=norm, drop=dropout_p, relu=True, X=X)
            self.flo_blocks[u + 1] = SlabbedConvLayers(2 * out_channels, X, num_conv_per_flow,  slab_kernel_sizes=slab_kernel_sizes, conv_kernel_sizes=conv_kernel_sizes,
                                                       slab_stride_sizes=slab_stride_sizes, mask=mask, bias=bias, norm=norm, drop=False, relu=True, X=X)

        self.dec_blocks[0] = StackedConvTransposeLayers(num_features[0], num_classes, num_conv_per_stage, conv_kernel_sizes, 1, bias=bias, norm=norm, drop=dropout_p, relu=True, X=X)
        self.flo_blocks[0] = SlabbedConvLayers(2 * num_classes, X, num_conv_per_flow, slab_kernel_sizes=slab_kernel_sizes, conv_kernel_sizes=conv_kernel_sizes, 
                                               slab_stride_sizes=slab_stride_sizes, mask=mask, bias=bias, norm=norm, drop=False, relu=True, X=X)

        # register all modules properly
        self.dec_blocks = nn.ModuleList(self.dec_blocks)
        self.enc_blocks = nn.ModuleList(self.enc_blocks)
        self.flo_blocks = nn.ModuleList(self.flo_blocks)

        if weightInitializer is not None:
            self.apply(weightInitializer)

    def forward(self, x):

        shape = x.shape
        skips = [None] * len(self.enc_blocks)
        sizes = [None] * len(self.enc_blocks)
        x = torch.cat(x.tensor_split(2, 1), 0) #.reshape([x.shape[0] * 2, x.shape[1] // 2] + list(x.shape[2:]))
        for d in range(len(self.enc_blocks)):
            x = skips[d] = self.enc_blocks[d](x)
            sizes[d] = list(x.shape[2:])

        flow = torch.zeros([x.shape[0] // 2, x.ndim - 2] + sizes[-1], device=x.device)
        delta = torch.zeros([x.shape[0] // 2, x.ndim - 2] + sizes[-1], device=x.device)
        flows = [None] * len(self.dec_blocks)

        for u in reversed(range(len(self.dec_blocks))):
            x = self.dec_blocks[u](x, skips[u]) #.tensor_split(2)
            y = [x.tensor_split(2, 0)[0], self.warp(x.tensor_split(2, 0)[1], flow)]; y = torch.cat([0.5 * (y[0] + y[1]), y[0] - y[1]], 1)

            flow, _ = self.flow_add(flow, self.flo_blocks[u](y)) # * self.seg_blocks[u](y)))
            flows[u] = torch.stack([flow[:, d - 2] * ((shape[d] - 1)/(flow.shape[d] - 1)) for d in range(2, flow.ndim)], 1).flip(1)
        return flows[0] #if self.ds and self.training else flows[0]

    def flow_add(self, flow, delta):
        flow = torch.stack([flow[:, d - 2] * ((delta.shape[d] - 1)/(flow.shape[d] - 1)) for d in range(2, delta.ndim)], 1)
        flow = F.interpolate(flow, size=delta.shape[2:], mode=self.mode, align_corners=True) if flow.shape[2:] != delta.shape[2:] else flow
        
        return flow + delta[:, :delta.ndim - 2], delta[:, delta.ndim - 2:]

class Flow_UNet_3stacks(nn.Module):
    def __init__(self, input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage=4, num_conv_per_flow=4,
                 featmul=2, mask=False, norm=False, dropout_p=0.0, weightInitializer=InitWeights_He(1e-2), pool_kernel_sizes=2,
                 slab_kernel_sizes=3, conv_kernel_sizes=3, slab_stride_sizes=1, conv_stride_sizes=1, convolutional_pooling=True,
                 normalize_splat=True, convolutional_upsampling=True, max_num_features=None, bias=False, shortcut=True, warp=True, X=3):

        super().__init__()

        num_features = [24, 32, 48, 64, 96, 128, 192, 256, 320] if isinstance(base_num_features, int) else base_num_features

        extra_bottleneck = True

        
        self.num_features = num_features
        self.enc_blocks = [None] * (num_pool + 1)
        self.dec_blocks = [None] * (num_pool + 1)
        self.flo_blocks = [None] * (num_pool + 1)

        if(extra_bottleneck):
            self.dec_blocks = [None] * (num_pool + 2)
            self.flo_blocks = [None] * (num_pool + 2)

        self.mode = 'linear' if X == 1 else 'bilinear' if X == 2 else 'trilinear' if X == 3 else 'nearest'
        self.mask = mask
        self.seg_outputs = []
        self.warp = WarpingLayer_3stacks(warp=warp, X=X)
        self.splat = WarpingLayer_3stacks(warp=warp, X=X, normalize=normalize_splat, transpose=True)
        self.interp = UpsampleLayer(X=X)
        

        # add convolutions
        self.enc_blocks[0] = StackedConvLayers(input_channels, num_features[0], num_conv_per_stage, conv_kernel_sizes, 1, bias=bias, norm=norm, drop=dropout_p, relu=True, X=X)
        for d in range(num_pool):
            # add convolutions
            in_channels = num_features[d + 0] #min(base_num_features * (featmul ** d), 384)
            out_channels = num_features[d + 1] #min(base_num_features * (featmul ** (d + 1)), 384)
            self.enc_blocks[d + 1] = StackedConvLayers(in_channels, out_channels, num_conv_per_stage, 
                                                       conv_kernel_sizes, pool_kernel_sizes,  bias=bias, norm=norm, drop=dropout_p, relu=True, X=X)
        

        if(extra_bottleneck):
            u =num_pool
            in_channels = num_features[u ] #min(base_num_features * (featmul ** (u + 1)), 384) * 2
            out_channels = num_features[u ] #min(base_num_features * (featmul ** u), 384) * 2
            self.dec_blocks[u + 1] = StackedConvTransposeLayers(in_channels, out_channels, num_conv_per_stage, 
                                                                                conv_kernel_sizes, pool_kernel_sizes, bias=bias, norm=norm, drop=dropout_p, relu=True, X=X)
            self.flo_blocks[u + 1] = SlabbedConvLayers(2 * out_channels, X, num_conv_per_flow,  slab_kernel_sizes=slab_kernel_sizes, conv_kernel_sizes=conv_kernel_sizes,
                                                        slab_stride_sizes=slab_stride_sizes, mask=mask, bias=bias, norm=norm, drop=False, relu=True, X=X)

        

        for u in reversed(range(num_pool)):
            # add convolution
            in_channels = num_features[u + 1] #min(base_num_features * (featmul ** (u + 1)), 384) * 2
            out_channels = num_features[u + 0] #min(base_num_features * (featmul ** u), 384) * 2
            self.dec_blocks[u + 1] = StackedConvTransposeLayers(in_channels, out_channels, num_conv_per_stage, 
                                                                conv_kernel_sizes, pool_kernel_sizes, bias=bias, norm=norm, drop=dropout_p, relu=True, X=X)
            # for simp2
           # self.flo_blocks[u + 1] = SlabbedConvLayers(4 * out_channels, X, num_conv_per_flow,  slab_kernel_sizes=slab_kernel_sizes, conv_kernel_sizes=conv_kernel_sizes,
                                    #                   slab_stride_sizes=slab_stride_sizes, mask=mask, bias=bias, norm=norm, drop=False, relu=True, X=X)
            self.flo_blocks[u + 1] = SlabbedConvLayers(2 * out_channels, X, num_conv_per_flow,  slab_kernel_sizes=slab_kernel_sizes, conv_kernel_sizes=conv_kernel_sizes,
                                                       slab_stride_sizes=slab_stride_sizes, mask=mask, bias=bias, norm=norm, drop=False, relu=True, X=X)

        self.dec_blocks[0] = StackedConvTransposeLayers(num_features[0], num_classes, num_conv_per_stage, conv_kernel_sizes, 1, bias=bias, norm=norm, drop=dropout_p, relu=True, X=X)
       
        # for simp2
      #  self.flo_blocks[0] = SlabbedConvLayers(8 * num_classes, X, num_conv_per_flow, slab_kernel_sizes=slab_kernel_sizes, conv_kernel_sizes=conv_kernel_sizes, 
                                        #       slab_stride_sizes=slab_stride_sizes, mask=mask, bias=bias, norm=norm, drop=False, relu=True, X=X)


        self.flo_blocks[0] = SlabbedConvLayers(2 * num_classes, X, num_conv_per_flow, slab_kernel_sizes=slab_kernel_sizes, conv_kernel_sizes=conv_kernel_sizes, 
                                               slab_stride_sizes=slab_stride_sizes, mask=mask, bias=bias, norm=norm, drop=False, relu=True, X=X)


        # register all modules properly
        self.dec_blocks = nn.ModuleList(self.dec_blocks)
        self.enc_blocks = nn.ModuleList(self.enc_blocks)
        self.flo_blocks = nn.ModuleList(self.flo_blocks)

        if weightInitializer is not None:
            self.apply(weightInitializer)
       # print("IN DEFINING U-NETS")


    def flow_add(self, flow, delta):
       # pdb.set_trace()
       # print("dif flow add")
        if(delta.shape[2:]!=flow.shape[2:] ):

            flow[:,0:] = flow[:,0:]*2
            
        else:
            1==1
           # print("og flow add")

       # flow = torch.stack([flow[:, d - 2] * ((delta.shape[d] - 1)/(flow.shape[d] - 1)) for d in range(2, delta.ndim)], 1)
        
        flow = F.interpolate(flow, size=delta.shape[2:], mode=self.mode, align_corners=True) if flow.shape[2:] != delta.shape[2:] else flow
        
        return flow + delta[:, :delta.ndim - 2], delta[:, delta.ndim - 2:]
    
    
 
class Flow_UNet_3stacks_encoder_only(nn.Module):
    def __init__(self, input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage=4, num_conv_per_flow=4,
                 featmul=2, mask=False, norm=False, dropout_p=0.0, weightInitializer=InitWeights_He(1e-2), pool_kernel_sizes=2,
                 slab_kernel_sizes=3, conv_kernel_sizes=3, slab_stride_sizes=1, conv_stride_sizes=1, convolutional_pooling=True,
                 normalize_splat=True, convolutional_upsampling=True, max_num_features=None, bias=False, shortcut=True, warp=True, X=3):

        super().__init__()

        num_features = [24, 32, 48, 64, 96, 128, 192, 256, 320] if isinstance(base_num_features, int) else base_num_features

        extra_bottleneck = True

        
        self.num_features = num_features
        self.enc_blocks = [None] * (num_pool + 1)

        self.mode = 'linear' if X == 1 else 'bilinear' if X == 2 else 'trilinear' if X == 3 else 'nearest'
        self.mask = mask
        self.seg_outputs = []
        self.warp = WarpingLayer_3stacks(warp=warp, X=X)
        self.splat = WarpingLayer_3stacks(warp=warp, X=X, normalize=normalize_splat, transpose=True)
        self.interp = UpsampleLayer(X=X)
        

        # add convolutions
        self.enc_blocks[0] = StackedConvLayers(input_channels, num_features[0], num_conv_per_stage, conv_kernel_sizes, 1, bias=bias, norm=norm, drop=dropout_p, relu=True, X=X)
        for d in range(num_pool):
            # add convolutions
            in_channels = num_features[d + 0] #min(base_num_features * (featmul ** d), 384)
            out_channels = num_features[d + 1] #min(base_num_features * (featmul ** (d + 1)), 384)
            self.enc_blocks[d + 1] = StackedConvLayers(in_channels, out_channels, num_conv_per_stage, 
                                                       conv_kernel_sizes, pool_kernel_sizes,  bias=bias, norm=norm, drop=dropout_p, relu=True, X=X)
        





        # register all modules properly
        self.enc_blocks = nn.ModuleList(self.enc_blocks)

        if weightInitializer is not None:
            self.apply(weightInitializer)
       # print("IN DEFINING U-NETS")

    def flow_add(self, flow, delta):
       # pdb.set_trace()
       # print("dif flow add")
        if(delta.shape[2:]!=flow.shape[2:] ):

            flow[:,0:] = flow[:,0:]*2
            
        else:
            1==1
           # print("og flow add")

       # flow = torch.stack([flow[:, d - 2] * ((delta.shape[d] - 1)/(flow.shape[d] - 1)) for d in range(2, delta.ndim)], 1)
        
        flow = F.interpolate(flow, size=delta.shape[2:], mode=self.mode, align_corners=True) if flow.shape[2:] != delta.shape[2:] else flow
        
        return flow + delta[:, :delta.ndim - 2], delta[:, delta.ndim - 2:]
    
    
    
   


def flow_UNet2d(input_channels, num_classes, **kwargs):
    return Flow_UNet(input_channels=input_channels, base_num_features=[16, 24, 32, 48, 64, 96, 128, 192, 256, 320], num_classes=2, num_pool=7, X=2)

def flow_UNet2d_postwarp(input_channels, num_classes, **kwargs):
    return Flow_UNet(input_channels=input_channels, base_num_features=[16, 24, 32, 48, 64, 96, 128, 192, 256, 320], num_classes=2, num_pool=7, X=2)

def flow_UNet2d_nowarp(input_channels, num_classes, **kwargs):
    return Flow_UNet(input_channels=input_channels, base_num_features=[16, 24, 32, 48, 64, 96, 128, 192, 256, 320], num_classes=2, num_pool=7, warp=False, X=2)

def flow_UNet3d(input_channels, num_classes, **kwargs):
    return Flow_UNet(input_channels=input_channels, base_num_features=[16, 24, 32, 48, 64, 96, 128, 192, 256, 320], num_classes=3, num_pool=6, X=3)

def flow_UNet3d_postwarp(input_channels, num_classes, **kwargs):
    return Flow_UNet(input_channels=input_channels, base_num_features=[16, 24, 32, 48, 64, 96, 128, 192, 256, 320], num_classes=3, num_pool=6, X=3)

def flow_UNet3d_nowarp(input_channels, num_classes, **kwargs):
    return Flow_UNet(input_channels=input_channels, base_num_features=[16, 24, 32, 48, 64, 96, 128, 192], num_classes=3, num_pool=6, warp=False, X=3)