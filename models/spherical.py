import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import interpol
#from skimage import measure
import pdb
from .flow_SNet4 import Flow_SNet_multi

from .grid_utils import make_grid_one

def cce_loss(outputs, targets, weights=1, reduction='mean', **kwargs):
    weights = torch.as_tensor(weights, device=outputs.device)
    numer = torch.sum(weights * -(targets * torch.log_softmax(outputs, dim=1)), keepdim=True, axis=1)
    denom = torch.sum(weights * (targets), keepdim=True, axis=1)

    if reduction == 'none':
        return numer
    
    return torch.sum(numer) / torch.sum(denom)

def dce_loss(outputs, targets, weights=1, reduction='mean', **kwargs):
    outputs = F.log_softmax(outputs, dim=1)
    axes = [0] + list(range(2, outputs.ndim))
    numer = torch.sum(targets * -outputs, keepdim=True, axis=axes)
    denom = torch.sum(targets, keepdim=True, axis=axes)

    return torch.mean(numer / denom)


def l2_loss(out, tar, weights=1, kernel_size=1, reduction='mean', **kwargs):
    batch, chans, stacks, *size = out.shape
    mask = tar[:,chans:] # if masked else torch.ones_like(tar[:,chans:])
    weights = torch.as_tensor(weights, device=out.device)

    return torch.mean(weights * (out[:,:chans] - tar[:,:chans]) ** 2)



def l22_loss_grid(out, tar, masked=True, eps=0e-4, reduction='mean', **kwargs):

    batch, chans, *size = out.shape #batch should always be 1 
    batch_tar, chans_tar, *size_tar = tar.shape #batch should always be 1 
    
    
    flow_factor = size[1]/size_tar[1]
    tar = F.interpolate(tar, size=size, mode='area') 
    tar[0,3] = (tar[0,3] > 0.5).float()
    tar[:,0:chans] = tar[:,0:chans]* flow_factor

    return torch.sum(tar[:,chans:] * (out[:,:chans] - tar[:,:chans]) ** 2)/ torch.sum(tar[:,chans:])


def classification_onehot_loss(out, tar, reduction='sum', **kwargs):
    """
    Computes cross-entropy loss for one-hot encoded targets using raw logits.
    Args:
        outputs (torch.Tensor): Raw output logits from the model. Shape [B, C].
        targets (torch.Tensor): Ground truth one-hot vectors. Shape [B, C].
        reduction (str): 'none' | 'mean' | 'sum'
    Returns:
        torch.Tensor: The computed loss.
    """
    print("in classification onehot loss")
    # Convert logits -> log probabilities (more numerically stable than softmax + log)
    log_probs = F.log_softmax(out, dim=1)
    print("Log probs")
    print(log_probs)

    # Cross-entropy with one-hot = -sum(y * log(p))
    loss = -torch.sum(tar * log_probs, dim=1)


    # Reduction handling
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss



def classification_correct_vec(out, tar, num_vec =6 ,reduction='sum', **kwargs):  
    print("CORRECT STACK LOSS") 
    print(tar[0, :,], out[:,])
    idx_stack = torch.argmax(out[:,0:num_vec], dim=1)
    one_hot_stack = F.one_hot(idx_stack, num_classes = num_vec).float()


    print(num_vec)
    print(tar[0,:, 0:num_vec].shape, one_hot_stack.shape)
    correct_stack = torch.sum((tar[0,:, 0:num_vec] == one_hot_stack).all(dim=1)).item()
    print("done comparing!")

    return correct_stack


def classification_correct_stack(out, tar, num_stacks=3, num_rots=4, num_order=2, reduction='sum', **kwargs):  
    print("CORRECT STACK LOSS") 
    print(tar[0, :,0:num_stacks], out[:,0:num_stacks])
    idx_stack = torch.argmax(out[:,0:num_stacks], dim=1)
    one_hot_stack = F.one_hot(idx_stack, num_classes = num_stacks).float()
    print(one_hot_stack )
    correct_stack = torch.sum((tar[0, :,0:num_stacks] == one_hot_stack).all(dim=1)).item()
    print(type(correct_stack))
    return correct_stack

def classification_correct_rot(out, tar, num_stacks=6, num_rots=4, num_order=2, reduction='sum', **kwargs): 
    print("CORRECT STACK ROT")   
    print(tar[0, :,num_stacks:num_stacks+num_rots].shape, out[:,num_stacks:num_stacks+num_rots].shape)
    idx_rots = torch.argmax(out[:,num_stacks:num_stacks+num_rots], dim=1)
    one_hot_rots = F.one_hot(idx_rots, num_classes = num_rots).float()

    correct_rot = torch.sum((tar[0, :,num_stacks:num_stacks+num_rots] == one_hot_rots).all(dim=1)).item()
    return correct_rot

def classification_correct_order(out, tar, num_stacks=3, num_rots=4, num_order=2, reduction='sum', **kwargs):   
    print("CORRECT STACK ORDER")   
    print(tar[0, :,-num_order].shape, out[:,-num_order].shape)
    print(tar[0, :,-num_order].shape, out[:,-num_order].shape)

    idx_order = torch.argmax(out[:,-num_order:], dim=1)
    one_hot_order = F.one_hot(idx_order, num_classes = num_order).float()

    correct_order = torch.sum((tar[0, :,-num_order:] == one_hot_order).all(dim=1)).item()
    return correct_order

 
def classification_multihot_loss(out, tar, num_stacks=3, num_rots=4, reduction='sum', **kwargs):
    """
    Computes cross-entropy loss for one-hot encoded targets using raw logits.
    Args:
        outputs (torch.Tensor): Raw output logits from the model. Shape [B, C].
        targets (torch.Tensor): Ground truth one-hot vectors. Shape [B, C].
        reduction (str): 'none' | 'mean' | 'sum'
    Returns:
        torch.Tensor: The computed loss.
    """
    print("in classification multihot loss")


    log_probs_stack = F.log_softmax(out[:,0:num_stacks], dim=1)
    print("Log probs stacks shape" , log_probs_stack.shape, tar[0,:,0:num_stacks].shape)
    loss_stack = -torch.sum(tar[0,:,0:num_stacks] * log_probs_stack, dim=1)
   
    
    log_probs_rot = F.log_softmax(out[:,num_stacks:], dim=1)
    print("Log probs rot shape" , log_probs_rot.shape, tar[0,:,num_stacks:].shape)
    loss_rot = -torch.sum(tar[0,:,num_stacks:] * log_probs_rot, dim=1)



    # Reduction handling
    if reduction == 'mean':
        return (loss_stack + loss_rot).mean()
    elif reduction == 'sum':
        return (loss_stack + loss_rot).sum()
    else:
        return (loss_stack + loss_rot)

def classification_multihot_loss2(
    out,
    tar,
    num_stacks=3,
    num_rots=4,
    reduction="mean",
    stack_weight=1.0,
    rot_weight=1.0,
    eps=0.05,   # label smoothing factor
):
    """
    Cross-entropy loss for multi-head one-hot targets using raw logits
    with optional label smoothing.
    """

    # Split logits and targets
    out_stack, out_rot = out[:, :num_stacks], out[:, num_stacks:]
    tar_stack, tar_rot = tar[0,:, :num_stacks], tar[0,:, num_stacks:]

    # --- Label smoothing ---
    if eps > 0:
        tar_stack = tar_stack * (1 - eps) + eps / num_stacks
        tar_rot   = tar_rot   * (1 - eps) + eps / num_rots

    # Log-softmax
    log_p_stack = F.log_softmax(out_stack, dim=1)
    log_p_rot   = F.log_softmax(out_rot, dim=1)

    # Per-sample losses
    loss_stack = -(tar_stack * log_p_stack).sum(dim=1)
    loss_rot   = -(tar_rot   * log_p_rot).sum(dim=1)

    loss = stack_weight * loss_stack + rot_weight * loss_rot

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Invalid reduction: {reduction}")
    
def classification_multihot_loss_stack_order(out, tar, num_stacks=6, num_rots=3, num_order = 0, reduction='sum', **kwargs):
    """
    Computes cross-entropy loss for one-hot encoded targets using raw logits.
    Args:
        outputs (torch.Tensor): Raw output logits from the model. Shape [B, C].
        targets (torch.Tensor): Ground truth one-hot vectors. Shape [B, C].
        reduction (str): 'none' | 'mean' | 'sum'
    Returns:
        torch.Tensor: The computed loss.
    """
    print("in classification multihot loss")

    print("tar shape")
    print(tar.shape)
   
    log_probs_stack = F.log_softmax(out[:,0:num_stacks], dim=1)
    print("Log probs stacks shape" , log_probs_stack.shape, tar[0,:,0:num_stacks].shape)
    loss_stack = -torch.sum(tar[0,:,0:num_stacks] * log_probs_stack, dim=1)
   
    # log_probs_rot = F.log_softmax(out[:,num_stacks:], dim=1)
    # print("Log probs rot shape" , log_probs_rot.shape, tar[0,:,num_stacks:].shape)
    # loss_rot = -torch.sum(tar[0,:,num_stacks:] * log_probs_rot, dim=1)


    
    log_probs_rot = F.log_softmax(out[:,num_stacks:num_stacks+num_rots], dim=1)
    print("Log probs rot shape" , log_probs_rot.shape, tar[0,:,num_stacks:num_stacks+num_rots].shape)
    loss_rot = -torch.sum(tar[0,:,num_stacks:num_stacks+num_rots] * log_probs_rot, dim=1)

    # log_probs_order = F.sigmoid(out[:,-num_order:])
    # print("Log probs order shape" , log_probs_order, tar[0,:,-num_order:].shape)
    # loss_order = F.binary_cross_entropy(log_probs_order, tar[0,:,-num_order:])
    # print(loss_order, type(loss_order))

    # Reduction handling
    if reduction == 'mean':
        return (loss_stack + loss_rot ).mean()
    elif reduction == 'sum':
        return (loss_stack + loss_rot ).sum()
    else:
        return (loss_stack + loss_rot )

 
def classification_cross_multihot_vec(out, tar, num_stacks=3, num_rots=4, num_order = 1, reduction='sum', **kwargs):
    """
    Computes cross-entropy loss for one-hot encoded targets using raw logits.
    Args:
        outputs (torch.Tensor): Raw output logits from the model. Shape [B, C].
        targets (torch.Tensor): Ground truth one-hot vectors. Shape [B, C].
        reduction (str): 'none' | 'mean' | 'sum'
    Returns:
        torch.Tensor: The computed loss.
    """



    loss_probs_vec = F.log_softmax(out, dim=1)
    print("Log probs rot shape" , loss_probs_vec.shape, tar[0,:,num_stacks:].shape)
    print(loss_probs_vec, tar[0,:,:])
    loss_vec= -torch.sum(tar[0,:,:] * loss_probs_vec, dim=1)



    return loss_vec.sum()

                #  if (MULTI_SCALE_LOSS):
                #     out_f = 1
                #     if(x_ratio!=16):
                #         flow_dim_ = flow_dim/2
                #     else:
                #         flow_dim_ = flow_dim
                #     if (x_ratio!=1):
                #         scale_num = int(flow_dim_/out_f )
                #         flow_ = flow[:,0:3] * scale_num
                #     else:
                #         scale_num = 1
                #         flow_ = flow[:,0:3]
                #     flow_  = F.interpolate(flow_, size=[ flow.shape[2],flow.shape[3]*scale_num,flow.shape[4]*scale_num], mode='trilinear', align_corners=False)
                #     if(x_ratio<16):
                #         all_flows.append(flow_)
               

def l22_loss_grid_multiscale_real(out_list, tar, masked=True, eps=0e-4, reduction='mean', **kwargs):
    tot_loss = 0
    mult_factors = [1,1,1,1,1,1]
   # mult_factors = [1,1,1,1,1,4]
   # mult_factors = [16,16,8,4,2,1]
   # mult_factors = [32,32,8,4,2,1]
    for i in range(len(out_list)):
        out = out_list[i]

        
        batch, chans, *size = out.shape #batch should always be 1 
        batch_tar, chans_tar, *size_tar = tar.shape #batch should always be 1 
        
        
        flow_factor = size[1]/size_tar[1]
        tar = F.interpolate(tar, size=size, mode='area') 
        tar[0,3] = (tar[0,3] > 0.5).float()
     #   tar[:,0:chans] = tar[:,0:chans] * flow_factor

        new_add = torch.sum(tar[:,chans:] * (out[:,:chans] - tar[:,:chans]) ** 2)/ torch.sum(tar[:,chans:]) 
        tot_loss = tot_loss + mult_factors[i] * new_add
        print("LOSS AT SCALE ", i, new_add)
    return tot_loss/len(out_list)


def l22_loss_grid_multiscale(out_list, tar, masked=True, eps=0e-4, reduction='mean', **kwargs):
    tot_loss = 0
    for i in range(len(out_list)):
        out = out_list[i]

        
        batch, chans, *size = out.shape #batch should always be 1 
        batch_tar, chans_tar, *size_tar = tar.shape #batch should always be 1 
        
        
        flow_factor = size[1]/size_tar[1]
        tar = F.interpolate(tar, size=size, mode='area') 
        tar[0,3] = (tar[0,3] > 0.5).float()
        tar[:,0:chans] = tar[:,0:chans]* flow_factor
        new_add = torch.sum(tar[:,chans:] * (out[:,:chans] - tar[:,:chans]) ** 2)/ torch.sum(tar[:,chans:])
        tot_loss = tot_loss + new_add
        print("LOSS AT SCALE ", i, new_add)
    return tot_loss/len(out_list)

def l22_loss_grid_slice_loss(out, tar, masked=True, eps=0e-4, reduction='mean', **kwargs):
    # if out.shape[0] > 1:
    #     print("IN HERE :((()))")
    #     loss = [l22_loss_grid(out[i:i+1], tar[i:i+1], masked, eps, reduction, **kwargs) for i in range(out.shape[0])]
    #     return sum(loss) / len(loss) #torch.stack(img1, 0), torch.stack(seg1, 0)


    if isinstance(out, tuple):
        batch, chans, *size = out[0].shape #batch should always be 1
        slice_loss = out[1]
        out = out[0]
        print("IS TUPLE :)")
        batch_tar, chans_tar, *size_tar = tar.shape
    else:
        batch, chans, *size = out.shape #batch should always be 1 
        batch_tar, chans_tar, *size_tar = tar.shape #batch should always be 1 
        
    
    flow_factor = size[1]/size_tar[1]
    tar = F.interpolate(tar, size=size, mode='area') 
    tar[0,3] = (tar[0,3] > 0.5).float()
    tar[:,0:chans] = tar[:,0:chans]* flow_factor

    return torch.sum(tar[:,chans:] * (out[:,:chans] - tar[:,:chans]) ** 2)/ torch.sum(tar[:,chans:]) + slice_loss









def l22_loss_grid_crop_affine(out, tar, masked=True, eps=0e-4, reduction='mean', **kwargs):
    # if out.shape[0] > 1:
    #     print("IN HERE :((()))")
    #     loss = [l22_loss_grid(out[i:i+1], tar[i:i+1], masked, eps, reduction, **kwargs) for i in range(out.shape[0])]
    #     return sum(loss) / len(loss) #torch.stack(img1, 0), torch.stack(seg1, 0)

    if isinstance(out, tuple):
        batch, chans, *size = out[0].shape #batch should always be 1
        ALL_STACKS = out[1]
        out = out[0]
        print("IS TUPLE :)")

    else:
        print("not tuple")
        batch, chans, *size = out.shape #batch should always be 1 

    batch_tar, chans_tar, *size_tar = tar.shape #batch should always be 1 
    mask = tar[:,chans:]
    
    spacing = 1 
    self_slice=0
    detach_=False

    

    # Define grid and ones to make homog. coords
   # pdb.set_trace()
    ones = torch.ones([1] + size, dtype=out.dtype, device=out.device) 
    grid = make_grid_one(ALL_STACKS, [out.shape[3],out.shape[4]],  voxel_size=1,  slice_dims=[1,1,1], device=out.device)
 
    warp = tar[:,:chans]+ grid[:out.shape[1]]

    # Combine grid with homog ones and add to flow
    grid_ones = torch.cat([grid, ones]) # grida shape - torch.Size([4, 192, 64, 64])
    grid_gt = out + grid[:out.shape[1]] # warp torch.Size([1, 3, 192, 64, 64])
    
    
    # Flatten mask, grid, and flow
    M = mask.expand([batch, chans] + size).unflatten(2 + self_slice, [-1, spacing]).movedim(2 + self_slice, 1).flatten(3).transpose(2,3)
    A = grid_gt.expand([batch, chans] + size).unflatten(2 + self_slice, [-1, spacing]).movedim(2 + self_slice, 1).flatten(3).transpose(2,3)
    A_ones = grid_ones.expand([batch, chans + 1] + size).unflatten(2 + self_slice, [-1, spacing]).movedim(2 + self_slice, 1).flatten(3).transpose(2,3)
  #  B = warp.unflatten(2 + self_slice, [-1, spacing]).movedim(2 + self_slice, 1).flatten(3).transpose(2,3) # B shape torch.Size([1, 192, 4096, 3])    
    B = warp.expand([batch, chans] + size).unflatten(2 + self_slice, [-1, spacing]).movedim(2 + self_slice, 1).flatten(3).transpose(2,3)

    
    # set up variables
    aff_arr = torch.zeros((4,4), device=out.device)
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
            torch.save(out.detach().cpu(),'out_error.pth')
            X = torch.eye(3, device=out.device)
            t_solve = torch.tensor([[0,0,0]],device=out.device)


    
    grid_unrolled = grid_gt.reshape(3,size[0]*size[1]*size[2]).to(out.device)
    grid_og = grid[:out.shape[1]].reshape(3,size[0]*size[1]*size[2]).to(out.device)

    affine_all = ((X @ grid_unrolled  + t_solve.T) - grid_og).reshape(3,size[0],size[1],size[2])

    aff_arr = torch.eye(4, device=out.device)
    aff_arr[0:3,0:3] = X            
    aff_arr[0:3,3] = t_solve.T[:,0]


  
    out = affine_all[None] 
    flow_factor = size[1]/size_tar[1]
    
    tar = F.interpolate(tar, size=size, mode='area') 
    tar[0,3] = (tar[0,3] > 0.5).float()
    tar[:,0:chans] = tar[:,0:chans]* flow_factor
    

    
# W = (tar[:,chans:].expand(tar[:,:chans].shape).masked_fill(tar[:,chans:] == 0, eps)).reshape(batch, chans, -1).transpose(1,2)

    # torch.sum(tar[:,chans:] * (affine_all[None][:,:chans] - tar[:,:chans]) ** 2)/ torch.sum(tar[:,chans:])
    return torch.sum(tar[:,chans:] * (out[:,:chans] - tar[:,:chans]) ** 2)/ torch.sum(tar[:,chans:])



def l22_loss_grid_masked(out, tar, masked=True, eps=0e-4, reduction='mean', **kwargs):
    if out.shape[0] > 1:
        print("IN HERE :((()))")
        loss = [l22_loss_grid(out[i:i+1], tar[i:i+1], masked, eps, reduction, **kwargs) for i in range(out.shape[0])]
        return sum(loss) / len(loss) #torch.stack(img1, 0), torch.stack(seg1, 0)

    batch, chans, *size = out.shape #batch should always be 1 
    batch_tar, chans_tar, *size_tar = tar.shape #batch should always be 1 
    
    
    
    flow_factor = size[1]/size_tar[1]
    tar = F.interpolate(tar, size=size, mode='area') 
    tar[0,3] = (tar[0,3] > 0.5).float()
    tar[:,0:chans] = tar[:,0:chans]* flow_factor
    
    
   # W = (tar[:,chans:].expand(tar[:,:chans].shape).masked_fill(tar[:,chans:] == 0, eps)).reshape(batch, chans, -1).transpose(1,2)
   # pdb.set_trace()
   # return torch.sum(tar[:,chans:] * (out[:,:chans] - tar[:,:chans]) ** 2)/ torch.sum(tar[:,chans:])
    mask = (tar[:,chans:] > 0)[0,0]

    term1 = ((out[0,0][mask] - tar[0,0][mask]) ** 2)
    term2 =  ((out[0,1][mask] - tar[0,1][mask]) ** 2)
    term3 =  ((out[0,2][mask] - tar[0,2][mask]) ** 2)

    
    return torch.mean(term1 + term2 + term3)


   # return torch.sum(tar[:,chans:] * (out[:,:chans] - tar[:,:chans]) ** 2)/ torch.sum(tar[:,chans:])

def l1_loss_grid(out, tar, masked=True, eps=0e-4, reduction='mean', **kwargs):
    if out.shape[0] > 1:
        print("IN HERE :((()))")
        loss = [l22_loss_grid(out[i:i+1], tar[i:i+1], masked, eps, reduction, **kwargs) for i in range(out.shape[0])]
        return sum(loss) / len(loss) #torch.stack(img1, 0), torch.stack(seg1, 0)

    batch, chans, *size = out.shape #batch should always be 1 
    batch_tar, chans_tar, *size_tar = tar.shape #batch should always be 1 
    
    
    
    flow_factor = size[1]/size_tar[1]
    tar = F.interpolate(tar, size=size, mode='area') 
    tar[0,3] = (tar[0,3] > 0.5).float()
    tar[:,0:chans] = tar[:,0:chans]* flow_factor
    
    
   # W = (tar[:,chans:].expand(tar[:,:chans].shape).masked_fill(tar[:,chans:] == 0, eps)).reshape(batch, chans, -1).transpose(1,2)
   # pdb.set_trace()
    return torch.sum(torch.abs(tar[:,chans:] * (out[:,:chans] - tar[:,:chans])))/ torch.sum(tar[:,chans:])



def TRE_loss_grid_latest(out, tar, masked=True, eps=0e-4, reduction='mean', **kwargs): # include size based on dimension, resample truth to be the output
    if out.shape[0] > 1:
        print("IN HERE :((()))")
        loss = [TRE_loss_grid(out[i:i+1], tar[i:i+1], masked, eps, reduction, **kwargs) for i in range(out.shape[0])]
        return sum(loss) / len(loss) #torch.stack(img1, 0), torch.stack(seg1, 0)

    batch, chans, *size = out.shape #batch should always be 1 
    batch_tar, chans_tar, *size_tar = tar.shape #batch should always be 1 
    
    

    flow_factor = size[1]/size_tar[1]
    tar = F.interpolate(tar, size=size, mode='area') 
    tar[0,3] = (tar[0,3] > 0.5).float()
    tar[:,0:chans] = tar[:,0:chans]* flow_factor


    slice_shape  = out.shape[4]
    mult_factor = (128/slice_shape)*1.6

   
   # W = (tar[:,chans:].expand(tar[:,:chans].shape).masked_fill(tar[:,chans:] == 0, eps)).reshape(batch, chans, -1).transpose(1,2)

 
    return mult_factor * torch.sum(torch.norm(tar[:,chans:] * torch.abs(out[:,:chans] - tar[:,:chans]) , dim=1))/ torch.sum(tar[:,chans:])



def TRE_loss_grid(out, tar, masked=True, eps=0e-4, reduction='mean', **kwargs): # include size based on dimension, resample truth to be the output
 
    if isinstance(out, list):
            out = out[-1]

    if isinstance(out, tuple):
            out = out[0]

            
    batch, chans, *size = out.shape #batch should always be 1 
    batch_tar, chans_tar, *size_tar = tar.shape #batch should always be 1 
    
    

    flow_factor = size[1]/size_tar[1]
    tar = F.interpolate(tar, size=size, mode='area') * flow_factor


    slice_shape  = out.shape[4]
    mult_factor = (128/slice_shape)*1.6
   # W = (tar[:,chans:].expand(tar[:,:chans].shape).masked_fill(tar[:,chans:] == 0, eps)).reshape(batch, chans, -1).transpose(1,2)


    return mult_factor * torch.sum(torch.norm(tar[:,chans:] * torch.abs(out[:,:chans] - tar[:,:chans]) , dim=1))/ torch.sum(tar[:,chans:])


def TRE_loss_soft_max(out, tar, masked=True, eps=0e-4, reduction='mean', **kwargs): # include size based on dimension, resample truth to be the output
    if out.shape[0] > 1:
        print("IN HERE :((()))")
        loss = [TRE_loss_soft_max(out[i:i+1], tar[i:i+1], masked, eps, reduction, **kwargs) for i in range(out.shape[0])]
        return sum(loss) / len(loss) #torch.stack(img1, 0), torch.stack(seg1, 0)

    batch, chans, *size = out.shape #batch should always be 1 
    batch_tar, chans_tar, *size_tar = tar.shape #batch should always be 1 
    
    

    flow_factor = size[1]/size_tar[1]
    tar = F.interpolate(tar, size=size, mode='area') * flow_factor


    slice_shape  = out.shape[4]
    mult_factor = (128/slice_shape)*1.6
   # W = (tar[:,chans:].expand(tar[:,:chans].shape).masked_fill(tar[:,chans:] == 0, eps)).reshape(batch, chans, -1).transpose(1,2)

   # return mult_factor * torch.sum(torch.norm(tar[:,chans:] * torch.abs(out[:,:chans] - tar[:,:chans]) , dim=1))/ torch.sum(tar[:,chans:])
    return mult_factor * torch.quantile(torch.norm(tar[:,chans:] * torch.abs(out[:,:chans] - tar[:,:chans]) , dim=1).flatten(), 0.97)




def l22_loss_affine_invariant_3stacks_grid_no_flip(out, tar, masked=True, eps=0e-4, reduction='mean', **kwargs):
    if out.shape[0] > 1:
        print("IN HERE :((()))")
        loss = [l22_loss_affine_invariant(out[i:i+1], tar[i:i+1], masked, eps, reduction, **kwargs) for i in range(out.shape[0])]
        return sum(loss) / len(loss) #torch.stack(img1, 0), torch.stack(seg1, 0)

    batch, chans, *size = out.shape #batch should always be 1 
    #print("L22 LOSS")
    # MOVE BACK IN
    size[0] = size[0]//3
    num_stacks = 3

 #   pdb.set_trace()
    
    grid = [torch.arange(0, size[d], dtype=torch.float, device=out.device) for d in range(len(size))]
    grid = torch.stack(torch.meshgrid(grid, indexing='ij'), 0)
    
    
    grid_list = [grid] * num_stacks
    grid = torch.cat((grid_list), dim=1)
    mask = tar[:,chans:] # if masked else torch.ones_like(tar[:,chans:])


    B = (out[:,:chans] + grid).masked_select(mask.bool()).reshape(batch, chans, -1).transpose(1,2)
    A = (tar[:,:chans]+ grid).masked_select(mask.bool()).reshape(batch, chans, -1).transpose(1,2)

    mean_B = B.mean(1, keepdim=True).detach()
    mean_A = A.mean(1, keepdim=True).detach()
    X = torch.linalg.svd(torch.linalg.lstsq(A - mean_A, B - mean_B).solution.detach())

    B = (out[:,:chans] + grid).reshape(batch, chans, -1).transpose(1,2)
    A = (tar[:,:chans] + grid).reshape(batch, chans, -1).transpose(1,2)

    W = (tar[:,chans:].expand(tar[:,:chans].shape).masked_fill(tar[:,chans:] == 0, eps)).reshape(batch, chans, -1).transpose(1,2)

    print("MEAN A")
    print(mean_A)
    print(A - mean_A)
    print(out[:,:chans,0].flatten().mean().detach() )
    print(out[:,:chans,1].flatten().mean().detach() )
    print(out[:,:chans,2].flatten().mean().detach() )

    print("MEAN B")
    print(mean_B)
    print(B - mean_B)
    print(tar[:,:chans,0].flatten().mean().detach())
    print(tar[:,:chans,1].flatten().mean().detach())
    print(tar[:,:chans,2].flatten().mean().detach())
    print("AFFINE BETWEEN THEM")
    print((X.U @ X.S.sign().diag_embed() @ X.Vh))
    return torch.sum(W * ((B - mean_B) - (A - mean_A) @ (X.U @ X.S.sign().diag_embed() @ X.Vh)) ** 2) / torch.sum(W)








# work on this one
def l22_loss_affine_invariant(out, tar, masked=True, eps=0e-4, reduction='mean', **kwargs):
    if out.shape[0] > 1:
        loss = [l22_loss_affine_invariant(out[i:i+1], tar[i:i+1], masked, eps, reduction, **kwargs) for i in range(out.shape[0])]
        return sum(loss) / len(loss) #torch.stack(img1, 0), torch.stack(seg1, 0)

    batch, chans, *size = out.shape #batch should always be 1 
    grid = torch.stack(torch.meshgrid([torch.arange(1., s + 1., device=out.device) for s in size], indexing='ij'))
    mask = tar[:,chans:] # if masked else torch.ones_like(tar[:,chans:])

    B = (out[:,:chans].flip(1) + grid).masked_select(mask.bool()).reshape(batch, chans, -1).transpose(1,2)
    A = (tar[:,:chans].flip(1) + grid).masked_select(mask.bool()).reshape(batch, chans, -1).transpose(1,2)
    
    mean_B = B.mean(1, keepdim=True).detach()
    mean_A = A.mean(1, keepdim=True).detach()
    X = torch.linalg.svd(torch.linalg.lstsq(A - mean_A, B - mean_B).solution.detach())
    # if not masked:
    B = (out[:,:chans].flip(1) + grid).reshape(batch, chans, -1).transpose(1,2)
    A = (tar[:,:chans].flip(1) + grid).reshape(batch, chans, -1).transpose(1,2)
    W = (tar[:,chans:].expand(tar[:,:chans].shape).masked_fill(tar[:,chans:] == 0, eps)).reshape(batch, chans, -1).transpose(1,2)
 #   pdb.set_trace()
    return torch.sum(W * ((B - mean_B) - (A - mean_A) @ (X.U @ X.S.sign().diag_embed() @ X.Vh)) ** 2) / torch.sum(W)

# add grid

def l22_loss_affine_invariant_1stack_grid(out, tar, masked=True, eps=0e-4, reduction='mean', **kwargs):
    if out.shape[0] > 1:
        loss = [l22_loss_affine_invariant(out[i:i+1], tar[i:i+1], masked, eps, reduction, **kwargs) for i in range(out.shape[0])]
        return sum(loss) / len(loss) #torch.stack(img1, 0), torch.stack(seg1, 0)

    batch, chans, *size = out.shape #batch should always be 1 
    
    # MOVE BACK IN
    size[0] = size[0]//3
 
        #grid2 = torch.stack(torch.meshgrid([torch.arange(1., s + 1.) for s in size], indexing='ij'))
    num_stacks = 3
    grid = torch.stack(torch.meshgrid([torch.arange(1., s + 1., device=out.device) for s in size], indexing='ij'))
    grid_list = [grid] * num_stacks
    grid = torch.cat((grid_list), dim=1)
   # print("new grid")
    # MOVE BACK IN
   # grid = torch.cat((grid, grid, grid), dim=1)
  
  
  #  pdb.set_trace()
    mask = tar[:,chans:] # if masked else torch.ones_like(tar[:,chans:])


    B = (out[:,:chans].flip(1) + grid).masked_select(mask.bool()).reshape(batch, chans, -1).transpose(1,2)
    A = (tar[:,:chans].flip(1) + grid).masked_select(mask.bool()).reshape(batch, chans, -1).transpose(1,2)

    mean_B = B.mean(1, keepdim=True).detach()
    mean_A = A.mean(1, keepdim=True).detach()

    X = torch.linalg.svd(torch.linalg.lstsq(A - mean_A, B - mean_B).solution.detach())
    B = (out[:,:chans].flip(1) + grid).reshape(batch, chans, -1).transpose(1,2)
    A = (tar[:,:chans].flip(1) + grid).reshape(batch, chans, -1).transpose(1,2)
    W = (tar[:,chans:].expand(tar[:,:chans].shape).masked_fill(tar[:,chans:] == 0, eps)).reshape(batch, chans, -1).transpose(1,2)

    return torch.sum(W * ((B - mean_B) - (A - mean_A) @ (X.U @ X.S.sign().diag_embed() @ X.Vh)) ** 2) / torch.sum(W)


def l22_loss_affine_invariant_3stacks_grid_crop(out, tar, masked=True, eps=0e-4, reduction='mean', **kwargs):
    if out.shape[0] > 1:
        print("IN HERE :((()))")
        loss = [l22_loss_affine_invariant_3stacks_grid_crop(out[i:i+1], tar[i:i+1], masked, eps, reduction, **kwargs) for i in range(out.shape[0])]
        return sum(loss) / len(loss) #torch.stack(img1, 0), torch.stack(seg1, 0)

    batch, chans, *size = out.shape #batch should always be 1 
   # print("L22 LOSS")
    # MOVE BACK IN
    size[0] = size[0]//3
    num_stacks = 3
 #   pdb.set_trace()
    grid = [torch.arange(0, size[d], dtype=torch.float, device=out.device) for d in range(len(size))]
    grid = torch.stack(torch.meshgrid(grid, indexing='ij'), 0)
    
    
    grid_list = [grid] * num_stacks
    grid = torch.cat((grid_list), dim=1)
    mask = tar[:,chans:] # if masked else torch.ones_like(tar[:,chans:])


    B = (out[:,:chans] + grid).masked_select(mask.bool()).reshape(batch, chans, -1).transpose(1,2)
    A = (tar[:,:chans] + grid).masked_select(mask.bool()).reshape(batch, chans, -1).transpose(1,2)

    mean_B = B.mean(1, keepdim=True).detach()
    mean_A = A.mean(1, keepdim=True).detach()

    print("Mean A")
    print(mean_A)


    print("Mean B")
    print(mean_B)
    X = torch.linalg.svd(torch.linalg.lstsq(A - mean_A, B - mean_B).solution.detach())

    B = (out[:,:chans] + grid).reshape(batch, chans, -1).transpose(1,2)
    A = (tar[:,:chans] + grid).reshape(batch, chans, -1).transpose(1,2)

    W = (tar[:,chans:].expand(tar[:,:chans].shape).masked_fill(tar[:,chans:] == 0, eps)).reshape(batch, chans, -1).transpose(1,2)
    #pdb.set_trace()
    print("Transform")
    print((X.U @ X.S.sign().diag_embed() @ X.Vh))

    return torch.sum(W * ((B - mean_B) - (A - mean_A) @ (X.U @ X.S.sign().diag_embed() @ X.Vh)) ** 2) / torch.sum(W)



def l22_loss_affine_invariant_3stacks_grid(out, tar, masked=True, eps=0e-4, reduction='mean', **kwargs):
    if out.shape[0] > 1:
        print("IN HERE :((()))")
        loss = [l22_loss_affine_invariant(out[i:i+1], tar[i:i+1], masked, eps, reduction, **kwargs) for i in range(out.shape[0])]
        return sum(loss) / len(loss) #torch.stack(img1, 0), torch.stack(seg1, 0)

    batch, chans, *size = out.shape #batch should always be 1 
   # print("L22 LOSS")
    # MOVE BACK IN
    size[0] = size[0]//3
    num_stacks = 3
 #   pdb.set_trace()
    grid = [torch.arange(0, size[d], dtype=torch.float, device=out.device) for d in range(len(size))]
    grid = torch.stack(torch.meshgrid(grid, indexing='ij'), 0)
    
    
    grid_list = [grid] * num_stacks
    grid = torch.cat((grid_list), dim=1)
    mask = tar[:,chans:] # if masked else torch.ones_like(tar[:,chans:])


    B = (out[:,:chans].flip(1) + grid).masked_select(mask.bool()).reshape(batch, chans, -1).transpose(1,2)
    A = (tar[:,:chans].flip(1) + grid).masked_select(mask.bool()).reshape(batch, chans, -1).transpose(1,2)

    mean_B = B.mean(1, keepdim=True).detach()
    mean_A = A.mean(1, keepdim=True).detach()

    print("Mean A")
    print(mean_A)


    print("Mean B")
    print(mean_B)
    X = torch.linalg.svd(torch.linalg.lstsq(A - mean_A, B - mean_B).solution.detach())

    B = (out[:,:chans].flip(1) + grid).reshape(batch, chans, -1).transpose(1,2)
    A = (tar[:,:chans].flip(1) + grid).reshape(batch, chans, -1).transpose(1,2)

    W = (tar[:,chans:].expand(tar[:,:chans].shape).masked_fill(tar[:,chans:] == 0, eps)).reshape(batch, chans, -1).transpose(1,2)
    #pdb.set_trace()
    print("Transform")
    print((X.U @ X.S.sign().diag_embed() @ X.Vh))

    return torch.sum(W * ((B - mean_B) - (A - mean_A) @ (X.U @ X.S.sign().diag_embed() @ X.Vh)) ** 2) / torch.sum(W)



def l21_loss_affine_invariant_3stacks_grid(out, tar, masked=True, eps=0e-4, reduction='mean', **kwargs):
    if out.shape[0] > 1:
        loss = [l22_loss_affine_invariant(out[i:i+1], tar[i:i+1], masked, eps, reduction, **kwargs) for i in range(out.shape[0])]
        return sum(loss) / len(loss) #torch.stack(img1, 0), torch.stack(seg1, 0)

    batch, chans, *size = out.shape #batch should always be 1 
    
    # MOVE BACK IN
    size[0] = size[0]//3
    num_stacks = 3
    
    grid = [torch.arange(0, size[d], dtype=torch.float, device=out.device) for d in range(len(size))]
    grid = torch.stack(torch.meshgrid(grid, indexing='ij'), 0)
    
    
    grid_list = [grid] * num_stacks
    grid = torch.cat((grid_list), dim=1)
    mask = tar[:,chans:] # if masked else torch.ones_like(tar[:,chans:])


    B = (out[:,:chans].flip(1) + grid).masked_select(mask.bool()).reshape(batch, chans, -1).transpose(1,2)
    A = (tar[:,:chans].flip(1) + grid).masked_select(mask.bool()).reshape(batch, chans, -1).transpose(1,2)

    mean_B = B.mean(1, keepdim=True).detach()
    mean_A = A.mean(1, keepdim=True).detach()
    X = torch.linalg.svd(torch.linalg.lstsq(A - mean_A, B - mean_B).solution.detach())

    B = (out[:,:chans].flip(1) + grid).reshape(batch, chans, -1).transpose(1,2)
    A = (tar[:,:chans].flip(1) + grid).reshape(batch, chans, -1).transpose(1,2)

    W = (tar[:,chans:].expand(tar[:,:chans].shape).masked_fill(tar[:,chans:] == 0, eps)).reshape(batch, chans, -1).transpose(1,2)
    return torch.sum(W * torch.abs((B - mean_B) - (A - mean_A) @ (X.U @ X.S.sign().diag_embed() @ X.Vh))) / torch.sum(W)

  #  return torch.sum(torch.sqrt(W * ((B - mean_B) - (A - mean_A) @ (X.U @ X.S.sign().diag_embed() @ X.Vh)) ** 2)) / torch.sum(W)



def l21_loss_affine_invariant_3stacks_grid_old(out, tar, masked=True, eps=0e-4, reduction='mean', **kwargs):
    if out.shape[0] > 1:
        loss = [l22_loss_affine_invariant(out[i:i+1], tar[i:i+1], masked, eps, reduction, **kwargs) for i in range(out.shape[0])]
        return sum(loss) / len(loss) #torch.stack(img1, 0), torch.stack(seg1, 0)

    batch, chans, *size = out.shape #batch should always be 1 
    size[0] = size[0]//3
        #grid2 = torch.stack(torch.meshgrid([torch.arange(1., s + 1.) for s in size], indexing='ij'))

    grid = torch.stack(torch.meshgrid([torch.arange(1., s + 1., device=out.device) for s in size], indexing='ij'))
    grid = torch.cat((grid, grid, grid), dim=1)
  #  pdb.set_trace()
    mask = tar[:,chans:] # if masked else torch.ones_like(tar[:,chans:])
   # mask = torch.ones_like(tar[:,chans:]) # try to see if mask is the problem??
    

    B = (out[:,:chans].flip(1) + grid).masked_select(mask.bool()).reshape(batch, chans, -1).transpose(1,2)
    A = (tar[:,:chans].flip(1) + grid).masked_select(mask.bool()).reshape(batch, chans, -1).transpose(1,2)
   # B = (out[:,:chans] + grid).masked_select(mask.bool()).reshape(batch, chans, -1).transpose(1,2)
    #A = (tar[:,:chans] + grid).masked_select(mask.bool()).reshape(batch, chans, -1).transpose(1,2)
    mean_B = B.mean(1, keepdim=True).detach()
    mean_A = A.mean(1, keepdim=True).detach()
    X = torch.linalg.svd(torch.linalg.lstsq(A - mean_A, B - mean_B).solution.detach())
    # Solve for (A-mean_A) = (B-mean_B) @ X


    # if not masked:
   # B = (out[:,:chans].flip(1) + grid).reshape(batch, chans, -1).transpose(1,2)
   # A = (tar[:,:chans].flip(1) + grid).reshape(batch, chans, -1).transpose(1,2)

    B = (out[:,:chans].flip(1) + grid).reshape(batch, chans, -1).transpose(1,2) 
    A = (tar[:,:chans].flip(1) + grid).reshape(batch, chans, -1).transpose(1,2)
    W = (tar[:,chans:].flip(1).expand(tar[:,:chans].shape).masked_fill(tar[:,chans:] == 0, eps)).reshape(batch, chans, -1).transpose(1,2)
    
    # try canceling W
   # W = torch.ones_like(W)
    return torch.sum(W * ((B - mean_B) - (A - mean_A) @ (X.U @ X.S.sign().diag_embed() @ X.Vh)) ** 2) / torch.sum(W)



# work on this one
def l22_loss_affine_invariant_3stacks_flip(out, tar, masked=True, eps=0e-4, reduction='mean', **kwargs):
    if out.shape[0] > 1:
        loss = [l22_loss_affine_invariant(out[i:i+1], tar[i:i+1], masked, eps, reduction, **kwargs) for i in range(out.shape[0])]
        return sum(loss) / len(loss) #torch.stack(img1, 0), torch.stack(seg1, 0)

    batch, chans, *size = out.shape #batch should always be 1 
    grid = torch.stack(torch.meshgrid([torch.arange(1., s + 1., device=out.device) for s in size], indexing='ij'))
    mask = tar[:,chans:] # if masked else torch.ones_like(tar[:,chans:])

    B = (out[:,:chans].flip(1) + grid).masked_select(mask.bool()).reshape(batch, chans, -1).transpose(1,2)
    A = (tar[:,:chans].flip(1) + grid).masked_select(mask.bool()).reshape(batch, chans, -1).transpose(1,2)
    
    mean_B = B.mean(1, keepdim=True).detach()
    mean_A = A.mean(1, keepdim=True).detach()
    X = torch.linalg.svd(torch.linalg.lstsq(A - mean_A, B - mean_B).solution.detach())
    # if not masked:
    B = (out[:,:chans].flip(1) + grid).reshape(batch, chans, -1).transpose(1,2)
    A = (tar[:,:chans].flip(1) + grid).reshape(batch, chans, -1).transpose(1,2)
    W = (tar[:,chans:].expand(tar[:,:chans].shape).masked_fill(tar[:,chans:] == 0, eps)).reshape(batch, chans, -1).transpose(1,2)

    return torch.sum(W * ((B - mean_B) - (A - mean_A) @ (X.U @ X.S.sign().diag_embed() @ X.Vh)) ** 2) / torch.sum(W)

def l22_loss_affine_invariant_3stacks(out, tar, masked=True, eps=0e-4, reduction='mean', **kwargs):
    if out.shape[0] > 1:
        loss = [l22_loss_affine_invariant(out[i:i+1], tar[i:i+1], masked, eps, reduction, **kwargs) for i in range(out.shape[0])]
        return sum(loss) / len(loss) #torch.stack(img1, 0), torch.stack(seg1, 0)

    batch, chans, *size = out.shape #batch should always be 1 
    grid = torch.stack(torch.meshgrid([torch.arange(1., s + 1., device=out.device) for s in size], indexing='ij'))
    mask = tar[:,chans:] # if masked else torch.ones_like(tar[:,chans:])

    B = (out[:,:chans].flip(1) + grid).masked_select(mask.bool()).reshape(batch, chans, -1).transpose(1,2)
    A = (tar[:,:chans].flip(1) + grid).masked_select(mask.bool()).reshape(batch, chans, -1).transpose(1,2)
    
    mean_B = B.mean(1, keepdim=True).detach()
    mean_A = A.mean(1, keepdim=True).detach()
    X = torch.linalg.svd(torch.linalg.lstsq(A - mean_A, B - mean_B).solution.detach())
    # if not masked:
    #X = torch.linalg.svd(torch.linalg.lstsq(A - mean_A, A - mean_A).solution.detach())

   # print(X)
    B = (out[:,:chans].flip(1) + grid).reshape(batch, chans, -1).transpose(1,2)
    A = (tar[:,:chans].flip(1) + grid).reshape(batch, chans, -1).transpose(1,2)
    W = (tar[:,chans:].expand(tar[:,:chans].shape).masked_fill(tar[:,chans:] == 0, eps)).reshape(batch, chans, -1).transpose(1,2)

    return torch.sum(W * ((B - mean_B) - (A - mean_A) @ (X.U @ X.S.sign().diag_embed() @ X.Vh)) ** 2) / torch.sum(W)


def ap_loss_affine_invariant(out, tar, masked=True, eps=0e-4, reduction='mean', **kwargs):
    if out.shape[0] > 1:
        loss = [l22_loss_affine_invariant(out[i:i+1], tar[i:i+1], masked, eps, reduction, **kwargs) for i in range(out.shape[0])]
        return sum(loss) / len(loss) #torch.stack(img1, 0), torch.stack(seg1, 0)

    batch, chans, *size = out.shape #batch should always be 1 
    grid = torch.stack(torch.meshgrid([torch.arange(1., s + 1., device=out.device) for s in size], indexing='ij'))
    mask = tar[:,chans:] # if masked else torch.ones_like(tar[:,chans:])

    B = (out[:,:chans].flip(1) + 0.0 * grid).masked_select(mask.bool()).reshape(batch, chans, -1).transpose(1,2)
    A = (tar[:,:chans].flip(1) + 1.0 * grid).masked_select(mask.bool()).reshape(batch, chans, -1).transpose(1,2)
    
    B = B - B.mean(1, keepdim=True).detach()
    A = A - A.mean(1, keepdim=True).detach()
    X = torch.linalg.svd(torch.linalg.lstsq(A, B).solution.detach())

    return torch.mean((B - A @ (X.U @ X.S.sign().diag_embed() @ X.Vh)) ** 2)

def l21_loss_affine_invariant(out, tar, masked=True, eps=0e-4, reduction='mean', **kwargs):
    if out.shape[0] > 1:
        loss = [l21_loss_affine_invariant(out[i:i+1], tar[i:i+1], masked, eps, reduction, **kwargs) for i in range(out.shape[0])]
        return sum(loss) / len(loss) #torch.stack(img1, 0), torch.stack(seg1, 0)

    batch, chans, *size = out.shape
    grid = torch.stack(torch.meshgrid([torch.arange(1., s + 1., device=out.device) for s in size], indexing='ij'))
    mask = tar[:,chans:] if masked else torch.ones_like(tar[:,chans:])

    B = (out[:,:chans].flip(1) + grid).masked_select(mask.bool()).reshape(batch, chans, -1).transpose(1,2)
    A = (tar[:,:chans].flip(1) + grid).masked_select(mask.bool()).reshape(batch, chans, -1).transpose(1,2)
    
    B = B - B.mean(1, keepdim=True).detach()
    A = A - A.mean(1, keepdim=True).detach()
    X = torch.linalg.svd(torch.linalg.lstsq(A, B).solution.detach())

    return torch.mean(torch.sqrt(torch.sum((B - A @ (X.U @ X.S.sign().diag_embed() @ X.Vh)) ** 2, 2)))