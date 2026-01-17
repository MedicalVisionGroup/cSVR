
import os
import sys
import numpy as np
import nibabel as nib
import torch
from scipy.ndimage import zoom
import pdb
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
    
    t = affine[0:3,3]    
    aff_m = affine[:3,:3]
    d = aff_m @ t
    affine[0:3,3] = t-d

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

def make_init_stacks(dim_stacks, device):
    sl_shape = 128
    STACK1 = og_slice_pos_pre(sl_shape, [1,1,1], 1, 0, [sl_shape,sl_shape,sl_shape], device=device)
    STACK2 = og_slice_pos_pre(sl_shape, [1,1,1], 1, 1, [sl_shape,sl_shape,sl_shape], device=device)
    STACK3 = og_slice_pos_pre(sl_shape, [1,1,1], 1, 2, [sl_shape,sl_shape,sl_shape], device=device)

    ALL_STACKS = torch.zeros((2, sl_shape*3,4,4), device=device)
    ALL_STACKS[0,0:sl_shape] =   STACK1
    ALL_STACKS[0, sl_shape:sl_shape*2] = STACK2
    ALL_STACKS[0, sl_shape*2:sl_shape*3] = STACK3

    ALL_STACKS[1,   0:sl_shape] =   STACK1
    ALL_STACKS[1, sl_shape:sl_shape*2] = STACK1
    ALL_STACKS[1, sl_shape*2:sl_shape*3] = STACK1

    half_sl = int(sl_shape/2)

    start_idx0 = int(half_sl - dim_stacks[0])
    start_idx1 = int(half_sl - dim_stacks[1])
    start_idx2 = int(half_sl - dim_stacks[2])


    part1 = ALL_STACKS[:, start_idx0:start_idx0+dim_stacks[0]*2:2]   # shape: [2, 40, 4, 4]

    part2 = ALL_STACKS[:, sl_shape + start_idx1 : sl_shape + start_idx1 + dim_stacks[1]*2:2] # shape: [2, 30, 4, 4]

    
    part3 = ALL_STACKS[:, sl_shape*2 + start_idx2 : sl_shape*2 + start_idx2 + dim_stacks[2]*2:2]


    init_stacks = torch.cat([part1, part2, part3], dim=1)  
    
    return init_stacks

def normalize_by_second_mode(img, mask, target_mean=0.5, verbose=False):
    vals, counts = np.unique(img[mask == 1], return_counts=True)

    # sort by frequency (ascending)
    sorted_idx = np.argsort(counts)
   

    if len(sorted_idx) > 1:
        second_val = vals[sorted_idx[-2]]
    
    else:
        second_val = vals[0] if len(vals) > 0 else 1.0

    scale = target_mean / second_val

    if verbose:
        print(f"first val: {vals[sorted_idx[-1]]}")
        print(f"second_val: {second_val}")
        print(f"third val: {vals[sorted_idx[-2]]}")
        print(f"scale: {scale}")

    return img * scale

def crop_around_mask_center_stack(imgs, masks, crop_size=128):

    N, H, W = imgs.shape

    # Get all foreground voxel coordinates
    coords = np.argwhere(masks > 0)

    if coords.shape[0] == 0:
        # No foreground: fallback to center crop
        center_y, center_x = H // 2, W // 2
    else:
        # Compute global mean center (ignore slice dimension)
        center_y = int(np.mean(coords[:, 1]))
        center_x = int(np.mean(coords[:, 2]))

    # Compute crop bounds (same for all slices)
    half = crop_size // 2
    y1 = max(center_y - half, 0)
    x1 = max(center_x - half, 0)
    y2 = y1 + crop_size
    x2 = x1 + crop_size

    # Clip to image dimensions
    if y2 > H:
        y1 = H - crop_size
        y2 = H
    if x2 > W:
        x1 = W - crop_size
        x2 = W

    # Apply same crop to all slices
    cropped_imgs = imgs[:, y1:y2, x1:x2]
    cropped_masks = masks[:, y1:y2, x1:x2]

    return cropped_imgs, cropped_masks

def permute_preserve_hand(arr_in):
    idx = np.argmin(arr_in.shape).item()
    if(idx==1):
        arr_in = np.transpose(arr_in, (1,2,0))
        return arr_in
    elif(idx==2):
        arr_in = np.transpose(arr_in, (2,0,1))
        return arr_in
    else:
        return arr_in

def standerdize_stack(root_dir, suffix="auto2", output_dir=None, tol=1e-3):
    
    nii_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if (filename.endswith(".nii") or filename.endswith(".nii.gz") ) and not filename.startswith("."):
                nii_files.append(os.path.join(dirpath, filename))
    
    old_files = False
    if old_files:
        
        sag_indexes = [i for i, f in enumerate(nii_files) if f.endswith("n_sag.nii")][0]    
        cor_indexes = [i for i, f in enumerate(nii_files) if f.endswith("n_cor.nii")][0]
        axi_indexes = [i for i, f in enumerate(nii_files) if f.endswith("n_axi.nii")][0]

        sag_mask_indexes = [i for i, f in enumerate(nii_files) if f.endswith("maskN_sag.nii")][0]
        cor_mask_indexes = [i for i, f in enumerate(nii_files) if f.endswith("maskN_cor.nii")][0]
        axi_mask_indexes = [i for i, f in enumerate(nii_files) if f.endswith("maskN_axi.nii")][0]
    else:
        print("Other file format")
        sag_indexes = [i for i, f in enumerate(nii_files) if "mask" not in f and f.endswith("_sag.nii")][0]    
        cor_indexes = [i for i, f in enumerate(nii_files) if "mask" not in f and f.endswith("_cor.nii")][0]
        axi_indexes = [i for i, f in enumerate(nii_files) if "mask" not in f and f.endswith("_axi.nii")][0]

        sag_mask_indexes = [i for i, f in enumerate(nii_files) if "mask" in f and f.endswith("_sag.nii")][0]
        cor_mask_indexes = [i for i, f in enumerate(nii_files) if "mask" in f and f.endswith("_cor.nii")][0]
        axi_mask_indexes = [i for i, f in enumerate(nii_files) if "mask" in f and f.endswith("_axi.nii")][0]
        
        

        
    img1_nii = nib.load(nii_files[sag_indexes])
    voxel_size1 = img1_nii.header.get_zooms()
    img1 = nib.as_closest_canonical(img1_nii).get_fdata()

    mask1 = nib.load(nii_files[sag_mask_indexes])
    mask1 = nib.as_closest_canonical(mask1).get_fdata()

    img2_nii = nib.load(nii_files[cor_indexes])
    voxel_size2 = img2_nii.header.get_zooms()
    img2 = nib.as_closest_canonical(img2_nii).get_fdata()

    mask2 = nib.load(nii_files[cor_mask_indexes])
    mask2 = nib.as_closest_canonical(mask2).get_fdata()

    img3_nii = nib.load(nii_files[axi_indexes])
    voxel_size3 = img3_nii.header.get_zooms()
    img3 = nib.as_closest_canonical(img3_nii).get_fdata()
    
    mask3 = nib.load(nii_files[axi_mask_indexes])
    mask3 = nib.as_closest_canonical(mask3).get_fdata()

    if not (abs(voxel_size1[0] - voxel_size2[0]) <= tol and abs(voxel_size1[0] - voxel_size3[0]) <= tol):
        raise ValueError(f"Inconsistent voxel sizes: {voxel_size1}, {voxel_size2}, {voxel_size3}")
    if not (abs(voxel_size1[2] - voxel_size2[2]) <= tol and abs(voxel_size1[2] - voxel_size3[2]) <= tol):
        raise ValueError(f"Inconsistent voxel sizes: {voxel_size1}, {voxel_size2}, {voxel_size3}")

    

    
    
    img1 = permute_preserve_hand(img1)
    img2 = permute_preserve_hand(img2)
    img3 = permute_preserve_hand(img3)
    
    
    mask1 = permute_preserve_hand(mask1)
    mask2 = permute_preserve_hand(mask2)
    mask3 = permute_preserve_hand(mask3)
    
    img1, mask1 = crop_around_mask_center_stack(img1, mask1)
    img2, mask2 = crop_around_mask_center_stack(img2, mask2)
    img3, mask3 =crop_around_mask_center_stack(img3, mask3) 


    
    img1 = normalize_by_second_mode(img1, mask1, verbose=False)
    img2 = normalize_by_second_mode(img2, mask2, verbose=False)
    img3 = normalize_by_second_mode(img3, mask3, verbose=False)



    # save img3 as nifty

    all_imgs = np.concatenate([img1, img2, img3], axis=0)
    all_masks = np.concatenate([mask1, mask2, mask3], axis=0)
    




    keep0 = mask1.reshape(mask1.shape[0], -1).any(axis=1).sum() #-8
 
    keep1 = mask2.reshape(mask2.shape[0], -1).any(axis=1).sum() #-6
 
    keep2 = mask3.reshape(mask3.shape[0], -1).any(axis=1).sum() #-3




    keep = all_masks.reshape(all_masks.shape[0], -1).any(axis=1)


    all_imgs = all_imgs[keep]
    all_masks = all_masks[keep]
    


    if(all_imgs.shape[0]%2==1):
        all_imgs = all_imgs[:-1]
        all_masks = all_masks[:-1]
        keep2 = keep2-1
        


    

    combined_tensor = torch.from_numpy(np.stack([all_imgs, all_masks], axis=0)).float()[None]
    all_imgs = combined_tensor[0,0]
    all_masks = combined_tensor[0,1]
    


        
    cropped_imgs, cropped_masks = all_imgs, all_masks
    
    
    
    combined_cropped = torch.stack([cropped_imgs*cropped_masks, cropped_masks])




    identity_affine = np.eye(4)
    img_nii = nib.Nifti1Image(combined_cropped[0].numpy() * combined_cropped[1].numpy(), identity_affine)

    mask_nii = nib.Nifti1Image(combined_cropped[1].numpy(),identity_affine)
        
    folder_name = os.path.basename(root_dir)
    
    if output_dir is None:
        save_dir = root_dir
    else:
        save_dir = output_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

    # Save to save_dir with suffix


    nib.save(img_nii, os.path.join(save_dir, f'{folder_name}_{suffix}.nii.gz'))
    nib.save(mask_nii, os.path.join(save_dir, f'{folder_name}_mask_{suffix}.nii.gz'))
    torch.save(combined_cropped[None], os.path.join(save_dir, f'{folder_name}_{suffix}.pt'))
    
    # Fix dim_stacks mismatch: duplicate indices to match make_init_stacks expectation if it still expects 6 (Wait, user updated make_init_stacks to take 3 things? No, user updated it to 3 stacks but indices are still separate? 
    # Let's check user's code for make_init_stacks input:
    # It seems to index dim_stacks[0], dim_stacks[1], dim_stacks[2]. The loop for parts uses 0, 1, 2.
    # Ah, the user's latest make_init_stacks update has `start_idx0`, `start_idx1`, `start_idx2` and creates `part1`, `part2`, `part3`.
    # It uses `dim_stacks[0]`, `dim_stacks[1]`, `dim_stacks[2]`.
    # So we only need 3 values in the list.
    
    out_stack = make_init_stacks([keep0,keep1,keep2], device=combined_tensor.device)

    
    torch.save(out_stack, os.path.join(save_dir, f'init_stack_{folder_name}_{suffix}.pt'))

    return out_stack, combined_cropped[None], voxel_size1
