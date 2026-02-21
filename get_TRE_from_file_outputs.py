#!/usr/bin/env python


import torch
import nibabel as nib



import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless environments
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors
# Add the directory containing the module to the path
import numpy as np
import os
np.set_printoptions(suppress=True) 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

import re
import glob
from scipy.io import loadmat
import json
import argparse
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from monai.losses import LocalNormalizedCrossCorrelationLoss
from monai.metrics import PSNRMetric




def calc_local_ncc(vol1, vol2, win_size=5, eps=1e-1):
    print("local size")
    print(vol1.shape, vol2.shape)
    if(len(vol1.shape)==4):
        loss_fn = LocalNormalizedCrossCorrelationLoss(spatial_dims=2, kernel_size=win_size)  # patch size
    else:
        loss_fn = LocalNormalizedCrossCorrelationLoss(spatial_dims=3, kernel_size=win_size)  # patch size
    vol1 = torch.tensor(vol1) #.unsqueeze(0).unsqueeze(0) #.squeeze(0).squeeze(0) #.squeeze(0) #.unsqueeze(0).unsqueeze(0)  # add batch & channel
    vol2 = torch.tensor(vol2) #).unsqueeze(0).unsqueeze(0) #. squeeze(0).squeeze(0) #.squeeze(0)#.unsqueeze(0).unsqueeze(0)

    ncc_loss = loss_fn(vol1, vol2)

    return -ncc_loss

def masked_psnr_scaled(pred, target, mask, max_val=1, eps=1e-8):
    mask = mask.float()

    # Compute masked means
    mean_pred = torch.sum(pred * mask) / (mask.sum() + eps)
    mean_target = torch.sum(target * mask) / (mask.sum() + eps)

    print("mean pred and target")
    print(mean_pred.item(), mean_target.item())
    # Scale pred to match target mean
    pred_scaled = pred * (mean_target / (mean_pred + eps))

    # Compute masked MSE
    diff2 = (pred_scaled - target) ** 2
    mse = torch.sum(diff2 * mask) / (mask.sum() + eps)

    # Compute PSNR
    psnr = 10 * torch.log10(max_val**2 / (mse + eps))

    return psnr

def masked_psnr(pred, target, mask, max_val=1, eps=1e-8):

    
    mask = mask.float() #.squeeze(-1)

    diff2 = (pred - target) ** 2
    mse = torch.sum(diff2 *mask ) / (mask.sum())
    psnr = 10 * torch.log10(max_val**2 / (mse))
    

    return psnr

def masked_ssim_crop_nifti_3d(
    pred,
    target,
    mask,
    data_range=None,
    eps=1e-8,
):

    assert pred.shape == target.shape == mask.shape, "Shape mismatch"

    # 3D mask bounding box
    print("mask shape", mask.shape)
    print("pred shape", pred.shape)
    print("target shape", target.shape)
    zs, ys, xs = torch.nonzero(mask[0,0],  as_tuple=True)
    z0, z1 = zs.min(), zs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1

    pred_crop = pred[:,:, z0:z1, y0:y1, x0:x1]
    target_crop = target[:,:, z0:z1, y0:y1, x0:x1]

    if(pred_crop.shape[-1]==1):
        pred_crop = pred_crop.squeeze_(-1)
        target_crop = target_crop.squeeze_(-1)

    ssim = StructuralSimilarityIndexMeasure(data_range=pred_crop.max() - target_crop.min())
  

    return ssim(pred_crop, target_crop)

def masked_ssim_crop_nifti_3d_scaled(
    pred,
    target,
    mask,
    data_range=None,
    eps=1e-8,
):

    assert pred.shape == target.shape == mask.shape, "Shape mismatch"

    # 3D mask bounding box
    print("mask shape", mask.shape)
    print("pred shape", pred.shape)
    print("target shape", target.shape)
    zs, ys, xs = torch.nonzero(mask[0,0],  as_tuple=True)
    z0, z1 = zs.min(), zs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1


    mean_pred = torch.sum(pred * mask) / (mask.sum() + eps)
    mean_target = torch.sum(target * mask) / (mask.sum() + eps)
    pred_scaled = pred * (mean_target / (mean_pred + eps))


    pred_crop = pred_scaled[:,:, z0:z1, y0:y1, x0:x1]
    target_crop = target[:,:, z0:z1, y0:y1, x0:x1]

    if(pred_crop.shape[-1]==1):
        pred_crop = pred_crop.squeeze_(-1)
        target_crop = target_crop.squeeze_(-1)

    ssim = StructuralSimilarityIndexMeasure(data_range=pred_crop.max() - target_crop.min())
  
    return ssim(pred_crop, target_crop)

def masked_ncc_scaled(pred, target, mask, eps=1e-8):

    # Number of valid voxels
    N = mask.sum()

    if N < 2:
        return torch.tensor(0.0, device=pred.device)

    mean_pred = torch.sum(pred * mask) / (mask.sum() + eps)
    mean_target = torch.sum(target * mask) / (mask.sum() + eps)
    pred_scaled = pred * (mean_target / (mean_pred + eps))

    # Masked means
    mx = (pred_scaled * mask).sum() / (N + eps)
    my = (target * mask).sum() / (N + eps)
    # Zero-mean inside mask
    xm = (pred_scaled - mx) * mask
    ym = (target - my) * mask

    # NCC
    num = (xm * ym).sum()
    den = torch.sqrt((xm**2).sum() * (ym**2).sum()) + eps

    return num / den

def masked_ncc(pred, target, mask, eps=1e-8):

    # Number of valid voxels
    N = mask.sum()

    if N < 2:
        return torch.tensor(0.0, device=pred.device)

    # Masked means
    mx = (pred * mask).sum() / (N + eps)
    my = (target * mask).sum() / (N + eps)
    # Zero-mean inside mask
    xm = (pred - mx) * mask
    ym = (target - my) * mask

    # NCC
    num = (xm * ym).sum()
    den = torch.sqrt((xm**2).sum() * (ym**2).sum()) + eps

    return num / den


def calc_metrics(path1, path2, mask_path, max_int_est):
    # Load NIfTI files and extract data
    nii0 = nib.load(path1).get_fdata()
    nii1 = nib.load(path2).get_fdata()
    print("ssim paths")
    print(path1)
    print(path2)
    

    print("masked!")
    if mask_path == None:
        nii0_masked = nii0
        nii1_masked = nii1
    else:
        mask = nib.load(mask_path).get_fdata()
        nii0_masked = nii0 * mask
        nii1_masked = nii1 * mask

    masked_img2 = nib.Nifti1Image(nii1_masked, nib.load(path2).affine, nib.load(path2).header)
    nib.save(masked_img2, path2)  # overwrite

    # Convert to torch tensors and reshape to [B, C, H, W]
    # For 3D images, compare slice-by-slice (e.g., take mid-slice for example)
    

    vol1 = torch.tensor(nii0_masked, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    vol2 = torch.tensor(nii1_masked, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    mask_ = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    
    # if(len(vol1.shape)==5):
    #     vol1 = vol1.squeeze(-1)
    #     vol2 = vol2.squeeze(-1)

    # Compute SSIM
    ssim = StructuralSimilarityIndexMeasure(data_range=vol1.max() - vol1.min())
  
    print("max_int_est")
    print(max_int_est)
    psnr_fun = PSNRMetric(max_val=max_int_est)


    
  #  score = ssim(vol1, vol2)
    score = masked_ssim_crop_nifti_3d_scaled(vol1, vol2, mask_)
    print("SSIM SCORE:" +str(score.item()))
    print("data_range=" + str(vol1.max() - vol1.min()))
   # psnr = psnr_fun(vol1, vol2)
    psnr = masked_psnr_scaled(vol1, vol2, mask_)


    # score_ncc_5 = calc_local_ncc(vol1, vol2,win_size=5)
    # score_ncc_11 = calc_local_ncc(vol1, vol2,win_size=11)
    # score_ncc_19 = calc_local_ncc(vol1, vol2,win_size=19)
    # score_ncc_25 = calc_local_ncc(vol1, vol2,win_size=25)

    score_ncc_5 = masked_ncc_scaled(vol1, vol2, mask_,)
    score_ncc_11 = torch.tensor([0])
    score_ncc_19 = torch.tensor([0])    
    score_ncc_25 = torch.tensor([0])


    if(vol2.min()==0 and vol2.max()==0):
        score = 0
     #   return 0, score_ncc.item()
   #return score.item(), score_ncc.item()
        return  0, score_ncc_5.item(),score_ncc_11.item(), score_ncc_19.item(), score_ncc_25.item(), 0 #, score_ncc_11.item(), score_ncc_19.item(), score_ncc_25.item()
    return  score.item(), score_ncc_5.item(), score_ncc_11.item(), score_ncc_19.item(), score_ncc_25.item(), psnr.item()




def ea_antsmat2mat_file(fname):
    """
    Convert ANTs affine transform to a full 4x4 matrix using center correction,
    following ITK's ComputeOffset() logic. Converts from RAS to LPS at the end.

    Parameters:
        afftransform (array-like): 12-element ANTs affine parameters
        m_Center (array-like): 3-element center point

    Returns:
        mat (np.ndarray): 4x4 affine matrix
    """
    aff_data = loadmat(fname)

    afftransform = aff_data['AffineTransform_double_3_3'].squeeze()
    m_Center = aff_data['fixed'].squeeze()

    afftransform = np.asarray(afftransform).flatten()
    m_Center = np.asarray(m_Center).flatten()

    # Build 3x3 rotation and 3x1 translation
    R = afftransform[:9].reshape(3, 3)
    T = afftransform[9:12]

    # Construct full 4x4 affine matrix
    mat = np.eye(4)
    mat[:3, :3] = R
    mat[:3, 3] = T

    # Compute offset using explicit loop (faithful to MATLAB)
    m_Offset = np.zeros(3)
    for i in range(3):
        m_Offset[i] = T[i] + m_Center[i]
        for j in range(3):
            m_Offset[i] = m_Offset[i] - R[i, j] * m_Center[j]

    mat[:3, 3] = m_Offset

    # Invert the matrix
    mat = np.linalg.inv(mat)

    # Convert RAS to LPS
    conversion = np.array([
        [ 1,  1, -1, -1],
        [ 1,  1, -1, -1],
        [-1, -1,  1,  1],
        [ 1,  1,  1,  1]
    ])
    mat = mat * conversion

    return mat



def plot_tensor_histogram(tensor, bins=100, title='Histogram of Tensor Values', color='skyblue'):
    """
    Plots a histogram of the values in a PyTorch tensor.

    Args:
        tensor (torch.Tensor): Input tensor of any shape.
        bins (int): Number of histogram bins.
        title (str): Title of the plot.
        color (str): Color of the histogram bars.
    """
    # Flatten the tensor to 1D
   # data_flat = tensor.view(-1).cpu().numpy()
   # data_flat = tensor.ravel()

    # Plot the histogram
    plt.hist(tensor, bins=bins, color=color, edgecolor='black')
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()



def homog_coords(H, W):
    # Create meshgrid of x and y coordinates
    x = np.arange(W)
    y = np.arange(H)
    xx, yy = np.meshgrid(x, y)

    # Flatten and create homogeneous coordinates
    xx = xx[..., np.newaxis, np.newaxis]  # shape: (H, W, 1, 1)
    yy = yy[..., np.newaxis, np.newaxis]  # shape: (H, W, 1, 1)
    zz = np.zeros_like(xx)                # z = 0 everywhere
    ones = np.ones_like(xx)               # homogeneous 1's

    homogeneous_coords = np.concatenate([xx, yy, zz, ones], axis=-1)

    # Reshape to (H*W, 4) and convert to torch tensor
   # new_g = torch.tensor(homogeneous_coords.reshape(H * W, 4), dtype=torch.float32)
    
    new_g = np.array(homogeneous_coords.reshape(H * W, 4), dtype=np.float32)

    return new_g


def extract_affines_from_folder(folder_path):
    def extract_number(filename):
        match = re.search(r'\d+', filename)
        return int(match.group()) if match else -1

    # Collect all NIfTI files and sort by number in filename
    nifti_files = sorted([
        f for f in os.listdir(folder_path)
        if ((f.endswith('.nii') or f.endswith('.nii.gz')) and not f.startswith("mask"))
    ], key=extract_number)

    # Extract affines
    affines = []
    max_intensity_all = 0
    for fname in nifti_files:
        img = nib.load(os.path.join(folder_path, fname))
        affines.append(img.affine)
        if(img.get_fdata().max()>max_intensity_all):
                max_intensity_all = img.get_fdata().max()

    # Return as (N, 4, 4) array
    return np.stack(affines, axis=0), max_intensity_all


def extract_index(fname):
    match = re.search(r'mask_(\d+)', os.path.basename(fname))
    return int(match.group(1)) if match else -1



def compute_errors(ground_truth_t, predicted_t, coord_transform, all_masks, offset=False, center=True, plot=False, max=True, clin=False):
    """
    Compute errors between predicted transformations and ground truth.

    Args:
        ground_truth_t (numpy array): ground truth transforms, shape (num_slices, 4, 4)
        coord_transform (numpy arrayr): 4x4 coordinate transformation matrix from predicted to ground truth
        predicted_t (list or array): estimated transforms, shape (num_slices, 4, 4)
        new_g (numpy array): (H*W, 4) tensor of grid points
        all_masks (list or array): (num_slices, H, W)

    Returns:
        all_means (numpy array): average of absolute mean errors across slices
    """

    H, W = all_masks.shape[1], all_masks.shape[1]
    ones = torch.ones([H * W, 1], dtype=torch.float)
    
    non_zero = 0
    all_means = 0
    all_means_list = []
    mask_t = 3600
    bad_preds = []
    if(clin):
        mask_t = 1600
    print("mask t")
    print(mask_t)
    slice_num = ground_truth_t.shape[0]
    if(center==True):
        g_t = homog_coords_center(H,W)
       
    else:
        g_t = homog_coords(H,W)
    if(offset):
        g_t[:,2] = 64
   # pdb.set_trace()
    print("SLICE NUM: "+str(slice_num))
    for i in range(slice_num):
       # pred_mine = torch.tensor(predicted_t[i], dtype=torch.float32)  # ensure tensor
        
 
        pred_mine = predicted_t[i]
       # print(coord_transform.dtype, pred_mine.dtype)
        print("pred mine:")
        print(pred_mine)



        pred_mine = coord_transform @ pred_mine

        print("pred mine transformed:")
        print(pred_mine)

        print("gt:")
        print(ground_truth_t[i])

        error_slice10 = (ground_truth_t[i] @ g_t.T) - (pred_mine @ g_t.T)

     #   print("mask shape:", all_masks[i].shape)
        mask_slice = all_masks[i].reshape(H * W)

       # vec_norm = torch.norm(error_slice10, dim=0, keepdim=True)
        vec_norm = np.linalg.norm(error_slice10, axis=0, keepdims=True) 
        vec_norm = vec_norm[0]
        vec_norm = vec_norm[mask_slice.astype(bool)]
        if(vec_norm.shape[0]>0):
            if(max):
                norms = np.max(vec_norm, axis=0)
           #     print("slice num: "+str(i))
             #   print("mean,min,max")
             #   print(np.mean(vec_norm, axis=0),np.min(vec_norm, axis=0),np.max(vec_norm, axis=0))
            else:
                norms = np.mean(vec_norm, axis=0)
        else:
            norms = 0
        norms = np.nan_to_num(np.abs(norms), nan=0.0)

        print("mask sum")
        print(mask_slice.sum())
        if(np.abs(norms)>3):
                bad_preds.append(i)
        if(mask_slice.sum()>mask_t): #was 36000
            vec_norm =  np.round(vec_norm, 3)
            print("slice num: "+str(i))
            print(norms)

            # print("truth vs mine")
            # print(ground_truth_t[i])
            # print(pred_mine)
            # print(type(vec_norm))
            if(plot):
                plot_tensor_histogram(vec_norm.tolist(), bins=10)
          #  plt.hist(vec_norm, bins=50)
           # plt.show()

            non_zero = non_zero+1
            all_means += np.abs(norms)
            all_means_list.append(np.abs(norms))
        else:
            print("excluded slice num: "+str(i))
            print(norms)


        

    all_means /= non_zero
    p95 = np.percentile(all_means_list, 95)

    print("COMPARE MEANS")
    med_max = np.median(np.array(all_means_list))
    print(all_means)

    med_max = np.median(np.array(all_means_list))
    return all_means,med_max, p95, all_means_list, bad_preds


def save_outlier_slices(bad_preds,og_slice_path,folder_est, folder_gt, img_num):
    # print()
    # print(og_slice_path)
    # print(folder_est)
    # print(folder_gt)
    print("og slice path")
    print(og_slice_path)
    print("folder est")
    print(folder_est)
    print("folder gt")
    print(folder_gt)
    folder_name = os.path.basename(og_slice_path)

    for idx in bad_preds:
        og_path = os.path.join(og_slice_path, f"{idx}.nii.gz")
        img_og = nib.load(og_path).get_fdata()       

        est_path = os.path.join(folder_est, f"{idx}.nii.gz")
        img_est = nib.load(est_path) .get_fdata()     

        gt_path = os.path.join(folder_gt, f"{idx}.nii.gz")
        img_gt = nib.load(gt_path).get_fdata()  

        # Ensure we have 2D slices (handle singleton dimension)
        print("slices shape!")
        print(img_og.shape, img_gt.shape, img_est.shape)
        def get_slice(data):
            if data.ndim == 3:
                return data[:, :, 0]  # take first slice if singleton
            return data

        slices = [get_slice(img_og), get_slice(img_gt), get_slice(img_est)]
        titles = ["Original", "Nesvor", "Mine"]

        # Plot side by side
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, sl, title in zip(axes, slices, titles):
            ax.imshow(sl.T, cmap="gray", origin="lower")
            ax.set_title(title)
            ax.axis("off")

        # Save the figure
        slice_result = f"slice_pngs_{img_num}_{folder_name}"
        out_path = os.path.join(slice_result, f"slice_{idx}.png")
        os.makedirs(slice_result, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved slice comparison -> {out_path}")


def compute_slice_sim(og_slice_path,folder_est, slice_num, max_int_est):
    # print()
    # print(og_slice_path)
    # print(folder_est)
    # print(folder_gt)

    ncc_slice_all5 = []
    ncc_slice_all11 = []
    ncc_slice_all19 = []
    ncc_slice_all25 = []

    ssim_slice_all = []
    psnr_slice_all = []
    print("COMPARING SLICE PATHS")
    print(og_slice_path)
    print(folder_est)
    for idx in range(slice_num):
        
        og_path = os.path.join(og_slice_path, f"{idx}.nii.gz")
        img_og = nib.load(og_path).get_fdata()       

        mask_path = os.path.join(folder_est, f"mask_{idx}.nii.gz")
        img_mask = nib.load(mask_path).get_fdata()

        # est_path = os.path.join(folder_est, f"{idx}.nii.gz")
        # img_est = nib.load(est_path).get_fdata()     
        est_path = os.path.join(folder_est, f"{idx}.nii.gz")

        if os.path.exists(est_path):
            img_est = nib.load(est_path).get_fdata()
        else:
            est_path = os.path.join(folder_est, f"slice{idx}.nii.gz")
            img_est = nib.load(est_path).get_fdata()


        def get_slice(data):
            if data.ndim == 3:
                return data[:, :, 0]  # take first slice if singleton
            return data
       # pdb.set_trace()
       # slice_og, slice_est = get_slice(img_og), get_slice(img_est)

        if(img_mask.sum()>1600): # was 1600 #was 36000
            print("METRICS BEFORE ASSIGN")

            ssim_1, score_ncc_5, score_ncc_11, score_ncc_19, score_ncc_25, psnr = calc_metrics(og_path, est_path, mask_path, max_int_est)
            print("IDX: "+str(idx))
            print("SSIM,, NCC, PSNR")
            print(ssim_1, score_ncc_5, psnr)
            ssim_slice_all.append(ssim_1)
            ncc_slice_all5.append(score_ncc_5)
            ncc_slice_all11.append(score_ncc_11)
            ncc_slice_all19.append(score_ncc_19)
            ncc_slice_all25.append(score_ncc_25)
            psnr_slice_all.append(psnr)

    ssim_slice_med = np.median(np.array(ssim_slice_all))
    ncc_slice_med5 = np.median(np.array(ncc_slice_all5))
    ncc_slice_med11 = np.median(np.array(ncc_slice_all11))
    ncc_slice_med19 = np.median(np.array(ncc_slice_all19))
    ncc_slice_med25 = np.median(np.array(ncc_slice_all25))


    ssim_slice_mean = np.mean(np.array(ssim_slice_all))
    ncc_slice_mean5 = np.mean(np.array(ncc_slice_all5))
    psnr_slice_mean = np.mean(np.array(psnr_slice_all))





    return ssim_slice_mean, ncc_slice_mean5, ssim_slice_med, ncc_slice_med5, ncc_slice_med11, ncc_slice_med19, ncc_slice_med25, psnr_slice_mean




def ea_antsmat2mat_file(fname):
    """
    Convert ANTs affine transform to a full 4x4 matrix using center correction,
    following ITK's ComputeOffset() logic. Converts from RAS to LPS at the end.

    Parameters:
        afftransform (array-like): 12-element ANTs affine parameters
        m_Center (array-like): 3-element center point

    Returns:
        mat (np.ndarray): 4x4 affine matrix
    """
    aff_data = loadmat(fname)

    afftransform = aff_data['AffineTransform_double_3_3'].squeeze()
    m_Center = aff_data['fixed'].squeeze()

    afftransform = np.asarray(afftransform).flatten()
    m_Center = np.asarray(m_Center).flatten()

    # Build 3x3 rotation and 3x1 translation
    R = afftransform[:9].reshape(3, 3)
    T = afftransform[9:12]

    # Construct full 4x4 affine matrix
    mat = np.eye(4)
    mat[:3, :3] = R
    mat[:3, 3] = T

    # Compute offset using explicit loop (faithful to MATLAB)
    m_Offset = np.zeros(3)
    for i in range(3):
        m_Offset[i] = T[i] + m_Center[i]
        for j in range(3):
            m_Offset[i] = m_Offset[i] - R[i, j] * m_Center[j]

    mat[:3, 3] = m_Offset

    # Invert the matrix
    mat = np.linalg.inv(mat)

    # Convert RAS to LPS
    conversion = np.array([
        [ 1,  1, -1, -1],
        [ 1,  1, -1, -1],
        [-1, -1,  1,  1],
        [ 1,  1,  1,  1]
    ])
    mat = mat * conversion

    return mat




parser = argparse.ArgumentParser(description='Get testing info')

parser.add_argument('--og_slices', type=str, required=True, help='folder with original slices that has masks')
parser.add_argument('--folder_est', type=str, required=True, help='folder with estimated slice poses')
parser.add_argument('--folder_gt', type=str, required=False, default="", help='folder with true slice poses')
parser.add_argument('--mat_file', type=str, required=False, default="", help='path with ANTS registration mat file')
parser.add_argument('--gt_file', type=str, required=False, default="", help='path with ANTS registration mat file')
parser.add_argument('--reg_file', type=str, required=False, default="", help='path with ANTS registration mat file')
parser.add_argument('--json_path', type=str, required=True, help='path where to save results')
parser.add_argument('--img_num', type=int, required=True, help='path where to save results')
parser.add_argument('--clin', type=str, required=False, default="False",help='path where to save results')




args = parser.parse_args()
og_slice_path = args.og_slices
folder_est = args.folder_est
folder_gt = args.folder_gt
mat_file = args.mat_file
gt_file = args.gt_file
reg_file = args.reg_file
json_path = args.json_path
img_num = args.img_num
clin = args.clin
svrtk_baseline = False


if(clin=="True"):
    clin=True
else:
    clin=False

print(folder_gt, folder_est)
only_slice_sim = True

if not svrtk_baseline and not only_slice_sim:
    affine_arr1, max_int_est = extract_affines_from_folder(folder_est)
    affine_arr2, max_int_gt = extract_affines_from_folder(folder_gt)

    print("folder est affines")
    print(folder_est)
    print("folder gt affines")
    print(folder_gt)

    mask_files = sorted(glob.glob(os.path.join(og_slice_path, 'mask_*.nii*')), key=extract_index)


    mask_arrays = [nib.load(f).get_fdata() for f in mask_files]
    mask_volume = np.stack(mask_arrays, axis=0)

    reg_aff =  ea_antsmat2mat_file(mat_file)

    print("reg aff")
    print(reg_aff)
    avg, med_max, p95, all_list, bad_preds = compute_errors(affine_arr2, affine_arr1, reg_aff, mask_volume[:,:,:,0], offset=False, center=False, plot=False, clin=clin)
    ssim_mean, ncc_mean, ssim_med, ncc_med, ncc_med11, ncc_med19, ncc_med25, psnr_mean = compute_slice_sim(og_slice_path,folder_est, slice_num=affine_arr1.shape[0], max_int_est=max_int_est)
    if (clin):
        save_outlier_slices(bad_preds,og_slice_path,folder_est, folder_gt, img_num)
        

if not svrtk_baseline and only_slice_sim:
    affine_arr1, max_int_est = extract_affines_from_folder(folder_est)
    ssim_mean, ncc_mean, ssim_med, ncc_med, ncc_med11, ncc_med19, ncc_med25, psnr_mean = compute_slice_sim(og_slice_path,folder_est, slice_num=affine_arr1.shape[0], max_int_est=max_int_est)

if not os.path.exists(json_path):
    with open(json_path, "w") as f:
        results={}
        json.dump(results, f, indent=2)

if os.path.exists(json_path) and os.path.getsize(json_path) > 0:
    with open(json_path, "r") as f:
        try:
            results = json.load(f)
        except json.JSONDecodeError:
            results = {}
else:
    results = {}

if( not clin):
   # gt_file_mask =  gt_file[0:-12]+"mask.nii.gz"
    gt_file_mask =  gt_file[0:-12]+"mask.nii.gz"
    print("GT FILE")
    print(gt_file)
    print("REG GILE")
    print(reg_file)
    print("GT FILE MASK")
    print(gt_file_mask)
    ssim_score, ncc_score, score_ncc_11, score_ncc_19, score_ncc_25, psnr_score = calc_metrics(gt_file, reg_file, gt_file_mask, 1)
else:
    ssim_score = 0
    ncc_score = 0
    psnr_score = 0


if svrtk_baseline or only_slice_sim:
    all_list = [0]
    avg = 0
    med_max = 0
    p95 = 0

# Step 2: Add a new subject
new_subject = f"subject_{img_num}"

results[new_subject] = {
        "mean_max": avg,
         "median_max": med_max,
        "95_per": p95,
        "ssim": ssim_score,
        "ncc": ncc_score,
        "psnr": psnr_score,
        "slice_scores": all_list,
        "slice_ssim_mean": ssim_mean,
        "slice_ncc_mean": ncc_mean,
        "slice_ssim_med": ssim_med,
        "slice_ncc_med": ncc_med,
        "slice_ncc_med11": ncc_med11,
        "slice_ncc_med19": ncc_med19,
        "slice_ncc_med25": ncc_med25,
        "slice_psnr_mean": psnr_mean
}

# Step 3: Save back to JSON
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)
print("full list")
print(all_list)
print("img num:")
print(img_num)
print("mean_score:")
print(avg)
print("ssim:")
print(ssim_score)
print("ncc:")
print(ncc_score)
print("slice ssim mean:")
print(ssim_mean)
print("slice ncc mean:")
print(ncc_mean)


