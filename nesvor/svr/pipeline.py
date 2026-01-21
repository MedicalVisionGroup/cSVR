import logging
from typing import List, Optional, Tuple, cast
import torch
import numpy as np
from .registration import SliceToVolumeRegistration
from .outlier import EM, global_ncc_exclusion, local_ssim_exclusion
from .reconstruction import (
    psf_reconstruction,
    srr_update,
    simulate_slices,
    slices_scale,
    simulated_error,
)
from ..utils import DeviceType, PathType, get_PSF
from ..image import Volume, Slice, load_volume, load_mask, Stack
from ..inr.data import PointDataset
import pdb
import numpy as np
import nibabel as nib
def _initial_mask(
    slices: List[Slice],
    output_resolution: float,
    sample_mask: Optional[PathType],
    sample_orientation: Optional[PathType],
    device: DeviceType,
) -> Tuple[Volume, bool]:
    dataset = PointDataset(slices)
    mask = dataset.mask
    if sample_mask is not None:
        mask = load_mask(sample_mask, device)
    transformation = None
    if sample_orientation is not None:
        transformation = load_volume(
            sample_orientation,
            device=device,
        ).transformation
    mask = mask.resample(output_resolution, transformation)
    mask.mask = mask.image > 0
    return mask, sample_mask is None


def _check_resolution_and_shape(slices: List[Slice]) -> List[Slice]:
    res_inplane = []
    thicknesses = []
    for s in slices:
        res_inplane.append(float(s.resolution_x))
        res_inplane.append(float(s.resolution_y))
        thicknesses.append(float(s.resolution_z))

    res_s = min(res_inplane)
    s_thick = np.mean(thicknesses).item()
    slices = [s.resample((res_s, res_s, s_thick)) for s in slices]
    slices = Stack.pad_stacks(slices)

    if max(thicknesses) - min(thicknesses) > 0.001:
        logging.warning("The input data have different thicknesses!")

    return slices


def _normalize(
    stack: Stack, output_intensity_mean: float
) -> Tuple[Stack, float, float]:
    masked_v = stack.slices[stack.mask]
    mean_intensity = masked_v.mean().item()
    max_intensity = masked_v.max().item()
    min_intensity = masked_v.min().item()
  #  print("original intensities")
  #  print(mean_intensity)
    # stack.slices = stack.slices * (output_intensity_mean / mean_intensity)
    # max_intensity = max_intensity * (output_intensity_mean / mean_intensity)
    # min_intensity = min_intensity * (output_intensity_mean / mean_intensity)
    return stack, max_intensity, min_intensity


def slice_to_volume_reconstruction(
    slices: List[Slice],
    *,
    with_background: bool = False,
    output_resolution: float = 0.8,
    output_intensity_mean: float = 700,
    delta: float = 150 / 700,
    n_iter: int = 3,
    n_iter_rec: List[int] = [7, 7, 21],
    global_ncc_threshold: float = 0.5,
    local_ssim_threshold: float = 0.4,
    no_slice_robust_statistics: bool = False,
    no_pixel_robust_statistics: bool = False,
    no_global_exclusion: bool = False,
    no_local_exclusion: bool = False,
    sample_mask: Optional[PathType] = None,
    sample_orientation: Optional[PathType] = None,
    psf: str = "gaussian",
    device: DeviceType = torch.device("cpu"),
    **unused
) -> Tuple[Volume, List[Slice], List[Slice]]:
    # check data

    for s in slices:
        s.image = s.image.to(device)
        s.mask = s.mask.to(device)



    slices = _check_resolution_and_shape(slices)
    stack = Stack.cat(slices)
    slices_mask_backup = stack.mask.clone()


    # init volume
    volume, is_refine_mask = _initial_mask(
        slices,
        output_resolution,
        sample_mask,
        sample_orientation,
        device,
    )

    # data normalization
    stack, max_intensity, min_intensity = _normalize(stack, output_intensity_mean)

    # define psf
    
    psf_tensor = get_PSF(
        res_ratio=(
            stack.resolution_x / output_resolution,
            stack.resolution_y / output_resolution,
            stack.thickness / output_resolution,
        ),
        device=volume.device,
        psf_type=psf,
    )

    # outer loop
    only_splat = False
    scale_volume = True
    

    # USE THIS TO CONTROL WHETHER REFINE HAPPENS USING COMMAND-LINE ARGUMENT

    if(no_slice_robust_statistics):
        only_splat = True
        no_slice_robust_statistics = False

    if (only_splat):
        n_iter = 1

        print("Only recon, no pose estimation")
    else:
        print("Pose refinement on")

   


    for i in range(n_iter):

        logging.info("outer %d", i)
        

        # slice-to-volume registration
        if i > 0 and not only_splat:  # skip slice-to-volume registration for the first iteration
            print("estimating motion...")
            svr = SliceToVolumeRegistration(
                num_levels=3,
                num_steps=5,
                step_size=2,
                max_iter=30,
            )

            # svr = SliceToVolumeRegistration(
            #     num_levels=5,
            #     num_steps=50,
            #     step_size=2,
            #     max_iter=100,
            # )
            slices_transform, _ = svr(
                stack,
                volume,
                use_mask=True,
            )
            stack.transformation = slices_transform

        # global structual exclusion
        if i > 0 and not no_global_exclusion:
            stack.mask = slices_mask_backup.clone()
            excluded = global_ncc_exclusion(stack, volume, global_ncc_threshold)
            stack.mask[excluded] = False
        # PSF reconstruction & volume mask
     
        
        volume = psf_reconstruction(
            stack,
            volume,
            update_mask=is_refine_mask,
            use_mask=not with_background,
            psf=psf_tensor,
        )
      #  pdb.set_trace()

        # init EM
        em = EM(max_intensity, min_intensity)
        p_voxel = torch.ones_like(stack.slices)
        # super-resolution reconstruction (inner loop)
        for j in range(n_iter_rec[i]):



      #  for j in range(10):
            logging.info("inner %d", j)
            # simulate slices
            slices_sim, slices_weight = cast(
                Tuple[Stack, Stack],
                simulate_slices(
                    stack,
                    volume,
                    return_weight=True,
                    use_mask=not with_background,
                    psf=psf_tensor,
                ),
            )
            # scale
            scale = slices_scale(stack, slices_sim, slices_weight, p_voxel, True)
            # err
     #       print("YOOOOO REMOVED SCALE!")
      #      print("!!!!!!!")
 
         #   scale = torch.ones_like(scale)
          #  print("KEPT SCALE")
      
            err = simulated_error(stack, slices_sim, scale)

      
            # EM robust statistics
            if (not no_pixel_robust_statistics) or (not no_slice_robust_statistics):
                p_voxel, p_slice = em(err, slices_weight, scale, 1)
                if no_pixel_robust_statistics:  # reset p_voxel
                    p_voxel = torch.ones_like(stack.slices)
                if no_slice_robust_statistics:  # reset p_slice
                    p_slice = torch.ones_like(p_slice)
            p = p_voxel

            if not no_slice_robust_statistics:
                p = p_voxel * p_slice.view(-1, 1, 1, 1)
            # local structural exclusion
      #      print("p slice")
       #     print(p_slice)

            if not no_local_exclusion:
                p = p * local_ssim_exclusion(stack, slices_sim, local_ssim_threshold)
            # super-resolution update
            
          #  i = int(j//2)
            if(only_splat):
                i = 5

            output_intensity_mean = 1
            # print("CHANGED OUTPUT_INTENSITY MEAN")
            beta = max(0.01, 0.08 / (2**i))
            
            alpha = min(1, 0.05 / beta)
           # beta = 0
           # alpha = 1.3
        #    print(f"alpha: %.4f, beta: %.4f" % (alpha, beta))
            
            volume = srr_update(
                err,
                volume,
                p,
                alpha,
                beta,
                delta * output_intensity_mean,
                use_mask=not with_background,
                psf=psf_tensor,
                )
         #   print("VOXEL VALUE")
         #   print(volume.image[64,84,36])
           # print(volume.image[50,50,50])
            
          #  pdb.set_trace()
          #  if(j==n_iter_rec[i]-1 and False==True):
            if False == True :
                valid = (scale > 0) & (scale < 2)
                mean_val = scale[valid].mean()
                vol_save= volume.image * (1/mean_val)

                affine_mat = np.array([[0.80000001, 0., 0., -45.08745956], [0., 0.80000001, 0., -44.65007019], [0., 0., 0.80000001, -34.85554504], [0., 0., 0., 1.]])
                vol = vol_save.detach().cpu().numpy()
                vol= vol.transpose(2, 1, 0)
                nii_image = nib.Nifti1Image(vol, affine=affine_mat)  
                nib.save(nii_image, f'/data/vision/polina/users/mfirenze/svr_my_train_2024/vol_splat_j0_{i}_exp.nii.gz')

    # reconstruction finished
    # prepare outputs
    slices_sim = cast(
        Stack,
        simulate_slices(
            stack, volume, return_weight=False, use_mask=True, psf=psf_tensor
        ),
    )
    simulated_slices = stack[:]
    output_slices = slices_sim[:]

#    # pdb.set_trace()
#     print("SCALING VOLUME")
    if(scale_volume):
        valid = (scale > 0) & (scale < 2)
        mean_val = scale[valid].mean()
        volume.image = volume.image * (1/mean_val)
        # print(scale)
        # print("SCALE FACTOR:", mean_val)


        slices_sim = cast(
        Stack,
        simulate_slices(
            stack, volume, return_weight=False, use_mask=True, psf=psf_tensor
        ), )
        simulated_slices = stack[:]
        output_slices = slices_sim[:]
#  #   volume.image = volume.image *10 
   # pdb.set_trace()
    # scaled_sim_images = simulated_slices
    # for n in range(len(simulated_slices)):
    #     scaled_sim_images[n].image = simulated_slices.slices[n].image * (1/scale[n])

    # scaled_out_images = [s.image * sc for s, sc in zip(output_slices, scale)]
    # simulated_slices = scaled_sim_images
    # output_slices = scaled_out_images


    return volume, output_slices, simulated_slices
