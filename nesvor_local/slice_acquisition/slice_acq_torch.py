from typing import Optional, cast, Sequence
import logging
import torch
import torch.nn.functional as F
from ..image import Volume, Slice
from ..transform import mat_transform_points
import pdb

# _cache: Dict = dict()
# _cache_id: Tuple = tuple()

BATCH_SIZE = 64


# def clean_cache(*args) -> None:
#     global _cache, _cache_id
#     del _cache
#     _cache = dict()
#     _cache_id = args
#     torch.cuda.empty_cache()


# def check_cache(transforms, vol_shape, slice_shape, psf, res_slice) -> bool:
#     global _cache, _cache_id
#     if not _cache_id:
#         return False
#     if id(transforms) != id(_cache_id[0]):
#         if transforms.shape != _cache_id[0].shape:
#             return False
#         if torch.any(transforms != _cache_id[0]):
#             return False
#     if vol_shape != _cache_id[1]:
#         return False
#     if slice_shape != _cache_id[2]:
#         return False
#     if id(psf) != id(_cache_id[3]):
#         if psf.shape != _cache_id[3].shape:
#             return False
#         if torch.any(psf != _cache_id[3]):
#             return False
#     if res_slice != _cache_id[4]:
#         return False
#     return True


def _construct_coef(
    idxs, transforms, vol_shape, slice_shape, vol_mask, slice_mask, psf, res_slice
):
    # if not check_cache(transforms, vol_shape, slice_shape, psf, res_slice):
    #    clean_cache(transforms, vol_shape, slice_shape, psf, res_slice)
    #    print("clean cache")
    # if False and idxs[0] in _cache:
    #     # print("cache")
    #     return _cache[idxs[0]].to(transforms.device)
    # else:

    slice_ids = []
    volume_ids = []
    psf_vs = []
    #print("helloooo")
    for i in range(len(idxs)):
      #  print("num slices:")
       # print(len(idxs))
        slice_id, volume_id, psf_v = _construct_slice_coef(
            i,
            transforms[idxs[i]],
            vol_shape,
            slice_shape,
            vol_mask,
            slice_mask[idxs[i]] if slice_mask is not None else None,
            psf,
            res_slice,
        )
       # pdb.set_trace()
        slice_ids.append(slice_id)
        volume_ids.append(volume_id)
        psf_vs.append(psf_v)

    slice_id = torch.cat(slice_ids)
    del slice_ids
    volume_id = torch.cat(volume_ids)
    del volume_ids
    ids = torch.stack((slice_id, volume_id), 0)
    del slice_id, volume_id
    psf_v = torch.cat(psf_vs)
    del psf_vs
    coef = torch.sparse_coo_tensor(
        ids,
        psf_v,
        [
            slice_shape[0] * slice_shape[1] * len(idxs),
            vol_shape[0] * vol_shape[1] * vol_shape[2],
        ],
    ).coalesce()
    # _cache[idxs[0]] = coef.cpu()
    return coef


def _construct_slice_coef(
    i, transform, vol_shape, slice_shape, vol_mask, slice_mask, psf, res_slice
):
    
    # how the 3D psf converted to 2D??
   # print("IN _CONSTRUCT SLICE COEF")
   # #pdb.set_trace()
    transform = transform[None]
    psf_volume = Volume(psf, psf > 0, resolution_x=1) # 3D psf geometric
    psf_xyz = psf_volume.xyz_masked_untransformed # corresponds which coordinates take care of
    psf_v = psf_volume.v_masked # psf squished in one dimension
    if slice_mask is not None:
        _slice = slice_mask
    else:
        _slice = torch.ones((1,) + slice_shape, dtype=torch.bool, device=psf.device)

    # z worked
    # all points are flat in x (all zero)
    slice_xyz = Slice(_slice, _slice, resolution_x=res_slice).xyz_masked_untransformed # transforms but keeps all points
   # pdb.set_trace()
    # transformation
    # maps to corresponding 3D coordinates
    slice_xyz = mat_transform_points(transform, slice_xyz, trans_first=True) 
  #  pdb.set_trace()
    
    # transform psf to same orientation as slice
    psf_xyz = mat_transform_points(transform, psf_xyz - transform[:, :, -1], trans_first=True)
    # this is not right way to go about it
  #  pdb.set_trace()

    shift_xyz = (
        torch.tensor(vol_shape[::-1], dtype=psf.dtype, device=psf.device) - 1
    ) / 2.0  # shift to center
  #  shift_xyz[0] = shift_xyz[0] + 0.5
   # pdb.set_trace()
   # if (psf.shape[0] % 2 == 0):
    #    shift_xyz = shift_xyz + 0.5
   # pdb.set_trace()
    #            torch.Size([3]) + torch.Size([1, 105, 3]) + torch.Size([65536, 1, 3])
    # (n_pixel, n_psf, 3) 3D coordinates of pixels and psf
    slice_xyz = shift_xyz + psf_xyz.reshape((1, -1, 3)) + slice_xyz.reshape((-1, 1, 3))

   # pdb.set_trace()
    # (n_pixel, n_psf)
    # checks if all three coordinates are in bounds, give only the index of that coordinate
    inside_mask = torch.all((slice_xyz > 0) & (slice_xyz < (shift_xyz * 2)), -1) 

   # pdb.set_trace()
    # (n_masked, 3)  # get back all the coordinates within a range
    slice_xyz = slice_xyz[inside_mask].round().long()

   # pdb.set_trace()
    # (n_masked,)
    slice_id = torch.arange(
        i * slice_shape[0] * slice_shape[1], # start from this value
        (i + 1) * slice_shape[0] * slice_shape[1], # go to this value
        dtype=torch.long,
        device=psf.device,
    )
    #pdb.set_trace()
    if slice_mask is not None:
        slice_id = slice_id.view_as(slice_mask)[slice_mask]
    slice_id = slice_id[..., None].expand(-1, psf_v.shape[0])[inside_mask] # slice identity 0 repeated 29 times, 1 ...
    psf_v = psf_v[None].expand(inside_mask.shape[0], -1)[inside_mask] # repeat inside_mask.shape[0] times
    volume_id = ( # indexing where it is in the volume
        slice_xyz[:, 0]
        + slice_xyz[:, 1] * vol_shape[2]
        + slice_xyz[:, 2] * (vol_shape[1] * vol_shape[2])
    )
   
  #  pdb.set_trace()
    return slice_id, volume_id, psf_v


# do slicing
def slice_acquisition_torch(
    transforms: torch.Tensor,
    vol: torch.Tensor,
    vol_mask: Optional[torch.Tensor],
    slices_mask: Optional[torch.Tensor],
    psf: torch.Tensor,
    slice_shape: Sequence,
    res_slice: float,
    need_weight: bool,
):
    slice_shape = tuple(slice_shape)
    global BATCH_SIZE
    if psf.numel() == 1 and need_weight == False:
      #  print("NO WEIGHT")
       
        return slice_acquisition_no_psf_torch(
            transforms, vol, vol_mask, slices_mask, slice_shape, res_slice
        )
    # print("f")
    if vol_mask is not None:
        vol = vol * vol_mask
    vol_shape = vol.shape[-3:]
    _slices = []
    _weights = []
    i = 0

    while i < transforms.shape[0]:
        succ = False
        try:
            coef = _construct_coef(
                list(range(i, min(i + BATCH_SIZE, transforms.shape[0]))),
                transforms,
                vol_shape,
                slice_shape,
                vol_mask,
                slices_mask,
                psf,
                res_slice,
            )
            #check size of coeficent 
           # pdb.set_trace()
            s = torch.mv(coef, vol.view(-1)).to_dense().reshape((-1, 1) + slice_shape)
            weight = torch.sparse.sum(coef, 1).to_dense().reshape_as(s)
            del coef
            succ = True
        except RuntimeError as e:
            if "out of memory" in str(e) and BATCH_SIZE > 0:
                logging.debug("OOM, reduce batch size")
                BATCH_SIZE = BATCH_SIZE // 2
                torch.cuda.empty_cache()
            else:
                raise e
        if succ:
            _slices.append(s)
            _weights.append(weight)
            i += BATCH_SIZE

    slices = torch.cat(_slices)
    weights = torch.cat(_weights)
    m = weights > 1e-2
    slices[m] = slices[m] / weights[m]
    if slices_mask is not None:
        slices = slices * slices_mask
    if need_weight:
        return slices, weights
    else:
        return slices


# do splatting
def slice_acquisition_adjoint_torch(
    transforms: torch.Tensor,
    psf: torch.Tensor,
    slices: torch.Tensor,
    slices_mask: Optional[torch.Tensor],
    vol_mask: Optional[torch.Tensor],
    vol_shape: Sequence,
    res_slice: float,
    equalize: bool,
):
    vol_shape = tuple(vol_shape)
    global BATCH_SIZE
    if slices_mask is not None:
        slices = slices * slices_mask
    vol = None
    weight = None
    slice_shape = slices.shape[-2:]
    #print("IN SLICE ACQUISITION TORCH")
    i = 0
   # #pdb.set_trace()
    while i < transforms.shape[0]:
        succ = False
        try:

            coef = _construct_coef(
                list(range(i, min(i + BATCH_SIZE, transforms.shape[0]))),
                transforms,
                vol_shape,
                slice_shape,
                vol_mask,
                slices_mask,
                psf,
                res_slice,
            ).t()
           # #pdb.set_trace()
            
            v = torch.mv(coef, slices[i : i + BATCH_SIZE].view(-1)) # matrix vector multiplication
            if equalize:
                w = torch.sparse.sum(coef, 1)
            del coef
            succ = True
        except RuntimeError as e:
            if "out of memory" in str(e) and BATCH_SIZE > 0:
                logging.debug("OOM, reduce batch size")
                BATCH_SIZE = BATCH_SIZE // 2
                torch.cuda.empty_cache()
            else:
                raise e
        if succ:
            if vol is None:
                vol = v
            else:
                vol += v
            if equalize:
                if weight is None:
                    weight = w
                else:
                    weight += w
            i += BATCH_SIZE
    vol = cast(torch.Tensor, vol)
    vol = vol.to_dense().reshape((1, 1) + vol_shape)
    if equalize:
        weight = cast(torch.Tensor, weight)
        weight = weight.to_dense().reshape_as(vol)
        m = weight > 1e-2
        vol[m] = vol[m] / weight[m]
    if vol_mask is not None:
        vol = vol * vol_mask
    return vol


def slice_acquisition_no_psf_torch(
    transforms: torch.Tensor,
    vol: torch.Tensor,
    vol_mask: Optional[torch.Tensor],
    slices_mask: Optional[torch.Tensor],
    slice_shape: Sequence,
    res_slice: float,
) -> torch.Tensor:
    slice_shape = tuple(slice_shape)
    device = transforms.device
    _slice = torch.ones((1,) + slice_shape, dtype=torch.bool, device=device)
    slice_xyz = Slice(_slice, _slice, resolution_x=res_slice).xyz_masked_untransformed
    # transformation
    slice_xyz = mat_transform_points(
        transforms[:, None], slice_xyz[None], trans_first=True
    ).view((transforms.shape[0], 1) + slice_shape + (3,))

    output_slices = torch.zeros_like(slice_xyz[..., 0])

    if slices_mask is not None:
        masked_xyz = slice_xyz[slices_mask]
    else:
        masked_xyz = slice_xyz

    # shape = xyz.shape[:-1]
    masked_xyz = masked_xyz / (
        (torch.tensor(vol.shape[-3:][::-1], dtype=masked_xyz.dtype, device=device) - 1)
        / 2
    )
    if vol_mask is not None:
        vol = vol * vol_mask
    masked_v = F.grid_sample(vol, masked_xyz.view(1, 1, 1, -1, 3), align_corners=True)
    if slices_mask is not None:
        output_slices[slices_mask] = masked_v
    else:
        output_slices = masked_v.reshape((transforms.shape[0], 1) + slice_shape)
    return output_slices
