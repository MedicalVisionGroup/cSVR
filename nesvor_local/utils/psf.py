from typing import List, Tuple, Optional, Callable, Union
import torch
from math import log, sqrt
from .types import DeviceType
import pdb

# Full width have maximum (width of kernel) in the through-plane direction
GAUSSIAN_FWHM = ( 1 / (2 * sqrt(2 * log(2)))) # divide by a constant to get smaller PSF

#GAUSSIAN_FWHM = GAUSSIAN_FWHM/4

# print("CHANGED PSFFFFF")
# GAUSSIAN_FWHM = GAUSSIAN_FWHM/2
# Full width have maximum (width of kernel) in the in-plane direction 
SINC_FWHM = (1.206709128803223 * GAUSSIAN_FWHM ) # make equal to gaussian if want isotropic



def resolution2sigma(rx, ry=None, rz=None, /, isotropic=False):
# get the sigma depending on input parameters
    if isotropic:
        fx = fy = fz = GAUSSIAN_FWHM
    else:
        fx = fy = SINC_FWHM
        fz = GAUSSIAN_FWHM
    assert not ((ry is None) ^ (rz is None))
    if ry is None: # infer ry, rz from rx if not given
        if isinstance(rx, float) or isinstance(rx, int):
            if isotropic:
                return fx * rx
            else:
                return fx * rx, fy * rx, fz * rx
        elif isinstance(rx, torch.Tensor):
            if isotropic:
                return fx * rx
            else:
                assert rx.shape[-1] == 3
                return rx * torch.tensor([fx, fy, fz], dtype=rx.dtype, device=rx.device)
        elif isinstance(rx, List) or isinstance(rx, Tuple):
            assert len(rx) == 3
            return resolution2sigma(rx[0], rx[1], rx[2], isotropic=isotropic)
        else:
            raise Exception(str(type(rx)))
    else:
        # this function is essentially just this line
        return fx * rx, fy * ry, fz * rz 



def get_PSF(
    r_max: Optional[int] = None,
    res_ratio: Tuple[float, float, float] = (1, 1, 3),
   # threshold: float = 1e-3,
    threshold: float = 1e-4, #4
    device: DeviceType = torch.device("cpu"),
    psf_type: str = "gaussian", #hard codee to change to another PSF to experiment 
) -> torch.Tensor:
   # print("RES RATIO")
   # print(res_ratio)
    sigma_x, sigma_y, sigma_z = resolution2sigma(res_ratio, isotropic=False)
   # print(GAUSSIAN_FWHM)
   # print(sigma_x, sigma_y, sigma_z)
    if r_max is None:
        r_max = max(int(2 * r + 1) for r in (sigma_x, sigma_y, sigma_z)) #SAME AS CEILING 
        r_max = max(r_max, 4)

    x = torch.linspace(-r_max, r_max, 2 * r_max + 1, dtype=torch.float32, device=device)
    grid_z, grid_y, grid_x = torch.meshgrid(x, x, x, indexing="ij")

    if psf_type == "gaussian":
       # print("GAUSSIAN!!")
        # sigma_x = sigma_x/1

        # sigma_y = sigma_y/1
        # sigma_z = sigma_z/2

        psf = torch.exp(
            -0.5
            * (
                grid_x**2 / sigma_x**2
                + grid_y**2 / sigma_y**2
                + grid_z**2 / sigma_z**2
            )
        )
    elif psf_type == "sinc": # this is actually implemented as sinc(x)^2

        psf = torch.sinc(
            torch.sqrt((grid_x / res_ratio[0]) ** 2 + (grid_y / res_ratio[1]) ** 2)
        ) ** 2 * torch.exp(-0.5 * grid_z**2 / sigma_z**2)

    elif psf_type == "sinc_neg": # implements sinc()

        res_ratio_0 = res_ratio[0]/1 #/0.8
        res_ratio_1 = res_ratio[1]/1 #/0.8


        sigma_g = 2
        psf = torch.sinc(
            torch.sqrt((grid_x / (res_ratio_0)) ** 2 + (grid_y / (res_ratio_1)) ** 2)
        ) * torch.exp(-0.5 * grid_z**2 / sigma_z**2) * torch.exp(-0.5*((grid_x )**2 / (sigma_g**2) + (grid_y )**2 / (sigma_g**2)))

    elif psf_type == 'smooth_box': # estimates box convolved with gaussain (no sharp edges)
        wz = (res_ratio[2]-1) #with of box
        tz = 0.1 # drop-off smoothness
        psf =  ((torch.exp(-0.5*((grid_x )**2 / (sigma_x**2) + (grid_y )**2 / (sigma_y**2))) *\
            (1 / (1 + torch.exp(-(grid_z + wz/2) / tz)) - 1 / (1 + torch.exp(-(grid_z - wz/2) / tz)))))
    
    elif psf_type == "sharp_box": 
        width = res_ratio[2]-1 # can vary this
        psf = (torch.exp(-0.5*((grid_x )**2 / (sigma_x**2) + (grid_y )**2 / (sigma_y**2))) *\
            torch.where((grid_z >=  - width / 2) & (grid_z <=  + width / 2), 1.0, 0.0))
    
    elif psf_type == "z_box": # PSF only through plane
        width = res_ratio[2] 
       # psf = torch.tensor([[[0.2]], [[0.2]], [[0.2]], [[0.2]], [[0.2]]]).to(device)
        psf = torch.tensor([[[0.25]], [[0.25]], [[0.25]], [[0.25]]]).to(device)
      #  psf = torch.tensor([[[0.333]], [[0.333]], [[0.333]]]).to(device)
      #  psf = torch.tensor([[[1]]]).to(device)


    # truncates PSF so that small values are cropeed
    psf[psf.abs() < threshold] = 0

    rx = int(torch.nonzero(psf.sum((0, 1)) > 0)[0, 0].item()) # if all entries are zero
    ry = int(torch.nonzero(psf.sum((0, 2)) > 0)[0, 0].item())
    rz = int(torch.nonzero(psf.sum((1, 2)) > 0)[0, 0].item())

  #  do_even = False
  #  if (res_ratio[2]% 2 ==0 and do_even ==True ):
   #     psf = psf[
   #         rz : 2 * r_max  - rz, ry : 2 * r_max  - ry, rx : 2 * r_max - rx
   #     ].contiguous()
    
    #Assumes odd PSF size (eg 5x5 not 4ßx4)
    psf = psf[rz : 2 * r_max + 1 - rz, ry : 2 * r_max + 1 - ry, rx : 2 * r_max + 1 - rx].contiguous()

    psf = psf / psf.sum() # normalizes to sum to 1
    


    return psf


