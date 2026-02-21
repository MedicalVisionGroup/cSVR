#!/usr/bin/env python3

import argparse
import nibabel as nib
import numpy as np


def apply_mask(image_nii_path, mask_nii_path, output_nii_path):
    # Load image and mask
    img_nii = nib.load(image_nii_path)
    img_nii = nib.as_closest_canonical(img_nii)
    mask_nii = nib.load(mask_nii_path)
    mask_nii = nib.as_closest_canonical(mask_nii)

    img_data = img_nii.get_fdata()
    mask_data = mask_nii.get_fdata()

    # Sanity check
    if img_data.shape != mask_data.shape:
        raise ValueError(
            f"Shape mismatch: image {img_data.shape}, mask {mask_data.shape}"
        )

    # Multiply image by mask
    masked_data = img_data * mask_data

    # Save with original image affine + header
    masked_nii = nib.Nifti1Image(
        masked_data,
        affine=img_nii.affine
    )
    masked_nii.set_data_dtype(img_nii.get_data_dtype())

    nib.save(masked_nii, output_nii_path)


def main():
    parser = argparse.ArgumentParser(
        description="Multiply a NIfTI image by a NIfTI mask and save with original affine."
    )
    parser.add_argument("image", help="Path to input image NIfTI (.nii or .nii.gz)")
    parser.add_argument("mask", help="Path to mask NIfTI (.nii or .nii.gz)")
    parser.add_argument("output", help="Path to output masked NIfTI")

    parser.add_argument(
        "--binarize-mask",
        action="store_true",
        help="Binarize mask before applying (mask > 0)"
    )

    args = parser.parse_args()

    # Load data
    img_nii_og = nib.load(args.image)
    img_nii = nib.as_closest_canonical(img_nii_og)
    mask_nii = nib.load(args.mask)
    mask_nii = nib.as_closest_canonical(mask_nii)

    img_data = img_nii.get_fdata()
    mask_data = mask_nii.get_fdata()

    if img_data.shape != mask_data.shape:
        raise ValueError(
            f"Shape mismatch: image {img_data.shape}, mask {mask_data.shape}"
        )

    if args.binarize_mask:
        mask_data = (mask_data > 0).astype(img_data.dtype)

    masked_data = img_data * mask_data

    masked_nii = nib.Nifti1Image(
        masked_data,
        affine=img_nii.affine
    )


  
    masked_nii.set_data_dtype(img_nii.get_data_dtype())

    nib.save(masked_nii, args.output)
    print(f"Saved masked image to {args.output}")


if __name__ == "__main__":
    main()
