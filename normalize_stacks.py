"""
Normalize NIfTI stacks in a directory using intensity normalization.

Usage:
    python normalize_stacks.py <root_dir> [<root_dir2> ...]
    python normalize_stacks.py --dir-list <file.txt> [--data-dir <base_dir>]

For each directory, finds _sag.nii, _cor.nii, _axi.nii stacks and their
corresponding masks, normalizes each by second mode intensity, and saves
the result as <original_name>_norm.nii.gz.
"""
import os
import sys
import argparse
import numpy as np
import nibabel as nib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nifti_utils import normalize_by_second_mode


def normalize_stack(root_dir):
    nii_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if (filename.endswith(".nii") or filename.endswith(".nii.gz")) and not filename.startswith("."):
                nii_files.append(os.path.join(dirpath, filename))

    sag_path   = [f for f in nii_files if "mask" not in os.path.basename(f) and f.endswith("_sag.nii")][0]
    cor_path   = [f for f in nii_files if "mask" not in os.path.basename(f) and f.endswith("_cor.nii")][0]
    axi_path   = [f for f in nii_files if "mask" not in os.path.basename(f) and f.endswith("_axi.nii")][0]

    mask_sag_path = [f for f in nii_files if "mask" in os.path.basename(f) and f.endswith("_sag.nii")][0]
    mask_cor_path = [f for f in nii_files if "mask" in os.path.basename(f) and f.endswith("_cor.nii")][0]
    mask_axi_path = [f for f in nii_files if "mask" in os.path.basename(f) and f.endswith("_axi.nii")][0]

    stacks = [
        (sag_path, mask_sag_path, "sag"),
        (cor_path, mask_cor_path, "cor"),
        (axi_path, mask_axi_path, "axi"),
    ]

    for img_path, mask_path, label in stacks:
        img_nii = nib.load(img_path)
        img = img_nii.get_fdata()
        mask = nib.load(mask_path).get_fdata()

        img_norm = normalize_by_second_mode(img, mask, verbose=False)

        base = os.path.basename(img_path)          # e.g. MAP-C581_sag.nii
        stem = base[:-4]                           # strip .nii
        out_name = stem + "_norm.nii.gz"
        out_path = os.path.join(os.path.dirname(img_path), out_name)

        nib.save(nib.Nifti1Image(img_norm, affine=img_nii.affine), out_path)
        print(f"  [{label}] saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Normalize NIfTI stacks by second-mode intensity.")
    parser.add_argument("directories", nargs="*", help="Directories containing stacks to normalize.")
    parser.add_argument("--dir-list", default=None, help="Path to a .txt file with one directory name per line.")
    parser.add_argument("--data-dir", default=None, help="Base directory prepended to each entry in --dir-list.")
    args = parser.parse_args()

    directories = list(args.directories)
    if args.dir_list:
        with open(args.dir_list) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if args.data_dir:
                    line = os.path.join(args.data_dir, line)
                directories.append(line)

    if not directories:
        parser.error("No directories specified. Provide positional arguments or --dir-list.")

    for d in directories:
        if not os.path.isdir(d):
            print(f"Skipping {d}: not a directory.")
            continue
        print(f"Normalizing {d} ...")
        try:
            normalize_stack(d)
        except Exception as e:
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()
