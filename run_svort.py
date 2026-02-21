#!/usr/bin/env python3
"""
Run SVoRT reconstruction on pre-saved slices.

Usage:
    python run_svort.py [directories] --suffix [suffix] --output-dir [dir]
    python run_svort.py --dir-list [file] --suffix [suffix]
"""
import argparse
import os
import traceback
import warnings

import torch
import run_svort

warnings.filterwarnings("ignore", category=UserWarning, message="torch.meshgrid")


def main():
    parser = argparse.ArgumentParser(description="Run SVoRT reconstruction on pre-saved slices.")
    parser.add_argument("directories", nargs='*', help="List of directories to process.")
    parser.add_argument("--dir-list", default=None, help="Path to a text file with one directory per line.")
    parser.add_argument("--suffix", default="run1", help="Suffix for output files.")
    parser.add_argument("--save-folder", default=None, help="Directory containing input slices (overrides default).")
    parser.add_argument("--output-volume", default=None, help="Directory to save output volume (if different from output-dir).")
    parser.add_argument("--n-iter", type=int, default=5, help="Number of SVoRT iterations.")
    parser.add_argument("--n-iter-rec", type=int, default=3, help="Number of reconstruction iterations.")

    args = parser.parse_args()

    directories = list(args.directories)
    if args.dir_list:
        with open(args.dir_list) as f:
            file_dirs = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        directories = file_dirs + directories
    if not directories:
        parser.error("No directories specified. Provide positional arguments or --dir-list.")

    for directory in directories:
        if not os.path.exists(directory):
            print(f"Error: Directory {directory} not found.")
            continue
        if not os.path.isdir(directory):
            print(f"Error: {directory} is not a directory.")
            continue

        try:
            output_directory = os.path.join(directory, "cSVR_files")
            folder_name = os.path.basename(directory.rstrip(os.sep))

            print(f"\nRunning SVoRT on {folder_name}...")

            # Ensure slices are where we expect them
            input_slices_path = args.save_folder if args.save_folder else os.path.join(output_directory, f"{folder_name}_slices")

            # Construct output paths
            volume_dir = args.output_volume if args.output_volume else output_directory
            output_volume_path = os.path.join(volume_dir, f"{folder_name}_cSVR_gd_recon{args.suffix}.nii.gz")
            sim_slices_path = os.path.join(volume_dir, f"{folder_name}_sim_slices")
            print("sim path: ", sim_slices_path)

            # Call run_svort.svr
            run_svort.svr(
                input_slices=run_svort.load_slices(input_slices_path, device=torch.device("cuda")),
                output_volume=output_volume_path,
                simulated_slices=sim_slices_path,
                no_global_exclusion=True,
                n_iter=args.n_iter,
                n_iter_rec=args.n_iter_rec,
            )
            print(f"Finished run_svort {folder_name}")

        except Exception as e:
            print(f"Failed to process {directory}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
