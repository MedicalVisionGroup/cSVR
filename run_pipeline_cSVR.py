#!/usr/bin/env python3
"""
Process NIfTI Stacks Script

This script processes directories containing NIfTI files by standardizing them
and optionally running the cSVR pipeline on the output.

Usage:
    python get_masks.py --input-dir [directorie/sub] --output-stack-masks [directorie/sub]
    python run_pipeline_cSVR.py [directories] --suffix [suffix] --output-dir [dir] --run-cSVR
"""

import argparse
import os
import subprocess
import traceback
import pdb
import warnings

# Local imports
from nifti_utils import standerdize_stack
import torch
import run_cSVR_fast
import inr_recon
import gd_recon
import models
import os.path as path
import time
warnings.filterwarnings("ignore", category=UserWarning, message="torch.meshgrid")

def main():
    """
    Main function to process NIfTI stacks and run cSVR.
    Parses command-line arguments and iterates over the provided directories.
    """
    parser = argparse.ArgumentParser(description="Process NIfTI stacks and run cSVR.")
    parser.add_argument("directories", nargs='+', help="List of directories to process.")
    parser.add_argument("--suffix", default="run1", help="Suffix for output files.")
    parser.add_argument("--run-cSVR", action="store_true", help="Run run_cSVR.py on the output.")
    parser.add_argument("--inr-recon", action="store_true", help="Run inr_recon.py on the output slices.")
    parser.add_argument("--save-slices", action="store_true", help="Run slice saver in cSVR.")
    parser.add_argument("--clin", action="store_true", help="Clin spacing flag for slice_saver.")
    parser.add_argument("--save-folder", default=None, help="Directory to save output slices.")
    parser.add_argument("--gd-recon", action="store_true", help="Run gd_recon.py on the output slices.")
    parser.add_argument("--output-volume", default=None, help="Directory to save output volume (if different from output-dir).")
    
    args = parser.parse_args()
    

    model = None
    model_mlp = None

    if args.run_cSVR:
        print("Loading models once...")
        # LOAD MODEL LOGIC (Copied/Adapted from run_cSVR_fast.py to ensure single load)
        model1024 = True

        trainee = models.segment(model=models.flow_SNet3d2_512_multi_crop())
        if model1024:
           start = time.time()
           trainee = models.segment(model=models.flow_SNet3d2_1024_multi_crop())
           end = time.time()
           print(f"Loading svr segment time {end - start:.6f} seconds")


        start = time.time()
        trainee.load_state_dict(torch.load(path.join('./model_checkpoints', 'UNet_last.ckpt'),map_location='cuda')['state_dict'])
        end = time.time()
        print(f"Loading SVR checkpoint time {end - start:.6f} seconds")
        model = trainee.model.cuda()

        start = time.time()
        trainee_mlp = models.segment(model=models.flow_SNet3d2_1024_MLP())
        trainee_mlp.load_state_dict(torch.load(path.join('./model_checkpoints', 'MLP_last.ckpt'), map_location='cuda')['state_dict'], strict = False)


        end = time.time()
        print(f"Loading MLP segment time {end - start:.6f} seconds")
        model_mlp = trainee_mlp.model.cuda()
        print("Models loaded successfully.")


    for directory in args.directories:
        # Check if the directory exists
        if not os.path.exists(directory):
            print(f"Error: Directory {directory} not found.")
            continue
        
        if not os.path.isdir(directory):
            print(f"Error: {directory} is not a directory.")
            continue

        try:
            print(f"Processing {directory} with suffix '{args.suffix}'...")
            
            effective_output_dir =  os.path.join(directory, "cSVR_files")
            os.makedirs(effective_output_dir, exist_ok=True)


            # Standardize the stack
            # Capture the returned tensors: out_stack (init_stacks), combined_cropped (input tensor)
            init_stacks_tensor, input_tensor, slice_res = standerdize_stack(directory, suffix=args.suffix, output_dir=effective_output_dir)
            
            print(f"Finished processing {directory}")
            
            # Run cSVR if requested
            if args.run_cSVR:
                # Determine the output directory and folder name
                output_directory = effective_output_dir
                folder_name = os.path.basename(directory.rstrip(os.sep))
                
                print(f"\nRunning cSVR on {folder_name} using in-memory tensors...")
                
                save_folder_path = args.save_folder if args.save_folder else os.path.join(output_directory, f"{folder_name}_slices")

                print("save folder path: ", save_folder_path)
         
                cSVR_args = argparse.Namespace(
                    init_stack_template=None, # Not used if tensor provided
                    input_template=None,      # Not used if tensor provided
                    save_slices=True,
                    clin=args.clin,
                    save_folder=save_folder_path,
                    slice_res = slice_res,
                    suffix=args.suffix
                )
                
                try:
                    # Pass the pre-loaded models to the fast inference function
                    run_cSVR_fast.run_svr_inference(
                        model=model, 
                        model_mlp=model_mlp, 
                        args=cSVR_args, 
                        init_stacks_input=init_stacks_tensor, 
                        downsampled_input_tensor=input_tensor, 
                        output_dir=directory
                    )
                    print(f"Finished running cSVR on {folder_name}")
                except Exception as e:
                    print(f"Failed to run cSVR on {folder_name}: {e}")
                    traceback.print_exc()

            # Run INR recon if requested
            if args.inr_recon:
                print(f"\nRunning INR reconstruction on {folder_name}...")
                try:
                    output_directory = effective_output_dir
                    folder_name = os.path.basename(directory.rstrip(os.sep))
                    
                    # Ensure slices are where we expect them
                    input_slices_path = args.save_folder if args.save_folder else os.path.join(output_directory, f"{folder_name}_slices")
                    
                    # Construct output filename
                    # Note: Using the same naming convention as the bash script but without explicitly running get_masks first if not done
                    volume_dir = directory
                    output_volume_path = os.path.join(volume_dir, f"{folder_name}_cSVR_inr_recon{args.suffix}.nii.gz")
                    
                    # Call inr_recon.inr
                    # Note: passing path string as input_slices, handling internally by inr_recon.inr
                    inr_recon.inr(
                        input_slices=input_slices_path,
                        output_volume=output_volume_path
                    )
                    print(f"Finished INR reconstruction for {folder_name}")
                except Exception as e:
                    print(f"Failed to run INR reconstruction on {folder_name}: {e}")
                    traceback.print_exc()

            # Run GD recon if requested
            if args.gd_recon:
                print(f"\nRunning GD reconstruction on {folder_name}...")
                try:
                    output_directory = effective_output_dir
                    folder_name = os.path.basename(directory.rstrip(os.sep))
                    
                    # Ensure slices are where we expect them
                    input_slices_path = args.save_folder if args.save_folder else os.path.join(output_directory, f"{folder_name}_slices")
                    
                    # Construct output filename
                    volume_dir = args.output_volume if args.output_volume else output_directory
                    output_volume_path = os.path.join(volume_dir, f"{folder_name}_cSVR_gd_recon{args.suffix}.nii.gz")
                    
                    # Call gd_recon.svr
                    gd_recon.svr(
                        input_slices=gd_recon.load_slices(input_slices_path, device=torch.device("cuda")),
                        output_volume=output_volume_path,
                        no_global_exclusion=True,
                        n_iter=5,
                        n_iter_rec=3,
                    )
                    print(f"Finished GD reconstruction for {folder_name}")
                except Exception as e:
                    print(f"Failed to run GD reconstruction on {folder_name}: {e}")
                    traceback.print_exc()

        except Exception as e:
            print(f"Failed to process {directory}: {e}")
            # Print the full traceback for debugging
            traceback.print_exc()


if __name__ == "__main__":
    main()
