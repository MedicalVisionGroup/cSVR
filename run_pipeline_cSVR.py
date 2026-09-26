#!/usr/bin/env python3
"""
cSVR pipeline: preprocessing -> learned slice pose estimation -> volume reconstruction.

Usage:
    python run_pipeline_cSVR.py SUBJECT_DIR [SUBJECT_DIR ...] --suffix run1 --run-cSVR --preprocess --gd-recon
    python run_pipeline_cSVR.py --dir-list subjects.txt --suffix run1 --run-cSVR --preprocess --gd-recon
    python run_pipeline_cSVR.py --help

See README.md for the input layout, the checkpoint files (CSVR_CHECKPOINT_DIR) and the
outputs. run_pipeline_cSVR_serve.py runs the same pipeline with a serve mode that keeps
the models loaded between subjects.
"""
import argparse
import copy
import os
import datasets
import subprocess
import traceback
import pdb
import warnings

# Local imports
from nifti_utils import standerdize_stack, standerdize_stack_synth, standerdize_stack_from_arrays
from preprocess_in_memory import preprocess_subject_in_memory
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
    parser.add_argument("directories", nargs='*', help="List of directories to process.")
    parser.add_argument("--dir-list", default=None, help="Path to a text file with one directory per line.")
    parser.add_argument("--suffix", default="run1", help="Suffix for output files.")
    parser.add_argument("--run-cSVR", action="store_true", help="Run run_cSVR.py on the output.")
    parser.add_argument("--inr-recon", action="store_true", help="Run inr_recon.py on the output slices.")
    parser.add_argument("--save-slices", action="store_true", help="Run slice saver in cSVR.")
    parser.add_argument("--clin", action="store_true", help="Clin spacing flag for slice_saver.")
    parser.add_argument("--synth", action="store_true", help="Synth data flag passed to standerdize_stack.")
    parser.add_argument("--save-folder", default=None, help="Directory to save output slices.")
    parser.add_argument("--gd-recon", action="store_true", help="Run gd_recon.py on the output slices.")
    parser.add_argument("--output-volume", default=None, help="Directory to save output volume (if different from output-dir).")
    parser.add_argument("--dest-folder", default=None, help="Folder name (relative to subject dir) to save reconstruction outputs instead of cSVR_files.")
    parser.add_argument("--no-mlp", action="store_true", help="Take the stack order from the filenames instead of the MLP: (sag, cor, axi) when named with _sag/_cor/_axi suffixes, else sorted filename order, which must then already be sagittal, coronal, axial. The MLP still decides the 180° in-plane flip of each stack.")
    parser.add_argument("--normalize-mean", action="store_true", help="Normalize stacks by mean instead of second mode.")
    parser.add_argument("--no-trim-odd", action="store_true", help="Keep all slices (don't crop the odd slice from any stack).")
    parser.add_argument("--bias-field-norm", action="store_true", help="Read the N4-corrected *_sag_bias_field/_cor_bias_field/_axi_bias_field stacks instead of the raw _sag/_cor/_axi files.")
    parser.add_argument("--save-slices-to-disk", action="store_true", help="Also write cSVR slices to disk as .nii.gz (default: keep slices in memory and hand them straight to the reconstruction).")
    parser.add_argument("--preprocess", action="store_true", help="Run masking + reorient + bias-field + normalize in-memory from the subject's three raw stacks (single env, no intermediate files), instead of reading pre-processed *_re_bias_field_norm files.")

    args = parser.parse_args()

    directories = list(args.directories)
    if args.dir_list:
        with open(args.dir_list) as f:
            file_dirs = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        directories = file_dirs + directories
    if not directories:
        parser.error("No directories specified. Provide positional arguments or --dir-list.")

    model = None
    model_mlp = None

    if args.run_cSVR:
        print("Loading models once...")
        # Checkpoints: $CSVR_CHECKPOINT_DIR (default <repo>/model_checkpoints) must hold
        #   cSVR_SVR.ckpt  -- multi-scale SVR network (models.flow_SNet3d2_1024_multi_crop)
        #   cSVR_MLP.ckpt  -- stack-ordering MLP    (models.flow_SNet3d2_1024_MLP)
        ckpt_dir = os.environ.get(
            "CSVR_CHECKPOINT_DIR",
            path.join(path.dirname(path.abspath(__file__)), "model_checkpoints"))
        svr_ckpt = path.join(ckpt_dir, "cSVR_SVR.ckpt")
        mlp_ckpt = path.join(ckpt_dir, "cSVR_MLP.ckpt")
        needed = [svr_ckpt, mlp_ckpt]
        missing = [p for p in needed if not path.exists(p)]
        if missing:
            raise SystemExit(
                "Missing checkpoint(s): " + ", ".join(missing)
                + "\nRun ./download_checkpoints.sh to fetch them, or point"
                " CSVR_CHECKPOINT_DIR at a directory that has them.")

        start = time.time()
        trainee = models.segment(model=models.flow_SNet3d2_1024_multi_crop())
        print(f"Loading svr segment time {time.time() - start:.6f} seconds")

        start = time.time()
        trainee.load_state_dict(torch.load(svr_ckpt, map_location='cuda')['state_dict'])
        print(f"Loading SVR checkpoint time {time.time() - start:.6f} seconds")
        model = trainee.model.cuda()

        # The MLP is loaded even with --no-mlp: that flag only replaces its stack-order
        # decision by the filenames, the 180° in-plane flips still come from the MLP.
        start = time.time()
        trainee_mlp = models.segment(model=models.flow_SNet3d2_1024_MLP())
        trainee_mlp.load_state_dict(torch.load(mlp_ckpt, map_location='cuda')['state_dict'], strict=False)
        print(f"Loading MLP segment time {time.time() - start:.6f} seconds")
        model_mlp = trainee_mlp.model.cuda()
        if args.no_mlp:
            print("--no-mlp: stack order from filenames; MLP kept for the 180° flips.")
        print("Models loaded successfully.")


    for directory in directories:
        # Check if the directory exists
        if not os.path.exists(directory):
            print(f"Error: Directory {directory} not found.")
            continue
        
        if not os.path.isdir(directory):
            print(f"Error: {directory} is not a directory.")
            continue

        try:
            print(f"Processing {directory} with suffix '{args.suffix}'...")
            
            if args.dest_folder:
                effective_output_dir = os.path.join(directory, args.dest_folder)
            elif args.inr_recon:
                effective_output_dir = os.path.join(directory, "cSVR_files_inr")
            else:
                effective_output_dir = os.path.join(directory, "cSVR_files")
            os.makedirs(effective_output_dir, exist_ok=True)

            # Used by the cSVR / INR / GD blocks below; define once so those
            # blocks don't depend on --run-cSVR having run first.
            output_directory = effective_output_dir
            folder_name = os.path.basename(directory.rstrip(os.sep))


            # Standardize the stack
            # Capture the returned tensors: out_stack (init_stacks), combined_cropped (input tensor)
            torch.cuda.synchronize()
            preprocess_start = time.time()
            if args.preprocess:
                # In-memory masking + reorient + bias + normalize from the raw stacks
                # (no intermediate *_re_bias_field_norm files). Validated bit-identical to
                # the disk preprocessing pipeline.
                print(f"Preprocessing {directory} in-memory (mask + bias + normalize)...")
                pp1, pp2, pp3, pm1, pm2, pm3, pp_vox = preprocess_subject_in_memory(
                    directory, torch.device("cuda"))
                init_stacks_tensor, input_tensor, slice_res = standerdize_stack_from_arrays(
                    pp1, pp2, pp3, pm1, pm2, pm3, pp_vox, directory, suffix=args.suffix,
                    output_dir=effective_output_dir, synth=args.synth,
                    normalize_mean=args.normalize_mean, trim_odd=not args.no_trim_odd,
                    save_outputs=True)
            else:
                init_stacks_tensor, input_tensor, slice_res = standerdize_stack(directory, suffix=args.suffix, output_dir=effective_output_dir, synth=args.synth, normalize_mean=args.normalize_mean, trim_odd=not args.no_trim_odd, bias_field_norm=args.bias_field_norm)
            torch.cuda.synchronize()
            print(f"[TIMING] preprocessing for {folder_name}: {time.time() - preprocess_start:.2f} s")
            if(args.synth):
                init_stacks_tensor_hres, input_tensor_hres, slice_res_hres = standerdize_stack_synth(directory, suffix=args.suffix, output_dir=effective_output_dir,synth=args.synth)
                print("INPUT TENSOR SHAPE")
                print(input_tensor_hres.shape)
            print(f"Finished processing {directory}")
            
            # Run cSVR if requested
            csvr_start_time = None
            csvr_slices = None
            if args.run_cSVR:
                # Determine the output directory and folder name
                output_directory = effective_output_dir
                folder_name = os.path.basename(directory.rstrip(os.sep))

                print(f"\nRunning cSVR on {folder_name} using in-memory tensors...")

                # Warm up the models once, BEFORE starting the timer, so the one-time ~15 s
                # GPU/extension init is not counted in any subject's total. Self-gates after
                # the first call, so this is a no-op for every subsequent subject.
                run_cSVR_fast.warmup(model, model_mlp, init_stacks_tensor, input_tensor)

                csvr_start_time = time.time()

                save_folder_path = args.save_folder if args.save_folder else os.path.join(output_directory, f"{folder_name}_slices")

                print("save folder path: ", save_folder_path)
         
                cSVR_args = argparse.Namespace(
                    init_stack_template=None, # Not used if tensor provided
                    input_template=None,      # Not used if tensor provided
                    save_slices=True,
                    save_slices_to_disk=args.save_slices_to_disk,
                    clin=args.clin,
                    synth=args.synth,
                    save_folder=save_folder_path,
                    slice_res = slice_res,
                    suffix=args.suffix,
                    no_mlp=args.no_mlp,
                )

                try:
                    # Pass the pre-loaded models to the fast inference function
                    torch.cuda.synchronize()
                    csvr_phase_start = time.time()
                    csvr_slices = run_cSVR_fast.run_svr_inference(
                        model=model,
                        model_mlp=model_mlp,
                        args=cSVR_args,
                        init_stacks_input=init_stacks_tensor,
                        downsampled_input_tensor=input_tensor,
                        output_dir=directory,
                        downsampled_input_tensor_hres=input_tensor_hres if args.synth else None
                    )
                    torch.cuda.synchronize()
                    print(f"[TIMING] cSVR inference for {folder_name}: {time.time() - csvr_phase_start:.2f} s")
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
                    volume_dir = args.output_volume if args.output_volume else output_directory
                    output_volume_path = os.path.join(volume_dir, f"{folder_name}_cSVR_inr_recon{args.suffix}.nii.gz")
                    sim_slices_path = os.path.join(volume_dir, f"{folder_name}_sim_slices")

                    # Prefer in-memory slices from cSVR; fall back to reading from disk.
                    # If GD also runs, deepcopy so it still gets a pristine copy.
                    if csvr_slices is not None:
                        inr_input_slices = copy.deepcopy(csvr_slices) if args.gd_recon else csvr_slices
                        print(f"Using {len(inr_input_slices)} in-memory slices for INR (skipping disk reload)")
                    else:
                        inr_input_slices = input_slices_path

                    # Call inr_recon.inr
                    torch.cuda.synchronize()
                    inr_phase_start = time.time()
                    inr_recon.inr(
                        input_slices=inr_input_slices,
                        simulated_slices= sim_slices_path,
                        output_volume=output_volume_path
                    )
                    torch.cuda.synchronize()
                    print(f"[TIMING] INR reconstruction for {folder_name}: {time.time() - inr_phase_start:.2f} s")
                    print(f"Finished INR reconstruction for {folder_name}")
                    if csvr_start_time is not None:
                        print(f"Total time (cSVR start -> reconstruction finish) for {folder_name}: {time.time() - csvr_start_time:.2f} seconds")
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
                    sim_slices_path = os.path.join(volume_dir, f"{folder_name}_sim_slices")
                    print("sim path: ", sim_slices_path)

                    # Call gd_recon.svr
                    torch.cuda.synchronize()
                    gd_load_start = time.time()
                    # Prefer in-memory slices from cSVR; fall back to reading from disk.
                    if csvr_slices is not None:
                        gd_input_slices = csvr_slices
                        print(f"Using {len(gd_input_slices)} in-memory slices for GD (skipping disk reload)")
                    else:
                        gd_input_slices = gd_recon.load_slices(input_slices_path, device=torch.device("cuda"))
                    torch.cuda.synchronize()
                    print(f"[TIMING] GD load_slices for {folder_name}: {time.time() - gd_load_start:.2f} s")
                    gd_recon_start = time.time()
                    gd_recon.svr(
                        input_slices=gd_input_slices,
                        output_volume=output_volume_path,
                        simulated_slices= sim_slices_path,
                        no_global_exclusion=True,
                        n_iter=5,
                        n_iter_rec=3,
                        output_resolution=0.8,
                    )
                    torch.cuda.synchronize()
                    print(f"[TIMING] GD reconstruction for {folder_name}: {time.time() - gd_recon_start:.2f} s")
                    print(f"Finished GD reconstruction for {folder_name}")
                    if csvr_start_time is not None:
                        print(f"Total time (cSVR start -> reconstruction finish) for {folder_name}: {time.time() - csvr_start_time:.2f} seconds")
                except Exception as e:
                    print(f"Failed to run GD reconstruction on {folder_name}: {e}")
                    traceback.print_exc()

        except Exception as e:
            print(f"Failed to process {directory}: {e}")
            # Print the full traceback for debugging
            traceback.print_exc()


if __name__ == "__main__":
    main()
