#!/usr/bin/env python3
"""
Speed-optimized version of run_pipeline_cSVR.py with server/REPL mode.

Loads models ONCE, then accepts jobs interactively so subsequent runs
skip the expensive model loading (~30s+ startup saved per run).

Usage:
    # Start server mode (loads models, then prompts for jobs):
    python run_pipeline_cSVR_serve.py --serve

    # Then at the cSVR> prompt, type your normal arguments:
    cSVR> /path/to/subject --suffix run1 --preprocess --gd-recon --dest-folder /path/to/subject/cSVR_run1

    # Can also be used as a drop-in replacement for run_pipeline_cSVR.py:
    python run_pipeline_cSVR_serve.py /path/to/subject --suffix run1 --run-cSVR --preprocess --gd-recon --dest-folder /path/to/subject/cSVR_run1
"""

import argparse
import copy
import gc
import json
import os
import shlex
import subprocess
import traceback
import pdb
import warnings
import datasets

from nifti_utils import (standerdize_stack, standerdize_stack_synth, standerdize_stack_from_arrays,
                         make_init_stacks, SLICE_SIZE)
from preprocess_in_memory import preprocess_subject_in_memory, CHAINS, SEG_DEFAULTS
import torch
import run_cSVR_fast
import inr_recon
import gd_recon
import models
import os.path as path
import time
warnings.filterwarnings("ignore", category=UserWarning, message="torch.meshgrid")

# Slices per stack used to build the dummy warm-up tensors. Any value up to SLICE_SIZE//2
# works (make_init_stacks indexes SLICE_SIZE/2 - dim); a real subject has a different slice
# count, but the warm-up only needs to trigger the one-time GPU init, not match shapes.
WARMUP_SLICES_PER_STACK = 30


def warmup_models(model, model_mlp, device=None):
    """Run every net once on dummy inputs so the one-time GPU init (CUDA context, cuDNN,
    custom CUDA-extension JIT -- ~15 s total) is paid at server start-up instead of inside
    the first job. run_pipeline_cSVR_prod.py does the same warm-up with the first subject's
    tensors; in serve mode no subject is loaded yet, so we build correctly-shaped zero
    tensors instead. The output is discarded either way, so results are unaffected."""
    device = device if device is not None else torch.device("cuda")

    # nesvor's slice_acq / transform_convert are JIT-compiled on first import, inside the
    # reconstruction call -- so that cost lands in time_reconstruction_s and dwarfs it
    # (~65 s compiling vs ~5 s reconstructing in cSVR_v2_aug15_trial2). Import them here,
    # before any timer starts. Also reports which implementation is live: the pytorch
    # fallback is ~6x slower and otherwise only whispers a warning to stderr.
    try:
        _t0 = time.time()
        from nesvor_local.slice_acquisition import slice_acq
        from nesvor_local.transform import transform_convert
        print(f"[TIMING] nesvor CUDA extension load: {time.time() - _t0:.2f} s")
        if slice_acq.USE_TORCH or transform_convert.USE_TORCH:
            print("WARNING: nesvor is using its pytorch fallback (slice_acq="
                  f"{not slice_acq.USE_TORCH}, transform_convert={not transform_convert.USE_TORCH}). "
                  "Reconstruction will be several times slower -- check CUDA_HOME.")
        else:
            print("nesvor CUDA extensions active (slice_acq, transform_convert).")
    except Exception as e:
        print(f"Warning: could not pre-load the nesvor CUDA extensions ({e}).")

    # SVR U-Net + MLP: same call run_cSVR_fast.warmup does in prod. It self-gates on
    # _WARMED_UP, so the gated warm-up inside run_svr_inference becomes a no-op.
    try:
        dims = [WARMUP_SLICES_PER_STACK] * 3
        init_stacks = make_init_stacks(dims, device=device)
        dummy_input = torch.zeros((1, 2, sum(dims), SLICE_SIZE, SLICE_SIZE), device=device)
        run_cSVR_fast.warmup(model, model_mlp, init_stacks, dummy_input)
        del init_stacks, dummy_input
    except Exception as e:
        print(f"Warning: SVR/MLP warmup skipped ({e}); it will warm up during the first job.")

    # Preprocessing segmentation U-Net (MONAIfbs DynUNet), used by --preprocess jobs.
    # Dummy batch matches what a real masking call feeds it: batch_size_seg slices at the
    # network's canonical padded patch size.
    try:
        # NOTE: import the names, not the module -- `nesvor_local.preprocessing.masking`
        # re-exports a *function* called brain_segmentation, which shadows the submodule.
        from nesvor_local.preprocessing.masking.brain_segmentation import (
            build_monaifbs_net, W_min, W_factor, H_min, H_factor)
        net = build_monaifbs_net(device)
        dummy_slices = torch.zeros(
            (SEG_DEFAULTS["batch_size_seg"], 1, W_min * W_factor, H_min * H_factor),
            device=device)
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            for _ in range(2):
                _ = net(dummy_slices)
        torch.cuda.synchronize()
        print(f"[TIMING] segmentation U-Net warmup: {time.time() - start:.2f} s")
        del dummy_slices
    except Exception as e:
        print(f"Warning: segmentation U-Net warmup skipped ({e}); "
              f"it will warm up on the first --preprocess job.")


def load_models(no_mlp=False):
    """Load SVR and MLP models once, return (model, model_mlp).

    The MLP is loaded even with --no-mlp: that flag only replaces its stack-order
    decision by the filenames, the 180° in-plane flips still come from the MLP.
    """
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
    end = time.time()
    print(f"Loading svr segment time {end - start:.6f} seconds")

    start = time.time()
    trainee.load_state_dict(torch.load(svr_ckpt, map_location='cuda')['state_dict'])
    end = time.time()
    print(f"Loading SVR checkpoint time {end - start:.6f} seconds")
    model = trainee.model.cuda()

    start = time.time()
    trainee_mlp = models.segment(model=models.flow_SNet3d2_1024_MLP())
    trainee_mlp.load_state_dict(torch.load(mlp_ckpt, map_location='cuda')['state_dict'], strict=False)
    end = time.time()
    print(f"Loading MLP segment+checkpoint time {end - start:.6f} seconds")
    model_mlp = trainee_mlp.model.cuda()
    if no_mlp:
        print("--no-mlp: stack order from filenames; MLP kept for the 180° flips.")

    # Pre-load the preprocessing (MONAIfbs segmentation) model once, so --preprocess jobs
    # don't build/load the masking checkpoint on their first call. Cached in
    # brain_segmentation so every later masking call reuses it.
    try:
        from nesvor_local.preprocessing.masking.brain_segmentation import build_monaifbs_net
        start = time.time()
        build_monaifbs_net(torch.device("cuda"))
        print(f"Loading segmentation (preprocessing) model time {time.time() - start:.6f} seconds")
    except Exception as e:
        print(f"Warning: could not pre-load segmentation model ({e}); it will load on first --preprocess job.")

    print("Models loaded successfully.")
    return model, model_mlp


def write_timings(output_dir, folder_name, suffix, timings):
    """Dump this subject's stage timings next to its reconstruction.

    get_TRE_from_file_outputs_og.py --timings-json folds the file into the subject's
    entry in the metrics JSON, so evaluation output carries the run cost alongside the
    quality numbers. Written even when a stage was skipped (value stays None).
    """
    path = os.path.join(output_dir, f"{folder_name}_timings_{suffix}.json")
    try:
        with open(path, "w") as fh:
            json.dump(timings, fh, indent=2)
        print(f"[TIMING] wrote {path}")
    except OSError as e:
        print(f"[TIMING] could not write {path}: {e}")
    return path


def process_directory(directory, args, model, model_mlp):
    """Process a single directory with pre-loaded models. Same logic as run_pipeline_cSVR.py."""
    if not os.path.exists(directory):
        print(f"Error: Directory {directory} not found.")
        return
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a directory.")
        return

    timings = {"time_preprocess_s": None, "time_pose_estimation_s": None,
               "time_reconstruction_s": None}
    # Timing runs measure compute, so every optional disk write is suppressed: the
    # standerdize dumps, the per-slice cSVR output and the simulated slices.
    timing_mode = getattr(args, "timing_mode", False)
    save_artifacts = not timing_mode

    try:
        print(f"Processing {directory} with suffix '{args.suffix}'...")

        if args.dest_folder:
            effective_output_dir = args.dest_folder
        elif args.inr_recon:
            effective_output_dir = os.path.join(directory, "cSVR_files_inr")
        else:
            effective_output_dir = os.path.join(directory, "cSVR_files")
        os.makedirs(effective_output_dir, exist_ok=True)

        torch.cuda.synchronize()
        preprocess_start = time.time()
        if getattr(args, "preprocess", False):
            # In-memory masking + reorient + bias + normalize from the raw stacks (uses the
            # pre-loaded, cached segmentation model). No intermediate *_re_bias_field_norm files.
            # Re-normalizing per stack here would undo the cross-stack intensity match
            # correct-bias-field leaves behind, which is the whole point of norm_bias.
            stack_normalize = args.stack_normalize
            if stack_normalize == "auto":
                stack_normalize = "none" if args.preprocess_chain == "norm_bias" else "per_stack"
            print(f"Preprocessing {directory} in-memory (chain={args.preprocess_chain}, "
                  f"standerdize normalize={stack_normalize})...")
            pp1, pp2, pp3, pm1, pm2, pm3, pp_vox = preprocess_subject_in_memory(
                directory, torch.device("cuda"), chain=args.preprocess_chain,
                dump_dir=(os.path.join(effective_output_dir, "preprocessed")
                          if args.dump_preprocessed else None))
            init_stacks_tensor, input_tensor, slice_res = standerdize_stack_from_arrays(
                pp1, pp2, pp3, pm1, pm2, pm3, pp_vox, directory, suffix=args.suffix,
                output_dir=effective_output_dir, synth=args.synth,
                normalize_mean=args.normalize_mean, save_outputs=save_artifacts,
                normalize=stack_normalize)
        else:
            init_stacks_tensor, input_tensor, slice_res = standerdize_stack(
                directory, suffix=args.suffix, output_dir=effective_output_dir,
                synth=args.synth, normalize_mean=args.normalize_mean
            )
        torch.cuda.synchronize()
        timings["time_preprocess_s"] = time.time() - preprocess_start
        print(f"[TIMING] preprocessing for {os.path.basename(directory.rstrip(os.sep))}: {timings['time_preprocess_s']:.2f} s")
        input_tensor_hres = None
        if args.synth:
            init_stacks_tensor_hres, input_tensor_hres, slice_res_hres = standerdize_stack_synth(
                directory, suffix=args.suffix, output_dir=effective_output_dir, synth=args.synth
            )
            print("INPUT TENSOR SHAPE:", input_tensor_hres.shape)
        print(f"Finished processing {directory}")

        # Run cSVR
        csvr_slices = None
        if args.run_cSVR and model is not None:
            output_directory = effective_output_dir
            folder_name = os.path.basename(directory.rstrip(os.sep))
            print(f"\nRunning cSVR on {folder_name} using in-memory tensors...")

            save_folder_path = args.save_folder if args.save_folder else os.path.join(output_directory, f"{folder_name}_slices")
            print("save folder path: ", save_folder_path)

            cSVR_args = argparse.Namespace(
                init_stack_template=None,
                input_template=None,
                save_slices=True,
                save_slices_to_disk=save_artifacts and getattr(args, "save_slices_to_disk", False),
                save_debug_images=save_artifacts,
                clin=args.clin,
                synth=args.synth,
                save_folder=save_folder_path,
                slice_res=slice_res,
                suffix=args.suffix,
                no_mlp=getattr(args, "no_mlp", False),
            )

            try:
                csvr_slices = run_cSVR_fast.run_svr_inference(
                    model=model,
                    model_mlp=model_mlp,
                    args=cSVR_args,
                    init_stacks_input=init_stacks_tensor,
                    downsampled_input_tensor=input_tensor,
                    output_dir=directory,
                    downsampled_input_tensor_hres=input_tensor_hres if args.synth else None
                )
                # MLP ordering + SVR forward + pose post-processing (splat_thin, project,
                # splat_thick), as reported by the "POSE ESTIMATION:" line.
                timings["time_pose_estimation_s"] = run_cSVR_fast.LAST_POSE_ESTIMATION_S
                print(f"Finished running cSVR on {folder_name}")
            except Exception as e:
                print(f"Failed to run cSVR on {folder_name}: {e}")
                traceback.print_exc()

        # Run INR recon
        if args.inr_recon:
            folder_name = os.path.basename(directory.rstrip(os.sep))
            print(f"\nRunning INR reconstruction on {folder_name}...")
            try:
                output_directory = effective_output_dir
                input_slices_path = args.save_folder if args.save_folder else os.path.join(output_directory, f"{folder_name}_slices")
                volume_dir = args.output_volume if args.output_volume else output_directory
                output_volume_path = os.path.join(volume_dir, f"{folder_name}_cSVR_inr_recon{args.suffix}.nii.gz")
                sim_slices_path = os.path.join(volume_dir, f"{folder_name}_sim_slices")
                # Prefer in-memory slices from cSVR; deepcopy if GD also runs so it keeps a pristine copy.
                if csvr_slices is not None:
                    inr_input_slices = copy.deepcopy(csvr_slices) if args.gd_recon else csvr_slices
                    print(f"Using {len(inr_input_slices)} in-memory slices for INR (skipping disk reload)")
                else:
                    inr_input_slices = input_slices_path
                torch.cuda.synchronize()
                recon_start = time.time()
                inr_recon.inr(
                    input_slices=inr_input_slices,
                    simulated_slices=sim_slices_path if save_artifacts else None,
                    output_volume=output_volume_path
                )
                torch.cuda.synchronize()
                # GD and INR share the key; at most one of them normally runs.
                timings["time_reconstruction_s"] = time.time() - recon_start
                print(f"[TIMING] inr reconstruction for {folder_name}: {timings['time_reconstruction_s']:.2f} s")
                print(f"Finished INR reconstruction for {folder_name}")
            except Exception as e:
                print(f"Failed to run INR reconstruction on {folder_name}: {e}")
                traceback.print_exc()

        # Run GD recon
        if args.gd_recon:
            folder_name = os.path.basename(directory.rstrip(os.sep))
            print(f"\nRunning GD reconstruction on {folder_name}...")
            try:
                output_directory = effective_output_dir
                input_slices_path = args.save_folder if args.save_folder else os.path.join(output_directory, f"{folder_name}_slices")
                volume_dir = args.output_volume if args.output_volume else output_directory
                output_volume_path = os.path.join(volume_dir, f"{folder_name}_cSVR_gd_recon{args.suffix}.nii.gz")
                sim_slices_path = os.path.join(volume_dir, f"{folder_name}_sim_slices")
                print("sim path: ", sim_slices_path)
                # Prefer in-memory slices from cSVR; fall back to reading from disk.
                if csvr_slices is not None:
                    gd_input_slices = csvr_slices
                    print(f"Using {len(gd_input_slices)} in-memory slices for GD (skipping disk reload)")
                else:
                    gd_input_slices = gd_recon.load_slices(input_slices_path, device=torch.device("cuda"))
                torch.cuda.synchronize()
                recon_start = time.time()
                gd_recon.svr(
                    input_slices=gd_input_slices,
                    output_volume=output_volume_path,
                    simulated_slices=sim_slices_path if save_artifacts else None,
                    no_global_exclusion=True,
                    n_iter=5,
                    n_iter_rec=3,
                )
                torch.cuda.synchronize()
                timings["time_reconstruction_s"] = time.time() - recon_start
                print(f"[TIMING] gd reconstruction for {folder_name}: {timings['time_reconstruction_s']:.2f} s")
                print(f"Finished GD reconstruction for {folder_name}")
            except Exception as e:
                print(f"Failed to run GD reconstruction on {folder_name}: {e}")
                traceback.print_exc()

    except Exception as e:
        print(f"Failed to process {directory}: {e}")
        traceback.print_exc()
    finally:
        # Written even on failure, so a partial run still records what it got through.
        try:
            write_timings(effective_output_dir,
                          os.path.basename(directory.rstrip(os.sep)), args.suffix, timings)
        except NameError:
            pass  # blew up before the output dir was decided

        # Free this subject's GPU memory (and any tensors left by a failed step) so it
        # doesn't accumulate across jobs in serve mode, which keeps the process alive.
        # Runs on success and failure. Note: this only releases THIS process's cache back
        # to the driver -- if the GPU is full because *other* processes are using it (as in
        # the 121 MiB-free OOM), use a free GPU (CUDA_VISIBLE_DEVICES) or the clear-gpu tool.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def make_parser():
    parser = argparse.ArgumentParser(description="Process NIfTI stacks and run cSVR (speed-optimized with serve mode).")
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
    parser.add_argument("--dest-folder", default=None, help="Destination folder to save reconstruction outputs.")
    parser.add_argument("--no-mlp", action="store_true", help="Take the stack order from the filenames instead of the MLP: (sag, cor, axi) when named with _sag/_cor/_axi suffixes, else sorted filename order, which must then already be sagittal, coronal, axial. The MLP still decides the 180° in-plane flip of each stack.")
    parser.add_argument("--normalize-mean", action="store_true", help="Normalize stacks by mean instead of second mode.")
    parser.add_argument("--save-slices-to-disk", action="store_true", help="Also write cSVR slices to disk as .nii.gz (default: keep slices in memory and hand them straight to the reconstruction).")
    parser.add_argument("--preprocess", action="store_true", help="Run masking + reorient + bias-field + normalize in-memory from the subject's three raw stacks (uses the pre-loaded segmentation model), instead of reading pre-processed *_re_bias_field_norm files.")
    parser.add_argument("--preprocess-chain", choices=CHAINS, default="norm_bias",
                        help="--preprocess step order: norm_bias (default) normalizes "
                             "then runs correct-bias-field, which is what the "
                             "nesvor/svort baselines read, so all three methods run on "
                             "the same stacks; bias_norm is the reverse, reproducing "
                             "the *_re_bias_field_norm files the cSVR network was "
                             "trained on.")
    parser.add_argument("--stack-normalize", choices=("auto", "per_stack", "joint", "none"),
                        default="auto",
                        help="Rescaling standerdize applies after cropping. auto "
                             "(default) means none when --preprocess-chain already "
                             "produced intensity-matched stacks (norm_bias), else "
                             "per_stack. per_stack rescales each stack on its own, "
                             "which discards any cross-stack match the inputs arrived "
                             "with; joint derives one scale from all three. The "
                             "ordering MLP always gets its own per-stack normalized "
                             "copy, whatever this is set to.")
    parser.add_argument("--dump-preprocessed", action="store_true",
                        help="Write the --preprocess intermediates (*_bias_field.nii.gz, "
                             "*_norm.nii.gz, mask_*.nii.gz) into <dest>/preprocessed for "
                             "debugging.")
    parser.add_argument("--serve", action="store_true", help="Load models once and warm them up, run any directories given on the command line, then enter the interactive loop for further jobs.")
    parser.add_argument("--timing-mode", action="store_true",
                        help="Measure speed without paying for artifacts: skip the standerdize "
                             "*.pt/*.nii.gz dumps, the per-slice cSVR output and the simulated "
                             "slices. The reconstructed volume and the timings JSON are still "
                             "written. Nothing downstream of this can be evaluated -- "
                             "get_TRE needs the slice folders.")
    return parser


def serve_loop(model, model_mlp, base_parser):
    """Interactive REPL: models stay loaded in GPU, each line is parsed as new arguments."""
    print("\n" + "=" * 60)
    print("SERVER MODE: Models are loaded and ready on GPU.")
    print("Enter arguments (same as CLI). --run-cSVR is implied.")
    print("Type 'quit' or Ctrl-D to exit.")
    print("=" * 60 + "\n")

    while True:
        try:
            line = input("cSVR> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting server mode.")
            break
        if not line:
            continue
        if line.lower() in ('quit', 'exit', 'q'):
            print("Exiting server mode.")
            break

        try:
            tokens = shlex.split(line)
            job_args = base_parser.parse_args(tokens)
            job_args.run_cSVR = True  # always implied in serve mode

            directories = list(job_args.directories)
            if job_args.dir_list:
                with open(job_args.dir_list) as f:
                    file_dirs = [l.strip() for l in f if l.strip() and not l.startswith('#')]
                directories = file_dirs + directories
            if not directories:
                print("Error: No directories specified.")
                continue

            start = time.time()
            for directory in directories:
                process_directory(directory, job_args, model, model_mlp)
            elapsed = time.time() - start
            print(f"\nJob done in {elapsed:.2f}s\n")

        except SystemExit:
            # argparse calls sys.exit on parse error; catch so the loop continues
            pass
        except Exception as e:
            print(f"Error processing job: {e}")
            traceback.print_exc()


def main():
    parser = make_parser()
    args = parser.parse_args()

    directories = list(args.directories)
    if args.dir_list:
        with open(args.dir_list) as f:
            file_dirs = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        directories = file_dirs + directories

    # Serve mode: load the checkpoints once, warm up on dummy inputs, run any
    # directories given on the command line, then take further jobs from the prompt.
    if args.serve:
        print("Loading models for server mode...")
        model, model_mlp = load_models(no_mlp=args.no_mlp)
        print("Warming up models on dummy inputs...")
        warmup_models(model, model_mlp)
        if directories:
            args.run_cSVR = True
            start = time.time()
            for directory in directories:
                process_directory(directory, args, model, model_mlp)
            print(f"\nCommand-line job done in {time.time() - start:.2f}s\n")
        serve_loop(model, model_mlp, make_parser())
        return

    # Normal mode: same behavior as original run_pipeline_cSVR.py, plus the warm-up
    if not directories:
        parser.error("No directories specified. Provide positional arguments or --dir-list.")

    model = None
    model_mlp = None
    if args.run_cSVR:
        print("Loading models once...")
        model, model_mlp = load_models(no_mlp=args.no_mlp)
        print("Warming up models on dummy inputs...")
        warmup_models(model, model_mlp)

    for directory in directories:
        process_directory(directory, args, model, model_mlp)


if __name__ == "__main__":
    main()
