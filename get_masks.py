import sys
import argparse
from nesvor_local.cli.main import main

import os
import glob

def segment_stack(input_stacks, output_stack_masks):
    args = ["segment-stack"]
    
    # input_stacks handling
    stacks_to_use = []
    if isinstance(input_stacks, str):
        input_stacks = [input_stacks]
        

    
    for item in input_stacks:
        stacks_to_use.append(item)
            
    print("stacks to use")
    print(stacks_to_use)
    args += ["--input-stacks"] + stacks_to_use
    
    # output_stack_masks handling
    if isinstance(output_stack_masks, str):
        args += ["--output-stack-masks", output_stack_masks]
    elif isinstance(output_stack_masks, list):
        args += ["--output-stack-masks"] + output_stack_masks

    old_argv = sys.argv
    sys.argv = ["nesvor"] + args
    try:
        main()
    finally:
        sys.argv = old_argv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, nargs='+', required=True, help="Directory containing input stacks")
    args = parser.parse_args()

    for input_dir in args.input_dir:
        files = glob.glob(os.path.join(input_dir, "*.nii.gz")) + glob.glob(os.path.join(input_dir, "*.nii"))
        filtered = [f for f in files if (f.endswith("_sag.nii.gz") or f.endswith("_cor.nii.gz") or f.endswith("_axi.nii.gz") or f.endswith("_sag.nii") or f.endswith("_cor.nii") or f.endswith("_axi.nii")) and not os.path.basename(f).startswith("mask_")]
        input_stacks = sorted(filtered)
        
        if not input_stacks:
            print(f"Warning: No .nii or .nii.gz files found in directory: {input_dir}")
            continue
            
        print(f"Found {len(input_stacks)} files in {input_dir}")
        segment_stack(input_stacks, input_dir)
