import sys
import argparse
from nesvor_local.cli.main import main

import os
import glob

def bias_correct(input_stacks, input_masks, output_stack_masks):
    args = ["correct-bias-field"]
    
    # Ensure inputs are lists
    if isinstance(input_stacks, str):
        input_stacks = [input_stacks]
    if isinstance(input_masks, str):
        input_masks = [input_masks]
    if isinstance(output_stack_masks, str):
        output_stack_masks = [output_stack_masks]

    print("stacks to use")
    print(input_stacks)
    
    args += ["--input-stacks"] + input_stacks
    args += ["--stack-masks"] + input_masks
    args += ["--output-corrected-stacks"] + output_stack_masks
    

    
    old_argv = sys.argv
    sys.argv = ["nesvor"] + args
    try:
        main()
    finally:
        sys.argv = old_argv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing input stacks")
    parser.add_argument("--stack-masks", type=str, required=False, help="Directory of masks")
    parser.add_argument("--output-stack-masks", type=str, required=False, help="Path(s) to output mask(s) or folder")
    args = parser.parse_args()

    input_dir = args.input_dir
    files = glob.glob(os.path.join(input_dir, "*.nii.gz")) + glob.glob(os.path.join(input_dir, "*.nii"))
    filtered = [f for f in files if (f.endswith("_sag.nii.gz") or f.endswith("_cor.nii.gz") or f.endswith("_axi.nii.gz") or f.endswith("_sag.nii") or f.endswith("_cor.nii") or f.endswith("_axi.nii")) and not os.path.basename(f).startswith("mask_")]
    input_stacks = sorted(filtered)

    filtered_masks = [f for f in files if (f.endswith("_sag.nii.gz") or f.endswith("_cor.nii.gz") or f.endswith("_axi.nii.gz") or f.endswith("_sag.nii") or f.endswith("_cor.nii") or f.endswith("_axi.nii")) and os.path.basename(f).startswith("mask_")]
    input_masks = sorted(filtered_masks)
    
    if not input_stacks:
        print(f"Error: No .nii or .nii.gz files found in directory: {input_dir}")
        sys.exit(1)
        
    print(f"Found {len(input_stacks)} files in {input_dir}")
    if (args.output_stack_masks == None):
        stack_masks = [f.replace(".nii.gz", "_bf.nii.gz") if f.endswith(".nii.gz") else f.replace(".nii", "_bf.nii") for f in input_stacks]
    else:
        stack_masks = args.output_stack_masks
    bias_correct(input_stacks, input_masks, stack_masks)
