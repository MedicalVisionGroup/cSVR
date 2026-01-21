import sys
from nesvor.cli.main import main
import torch 


def inr(
    input_slices,
    output_volume,

):
    args = ["reconstruct"]

    if isinstance(input_slices, str):
        args += ["--input-slices", str(input_slices)]
        
    args += ["--output-volume", output_volume]


    old_argv = sys.argv
    sys.argv = ["nesvor"] + args
    try:
        if isinstance(input_slices, str):
            main()
        else:
            main(input_slices=input_slices)
    finally:
        sys.argv = old_argv

from nesvor.image import load_slices

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-slices", type=str, required=True)
    parser.add_argument("--output-volume", type=str, required=True)
    args = parser.parse_args()

    print("before running svr")
    inr(
        input_slices=load_slices(args.input_slices, device=torch.device("cuda")),
        output_volume=args.output_volume,
    )


