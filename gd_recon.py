import sys
from nesvor.cli.main import main
import torch 
import argparse



def svr(
    input_slices,
    output_volume,
    no_global_exclusion=False,
    n_iter=None,
    n_iter_rec=None,
):
    args = ["svr"]

    if isinstance(input_slices, str):
        args += ["--input-slices", str(input_slices)]
    
    args += ["--output-volume", str(output_volume)]

    if no_global_exclusion:
        args += ["--no-global-exclusion"]

    if n_iter is not None:
        args += ["--n-iter", str(n_iter)]

    if n_iter_rec is not None:
        args += ["--n-iter-rec", str(n_iter_rec)]

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-slices", type=str, required=True)
    parser.add_argument("--output-volume", type=str, required=True)
    args = parser.parse_args()

    print("before running svr")
    svr(
        input_slices=load_slices(args.input_slices, device=torch.device("cuda")),
        output_volume=args.output_volume,
        no_global_exclusion=True,
        n_iter=5,
        n_iter_rec=3,
    )


