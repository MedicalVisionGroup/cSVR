#!/bin/bash
## SLURM training job for cSVR: one config per submission.
##
## Required (export before sbatch):
##   CONFIG        training config, e.g. experiments/feb17_node5_repeat_180_in_plane.yaml
## Optional:
##   CONDA_ENV     conda environment to activate. Training needs pytorch-lightning 1.9
##                 (see experiments/README.md) -- the pinned uv environment of this
##                 repository is for inference and will NOT run the training script.
##   WANDB_API_KEY wandb credentials (or set WANDB_MODE=offline to skip logging in).
##   WANDB_ENTITY / WANDB_PROJECT   where runs are logged (defaults are the maintainer's).
##
## Launch from the repository root (SLURM_SUBMIT_DIR locates the code):
##   mkdir -p train_outs
##   export CONFIG=experiments/feb17_node5_repeat_180_in_plane.yaml
##   sbatch -p <partition> --gres=gpu:1 job_train_slurm.sh
##
## Checkpoints go to ./checkpoints/<run name>/ (best.ckpt + last.ckpt). Resubmitting
## with the same config resumes automatically from last.ckpt (same wandb run id, too).
##
#SBATCH --job-name cSVR_train
#SBATCH --output=train_outs/o_%x_%j.out
#SBATCH --error=train_outs/err_%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --ntasks=1
#SBATCH --time=7-00:00:00

set -eu
cd "${SLURM_SUBMIT_DIR:-.}"

if [ -n "${CONDA_ENV:-}" ]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

: "${CONFIG:?export CONFIG=experiments/<config>.yaml before sbatch}"

which python
python --version
python train_from_bulk_wdb_yaml.py --config "$CONFIG"
