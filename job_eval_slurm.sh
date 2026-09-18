#!/bin/bash
## SLURM array job: one subject per task. Runs the cSVR pipeline in serve mode with
## in-memory preprocessing and gradient-descent reconstruction, then scores the result
## with get_TRE_from_file_outputs_og.py (slice similarity: NCC / SSIM / PSNR).
##
## Required (export before sbatch):
##   DATA_DIR      directory holding one sub-directory per subject (three stacks each)
##   SUFFIX        run name; outputs go to $DATA_DIR/<subject>/cSVR_$SUFFIX
## Optional:
##   SUBJECT_LIST  text file, one subject directory name per line
##                 (default: $DATA_DIR/subjects.txt). --array must cover its line indices.
##   MODE          eval (default) | timing. Also accepted as the first positional argument.
##                 timing measures compute only: no slice / simulated-slice output, and
##                 the evaluation step is skipped.
##
## Launch from the repository root (SLURM_SUBMIT_DIR locates the code and .venv):
##   mkdir -p train_outs evaluate_metrics
##   export DATA_DIR=/path/to/data SUFFIX=run1
##   sbatch --array=0-17 -p <partition> --gres=gpu:1 job_eval_slurm.sh
##
#SBATCH --job-name cSVR
#SBATCH --output=train_outs/o_%x_%A_%a.out
#SBATCH --error=train_outs/err_%x_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --ntasks=1
#SBATCH --time=00-10:00:00
#SBATCH --array=0

set -eo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DATA_DIR="${DATA_DIR:?export DATA_DIR=/path/to/data}"
SUFFIX="${SUFFIX:?export SUFFIX=<run name>}"
SUBJECT_LIST="${SUBJECT_LIST:-$DATA_DIR/subjects.txt}"
MODE="${1:-${MODE:-eval}}"

# Environment: the uv venv created with `uv sync` (README, "Installation").
source "$SCRIPT_DIR/.venv/bin/activate"

# nesvor's slice_acq / transform_convert CUDA kernels are JIT-compiled by torch on first
# use and need a CUDA 11.7 toolkit (nvcc + headers) via CUDA_HOME. Without it they silently
# fall back to pytorch kernels (~6x slower reconstruction; stderr says "Fail to load CUDA
# extention"). Pin the arch list so one cached build serves every GPU model you run on:
# torch's extension cache ignores the arch, so alternating GPU types otherwise rebuilds.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5;8.6}"
# Keep the JIT build cache per checkout: torch's default (~/.cache/torch_extensions) is
# shared by every environment on the machine, and a build made by one environment is
# invalidated by the next (different torch include paths), costing ~100 s per job.
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$SCRIPT_DIR/.torch_extensions}"
if [ -z "${CUDA_HOME:-}" ] && [ -e "$SCRIPT_DIR/.cuda_home/bin/nvcc" ]; then
    export CUDA_HOME="$SCRIPT_DIR/.cuda_home"
fi
if [ -z "${CUDA_HOME:-}" ]; then
    echo "WARNING: CUDA_HOME not set; nesvor's CUDA extensions will fall back to pytorch." >&2
fi

mapfile -t SUBJECTS < "$SUBJECT_LIST"
IDX="${SLURM_ARRAY_TASK_ID:-0}"
SUBJECT="${SUBJECTS[$IDX]}"
NAME="$(basename "$SUBJECT")"
OUT_DIR="$DATA_DIR/$NAME/cSVR_$SUFFIX"

case "$MODE" in
    # get_TRE reads the slices from disk, so eval must write them.
    eval)   PIPELINE_FLAGS="--save-slices-to-disk"; RUN_EVAL=1 ;;
    timing) PIPELINE_FLAGS="--timing-mode";         RUN_EVAL=0 ;;
    *) echo "Unknown MODE '$MODE' (expected eval or timing)" >&2; exit 1 ;;
esac

echo "Running: idx=$IDX subject=$NAME mode=$MODE"
cd "$SCRIPT_DIR"

# Serve mode reads one argument line per job from stdin; --run-cSVR is implied.
# --dest-folder is taken as-is (absolute), not relative to the subject directory.
echo "$DATA_DIR/$SUBJECT --suffix $SUFFIX --preprocess --gd-recon $PIPELINE_FLAGS --dest-folder $OUT_DIR" \
  | python run_pipeline_cSVR_serve.py --serve

# Stage timings are written next to the reconstruction in both modes.
TIMINGS="$OUT_DIR/${NAME}_timings_${SUFFIX}.json"

if [ "$RUN_EVAL" -eq 1 ]; then
    mkdir -p evaluate_metrics
    # --timings_json folds the stage times into the subject's metrics entry.
    python get_TRE_from_file_outputs_og.py \
        --og_slices "$OUT_DIR/${NAME}_sim_slices" \
        --folder_est "$OUT_DIR/${NAME}_slices" \
        --json_path "evaluate_metrics/${SUFFIX}${IDX}.json" \
        --img_num "$IDX" --img_name "$NAME" \
        --timings_json "$TIMINGS" \
        --clin "True" --only_slice_sim
else
    echo "timing mode: skipping evaluation. Timings in $TIMINGS"
    cat "$TIMINGS"
fi
