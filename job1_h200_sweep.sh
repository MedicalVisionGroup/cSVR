#!/bin/bash
## SLURM Variables:
#SBATCH --job-name="${JOB_NAME}"
#SBATCH --output=/data/vision/polina/users/mfirenze/cSVR/train_outs/out_${JOB_NAME}_%a.out
#SBATCH -e /data/vision/polina/users/mfirenze/cSVR/train_outs/err_${JOB_NAME}_%a.out
#SBATCH -o /data/vision/polina/users/mfirenze/cSVR/train_outs/o_${JOB_NAME}_%a.out
#SBATCH --partition=csail-shared-h200
#SBATCH --qos=lab-free
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=170G
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --exclude=aia-h200-8,quanta-h200-1
#SBATCH --time=1-00:00:00
#SBATCH --array=0-2
#SBATCH --ntasks=1

# activate virtual environment #SBATCH --exclude=rosemary, aia-h200-13



source /data/vision/polina/users/mfirenze/.bashrc
source /data/vision/polina/users/mfirenze/miniconda3/etc/profile.d/conda.sh
#conda activate 4DCNN_env_freesurfer
conda activate 4DCNN_env_h200
export WANDB_API_KEY=37fc89406827ac32962cc9a582cd3748f11cb5a0
export PYTHONPATH="/data/vision/polina/users/mfirenze/miniconda3/envs/4DCNN_env_freesurfer/lib/python3.10"
which python
python --version
cd  /data/vision/polina/users/mfirenze/cSVR
## EXECUTION OF PYTHON CODE:

git add -u
git commit -m "$JOB_NAME" || echo "No commited changes"

set -e



n=4
# 2. Build the array (more compatible syntax)
RUN_NAMES=()
for i in $(seq 0 $((n-1))); do
    RUN_NAMES+=("jan26_sz$i")
done

# 3. Handle the Index (Default to 0 if Slurm ID is empty)
IDX=${SLURM_ARRAY_TASK_ID:-0}

# 4. Extract the run name
CURRENT_RUN="${RUN_NAMES[$IDX]}"

# --- DEBUG INFO ---
echo "Slurm ID: $SLURM_ARRAY_TASK_ID"
echo "Target Run: $CURRENT_RUN"
echo "All Runs: ${RUN_NAMES[@]}"
# ------------------
# export JOB_NAME="jan13_train_params_check_new_config_fix_lr_rep_rot100_h200"
# envsubst < job1_h200.sh | sbatch 
# 5. Run the command
python /data/vision/polina/users/mfirenze/cSVR/train_from_bulk_wdb_yaml.py --config "./experiments/${CURRENT_RUN}.yaml"
# export JOB_NAME="sweep_h200"

# envsubst < job1_h200_sweep.sh | sbatch 
