#!/bin/bash
## SLURM Variables:
#SBATCH --job-name teval_clin
#SBATCH --output=/data/vision/polina/users/mfirenze/cSVR/train_outs/o_${SUFFIX}_%a.out
#SBATCH --error=/data/vision/polina/users/mfirenze/cSVR/train_outs/err_${SUFFIX}_%a.err
#SBATCH --partition=polina-all
#SBATCH -A vision-polina
#SBATCH --qos=vision-polina-main
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --ntasks=1
#SBATCH --time=00-10:00:00
#SBATCH --array=0-17
#SBATCH --exclude=chili,pandan

set -e




source /data/vision/polina/users/mfirenze/.bashrc
source /data/vision/polina/users/mfirenze/miniconda3/etc/profile.d/conda.sh
conda activate cSVR_env_v5
export PYTHONPATH="/data/vision/polina/users/mfirenze/miniconda3/envs/cSVR_env_v5/lib/python3.10"


## EXECUTION OF PYTHON CODE:


#export SUFFIX="feb19_sbatch2_time"
SCRIPT_DIR="/data/vision/polina/users/mfirenze/cSVR"
DATA_DIR="/data/vision/polina/users/mfirenze/clin_data_processed_svr"
mapfile -t FILES < "${DATA_DIR}/test_subs.txt"

IDX=${SLURM_ARRAY_TASK_ID}
FILE=${FILES[$IDX]}
FILE_NAME=$(basename "$FILE")

echo "Running: idx=$IDX, file=$FILE"
python "${SCRIPT_DIR}/run_pipeline_cSVR.py" "${DATA_DIR}/$FILE" --suffix "$SUFFIX" --run-cSVR --gd-recon


python /data/vision/polina/users/mfirenze/cSVR/get_TRE_from_file_outputs.py \
    --og_slices "${DATA_DIR}/${FILE_NAME}/cSVR_files/${FILE_NAME}_sim_slices" \
    --folder_est "${DATA_DIR}/${FILE_NAME}/cSVR_files/${FILE_NAME}_slices" \
    --json_path "${SCRIPT_DIR}/evaluate_metrics/${SUFFIX}${IDX}.json" \
    --img_num $IDX \
    --clin "True"



# python /data/vision/polina/users/mfirenze/cSVR/get_TRE_from_file_outputs.py \
#     --og_slices "${DATA_DIR}/${FILE_NAME}/cSVR_files_inr/${FILE_NAME}_sim_slices" \
#     --folder_est "${DATA_DIR}/${FILE_NAME}/cSVR_files_inr/${FILE_NAME}_slices" \
#     --json_path "${SCRIPT_DIR}/evaluate_metrics/${SUFFIX}${IDX}.json" \
#     --img_num $IDX \
#     --clin "True"


#export SUFFIX="cSVR_feb20_gd_recon_rep6_"
#jobid=$(envsubst '$SUFFIX' < job_eval_clin.sh | sbatch | awk '{print $4}')
#sbatch --dependency=afterok:$jobid summarize_results.sh $SUFFIX


#export SUFFIX="cSVR_feb20_gd_recon_rep6_traj"
#jobid=$(envsubst '$SUFFIX' < job_eval_clin.sh | sbatch | awk '{print $4}')
#sbatch --dependency=afterok:$jobid summarize_results.sh $SUFFIX