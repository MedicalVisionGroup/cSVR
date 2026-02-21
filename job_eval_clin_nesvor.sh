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
conda deactivate 
conda activate nesvor_env2_copy_monai
export PYTHONPATH="/data/vision/polina/users/mfirenze/miniconda3/envs/nesvor_env2_copy_monai/lib/python3.10"

cd  /data/vision/polina/users/mfirenze/cSVR
## EXECUTION OF PYTHON CODE:


#export SUFFIX="feb19_sbatch2_time"
SCRIPT_DIR="/data/vision/polina/users/mfirenze/cSVR"
DATA_DIR="/data/vision/polina/users/mfirenze/clin_data_processed_svr"
mapfile -t FILES < "${DATA_DIR}/test_subs.txt"

IDX=${SLURM_ARRAY_TASK_ID}
FILE=${FILES[$IDX]}
FILE_NAME=$(basename "$FILE")

stack0=$(find "${DATA_DIR}/${FILE_NAME}/" -maxdepth 1 \( -name "*_sag_norm.nii" -o -name "*_sag_norm.nii.gz" \) ! -name "mask_*")
stack1=$(find "${DATA_DIR}/${FILE_NAME}/" -maxdepth 1 \( -name "*_cor_norm.nii" -o -name "*_cor_norm.nii.gz" \) ! -name "mask_*")
stack2=$(find "${DATA_DIR}/${FILE_NAME}/" -maxdepth 1 \( -name "*_axi_norm.nii" -o -name "*_axi_norm.nii.gz" \) ! -name "mask_*")

mask0=$(find "${DATA_DIR}/${FILE_NAME}/" -maxdepth 1 \( -name "mask*_sag.nii" -o -name "mask*_sag.nii.gz" \))
mask1=$(find "${DATA_DIR}/${FILE_NAME}/" -maxdepth 1 \( -name "mask*_cor.nii" -o -name "mask*_cor.nii.gz" \))
mask2=$(find "${DATA_DIR}/${FILE_NAME}/" -maxdepth 1 \( -name "mask*_axi.nii" -o -name "mask*_axi.nii.gz" \))

echo "stack0: $stack0"
echo "stack1: $stack1"
echo "stack2: $stack2"
echo "mask0: $mask0"
echo "mask1: $mask1"
echo "mask2: $mask2"

mkdir -p "${DATA_DIR}/${FILE_NAME}/nesvor_files"

nesvor reconstruct --input-stacks $stack0 $stack1 $stack2 \
    --stack-masks $mask0 $mask1 $mask2 \
    --simulated-slices "${DATA_DIR}/${FILE_NAME}/nesvor_files/${FILE_NAME}_sim_slices_${SUFFIX}" \
    --output-slices "${DATA_DIR}/${FILE_NAME}/nesvor_files/${FILE_NAME}_out_slices_${SUFFIX}" \
    --output-volume "${DATA_DIR}/${FILE_NAME}/nesvor_files/${FILE_NAME}_${SUFFIX}.nii.gz" \

    
python /data/vision/polina/users/mfirenze/cSVR/get_TRE_from_file_outputs.py \
    --og_slices "${DATA_DIR}/${FILE_NAME}/nesvor_files/${FILE_NAME}_sim_slices_${SUFFIX}" \
    --folder_est "${DATA_DIR}/${FILE_NAME}/nesvor_files/${FILE_NAME}_out_slices_${SUFFIX}"  \
    --json_path "${SCRIPT_DIR}/evaluate_metrics/${SUFFIX}${IDX}.json" \
    --img_num 0 \
    --clin "True"


#export SUFFIX="nesvor_norm_v3_"
#jobid=$(envsubst '$SUFFIX' < job_eval_clin_nesvor.sh | sbatch | awk '{print $4}')
#sbatch --dependency=afterok:$jobid summarize_results.sh $SUFFIX