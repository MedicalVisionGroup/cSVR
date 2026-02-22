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


# # python /data/vision/polina/users/mfirenze/cSVR/get_TRE_from_file_outputs.py \
# #     --og_slices "${DATA_DIR}/${FILE_NAME}/cSVR_files/${FILE_NAME}_sim_slices" \
# #     --folder_est "${DATA_DIR}/${FILE_NAME}/cSVR_files/${FILE_NAME}_slices" \
# #     --json_path "${SCRIPT_DIR}/evaluate_metrics/${SUFFIX}${IDX}.json" \
# #     --img_num $IDX \
# #     --clin "True"

# run_name="synth_${IDX}_gt"

# cd /data/vision/polina/users/mfirenze/splatting_exp/inference_recons
# #reference_file="svr_val15_gt.nii.gz"
# reference_file="svr_${run_name}_gt.nii.gz"
# reference_gt_file="svr_${run_name}_gt_brain.nii.gz"
# reg_file="reg_${output_file}"
# reg_gt_file="reg_gt_${output_file}"


# # using only one line

# reference_gt_file="svr_${gt_run_name}${img_num}"_gt_brain.nii.gz
# output_sim_gt="sim_${gt_run_name}${img_num}_gt"
# reference_file="svr_${gt_run_name}${img_num}_gt.nii.gz"

# echo $reg_gt_file
# echo $reference_gt_file
# echo $output_sim_gt
# echo $reference_file

# # register to ground truth poses
# echo "Run ANTS registration"
# /data/vision/polina/users/dey/libraries/ants/install/bin/antsRegistration \
# -d 3 \
# -o ["reg_${img_num}_${run_name}", $reg_file] \
# -r [$reference_file, $output_file, 1] \
# -n Linear \
# -m MeanSquares[$reference_file,  $output_file,  1, , Regular, 1., 1] \
# -t Rigid[0.2] \
# -s 3x2x1x0x2x1x0 \
# -f 2x2x2x2x1x1x1 \
# -c 100x100x100x100x100x100x100 \
# -v

# # register to original volume
# echo "Run ANTS registration not CLIN"
# /data/vision/polina/users/dey/libraries/ants/install/bin/antsRegistration \
# -d 3 \
# -o ["reg_gt_${img_num}_${run_name}", $reg_gt_file] \
# -r [$reference_gt_file, $output_file, 1] \
# -n Linear \
# -m MeanSquares[$reference_gt_file,  $output_file,  1, , Regular, 1., 1] \
# -t Rigid[0.2] \
# -s 3x2x1x0x2x1x0 \
# -f 2x2x2x2x1x1x1 \
# -c 100x100x100x100x100x100x100 \
# -v

# python /data/vision/polina/users/mfirenze/svr_my_train_2024/get_TRE_from_file_outputs.py \
#     --mask_folder /data/vision/polina/users/mfirenze/synth_data_svr/$run_name/slices_$run_name \
#     --folder_est /data/vision/polina/users/mfirenze/synth_data_svr/$run_name/cSVR_files/$output_sim \
#     --folder_gt /data/vision/polina/users/mfirenze/synth_data_svr/$run_name/cSVR_files/$output_sim_gt \
#     --mat_file /data/vision/polina/users/mfirenze/synth_data_svr/$run_name/cSVR_files/"reg_"$img_num"_"$run_name"0GenericAffine.mat" \
#     --gt_file  /data/vision/polina/users/mfirenze/synth_data_svr/$run_name/cSVR_files/$reference_gt_file \
#     --reg_file /data/vision/polina/users/mfirenze/synth_data_svr/$run_name/cSVR_files/$reg_gt_file \
#     --json_path /data/vision/polina/users/mfirenze/cSVR/evaluate_metrics/$json_path \
#     --img_num $img_num \
#     --og_slice_path /data/vision/polina/users/mfirenze/synth_data_svr/$run_name/slices_$run_name 
