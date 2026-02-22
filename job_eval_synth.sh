#!/bin/bash
## SLURM Variables:
#SBATCH --job-name eval_synth
#SBATCH --output=/data/vision/polina/users/mfirenze/cSVR/train_outs/o_${ROOT_NAME}_%a.out
#SBATCH --error=/data/vision/polina/users/mfirenze/cSVR/train_outs/err_${ROOT_NAME}_%a.err
#SBATCH --partition=polina-all
#SBATCH -A vision-polina
#SBATCH --qos=vision-polina-main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --ntasks=1
#SBATCH --time=00-10:00:00
#SBATCH --array=0-11
#SBATCH --gres=gpu:a6000:1


set -e


# Number of run names
n=12  # change this to whatever value you want
RUN_NAMES=()

for ((i=0; i<n; i++)); do
    RUN_NAMES+=("synth_$i")
done
for ((i=0; i<n; i++)); do
    IMG_NUMS+=($i)
done



# Get task index
IDX=${SLURM_ARRAY_TASK_ID}
RUN_NAME=${RUN_NAMES[$IDX]}
RUN_NAME_GT="${RUN_NAMES[$IDX]}_gt"
IMG_NUM=${IMG_NUMS[$IDX]}
HRES="False" #True
CLIN="False"
ROT="20"
TRANS="0.03"
NOISE="0"
GT_File="synth_gt_precomp_v2_tr0.03_rot20_"



# Scripts and parameters
GT_SCRIPT="evaluate_pipeline_crop.sh"
EVAL_SCRIPT="evaluate_pipeline_crop.sh"
RESULT_FILE="${ROOT_NAME}${IDX}.json" #"only_nesvor_svr_recon" #"only_nesvor" #"only_svr" #"mine_nesvor_iter"
MODEL_PATH="feta3d0_multi_stack_svr_final_sb2_crop_flow_SNet3d2_1024_multi_crop_l22_loss_grid_multiscale_40_0.03_0.1_180_12_lr0.0001_feb17_node5_repeat_180_in_plane" #"feta3d0_multi_stack_svr_final_sb2_crop_flow_SNet3d2_512_multi_crop_l22_loss_grid_no_aug_flip" #"feta3d0_multi_stack_svr_final_sb2_crop_hr_flow_SNet3d2_512_multi_crop_hr_low_features_l22_loss_grid_256_low_hres_rep2" #"feta3d0_multi_stack_svr_final_sb2_crop_flow_SNet3d2_512_multi_crop_l22_loss_grid_no_aug_flip" #"feta3d0_multi_stack_svr_final_sb2_crop_hr_flow_SNet3d2_512_multi_crop_hr_low_features_l22_loss_grid_256_low_hres_rep2" #"feta3d0_multi_stack_svr_final_sb2_crop_flow_SNet3d2_512_multi_crop_l22_loss_grid_no_aug_flip" #"feta3d0_multi_stack_svr_final_sb2_crop_flow_SNet3d2_512_multi_crop_l22_loss_grid_no_repeat_proj_psf_v3" #"feta3d0_multi_stack_svr_final_sb2_crop_hr_flow_SNet3d2_512_multi_crop_hr_low_features_l22_loss_grid_no_repeat_proj_psf_HR_low_features_psf4" #"feta3d0_multi_stack_svr_final_sb2_crop_flow_SNet3d2_512_multi_crop_l22_loss_grid_no_repeat_proj_psf_v3"
#model_path="feta3d0_multi_stack_svr_final_sb2_flow_SNet3d2_512_multi_l22_loss_gridbulkrot_0_no_crop_v9_sb4_no_l2_loss_mask_mot"



echo "=== Variable Check ==="
echo "IDX=$IDX"
echo "ROOT_NAME=$ROOT_NAME"
echo "RECON_METHOD=$RECON_METHOD"
echo "RUN_NAME=$RUN_NAME"
echo "RUN_NAME_GT=$RUN_NAME_GT"
echo "IMG_NUM=$IMG_NUM"
echo "RESULT_FILE=$RESULT_FILE"
echo "MODEL_PATH=$MODEL_PATH"
echo "EVAL_SCRIPT=$EVAL_SCRIPT"
echo "HRES=$HRES"
echo "CLIN=$CLIN"
echo "GT_File=$GT_File"
echo "ROT=$ROT"
echo "TRANS=$TRANS"
echo "NOISE=$NOISE"
echo "====================="
echo "Running: run_name=$RUN_NAME, img_num=$IMG_NUM"


bash "$EVAL_SCRIPT" "$RUN_NAME_GT" "$IMG_NUM" "$RESULT_FILE" "$RECON_METHOD" "$MODEL_PATH" "True" "$HRES" "$CLIN" "$GT_File" "$ROT" "$TRANS" "$NOISE"
#bash "$EVAL_SCRIPT" "$RUN_NAME" "$IMG_NUM" "$RESULT_FILE" "$RECON_METHOD" "$MODEL_PATH" "False" "$HRES" "$CLIN" "$GT_File" "$ROT" "$TRANS" "$NOISE"

# export ROOT_NAME="try_synth_cSVR9"
# export RECON_METHOD="only_svr_no_refine"
# jobid=$(envsubst '$ROOT_NAME' < job_eval_synth.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME



# export ROOT_NAME="svort_vf_jan28_tr0.03_rot20_"
# export RECON_METHOD="only_svort"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME

# export ROOT_NAME="nesvor_only_vf_jan28_tr0.03_rot20_"
# export RECON_METHOD="only_nesvor"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME

# export ROOT_NAME="nesvor_mine_vf_jan28_tr0.03_rot20_"
# export RECON_METHOD="mine_nesvor_iter"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME

# export ROOT_NAME="csvr_ref_vf_jan28_tr0.03_rot20_"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME




# export ROOT_NAME="csvr_only_vf_jan28_tr0.03_rot20_"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME


# export ROOT_NAME="csvr_only_jan28_tr0.03_rot20_"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME



# export ROOT_NAME="csvr_ref_v2_jan28_tr0.03_rot20_"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME


# export ROOT_NAME="csvr_r_precomp_v2_tr0.03_rot20_"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME



# export ROOT_NAME="synth_gt_precomp_v2_tr0.03_rot20_"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME




# export ROOT_NAME="svort_tr0.06_rep2"
# export RECON_METHOD="only_svort"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME


# export ROOT_NAME="csvr_r_tr0.06_rep2"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME


# export ROOT_NAME="nesvor_mine_ns0.3_bulk70_v11_"
# export RECON_METHOD="mine_nesvor_iter"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME


# export ROOT_NAME="nesvor_mine_ns0.3_bulk70_v7_"
# export RECON_METHOD="mine_nesvor_iter"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME


# export ROOT_NAME="only_nesvor_ns0.3_bulk70_v7_"
# export RECON_METHOD="only_nesvor"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME



# export ROOT_NAME="svort_mine_ns0.3_bulk70_v7_"
# export RECON_METHOD="only_svort"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME


# export ROOT_NAME="svr_refine_ns0.1_bulk70_v8_"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME


# export ROOT_NAME="svr_only_ns0.3_bulk70_v7_"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME



# export ROOT_NAME="synth_gt_precomp_v4_ns0.3_TESTCSVR"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME



# export ROOT_NAME="svr_refine_ns0.3_bulk70_v7_a6000_"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME


# export ROOT_NAME="only_nesvor_ns0.3_bulk70_v6_"
# export RECON_METHOD="only_nesvor"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME




# export ROOT_NAME="nesvor_mine_ns0.3_bulk70_v5_"
# export RECON_METHOD="mine_nesvor_iter"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME



# export ROOT_NAME="nesvor_mine_ns0.3_bulk70_v5_"
# export RECON_METHOD="mine_nesvor_iter"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME


# export ROOT_NAME="svr_only_ns0.3_bulk70_v5_"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME


# export ROOT_NAME="synth_gt_precomp_v3_ns0.1"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME



# export ROOT_NAME="svr_refine_aug12_ns0.3_synth_f12_"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME


# export ROOT_NAME="nesvor_mine_aug12_ns0.3_synth_f6_"
# export RECON_METHOD="mine_nesvor_iter"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME



# export ROOT_NAME="nesvor_mine_aug12_tr0.12_synth_f3_"
# export RECON_METHOD="mine_nesvor_iter"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME


# export ROOT_NAME="nesvor_mine_aug12_ns0.3_synth_f3_"
# export RECON_METHOD="mine_nesvor_iter"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME


# export ROOT_NAME="svr_layer5_rot20tr0.03_v2_"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME



# export ROOT_NAME="svr_ns0.3_synth_m1_"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME


# export ROOT_NAME="nesvor_mine_tr0.09_bulk70_m1_"
# export RECON_METHOD="mine_nesvor_iter"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME





# export ROOT_NAME="nesvor_mine_tr0.09_bulk70_v2_"
# export RECON_METHOD="mine_nesvor_iter"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME



# export ROOT_NAME="svr_tr0.09_synth_f3_"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME

# export ROOT_NAME="svort_tr0.09_synth_f3_"
# export RECON_METHOD="only_svort"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME

# export ROOT_NAME="nesvor_tr0.09_synth_f3_"
# export RECON_METHOD="only_nesvor"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME








# export ROOT_NAME="nesvor_mine_tr0.12_bulk70_v2_"
# export RECON_METHOD="mine_nesvor_iter"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME



# export ROOT_NAME="synth_gt_precomp_v2_tr0.12_"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME



# export ROOT_NAME="svr_only_ns0.1_bulk70_v2_"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME





# export ROOT_NAME="svr_only_tr0.12_bulk70_v2_"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME

# export ROOT_NAME="svr_only_tr0.12_bulk70_v2_"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME



# export ROOT_NAME="svr_only_rot50_bulk70_v2_"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME



# export ROOT_NAME="nesvor_mine_ns0.3_bulk70_v2_"
# export RECON_METHOD="mine_nesvor_iter"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME




# export ROOT_NAME="nesvor_mine_rot10_bulk70_v2_"
# export RECON_METHOD="mine_nesvor_iter"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME

# export ROOT_NAME="synth_gt_precomp_v2_tr0.12_"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME


# export ROOT_NAME="synth_gt_precomp_v2_ns0.4"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME



# export ROOT_NAME="svr_only_rot60_bulk70_v2_"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME




# export ROOT_NAME="nesvor_mine_rot60_bulk70_v2_"
# export RECON_METHOD="mine_nesvor_iter"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME


# export ROOT_NAME="svr_only_rot60_bulk70_v2_"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME



# export ROOT_NAME="synth_gt_precomp_v2_rot50_"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME



#############################

# export ROOT_NAME="svr_refine_rot60_bulk70_"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME





# export ROOT_NAME="synth_gt_precomp_v1_ns0.4"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME



# export ROOT_NAME="synth_gt_precomp_v1_tr0.12"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME



# export ROOT_NAME="synth_gt_precomp_v1_rot60_"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME




# export ROOT_NAME="synth_mot_test_v7"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME


# export ROOT_NAME="gt_synth_rot30_var2_"
# export RECON_METHOD="only_nesvor"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME


# export ROOT_NAME="nesvor_iter_rot60_synth_f0_"
# export RECON_METHOD="mine_nesvor_iter"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME


# export ROOT_NAME="svr_raw_rot60_synth_f0_"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME




# export ROOT_NAME="svort_ns0.2_synth_f3_"
# export RECON_METHOD="only_svort"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME


# export ROOT_NAME="svr_ns0.2_synth_f3_"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME

# export ROOT_NAME="nesvor_ns0.2_synth_f3_"
# export RECON_METHOD="only_nesvor"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME




# export ROOT_NAME="svort_tr0.12_synth_f3_"
# export RECON_METHOD="only_svort"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME


# export ROOT_NAME="svr_tr0.12_synth_f3_"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME

# export ROOT_NAME="nesvor_tr0.12_synth_f3_"
# export RECON_METHOD="only_nesvor"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME





# export ROOT_NAME="svort_rot60_synth_check2_"
# export RECON_METHOD="only_svort"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME


# export ROOT_NAME="svr_rot60_synth_f2_"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME

# export ROOT_NAME="nesvor_rot60_synth_f2_"
# export RECON_METHOD="only_nesvor"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME






# export ROOT_NAME="only_nesvor_rot10_nodes_100_v3_"
# export RECON_METHOD="only_nesvor"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME








# export ROOT_NAME="only_svr_rot10_nodes_100_v3_"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME

# export ROOT_NAME="only_svr_rot40_nodes_100_v3_"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME




# export ROOT_NAME="svort_rot40_nodes_100_v3_"
# export RECON_METHOD="only_svort"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME



# export ROOT_NAME="nesvor_synth_rot40_nodes_100_v3_"
# export RECON_METHOD="only_nesvor"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME


# export ROOT_NAME="svr_synth_rot40_nodes_100_v3_"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME



# export ROOT_NAME="svort_noise1.4"
# export RECON_METHOD="only_svort"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME


# export ROOT_NAME="svr_save_affine_of_bulkrot3_"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME




# export ROOT_NAME="svr_new_100_refine4_"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME



# export ROOT_NAME="svr_new_model_refine4_"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME



# export ROOT_NAME="nesvor_check10_"
# export RECON_METHOD="only_nesvor"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME


# export ROOT_NAME="nesvor_uniform_params_"
# export RECON_METHOD="only_nesvor"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME

# export ROOT_NAME="nesvor_noise0.1"
# export RECON_METHOD="only_nesvor"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME


# export ROOT_NAME="svr_ref_noise0.8"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME

# export ROOT_NAME="svr_ref_rot40"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME




# export ROOT_NAME="nesvor_only_noise0.8_"
# export RECON_METHOD="only_nesvor"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME




# export ROOT_NAME="svort_rep_scale_masks2_psnr_"
# export RECON_METHOD="only_svort"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME



# export ROOT_NAME="svr_refine_rep_scale_masks2_psnr_"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME


# export ROOT_NAME="nesvor_only_synth_rep_"
# export RECON_METHOD="only_nesvor"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME




##SBATCH --nodelist=marjoram,zaatar,thyme 
# export ROOT_NAME="svrtk_synth_"
# export RECON_METHOD="svrtk"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME


# export ROOT_NAME="refine_svr_only_model_inter_simp_oct20_v2_"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME




# export ROOT_NAME="svr_only_model_inter_simp_oct20_v2_"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME




# export ROOT_NAME="svort_model_inter_simp_oct20_v2_"
# export RECON_METHOD="only_svort"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME


# export ROOT_NAME="svort_model_inter_simp_oct19_"
# export RECON_METHOD="only_svort"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME


# export ROOT_NAME="only_nesvor_inter_simp_oct19_"
# export RECON_METHOD="only_nesvor"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME




# export ROOT_NAME="opt_nesvor_inter_simp_oct19_"
# export RECON_METHOD="mine_nesvor_iter"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME



# export ROOT_NAME="only_svort_inter_simp_oct19_"
# export RECON_METHOD="only_svort"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME



# export ROOT_NAME="opt_nesvor_inter_simp_oct19_"
# export RECON_METHOD="mine_nesvor_iter"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME




# export ROOT_NAME="model_more_bulk_svr_only"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME


# export ROOT_NAME="model_inter_model_svr_only"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME

# export ROOT_NAME="model_mask_model_svr_refine"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME

# export ROOT_NAME="synth_model_t_ants_only_svort_f_"
# export RECON_METHOD="only_svort"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME

# export ROOT_NAME="synth_model_t_ants_only_nesvor_rigid"
# export RECON_METHOD="only_nesvor"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME



# export ROOT_NAME="model_test_rep_conv_v2_hard"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME


# export ROOT_NAME="model_test_rep_conv_v2_"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME


# export ROOT_NAME="synth_rep_inter_"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME

#sbatch summarize_results.sh $ROOT_NAME

# export ROOT_NAME="synth_nesvor_only_sept19_v21_"
# export RECON_METHOD="only_nesvor"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME


# export ROOT_NAME="synth_gd_only_sept30_v21_"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME



# export ROOT_NAME="synth_gd_only_sept19_v21_"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME


# export ROOT_NAME="synth_gd_refine_sept19_v21_"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME


# export ROOT_NAME="synth_gd_refine_sept19_v12_"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME

# export ROOT_NAME="synth_svort_only_sept19_21_"
# export RECON_METHOD="only_svort"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME


# export ROOT_NAME="synth_nesvor_iter_sept19_v21_"
# export RECON_METHOD="mine_nesvor_iter"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME


# export ROOT_NAME="synth_gd_bottleneck_model_refine"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME




# export ROOT_NAME="synth_gd_bottleneck_model_"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME



# export ROOT_NAME="svr_simp_model_test2__"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME



# export ROOT_NAME="svr_simp_model_test2__"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME




# export ROOT_NAME="svr_only_1024_only_gt_test_"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME



# export ROOT_NAME="svr_only_1024_same_zoom2_"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME



# export ROOT_NAME="nesvor_only_synth_v8_"
# export RECON_METHOD="only_nesvor"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME

# export ROOT_NAME="nesvor_no_opt_synth_v8_"
# export RECON_METHOD="mine_nesvor_no_opt"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME



# export ROOT_NAME="svort_no_refine_synth_v8_"
# export RECON_METHOD="only_svort"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME


# export ROOT_NAME="svort_only_synth_v8_"
# export RECON_METHOD="only_svort"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME


# export ROOT_NAME="svr_refine"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME


# export ROOT_NAME="synth_edges_only_nesvor_all9_more_rot_no_thres"
# export RECON_METHOD="only_nesvor"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME




# export ROOT_NAME="synth_edges_only_nesvor_all9_more_rot_no_edges"
# export RECON_METHOD="only_nesvor"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME




# export ROOT_NAME="synth_edges_same_motionv5_no_zoom_"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME



# export ROOT_NAME="only_svr_iter_ds_aug10_high_res2_run5_combo5_low3"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
#sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME


# export ROOT_NAME="clin_svr_nesvor_try_both"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME


# export ROOT_NAME="clin_svr_nesvor_try_both"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME


# export ROOT_NAME="recon_only_128_10_fix2_"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME

# export ROOT_NAME="nesvor_mine_128_10_fix2_"
# export RECON_METHOD="mine_nesvor_iter"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME

# export ROOT_NAME="only_svr2_256_10_"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME

# export ROOT_NAME="only_svr_128_10_rep"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME

# export ROOT_NAME="only_svr_128_10_rep_v3"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME

# export ROOT_NAME="only_svr_128_1_test"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME

# export ROOT_NAME="only_svr_128_1_test2"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME



# export ROOT_NAME="only_svr_360"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME


# export ROOT_NAME="only_svr_360_10_"
# export RECON_METHOD="only_svr"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME


# export ROOT_NAME="nesvor_only_360_v210_"
# export RECON_METHOD="only_nesvor"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME




# export ROOT_NAME="nesvor_only_128_10"
# export RECON_METHOD="only_nesvor"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME

# export ROOT_NAME="nesvor_only_128_10_fixx_"
# export RECON_METHOD="only_nesvor"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME

# export ROOT_NAME="mine_nesvor_128_10"
# export RECON_METHOD="mine_nesvor_iter"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME

# export ROOT_NAME="only_nesvor_1k_1"
# export RECON_METHOD="only_nesvor_1k"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME


# export ROOT_NAME="mine_nesvor_1k_1"
# export RECON_METHOD="mine_nesvor_iter1k"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME

# export ROOT_NAME="mine_nesvor_128_10_rep"
# export RECON_METHOD="mine_nesvor_iter"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME


# export ROOT_NAME="only_svort_128_10_rep_"
# export RECON_METHOD="only_svort"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME

# export ROOT_NAME="mine_nesvor_no_opt_128_10_rep2"
# export RECON_METHOD="mine_nesvor_no_opt"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME

# export ROOT_NAME="mine_nesvor_no_opt_128_10_rep3_"
# export RECON_METHOD="mine_nesvor_no_opt"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME


# export ROOT_NAME="mine_nesvor_no_opt_128_10_rep3_h"
# export RECON_METHOD="mine_nesvor_no_opt"
# jobid=$(envsubst '$ROOT_NAME' < job_evaluate_crop.sh | sbatch | awk '{print $4}')
# echo "Submitted job with ID: $jobid"
# sbatch --dependency=afterok:$jobid summarize_results.sh $ROOT_NAME
