#!/bin/bash
## SLURM Variables:
#SBATCH --job-name="${JOB_NAME}"
#SBATCH --output=/data/vision/polina/users/mfirenze/cSVR/train_outs/out_${JOB_NAME}.out
#SBATCH -e /data/vision/polina/users/mfirenze/cSVR/train_outs/err_${JOB_NAME}.out
#SBATCH -o /data/vision/polina/users/mfirenze/cSVR/train_outs/o_${JOB_NAME}.out
#SBATCH --partition=polina-all
#SBATCH -A vision-polina
#SBATCH --qos=vision-polina-main
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1




# activate virtual environment #SBATCH --exclude=rosemary

# #SBATCH --gres=gpu:rtx_6000_ada:1

source /data/vision/polina/users/mfirenze/.bashrc
source /data/vision/polina/users/mfirenze/miniconda3/etc/profile.d/conda.sh
conda activate 4DCNN_env_freesurfer
export WANDB_API_KEY=37fc89406827ac32962cc9a582cd3748f11cb5a0
export PYTHONPATH="/data/vision/polina/users/mfirenze/miniconda3/envs/4DCNN_env_freesurfer/lib/python3.10"
which python
python --version
cd  /data/vision/polina/users/mfirenze/cSVR
## EXECUTION OF PYTHON CODE:

git add -u
git commit -m "$JOB_NAME" || echo "No commited changes"


#bash /data/vision/polina/users/mfirenze/cSVR/feta3d_svr_train_reorient_l2.sh

#bash /data/vision/polina/users/mfirenze/cSVR/feta3d_svr_train_reorient_wb_config.sh
#bash /data/vision/polina/users/mfirenze/cSVR/feta3d_svr_mlp_train_reorient_wb_config.sh


#python /data/vision/polina/users/mfirenze/cSVR/train_from_bulk_wdb_yaml.py --config ./experiments/train_params_mlp_lr0.05_stack_order_more_channels_bigger_image.yaml


#python /data/vision/polina/users/mfirenze/cSVR/train_from_bulk_wdb_yaml.py --config ./experiments/train_params_mlp_lr0.05_stack_order_more_channels_bigger_image_more_slices32_fix_loss_no_rot_h200_no_plane_rot_less_mot_4096snet.yaml

#python /data/vision/polina/users/mfirenze/cSVR/train_from_bulk_wdb_yaml.py --config ./experiments/train_params_mlp_lr0.05_stack_order_more_channels_bigger_image_more_slices32_fix_loss_no_rot_h200_low_mot_vec.yaml
#python /data/vision/polina/users/mfirenze/cSVR/train_from_bulk_wdb_yaml.py --config ./experiments/train_params_low_mot_vec_new_mlp_imp_drop_out0.2.yaml

#bash /data/vision/polina/users/mfirenze/cSVR/feta3d_svr_train_reorient_l2_wb_resume.sh

#python /data/vision/polina/users/mfirenze/cSVR/train_from_bulk_wdb_yaml.py --config ./experiments/jan9_train_params_check_new_config_fix_lr_rep_clean_a6000_real_multi_scale_debug.yaml
python /data/vision/polina/users/mfirenze/cSVR/train_from_bulk_wdb_yaml.py --config ./experiments/jan14__mlp_norm_64size_vec_rots_rep_small_unet_small_mlp_remove_3D_2D.yaml
# export JOB_NAME="jan14__mlp_norm_64size_vec_rots_rep_small_unet_small_mlp_remove_3D_2D"
# envsubst < job1.sh | sbatch 

#python /data/vision/polina/users/mfirenze/cSVR/train_from_bulk_wdb_yaml.py --config ./experiments/train_params_low_mot_vec_new_mlp_imp_mlp_norm_64size_no_rot_vec_correct_flips_in_plane_rots.yaml


#python /data/vision/polina/users/mfirenze/cSVR/train_from_bulk_wdb_yaml.py --config ./experiments/train_params_low_mot_vec_new_mlp_imp_mlp_norm_64size_vec_rots.yaml
#python /data/vision/polina/users/mfirenze/cSVR/train_from_bulk_wdb_yaml.py --config ./experiments/jan4_train_params_check_new_config.yaml


# export JOB_NAME="jan4_train_params_check_new_config_a6000"
# envsubst < job1.sh | sbatch 


# export JOB_NAME="order_rots_a6000"
# envsubst < job1.sh | sbatch 


# export JOB_NAME="new_imp_mlp_dr0.2"
# envsubst < job1.sh | sbatch 

# export JOB_NAME="low_mot_vec"
# envsubst < job1.sh | sbatch 


# export JOB_NAME="256_snet_h200"
# envsubst < job1.sh | sbatch 


# export JOB_NAME="fix_loss_no_plane_rot"
# envsubst < job1.sh | sbatch 

# export JOB_NAME="fix_loss_no_rot"
# envsubst < job1.sh | sbatch 

# export JOB_NAME="fix_loss__"
# envsubst < job1.sh | sbatch 

# export JOB_NAME="fix_loss_256_"
# envsubst < job1.sh | sbatch 

# export JOB_NAME="more_slices_more_images_more_channels32"
# envsubst < job1.sh | sbatch

# export JOB_NAME="more_slices_more_images_more_channels"
# envsubst < job1.sh | sbatch

# export JOB_NAME="bigger_unet"
# envsubst < job1.sh | sbatch


# export JOB_NAME="slice_attention_slice_features"
# envsubst < job1.sh | sbatch

# export JOB_NAME="only_slice_feat"
# envsubst < job1.sh | sbatch

# export JOB_NAME="more_slice_feat"
# envsubst < job1.sh | sbatch

# export JOB_NAME="more_slice_feat"
# envsubst < job1.sh | sbatch

# export JOB_NAME="stack_predict_loss_fix_loss_sigmoid3_print_val"
# envsubst < job1.sh | sbatch

# export JOB_NAME="mlp_9_out_flip_second"
# envsubst < job1.sh | sbatch


# export JOB_NAME="mlp_config_redo_check_randomness"
# envsubst < job1.sh | sbatch

# export JOB_NAME="mlp_stack_harder360_rot_out_45"
# envsubst < job1.sh | sbatch

# export JOB_NAME="mlp_stack_fixed_neg_360_rot_out_180"git remote -v


# envsubst < job1.sh | sbatch


# export JOB_NAME="mlp_stack_fixed_neg_360_rot_also_multi2"
# envsubst < job1.sh | sbatch
# export JOB_NAME="rot40_nodes100_bigger_model_v2_out_plane12_augs_v2_no_ms_loss_rep_a6000"
# envsubst < job1.sh | sbatch

# export JOB_NAME="argparse_test"
# envsubst < job1.sh | sbatch

# export JOB_NAME="test_old512"
# envsubst < job1.sh | sbatch

# export JOB_NAME="print_params128"
# envsubst < job1.sh | sbatch

# export JOB_NAME="two_layer_a6000_"
# envsubst < job1.sh | sbatch

# export JOB_NAME="rot40_nodes100_bigger_model_v2_out_plane12_augs_v2_no_ms_loss_rep_a6000"
# envsubst < job1.sh | sbatch


# export JOB_NAME="subtle_augs"
# envsubst < job1.sh | sbatch

# export JOB_NAME="nter_project_cd_ot_false2_no_clamp"
# envsubst < job1.sh | sbatch

# export JOB_NAME="_inter_project_cd"
# envsubst < job1.sh | sbatch

# export JOB_NAME="edges_same_mot_correct_mask_random_rep"
# envsubst < job1.sh | sbatch

# export JOB_NAME="simplified2_no_conv_full_mask"
# envsubst < job1.sh | sbatch


# export JOB_NAME="simp2_test"
# envsubst < job1.sh | sbatch


# export JOB_NAME="flow_SNet3d2_256_multi_crop_redistribute2"
# envsubst < job1.sh | sbatch

# export JOB_NAME="simp_l2_slice_loss_v2"
# envsubst < job1.sh | sbatch

# export JOB_NAME="simp_l2_slice_w10000"
# envsubst < job1.sh | sbatch


# export JOB_NAME="simp_bottleneck_simp_mask_correct_no_aug_a6000"
# envsubst < job1.sh | sbatch

# export JOB_NAME="simp_slice_dif_flow_block_rep_a6000"
# envsubst < job1.sh | sbatch



# export JOB_NAME="_feta3d0_multi_stack_svr_final_sb2_crop_flow_SNet3d2_512_multi_crop_l22_loss_grid_btneck_interp_no_ot_resume"
# envsubst < job1.sh | sbatch

# export JOB_NAME="restart_feta3d0_multi_stack_svr_final_sb2_crop_flow_SNet3d2_512_multi_crop_l22_loss_grid_btneck_interp_no_ot_resume"
# envsubst < job1.sh | sbatch
