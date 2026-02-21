#!/bin/bash
## SLURM Variables:
#SBATCH --job-name="${JOB_NAME}"
#SBATCH --output=/data/vision/polina/users/mfirenze/cSVR/train_outs/out_${JOB_NAME}.out
#SBATCH -e /data/vision/polina/users/mfirenze/cSVR/train_outs/err_${JOB_NAME}.out
#SBATCH -o /data/vision/polina/users/mfirenze/cSVR/train_outs/o_${JOB_NAME}.out
#SBATCH --partition=csail-shared-h200
#SBATCH --qos=lab-free
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=170G
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --exclude=aia-h200-8,quanta-h200-1

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

git config --global --add safe.directory /data/vision/polina/users/mfirenze/cSVR
git add -u
git commit -m "$JOB_NAME" || echo "No commited changes"

#python /data/vision/polina/users/mfirenze/cSVR/train_from_bulk_wdb_yaml.py --config ./experiments/train_params_mlp_lr0.05_stack_order_more_channels_bigger_image_more_slices32_fix_loss_no_rot_h200_mask.yaml
#python /data/vision/polina/users/mfirenze/cSVR/train_from_bulk_wdb_yaml.py --config ./experiments/train_params_mlp_lr0.05_stack_order_more_channels_bigger_image_more_slices32_fix_loss_no_rot_h200_no_plane_rot_less_mot.yaml



#python /data/vision/polina/users/mfirenze/cSVR/train_from_bulk_wdb_yaml.py --config ./experiments/train_params_mlp_lr0.05_stack_order_more_channels_bigger_image_more_slices32_fix_loss_no_rot_h200_no_plane_rot_less_mot_4096snet.yaml
#python /data/vision/polina/users/mfirenze/cSVR/train_from_bulk_wdb_yaml.py --config ./experiments/train_params_low_mot_vec_new_mlp_imp.yaml

#python /data/vision/polina/users/mfirenze/cSVR/train_from_bulk_wdb_yaml.py --config ./experiments/train_params_low_mot_vec_new_mlp_imp_drop_out0.yaml
#python /data/vision/polina/users/mfirenze/cSVR/train_from_bulk_wdb_yaml.py --config ./experiments/train_params_low_mot_vec_new_mlp_imp_mlp_norm_64size_no_rot_vec_correct_flips_in_plane_rots.yaml

#python /data/vision/polina/users/mfirenze/cSVR/train_from_bulk_wdb_yaml.py --config ./experiments/jan6_train_params_check_new_config_fix_lr.yaml


# CHANGE SNET BEFORE RUNNING THIS
python /data/vision/polina/users/mfirenze/cSVR/train_from_bulk_wdb_yaml.py --config ./experiments/feb21_node5_repeat_180_svort_no_slice_param.yaml # export JOB_NAME="jan14__mlp_norm_64size_vec_rots_rep_large_unet"

#python /data/vision/polina/users/mfirenze/cSVR/train_from_bulk_wdb_yaml.py --config ./experiments/feb17_node5_repeat_180_in_plane.yaml # export JOB_NAME="jan14__mlp_norm_64size_vec_rots_rep_large_unet"
# export JOB_NAME="feb21_node5_repeat_180_svort_no_slice_param"

# envsubst < job1_h200_2.sh | sbatch 


# export JOB_NAME="feb17_node5_repeat_clean"
# envsubst < job1_h200_2.sh | sbatch 



# export JOB_NAME="no_rot_fix_flips"
# envsubst < job1_h200.sh | sbatch 



# export JOB_NAME="mlp_norm_64size_no_rot_vec"
# envsubst < job1_h200.sh | sbatch 


# export JOB_NAME="mlp_norm_64size"
# envsubst < job1_h200.sh | sbatch 

#bash /data/vision/polina/users/mfirenze/cSVR/feta3d_svr_train_reorient_l2_wb_resume.sh


# export JOB_NAME="new_imp_mlp_dr0_h200"
# envsubst < job1_h200.sh | sbatch 

# export JOB_NAME="new_imp_mlp_dr0.1_h200"
# envsubst < job1_h200.sh | sbatch 

# export JOB_NAME="new_imp_mlp_dr0.1"
# envsubst < job1_h200.sh | sbatch 


# export JOB_NAME="new_imp_mlp"
# envsubst < job1_h200.sh | sbatch 


# export JOB_NAME="1024_snet_h200_"
# export JOB_NAME="2056_snet_h200_"
# envsubst < job1_h200.sh | sbatch 

# export JOB_NAME="4096_snet_h200_"
# envsubst < job1_h200.sh | sbatch 

# export JOB_NAME="less_mot_just_rot"
# envsubst < job1_h200.sh | sbatch 

# export JOB_NAME="no_rot_h200_no_plane"
# envsubst < job1_h200.sh | sbatch 

# export JOB_NAME="no_rot_h200"
# envsubst < job1_h200.sh | sbatch

# export JOB_NAME="no_rot_h200_mask"
# envsubst < job1_h200.sh | sbatch




# export JOB_NAME="no_rot_h200_mask"
# envsubst < job1_h200.sh | sbatch

#bash /data/vision/polina/users/mfirenze/cSVR/feta3d_svr_train_reorient_l2_wb.sh
#bash /data/vision/polina/users/mfirenze/cSVR/feta3d_svr_train_reorient_l2_wb_resume.sh

#bash /data/vision/polina/users/mfirenze/cSVR/feta3d_svr_train_reorient_l2_resume.sh
# export JOB_NAME="repeat_rot40_v2_rep_with_config_v2"
# envsubst < job1_h200.sh | sbatch

# export JOB_NAME="three_layer_h200"
# envsubst < job1_h200.sh | sbatch

# export JOB_NAME="two_layer_h200"
# envsubst < job1_h200.sh | sbatch

# export JOB_NAME="one_layer_"
# envsubst < job1_h200.sh | sbatch


# export JOB_NAME="512_model"
# envsubst < job1_h200.sh | sbatch



# export JOB_NAME="rot30_bulk180_zooms"
# envsubst < job1_h200.sh | sbatch

# export JOB_NAME="rot0_bulk180_zooms_small_model"
# envsubst < job1_h200.sh | sbatch

# export JOB_NAME="rot20_bulk180_zooms"
# envsubst < job1_h200.sh | sbatch


# export JOB_NAME="repeat_rot40_v2_rep_small_180"
# envsubst < job1_h200.sh | sbatch

# export JOB_NAME="repeat_rot40_v2_rep_small_180_tr10"
# envsubst < job1_h200.sh | sbatch

# export JOB_NAME="repeat_rot40_v2_rep_small_180_tr10_big"
# envsubst < job1_h200.sh | sbatch


# export JOB_NAME="repeat_rot40_v2_rep_small_180_tr10_zooms"
# envsubst < job1_h200.sh | sbatch

# export JOB_NAME="repeat_rot40_v2_rep_small_180_tr10_smooth_zooms"
# envsubst < job1_h200.sh | sbatch

# export JOB_NAME="rep_no_zoom_config"
# envsubst < job1_h200.sh | sbatch


# export JOB_NAME="repeat_rot40_v2_with_bulk_rotations180_no_smooth"
# envsubst < job1_h200.sh | sbatch


# export JOB_NAME="repeat_rot40_v2_with_bulk_rotations180_tr10_big"
# envsubst < job1_h200.sh | sbatch

# export JOB_NAME="repeat_rot40_v2"
# envsubst < job1_h200.sh | sbatch

# export JOB_NAME="_rot40_nodes100_bulk_rot_config_big"
# envsubst < job1_h200.sh | sbatch

# export JOB_NAME="_rot40_nodes100_smaller_model_more_zooms_smooth_also_h200"
# envsubst < job1_h200.sh | sbatch

# export JOB_NAME="rot40_nodes100_smaller_model_smooth_augs"
# envsubst < job1_h200.sh | sbatch

# export JOB_NAME="rot40_nodes100_bigger_model_v2_out_plane12_augs_v2_no_ms_loss_rep"
# envsubst < job1_h200.sh | sbatch

# export JOB_NAME="rot40_nodes100_bigger_model_v2_out_plane12_augs_v2_no_stack_info_"
# envsubst < job1_h200.sh | sbatch

# export JOB_NAME="rot40_nodes100_bigger_model_v2"
# envsubst < job1_h200.sh | sbatch

# export JOB_NAME="rot40_nodes100_f"
# envsubst < job1_h200.sh | sbatch

# export JOB_NAME="rot40_nodes100_super_big_model"
# envsubst < job1_h200.sh | sbatch

# export JOB_NAME="inter_project_cd_h200_ot_false2_less_restrictions_bulk180"
# envsubst < job1_h200.sh | sbatch

# export JOB_NAME="inter_project_cd_h200_ot_false2_less_restrictions_smaller_eps"
# envsubst < job1_h200.sh | sbatch

# export JOB_NAME="inter_plane_rot180"
# envsubst < job1_h200.sh | sbatch

# export JOB_NAME="hres_compare_to_hres"
# envsubst < job1_h200.sh | sbatch

# export JOB_NAME="inter_hres_no_middle_project"
# envsubst < job1_h200.sh | sbatch


# export JOB_NAME="reg_model_512_hres"
# envsubst < job1_h200.sh | sbatch


# export JOB_NAME="reg_model_512_hres"
# envsubst < job1_h200.sh | sbatch

# export JOB_NAME="edges_same_mot_correct_mask_random_rep_anyexcept1"
# envsubst < job1_h200.sh | sbatch

# export JOB_NAME="_proj_try_200_simp"
# envsubst < job1_h200.sh | sbatch

# export JOB_NAME="simp2"
# envsubst < job1_h200.sh | sbatch

# export JOB_NAME="repeat_simp_no_edges_extra_layer_"
# envsubst < job1_h200.sh | sbatch

# export JOB_NAME="proj_clamp_H_eq"
# envsubst < job1_h200.sh | sbatch

# export JOB_NAME="simp_extra_bottleneck_check"
# envsubst < job1_h200.sh | sbatch

# export JOB_NAME="small_arch_128_v2"
# envsubst < job1_h200.sh | sbatch

# export JOB_NAME="simp_l2_affine_loss_h200_order_fix"
# envsubst < job1_h200.sh | sbatch

# export JOB_NAME="simp_l2_slice_loss_weight100"
# envsubst < job1_h200.sh | sbatch


# export JOB_NAME="simp_l2_slice_loss_v3"
# envsubst < job1_h200.sh | sbatch

# export JOB_NAME="simp_l2_affine_loss_rep"
# envsubst < job1_h200.sh | sbatch

# export JOB_NAME="simp_inter_loss"
# envsubst < job1_h200.sh | sbatch



# export JOB_NAME="simp_slice_dif_flow_block_rep"
# envsubst < job1_h200.sh | sbatch


# export JOB_NAME="simp_conv_slice_diff_correct_m_h6"
# envsubst < job1_h200.sh | sbatch

# export JOB_NAME="simp_conv_bulk_rot"
# envsubst < job1_h200.sh | sbatch

# export JOB_NAME="resume_h200"
# envsubst < job1_h200.sh | sbatch


# export JOB_NAME="feta3d0_multi_stack_svr_final_sb2_crop_flow_SNet3d2_512_multi_crop_l22_loss_grid_btneck_interp_no_ot_resume_one_layer_l2_scaled"
# envsubst < job1_h200.sh | sbatch



# export JOB_NAME="feta3d0_multi_stack_svr_final_sb2_crop_flow_SNet3d2_512_multi_crop_l22_loss_grid_multiscale_simp_inter_loss_lr0.6"
# envsubst < job1_h200.sh | sbatch

# export JOB_NAME="feta3d0_multi_stack_svr_final_sb2_crop_flow_SNet3d2_512_multi_crop_l22_loss_grid_multiscale_simp_inter_loss_restart"
# envsubst < job1_h200.sh | sbatch

# export JOB_NAME="feta3d0_multi_stack_svr_final_sb2_crop_flow_SNet3d2_512_multi_crop_l22_loss_grid_multiscale_simp_inter_loss_restart_v2"
# envsubst < job1_h200.sh | sbatch

# export JOB_NAME="inter_loss_edges_include_simp"
# envsubst < job1_h200.sh | sbatch

# export JOB_NAME="conv_splat_mask_false"
# envsubst < job1_h200.sh | sbatch

# export JOB_NAME="mask_correct_actually_no_aug_v2"
# envsubst < job1_h200.sh | sbatch

# export JOB_NAME="conv_splat_mask_false_v3"
# envsubst < job1_h200.sh | sbatch