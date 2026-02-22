#!/bin/bash
set -e

# python utils_evaluation_crop.py --folder_save test_save --model_folder feta3d0_multi_stack_svr_final_sb2_crop_flow_SNet3d2_512_multi_crop_l22_loss_grid_no_aug_flip --img_num 0 --gt False --hres False
# python utils_evaluation_crop.py --folder_save clin_case1 --model_folder /data/vision/polina/users/mfirenze/svr_my_train_2024/checkpoints/feta3d0_multi_stack_svr_final_sb2_crop_flow_SNet3d2_512_multi_crop_l22_loss_grid_no_aug_flip --img_num 0 --gt False --hres False


# bash evaluate_pipeline_crop.sh val_17__ 0 evaluation_results.json only_svr feta3d0_multi_stack_svr_final_sb2_crop_flow_SNet3d2_512_multi_crop_l22_loss_grid_no_repeat_proj_psf_v3
# bash evaluate_pipeline_crop.sh clin_0_order_180_test 0 evaluation_results_clin.json only_svr feta3d0_multi_stack_svr_final_sb2_crop_flow_SNet3d2_512_multi_crop_l22_loss_grid_clin_correct_zoom_plane_rot180 False True True True
# bash evaluate_pipeline_crop.sh clin_0_order_180_nesvor_gt 0 evaluation_results_clin.json only_svr feta3d0_multi_stack_svr_final_sb2_crop_flow_SNet3d2_512_multi_crop_l22_loss_grid_clin_correct_zoom_plane_rot180 True True True True
# bash evaluate_pipeline_crop.sh clin_0 0 evaluation_results_clin.json only_svr feta3d0_multi_stack_svr_final_sb2_crop_flow_SNet3d2_512_multi_crop_l22_loss_grid_clin_correct_zoom_plane_rot180
run_name="$1"
img_num="$2"
json_path="$3"
recon_method="$4"
model_path="$5"
gt="$6"
hres="$7"
clin="$8"

#rot="$9"  # specify the commit hash you want to run
gt_run_name="${9}"
rot="${10:-0}"
translation="${11:-0}"
noise="${12:-0}"
echo $recon_method

# new to do versioning
# git checkout $GIT_COMMIT
# echo "Running commit: $GIT_COMMIT"

if [ -z "$run_name" ] || [ -z "$img_num" ]; then
  exit 1
fi

echo "Using run_name: $run_name"
echo "Using img_num: $img_num"
echo "Using gt run name: $gt_run_name"


echo "Running inference"
source /data/vision/polina/users/mfirenze/.bashrc
source /data/vision/polina/users/mfirenze/miniconda3/etc/profile.d/conda.sh
conda activate 4DCNN_env_freesurfer
# conda activate 4DCNN_env_h200
# run_name="val15"
# img_num=0


output_file="svr_${run_name}.nii.gz"
output_sim="sim_${run_name}"
output_sim_o="sim_o_${run_name}"

ckpt_path="/data/vision/polina/users/mfirenze/svr_my_train_2024/checkpoints/${model_path}"
ckpt_path="/data/vision/polina/users/mfirenze/cSVR/checkpoints/${model_path}"

folder_save="/data/vision/polina/users/mfirenze/svr_my_train_2024/saved_outputs/$run_name"
folder_save="/data/vision/polina/users/mfirenze/synth_data_svr/$run_name"

save_gt="/data/vision/polina/users/mfirenze/splatting_exp/inference_recons/$output_file"
save_gt="/data/vision/polina/users/mfirenze/synth_data_svr/$run_name/$output_file"
echo "Input in"
echo $rot
echo $translation
echo $noise
python utils_evaluation_crop.py --folder_save $folder_save --model_folder $ckpt_path --img_num $img_num --gt "$gt" --hres "$hres" --save_gt $save_gt --clin "$clin" --rot "$rot" --translation "$translation" --noise "$noise"

echo "Saving slices in 3D"
python /data/vision/polina/users/mfirenze/svr_my_train_2024/slice_saver.py \
    --num_vols 3 --fold_save $folder_save  \
    --folder slices_$run_name --vol_folder vol_$run_name --pth_folder pth_slices_stack0_$run_name --img_num $img_num --clin "$clin"

echo "Reconstruct in 3D"
source /data/vision/polina/users/mfirenze/.bashrc
source /data/vision/polina/users/mfirenze/miniconda3/etc/profile.d/conda.sh
#conda activate nesvor_env2

#conda activate nesvor_env2_clone
#conda activate nesvor_env2_copy
conda activate nesvor_env2_copy_monai


# if [ -z "$gt_run_name" ]; then
#     echo "Using nesvor_env2 for non gt recon"
#     conda activate nesvor_env2
# else
#     echo "Using nesvor_env2_clone for gt recon"
#     conda activate nesvor_env2_clone
# fi
cd /data/vision/polina/users/mfirenze/splatting_exp/outputs_gen_vols




echo "Run SVR recon!"
echo "GT AND CLIN"
echo $recon_method
echo $gt

echo $clin
# if recon method only_svr if [ "$recon_method" == "only_svr" ] || [ "$gt" == "True" ];  then
if [[ ( "$recon_method" == "only_svr" && "$gt" == "False" ) || ( "$gt" == "True" && "$clin" == "False" && "$recon_method" != "only_svr_no_refine" ) ]]; then

    echo "Running nesvor svr"
    nesvor svr --input-slice slices_$run_name \
        --output-volume /data/vision/polina/users/mfirenze/splatting_exp/inference_recons/$output_file  \
        --simulated-slices  /data/vision/polina/users/mfirenze/splatting_exp/inference_recons/$output_sim_o \
        --output-slices  /data/vision/polina/users/mfirenze/splatting_exp/inference_recons/$output_sim --no-global-exclusion \
        --n-iter 5 --n-iter-rec 3 

elif [[ ( "$recon_method" == "only_svr_no_refine" ) ]]; then

    echo "Running only_svr_no_refine"
    nesvor svr --input-slice slices_$run_name \
        --output-volume $folder_save/$output_file  \
        --simulated-slices  $folder_save/$output_sim_o \
        --output-slices  $folder_save/$output_sim --no-global-exclusion \
        --n-iter 5 --n-iter-rec 3 --no-pose-refine




elif [[ "$recon_method" == "only_nesvor" || ( "$gt" = "True" && "$clin" = "True" )]]; then
    echo "Run only nesvor"
    echo $output_file
    cd /data/vision/polina/users/mfirenze/svr_my_train_2024/saved_outputs/$run_name

    cd $folder_save
    stack0="${img_num}_stack0.nii.gz"
    stack1="${img_num}_stack1.nii.gz"
    stack2="${img_num}_stack2.nii.gz"


    mask0="${img_num}_stack0_mask.nii.gz"
    mask1="${img_num}_stack1_mask.nii.gz"
    mask2="${img_num}_stack2_mask.nii.gz" # --no-slice-scale

    # nesvor reconstruct --input-stacks $stack0 $stack1 $stack2 \
    #     --stack-masks $mask0 $mask1 $mask2 \
    #     --seed 1 --image-regularization L2 \
    #     --output-volume /data/vision/polina/users/mfirenze/splatting_exp/inference_recons/$output_file  \
    #     --simulated-slices  /data/vision/polina/users/mfirenze/splatting_exp/inference_recons/$output_sim  \

    nesvor reconstruct --input-stacks $stack0 $stack1 $stack2 \
        --stack-masks $mask0 $mask1 $mask2 \
        --seed 1 \
        --output-volume /data/vision/polina/users/mfirenze/splatting_exp/inference_recons/$output_file  \
        --simulated-slices  /data/vision/polina/users/mfirenze/splatting_exp/inference_recons/$output_sim  \

    # nesvor reconstruct --input-stacks $stack0 $stack1 $stack2 \
    #     --output-volume /data/vision/polina/users/mfirenze/splatting_exp/inference_recons/$output_file  \
    #     --simulated-slices  /data/vision/polina/users/mfirenze/splatting_exp/inference_recons/$output_sim 


elif [[ "$recon_method" == "only_svort" ]]; then
    echo "Run only svort"
    cd /data/vision/polina/users/mfirenze/svr_my_train_2024/saved_outputs/$run_name

    cd $folder_save
    stack0="${img_num}_stack0.nii.gz"
    stack1="${img_num}_stack1.nii.gz"
    stack2="${img_num}_stack2.nii.gz"

    mask0="${img_num}_stack0_mask.nii.gz"
    mask1="${img_num}_stack1_mask.nii.gz"
    mask2="${img_num}_stack2_mask.nii.gz"


    nesvor svr --input-stacks $stack0 $stack1 $stack2 \
        --stack-masks $mask0 $mask1 $mask2 \
        --output-volume /data/vision/polina/users/mfirenze/splatting_exp/inference_recons/$output_file  \
        --output-slices /data/vision/polina/users/mfirenze/splatting_exp/inference_recons/$output_sim \
        --no-global-exclusion --n-iter 5 --n-iter-rec 3 \
        --verbose 2 
        #--debug 

elif [[ "$recon_method" == "svrtk" ]]; then
    echo "Run svrtk"

    echo "DONE WITH DOCKER"

    cd /data/vision/polina/users/mfirenze/svr_my_train_2024/saved_outputs/$run_name
    stack0="${img_num}_stack0.nii.gz"
    stack1="${img_num}_stack1.nii.gz"
    stack2="${img_num}_stack2.nii.gz"

    mask0="${img_num}_stack0_mask.nii.gz"
    mask1="${img_num}_stack1_mask.nii.gz"
    mask2="${img_num}_stack2_mask.nii.gz"

    newgrp docker
    echo "RUN DOCKER"


    docker run --rm \
    --user $(id -u) \
    --mount type=bind,source=/data/vision/polina/users/mfirenze,target=/data/vision/polina/users/mfirenze \
    -e run_name="$run_name" \
    -e output_sim="$output_sim" \
    -e output_file="$output_file" \
    -e img_num="$img_num" \
    fetalsvrtk/svrtk \
    bash -c '
            echo "RUN svrtk"
            cd /data/vision/polina/users/mfirenze/svr_my_train_2024/saved_outputs/$run_name
            stack0="${img_num}_stack0.nii.gz"
            stack1="${img_num}_stack1.nii.gz"
            stack2="${img_num}_stack2.nii.gz"

            mask0="${img_num}_stack0_mask.nii.gz"
            mask1="${img_num}_stack1_mask.nii.gz"
            mask2="${img_num}_stack2_mask.nii.gz"
            mirtk reconstruct /data/vision/polina/users/mfirenze/splatting_exp/inference_recons/$output_file 3 $stack1 $stack1 $stack0 -mask $mask1 -resolution 0.8 -debug

    '

    echo $output_file
    cd /data/vision/polina/users/mfirenze/svr_my_train_2024
    echo "running python copy_slices.py"
    python copy_slices.py --src /data/vision/polina/users/mfirenze/svr_my_train_2024/saved_outputs/$run_name --dest /data/vision/polina/users/mfirenze/splatting_exp/inference_recons/$output_sim


elif [[ "$recon_method" = "mine_nesvor_iter"  ]]; then
    echo "Running mine nesvor iter"
    nesvor reconstruct --input-slice slices_$run_name \
        --output-volume /data/vision/polina/users/mfirenze/splatting_exp/inference_recons/$output_file  \
        --simulated-slices  /data/vision/polina/users/mfirenze/splatting_exp/inference_recons/$output_sim

# if recon method mine_nesvor_iter
elif [ "$recon_method" == "only_svort" ]; then


    echo "Running svort"
    cd /data/vision/polina/users/mfirenze/svr_my_train_2024/saved_outputs/$run_name

    cd $folder_save

    stack0="${img_num}_stack0.nii.gz"
    stack1="${img_num}_stack1.nii.gz"
    stack2="${img_num}_stack2.nii.gz"

    mask0="${img_num}_stack0_mask.nii.gz"
    mask1="${img_num}_stack1_mask.nii.gz"
    mask2="${img_num}_stack2_mask.nii.gz"


    nesvor svr --input-stacks $stack0 $stack1 $stack2 \
        --stack-masks $mask0 $mask1 $mask2 \
        --output-volume /data/vision/polina/users/mfirenze/splatting_exp/inference_recons/$output_file  \
        --output-slices  /data/vision/polina/users/mfirenze/splatting_exp/inference_recons/$output_sim \
        --verbose 2


# if recon method mine_nesvor_iter
elif [ "$recon_method" == "mine_nesvor_no_opt" ]; then
    echo "Running mine nesvor iter"
    nesvor reconstruct --input-slice slices_$run_name \
        --output-volume /data/vision/polina/users/mfirenze/splatting_exp/inference_recons/$output_file  \
        --no-transformation-optimization --simulated-slices  /data/vision/polina/users/mfirenze/splatting_exp/inference_recons/$output_sim 

elif [ "$recon_method" == "mine_nesvor_iter1k" ]; then
    echo "Running mine nesvor iter"
    nesvor reconstruct --input-slice slices_$run_name \
        --output-volume /data/vision/polina/users/mfirenze/splatting_exp/inference_recons/$output_file  \
        --n-iter 1000  --simulated-slices  /data/vision/polina/users/mfirenze/splatting_exp/inference_recons/$output_sim 


elif [ "$recon_method" == "only_nesvor_1k" ]; then
    echo "Run only nesvor"
    cd /data/vision/polina/users/mfirenze/svr_my_train_2024/saved_outputs/$run_name
    cd $folder_save
    stack0="${img_num}_stack0.nii.gz"
    stack1="${img_num}_stack1.nii.gz"
    stack2="${img_num}_stack2.nii.gz"

    mask0="${img_num}_stack0_mask.nii.gz"
    mask1="${img_num}_stack1_mask.nii.gz"
    mask2="${img_num}_stack2_mask.nii.gz"

    nesvor reconstruct --input-stacks $stack0 $stack1 $stack2 \
        --stack-masks $mask0 $mask1 $mask2 \
        --output-volume /data/vision/polina/users/mfirenze/splatting_exp/inference_recons/$output_file  \
        --n-iter 1000 --simulated-slices  /data/vision/polina/users/mfirenze/splatting_exp/inference_recons/$output_sim 


elif [ "$recon_method" == "only_nesvor_svr_recon" ]; then
    echo "Run only nesvor with SVR recon"
    cd /data/vision/polina/users/mfirenze/svr_my_train_2024/saved_outputs/$run_name
    cd $folder_save
    stack0="${img_num}_stack0.nii.gz"
    stack1="${img_num}_stack1.nii.gz"
    stack2="${img_num}_stack2.nii.gz"


    mask0="${img_num}_stack0_mask.nii.gz"
    mask1="${img_num}_stack1_mask.nii.gz"
    mask2="${img_num}_stack2_mask.nii.gz"
    output_file_nesvor="nesvor_${run_name}.nii.gz"
    output_sim_nesvor="sim_${run_name}_nesvor"

    echo $output_file_nesvor
    echo $output_sim_nesvor

    nesvor reconstruct --input-stacks $stack0 $stack1 $stack2\
        --stack-masks $mask0 $mask1 $mask2 \
        --output-volume /data/vision/polina/users/mfirenze/splatting_exp/inference_recons/$output_file_nesvor \
        --simulated-slices  /data/vision/polina/users/mfirenze/splatting_exp/inference_recons/$output_sim_nesvor \

    nesvor svr --input-slice /data/vision/polina/users/mfirenze/splatting_exp/inference_recons/$output_sim_nesvor \
        --output-volume /data/vision/polina/users/mfirenze/splatting_exp/inference_recons/$output_file  \
        --simulated-slices  /data/vision/polina/users/mfirenze/splatting_exp/inference_recons/$output_sim \

fi

echo "GT AND CLIN"
echo $gt
echo $clin

if [[ "$gt" == "False" && "$clin" == "False" ]]; then
    cd /data/vision/polina/users/mfirenze/splatting_exp/inference_recons
    #reference_file="svr_val15_gt.nii.gz"
    reference_file="svr_${run_name}_gt.nii.gz"
    reference_gt_file="svr_${run_name}_gt_brain.nii.gz"
    reg_file="reg_${output_file}"
    reg_gt_file="reg_gt_${output_file}"


    # using only one line
   
    reference_gt_file="svr_${gt_run_name}${img_num}"_gt_brain.nii.gz
    output_sim_gt="sim_${gt_run_name}${img_num}_gt"
    reference_file="svr_${gt_run_name}${img_num}_gt.nii.gz"

    echo $reg_gt_file
    echo $reference_gt_file
    echo $output_sim_gt
    echo $reference_file

    # register to ground truth poses
    echo "Run ANTS registration"
    /data/vision/polina/users/dey/libraries/ants/install/bin/antsRegistration \
    -d 3 \
    -o ["reg_${img_num}_${run_name}", $reg_file] \
    -r [$reference_file, $output_file, 1] \
    -n Linear \
    -m MeanSquares[$reference_file,  $output_file,  1, , Regular, 1., 1] \
    -t Rigid[0.2] \
    -s 3x2x1x0x2x1x0 \
    -f 2x2x2x2x1x1x1 \
    -c 100x100x100x100x100x100x100 \
    -v

    # register to original volume
    echo "Run ANTS registration not CLIN"
    /data/vision/polina/users/dey/libraries/ants/install/bin/antsRegistration \
    -d 3 \
    -o ["reg_gt_${img_num}_${run_name}", $reg_gt_file] \
    -r [$reference_gt_file, $output_file, 1] \
    -n Linear \
    -m MeanSquares[$reference_gt_file,  $output_file,  1, , Regular, 1., 1] \
    -t Rigid[0.2] \
    -s 3x2x1x0x2x1x0 \
    -f 2x2x2x2x1x1x1 \
    -c 100x100x100x100x100x100x100 \
    -v

    python /data/vision/polina/users/mfirenze/svr_my_train_2024/get_TRE_from_file_outputs.py \
        --mask_folder /data/vision/polina/users/mfirenze/splatting_exp/outputs_gen_vols/slices_$run_name \
        --folder_est /data/vision/polina/users/mfirenze/splatting_exp/inference_recons/$output_sim \
        --folder_gt /data/vision/polina/users/mfirenze/splatting_exp/inference_recons/$output_sim_gt \
        --mat_file /data/vision/polina/users/mfirenze/splatting_exp/inference_recons/"reg_"$img_num"_"$run_name"0GenericAffine.mat" \
        --gt_file  /data/vision/polina/users/mfirenze/splatting_exp/inference_recons/$reference_gt_file \
        --reg_file /data/vision/polina/users/mfirenze/splatting_exp/inference_recons/$reg_gt_file \
        --json_path /data/vision/polina/users/mfirenze/svr_my_train_2024/evaluate_metrics/$json_path \
        --img_num $img_num \
        --og_slice_path /data/vision/polina/users/mfirenze/splatting_exp/outputs_gen_vols/slices_$run_name 
fi




if [[ "$gt" == "False" && "$clin" == "True" ]]; then
    cd /data/vision/polina/users/mfirenze/splatting_exp/inference_recons
    #reference_file="svr_val15_gt.nii.gz"
    reference_file="svr_${run_name}_gt.nii.gz"
    reg_file="reg_${output_file}"


    # register to ground truth poses
    echo "Run ANTS registration and CLIN"
    /data/vision/polina/users/dey/libraries/ants/install/bin/antsRegistration \
    -d 3 \
    -o ["reg_${run_name}_${img_num}", $reg_file] \
    -r [$reference_file, $output_file, 1] \
    -n Linear \
    -m MeanSquares[$reference_file,  $output_file,  1, , Regular, 1., 1] \
    -t Rigid[0.2] \
    -s 3x2x1x0x2x1x0 \
    -f 2x2x2x2x1x1x1 \
    -c 100x100x100x100x100x100x100 \
    -v



    python /data/vision/polina/users/mfirenze/svr_my_train_2024/get_TRE_from_file_outputs.py \
        --mask_folder /data/vision/polina/users/mfirenze/splatting_exp/outputs_gen_vols/slices_$run_name \
        --folder_est /data/vision/polina/users/mfirenze/splatting_exp/inference_recons/$output_sim \
        --folder_gt /data/vision/polina/users/mfirenze/splatting_exp/inference_recons/"$output_sim"_gt \
        --mat_file /data/vision/polina/users/mfirenze/splatting_exp/inference_recons/"reg_"$run_name"_"$img_num"0GenericAffine.mat" \
        --gt_file  /data/vision/polina/users/mfirenze/splatting_exp/inference_recons/$reference_file \
        --reg_file /data/vision/polina/users/mfirenze/splatting_exp/inference_recons/$reg_file \
        --json_path /data/vision/polina/users/mfirenze/svr_my_train_2024/evaluate_metrics/$json_path \
        --img_num $img_num --clin "True" \
        --og_slice_path /data/vision/polina/users/mfirenze/splatting_exp/outputs_gen_vols/slices_$run_name
fi 