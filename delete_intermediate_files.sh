#!/bin/bash

# Define paths
data_dir="/data/vision/polina/users/mfirenze/clin_data_processed_svr"
subjects_file="/data/vision/polina/users/mfirenze/clin_data_processed_svr/all_subjects_list.txt"

# Array to store directories for step 2
dirs_to_process=()

while IFS= read -r sub || [ -n "$sub" ]; do
    [ -z "$sub" ] && continue
    echo "Adding subject: $sub"
    dirs_to_process+=("${data_dir}/${sub}")
done < "$subjects_file"


# Delete intermediate files


# Delete files in subject directories that do not match the kept patterns
for dir in "${dirs_to_process[@]}"; do
    if [ -d "$dir" ]; then
        echo "Cleaning directory: $dir"
        find "$dir" -maxdepth 1 -type f \
            ! -name "*_sag.nii" \
            ! -name "*_cor.nii" \
            ! -name "*_axi.nii" \
            ! -name "*_sag.nii.gz" \
            ! -name "*_cor.nii.gz" \
            ! -name "*_axi.nii.gz" \
            -delete
        
        # Remove subdirectories
        rm -rf "${dir}/cSVR_files"
        rm -rf "${dir}/saved_slices"


    else
        echo "Warning: Directory $dir not found"
    fi
done