#!/bin/bash
## SLURM Variables:
#SBATCH --job-name="summarize_results"
#SBATCH --output=/data/vision/polina/users/mfirenze/cSVR/train_outs/out_summ_results.out
#SBATCH -e /data/vision/polina/users/mfirenze/cSVR/train_outs/err_summ_results.out
#SBATCH -o /data/vision/polina/users/mfirenze/cSVR/train_outs/o_summ_results.out
#SBATCH --partition=polina-all
#SBATCH -A vision-polina
#SBATCH --qos=vision-polina-main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00-00:10:00


ROOT=$1
DIR=${2:-evaluate_metrics}  # default to 'evaluate_metrics' if not given
OUT_F="combined_${ROOT}.json"
OUT="$DIR/combined_${ROOT}.json"

echo "[" > "$OUT"

first=true
for f in "$DIR"/${ROOT}*.json; do
    if [ "$first" = true ]; then
        cat "$f" >> "$OUT"
        first=false
    else
        echo "," >> "$OUT"
        echo "Added comma before $f"
        cat "$f" >> "$OUT"
    fi
done

echo "]" >> "$OUT"

echo "Running inference"
source /data/vision/polina/users/mfirenze/.bashrc
source /data/vision/polina/users/mfirenze/miniconda3/etc/profile.d/conda.sh
conda activate 4DCNN_env_freesurfer

echo "Combined JSON saved to $OUT"
#python analyze_combined.py $OUT_F

python analyze_combined_c.py $OUT