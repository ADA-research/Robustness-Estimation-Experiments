#!/bin/bash
#SBATCH --job-name=LION_exps
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=20:30:00
#SBATCH --array=0-9

# Submit from experiments/imagenet/ (sbatch run_batches.sh); adapt SCRATCH_DIR to your cluster.

# Load Conda environment
export PATH="$HOME/miniconda3/bin:$PATH"
source ~/.bashrc
conda activate verona_env

# Temp directories
SCRATCH_DIR="${SCRATCH_DIR:-/path/to/scratch}"
export TMP="$SCRATCH_DIR"
export TEMP="$SCRATCH_DIR"
export TMPDIR="$SCRATCH_DIR"
mkdir -p $TMPDIR

# Paths
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CSV="$SCRIPT_DIR/arguments_eva.txt"
SCRIPT="$SCRIPT_DIR/main.py"

# Pick the correct line for this array task (skip header)
LINE=$(awk -v idx=$SLURM_ARRAY_TASK_ID 'NR==idx+2 {print; exit}' "$CSV")

# Split line into variables
IFS=, read -r model_name attack_method indice batch_size start <<< "$LINE"

echo "=============================================================="
echo "Job $SLURM_ARRAY_TASK_ID"
echo "Model:  $model_name"
echo "Attack: $attack_method"
echo "Indice: $indice"
echo "Batch:  $batch_size"
echo "=============================================================="

# Run Python
python "$SCRIPT" \
  --model_name "$model_name" \
  --attack_method "$attack_method" \
  --indices "$indice" \
  --batch_size "$batch_size"\
  --start "$start"
