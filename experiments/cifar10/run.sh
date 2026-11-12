#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=20:30:00

# Submit from experiments/cifar10/ (sbatch run.sh); adapt SCRATCH_DIR to your cluster.

export PATH="$HOME/miniconda3/bin:$PATH"
source ~/.bashrc
conda activate verona_env

SCRATCH_DIR="${SCRATCH_DIR:-/path/to/scratch}"
export TMP="$SCRATCH_DIR"
export TEMP="$SCRATCH_DIR"
export TMPDIR="$SCRATCH_DIR"
mkdir -p $TMPDIR

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

# The CIFAR-10 test set was split into chunks of 500 images (indices 0-20), one job per chunk.
# Set INDEX to the chunk to run (default: 20, the last job that was submitted).
INDEX="${INDEX:-20}"
python "$SCRIPT_DIR/main_cifar.py" --attack_method autoattack --indice "$INDEX" --batch_size 500
