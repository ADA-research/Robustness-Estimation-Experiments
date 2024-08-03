#!/bin/zsh
#SBATCH --job-name=multiprocessing_abcrown
#SBATCH --time=01:00:00
#SBATCH --err ./logs/multiprocessing_errors.err
#SBATCH --out ./logs/multiprocessing_output.out
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --account=rwth1650
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=6
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aaronberger@hotmail.de

export CONDA_ROOT=$HOME/miniconda3 
source $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"

conda activate robox

python -u create_dist_multi_processing.py