#!/bin/zsh
#SBATCH --job-name=multiprocessing_abcrown
#SBATCH --time=20:00:00
#SBATCH --err /home/rwth1650/job_logs/multiprocessing_abcrown_err_%J.txt
#SBATCH --out /home/rwth1650/job_logs/multiprocessing_abcrown_out_%J.txt
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
cd ..

python -u run_abcrown_mnist_experiment.py