#!/bin/zsh
#SBATCH --job-name=sequential_autoattack
#SBATCH --time=01:00:00
#SBATCH --err /home/rwth1650/job_logs/sequential_autoattack_err_%J.txt
#SBATCH --out /home/rwth1650/job_logs/sequential_autoattack_out_%J.txt
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --account=rwth1650
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nils.eberhardt@rwth-aachen.de

export CONDA_ROOT=$HOME/miniconda3 
source $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"

conda activate robox
cd ..

python -u run_autoattack_mnist_experiment_sequential.py