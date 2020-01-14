#!/usr/bin/env bash

#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=5GB
#SBATCH --mail-type=FAIL

source activate tfgpu
echo "FILENAME: " ${filename}
module purge
module load cudnn/9.0v7.3.0.29
bash ${filename}
