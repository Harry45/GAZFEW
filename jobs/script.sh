#!/bin/bash
#SBATCH --mail-type=BEGIN,END 
#SBATCH --mail-user=arrykrish@gmail.com
#SBATCH --time=00:10:00
#SBATCH --job-name=deeplearning_galaxyzoo
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=short
#SBATCH --cluster=htc
#SBATCH --gres=gpu:1

cd $SCRATCH || exit 1

rsync -av $HOME/example-torch/test_plot.py ./

module purge
module load CUDA/11.3.1
module load Anaconda3
export CONPREFIX=$DATA/pytorch-env39
source activate $CONPREFIX

python test_plot.py

nvidia-smi > nvidia.txt 

rsync -av --exclude=test_plot.py ./ $HOME/example-torch/outputs/