#!/bin/bash
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=arrykrish@gmail.com
#SBATCH --time=00:20:00
#SBATCH --job-name=siamese_network
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=short
#SBATCH --cluster=htc
#SBATCH --gres=gpu:1

cd $SCRATCH || exit 1

rsync -r $HOME/GAZFEW/ ./

module purge
module load CUDA/11.3.1
module load Anaconda3
export CONPREFIX=$DATA/pytorch-env39
source activate $CONPREFIX

python train.py

rsync -r ./outputs/ $HOME/GAZFEW/