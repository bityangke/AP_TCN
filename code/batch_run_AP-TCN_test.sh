#!/bin/bash -l
#SBATCH -p long
#SBATCH --gres=gpu:k80:1
#SBATCH -w werbos
#SBATCH -J test
#SBATCH -o test.log
hostname
echo $CUDA_VISIBLE_DEVICES
source activate tf-ap
export PYTHONHOME="/home/jinchoi/anaconda2/envs/tf-ap"
which python
srun python /home/jinchoi/src/rehab/action-recog/action_proposal/AP-TCN/code/test.py
