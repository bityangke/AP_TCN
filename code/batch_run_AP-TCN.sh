#!/bin/bash -l
#SBATCH -p long
#SBATCH --gres=gpu:k80:1
#SBATCH -J batch_run_AP-TCN
#SBATCH -o batch_run_AP-TCN-batch_exp_part2.sh.log
hostname
echo $CUDA_VISIBLE_DEVICES
source activate tf-ap
export PYTHONHOME="/home/jinchoi/anaconda2/envs/tf-ap"
which python
srun python /home/jinchoi/src/rehab/action-recog/action_proposal/AP-TCN/code/AP_TCN_dev_v0.4_batch_exp.py
