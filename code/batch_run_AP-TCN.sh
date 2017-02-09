#!/bin/sh -l
#SBATCH -p short
#SBATCH --gres=gpu:k80:1
#SBATCH -w werbos
#SBATCH -J my_short_job
#SBATCH -o my_short_job.log
hostname
echo $CUDA_VISIBLE_DEVICES
srun nohup python AP_TCN_dev_v0.3.py run_AP-TCN.log &
