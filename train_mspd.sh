#!/bin/bash

#SBATCH -J MSPD
#SBATCH -o txt/train/train_mspd.txt

echo "### START DATE=$(date)"
echo "### HOSTNAME=$(hostname)"
echo "### CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

WANDB__SERVICE_WAIT=300 python train_mspd.py --model='mspd' --result_path='results/results_mspd' --batch_size=2

echo "###"
echo "### END DATE=$(date)"
