#!/bin/bash

#SBATCH -J T_MSPD
#SBATCH -o txt/test/test_mspd.txt

echo "### START DATE=$(date)"
echo "### HOSTNAME=$(hostname)"
echo "### CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

WANDB__SERVICE_WAIT=300 python test_mspd.py --model='samspd_conv1' --result_path='results/results_mspd'

echo "###"
echo "### END DATE=$(date)"
