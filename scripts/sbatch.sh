#!/usr/bin/bash

#SBATCH -J U-net-8down
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_ugrad
#SBATCH -w aurora-g2
#SBATCH -t 1-0
#SBATCH -o /data/whgdk0911/logs/U-net-8down-%A.out

# Print current working directory
pwd

# Print the path to the Python interpreter
which python

# Print the hostname of the compute node
hostname

# Export PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/data/whgdk0911/repos/capstone_recycling/src/"

# Change directory to the location of train.py
cd /data/whgdk0911/repos/capstone_recycling/src/

# Run the Python script
python train.py logger=wandb.yaml

# Exit with status 0 (indicating successful execution)
exit 0
