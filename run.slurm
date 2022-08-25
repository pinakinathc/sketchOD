#!/bin/bash

# ##############################################################################
# Slurm submission file
# Submit via "sbatch <THIS_FILE>"
# More info: https://slurm.schedmd.com/sbatch.html
#
# EXAMPLE 04: TRAIN AND RUN A VAE ON MNIST.
# IMPORTANT: make sure that you have activated your conda environment BEFORE
# calling sbatch on this script.
# ##############################################################################

# Job identifier
#SBATCH --job-name=Generic_AA

# output pathing for logs. The "logs" directory must exist, otherwise no logs!
# Add username with %u. More info: https://slurm.schedmd.com/sbatch.html#lbAH
#SBATCH --error  logs/%x_%j.err
#SBATCH --output  logs/%x_%j.out

# Requested resources (time in hh:mm:ss)
# Jobs with more than 1 GPU should request a "big" partition, otherwise "small"
#SBATCH --nodes=1
#SBATCH --time=96:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=small
# request CPUs, see https://stackoverflow.com/a/54845007
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64GB


# main routine. Note the -u that allows Python "print" statements to be flushed
# right away (instead of buffered). This allows to check the logs in real time.

python main.py --data_dir