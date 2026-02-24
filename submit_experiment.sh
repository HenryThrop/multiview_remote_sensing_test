#!/bin/bash

#SBATCH --job-name=dino_full_pipeline
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1                      # CRITICAL: Request 1 GPU for fine-tuning
#SBATCH --time=24:00:00                   # Request 24 hours
#SBATCH --partition=gpu                   # Make sure to use the GPU partition!
#SBATCH --output=slurm-%j.out

# ==========================================
# 1. ACTIVATE YOUR ENVIRONMENT HERE
# (e.g., module load Anaconda3, conda activate my_env)
# ==========================================
module purge
module load Anaconda3
source activate kidsat


# ==========================================
# 2. NAVIGATE TO THE REPOSITORY ROOT
# ==========================================
# Make sure to change to  actual folder path!
cd /path/to/your/MLGlobalHealth/multiview_remote_sensing/


# ==========================================
# 3. RUN THE PIPELINE
# ==========================================
echo "Starting rigorous fine-tuning and evaluation pipeline..."

# Run the master pipeline script hiding inside the dino folder.
# Make sure to change the imagery_path to your actual imagery folder!
# Run the master pipeline script hiding inside the dino folder.
python modelling/dino/run_random_experiment.py \
    --imagery_path /path/to/your/imagery \
    --num_random_tests 3

echo "Job finished!"
