#!/bin/bash
#SBATCH --job-name spec-clustering
#SBATCH --output %A_%a.out
#SBATCH --time 5-0:00:00
#SBATCH -c 40
#SBATCH --mem 60000
#SBATCH --gres gpu:1
#SBATCH --array 0-7


python -u -m sufield.spec_cluster ${SLURM_ARRAY_TASK_ID} 8
