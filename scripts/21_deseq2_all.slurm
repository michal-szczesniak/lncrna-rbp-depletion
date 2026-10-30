#!/bin/bash
#SBATCH --job-name=deseq2_all
#SBATCH --array=1-50
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=logs/21_deseq2_%A_%a.log
# Faza 2, krok 2.2: DESeq2 dla 876 knockdownow w 50 rownoleglych zadaniach (ok. 18 knockdownow na zadanie)
set -e
source activate_rbp.sh
conda activate "$PROJ/envs/rbp-r"
Rscript scripts/21_deseq2_all.R "$SLURM_ARRAY_TASK_ID" 50
