#!/bin/bash
#SBATCH --job-name=tsp64AdaVCA
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
mkdir -p logs

module load StdEnv/2023
module load gcc/12.3
module load cuda/12.2

# Activate your venv (puts the right python first on PATH)
source /home/idrissm/projects/def-mh541-ab/idrissm/vca/bin/activate

PY=/home/idrissm/projects/def-mh541-ab/idrissm/vca/bin/python
SCRIPT=/home/idrissm/projects/def-mh541-ab/idrissm/neighborVCA/experiments/TSP/TSP64/adavca.py
CFG=/home/idrissm/projects/def-mh541-ab/idrissm/neighborVCA/experiments/configs/TSP64.json

# Optional: default seed if not set
SEED=${SEED:-0}

srun --export=ALL "$PY" \
  "$SCRIPT" \
  --config "$CFG" \
  --seed "$SEED"