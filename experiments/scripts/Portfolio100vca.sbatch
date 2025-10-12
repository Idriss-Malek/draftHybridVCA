#!/bin/bash
#SBATCH --job-name=portfolio100VCA
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=11:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
mkdir -p logs

module load StdEnv/2023
module load gcc/12.3
module load cuda/12.2

# Activate venv
source /home/idrissm/projects/def-mh541-ab/idrissm/vca/bin/activate

PY=/home/idrissm/projects/def-mh541-ab/idrissm/vca/bin/python
SCRIPT=/home/idrissm/projects/def-mh541-ab/idrissm/neighborVCA/experiments/Portfolio/Portfolio100/vca.py
CFG=/home/idrissm/projects/def-mh541-ab/idrissm/neighborVCA/experiments/configs/Portfolio100.json

# Default seed if not set
SEED=${SEED:-0}

srun --export=ALL "$PY" \
  "$SCRIPT" \
  --config "$CFG" \
  --seed "$SEED"
