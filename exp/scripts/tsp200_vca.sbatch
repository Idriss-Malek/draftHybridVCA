#!/bin/bash
#SBATCH --job-name=tsp200VCA
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=30:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
mkdir -p logs

module load StdEnv/2023
module load gcc/12.3
module load cuda/12.2

source /home/idrissm/projects/def-mh541-ab/idrissm/vca/bin/activate

PY=/home/idrissm/projects/def-mh541-ab/idrissm/vca/bin/python
SCRIPT=/home/idrissm/projects/def-mh541-ab/idrissm/neighborVCA/exp/TSP/tsp200_vca.py
CFG=/home/idrissm/projects/def-mh541-ab/idrissm/neighborVCA/exp/configs/TSP200.json
DATA=/home/idrissm/projects/def-mh541-ab/idrissm/neighborVCA/dataset/ml4co_tsp/tsp200_concorde_10.71908.txt

SEED=${SEED:-0}
INSTANCES=${INSTANCES:-3}

srun --export=ALL "$PY" \
  "$SCRIPT" \
  --config "$CFG" \
  --dataset "$DATA" \
  --seed "$SEED" \
  --max-instances "$INSTANCES"
