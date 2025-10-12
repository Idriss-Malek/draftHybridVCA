#!/bin/bash
#SBATCH --job-name=tsp64_rnn_PPOVCA
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1                 # or: --gres=gpu:a100:1  (site-dependent)
# #SBATCH --account=def-mh541-ab      # uncomment if required
#SBATCH --output=logs/%x-%j.out      # ensure logs/ exists BEFORE sbatch

set -euo pipefail

module --force purge
module load StdEnv/2023
module load gcc/12.3
module load cuda/12.2

# Env for Triton / CUDA JIT
export CC="$(which gcc)"
export CXX="$(which g++)"
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/triton-cache"
mkdir -p "$TRITON_CACHE_DIR"

hostname
source /home/idrissm/projects/def-mh541-ab/idrissm/vca/bin/activate

echo "==== $(date) | job ${SLURM_JOB_ID:-N/A} | node ${HOSTNAME:-N/A} ===="
echo "model=${model-} instance=${instance-} sa_anneal=${sa_anneal-}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-}"
nvidia-smi --query-gpu=index,uuid,name --format=csv,noheader

srun python /home/idrissm/projects/def-mh541-ab/idrissm/neighborVCA/runs/tsp_ppo.py\
    --config /home/idrissm/projects/def-mh541-ab/idrissm/neighborVCA/scripts/VCA/tsp/configVCA64.json
