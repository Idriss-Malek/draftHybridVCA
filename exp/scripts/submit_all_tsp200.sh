#!/usr/bin/env bash
#SBATCH --job-name=tsp200All
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
mkdir -p logs

SCRIPTS_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Submit each job with seeds 0,1,2 covering first 3 instances
for job in tsp200_vca tsp200_adavca tsp200_gflownet tsp200_adagfn tsp200_sa; do
  for SEED in 0; do
    sbatch --export=SEED=${SEED},INSTANCES=3 "$SCRIPTS_DIR/${job}.sbatch"
  done
  # ensure we don't overwhelm scheduler
  sleep 1
done
