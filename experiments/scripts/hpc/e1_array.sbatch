#!/bin/bash
#SBATCH --job-name=bs4_e1
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH -o /home/anon/benchscope4/logs/e1_%A_%a.log
# E1 (WN18RR as published), seeds from e1_tasks.txt line SLURM_ARRAY_TASK_ID ("model seed").
# Unmodified Ruffinelli et al. 2020 config; LibKGE test on the best checkpoint; then the
# per-query scorer on the mapped benchmark test triples with the benchmark's own filter.
set -euo pipefail
R=/home/anon/benchscope4
export PATH=$R/env/bin:$PATH
read M SEED < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $R/scripts/hpc/e1_tasks.txt)
F=$R/runs/E1/${M}_s${SEED}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "task $SLURM_ARRAY_TASK_ID model $M seed $SEED"
cd $R/kge
if [ ! -f $F/checkpoint_best.pt ] || ! grep -q "Maximum number of epochs\|Stopping early" $F/kge.log 2>/dev/null; then
  rm -rf $F
  kge start $R/configs/e1/wnrr-$M.yaml --folder $F --job.device cuda --dataset.name wnrr --random_seed.default $SEED
fi
kge test $F --job.device cuda > $F/test.log 2>&1
grep -o "  mean_reciprocal_rank_filtered: [0-9.]*" $F/test.log | tail -1
cd /tmp
python $R/scripts/score_wn18rr.py --checkpoint $F/checkpoint_best.pt --mode E1 --target $R/target \
  --queries bench_test --filter splits --out $F/score_bench_test
echo TASK_OK
