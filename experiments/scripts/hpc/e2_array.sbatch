#!/bin/bash
#SBATCH --job-name=bs4_e2
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH -o /home/anon/benchscope4/logs/e2_%A_%a.log
# E2 (declared target), task line SLURM_ARRAY_TASK_ID of e2_tasks.txt ("model seed").
# Same config as E1 except dataset wnrr_target and O-only validation
# (user.candidate_entities_file), i.e. model selection on bench_valid over O.
# Scoring: bench_test (primary) and u_test (query-side secondary), filtered with every
# target triple. A LibKGE test restricted to O is kept as a cross-check of the scorer.
set -euo pipefail
R=/home/anon/benchscope4
export PATH=$R/env/bin:$PATH
read M SEED < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $R/scripts/hpc/e2_tasks.txt)
F=$R/runs/E2/${M}_s${SEED}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "task $SLURM_ARRAY_TASK_ID model $M seed $SEED"
cd $R/kge
if [ ! -f $F/checkpoint_best.pt ] || ! grep -q "Maximum number of epochs\|Stopping early" $F/kge.log 2>/dev/null; then
  rm -rf $F
  kge start $R/configs/e2/e2-$M.yaml --folder $F --job.device cuda --random_seed.default $SEED
fi
kge test $F --job.device cuda > $F/test_O.log 2>&1
grep -o "  mean_reciprocal_rank_filtered: [0-9.]*" $F/test_O.log | tail -1
cd /tmp
for Q in bench_test u_test; do
  python $R/scripts/score_wn18rr.py --checkpoint $F/checkpoint_best.pt --mode E2 --target $R/target \
    --queries $Q --filter target --out $F/score_$Q
done
echo TASK_OK
