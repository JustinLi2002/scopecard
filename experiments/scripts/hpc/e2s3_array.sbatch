#!/bin/bash
#SBATCH --job-name=bs4_e2s3
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH -o /home/anon/benchscope4/logs/e2s3_%A_%a.log
# S3 (MEASURED_C_DESIGN.md Sect. 5, benchmark identity): E2 exactly as e2_array.sbatch, on the target keyed
# by bare offset (target_s3, LibKGE dataset wnrr_target_s3). Task line SLURM_ARRAY_TASK_ID of e2_tasks.txt
# ("model seed"); the configs differ from E2's only in the dataset. Only bench_test is scored: S3 is a
# sensitivity analysis of the primary estimand, and u_test is not part of it.
set -euo pipefail
R=/home/anon/benchscope4
export PATH=$R/env/bin:$PATH
read M SEED < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $R/scripts/hpc/e2_tasks.txt)
F=$R/runs/E2_S3/${M}_s${SEED}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "task $SLURM_ARRAY_TASK_ID model $M seed $SEED"
cd $R/kge
if [ ! -f $F/checkpoint_best.pt ] || ! grep -q "Maximum number of epochs\|Stopping early" $F/kge.log 2>/dev/null; then
  rm -rf $F
  kge start $R/configs/e2s3/e2s3-$M.yaml --folder $F --job.device cuda --random_seed.default $SEED
fi
kge test $F --job.device cuda > $F/test_O.log 2>&1
grep -o "  mean_reciprocal_rank_filtered: [0-9.]*" $F/test_O.log | tail -1
cd /tmp
python $R/scripts/score_wn18rr.py --checkpoint $F/checkpoint_best.pt --mode E2 --target $R/target_s3 \
  --queries bench_test --filter target --out $F/score_bench_test
echo TASK_OK
