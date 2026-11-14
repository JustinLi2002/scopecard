#!/bin/bash -l
#SBATCH --job-name=bs4_nbfE1
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH -o /home/anon/benchscope4/logs/nbfE1_%A_%a.log
# NBFNet E1 (MEASURED_C_DESIGN.md Sect. 13, item 1): official transductive WN18RR config,
# seed = SLURM_ARRAY_TASK_ID, rspmm kernel built with the cuda/12.6 module (worked in the
# feasibility job). After training, run.py selects the best validation checkpoint and
# tests it; the adapter then scores that checkpoint on bench_test (E1) and on the E2
# training graph (deployment test).
set -uo pipefail
R=/home/anon/benchscope4
SEED=$SLURM_ARRAY_TASK_ID
export PATH=$R/env_nbf/bin:$PATH
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
CUDAMOD=$(module -t avail cuda 2>&1 | grep -E '^cuda/12' | sort -V | tail -1)
module load $CUDAMOD && export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export TORCH_EXTENSIONS_DIR=$R/torch_ext_s$SEED
[ -d $TORCH_EXTENSIONS_DIR ] || cp -r $R/torch_ext $TORCH_EXTENSIONS_DIR 2>/dev/null || true
OUT=$R/runs/nbf_E1/s$SEED
mkdir -p $OUT
CFG=$OUT/config.yaml
sed -e "s#root: ~/datasets/knowledge_graphs/#root: $R/nbf_data/WN18RR#" \
    -e "s#output_dir: ~/experiments/#output_dir: $OUT/#" \
    $R/NBFNet-PyG/config/transductive/wn18rr.yaml > $CFG
cd $R/NBFNet-PyG
python script/run.py -c $CFG --gpus [0] -s $SEED > $OUT/train.log 2>&1
echo "run.py exit $?"
WD=$(dirname $(ls -t $OUT/NBFNet/WN18RR/*/log.txt 2>/dev/null | head -1) 2>/dev/null)
BEST=$(grep -o "Load checkpoint from model_epoch_[0-9]*.pth" $OUT/train.log | tail -1 | grep -o "model_epoch_[0-9]*.pth")
echo "working dir $WD best $BEST"
grep -E "mrr:|hits@10:" $OUT/train.log | tail -4
for MODE in E1 deploy; do
  python $R/scripts/score_nbfnet.py --config $CFG --checkpoint $WD/$BEST --mode $MODE --target $R/target \
    --raw $R/nbf_data/WN18RR/raw --queries bench_test --batch 16 --out $OUT/score_${MODE}_bench_test
done
echo TASK_OK
