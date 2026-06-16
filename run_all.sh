#!/bin/bash
# FixMatch CIFAR-10 全量训练脚本 (40 → 250 → 4000 labels)
# RTX 4090 预计总耗时: ~15 小时
#
# 用法:
#   chmod +x run_all.sh
#   ./run_all.sh
#   nohup ./run_all.sh > logs/run_all.log 2>&1 &  # 后台运行

set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="./logs"
mkdir -p "$LOGDIR"

ARGS_COMMON="--batch_size 64 --uratio 7 --lr 0.03 --total_steps 1048576 \
  --confidence_threshold 0.95 --ema_decay 0.999 --use_amp --cudnn_benchmark \
  --seed 0 --eval_interval 1024 --log_interval 128 --save_interval 65536"

train_one() {
    local NUM_LABELS=$1
    local CKPT_DIR="./checkpoints/fixmatch_${NUM_LABELS}"
    local LOG_FILE="$LOGDIR/fixmatch_${NUM_LABELS}_${TIMESTAMP}.log"

    echo "============================================================"
    echo "  Training FixMatch with ${NUM_LABELS} labels"
    echo "  Start: $(date)"
    echo "  Log:   $LOG_FILE"
    echo "============================================================"

    python3 -u train.py \
        --num_labels "$NUM_LABELS" \
        --checkpoint_dir "$CKPT_DIR" \
        --log_dir "$LOGDIR/fixmatch_${NUM_LABELS}" \
        $ARGS_COMMON \
        2>&1 | tee "$LOG_FILE"

    echo "  Finished: $(date)"
    echo ""
}

echo "FixMatch CIFAR-10 Training Pipeline"
echo "Start time: $(date)"
echo "Expected: ~5h per run, ~15h total on RTX 4090"
echo ""

train_one 40
train_one 250
train_one 4000

echo "============================================================"
echo "  ALL DONE at $(date)"
echo "============================================================"
echo ""
echo "Results summary:"
grep -h "Result" "$LOGDIR"/fixmatch_*_"$TIMESTAMP".log 2>/dev/null || echo "  (check logs)"
grep -h "Final" "$LOGDIR"/fixmatch_*_"$TIMESTAMP".log 2>/dev/null || true
