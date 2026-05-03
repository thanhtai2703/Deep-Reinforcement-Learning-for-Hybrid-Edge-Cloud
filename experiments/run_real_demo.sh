#!/bin/bash
# Chạy tất cả policies trên hạ tầng K8s thật + gom summary vào 1 CSV.
# Sau đó dùng plot_real_comparison.py để vẽ.

set -e

SEED=${SEED:-42}
NUM_TASKS=${NUM_TASKS:-50}
CONCURRENCY=${CONCURRENCY:-5}
PROMETHEUS=${PROMETHEUS:-http://localhost:9090}
OUTPUT_CSV=${OUTPUT_CSV:-experiments/logs/real_comparison.csv}

DQN_CAL_MODEL=models/checkpoints/dqn_calibrated/dqn_best.pth
DQN_UNCAL_MODEL=models/checkpoints/dqn_uncalibrated/dqn_best.pth

# Reset CSV nếu tồn tại
[ -f "$OUTPUT_CSV" ] && rm "$OUTPUT_CSV"
mkdir -p "$(dirname "$OUTPUT_CSV")"

echo "════════════════════════════════════════════════════════════════"
echo "  Real K8s Comparison Run"
echo "  Tasks: $NUM_TASKS | Concurrency: $CONCURRENCY | Seed: $SEED"
echo "  Output: $OUTPUT_CSV"
echo "════════════════════════════════════════════════════════════════"

run_policy() {
  local POLICY=$1
  local MODEL_FLAG=$2
  echo ""
  echo "▶ Running policy: $POLICY"
  python -m dispatcher.dispatcher_cli --policy $POLICY $MODEL_FLAG \
    --num-tasks $NUM_TASKS --concurrency $CONCURRENCY --seed $SEED \
    --prometheus $PROMETHEUS \
    --save-summary "$OUTPUT_CSV"
}

# Baselines
run_policy random          ""
run_policy round_robin     ""
run_policy least_connection ""
run_policy edge_only       ""
run_policy cloud_only      ""

# RL models (chỉ chạy nếu có model)
if [ -f "$DQN_CAL_MODEL" ]; then
  run_policy dqn "--dqn-model $DQN_CAL_MODEL"
fi

if [ -f "$DQN_UNCAL_MODEL" ]; then
  echo ""
  echo "▶ Running policy: dqn_uncal (cùng dispatcher_cli, model khác)"
  python -m dispatcher.dispatcher_cli --policy dqn \
    --dqn-model $DQN_UNCAL_MODEL \
    --num-tasks $NUM_TASKS --concurrency $CONCURRENCY --seed $SEED \
    --prometheus $PROMETHEUS \
    --save-summary "$OUTPUT_CSV"
  # Mark as dqn_uncal trong CSV (sửa label sau khi run nếu cần)
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "✅ Done. CSV saved: $OUTPUT_CSV"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Next step:"
echo "  python experiments/plot_real_comparison.py $OUTPUT_CSV"
