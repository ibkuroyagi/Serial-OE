#!/bin/bash

# shellcheck disable=SC1091
. ./cmd.sh || exit 1
. ./path.sh || exit 1

tag=serial_oe.normal
epochs="50 100"
feature=""
# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1
set -euo pipefail

echo "Start scoring arregated anomaly scores."
agg_checkpoints=""
for machine in "fan" "pump" "slider" "ToyCar" "ToyConveyor" "valve"; do
    agg_checkpoints+="exp/${machine}/${tag}/best_loss/best_loss${feature}_agg.csv "
done
mkdir -p "exp/all/${tag}/best_loss"
echo "See log via  exp/all/${tag}/best_loss/scoring${feature}.log"
# shellcheck disable=SC2154,SC2086
${train_cmd} "exp/all/${tag}/best_loss/scoring${feature}.log" \
    python -m serial_oe.bin.scoring --feature "${feature}" --agg_checkpoints ${agg_checkpoints}
score_checkpoints="exp/all/${tag}/best_loss/score${feature}.csv "
# score_checkpoints=""
for epoch in ${epochs}; do
    echo "Start scoring arregated anomaly scores in ${epoch} epoch."
    agg_checkpoints=""
    for machine in "fan" "pump" "slider" "ToyCar" "ToyConveyor" "valve"; do
        agg_checkpoints+="exp/${machine}/${tag}/checkpoint-${epoch}epochs/checkpoint-${epoch}epochs${feature}_agg.csv "
    done
    mkdir -p "exp/all/${tag}/checkpoint-${epoch}epochs"
    echo "See log via  exp/all/${tag}/checkpoint-${epoch}epochs/scoring${feature}.log"
    # shellcheck disable=SC2154,SC2086
    ${train_cmd} "exp/all/${tag}/checkpoint-${epoch}epochs/scoring${feature}.log" \
        python -m serial_oe.bin.scoring --feature "${feature}" --agg_checkpoints ${agg_checkpoints}
    score_checkpoints+="exp/all/${tag}/checkpoint-${epoch}epochs/score${feature}.csv "
done
# shellcheck disable=SC2086
python -m serial_oe.bin.scoring --feature "${feature}" --agg_checkpoints ${score_checkpoints} --concat
echo "Successfully finished scoring${feature}."
