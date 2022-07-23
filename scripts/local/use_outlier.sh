#!/bin/bash

# Copyright 2022 Ibuki Kuroyanagi
# shellcheck disable=SC1091
. ./cmd.sh || exit 1
. ./path.sh || exit 1

dumpdir="dump"
pos_machine="ToyCar"
checkpoint="checkpoint-100epochs"
no="audioset_v000"
stage=3
stop_stage=3
# training model related
run_stage=3
resume=""
epochs=100
feature=_embed
valid_ratio=0.15
# use outlier related
threshold=0.999
col_name=outlier
# anomaly related
seed=0

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y/%m/%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1
set -euo pipefail

end_str="_${valid_ratio}"
conf="conf/tuning/asd_model.${no}.yaml"
tag="${no}${end_str}_seed${seed}"
outdir="exp/${pos_machine}/${tag}/${checkpoint}"

if [ "${pos_machine}" = "fan" ]; then
    neg_machines=("pump" "slider" "ToyCar" "ToyConveyor" "valve")
elif [ "${pos_machine}" = "pump" ]; then
    neg_machines=("pump" "slider" "ToyCar" "ToyConveyor" "valve")
elif [ "${pos_machine}" = "slider" ]; then
    neg_machines=("fan" "pump" "ToyCar" "ToyConveyor" "valve")
elif [ "${pos_machine}" = "ToyCar" ]; then
    neg_machines=("fan" "pump" "slider" "ToyConveyor" "valve")
elif [ "${pos_machine}" = "ToyConveyor" ]; then
    neg_machines=("fan" "pump" "slider" "ToyCar" "valve")
elif [ "${pos_machine}" = "valve" ]; then
    neg_machines=("fan" "pump" "slider" "ToyCar" "ToyConveyor")
fi

train_neg_machine_scps=""
valid_neg_machine_scps=""
for neg_machine in "${neg_machines[@]}"; do
    train_neg_machine_scps+="${dumpdir}/dev/${neg_machine}/train/train${end_str}.scp "
    valid_neg_machine_scps+="${dumpdir}/dev/${neg_machine}/train/valid${end_str}.scp "
done

if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    log "Stage 1: Get outlier embedding start. See the progress via ${outdir}/outlier_embed_${pos_machine}_${tag}.log."
    outlier_scps="${dumpdir}/idmt/idmt.scp ${dumpdir}/uav/uav.scp ${dumpdir}/audioset/balanced_train_segments/audioset.scp"
    audioset_scps="${dumpdir}/audioset/unbalanced_train_segments/audioset_2__21.scp"
    # shellcheck disable=SC2086,SC2154
    ${cuda_cmd} --gpu 1 "${outdir}/outlier_embed_${pos_machine}_${tag}.log" \
        python local/get_outlier_embed.py \
        --train_pos_machine_scp "${dumpdir}/dev/${pos_machine}/train/train${end_str}.scp" \
        --train_neg_machine_scps ${train_neg_machine_scps} \
        --valid_pos_machine_scp "${dumpdir}/dev/${pos_machine}/train/valid${end_str}.scp" \
        --valid_neg_machine_scps ${valid_neg_machine_scps} \
        --pos_machine "${pos_machine}" \
        --audioset_scps ${audioset_scps} \
        --outlier_scps ${outlier_scps} \
        --statistic_path "${dumpdir}/dev/${pos_machine}/train/statistic.json" \
        --checkpoint "${outdir}/${checkpoint}.pkl" \
        --config "${conf}" \
        --verbose "1"
    log "Successfully finished extracting embedding."
fi
if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    log "Stage 2: Select using outlier data."
    # shellcheck disable=SC2086,SC2154
    ${train_cmd} "${outdir}/select_outlier_${pos_machine}_${tag}.log" \
        python local/select_outlier.py \
        --pos_machine "${pos_machine}" \
        --outlier_mean_csv "${outdir}/${checkpoint}_outlier_mean.csv" \
        --audioset_mean_csv "${outdir}/${checkpoint}_audioset_mean.csv" \
        --outlier_scp_save_dir "${outdir}"
    log "Successfully selected outliers."
fi

outlier_scps="exp/${pos_machine}/${tag}/${checkpoint}/${col_name}_${threshold}.scp"
if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
    log "Start model training ${pos_machine}/${no}_${col_name}${threshold}_${valid_ratio}."
    ./run.sh \
        --stage "${run_stage}" \
        --stop_stage "5" \
        --conf "conf/tuning/asd_model.${no}.yaml" \
        --pos_machine "${pos_machine}" \
        --resume "${resume}" \
        --tag "${no}_${col_name}${threshold}" \
        --feature "${feature}" \
        --epochs "${epochs}" \
        --valid_ratio "${valid_ratio}" \
        --audioset_pow "0" \
        --use_uav "false" \
        --use_idmt "false" \
        --n_anomaly "-1" \
        --max_anomaly_pow "6" \
        --outlier_scps "${outlier_scps}" \
        --seed "${seed}"
fi
