#!/bin/bash

# Copyright 2022 Ibuki Kuroyanagi
# shellcheck disable=SC1091
. ./cmd.sh || exit 1
. ./path.sh || exit 1

dumpdir="dump"
pos_machine="fan"
checkpoint="checkpoint-100epochs"
no="audioset_v000"

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y/%m/%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1
conf="conf/tuning/asd_model.${no}.yaml"
tag="${no}_0.15"
outdir="exp/${pos_machine}/${tag}/${checkpoint}"

log "Get outlier embedding start. See the progress via ${outdir}/outlier_embed_${pos_machine}_${tag}.log."
outlier_scps="${dumpdir}/idmt/idmt.scp ${dumpdir}/uav/uav.scp ${dumpdir}/audioset/balanced_train_segments/audioset.scp"
audioset_scps="${dumpdir}/audioset/unbalanced_train_segments/audioset_2__21.scp"
# shellcheck disable=SC2086
${cuda_cmd} --gpu 1 "${outdir}/outlier_embed_${pos_machine}_${tag}.log" \
    python local/get_outlier_embed.py \
    --pos_machine "${pos_machine}" \
    --audioset_scps ${audioset_scps} \
    --outlier_scps ${outlier_scps} \
    --statistic_path "${dumpdir}/dev/${pos_machine}/train/statistic.json" \
    --checkpoint "${outdir}/${checkpoint}.pkl" \
    --config "${conf}" \
    --verbose "1"
log "Successfully finished extracting embedding."
