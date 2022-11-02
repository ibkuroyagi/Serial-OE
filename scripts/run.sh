#!/bin/bash

# Copyright 2022 Ibuki Kuroyanagi
# shellcheck disable=SC1091
. ./cmd.sh || exit 1
. ./path.sh || exit 1
# basic settings
stage=0      # stage to start
stop_stage=5 # stage to stop
verbose=1    # verbosity level (lower is less info)
n_gpus=1     # number of gpus in training
conf=conf/tuning/serial_oe.normal.yaml
resume=""
pos_machine=fan
tag=serial_oe.normal # tag for directory to save model
seed=-1              # Seed of scp file.
# directory path setting
download_dir=downloads
dumpdir=dump # directory to dump features
expdir=exp
# training related setting
valid_ratio=0.15
max_anomaly_pow=6 # The number of anomaly sample for training. 2**(6-1) = 32
n_anomaly=-1      # -1: eval.scp, 0>=: eval_max*_seed*.scp
# inference related setting
epochs="50 100"
checkpoints=""
feature=_embed # "_none": use output of ID classification.
# _embed: use only embedding features in training an anomalous detector.
# _prediction: use machine prediction and ID predictions in training an anomalous detector.

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y/%m/%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1

set -euo pipefail

log "Start run.sh"
machines=("fan" "pump" "slider" "valve" "ToyCar" "ToyConveyor")
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

if [ "${stage}" -le 0 ] && [ "${stop_stage}" -ge 0 ]; then
    log "Download data."
    local/download_data.sh "${download_dir}"
    # To use the eval directory of the DCASE 2020 Task2 Challenge dataset,
    # put it in the dev directory in the same format.

    # downloads
    # |--dev  (We expect dev directory contain all IDs {00,01,..,06}.)
    #    |--fan
    #    |  |--test
    #    |  |  |--anomaly_id_00_00000000.wav
    #    |  |  |--anomaly_id_00_00000001.wav
    #    |  |  |--**
    #    |  |  |--normal_id_06_00000099.wav
    #    |  |--train
    #    |  |  |--normal_id_00_00000000.wav
    #    |  |  |--normal_id_00_00000001.wav
    #    |  |  |--**
    #    |  |  |--normal_id_06_00000914.wav
    #    |--pump
    #    |  |--test
    #    |  |  |--anomaly_id_00_00000000.wav
    #    |  |  |--anomaly_id_00_00000001.wav
    #    |  |  |--**
    #    |  |  |--normal_id_06_00000099.wav
    #    |  |--train
    #    |  |  |--normal_id_00_00000000.wav
    #    |  |  |--normal_id_00_00000001.wav
    #    |  |  |--**
    #    |  |  |--normal_id_06_00000914.wav
fi

if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    log "Stage 1: Dump wave data."
    for machine in "${machines[@]}"; do
        train_set="dev/${machine}/train"
        eval_set="dev/${machine}/test"
        statistic_path="${dumpdir}/dev/${machine}/train/statistic.json"
        for name in ${train_set} ${eval_set}; do
            [ ! -e "${dumpdir}/${name}" ] && mkdir -p "${dumpdir}/${name}"
            log "See the progress via ${dumpdir}/${name}/normalize_wave.log."
            # shellcheck disable=SC2154
            ${train_cmd} --num_threads 1 "${dumpdir}/${name}/normalize_wave.log" \
                python -m serial_oe.bin.normalize_wave \
                --download_dir "${download_dir}/${name}" \
                --dumpdir "${dumpdir}/${name}" \
                --statistic_path "${statistic_path}" \
                --verbose "${verbose}"
            log "Successfully finished dump ${name}."
        done
    done
fi

end_str+="_${valid_ratio}"
if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    log "Stage 2: Write scp."
    log "Split training data."
    for machine in "${machines[@]}"; do
        train_set="dev/${machine}/train"
        # shellcheck disable=SC2154,SC2086
        ${train_cmd} "${dumpdir}/${train_set}/write_scp${end_str}.log" \
            python local/write_scp.py \
            --dumpdir ${dumpdir}/${train_set} \
            --valid_ratio "${valid_ratio}"
        log "Successfully splited ${dumpdir}/${train_set} train and valid data."
        log "Scp files are in train${end_str}.scp and train${end_str}.scp."
        eval_set="dev/${machine}/test"
        # shellcheck disable=SC2154
        ${train_cmd} "${dumpdir}/${eval_set}/write_scp.log" \
            python local/write_scp.py \
            --dumpdir "${dumpdir}/${eval_set}" \
            --max_anomaly_pow "${max_anomaly_pow}" \
            --valid_ratio 0
        log "Successfully write ${dumpdir}/${eval_set}/eval.scp."
    done
fi

tag+="${end_str}"
if [ ${seed} -ge 0 ] && [ "${n_anomaly}" -le -1 ]; then
    log "For simple ASD model's seed : ${seed}."
    tag+="_seed${seed}"
fi

anomaly_end_str=""
train_anomaly_scp=""
if [ ${n_anomaly} -ge 0 ]; then
    anomaly_end_str+="_max${max_anomaly_pow}_seed${seed}"
    train_anomaly_scp+="${dumpdir}/dev/${pos_machine}/test/train_anomaly${n_anomaly}${anomaly_end_str}.scp"
    tag+="_anomaly${n_anomaly}${anomaly_end_str}"
fi
outdir="${expdir}/${pos_machine}/${tag}"
if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
    log "Stage 3: Train embedding. resume: ${resume}"
    [ ! -e "${outdir}" ] && mkdir -p "${outdir}"
    train_neg_machine_scps=""
    valid_neg_machine_scps=""
    for neg_machine in "${neg_machines[@]}"; do
        train_neg_machine_scps+="${dumpdir}/dev/${neg_machine}/train/train${end_str}.scp "
        valid_neg_machine_scps+="${dumpdir}/dev/${neg_machine}/train/valid${end_str}.scp "
    done
    log "Training start. See the progress via ${outdir}/train_${pos_machine}_${tag}.log."
    # shellcheck disable=SC2154,SC2086
    ${cuda_cmd} --gpu "${n_gpus}" "${outdir}/train_${pos_machine}_${tag}.log" \
        python -m serial_oe.bin.train \
        --pos_machine "${pos_machine}" \
        --train_pos_machine_scp "${dumpdir}/dev/${pos_machine}/train/train${end_str}.scp" \
        --train_pos_anomaly_machine_scp "${train_anomaly_scp}" \
        --train_neg_machine_scps ${train_neg_machine_scps} \
        --valid_pos_machine_scp "${dumpdir}/dev/${pos_machine}/train/valid${end_str}.scp" \
        --valid_neg_machine_scps ${valid_neg_machine_scps} \
        --statistic_path "${dumpdir}/dev/${pos_machine}/train/statistic.json" \
        --outdir "${outdir}" \
        --config "${conf}" \
        --resume "${resume}" \
        --verbose "${verbose}"
    log "Successfully finished source training."
fi

# shellcheck disable=SC2086
if [ -z ${checkpoints} ]; then
    checkpoints+="${outdir}/best_loss/best_loss.pkl "
    for epoch in ${epochs}; do
        checkpoints+="${outdir}/checkpoint-${epoch}epochs/checkpoint-${epoch}epochs.pkl "
    done
fi

if [ "${stage}" -le 4 ] && [ "${stop_stage}" -ge 4 ]; then
    log "Stage 4: Embedding calculation start. See the progress via ${outdir}/embed_${pos_machine}_${tag}.log."
    # shellcheck disable=SC2154,SC2086
    ${cuda_cmd} --gpu "${n_gpus}" "${outdir}/embed_${pos_machine}_${tag}.log" \
        python -m serial_oe.bin.embed \
        --valid_pos_machine_scp "${dumpdir}/dev/${pos_machine}/train/valid${end_str}.scp" \
        --eval_pos_machine_scp "${dumpdir}/dev/${pos_machine}/test/eval${anomaly_end_str}.scp" \
        --statistic_path "${dumpdir}/dev/${pos_machine}/train/statistic.json" \
        --checkpoints ${checkpoints} \
        --config "${conf}" \
        --verbose "${verbose}"
    log "Successfully finished extracting embedding."
fi

if [ "${stage}" -le 5 ] && [ "${stop_stage}" -ge 5 ]; then
    log "Stage 5: Inference start. See the progress via ${outdir}/infer_${pos_machine}${feature}_${tag}.log."
    # shellcheck disable=SC2154,SC2086
    ${cuda_cmd} "${outdir}/infer_${pos_machine}${feature}_${tag}.log" \
        python -m serial_oe.bin.infer \
        --pos_machine "${pos_machine}" \
        --checkpoints ${checkpoints} \
        --config "${conf}" \
        --feature "${feature}" \
        --verbose "${verbose}"
    log "Successfully finished Inference."
fi
