#!/bin/bash

# Copyright 2022 Ibuki Kuroyanagi
# shellcheck disable=SC1091
. ./cmd.sh || exit 1
. ./path.sh || exit 1
# basic settings
stage=1      # stage to start
stop_stage=1 # stage to stop
verbose=1    # verbosity level (lower is less info)
n_gpus=1     # number of gpus in training
n_jobs=128
conf=conf/tuning/asd_model.audioset_v000.yaml
resume=""
pos_machine=fan
seed=0 # Seed of scp file.
# directory path setting
download_dir=downloads
dumpdir=dump # directory to dump features
expdir=exp
# training related setting
tag=audioset_v000 # tag for directory to save model
valid_ratio=0.1
max_anomaly_pow=6
n_anomaly=-1 # -1: eval.scp, 0>=: eval_max*_seed*.scp,
# outlier setting
audioset_dir=/path/to/AudioSet/audios
audioset_pow=21
use_uav=false
use_idmt=false
# inference related setting
epochs="20 40 60 80 100"
checkpoints=""
use_10sec=false
feature=_embed # "": using all features in training an anomalous detector.
# _embed: using only embedding features in training an anomalous detector.
# _prediction: using only predictions in training an anomalous detector.
inlier_scp=""

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y/%m/%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
log "1. resume:${resume}"
# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1

set -euo pipefail
if [ ${audioset_pow} -gt 0 ]; then
    use_audioset=true
else
    use_audioset=false
fi

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
                python -m asd_tools.bin.normalize_wave \
                --download_dir "${download_dir}/${name}" \
                --dumpdir "${dumpdir}/${name}" \
                --statistic_path "${statistic_path}" \
                --verbose "${verbose}"
            log "Successfully finished dump ${name}."
        done
    done
    if "${use_audioset}"; then
        log "Creat Audioset scp."
        # # shellcheck disable=SC1091
        local/get_audioset_scp.sh "${audioset_dir}" "${dumpdir}/audioset"
        for segments in balanced_train_segments unbalanced_train_segments; do
            split_scps=""
            for i in $(seq 1 "${n_jobs}"); do
                split_scps+=" ${dumpdir}/audioset/${segments}/log/wav.${i}.scp"
            done
            # shellcheck disable=SC2086
            utils/split_scp.pl "${dumpdir}/audioset/${segments}/wav.scp" ${split_scps}
            log "Creat dump file start. ${dumpdir}/audioset/${segments}/log/preprocess.*.log"
            pids=()
            (
                # shellcheck disable=SC2086,SC2154
                ${train_cmd} --max-jobs-run 64 JOB=1:"${n_jobs}" "${dumpdir}/audioset/${segments}/log/preprocess.JOB.log" \
                    python local/dump_outlier.py \
                    --download_dir "${dumpdir}/audioset/${segments}/log/wav.JOB.scp" \
                    --dumpdir "${dumpdir}/audioset/${segments}/part.JOB" \
                    --dataset "audioset" \
                    --verbose "${verbose}"
            ) &
            pids+=($!)
            i=0
            for pid in "${pids[@]}"; do wait "${pid}" || ((++i)); done
            [ "${i}" -gt 0 ] && echo "$0: ${i} background jobs are failed." && exit 1
            log "Successfully created Audioset ${segments} dump."
        done
    fi
    if "${use_uav}"; then
        log "Creat dump of UAV-Propeller-Anomaly-Audio-Dataset."
        log "See the progress via ${dumpdir}/uav/dump_outlier.log."
        # shellcheck disable=SC2154,SC2086
        ${train_cmd} "${dumpdir}/uav/dump_outlier.log" \
            python local/dump_outlier.py \
            --download_dir "${download_dir}/UAV-Propeller-Anomaly-Audio-Dataset" \
            --dumpdir "${dumpdir}/uav" \
            --dataset "uav" \
            --verbose "${verbose}"
        log "Created dump of UAV-Propeller-Anomaly-Audio-Dataset."
    fi
    if "${use_idmt}"; then
        log "Creat dump of IDMT-ISA-ELECTRIC-ENGINE."
        log "See the progress via ${dumpdir}/idmt/dump_outlier.log."
        # shellcheck disable=SC2154,SC2086
        ${train_cmd} "${dumpdir}/idmt/dump_outlier.log" \
            python local/dump_outlier.py \
            --download_dir "${download_dir}/IDMT-ISA-ELECTRIC-ENGINE" \
            --dumpdir "${dumpdir}/idmt" \
            --dataset "idmt" \
            --verbose "${verbose}"
        log "Created dump of IDMT-ISA-ELECTRIC-ENGINE."
    fi
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
            --dataset "dcase" \
            --valid_ratio "${valid_ratio}"
        log "Successfully splited ${dumpdir}/${train_set} train and valid data."
        log "Scp files are in train${end_str}.scp and train${end_str}.scp."
        eval_set="dev/${machine}/test"
        # shellcheck disable=SC2154
        ${train_cmd} "${dumpdir}/${eval_set}/write_scp.log" \
            python local/write_scp.py \
            --dumpdir "${dumpdir}/${eval_set}" \
            --dataset "dcase" \
            --max_anomaly_pow "${max_anomaly_pow}" \
            --valid_ratio 0
        log "Successfully write ${dumpdir}/${eval_set}/eval.scp."
    done
    if "${use_audioset}"; then
        log "Creat scp of Audioset."
        log "See the progress via ${dumpdir}/audioset/unbalanced_train_segments/log/write_scp.log."
        # shellcheck disable=SC2086
        ${train_cmd} "${dumpdir}/audioset/unbalanced_train_segments/log/write_scp.log" \
            python local/write_scp.py \
            --dumpdir "${dumpdir}/audioset/unbalanced_train_segments" \
            --dataset "audioset" \
            --max_audioset_size_2_pow 22
        log "Successfully write ${dumpdir}/audioset/audioset_2__*.scp."
        log "See the progress via ${dumpdir}/audioset/balanced_train_segments/log/write_scp.log."
        # shellcheck disable=SC2086
        ${train_cmd} "${dumpdir}/audioset/balanced_train_segments/log/write_scp.log" \
            python local/write_scp.py \
            --dumpdir "${dumpdir}/audioset/balanced_train_segments" \
            --dataset "audioset" \
            --max_audioset_size_2_pow 0
        log "Successfully write ${dumpdir}/audioset/balanced_train_segments/audioset.scp."
    fi
    if "${use_uav}"; then
        log "Creat scp of UAV-Propeller-Anomaly-Audio-Dataset."
        log "See the progress via ${dumpdir}/uav/write_scp.log."
        # shellcheck disable=SC2154,SC2086
        ${train_cmd} "${dumpdir}/uav/write_scp.log" \
            python local/write_scp.py \
            --dumpdir "${dumpdir}/uav" \
            --dataset "uav"
        log "Created scp of UAV-Propeller-Anomaly-Audio-Dataset."
    fi
    if "${use_idmt}"; then
        log "Creat scp of IDMT-ISA-ELECTRIC-ENGINE."
        log "See the progress via ${dumpdir}/idmt/write_scp.log."
        # shellcheck disable=SC2154,SC2086
        ${train_cmd} "${dumpdir}/idmt/write_scp.log" \
            python local/write_scp.py \
            --dumpdir "${dumpdir}/idmt" \
            --dataset "idmt"
        log "Created scp of IDMT-ISA-ELECTRIC-ENGINE."
    fi
fi

tag+="${end_str}"
outlier_scps=""
valid_outlier_scps=""
if "${use_audioset}"; then
    tag+="_p${audioset_pow}"
    outlier_scps+="${dumpdir}/audioset/unbalanced_train_segments/audioset_2__${audioset_pow}.scp "
    valid_outlier_scps+="${dumpdir}/audioset/balanced_train_segments/audioset.scp"
fi
if "${use_uav}"; then
    tag+="_uav"
    outlier_scps+="${dumpdir}/uav/uav.scp "
fi
if "${use_idmt}"; then
    tag+="_idmt"
    outlier_scps+="${dumpdir}/idmt/idmt.scp "
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
        python -m asd_tools.bin.train \
        --pos_machine "${pos_machine}" \
        --train_pos_machine_scp "${dumpdir}/dev/${pos_machine}/train/train${end_str}.scp" \
        --train_pos_anomaly_machine_scp "${train_anomaly_scp}" \
        --train_neg_machine_scps ${train_neg_machine_scps} \
        --valid_pos_machine_scp "${dumpdir}/dev/${pos_machine}/train/valid${end_str}.scp" \
        --valid_neg_machine_scps ${valid_neg_machine_scps} \
        --outlier_scp ${outlier_scps} \
        --valid_outlier_scp ${valid_outlier_scps} \
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
if [ -z "${inlier_scp}" ]; then
    inlier_scp="${dumpdir}/dev/${pos_machine}/train/valid${end_str}.scp"
    tail_name=""
else
    tail_name="_dev"
fi
if [ "${stage}" -le 4 ] && [ "${stop_stage}" -ge 4 ]; then
    log "Stage 4: Embedding calculation start. See the progress via ${outdir}/embed_${pos_machine}_${tag}${tail_name}.log."
    # shellcheck disable=SC2154,SC2086
    ${cuda_cmd} --gpu "${n_gpus}" "${outdir}/embed_${pos_machine}_${tag}${tail_name}.log" \
        python -m asd_tools.bin.embed \
        --valid_pos_machine_scp "${dumpdir}/dev/${pos_machine}/train/valid${end_str}.scp" \
        --eval_pos_machine_scp "${dumpdir}/dev/${pos_machine}/test/eval${anomaly_end_str}.scp" \
        --statistic_path "${dumpdir}/dev/${pos_machine}/train/statistic.json" \
        --checkpoints ${checkpoints} \
        --config "${conf}" \
        --use_10sec "${use_10sec}" \
        --tail_name "${tail_name}" \
        --verbose "${verbose}"
    log "Successfully finished extracting embedding."
fi

if [ "${stage}" -le 5 ] && [ "${stop_stage}" -ge 5 ]; then
    log "Stage 5: Inference start. See the progress via ${outdir}/infer_${pos_machine}${feature}${tail_name}_${tag}.log."
    # shellcheck disable=SC2154,SC2086
    ${cuda_cmd} "${outdir}/infer_${pos_machine}${feature}${tail_name}_${tag}.log" \
        python -m asd_tools.bin.infer \
        --pos_machine "${pos_machine}" \
        --checkpoints ${checkpoints} \
        --config "${conf}" \
        --feature "${feature}" \
        --use_10sec "${use_10sec}" \
        --tail_name "${tail_name}" \
        --verbose "${verbose}"
    log "Successfully finished Inference."
fi
