#!/bin/bash

# Combine data direcotries into a single data direcotry

# Copyright 2022 Ibuki Kuroyanagi
#  MIT License (https://opensource.org/licenses/MIT)
log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y/%m/%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
slurm_gpu_scheduler() {
    if [ $# != 1 ]; then
        echo "Usage: $0 <available_gpus>"
        echo "e.g.: $0 3"
        exit 1
    fi
    while :; do
        sleep 8 # Wait for the previous tasks to be assigned to the GPUs.
        local cnt_gpus=0
        local using_gpu_list=$(squeue -u $(whoami) -h -o "%.5b")
        for token in $using_gpu_list; do
            use_gpu=$(echo ${token} | sed -e 's/[^0-9]//g')
            if [ -n "${use_gpu}" ]; then
                cnt_gpus=$((${cnt_gpus} + ${use_gpu}))
            fi
        done
        if [ "${cnt_gpus}" -le "$1" ]; then
            break
        fi
    done
}
