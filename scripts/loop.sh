#!/bin/bash

# shellcheck disable=SC1091
. utils/original_funcs.sh || exit 1
# training stage
for method in only_l_product only_l_machine serial_oe serial_method wo_l_machine wo_l_product wo_mixup; do
    ./job.sh --model_name "${method}.normal"
    ./job.sh --model_name "${method}.utilize" --n_anomaly 32
    ./job.sh --model_name "${method}.contaminated" --n_anomaly 32
done
sleep 10000
# scoring stage
for method in only_l_product only_l_machine serial_oe serial_method wo_l_machine wo_l_product wo_mixup; do
    ./job.sh --model_name "${method}.normal" --stage 2
    ./job.sh --model_name "${method}.utilize" --n_anomaly 32 --stage 2
    ./job.sh --model_name "${method}.contaminated" --n_anomaly 32 --stage 2
done
