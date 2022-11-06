#!/bin/bash

# training stage
for method in only_l_product only_l_machine serial_oe serial_method wo_l_machine wo_l_product wo_mixup; do
    ./job.sh --model_name "${method}.normal"
    ./job.sh --model_name "${method}.utilize" --n_anomaly 32
    ./job.sh --model_name "${method}.contaminated" --n_anomaly 32
done

# scoring stage
for method in only_l_product only_l_machine serial_oe serial_method wo_l_machine wo_l_product wo_mixup; do
    ./job.sh --model_name "${method}.normal" --stage 2
    ./job.sh --model_name "${method}.utilize" --n_anomaly 32 --stage 2
    ./job.sh --model_name "${method}.contaminated" --n_anomaly 32 --stage 2
done
