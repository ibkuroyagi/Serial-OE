#!/bin/bash
# rm -rf dump/uav dump/idmt
# ln -s /fsws1/i_kuroyanagi/dcase2020/uav dump
# ln -s /fsws1/i_kuroyanagi/dcase2020/idmt dump
# echo "FInish"

for no in 000 001; do
    for valid_ratio in 0.1 0.15; do
        ./audioset_job.sh --no "audioset_v${no}" --audioset_pow 0 --stage 1 --start_stage 3 --valid_ratio ${valid_ratio} --use_uav true --use_idmt true
    done
    sleep 3600
done

sleep 3600
for no in 000 001; do
    for valid_ratio in 0.1 0.15; do
        sbatch ./audioset_job.sh --no "audioset_v${no}" --audioset_pow 0 --stage 2 --valid_ratio ${valid_ratio} --use_uav true --use_idmt true
    done
done
