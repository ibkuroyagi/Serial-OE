#!/bin/bash

for no in 009 010 011 012; do
    for valid_ratio in 0.1 0.15; do
        ./audioset_job.sh --no "audioset_v${no}" --audioset_pow 0 --stage 1 --start_stage 3 --valid_ratio ${valid_ratio}
    done
    sleep 3600
done

sleep 7200
for no in 009 010 011 012; do
    for valid_ratio in 0.1 0.15; do
        ./audioset_job.sh --no "audioset_v${no}" --audioset_pow 0 --stage 2 --valid_ratio ${valid_ratio}
    done
done

# rm -rf dump/uav dump/idmt
# ln -s /fsws1/i_kuroyanagi/dcase2020/uav dump
# ln -s /fsws1/i_kuroyanagi/dcase2020/idmt dump
echo "FInish"
