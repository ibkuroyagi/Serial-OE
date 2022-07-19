#!/bin/bash

cp -r exp /fsws1/i_kuroyanagi/dcase2020/
echo "finished"
# no=100
# valid_ratio=0.15
# seed=0

# no=200
# for seed in 0 1 2 3 4; do
#     for n_anomaly in 0 1 2 4 8 16 32; do
#         ./job.sh \
#             --no "audioset_v${no}" \
#             --audioset_pow "0" \
#             --stage "1" \
#             --start_stage "3" \
#             --valid_ratio "${valid_ratio}" \
#             --use_uav "false" \
#             --use_idmt "false" \
#             --n_anomaly "${n_anomaly}" \
#             --seed "${seed}"
#         sleep 1800
#     done
# done
# sleep 3600
# for seed in 0 1 2 3 4; do
#     for n_anomaly in 0 1 2 4 8 16 32; do
#         sbatch ./job.sh \
#             --no "audioset_v${no}" \
#             --audioset_pow "0" \
#             --stage "2" \
#             --valid_ratio "${valid_ratio}" \
#             --use_uav "false" \
#             --use_idmt "false" \
#             --n_anomaly "${n_anomaly}" \
#             --seed "${seed}"
#     done
# done
# bo=true
# for no in 000 001; do
#     for valid_ratio in 0.1 0.15; do
#         sbatch ./job.sh --no "audioset_v${no}" --audioset_pow 0 --stage 2 --valid_ratio ${valid_ratio} --use_uav ${bo} --use_idmt ${bo}
#     done
# done
# no=000
# valid_ratio=0.15
# sbatch ./job.sh --no "audioset_v${no}" --audioset_pow 0 --stage 1 --valid_ratio ${valid_ratio} --use_uav ${bo} --use_idmt ${bo}
# sleep 7200
# sbatch ./job.sh --no "audioset_v${no}" --audioset_pow 21 --stage 2 --valid_ratio ${valid_ratio} --use_uav ${bo} --use_idmt ${bo}
# machines=("fan" "pump" "slider" "ToyCar" "ToyConveyor" "valve")
# for machine in "${machines[@]}"; do
#     rm dump/dev/${machine}/test/train*
#     echo "finish ${machine}"
# done
