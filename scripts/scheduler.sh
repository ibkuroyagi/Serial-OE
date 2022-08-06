#!/bin/bash
no=audioset_v000
seed=4
col_name=machine
# shellcheck disable=SC1091
. utils/original_funcs.sh || exit 1
# for seed in 3 1; do
for threshold in 0 0.001 0.005 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1; do
    sbatch ./use_outliers_job.sh --stage 2 --threshold "${threshold}" --col_name "${col_name}" --seed "${seed}" --no "${no}"
done
# done
