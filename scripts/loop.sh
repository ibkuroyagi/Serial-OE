#!/bin/bash

# shellcheck disable=SC1091
. utils/original_funcs.sh || exit 1
# for seed in 0 1 2 3 4; do
#     sbatch ./job.sh --stage 1 --start_stage 5 --no audioset_v020 --seed ${seed} --feature _none
# done
for seed in 0 1 2 3 4; do
    sbatch ./job.sh --stage 2 --start_stage 5 --no audioset_v020 --seed ${seed} --feature _none
done
