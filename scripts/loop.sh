#!/bin/bash

# for no in {000..003}; do
#     for audioset_pow in 0 21; do
#         sbatch ./audioset_job.sh --no "audioset_v${no}" --audioset_pow ${audioset_pow} --stage 1 --feature "" --start_stage 5
#     done
# done
for no in {000..003}; do
    for audioset_pow in 0 21; do
        sbatch ./audioset_job.sh --no "audioset_v${no}" --audioset_pow ${audioset_pow} --stage 2 --feature "_embed"
    done
done
