#!/bin/bash

# for col_name in section outlier; do
#     for seed in 0 1 2 3 4; do
#         ./use_outliers_job.sh --start_stage 1 --run_stage 3 --threshold "0" --col_name "${col_name}" --seed "${seed}"
#         sleep 1200
#     done
# done
col_name=outlier
for seed in 0 1 2 3 4; do
    ./use_outliers_job.sh --start_stage 3 --run_stage 3 --threshold "0" --col_name "${col_name}" --seed "${seed}"
    sleep 1200
done
seed=0
for col_name in section outlier; do
    for threshold in 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.99 0.999 0.9995 0.9999 1; do
        ./use_outliers_job.sh --start_stage 3 --run_stage 3 --threshold "${threshold}" --col_name "${col_name}" --seed "${seed}"
        sleep 1800
    done
done

for col_name in section outlier; do
    for seed in 1 2 3 4; do
        for threshold in 0 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.99 0.999 0.9995 0.9999 1; do
            ./use_outliers_job.sh --start_stage 3 --run_stage 3 --threshold "${threshold}" --col_name "${col_name}" --seed "${seed}"
            sleep 1800
        done
    done
done

for col_name in section outlier; do
    for seed in 0 1 2 3 4; do
        for threshold in 0 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.99 0.999 0.9995 0.9999 1; do
            ./use_outliers_job.sh --stage 2 --threshold "${threshold}" --col_name "${col_name}" --seed "${seed}"
        done
    done
done

# sleep 7200
# for threshold in 0 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.99 0.999 0.9995 0.9999 0.99995 0.99999 0.999995 1; do
#     sbatch ./use_outliers_job.sh --start_stage 3 --threshold "${threshold}" --stage 2 --col_name "section"
# done
