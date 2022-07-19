#!/bin/bash

./use_outliers_job.sh --start_stage 2 --threshold 0
sleep 600
for threshold in 0.05 0.1 0.5 0.9 0.95 0.99 0.999; do
    ./use_outliers_job.sh --start_stage 3 --threshold "${threshold}"
    sleep 3600
done

sleep 3600
for threshold in 0 0.05 0.1 0.5 0.9 0.95 0.99 0.999; do
    sbatch ./use_outliers_job.sh --start_stage 3 --threshold "${threshold}" --stage 2
done
