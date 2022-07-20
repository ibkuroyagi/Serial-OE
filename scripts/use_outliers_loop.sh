#!/bin/bash

for threshold in 0.9995 0.9999 0.99995 0.99999 0.999995; do
    ./use_outliers_job.sh --start_stage 3 --threshold "${threshold}"
    sleep 1500
done

sleep 3600
for threshold in 0.9995 0.9999 0.99995 0.99999 0.999995 1; do
    sbatch ./use_outliers_job.sh --start_stage 3 --threshold "${threshold}" --stage 2
done
