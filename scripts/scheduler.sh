#!/bin/bash

# shellcheck disable=SC1091
. utils/original_funcs.sh || exit 1
available_gpus=15

./sub_job.sh --no audioset_v126 --stage 1 --start_stage 3 --n_anomaly 0 --seed 0 --machine fan
./sub_job.sh --no audioset_v126 --stage 1 --start_stage 3 --n_anomaly 0 --seed 0 --machine slider
./sub_job.sh --no audioset_v126 --stage 1 --start_stage 4 --n_anomaly 0 --seed 0 --machine valve
./sub_job.sh --no audioset_v126 --stage 1 --start_stage 4 --n_anomaly 0 --seed 0 --machine ToyCar
./sub_job.sh --no audioset_v126 --stage 1 --start_stage 4 --n_anomaly 1 --seed 0 --machine fan
./sub_job.sh --no audioset_v126 --stage 1 --start_stage 3 --n_anomaly 1 --seed 0 --machine pump
./sub_job.sh --no audioset_v126 --stage 1 --start_stage 3 --n_anomaly 1 --seed 0 --machine ToyCar
./sub_job.sh --no audioset_v126 --stage 1 --start_stage 3 --n_anomaly 2 --seed 0 --machine fan
./sub_job.sh --no audioset_v126 --stage 1 --start_stage 3 --n_anomaly 2 --seed 0 --machine valve
./sub_job.sh --no audioset_v126 --stage 1 --start_stage 3 --n_anomaly 4 --seed 0 --machine slider
./sub_job.sh --no audioset_v126 --stage 1 --start_stage 4 --n_anomaly 4 --seed 0 --machine ToyCar
./sub_job.sh --no audioset_v126 --stage 1 --start_stage 3 --n_anomaly 4 --seed 0 --machine ToyConveyor
./sub_job.sh --no audioset_v126 --stage 1 --start_stage 4 --n_anomaly 8 --seed 0 --machine fan
./sub_job.sh --no audioset_v126 --stage 1 --start_stage 4 --n_anomaly 8 --seed 0 --machine slider
./sub_job.sh --no audioset_v126 --stage 1 --start_stage 3 --n_anomaly 16 --seed 0 --machine fan
./sub_job.sh --no audioset_v126 --stage 1 --start_stage 3 --n_anomaly 16 --seed 0 --machine slider
./sub_job.sh --no audioset_v126 --stage 1 --start_stage 3 --n_anomaly 16 --seed 0 --machine ToyCar
./sub_job.sh --no audioset_v126 --stage 1 --start_stage 3 --n_anomaly 32 --seed 0 --machine pump
./sub_job.sh --no audioset_v126 --stage 1 --start_stage 4 --n_anomaly 32 --seed 0 --machine slider
./sub_job.sh --no audioset_v126 --stage 1 --start_stage 3 --n_anomaly 32 --seed 0 --machine valve
./sub_job.sh --no audioset_v126 --stage 1 --start_stage 3 --n_anomaly 32 --seed 0 --machine ToyConveyor
./sub_job.sh --no audioset_v126 --stage 1 --start_stage 3 --n_anomaly 0 --seed 1 --machine ToyConveyor
./sub_job.sh --no audioset_v126 --stage 1 --start_stage 4 --n_anomaly 1 --seed 4 --machine pump
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 4 --n_anomaly 32 --seed 0 --machine pump
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 4 --n_anomaly 0 --seed 1 --machine slider
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 4 --n_anomaly 0 --seed 1 --machine ToyCar
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 0 --seed 1 --machine ToyConveyor
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 1 --seed 1 --machine fan
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 1 --seed 1 --machine slider
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 1 --seed 1 --machine valve
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 1 --seed 1 --machine ToyCar
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 1 --seed 1 --machine ToyConveyor
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 2 --seed 1 --machine fan
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 2 --seed 1 --machine slider
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 2 --seed 1 --machine valve
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 2 --seed 1 --machine ToyCar
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 2 --seed 1 --machine ToyConveyor
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 4 --seed 1 --machine fan
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 4 --seed 1 --machine pump
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 4 --seed 1 --machine slider
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 4 --seed 1 --machine valve
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 4 --seed 1 --machine ToyCar
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 8 --seed 1 --machine fan
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 8 --seed 1 --machine pump
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 8 --seed 1 --machine slider
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 8 --seed 1 --machine valve
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 8 --seed 1 --machine ToyCar
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 8 --seed 1 --machine ToyConveyor
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 16 --seed 1 --machine fan
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 16 --seed 1 --machine pump
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 16 --seed 1 --machine slider
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 16 --seed 1 --machine valve
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 16 --seed 1 --machine ToyCar
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 16 --seed 1 --machine ToyConveyor
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 32 --seed 1 --machine fan
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 32 --seed 1 --machine pump
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 32 --seed 1 --machine slider
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 32 --seed 1 --machine ToyCar
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 32 --seed 1 --machine ToyConveyor
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 0 --seed 2 --machine fan
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 0 --seed 2 --machine pump
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 0 --seed 2 --machine valve
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 0 --seed 2 --machine ToyCar
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 0 --seed 2 --machine ToyConveyor
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 1 --seed 2 --machine fan
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 4 --n_anomaly 1 --seed 2 --machine valve
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 1 --seed 2 --machine ToyConveyor
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 2 --seed 2 --machine slider
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 4 --n_anomaly 2 --seed 2 --machine valve
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 4 --n_anomaly 2 --seed 2 --machine ToyCar
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 2 --seed 2 --machine ToyConveyor
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 4 --seed 2 --machine pump
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 4 --seed 2 --machine ToyCar
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 8 --seed 2 --machine fan
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 4 --n_anomaly 8 --seed 2 --machine pump
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 8 --seed 2 --machine slider
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 8 --seed 2 --machine valve
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 16 --seed 2 --machine slider
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 4 --n_anomaly 16 --seed 2 --machine ToyCar
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 4 --n_anomaly 16 --seed 2 --machine ToyConveyor
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 32 --seed 2 --machine fan
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 4 --n_anomaly 32 --seed 2 --machine ToyCar
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 4 --n_anomaly 0 --seed 3 --machine fan
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 0 --seed 3 --machine pump
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 0 --seed 3 --machine valve
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 0 --seed 3 --machine ToyConveyor
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 1 --seed 3 --machine pump
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 1 --seed 3 --machine slider
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 1 --seed 3 --machine valve
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 1 --seed 3 --machine ToyCar
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 1 --seed 3 --machine ToyConveyor
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 2 --seed 3 --machine fan
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 2 --seed 3 --machine pump
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 2 --seed 3 --machine slider
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 2 --seed 3 --machine valve
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 2 --seed 3 --machine ToyCar
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 2 --seed 3 --machine ToyConveyor
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 4 --seed 3 --machine fan
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 4 --seed 3 --machine pump
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 4 --seed 3 --machine slider
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 4 --seed 3 --machine ToyCar
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 4 --seed 3 --machine ToyConveyor
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 8 --seed 3 --machine fan
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 8 --seed 3 --machine pump
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 8 --seed 3 --machine slider
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 8 --seed 3 --machine valve
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 8 --seed 3 --machine ToyCar
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 8 --seed 3 --machine ToyConveyor
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 16 --seed 3 --machine fan
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 16 --seed 3 --machine pump
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 16 --seed 3 --machine slider
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 16 --seed 3 --machine valve
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 16 --seed 3 --machine ToyCar
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 32 --seed 3 --machine pump
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 32 --seed 3 --machine slider
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 32 --seed 3 --machine valve
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 4 --n_anomaly 32 --seed 3 --machine ToyCar
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 4 --n_anomaly 32 --seed 3 --machine ToyConveyor
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 0 --seed 4 --machine slider
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 0 --seed 4 --machine valve
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 2 --seed 4 --machine slider
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 4 --n_anomaly 2 --seed 4 --machine valve
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 4 --n_anomaly 2 --seed 4 --machine ToyCar
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 2 --seed 4 --machine ToyConveyor
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 4 --seed 4 --machine slider
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 4 --seed 4 --machine ToyCar
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 4 --n_anomaly 4 --seed 4 --machine ToyConveyor
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 8 --seed 4 --machine pump
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 8 --seed 4 --machine slider
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 16 --seed 4 --machine ToyCar
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 16 --seed 4 --machine ToyConveyor
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 32 --seed 4 --machine pump
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 32 --seed 4 --machine slider
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 32 --seed 4 --machine valve
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 32 --seed 4 --machine ToyCar
./sub_job.sh --no audioset_v127 --stage 1 --start_stage 3 --n_anomaly 32 --seed 4 --machine ToyConveyor

./sub_job.sh --no audioset_v128 --stage 1 --start_stage 4 --n_anomaly 4 --seed 0 --machine valve
./sub_job.sh --no audioset_v128 --stage 1 --start_stage 4 --n_anomaly 32 --seed 3 --machine fan
./sub_job.sh --no audioset_v129 --stage 1 --start_stage 4 --n_anomaly 1 --seed 1 --machine slider
./sub_job.sh --no audioset_v130 --stage 1 --start_stage 4 --n_anomaly 16 --seed 3 --machine valve
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 0 --seed 0 --machine fan
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 0 --seed 0 --machine pump
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 0 --seed 0 --machine slider
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 0 --seed 0 --machine valve
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 0 --seed 0 --machine ToyCar
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 0 --seed 0 --machine ToyConveyor
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 1 --seed 0 --machine fan
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 1 --seed 0 --machine pump
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 1 --seed 0 --machine slider
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 1 --seed 0 --machine valve
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 1 --seed 0 --machine ToyCar
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 1 --seed 0 --machine ToyConveyor
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 2 --seed 0 --machine pump
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 2 --seed 0 --machine slider
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 2 --seed 0 --machine valve
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 2 --seed 0 --machine ToyCar
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 2 --seed 0 --machine ToyConveyor
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 4 --seed 0 --machine pump
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 4 --seed 0 --machine slider
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 4 --seed 0 --machine valve
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 8 --seed 0 --machine pump
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 4 --n_anomaly 32 --seed 0 --machine pump
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 1 --seed 1 --machine pump
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 1 --seed 1 --machine valve
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 2 --seed 1 --machine pump
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 4 --n_anomaly 2 --seed 1 --machine slider
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 4 --n_anomaly 2 --seed 1 --machine valve
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 2 --seed 1 --machine ToyConveyor
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 4 --seed 1 --machine pump
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 4 --seed 1 --machine slider
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 4 --n_anomaly 4 --seed 1 --machine ToyCar
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 8 --seed 1 --machine fan
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 4 --n_anomaly 8 --seed 1 --machine pump
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 8 --seed 1 --machine valve
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 16 --seed 1 --machine fan
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 16 --seed 1 --machine ToyConveyor
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 32 --seed 1 --machine slider
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 4 --n_anomaly 32 --seed 1 --machine ToyCar
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 4 --n_anomaly 0 --seed 2 --machine slider
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 4 --n_anomaly 0 --seed 2 --machine valve
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 4 --n_anomaly 0 --seed 2 --machine ToyCar
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 1 --seed 2 --machine fan
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 1 --seed 2 --machine valve
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 1 --seed 2 --machine ToyCar
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 2 --seed 2 --machine slider
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 2 --seed 2 --machine valve
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 4 --seed 2 --machine fan
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 8 --seed 2 --machine fan
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 8 --seed 2 --machine valve
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 4 --n_anomaly 8 --seed 2 --machine ToyCar
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 4 --n_anomaly 16 --seed 2 --machine pump
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 16 --seed 2 --machine slider
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 16 --seed 2 --machine ToyConveyor
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 32 --seed 2 --machine fan
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 4 --n_anomaly 32 --seed 2 --machine pump
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 0 --seed 3 --machine fan
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 4 --n_anomaly 0 --seed 3 --machine slider
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 0 --seed 3 --machine ToyCar
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 0 --seed 3 --machine ToyConveyor
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 4 --n_anomaly 1 --seed 3 --machine pump
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 1 --seed 3 --machine ToyConveyor
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 4 --n_anomaly 2 --seed 3 --machine ToyCar
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 2 --seed 3 --machine ToyConveyor
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 4 --seed 3 --machine pump
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 4 --seed 3 --machine ToyConveyor
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 4 --n_anomaly 8 --seed 3 --machine fan
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 8 --seed 3 --machine pump
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 4 --n_anomaly 8 --seed 3 --machine ToyCar
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 8 --seed 3 --machine ToyConveyor
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 4 --n_anomaly 16 --seed 3 --machine fan
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 4 --n_anomaly 16 --seed 3 --machine ToyConveyor
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 32 --seed 3 --machine fan
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 4 --n_anomaly 0 --seed 4 --machine valve
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 4 --n_anomaly 0 --seed 4 --machine ToyCar
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 0 --seed 4 --machine ToyConveyor
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 1 --seed 4 --machine pump
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 1 --seed 4 --machine valve
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 2 --seed 4 --machine pump
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 2 --seed 4 --machine ToyConveyor
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 4 --seed 4 --machine valve
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 8 --seed 4 --machine ToyCar
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 8 --seed 4 --machine ToyConveyor
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 16 --seed 4 --machine ToyCar
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 32 --seed 4 --machine pump
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 4 --n_anomaly 32 --seed 4 --machine valve
./sub_job.sh --no audioset_v227 --stage 1 --start_stage 3 --n_anomaly 32 --seed 4 --machine ToyCar

# valid_ratio=0.15
# n_anomaly=-1
# for no in 029 030; do
#     for seed in 0 1 2 3 4; do
#         slurm_gpu_scheduler "${available_gpus}"
#         ./job.sh \
#             --no "audioset_v${no}" \
#             --stage "1" \
#             --start_stage "3" \
#             --seed "${seed}"
#     done
# done
# for no in 129 130 229 230; do
#     for seed in 0 1 2 3 4; do
#         for n_anomaly in 0 1 2 4 8 16 32; do
#             slurm_gpu_scheduler "${available_gpus}"
#             ./job.sh \
#                 --no "audioset_v${no}" \
#                 --audioset_pow "0" \
#                 --stage "1" \
#                 --start_stage "3" \
#                 --valid_ratio "${valid_ratio}" \
#                 --use_uav "false" \
#                 --use_idmt "false" \
#                 --n_anomaly "${n_anomaly}" \
#                 --seed "${seed}"
#         done
#     done
# done

# for no in 029 030; do
#     for seed in 0 1 2 3 4; do
#         ./job.sh \
#             --no "audioset_v${no}" \
#             --stage "2" \
#             --start_stage "3" \
#             --seed "${seed}"
#     done
# done
# for no in 129 130 229 230; do
#     for seed in 0 1 2 3 4; do
#         for n_anomaly in 0 1 2 4 8 16 32; do
#             ./job.sh \
#                 --no "audioset_v${no}" \
#                 --audioset_pow "0" \
#                 --stage "2" \
#                 --start_stage "3" \
#                 --valid_ratio "${valid_ratio}" \
#                 --use_uav "false" \
#                 --use_idmt "false" \
#                 --n_anomaly "${n_anomaly}" \
#                 --seed "${seed}"
#         done
#     done
# done
