# Anomalous Sound Detection using Serial method with Outlier Exposure

## Requirements

- Python 3.9+
- Cuda 11.3

## Setup

```bash
git clone https://github.com/ibkuroyagi/Serial-OE.git
cd Serial-OE/tools
make
```

## Dataset

To use the eval directory of the DCASE 2020 Task2 Challenge dataset, put it in the dev directory in the same format.

```bash
scripts/downloads  (We expect dev directory contain all IDs {00,01,..,06}.)
|--dev
   |--fan
   |  |--test
   |  |  |--anomaly_id_00_00000000.wav
   |  |  |--anomaly_id_00_00000001.wav
   |  |  |--**
   |  |  |--normal_id_06_00000099.wav
   |  |--train
   |  |  |--normal_id_00_00000000.wav
   |  |  |--normal_id_00_00000001.wav
   |  |  |--**
   |  |  |--normal_id_06_00000914.wav
   |--pump
   |  |--test
   |  |  |--anomaly_id_00_00000000.wav
   |  |  |--anomaly_id_00_00000001.wav
   |  |  |--**
   |  |  |--normal_id_06_00000099.wav
   |  |--train
   |  |  |--normal_id_00_00000000.wav
   |  |  |--normal_id_00_00000001.wav
   |  |  |--**
   |  |  |--normal_id_06_00000914.wav
```

## Recipe

To run the recipe, please follow the below instruction.

```bash
# Let us move on the recipe directory.
$ cd scripts

# Run the recipe from scratch.
$ ./run.sh

# job.sh is a script for running jobs on multiple machine types.
# You can select the stage to start.
$ ./job.sh --stage 1 --start_stage 3

# You can change config via command line.
$ ./job.sh --model_name serial_oe.utilize --n_anomaly 16

# You can see the progress of the experiments.
$ . ./path.sh
$ python -m tensorboard.main --logdir exp

# After all machine types have completed Stage 5, starting Stage 2.
# You can see the results at exp/all/**/score*.csv.
$ ./job.sh --stage 2

```

## Author

Ibuki Kuroyanagi ([@ibkuroyagi](https://github.com/ibkuroyagi))  
E-mail: `kuroyanagi.ibuki<at>g.sp.m.is.nagoya-u.ac.jp`
