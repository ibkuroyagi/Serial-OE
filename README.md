# Anomalous Sound Detection using Serial method with Outlier Exposure

## Requirements
- Python 3.9+
- Cuda 11.3



## Setup
```bash
$ git clone https://github.com/ibkuroyagi/Serial-OE.git
$ cd Serial-OE/tools
$ make
```


## Recipe
To run the recipe, please follow the below instruction.

```bash
# Let us move on the recipe directory
$ cd scripts

# Run the recipe from scratch
$ ./job.sh

# You can change config via command line
$ ./job.sh --no <the_number_of_your_customized_yaml_config>

# You can select the stage to start and stop
$ ./job.sh --stage 1 --start_stage 3

# You can see the progress of the experiments.
$ . ./path.sh
$ python -m tensorboard.main --logdir exp

# After all machine types have completed Stage 5, starting Stage 2.
# You can see the results at exp/all/**/score*.csv
$ ./job.sh --stage 2

```


## Author

Ibuki Kuroyanagi ([@ibkuroyagi](https://github.com/ibkuroyagi))  
E-mail: `kuroyanagi.ibuki<at>g.sp.m.is.nagoya-u.ac.jp`
