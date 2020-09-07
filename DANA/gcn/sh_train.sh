#!/usr/bin/env bash
export PYTHONWARNINGS="ignore"


PYTHONPATH='..' python -W ignore train.py --dataset "douban_weibo" --left_directed=False --right_directed=False --learning_rate 3e-4 --lr_decay_step 200 --epochs 2000 --early_stopping 1200 --suffix "DANA" --logging_freq 50 --output_dim 100 --split_rate "82" >> douban_weibo_DANA_split82.out


 PYTHONPATH='..' python -W ignore train.py --dataset "dblp01" --left_directed=False --right_directed=False --learning_rate 3e-4 --lr_decay_step 50 --epochs 1200 --early_stopping 200 --suffix "DANA" --logging_freq 50 --output_dim 100 --split_rate "82" >> dblp01_DANA_split82.out

