#!/bin/bash
CUDA_VISIBLE_DEVICES=3,4 torchrun --standalone --nproc_per_node=2 main.py --config ./exps/cifar.json
#python main.py --config=./exps/cifar.json
#python main.py --config=./exps/cub.json
#python main.py --config=./exps/inr.json
#python main.py --config=./exps/ina.json
#python main.py --config=./exps/omn.json
#python main.py --config=./exps/vtab.json