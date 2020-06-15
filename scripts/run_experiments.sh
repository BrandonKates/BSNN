#!/bin/bash

# NOTE: "canonical" logs should be in `./logs` and checked in to git
# This script outputs logs in the `scripts` directory to compare against the
# canonical logs when making changes

# first, lenet5 on mnist
python main.py -m lenet5 -d mnist --resize-input --deterministic --gpu 0 --epochs 15 --batch-size 64 > ./scripts/lenet5_deterministic.log 2> ./scripts/lenet5_deterministic.err &
python main.py -m lenet5 -d mnist --resize-input  --gpu 1 --epochs 15 --batch-size 64 > ./scripts/lenet5_stochastic.log 2> ./scripts/lenet5_stochastic.err &
python main.py -m lenet5 -d mnist --resize-input --gpu 2 --epochs 15 --batch-size 64 --normalize > ./scripts/lenet5_batchnorm.log 2> ./scripts/lenet5_batchnorm.err &

# then simpleconv on cifar10
python main.py -m simpleconv -d cifar10 --deterministic --gpu 3 --epochs 15 --batch-size 128 --lr .005 > ./scripts/simpleconv_deterministic.log 2> ./scripts/simpleconv_deterministic.err &
wait
python main.py -m simpleconv -d cifar10 --gpu 0 --epochs 15 --batch-size 128 --lr .005 > ./scripts/simpleconv_stochastic.log 2> ./scripts/simpleconv_stochastic.err &
python main.py -m simpleconv -d cifar10 --gpu 1 --epochs 15 --batch-size 128 -n > ./scripts/simpleconv_batchnorm.log 2> ./scripts/simpleconv_batchnorm.err &

# finally, complex conv on cifar10
python main.py -m complexconv -d cifar10 --gpu 2 --epochs 20 --batch-size 128 --lr .001 --deterministic > ./scripts/complexconv_deterministic.log 2> ./scripts/complexconv_deterministic.err &
python main.py -m complexconv -d cifar10 --gpu 3 --epochs 20 --batch-size 128 --lr .001 > ./scripts/complexconv_stochastic.log 2> ./scripts/complexconv_stochastic.err &
wait
python main.py -m complexconv -d cifar10 --gpu 0 --epochs 20 --batch-size 128 --normalize > ./scripts/complexconv_batchnorm.log 2> ./scripts/complexconv_batchnorm.err &
