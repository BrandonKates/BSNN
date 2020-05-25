#!/bin/bash
# Baseline Architectures
python -u main.py --no-cuda --plot-boundary --num-samples 400 --dataset linear --model bernoulli  --epochs 500 --hidden-layers 3  > ./log/baseline/linear.out 2> ./log/baseline/linear.err &
python -u main.py --no-cuda --plot-boundary --batch-size 10 --num-samples 400 --dataset xor --model bernoulli  --epochs 500 --hidden-layers 6 6 5  > ./log/baseline/xor.out 2> ./log/baseline/xor.err &
python -u main.py --no-cuda --plot-boundary --batch-size 10 --num-samples 400 --dataset circle --model bernoulli  --num-labels 4 --epochs 1000 --hidden-layers 6 6 6 5  > ./log/baseline/circle.out 2> ./log/baseline/circle.err &
python -u main.py --no-cuda --plot-boundary --batch-size 10 --num-samples 400 --dataset spiral --model bernoulli  --epochs 500 --hidden-layers 6 6 5  > ./log/baseline/spiral.out 2> ./log/baseline/spiral.err &