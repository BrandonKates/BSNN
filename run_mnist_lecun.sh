#!/bin/bash

python -u main.py --batch-size 64 --log-interval 100 --dataset mnist --model bernoulli  --epochs 1000 --input-size 784 --save-model --save-location "./checkpoints/mnist300_" --hidden-layers 300 > ./log/mnist/300.out 2> ./log/mnist/300.err &
python -u main.py --batch-size 64 --log-interval 100 --dataset mnist --model bernoulli  --epochs 1000 --input-size 784 --save-model --save-location "./checkpoints/mnist1000_" --hidden-layers 1000 > ./log/mnist/1000.out 2> ./log/mnist/1000.err &

python -u main.py --batch-size 64 --log-interval 100 --dataset mnist --model bernoulli  --epochs 1000 --input-size 784 --save-model --save-location "./checkpoints/mnist300_100_" --hidden-layers 300 100 > ./log/mnist/300_100.out 2> ./log/mnist/300_100.err &
python -u main.py --batch-size 64 --log-interval 100 --dataset mnist --model bernoulli  --epochs 1000 --input-size 784 --save-model --save-location "./checkpoints/mnist1000_150_" --hidden-layers 1000 150 > ./log/mnist/1000_150.out 2> ./log/mnist/1000_150.err &


