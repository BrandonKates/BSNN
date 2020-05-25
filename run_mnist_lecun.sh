#!/bin/bash

# MNIST architecture (specified with lecun)
#python -u main.py --no-cuda --batch-size 64 --log-interval 100 --dataset mnist --model bernoulli  --epochs 1000 --input-size 784 --save-model --save-location "./checkpoints/mnist300_epoch" --hidden-layers 300 > ./log/mnist/300.out 2> ./log/mnist/300.err &
#python -u main.py --no-cuda --batch-size 64 --log-interval 100 --dataset mnist --model bernoulli  --epochs 1000 --input-size 784 --save-model --save-location "./checkpoints/mnist1000_epoch" --hidden-layers 1000 > ./log/mnist/1000.out 2> ./log/mnist/1000.err &

#python -u main.py --no-cuda --batch-size 64 --log-interval 100 --dataset mnist --model bernoulli  --epochs 1000 --input-size 784 --save-model --save-location "./checkpoints/mnist300_100_epoch" --hidden-layers 300 100 > ./log/mnist/300_100.out 2> ./log/mnist/300_100.err &
#python -u main.py --no-cuda --batch-size 64 --log-interval 100 --dataset mnist --model bernoulli  --epochs 1000 --input-size 784 --save-model --save-location "./checkpoints/mnist1000_150_epoch" --hidden-layers 1000 150 > ./log/mnist/1000_150.out 2> ./log/mnist/1000_150.err &


# MNIST Architecture with many layers
python -u main.py --no-cuda --batch-size 64 --log-interval 100 --dataset mnist --model bernoulli  --epochs 1000 --input-size 784 --save-model --save-location "./checkpoints/mnist_pyramid1pass_epoch" --hidden-layers 100 95 90 85 80 75 70 65 60 55 50 45 40 35 30 25 20 15 10 > ./log/mnist/pyramid_1pass.out 2> ./log/mnist/pyramid_1pass.err &
#python -u main.py --no-cuda --batch-size 64 --log-interval 100 --dataset mnist --model bernoulli  --epochs 1000 --input-size 784 --save-model --save-location "./checkpoints/mnist1000_150_epoch" --hidden-layers 1000 150 > ./log/mnist/1000_150.out 2> ./log/mnist/1000_150.err &
