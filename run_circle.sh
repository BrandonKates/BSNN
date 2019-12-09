#!/bin/bash

#python -u main.py --no-cuda --plot-boundary --batch-size 16 --num-samples 10000 --dataset circle --model bernoulli  --num-labels 10 --epochs 1000 --hidden-layers 15 14 13 12 11  > ./log/baseline/circle10class_5layer.out 2> ./log/baseline/circle.err &
#python -u main.py --no-cuda --plot-boundary --batch-size 64 --num-samples 60000 --dataset circle --model bernoulli  --num-labels 10 --epochs 2000 --hidden-layers 20 19 18 17 16 15 14 13 12 11 > ./log/baseline/circle10class_10layer.out 2> ./log/baseline/circle.err &
#python -u main.py --no-cuda --plot-boundary --batch-size 16 --num-samples 10000 --dataset circle --model bernoulli  --num-labels 10 --epochs 1000 --hidden-layers 200 100 50 25 15  > ./log/baseline/circle10class_5layer_large.out 2> ./log/baseline/circle.err &



function setup {
	DIR=$1
	shift
	LAYERS="$@"
	LAYERS=${LAYERS// /_}

	LOG_DIR="./log/${DIR}"
	CHECKPOINT_DIR="./checkpoints/${DIR}"

	mkdir -p $LOG_DIR
	mkdir -p $CHECKPOINT_DIR

	LOG_OUT="${LOG_DIR}layers${LAYERS}"
	CHECKPOINT_OUT="${CHECKPOINT_DIR}layers${LAYERS}"
	echo "$LOG_OUT" "$CHECKPOINT_OUT"
}


# Circle 2 Class

# Circle 10 Class
TRAIN_PASSES=$1
TEST_PASSES=$2
DIR_="circle/circle_10class/"
LAYERS_="${@:3}"
read LOG_SAVE_LOC CHECKPOINT_SAVE_LOC < <( setup $DIR_ $LAYERS_ )

python -u main.py --no-cuda --plot-boundary --batch-size 16 --num-samples 12000 --dataset circle --model bernoulli \
--num-labels 10 --epochs 3000 --save-model \
--save-location "${CHECKPOINT_SAVE_LOC}.pt" \
--hidden-layers $LAYERS_ \
--t-passes $TRAIN_PASSES \
--i-passes $TEST_PASSES \
> "${LOG_SAVE_LOC}.out" 2> "${LOG_SAVE_LOC}.err" &
