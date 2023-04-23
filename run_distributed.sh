#!/bin/bash

RANK=0
HOSTS=("austin" "baton-rouge" "hartford" "oklahoma-city")
NUM_NODES=${#HOSTS[@]}
RENDEVOUS_HOST="${HOSTS[0]}:42424"
TRAIN_SCRIPT=`pwd`/train_script.py

for host in ${HOSTS[@]}; do
    ssh $host torchrun \
        --nproc_per_node=1 \
        --nnodes=$NUM_NODES \
        --node_rank=$RANK \
        --rdzv_id=42424 \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$RENDEVOUS_HOST \
        $TRAIN_SCRIPT "$@" &
    ((RANK=RANK+1))
done
