#!/bin/bash


# python main.py --env-name "mreacher-v0" --algo ppo --use-gae --vis-interval 1  \
# --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 1 \
# --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --tau 0.95 --num-frames 1000000


python main.py --env-name "mreacher-v0" \
--algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 \
--value-loss-coef 1 --num-processes 8 --num-steps 128 \
--num-mini-batch 4 --vis-interval 1 --log-interval 1
