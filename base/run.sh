#!/bin/bash
python main.py --data data/gigaspeech --cuda --epochs 6 --model Transformer --dropout 0.1 --nlayers 6 --nhid 512 --emsize 512 --nhead 16 --lr 1
python main.py --data data/gigaspeech --cuda --epochs 6 --model LSTM --dropout 0.2 --nlayers 2 --nhid 512 --emsize 512 --lr 20 --clip 0.1
python main.py --data data/gigaspeech --cuda --epochs 6 --model GRU --nlayers 2 --nhid 512 --emsize 512 --lr 10 --clip 0.4 --dropout 0.2
python main.py --data data/gigaspeech --cuda --epochs 6 --model RNN_TANH --nhid 512 --emsize 512 --lr 1 --nlayers 4 --clip 0.4 --dropout 0.1
python main.py --data data/gigaspeech --cuda --epochs 6 --model RNN_RELU --nhid 512 --emsize 512 --lr 1 --nlayers 2 --clip 0.6 --dropout 0.1
