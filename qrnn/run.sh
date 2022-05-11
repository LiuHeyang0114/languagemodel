#!/bin/bash
python main_qrnn.py --data data/gigaspeech --cuda --epochs 6 --model QRNN --nhid 512 --emsize 512 --lr 10 --nlayers 4 --clip 0.2 --dropouth 0.2 --dropouti 0.2 --dropoute 0.1

