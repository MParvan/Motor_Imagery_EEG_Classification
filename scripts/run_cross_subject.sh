#!/usr/bin/env bash
set -e
python -m src.eeg_bci.train --dataset 2a --model eegnet --mode cross_subject --epochs 40 --batch-size 64
python -m src.eeg_bci.train --dataset 2b --model eegnet --mode cross_subject --epochs 30 --batch-size 64