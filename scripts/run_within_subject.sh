#!/usr/bin/env bash
set -e
python -m src.eeg_bci.train --dataset 2a --model shallow --mode within_subject --epochs 30 --batch-size 64
python -m src.eeg_bci.train --dataset 2b --model shallow --mode within_subject --epochs 30 --batch-size 64