# EEG Deep Learning Baselines on BCI Competition IV (2a & 2b)

Reproducible PyTorch baselines (EEGNet & ShallowConvNet) for **motor imagery** on **BCI Competition IV** datasets:
- **2a** = BNCI2014-001 (4 classes: left hand, right hand, feet, tongue)
- **2b** = BNCI2014-004 (2 classes: left hand, right hand)

Data are fetched via **[MOABB](https://moabb.neurotechx.com/)** and preprocessed with **MNE**.

## Quickstart

```bash
# 1) Create env
python -m venv .venv && source .venv/bin/activate  # (Linux/Mac)
# or on Windows: .venv\Scripts\activate

pip install -r requirements.txt

# 2) Train & evaluate EEGNet on 2a with cross-subject LOSO
python -m src.eeg_bci.train --dataset 2a --model eegnet --mode cross_subject --epochs 40 --batch-size 64

# 3) Within-subject (per subject 5-fold CV) on 2b
python -m src.eeg_bci.train --dataset 2b --model shallow --mode within_subject --epochs 30
```

Outputs (checkpoints, metrics) are stored under `outputs/`.

## Datasets

We use MOABB's dataset wrappers:
- **BNCI2014_001** (BCI-IV 2a) → 9 subjects, 22 EEG channels, 4-class MI (left hand, right hand, feet, tongue).
- **BNCI2014_004** (BCI-IV 2b) → 9 subjects, 3 EEG channels (C3, Cz, C4), 2-class MI (left vs right hand).

MOABB will **auto-download** the data on first run into your local cache (MNE data dir). See MOABB docs for details.

## Protocols

- **Cross-subject**: Leave-One-Subject-Out (LOSO). Train on N-1 subjects, test on the held-out subject. Reports per-fold and macro stats.
- **Within-subject**: For each subject, 5-fold stratified CV on that subject's trials.

Key preprocessing steps (via MOABB + MNE):
- Band-pass (default `fmin=4, fmax=38` Hz), notch at 50/60 Hz if provided by MOABB defaults.
- Epoching w.r.t. MI cues using [`MotorImagery` paradigm].
- Optional resampling to 128 Hz (default).
- Standardization (z-score) **fit on training data only**, applied to validation/test.

## Models
- **EEGNet** (depthwise-separable CNN) — compact and strong baseline.
- **ShallowConvNet** (Schirrmeister et al.) — simple, fast, robust.

Model hyperparameters are exposed as CLI flags; see `python -m src.eeg_bci.train -h`.

## Citation
If you use this code, please cite the data owners and MOABB:

- Brunner et al. *BCI Competition IV 2a/2b* (BNCI2014-001/004).
- Jayaram & Barachant. *MOABB: trustworthy algorithm benchmarking for BCIs*, J Neural Eng., 2018.
- Gramfort et al. *MNE-Python*, NeuroImage, 2014.
- Lawhern et al. *EEGNet*, J Neural Eng., 2018.
- Schirrmeister et al. *Deep learning with CNNs for EEG decoding and visualization*, Hum Brain Mapp., 2017.

## License
MIT (see `LICENSE`).