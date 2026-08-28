import argparse
import json
import os
import time
from typing import Dict, List, Optional

import numpy as np

from .data import (
    DevelopmentSessionConfig,
    EEGTrials,
    ValidationSplitConfig,
    get_default_validation_group_by,
    get_dataset_session_config,
    get_loso_subject_splits,
    get_subject_specific_splits,
    load_mi_data,
)
from .utils import ZScoreScalerTorch, set_seed


def _import_torch():
    import torch
    from torch.utils.data import DataLoader, Dataset

    return torch, DataLoader, Dataset


def _import_training_metrics():
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    return accuracy_score, confusion_matrix, f1_score


def _import_tqdm():
    from tqdm import tqdm

    return tqdm


def create_arg_parser():
    parser = argparse.ArgumentParser(description="EEG DL on BCI-IV 2a/2b via MOABB")
    parser.add_argument("--dataset", choices=["2a", "2b"], required=True)
    parser.add_argument(
        "--model",
        choices=["eegnet", "shallow", "deepconvnet", "tcn", "eeginception", "fbcnet", "mbma_ciac"],
        default="eegnet",
    )
    parser.add_argument("--mode", choices=["cross_subject", "within_subject"], default="cross_subject")
    parser.add_argument(
        "--augment",
        choices=["none"],
        default="none",
        help="Augmentation is disabled in the supported training path.",
    )
    parser.add_argument(
        "--session-role-map-json",
        default=None,
        help='Advanced override for development/test sessions, e.g. {"development_sessions":["0train"],"test_sessions":["1test"]}.',
    )
    parser.add_argument(
        "--validation-group-by",
        default=None,
        help="Metadata field used to split development data into train/validation for within-subject mode.",
    )
    parser.add_argument(
        "--validation-group-count",
        type=int,
        default=1,
        help="Number of development groups to reserve for validation in within-subject mode.",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--resample", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")
    return parser


def _build_dataset(X, y):
    torch, _, dataset_base = _import_torch()

    class EEGDataset(dataset_base):
        def __init__(self, X_inner, y_inner):
            self.X = torch.from_numpy(X_inner).float()
            self.y = torch.from_numpy(y_inner).long()

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            return self.X[i], self.y[i]

    return EEGDataset(X, y)


def build_model(name, n_channels, n_classes):
    name = name.lower()
    if name in ["eegnet", "eegnetv4", "eegnet_v4"]:
        from .models.eegnet import EEGNet

        return EEGNet(n_channels=n_channels, n_classes=n_classes)
    if name in ["shallow", "shallowconvnet", "shallow_convnet"]:
        from .models.shallowconvnet import ShallowConvNet

        return ShallowConvNet(n_channels=n_channels, n_classes=n_classes)
    if name in ["deep", "deepconvnet"]:
        from .models.deepconvnet import DeepConvNet

        return DeepConvNet(n_channels=n_channels, n_classes=n_classes)
    if name in ["tcn", "tcnet"]:
        from .models.tcn import TCN

        return TCN(n_channels=n_channels, n_classes=n_classes)
    if name in ["eeg-inception", "eeginception", "inception"]:
        from .models.eeg_inception import EEGInception

        return EEGInception(n_channels=n_channels, n_classes=n_classes)
    if name in ["fbcnet", "fbc"]:
        from .models.fbcnet import FBCNet

        return FBCNet(n_channels=n_channels, n_classes=n_classes)
    if name in ["mbma", "ciac", "mbma_ciac", "ciacnet"]:
        from .models.mbma_ciac_lite import MBMA_CIAC_Lite

        return MBMA_CIAC_Lite(n_channels=n_channels, n_classes=n_classes)
    raise ValueError(f"Unknown model: {name}")


def _make_loader(X, y, batch_size, shuffle, drop_last):
    _, data_loader, _ = _import_torch()
    dataset = _build_dataset(X, y)
    return data_loader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


def evaluate_loader(model, loader, device):
    torch, _, _ = _import_torch()
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            gts.append(yb.numpy())
    return np.concatenate(preds), np.concatenate(gts)


def train_one(model, train_loader, val_loader, device, epochs=40, lr=1e-3, weight_decay=0.0, patience=8):
    torch, _, _ = _import_torch()
    accuracy_score, _, _ = _import_training_metrics()
    tqdm = _import_tqdm()
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = torch.nn.CrossEntropyLoss()
    best_acc, best_state, no_improve = -1.0, None, 0
    for _ in tqdm(range(1, epochs + 1)):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
        preds, gts = evaluate_loader(model, val_loader, device)
        acc = accuracy_score(gts, preds)
        if acc > best_acc:
            best_acc = float(acc)
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_acc


def summarize_accuracy(values: List[float]) -> float:
    if not values:
        raise ValueError("Cannot summarize accuracy for an empty list.")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _parse_session_config(dataset: str, raw_json: Optional[str]) -> DevelopmentSessionConfig:
    if raw_json is None:
        return get_dataset_session_config(dataset)
    return DevelopmentSessionConfig.from_json(dataset_name=dataset, raw_json=raw_json)


def _require_validation_config(args) -> ValidationSplitConfig:
    group_by = args.validation_group_by or get_default_validation_group_by(args.dataset)
    return ValidationSplitConfig(
        group_by=group_by,
        val_group_count=args.validation_group_count,
        seed=args.seed,
    ).validate()


def _fit_and_transform(trials: EEGTrials, train_idx, val_idx, test_idx):
    scaler = ZScoreScalerTorch()
    scaler.fit(trials.X[list(train_idx)])
    return (
        scaler.transform(trials.X[list(train_idx)]),
        scaler.transform(trials.X[list(val_idx)]),
        scaler.transform(trials.X[list(test_idx)]),
        scaler,
    )


def _collect_split_metadata(trials: EEGTrials, split) -> Dict[str, List[str]]:
    return {
        "train_subjects": list(split.metadata_values(trials, "train", "subject")),
        "val_subjects": list(split.metadata_values(trials, "val", "subject")),
        "test_subjects": list(split.metadata_values(trials, "test", "subject")),
        "train_sessions": list(split.metadata_values(trials, "train", "session")),
        "val_sessions": list(split.metadata_values(trials, "val", "session")),
        "test_sessions": list(split.metadata_values(trials, "test", "session")),
        "train_runs": list(split.metadata_values(trials, "train", "run")),
        "val_runs": list(split.metadata_values(trials, "val", "run")),
        "test_runs": list(split.metadata_values(trials, "test", "run")),
        "val_groups": list(split.val_groups),
    }


def run_cross_subject(args):
    torch, _, _ = _import_torch()
    accuracy_score, confusion_matrix, f1_score = _import_training_metrics()
    trials = load_mi_data(args.dataset, resample=args.resample)
    n_channels = trials.X.shape[1]
    n_classes = len(np.unique(trials.y))
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    results = []
    accuracies = []
    out_dir = os.path.join("outputs", args.dataset, "cross_subject", args.model, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    for fold, split in enumerate(get_loso_subject_splits(trials), start=1):
        Xtr, Xva, Xte, _ = _fit_and_transform(trials, split.train_idx, split.val_idx, split.test_idx)
        tl = _make_loader(Xtr, trials.y[list(split.train_idx)], batch_size=args.batch_size, shuffle=True, drop_last=True)
        vl = _make_loader(Xva, trials.y[list(split.val_idx)], batch_size=args.batch_size, shuffle=False, drop_last=False)
        te = _make_loader(Xte, trials.y[list(split.test_idx)], batch_size=args.batch_size, shuffle=False, drop_last=False)
        model = build_model(args.model, n_channels, n_classes)
        model, best_val = train_one(
            model,
            tl,
            vl,
            device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
        )
        preds, gts = evaluate_loader(model, te, device)
        acc = float(accuracy_score(gts, preds))
        accuracies.append(acc)
        results.append(
            {
                "fold": fold,
                "test_subject": split.subject,
                "val_subject": split.val_subject,
                "val_best_acc": float(best_val),
                "test_acc": acc,
                "test_f1_macro": float(f1_score(gts, preds, average="macro")),
                "confusion_matrix": confusion_matrix(gts, preds).tolist(),
                **_collect_split_metadata(trials, split),
            }
        )
        torch.save(model.state_dict(), os.path.join(out_dir, f"ckpt_sub-{split.subject}.pt"))
    payload = {"folds": results, "summary": {"mean_test_acc": summarize_accuracy(accuracies)}}
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print("Saved:", out_dir)
    print("Mean accuracy for all subjects:", payload["summary"]["mean_test_acc"])


def run_within_subject(args):
    torch, _, _ = _import_torch()
    accuracy_score, confusion_matrix, f1_score = _import_training_metrics()
    trials = load_mi_data(args.dataset, resample=args.resample)
    session_config = _parse_session_config(args.dataset, args.session_role_map_json)
    validation_config = _require_validation_config(args)
    n_channels = trials.X.shape[1]
    n_classes = len(np.unique(trials.y))
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out_dir = os.path.join("outputs", args.dataset, "within_subject", args.model, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    results = []
    accuracies = []
    for fold, split in enumerate(
        get_subject_specific_splits(
            trials,
            dataset=args.dataset,
            validation_config=validation_config,
            session_config=session_config,
        ),
        start=1,
    ):
        Xtr, Xva, Xte, _ = _fit_and_transform(trials, split.train_idx, split.val_idx, split.test_idx)
        tl = _make_loader(Xtr, trials.y[list(split.train_idx)], batch_size=args.batch_size, shuffle=True, drop_last=True)
        vl = _make_loader(Xva, trials.y[list(split.val_idx)], batch_size=args.batch_size, shuffle=False, drop_last=False)
        te = _make_loader(Xte, trials.y[list(split.test_idx)], batch_size=args.batch_size, shuffle=False, drop_last=False)
        model = build_model(args.model, n_channels, n_classes)
        print(f"Training subject {split.subject}, fold {fold}...")
        model, best_val = train_one(
            model,
            tl,
            vl,
            device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
        )
        preds, gts = evaluate_loader(model, te, device)
        acc = float(accuracy_score(gts, preds))
        accuracies.append(acc)
        results.append(
            {
                "fold": fold,
                "subject": split.subject,
                "val_best_acc": float(best_val),
                "test_acc": acc,
                "test_f1_macro": float(f1_score(gts, preds, average="macro")),
                "confusion_matrix": confusion_matrix(gts, preds).tolist(),
                **_collect_split_metadata(trials, split),
            }
        )
    payload = {"folds": results, "summary": {"mean_test_acc": summarize_accuracy(accuracies)}}
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print("Saved:", out_dir)
    print("Mean accuracy for all subjects:", payload["summary"]["mean_test_acc"])


def main():
    parser = create_arg_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    if args.mode == "cross_subject":
        run_cross_subject(args)
    else:
        run_within_subject(args)


if __name__ == "__main__":
    main()
