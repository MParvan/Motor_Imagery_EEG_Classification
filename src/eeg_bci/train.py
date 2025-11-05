import argparse, os, time, json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm
from .data import load_mi_data, get_loso_splits, get_within_subject_splits, zscore_fit_transform_train_test
from .models.eegnet import EEGNet
from .models.shallowconvnet import ShallowConvNet
from .models.deepconvnet import DeepConvNet
from .models.tcn import TCN 
from .utils import set_seed

class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return self.X[i], self.y[i]

def build_model(name, n_channels, n_classes):
    name = name.lower()
    if name in ["eegnet", "eegnetv4", "eegnet_v4"]:
        return EEGNet(n_channels=n_channels, n_classes=n_classes)
    elif name in ["shallow", "shallowconvnet", "shallow_convnet"]:
        return ShallowConvNet(n_channels=n_channels, n_classes=n_classes)
    elif name in ["deep", "deepconvnet"]:
        return DeepConvNet(n_channels=n_channels, n_classes=n_classes)
    elif name in ["tcn", "eeG-tcnet", "tcnet"]:
        return TCN(n_channels=n_channels, n_classes=n_classes)
    else:
        raise ValueError("Unknown model: %s" % name)

def train_one(model, train_loader, val_loader, device, epochs=40, lr=1e-3, weight_decay=0.0, patience=8):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = torch.nn.CrossEntropyLoss()
    best_acc, best_state, no_improve = -1, None, 0
    for ep in tqdm(range(1, epochs+1)):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
        # val
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                preds.append(torch.argmax(logits, dim=1).cpu().numpy())
                gts.append(yb.numpy())
        preds = np.concatenate(preds); gts = np.concatenate(gts)
        acc = accuracy_score(gts, preds)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break
    # load best
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_acc

def run_cross_subject(args):
    X, y, groups = load_mi_data(args.dataset, resample=args.resample)
    n_channels = X.shape[1]
    n_classes = len(np.unique(y))
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    results = []
    out_dir = os.path.join("outputs", args.dataset, "cross_subject", args.model, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    fold = 0
    for train_idx, test_idx, test_sub in get_loso_splits(groups):
        fold += 1
        Xtr, Xte, scaler = zscore_fit_transform_train_test(X, train_idx, test_idx)
        # split train into train/val
        n = len(Xtr)
        idx = np.arange(n)
        np.random.shuffle(idx)
        split = int(0.9 * n)
        tr_idx, val_idx = idx[:split], idx[split:]
        ds_tr = EEGDataset(Xtr[tr_idx], y[train_idx][tr_idx])
        ds_va = EEGDataset(Xtr[val_idx], y[train_idx][val_idx])
        ds_te = EEGDataset(Xte, y[test_idx])

        tl = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, drop_last=True)
        vl = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False)
        te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False)

        model = build_model(args.model, n_channels, n_classes)
        model, best_val = train_one(model, tl, vl, device, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, patience=args.patience)
        # test
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for xb, yb in te:
                xb = xb.to(device)
                logits = model(xb)
                preds.append(torch.argmax(logits, dim=1).cpu().numpy())
                gts.append(yb.numpy())
        preds = np.concatenate(preds); gts = np.concatenate(gts)
        acc = accuracy_score(gts, preds)
        f1 = f1_score(gts, preds, average="macro")
        cm = confusion_matrix(gts, preds).tolist()
        results.append({"fold": fold, "test_subject": int(test_sub), "val_best_acc": float(best_val), "test_acc": float(acc), "test_f1_macro": float(f1), "confusion_matrix": cm})
        # save checkpoint per fold
        torch.save(model.state_dict(), os.path.join(out_dir, f"ckpt_sub-{test_sub}.pt"))
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("Saved:", out_dir)

def run_within_subject(args):
    X, y, groups = load_mi_data(args.dataset, resample=args.resample)
    n_channels = X.shape[1]
    n_classes = len(np.unique(y))
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out_dir = os.path.join("outputs", args.dataset, "within_subject", args.model, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    results = []
    current_sub = None
    fold = 0
    for train_idx, val_idx, sub in get_within_subject_splits(groups, y, n_folds=5):
        if current_sub != sub:
            current_sub = sub
            fold = 0
        fold += 1
        Xtr, Xva, _ = zscore_fit_transform_train_test(X, train_idx, val_idx)
        ds_tr = EEGDataset(Xtr, y[train_idx])
        ds_va = EEGDataset(Xva, y[val_idx])
        tl = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, drop_last=True)
        vl = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False)
        model = build_model(args.model, n_channels, n_classes)
        print(f"Training subject {sub}, fold {fold}...")
        model, best_val = train_one(model, tl, vl, device, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, patience=args.patience)
        preds, gts = [], []
        model.eval()
        with torch.no_grad():
            for xb, yb in vl:
                xb = xb.to(device)
                logits = model(xb)
                preds.append(torch.argmax(logits, dim=1).cpu().numpy())
                gts.append(yb.numpy())
        preds = np.concatenate(preds); gts = np.concatenate(gts)
        acc = accuracy_score(gts, preds)
        f1 = f1_score(gts, preds, average="macro")
        results.append({"fold": fold, "subject": int(sub), "val_best_acc": float(best_val), "fold_acc": float(acc), "fold_f1_macro": float(f1)})
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("Saved:", out_dir)

def main():
    parser = argparse.ArgumentParser(description="EEG DL on BCI-IV 2a/2b via MOABB")
    parser.add_argument("--dataset", choices=["2a","2b"], required=True)
    parser.add_argument("--model", choices=["eegnet","shallow"], default="eegnet")
    parser.add_argument("--mode", choices=["cross_subject","within_subject"], default="cross_subject")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--resample", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")
    args = parser.parse_args()
    set_seed(args.seed)
    if args.mode == "cross_subject":
        run_cross_subject(args)
    else:
        run_within_subject(args)

if __name__ == "__main__":
    main()