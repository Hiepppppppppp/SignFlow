
"""Train a landmark classifier with horizontal-flip augmentation and class balancing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, Subset

try:
    from .config import DEFAULT_CACHE_PATH, DEFAULT_MODEL_PATH, FEATURE_SIZE
    from .model import LandmarkClassifier
except ImportError:
    from config import DEFAULT_CACHE_PATH, DEFAULT_MODEL_PATH, FEATURE_SIZE
    from model import LandmarkClassifier


def stratified_indices(y: np.ndarray, val_ratio: float, seed: int = 42):
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []

    for label in np.unique(y):
        idx = np.where(y == label)[0]
        rng.shuffle(idx)

        n_val = max(1, int(round(len(idx) * val_ratio))) if len(idx) >= 2 else 0
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def horizontal_flip(features: np.ndarray) -> np.ndarray:
    """Mirror normalized landmarks across the vertical image axis."""
    points = features.reshape(-1, 21, 3).copy()
    points[:, :, 0] *= -1.0
    return points.reshape(-1, 63)


def macro_accuracy(y_true, y_pred, num_classes):
    scores = []

    for class_id in range(num_classes):
        mask = y_true == class_id
        if np.any(mask):
            scores.append(np.mean(y_pred[mask] == y_true[mask]))

    return float(np.mean(scores)) if scores else 0.0


def main(data_path: Path, model_path: Path, epochs: int,
         batch_size: int, learning_rate: float):

    if not data_path.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {data_path}. Run build_dataset.py first."
        )

    saved = np.load(data_path, allow_pickle=False)
    X = saved["X"].astype(np.float32)
    y = saved["y"].astype(np.int64)
    class_names = saved["class_names"].astype(str).tolist()

    if X.ndim != 2 or X.shape[1] != FEATURE_SIZE:
        raise ValueError(f"Expected X shape (N, {FEATURE_SIZE}), got {X.shape}")

    if len(X) != len(y):
        raise ValueError("X and y have different sample counts.")

    if len(np.unique(y)) < 2:
        raise ValueError("Training requires at least two classes.")

    if np.any(y < 0) or np.any(y >= len(class_names)):
        raise ValueError("Labels do not match class_names.")

    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Split original samples BEFORE augmentation.
    train_idx, val_idx = stratified_indices(y, 0.2)

    X_train = X[train_idx]
    y_train = y[train_idx]

    X_val = X[val_idx]
    y_val = y[val_idx]

    # Add mirrored copies only to the training set.
    X_mirrored = horizontal_flip(X_train)
    X_train = np.concatenate([X_train, X_mirrored], axis=0)
    y_train = np.concatenate([y_train, y_train.copy()], axis=0)

    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train)
    )

    val_dataset = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # Class weights are calculated from original training samples.
    counts = np.bincount(y[train_idx], minlength=len(class_names))
    weights = np.zeros(len(class_names), dtype=np.float32)

    present = counts > 0
    weights[present] = len(train_idx) / (
        int(present.sum()) * counts[present]
    )

    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    model = LandmarkClassifier(FEATURE_SIZE, len(class_names)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_macro = -1.0
    best_accuracy = 0.0

    print(f"Device: {device}")
    print(f"Original samples: {len(X)}")
    print(f"Training samples after flip augmentation: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print("\nOriginal training samples per class:")

    for i, name in enumerate(class_names):
        print(f"  {name:12s}: {counts[i]}")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(yb)
            total_count += len(yb)

        model.eval()
        all_true, all_pred = [], []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                predictions = model(xb).argmax(dim=1).cpu().numpy()

                all_pred.extend(predictions.tolist())
                all_true.extend(yb.numpy().tolist())

        all_true = np.asarray(all_true)
        all_pred = np.asarray(all_pred)

        val_accuracy = float(np.mean(all_true == all_pred))
        val_macro = macro_accuracy(
            all_true, all_pred, len(class_names)
        )

        avg_loss = total_loss / max(total_count, 1)

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"loss={avg_loss:.4f} | "
            f"val_acc={val_accuracy:.4f} | "
            f"val_macro={val_macro:.4f}"
        )

        # Select the model using macro accuracy, not overall accuracy alone.
        if val_macro > best_macro:
            best_macro = val_macro
            best_accuracy = val_accuracy

            model_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "input_size": FEATURE_SIZE,
                    "num_classes": len(class_names),
                    "val_accuracy": val_accuracy,
                    "val_macro_accuracy": val_macro,
                    "horizontal_flip_augmentation": True,
                },
                model_path
            )

            json_path = model_path.parent / "landmark_classes.json"
            json_path.write_text(
                json.dumps(class_names, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )

    print("\nTraining finished.")
    print(f"Best validation accuracy: {best_accuracy:.4f}")
    print(f"Best validation macro accuracy: {best_macro:.4f}")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)

    args = parser.parse_args()

    main(
        args.data,
        args.output,
        args.epochs,
        args.batch_size,
        args.lr
    )
