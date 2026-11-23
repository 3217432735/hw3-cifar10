from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn

from data import DEFAULT_DATA_ROOT, make_loaders
from model import build_model


ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = ROOT / "logs"
RESULTS_DIR = ROOT / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CIFAR-10 models")
    parser.add_argument("--model", choices=("cnn", "resnet18"), default="cnn")
    parser.add_argument(
        "--aug", choices=("none", "basic", "strong"), default="basic"
    )
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument(
        "--scheduler", choices=("none", "cosine"), default="cosine"
    )
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    use_amp = scaler is not None and scaler.is_enabled()

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        total_loss += loss.item() * targets.size(0)
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_samples += targets.size(0)
    return total_loss / total_samples, total_correct / total_samples


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = ("segoeuib.ttf", "arialbd.ttf") if bold else ("segoeui.ttf", "arial.ttf")
    for name in names:
        path = Path(r"C:\Windows\Fonts") / name
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def save_training_curve(history: list[dict], path: Path) -> None:
    width, height = 1400, 520
    margin = 90
    panel_gap = 100
    panel_width = (width - margin * 2 - panel_gap) // 2
    panel_height = height - margin * 2
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = _font(22)
    small = _font(16)
    title_font = _font(26, bold=True)

    metrics = (
        ("loss", "Loss", ("train_loss", "val_loss")),
        ("accuracy", "Accuracy", ("train_acc", "val_acc")),
    )
    colors = {"train": (35, 103, 194), "val": (210, 70, 55)}
    for panel_index, (metric, title, keys) in enumerate(metrics):
        left = margin + panel_index * (panel_width + panel_gap)
        top = margin
        right = left + panel_width
        bottom = top + panel_height
        draw.text((left, 24), title, fill="black", font=title_font)
        maximum = max(
            1.0,
            max(item[key] for item in history for key in keys),
        )
        for tick in range(6):
            y = bottom - tick * panel_height / 5
            draw.line((left, y, right, y), fill=(228, 230, 234), width=1)
            value = tick / 5 * maximum
            label = f"{value:.1f}"
            draw.text((14, y - 10), label, fill=(80, 80, 80), font=small)
        draw.line((left, top, left, bottom), fill="black", width=2)
        draw.line((left, bottom, right, bottom), fill="black", width=2)
        epochs = [item["epoch"] for item in history]
        x_values = epochs if len(epochs) > 1 else [0, 1]
        for split, key in zip(("train", "val"), keys):
            values = [item[key] for item in history]
            points = []
            for index, epoch in enumerate(epochs):
                x = left + (epoch - 1) / max(1, epochs[-1] - 1) * panel_width
                y = bottom - values[index] / maximum * panel_height
                points.append((x, y))
            if len(points) == 1:
                points = [(left, points[0][1]), (right, points[0][1])]
            if len(points) >= 2:
                draw.line(points, fill=colors[split], width=4)
            draw.text(
                (left + 16 + (0 if split == "train" else 115), bottom - 32),
                split,
                fill=colors[split],
                font=font,
            )
        draw.text(
            (left, bottom + 12),
            f"epoch 1 to {x_values[-1]}",
            fill="black",
            font=small,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def train(args: argparse.Namespace) -> dict:
    experiment = args.experiment or f"{args.model}_{args.aug}"
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _ = make_loaders(
        data_root=args.data_root,
        aug=args.aug,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    model = build_model(
        args.model, num_classes=10, pretrained=args.model == "resnet18"
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
        )
        if args.scheduler == "cosine"
        else None
    )
    amp_enabled = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    history = []
    best_state = None
    best_val_acc = -math.inf
    best_val_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler
        )
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.perf_counter() - epoch_started
        item = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
            "learning_rate": current_lr,
            "epoch_seconds": round(elapsed, 3),
        }
        history.append(item)
        print(
            f"[{experiment}] epoch {epoch:02d}/{args.epochs} "
            f"train loss={train_loss:.4f} acc={train_acc:.4f} "
            f"val loss={val_loss:.4f} acc={val_acc:.4f} "
            f"lr={current_lr:.2e} {elapsed:.1f}s",
            flush=True,
        )

        improved = (
            val_acc > best_val_acc + args.min_delta
            or (
                abs(val_acc - best_val_acc) <= args.min_delta
                and val_loss < best_val_loss
            )
        )
        if improved:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if scheduler is not None:
            scheduler.step()
        if (
            args.early_stop_patience > 0
            and epochs_without_improvement >= args.early_stop_patience
        ):
            print(f"[{experiment}] early stopping after epoch {epoch}", flush=True)
            break

    total_seconds = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("No checkpoint was saved")
    checkpoint_path = LOGS_DIR / f"{experiment}_best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "model": args.model,
            "experiment": experiment,
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "args": vars(args),
        },
        checkpoint_path,
    )
    curve_path = RESULTS_DIR / f"{experiment}_training_curves.png"
    save_training_curve(history, curve_path)
    payload = {
        "experiment": experiment,
        "model": args.model,
        "augmentation": args.aug,
        "seed": args.seed,
        "split": {"train": 45000, "validation": 5000, "test": 10000},
        "config": {
            **vars(args),
            "data_root": str(args.data_root),
            "amp": amp_enabled,
            "optimizer": "AdamW",
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
            ),
        },
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_val_acc,
        "best_validation_loss": best_val_loss,
        "epochs_completed": len(history),
        "early_stopped": len(history) < args.epochs,
        "training_seconds": round(total_seconds, 3),
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "training_curve": str(curve_path.relative_to(ROOT)),
        "history": history,
    }
    save_json(LOGS_DIR / f"{experiment}_train.json", payload)
    return payload


if __name__ == "__main__":
    result = train(parse_args())
    print(
        f"finished {result['experiment']}: "
        f"best val acc={result['best_validation_accuracy']:.4f}"
    )
