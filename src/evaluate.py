from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from data import CLASS_NAMES, DEFAULT_DATA_ROOT, denormalize, make_loaders
from model import build_model


ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = ROOT / "logs"
RESULTS_DIR = ROOT / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a CIFAR-10 checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    return parser.parse_args()


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = ("segoeuib.ttf", "arialbd.ttf") if bold else ("segoeui.ttf", "arial.ttf")
    for name in names:
        path = Path(r"C:\Windows\Fonts") / name
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _text_color(value: float) -> tuple[int, int, int]:
    return (255, 255, 255) if value > 0.55 else (20, 20, 20)


def save_confusion_matrix(matrix: np.ndarray, path: Path) -> None:
    cell = 78
    left, top = 150, 125
    width = left + cell * 10 + 45
    height = top + cell * 10 + 90
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = _font(28, bold=True)
    label_font = _font(17)
    cell_font = _font(16)
    draw.text((left, 28), "Confusion Matrix (rows: true, columns: predicted)", fill="black", font=title_font)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = matrix / np.maximum(row_sums, 1)
    for row in range(10):
        draw.text((15, top + row * cell + 25), CLASS_NAMES[row], fill="black", font=label_font)
        for column in range(10):
            value = float(normalized[row, column])
            color = (
                int(244 - 190 * value),
                int(247 - 120 * value),
                int(250 - 55 * value),
            )
            x0 = left + column * cell
            y0 = top + row * cell
            draw.rectangle((x0, y0, x0 + cell - 2, y0 + cell - 2), fill=color)
            text = str(int(matrix[row, column]))
            bbox = draw.textbbox((0, 0), text, font=cell_font)
            tx = x0 + (cell - (bbox[2] - bbox[0])) / 2
            ty = y0 + (cell - (bbox[3] - bbox[1])) / 2 - 2
            draw.text((tx, ty), text, fill=_text_color(value), font=cell_font)
    for column, name in enumerate(CLASS_NAMES):
        angle_image = Image.new("RGBA", (260, 30), (255, 255, 255, 0))
        angle_draw = ImageDraw.Draw(angle_image)
        angle_draw.text((4, 4), name, fill="black", font=label_font)
        rotated = angle_image.rotate(35, expand=True)
        x = left + column * cell + 8
        image.paste(rotated, (x, top + cell * 10 + 8), rotated)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def save_class_accuracy(per_class: dict, overall: float, path: Path) -> None:
    width, height = 1250, 680
    left, top, right, bottom = 100, 70, width - 40, height - 170
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = _font(28, bold=True)
    label_font = _font(17)
    draw.text((left, 20), f"Per-class accuracy (overall {overall:.2%})", fill="black", font=title_font)
    for tick in range(6):
        y = bottom - tick * (bottom - top) / 5
        draw.line((left, y, right, y), fill=(228, 230, 234), width=1)
        draw.text((28, y - 9), f"{tick * 20}%", fill=(70, 70, 70), font=label_font)
    slot = (right - left) / 10
    for index, name in enumerate(CLASS_NAMES):
        value = per_class[name]
        x0 = left + index * slot + 12
        x1 = left + (index + 1) * slot - 12
        y = bottom - value * (bottom - top)
        draw.rectangle((x0, y, x1, bottom), fill=(48, 119, 181))
        text = f"{value:.1%}"
        bbox = draw.textbbox((0, 0), text, font=label_font)
        draw.text(((x0 + x1 - bbox[2]) / 2, y - 27), text, fill="black", font=label_font)
        axis_image = Image.new("RGBA", (190, 30), (255, 255, 255, 0))
        axis_draw = ImageDraw.Draw(axis_image)
        axis_draw.text((3, 3), name, fill="black", font=label_font)
        rotated = axis_image.rotate(28, expand=True)
        image.paste(rotated, (int(x0) - 25, bottom + 8), rotated)
    image.save(path)


def save_misclassified(errors: list[dict], path: Path) -> None:
    count = min(len(errors), 12)
    columns = 4
    rows = max(1, (count + columns - 1) // columns)
    cell_w, cell_h = 300, 265
    image = Image.new("RGB", (columns * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(image)
    title_font = _font(18, bold=True)
    small = _font(15)
    for index, error in enumerate(errors[:count]):
        row, column = divmod(index, columns)
        x, y = column * cell_w, row * cell_h
        tensor = error.pop("_image")
        image.paste(tensor, (x + 12, y + 12))
        draw.text(
            (x + 12, y + 205),
            f"true: {error['true_class']}",
            fill=(22, 110, 50),
            font=title_font,
        )
        draw.text(
            (x + 12, y + 230),
            f"pred: {error['predicted_class']} ({error['confidence']:.1%})",
            fill=(180, 45, 35),
            font=small,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def evaluate(args: argparse.Namespace) -> dict:
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    experiment = args.experiment or checkpoint.get("experiment") or checkpoint_path.stem
    model_name = checkpoint["model"]
    aug = checkpoint["args"]["aug"]
    seed = checkpoint["args"]["seed"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = make_loaders(
        data_root=args.data_root,
        aug=aug,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=seed,
    )
    model = build_model(model_name, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    targets_all = []
    predictions_all = []
    confidences_all = []
    previews = []
    with torch.inference_mode():
        for images, targets in test_loader:
            logits = model(images.to(device, non_blocking=True))
            probabilities = logits.softmax(dim=1)
            confidence, predictions = probabilities.max(dim=1)
            targets_all.append(targets)
            predictions_all.append(predictions.cpu())
            confidences_all.append(confidence.cpu())
            previews.append((images.cpu(), targets, predictions.cpu(), confidence.cpu()))

    targets = torch.cat(targets_all).numpy()
    predictions = torch.cat(predictions_all).numpy()
    confidences = torch.cat(confidences_all).numpy()
    confusion = np.zeros((10, 10), dtype=np.int64)
    np.add.at(confusion, (targets, predictions), 1)
    accuracy = float((targets == predictions).mean())
    per_class = {
        name: float(confusion[index, index] / confusion[index].sum())
        for index, name in enumerate(CLASS_NAMES)
    }
    top_confusions = []
    for true_index in range(10):
        for predicted_index in range(10):
            if true_index != predicted_index:
                top_confusions.append(
                    {
                        "true_class": CLASS_NAMES[true_index],
                        "predicted_class": CLASS_NAMES[predicted_index],
                        "count": int(confusion[true_index, predicted_index]),
                    }
                )
    top_confusions.sort(key=lambda item: item["count"], reverse=True)

    wrong_indices = np.flatnonzero(targets != predictions)
    wrong_indices = wrong_indices[np.argsort(-confidences[wrong_indices])]
    flat_images = []
    flat_targets = []
    flat_predictions = []
    flat_confidences = []
    for images, batch_targets, batch_predictions, batch_confidences in previews:
        flat_images.append(images)
        flat_targets.append(batch_targets)
        flat_predictions.append(batch_predictions)
        flat_confidences.append(batch_confidences)
    all_images = torch.cat(flat_images)
    flat_targets = torch.cat(flat_targets).numpy()
    flat_predictions = torch.cat(flat_predictions).numpy()
    flat_confidences = torch.cat(flat_confidences).numpy()
    error_records = []
    for index in wrong_indices[:20].tolist():
        image_tensor = denormalize(all_images[index].unsqueeze(0))[0]
        pil_image = Image.fromarray(
            (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        ).resize((276, 185), Image.Resampling.NEAREST)
        record = {
            "test_index": int(index),
            "true_class": CLASS_NAMES[int(flat_targets[index])],
            "predicted_class": CLASS_NAMES[int(flat_predictions[index])],
            "confidence": float(flat_confidences[index]),
            "_image": pil_image,
        }
        error_records.append(record)

    confusion_path = RESULTS_DIR / f"{experiment}_confusion_matrix.png"
    per_class_path = RESULTS_DIR / f"{experiment}_per_class_accuracy.png"
    examples_path = RESULTS_DIR / f"{experiment}_misclassified.png"
    save_confusion_matrix(confusion, confusion_path)
    save_class_accuracy(per_class, accuracy, per_class_path)
    save_misclassified(error_records, examples_path)
    serializable_errors = [
        {key: value for key, value in record.items() if key != "_image"}
        for record in error_records
    ]
    payload = {
        "experiment": experiment,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "split": "official test set (10000 images, final evaluation only)",
        "test_samples": int(len(targets)),
        "total_accuracy": accuracy,
        "total_error_rate": 1 - accuracy,
        "per_class_accuracy": per_class,
        "macro_average_accuracy": float(np.mean(list(per_class.values()))),
        "confusion_matrix": confusion.tolist(),
        "top_confusions": top_confusions[:15],
        "misclassified_examples": serializable_errors,
        "artifacts": {
            "confusion_matrix": str(confusion_path.relative_to(ROOT)),
            "per_class_accuracy": str(per_class_path.relative_to(ROOT)),
            "misclassified_examples": str(examples_path.relative_to(ROOT)),
        },
    }
    log_path = LOGS_DIR / f"{experiment}_test.json"
    with log_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(
        f"[{experiment}] test accuracy={accuracy:.4f} "
        f"macro={payload['macro_average_accuracy']:.4f}"
    )
    return payload


if __name__ == "__main__":
    evaluate(parse_args())
