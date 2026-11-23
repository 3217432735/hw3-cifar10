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
    parser = argparse.ArgumentParser(description="Generate Grad-CAM examples")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    return parser.parse_args()


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.activations = None
        self.gradients = None
        self.activation_handle = target_layer.register_forward_hook(self._forward_hook)
        self.gradient_handle = target_layer.register_full_backward_hook(
            self._backward_hook
        )

    def _forward_hook(self, _module, _inputs, output) -> None:
        self.activations = output

    def _backward_hook(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0]

    def generate(self, image: torch.Tensor, class_index: int | None = None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image)
        if class_index is None:
            class_index = int(logits.argmax(dim=1).item())
        score = logits[0, class_index]
        score.backward(retain_graph=False)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).relu()
        cam = cam[0].detach().cpu().numpy()
        cam -= cam.min()
        cam /= max(cam.max(), 1e-8)
        return cam, class_index

    def close(self) -> None:
        self.activation_handle.remove()
        self.gradient_handle.remove()


def _heatmap(cam: np.ndarray) -> np.ndarray:
    value = np.clip(cam, 0, 1)
    red = np.clip(1.5 - np.abs(4 * value - 3), 0, 1)
    green = np.clip(1.5 - np.abs(4 * value - 2), 0, 1)
    blue = np.clip(1.5 - np.abs(4 * value - 1), 0, 1)
    return np.stack((red, green, blue), axis=-1)


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = ("segoeuib.ttf", "arialbd.ttf") if bold else ("segoeui.ttf", "arial.ttf")
    for name in names:
        path = Path(r"C:\Windows\Fonts") / name
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def generate(args: argparse.Namespace) -> dict:
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    experiment = args.experiment or checkpoint.get("experiment") or checkpoint_path.stem
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_args = checkpoint["args"]
    _, _, test_loader = make_loaders(
        data_root=args.data_root,
        aug=train_args["aug"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=train_args["seed"],
    )
    model = build_model(checkpoint["model"], pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    cam_generator = GradCAM(model, model.target_layer())

    targets_all, predictions_all, confidences_all = [], [], []
    with torch.inference_mode():
        for images, targets in test_loader:
            logits = model(images.to(device, non_blocking=True))
            confidence, predictions = logits.softmax(dim=1).max(dim=1)
            targets_all.append(targets)
            predictions_all.append(predictions.cpu())
            confidences_all.append(confidence.cpu())
    targets = torch.cat(targets_all).numpy()
    predictions = torch.cat(predictions_all).numpy()
    confidences = torch.cat(confidences_all).numpy()

    selected_classes = (0, 1, 3, 5, 7, 9)
    selected = []
    for class_index in selected_classes:
        candidates = np.flatnonzero(
            (targets == class_index) & (predictions == class_index)
        )
        if len(candidates):
            selected.append(int(candidates[np.argmax(confidences[candidates])]))
    if len(selected) < 6:
        correct = np.flatnonzero(targets == predictions)
        selected.extend(
            int(index) for index in correct[: 6 - len(selected)].tolist()
        )

    dataset = test_loader.dataset
    records = []
    tiles = []
    for test_index in selected:
        image_tensor, target = dataset[test_index]
        image_tensor = image_tensor.unsqueeze(0).to(device)
        model.zero_grad(set_to_none=True)
        logits = model(image_tensor)
        predicted = int(logits.argmax(dim=1).item())
        confidence = float(logits.softmax(dim=1)[0, predicted].item())
        cam, _ = cam_generator.generate(image_tensor, predicted)
        base = denormalize(image_tensor.detach())[0].permute(1, 2, 0).cpu().numpy()
        base_image = Image.fromarray((base * 255).astype(np.uint8)).resize(
            (232, 232), Image.Resampling.NEAREST
        )
        heat = Image.fromarray((_heatmap(cam) * 255).astype(np.uint8)).resize(
            (232, 232), Image.Resampling.BILINEAR
        )
        overlay = Image.blend(base_image, heat, alpha=0.48)
        tile = Image.new("RGB", (486, 280), "white")
        tile.paste(base_image, (0, 0))
        tile.paste(overlay, (254, 0))
        draw = ImageDraw.Draw(tile)
        label = (
            f"true={CLASS_NAMES[int(target)]}, pred={CLASS_NAMES[predicted]} "
            f"({confidence:.1%})"
        )
        draw.text((0, 246), label, fill="black", font=_font(14))
        tiles.append(tile)
        records.append(
            {
                "test_index": test_index,
                "true_class": CLASS_NAMES[int(target)],
                "predicted_class": CLASS_NAMES[predicted],
                "confidence": confidence,
                "correct": int(target) == predicted,
            }
        )

    columns = 2
    rows = (len(tiles) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * 486, rows * 280), "white")
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        canvas.paste(tile, (column * 486, row * 280))
    output_path = RESULTS_DIR / f"{experiment}_gradcam.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    cam_generator.close()
    payload = {
        "experiment": experiment,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "method": "Grad-CAM on the last convolutional layer",
        "samples": records,
        "artifact": str(output_path.relative_to(ROOT)),
    }
    with (LOGS_DIR / f"{experiment}_gradcam.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"[{experiment}] saved {output_path}")
    return payload


if __name__ == "__main__":
    generate(parse_args())
