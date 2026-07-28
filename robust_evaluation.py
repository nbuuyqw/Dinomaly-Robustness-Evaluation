
"""
Dinomaly robustness evaluation
- Clean baseline (no corruption)
- Gaussian blur: 5 severity levels
- Low light: 5 severity levels
- Standard inference and horizontal-flip TTA
- Save each category result immediately to JSON
- Resume automatically after interruption

Run this script on a SLURM GPU compute node, not directly on the tinyx login node.
"""

import gc
import json
import os
import time
from functools import partial
from typing import Any, Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset

from dataset import MVTecDataset, get_data_transforms
from models import vit_encoder
from models.uad import ViTill
from models.vision_transformer import Block as VitBlock, LinearAttention2, bMlp
from utils import evaluation_batch


# -----------------------------------------------------------------------------
# Global configuration
# -----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# These values must match the normalization used by dataset.py.
MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

SEVERITIES = [1, 2, 3, 4, 5]

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
    "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
    "transistor", "wood", "zipper",
]


# -----------------------------------------------------------------------------
# Corruptions
# -----------------------------------------------------------------------------
def _check_severity(severity: int) -> None:
    if severity not in SEVERITIES:
        raise ValueError(f"severity must be one of {SEVERITIES}, got {severity}")


def gaussian_blur_corrupt(img_np: np.ndarray, severity: int) -> np.ndarray:
    """Apply Gaussian blur. Input/output: uint8 HWC RGB array."""
    _check_severity(severity)
    radius_levels = [1, 2, 3, 4, 6]
    radius = radius_levels[severity - 1]
    image = Image.fromarray(img_np)
    image = image.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(image)


def low_light_corrupt(img_np: np.ndarray, severity: int) -> np.ndarray:
    """Simulate low light using gamma correction; larger gamma is darker."""
    _check_severity(severity)
    gamma_levels = [1.5, 2.0, 2.8, 3.6, 4.5]
    gamma = gamma_levels[severity - 1]
    img_float = img_np.astype(np.float32) / 255.0
    img_dark = np.power(img_float, gamma)
    return np.clip(img_dark * 255.0, 0, 255).astype(np.uint8)


CORRUPTIONS: Dict[str, Callable[[np.ndarray, int], np.ndarray]] = {
    "gaussian_blur": gaussian_blur_corrupt,
    "low_light": low_light_corrupt,
}


# -----------------------------------------------------------------------------
# Dataset wrapper: corrupt only test image; preserve mask/label/path
# -----------------------------------------------------------------------------
class CorruptedWrapper(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        corruption_fn: Optional[Callable[[np.ndarray, int], np.ndarray]],
        severity: int,
    ) -> None:
        self.base_dataset = base_dataset
        self.corruption_fn = corruption_fn
        self.severity = severity

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        img, gt, label, img_path = self.base_dataset[index]

        if self.corruption_fn is not None:
            # Undo normalization -> uint8 HWC.
            img_unnormalized = torch.clamp(img.cpu() * STD + MEAN, 0.0, 1.0)
            img_np = (
                img_unnormalized.permute(1, 2, 0).numpy() * 255.0
            ).round().astype(np.uint8)

            # Apply corruption on CPU.
            img_np = self.corruption_fn(img_np, self.severity)

            # uint8 HWC -> normalized float tensor CHW.
            img_float = img_np.astype(np.float32) / 255.0
            img = torch.from_numpy(img_float).permute(2, 0, 1)
            img = (img - MEAN) / STD

        return img, gt, label, img_path


# -----------------------------------------------------------------------------
# TTA wrapper
# -----------------------------------------------------------------------------
def _aggregate_tta_outputs(original: Any, flipped: Any) -> Any:
    """
    Aggregate original and horizontally flipped model outputs.

    Spatial tensors (dim >= 3) are flipped back before averaging.
    Lists, tuples and dictionaries are processed recursively.
    Non-tensor objects fall back to the original output.
    """
    if isinstance(original, torch.Tensor) and isinstance(flipped, torch.Tensor):
        if original.shape != flipped.shape:
            raise ValueError(
                f"TTA output shapes differ: {tuple(original.shape)} vs "
                f"{tuple(flipped.shape)}"
            )
        flipped_aligned = torch.flip(flipped, dims=[-1]) if original.dim() >= 3 else flipped
        return (original + flipped_aligned) / 2.0

    if isinstance(original, tuple) and isinstance(flipped, tuple):
        if len(original) != len(flipped):
            raise ValueError("TTA tuple outputs have different lengths")
        return tuple(
            _aggregate_tta_outputs(a, b) for a, b in zip(original, flipped)
        )

    if isinstance(original, list) and isinstance(flipped, list):
        if len(original) != len(flipped):
            raise ValueError("TTA list outputs have different lengths")
        return [
            _aggregate_tta_outputs(a, b) for a, b in zip(original, flipped)
        ]

    if isinstance(original, dict) and isinstance(flipped, dict):
        if original.keys() != flipped.keys():
            raise ValueError("TTA dictionary outputs have different keys")
        return {
            key: _aggregate_tta_outputs(original[key], flipped[key])
            for key in original
        }

    return original


class TTAWrapper(nn.Module):
    """Horizontal-flip test-time augmentation."""

    def __init__(self, base_model: nn.Module, enabled: bool = True) -> None:
        super().__init__()
        self.base_model = base_model
        self.enabled = enabled

    def forward(self, x: torch.Tensor):
        if not self.enabled:
            return self.base_model(x)

        output_original = self.base_model(x)
        x_flipped = torch.flip(x, dims=[-1])
        output_flipped = self.base_model(x_flipped)
        return _aggregate_tta_outputs(output_original, output_flipped)


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------
def load_model(checkpoint_path: str) -> nn.Module:
    encoder_name = "dinov2reg_vit_base_14"
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    encoder = vit_encoder.load(encoder_name)

    # small/base/large describe the DINOv2 backbone size, not corruption severity.
    if "small" in encoder_name:
        embed_dim, num_heads = 384, 6
    elif "base" in encoder_name:
        embed_dim, num_heads = 768, 12
    elif "large" in encoder_name:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise ValueError(f"Unsupported encoder architecture: {encoder_name}")

    bottleneck = nn.ModuleList([
        bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2)
    ])

    decoder_blocks = []
    for _ in range(8):
        decoder_blocks.append(
            VitBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-8),
                attn=LinearAttention2,
            )
        )
    decoder = nn.ModuleList(decoder_blocks)

    model = ViTill(
        encoder=encoder,
        bottleneck=bottleneck,
        decoder=decoder,
        target_layers=target_layers,
        mask_neighbor_size=0,
        fuse_layer_encoder=fuse_layer_encoder,
        fuse_layer_decoder=fuse_layer_decoder,
    )

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Also tolerate checkpoints wrapped as {'model': state_dict} or
    # {'state_dict': state_dict}.
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    model.load_state_dict(checkpoint, strict=True)
    model = model.to(DEVICE)
    model.eval()
    return model


# -----------------------------------------------------------------------------
# Persistent result handling
# -----------------------------------------------------------------------------
def empty_results() -> Dict[str, Any]:
    results: Dict[str, Any] = {
        "metadata": {},
        "clean": {},
    }
    for corruption_name in CORRUPTIONS:
        results[corruption_name] = {
            str(severity): {} for severity in SEVERITIES
        }
    return results


def normalize_result_structure(results: Dict[str, Any]) -> Dict[str, Any]:
    """Add missing sections so older/partial result files remain resumable."""
    results.setdefault("metadata", {})
    results.setdefault("clean", {})
    for corruption_name in CORRUPTIONS:
        results.setdefault(corruption_name, {})
        for severity in SEVERITIES:
            results[corruption_name].setdefault(str(severity), {})
    return results


def save_results_atomic(results: Dict[str, Any], save_path: str) -> None:
    """Write to a temporary file and atomically replace the old JSON."""
    save_directory = os.path.dirname(save_path)
    if save_directory:
        os.makedirs(save_directory, exist_ok=True)

    temporary_path = save_path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.flush()
        os.fsync(file.fileno())

    os.replace(temporary_path, save_path)


def load_saved_results(save_path: str) -> Dict[str, Any]:
    if not os.path.exists(save_path):
        return empty_results()

    try:
        with open(save_path, "r", encoding="utf-8") as file:
            results = json.load(file)
        print(f"[Resume] Loaded saved results: {save_path}")
        return normalize_result_structure(results)
    except (OSError, json.JSONDecodeError) as error:
        backup_path = save_path + ".broken"
        try:
            os.replace(save_path, backup_path)
            print(f"[Warning] Broken JSON moved to: {backup_path}")
        except OSError:
            pass
        print(f"[Warning] Could not read previous results: {error}")
        return empty_results()


def metric_to_float(metric: Any) -> float:
    if isinstance(metric, torch.Tensor):
        metric = metric.detach().cpu().item()
    value = float(metric)
    if not np.isfinite(value):
        raise ValueError(f"Metric is not finite: {value}")
    return value


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
def create_loader(dataset: Dataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )


def evaluate_one_dataset(
    eval_model: nn.Module,
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
) -> float:
    loader = create_loader(dataset, batch_size=batch_size, num_workers=num_workers)

    with torch.inference_mode():
        metrics = evaluation_batch(
            eval_model,
            loader,
            DEVICE,
            max_ratio=0.01,
            resize_mask=256,
        )

    return metric_to_float(metrics[0])


def cleanup_after_experiment() -> None:
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


def run_experiments(
    model: nn.Module,
    categories: List[str],
    data_root: str,
    use_tta: bool,
    save_path: str,
    batch_size: int = 16,
    num_workers: int = 4,
) -> Dict[str, Any]:
    """
    Run clean and corrupted evaluations.

    Each completed category is saved immediately. When restarted, completed
    entries are skipped and execution continues at the first missing entry.
    """
    eval_model: nn.Module = TTAWrapper(model, enabled=True) if use_tta else model
    eval_model = eval_model.to(DEVICE)
    eval_model.eval()

    data_transform, gt_transform = get_data_transforms(392, 392)
    results = load_saved_results(save_path)

    results["metadata"].update({
        "use_tta": use_tta,
        "device": str(DEVICE),
        "batch_size": batch_size,
        "num_workers": num_workers,
        "categories": categories,
        "severities": SEVERITIES,
        "corruptions": list(CORRUPTIONS.keys()),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    save_results_atomic(results, save_path)

    # 1. Clean baseline: original MVTec-AD test images, no corruption.
    for category in categories:
        if category in results["clean"]:
            print(
                f"[Skip] clean | {category} | "
                f"I-AUROC={results['clean'][category]:.4f}"
            )
            continue

        print(f"\n[Start] clean | category={category}", flush=True)
        try:
            dataset = MVTecDataset(
                root=os.path.join(data_root, category),
                transform=data_transform,
                gt_transform=gt_transform,
                phase="test",
            )
            auroc = evaluate_one_dataset(
                eval_model, dataset, batch_size=batch_size, num_workers=num_workers
            )
            results["clean"][category] = auroc
            results["metadata"]["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            save_results_atomic(results, save_path)
            print(f"[Saved] clean | {category} | I-AUROC={auroc:.4f}", flush=True)
        except BaseException:
            # Save all previously completed results even for KeyboardInterrupt,
            # scheduler termination signals converted into exceptions, etc.
            save_results_atomic(results, save_path)
            raise
        finally:
            cleanup_after_experiment()

    # 2. Corrupted test sets: 2 corruptions x 5 severity levels.
    for corruption_name, corruption_fn in CORRUPTIONS.items():
        for severity in SEVERITIES:
            severity_key = str(severity)

            for category in categories:
                saved = results[corruption_name][severity_key]
                if category in saved:
                    print(
                        f"[Skip] {corruption_name} | severity={severity} | "
                        f"{category} | I-AUROC={saved[category]:.4f}"
                    )
                    continue

                print(
                    f"\n[Start] {corruption_name} | severity={severity} | "
                    f"category={category}",
                    flush=True,
                )

                try:
                    base_dataset = MVTecDataset(
                        root=os.path.join(data_root, category),
                        transform=data_transform,
                        gt_transform=gt_transform,
                        phase="test",
                    )
                    corrupted_dataset = CorruptedWrapper(
                        base_dataset=base_dataset,
                        corruption_fn=corruption_fn,
                        severity=severity,
                    )
                    auroc = evaluate_one_dataset(
                        eval_model,
                        corrupted_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                    )
                    results[corruption_name][severity_key][category] = auroc
                    results["metadata"]["updated_at"] = time.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    save_results_atomic(results, save_path)
                    print(
                        f"[Saved] {corruption_name} | severity={severity} | "
                        f"{category} | I-AUROC={auroc:.4f}",
                        flush=True,
                    )
                except BaseException:
                    save_results_atomic(results, save_path)
                    raise
                finally:
                    cleanup_after_experiment()

    print(f"\n[Done] All results saved to: {save_path}")
    return results


# -----------------------------------------------------------------------------
# Analysis and plotting
# -----------------------------------------------------------------------------
def get_mean_result(
    results: Dict[str, Any],
    experiment_name: str,
    severity: Optional[int] = None,
) -> float:
    """Return the mean I-AUROC across currently completed categories."""
    if experiment_name == "clean":
        values = list(results["clean"].values())
    else:
        if severity is None:
            raise ValueError("severity is required for a corruption experiment")
        values = list(results[experiment_name][str(severity)].values())

    if not values:
        return float("nan")
    return float(np.mean(values))


def print_summary(results: Dict[str, Any], title: str) -> None:
    clean_mean = get_mean_result(results, "clean")
    clean_count = len(results["clean"])
    print(f"\n=== {title} summary ===")
    print(f"Clean: mean I-AUROC={clean_mean:.4f}, completed={clean_count}/{len(CATEGORIES)}")

    for corruption_name in CORRUPTIONS:
        print(f"{corruption_name}:")
        for severity in SEVERITIES:
            values = results[corruption_name][str(severity)]
            mean_value = get_mean_result(results, corruption_name, severity)
            if np.isnan(mean_value) or np.isnan(clean_mean) or clean_mean == 0:
                relative_drop = float("nan")
            else:
                relative_drop = (clean_mean - mean_value) / clean_mean * 100.0
            print(
                f"  severity {severity}: mean={mean_value:.4f}, "
                f"relative_drop={relative_drop:.2f}%, "
                f"completed={len(values)}/{len(CATEGORIES)}"
            )


def plot_comparison(
    results_standard: Dict[str, Any],
    results_tta: Dict[str, Any],
    save_path: str,
) -> None:
    clean_mean_standard = get_mean_result(results_standard, "clean")
    clean_mean_tta = get_mean_result(results_tta, "clean")

    plt.figure(figsize=(9, 6))

    for corruption_name in CORRUPTIONS:
        standard_means = [
            get_mean_result(results_standard, corruption_name, severity)
            for severity in SEVERITIES
        ]
        plt.plot(
            [0] + SEVERITIES,
            [clean_mean_standard] + standard_means,
            marker="o",
            label=f"{corruption_name} (Standard)",
        )

    for corruption_name in CORRUPTIONS:
        tta_means = [
            get_mean_result(results_tta, corruption_name, severity)
            for severity in SEVERITIES
        ]
        plt.plot(
            [0] + SEVERITIES,
            [clean_mean_tta] + tta_means,
            marker="^",
            linestyle="--",
            label=f"{corruption_name} (+TTA)",
        )

    plt.xlabel("Severity (0 = clean)")
    plt.ylabel("Mean image-level AUROC")
    plt.title("Dinomaly Robustness and TTA under Domain Shifts")
    plt.xticks([0] + SEVERITIES)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    save_directory = os.path.dirname(save_path)
    if save_directory:
        os.makedirs(save_directory, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[Saved] Comparison plot: {save_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    checkpoint_path = "checkpoints/dinomaly_mvtec.pth"
    data_root = os.path.expanduser("~/workspace/mvtec_anomaly_detection")
    output_dir = "experiment_results"

    # Reduce batch_size to 8 or 4 if CUDA out-of-memory occurs.
    batch_size = 16
    num_workers = 4

    print("=== Environment ===")
    print(f"Host: {os.uname().nodename}")
    print(f"Device: {DEVICE}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(
            "WARNING: CUDA is unavailable. The evaluation will run on CPU. "
            "On FAU HPC, request a GPU compute node before starting this script."
        )

    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"MVTec-AD root not found: {data_root}")

    os.makedirs(output_dir, exist_ok=True)
    model = load_model(checkpoint_path)

    print("\n=== 1. Standard Dinomaly (no TTA) ===")
    results_standard = run_experiments(
        model=model,
        categories=CATEGORIES,
        data_root=data_root,
        use_tta=False,
        save_path=os.path.join(output_dir, "results_standard.json"),
        batch_size=batch_size,
        num_workers=num_workers,
    )
    print_summary(results_standard, "Standard")

    print("\n=== 2. Dinomaly + horizontal-flip TTA ===")
    results_tta = run_experiments(
        model=model,
        categories=CATEGORIES,
        data_root=data_root,
        use_tta=True,
        save_path=os.path.join(output_dir, "results_tta.json"),
        batch_size=batch_size,
        num_workers=num_workers,
    )
    print_summary(results_tta, "TTA")

    plot_comparison(
        results_standard=results_standard,
        results_tta=results_tta,
        save_path=os.path.join(output_dir, "robustness_tta_comparison.png"),
    )


if __name__ == "__main__":
    main()