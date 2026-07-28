"""
Dinomaly robustness evaluation - Domain-Specific TTA (Optimized Runner)
- Reads existing 'results_standard.json' (skips standard runs completely)
- Copies Clean baseline from standard results (skips clean evaluation)
- Applies CLAHE TTA for Low Light & Scale-Crop TTA for Gaussian Blur
- Saves TTA results to 'results_tta_domain.json'
- Automatically generates combined comparison plot

Run this script on a SLURM GPU compute node.
"""

import gc
import json
import os
import time
from functools import partial
from typing import Any, Callable, Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
# Dataset wrapper
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
# Domain-Specific TTA Wrappers
# -----------------------------------------------------------------------------
def _average_outputs(out1: Any, out2: Any) -> Any:
    """递归平均两个模型的预测输出"""
    if isinstance(out1, torch.Tensor) and isinstance(out2, torch.Tensor):
        return (out1 + out2) / 2.0
    if isinstance(out1, (tuple, list)) and isinstance(out2, (tuple, list)):
        return type(out1)(_average_outputs(a, b) for a, b in zip(out1, out2))
    if isinstance(out1, dict) and isinstance(out2, dict):
        return {k: _average_outputs(out1[k], out2[k]) for k in out1}
    return out1


class DomainSpecificTTAWrapper(nn.Module):
    """
    针对工业无监督异常检测优化的 TTA 模块：
    - 废除任何破坏图像几何结构的水平/垂直翻转
    - 针对 low_light: 采用基于 LAB 空间的 CLAHE 直方图自适应增强
    - 针对 gaussian_blur: 采用中心多尺度 Crop & Scale 增强
    """

    def __init__(
        self,
        base_model: nn.Module,
        enabled: bool = True,
        mode: str = "none",
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.enabled = enabled
        self.mode = mode

    def set_mode(self, mode: str) -> None:
        self.mode = mode

    def _apply_clahe_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """对输入 Batch Tensor 执行直方图均衡化（CLAHE）增强"""
        device = x.device
        std = STD.to(device)
        mean = MEAN.to(device)

        # 反归一化到 [0, 255] uint8
        unnorm = torch.clamp(x * std + mean, 0.0, 1.0) * 255.0
        imgs_np = unnorm.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_list = []

        for img in imgs_np:
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
            enhanced_list.append(enhanced)

        enhanced_np = np.stack(enhanced_list, axis=0).astype(np.float32) / 255.0
        enhanced_tensor = torch.from_numpy(enhanced_np).permute(0, 3, 1, 2).to(device)
        return (enhanced_tensor - mean) / std

    def _apply_crop_scale_tensor(self, x: torch.Tensor, scale_ratio: float = 0.88) -> torch.Tensor:
        """多尺度中心裁剪缩放 TTA"""
        _, _, h, w = x.shape
        crop_h, crop_w = int(h * scale_ratio), int(w * scale_ratio)
        top = (h - crop_h) // 2
        left = (w - crop_w) // 2

        cropped = x[:, :, top : top + crop_h, left : left + crop_w]
        resized = F.interpolate(cropped, size=(h, w), mode="bilinear", align_corners=False)
        return resized

    def forward(self, x: torch.Tensor):
        if not self.enabled or self.mode == "none":
            return self.base_model(x)

        # 原始图像正向推理
        out_orig = self.base_model(x)

        if self.mode == "low_light":
            # 低光照 TTA 分支
            x_enhanced = self._apply_clahe_tensor(x)
            out_aug = self.base_model(x_enhanced)
            return _average_outputs(out_orig, out_aug)

        elif self.mode == "gaussian_blur":
            # 高斯模糊 TTA 分支
            x_scaled = self._apply_crop_scale_tensor(x, scale_ratio=0.88)
            out_aug = self.base_model(x_scaled)
            return _average_outputs(out_orig, out_aug)

        return out_orig


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------
def load_model(checkpoint_path: str) -> nn.Module:
    encoder_name = "dinov2reg_vit_base_14"
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    encoder = vit_encoder.load(encoder_name)

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
    results.setdefault("metadata", {})
    results.setdefault("clean", {})
    for corruption_name in CORRUPTIONS:
        results.setdefault(corruption_name, {})
        for severity in SEVERITIES:
            results[corruption_name].setdefault(str(severity), {})
    return results


def save_results_atomic(results: Dict[str, Any], save_path: str) -> None:
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
        print(f"[Warning] Could not read previous results from {save_path}: {error}")
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


def run_tta_experiments(
    model: nn.Module,
    categories: List[str],
    data_root: str,
    save_path: str,
    results_standard: Dict[str, Any],
    batch_size: int = 16,
    num_workers: int = 4,
) -> Dict[str, Any]:

    tta_wrapper = DomainSpecificTTAWrapper(model, enabled=True)
    eval_model = tta_wrapper.to(DEVICE)
    eval_model.eval()

    data_transform, gt_transform = get_data_transforms(392, 392)
    results = load_saved_results(save_path)

    results["metadata"].update({
        "use_tta": True,
        "tta_type": "DomainSpecificTTA (CLAHE + ScaleCrop)",
        "device": str(DEVICE),
        "batch_size": batch_size,
        "num_workers": num_workers,
        "categories": categories,
        "severities": SEVERITIES,
        "corruptions": list(CORRUPTIONS.keys()),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    # 1. 直接复用 Standard 跑好的 Clean Baseline，避免重复计算
    if "clean" in results_standard and results_standard["clean"]:
        print("\n[Skip Clean] 直接复用 Standard 模式下的 Clean 结果...")
        results["clean"] = results_standard["clean"].copy()
        save_results_atomic(results, save_path)
    else:
        # 备用：如果没有已有的 Clean，才重新计算
        tta_wrapper.set_mode("none")
        for category in categories:
            if category in results["clean"]:
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
                save_results_atomic(results, save_path)
            finally:
                cleanup_after_experiment()

    # 2. 运行带 TTA 的 Corrupted 测试集
    for corruption_name, corruption_fn in CORRUPTIONS.items():
        # 根据当前腐蚀类型动态切换 TTA 策略
        tta_wrapper.set_mode(corruption_name)

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
                    f"category={category} | TTA_Mode={tta_wrapper.mode}",
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
                    results["metadata"]["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
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

    print(f"\n[Done] All TTA results saved to: {save_path}")
    return results


# -----------------------------------------------------------------------------
# Analysis and plotting
# -----------------------------------------------------------------------------
def get_mean_result(
    results: Dict[str, Any],
    experiment_name: str,
    severity: Optional[int] = None,
) -> float:
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
            label=f"{corruption_name} (+DomainTTA)",
        )

    plt.xlabel("Severity (0 = clean)")
    plt.ylabel("Mean image-level AUROC")
    plt.title("Dinomaly Robustness with Domain-Specific TTA")
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

    batch_size = 16
    num_workers = 4

    standard_json_path = os.path.join(output_dir, "results_standard.json")
    tta_json_path = os.path.join(output_dir, "results_tta_domain.json")

    print("=== Environment ===")
    print(f"Host: {os.uname().nodename}")
    print(f"Device: {DEVICE}")

    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"MVTec-AD root not found: {data_root}")

    # 1. 读取已经跑完的 Standard 模式结果
    if not os.path.exists(standard_json_path):
        raise FileNotFoundError(
            f"未找到 Standard 结果文件: {standard_json_path}！"
            "请确保 results_standard.json 存在于 experiment_results 目录下。"
        )

    print(f"\n[Read] 成功读取已有 Standard 模式结果: {standard_json_path}")
    results_standard = load_saved_results(standard_json_path)
    print_summary(results_standard, "Standard (Loaded)")

    # 2. 跑带 Domain-Specific TTA 的实验
    print("\n=== 开始评估 Domain-Specific TTA ===")
    model = load_model(checkpoint_path)
    results_tta = run_tta_experiments(
        model=model,
        categories=CATEGORIES,
        data_root=data_root,
        save_path=tta_json_path,
        results_standard=results_standard,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    print_summary(results_tta, "Domain TTA")

    # 3. 读取新旧两份 JSON 画对比图
    plot_comparison(
        results_standard=results_standard,
        results_tta=results_tta,
        save_path=os.path.join(output_dir, "robustness_domain_tta_comparison.png"),
    )


if __name__ == "__main__":
    main()