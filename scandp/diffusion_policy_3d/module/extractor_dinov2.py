"""DINOv2 Feature Extractor with Full Torch Tensor Input Support (No PIL)"""

import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F


class DINOv2FeatureExtractor:
    """DINOv2特徴抽出とポイントクラウド生成のためのクラス (Torch Tensor対応版)"""

    def __init__(
        self,
        model_name: str = "dinov2_vitl14",
        device: Optional[torch.device] = None,
        resize: int = 256,
        crop: int = 224,
        batch_size: int = 4
    ):
        self.logger = logging.getLogger(self.__class__.__name__)

        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.batch_size = batch_size
        self.crop = crop
        self.resize = resize

        self.logger.info(f"Using device: {self.device}")

        # モデルのロード
        t0 = time.time()
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model.eval()
        self.model.to(self.device)
        self.logger.info(f"Model loaded in {time.time() - t0:.2f}s")

        # Patch情報
        self.patch_size = getattr(self.model, "patch_size", 14)
        self.patch_h = crop // self.patch_size
        self.patch_w = self.patch_h

    def preprocess_tensor(
        self, image_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        入力画像 (H, W, C) → (C, crop, crop) へ変換
        """
        if image_tensor.ndim != 3 or image_tensor.shape[-1] != 3:
            raise ValueError("Input image must have shape [H, W, 3].")

        # [H, W, C] → [C, H, W]
        img = image_tensor.permute(2, 0, 1).float()

        # Resize
        img = F.interpolate(
            img.unsqueeze(0),
            size=(self.resize, self.resize),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        # Center crop
        _, h, w = img.shape
        top = (h - self.crop) // 2
        left = (w - self.crop) // 2
        img = img[:, top:top + self.crop, left:left + self.crop]

        # Normalize to [0,1] if necessary
        if img.max() > 1.0:
            img = img / 255.0

        return img

    def extract_features_batch(
        self,
        images: torch.Tensor
    ) -> Tuple[torch.Tensor, int, int]:
        """
        複数画像からバッチで特徴を抽出
        Args:
            images: torch.Tensor of shape [N, H, W, C]
        Returns:
            features: (N, num_patches, feat_dim)
            patch_h, patch_w
        """
        if not isinstance(images, torch.Tensor):
            raise TypeError("Input images must be a torch.Tensor")
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError("Expected images of shape [N, H, W, 3]")

        all_features = []

        for i in range(0, len(images), self.batch_size):
            batch = images[i:i + self.batch_size]
            batch_preprocessed = torch.stack(
                [self.preprocess_tensor(img) for img in batch]
            ).to(self.device)

            with torch.no_grad():
                out = self.model.forward_features(batch_preprocessed)
                feats = out["x_norm_patchtokens"]  # (B, num_patches, feat_dim)
            all_features.append(feats)

        all_features_tensor = torch.cat(all_features, dim=0)
        return all_features_tensor, self.patch_h, self.patch_w

    @staticmethod
    def torch_pca_gpu(features: torch.Tensor, n_components: int = 3) -> torch.Tensor:
        """GPU上でPCAを実行"""
        mean = features.mean(dim=0, keepdim=True)
        centered = features - mean
        cov = (centered.T @ centered) / (features.shape[0] - 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        idx = torch.argsort(eigenvalues, descending=True)
        components = eigenvectors[:, idx][:, :n_components]
        pca_features = centered @ components
        return pca_features

    def pca_segment_gpu(
        self,
        features: torch.Tensor,
        patch_h: int,
        patch_w: int,
        n_components: int = 3,
        bg_thresh: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """PCAによるセグメンテーションとRGB可視化"""
        pca_feats = self.torch_pca_gpu(features, n_components=n_components)
        comp0 = pca_feats[:, 0]
        comp0_norm = (comp0 - comp0.min()) / \
            (comp0.max() - comp0.min() + 1e-12)
        bg_mask = comp0_norm > bg_thresh

        bg_mask = bg_mask.reshape(patch_h, patch_w)

        if not bg_mask.all():
            bg_mask_flat = bg_mask.reshape(-1)
            fg_feats = features[~bg_mask_flat]
            fg_pca = self.torch_pca_gpu(fg_feats, n_components=n_components)
            for i in range(n_components):
                col = fg_pca[:, i]
                fg_pca[:, i] = (col - col.min()) / \
                    (col.max() - col.min() + 1e-12)
            pca_rgb = torch.zeros_like(pca_feats)
            pca_rgb[~bg_mask_flat] = fg_pca
        else:
            pca_rgb = pca_feats

        pca_rgb = pca_rgb.reshape(patch_h, patch_w, n_components)
        if n_components < 3:
            pad = 3 - n_components
            pca_rgb = F.pad(pca_rgb.permute(2, 0, 1),
                            (0, 0, 0, 0, 0, pad)).permute(1, 2, 0)

        # Reshape pca_feats to the same spatial/channel format as pca_rgb
        try:
            pca_feats_img = pca_feats.reshape(patch_h, patch_w, n_components)
        except RuntimeError:
            # If reshape fails, fall back to .view (shouldn't happen)
            pca_feats_img = pca_feats.view(patch_h, patch_w, n_components)

        if n_components < 3:
            pad = 3 - n_components
            pca_feats_img = F.pad(pca_feats_img.permute(2, 0, 1),
                                  (0, 0, 0, 0, 0, pad)).permute(1, 2, 0)

        return bg_mask, pca_rgb, pca_feats_img


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torchvision.io as io

    # Torch形式の画像読み込み ([H, W, C])
    img_path = "/home/user/workspace/test/bunny.png"  # 任意の画像ファイルパス
    img_tensor = io.read_image(img_path).permute(1, 2, 0)  # [H, W, C]

    extractor = DINOv2FeatureExtractor(
        model_name="dinov2_vitl14", batch_size=1)

    feats, ph, pw = extractor.extract_features_batch(img_tensor.unsqueeze(0))
    print(f"Features shape: {feats.shape}, patch grid: {ph}x{pw}")

    bg_mask, pca_rgb, _ = extractor.pca_segment_gpu(
        feats[0].reshape(-1, feats.shape[-1]), ph, pw
    )
    print(bg_mask.shape, pca_rgb.shape)

    plt.subplot(1, 3, 1)
    plt.imshow(img_tensor.cpu())
    plt.subplot(1, 3, 2)
    plt.imshow(bg_mask.cpu())
    plt.subplot(1, 3, 3)
    plt.imshow(pca_rgb.cpu())
    plt.show()
