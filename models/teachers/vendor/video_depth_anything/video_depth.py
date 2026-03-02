# Copyright (2025) Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
# Vendored from: https://github.com/DepthAnything/Video-Depth-Anything
#
# Modified: removed infer_video_depth() and its numpy/cv2/tqdm dependencies.
# Only forward() is needed for distillation training.

import torch
import torch.nn.functional as F
import torch.nn as nn

from .dinov2 import DINOv2
from .dpt_temporal import DPTHeadTemporal


class VideoDepthAnything(nn.Module):
    def __init__(
        self,
        encoder='vitl',
        features=256,
        out_channels=[256, 512, 1024, 1024],
        use_bn=False,
        use_clstoken=False,
        num_frames=32,
        pe='ape',
        metric=False,
    ):
        super(VideoDepthAnything, self).__init__()

        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            "vitb": [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23]
        }

        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder)

        self.head = DPTHeadTemporal(
            self.pretrained.embed_dim, features, use_bn,
            out_channels=out_channels, use_clstoken=use_clstoken,
            num_frames=num_frames, pe=pe,
        )
        self.metric = metric

    def forward(self, x):
        """Forward pass for video depth estimation.

        Args:
            x: (B, T, C, H, W) input video clip, float32,
               ImageNet-normalised (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]).
               H and W must be divisible by 14 (DINOv2 patch size).

        Returns:
            depth: (B, T, H, W) — metric depth (ReLU-activated, non-negative).
        """
        B, T, C, H, W = x.shape
        patch_h, patch_w = H // 14, W // 14
        features = self.pretrained.get_intermediate_layers(
            x.flatten(0, 1),
            self.intermediate_layer_idx[self.encoder],
            return_class_token=True,
        )
        depth = self.head(features, patch_h, patch_w, T)[0]
        depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=True)
        depth = F.relu(depth)
        return depth.squeeze(1).unflatten(0, (B, T))  # (B, T, H, W)
