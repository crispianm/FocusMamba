# Copyright (2025) Bytedance Ltd. and/or its affiliates 

# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 

#     http://www.apache.org/licenses/LICENSE-2.0 

# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 
import torch
import torch.nn.functional as F
import torch.nn as nn
from .dpt import DPTHead
from .motion_module.motion_module import TemporalModule
# easydict removed — use plain dict instead
# from easydict import EasyDict


class StageResidualCacheGate(nn.Module):
    """Predict a stage-wise channel gate from current-vs-cached feature residuals."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(int(channels) // max(int(reduction), 1), 8)
        self.mlp = nn.Sequential(
            nn.Linear(int(channels) * 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, int(channels)),
            nn.Sigmoid(),
        )

    def forward(self, current_summary: torch.Tensor, cached_summary: torch.Tensor) -> torch.Tensor:
        residual = (current_summary - cached_summary).abs()
        gate_input = torch.cat([current_summary, cached_summary, residual], dim=-1)
        return self.mlp(gate_input)


class DPTHeadTemporal(DPTHead):
    def __init__(self, 
        in_channels, 
        features=256, 
        use_bn=False, 
        out_channels=[256, 512, 1024, 1024], 
        use_clstoken=False,
        num_frames=32,
        pe='ape',
        patch_size: int = 14,
        output_activation: str = "relu",
        state_gate_enabled: bool = False,
        state_gate_reduction: int = 8,
        state_gate_stage_mask=None,
    ):
        super().__init__(
            in_channels,
            features,
            use_bn,
            out_channels,
            use_clstoken,
            patch_size=patch_size,
            output_activation=output_activation,
        )

        assert num_frames > 0
        motion_module_kwargs = dict(num_attention_heads                = 8,
                                        num_transformer_block              = 1,
                                        num_attention_blocks               = 2,
                                        temporal_max_len                   = num_frames,
                                        zero_initialize                    = True,
                                        pos_embedding_type                 = pe)

        self.motion_modules = nn.ModuleList([
            TemporalModule(in_channels=out_channels[2], 
                           **motion_module_kwargs),
            TemporalModule(in_channels=out_channels[3],
                           **motion_module_kwargs),
            TemporalModule(in_channels=features,
                           **motion_module_kwargs),
            TemporalModule(in_channels=features,
                           **motion_module_kwargs)
        ])
        self.state_gate_enabled = bool(state_gate_enabled)
        if state_gate_stage_mask is None:
            self.state_gate_stage_mask = [True, True, True, True]
        else:
            if len(state_gate_stage_mask) != 4:
                raise ValueError(
                    "state_gate_stage_mask must have four booleans, "
                    f"got {state_gate_stage_mask!r}"
                )
            self.state_gate_stage_mask = [bool(v) for v in state_gate_stage_mask]
        self.stage_gates = nn.ModuleList([
            StageResidualCacheGate(out_channels[2], reduction=state_gate_reduction),
            StageResidualCacheGate(out_channels[3], reduction=state_gate_reduction),
            StageResidualCacheGate(features, reduction=state_gate_reduction),
            StageResidualCacheGate(features, reduction=state_gate_reduction),
        ]) if self.state_gate_enabled else None

    def _gate_stage_cache(self, current_stage, cached_stage_list, stage_idx):
        batch_size = current_stage.shape[0]
        device = current_stage.device
        dtype = current_stage.dtype
        default_gate = torch.ones(batch_size, device=device, dtype=dtype)

        if (
            not self.state_gate_enabled
            or not self.state_gate_stage_mask[stage_idx]
            or not cached_stage_list
        ):
            return cached_stage_list, default_gate

        channels = current_stage.shape[1]
        current_summary = current_stage.mean(dim=(2, 3, 4))
        cached_summary_parts = []
        reshaped_cache = []

        for tensor in cached_stage_list:
            if tensor.ndim != 3:
                raise ValueError(f"Expected cached temporal tensor with ndim=3, got {tensor.ndim}")
            if tensor.shape[-1] != channels:
                raise ValueError(
                    f"Cached temporal tensor has channel dim {tensor.shape[-1]}, expected {channels}"
                )
            if tensor.shape[0] % batch_size != 0:
                raise ValueError(
                    f"Cached temporal tensor leading dim {tensor.shape[0]} is not divisible by batch={batch_size}"
                )
            spatial_tokens = tensor.shape[0] // batch_size
            view = tensor.reshape(batch_size, spatial_tokens, tensor.shape[1], channels)
            reshaped_cache.append((tensor.shape, view))
            cached_summary_parts.append(view.mean(dim=(1, 2)))

        cached_summary = torch.stack(cached_summary_parts, dim=0).mean(dim=0)
        gate = self.stage_gates[stage_idx](current_summary, cached_summary).to(dtype=dtype)

        gated_cache = []
        for original_shape, view in reshaped_cache:
            gated_view = view * gate[:, None, None, :]
            gated_cache.append(gated_view.reshape(original_shape))
        return gated_cache, gate.mean(dim=1)

    def forward(
        self,
        out_features,
        patch_h,
        patch_w,
        frame_length,
        micro_batch_size=4,
        cached_hidden_state_list=None,
        return_gate_stats: bool = False,
    ):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w)).contiguous()

            B, T = x.shape[0] // frame_length, frame_length
            x = self.projects[i](x)
            x = self.resize_layers[i](x)

            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        B, T = layer_1.shape[0] // frame_length, frame_length
        if cached_hidden_state_list is not None:
            N = len(cached_hidden_state_list) // len(self.motion_modules)
        else:
            N = 0
        gate_means = []

        layer_3_seq = layer_3.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4)
        layer_3_cache, gate_mean = self._gate_stage_cache(layer_3_seq, list(cached_hidden_state_list[0:N]) if N else None, 0)
        gate_means.append(gate_mean)
        layer_3, h0 = self.motion_modules[0](layer_3_seq, None, None, layer_3_cache)
        layer_3 = layer_3.permute(0, 2, 1, 3, 4).flatten(0, 1)
        layer_4_seq = layer_4.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4)
        layer_4_cache, gate_mean = self._gate_stage_cache(layer_4_seq, list(cached_hidden_state_list[N:2*N]) if N else None, 1)
        gate_means.append(gate_mean)
        layer_4, h1 = self.motion_modules[1](layer_4_seq, None, None, layer_4_cache)
        layer_4 = layer_4.permute(0, 2, 1, 3, 4).flatten(0, 1)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_4_seq = path_4.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4)
        path_4_cache, gate_mean = self._gate_stage_cache(path_4_seq, list(cached_hidden_state_list[2*N:3*N]) if N else None, 2)
        gate_means.append(gate_mean)
        path_4, h2 = self.motion_modules[2](path_4_seq, None, None, path_4_cache)
        path_4 = path_4.permute(0, 2, 1, 3, 4).flatten(0, 1)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_3_seq = path_3.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4)
        path_3_cache, gate_mean = self._gate_stage_cache(path_3_seq, list(cached_hidden_state_list[3*N:]) if N else None, 3)
        gate_means.append(gate_mean)
        path_3, h3 = self.motion_modules[3](path_3_seq, None, None, path_3_cache)
        path_3 = path_3.permute(0, 2, 1, 3, 4).flatten(0, 1)

        batch_size = layer_1_rn.shape[0]
        if batch_size <= micro_batch_size or batch_size % micro_batch_size != 0:
            path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
            path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

            out = self.scratch.output_conv1(path_1)
            out = F.interpolate(
                out, (int(patch_h * self.patch_size), int(patch_w * self.patch_size)),
                mode="bilinear", align_corners=True
            )
            ori_type = out.dtype
            with torch.autocast(device_type="cuda", enabled=False):
                out = self.scratch.output_conv2(out.float())

            output = self._apply_output_activation(out.to(ori_type))
        else:
            ret = []
            for i in range(0, batch_size, micro_batch_size):
                path_2 = self.scratch.refinenet2(path_3[i:i + micro_batch_size], layer_2_rn[i:i + micro_batch_size], size=layer_1_rn[i:i + micro_batch_size].shape[2:])
                path_1 = self.scratch.refinenet1(path_2, layer_1_rn[i:i + micro_batch_size])
                out = self.scratch.output_conv1(path_1)
                out = F.interpolate(
                    out, (int(patch_h * self.patch_size), int(patch_w * self.patch_size)),
                    mode="bilinear", align_corners=True
                )
                ori_type = out.dtype
                with torch.autocast(device_type="cuda", enabled=False):
                    out = self.scratch.output_conv2(out.float())
                ret.append(self._apply_output_activation(out.to(ori_type)))
            output = torch.cat(ret, dim=0)

        if return_gate_stats:
            return output, h0 + h1 + h2 + h3, torch.stack(gate_means, dim=1)
        return output, h0 + h1 + h2 + h3
