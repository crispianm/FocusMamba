"""
Real-time webcam / video depth demo for streaming checkpoints.

Examples:
    python -m inference.demo \
        --checkpoint checkpoints/focused_redirection_20260310/04_hetero_gate_stats_align/best.pt

    python -m inference.demo \
        --checkpoint checkpoints/metric_video_depth_anything_vits.pth \
        --variant small \
        --max-side 320
"""

from __future__ import annotations

import argparse
from collections import deque
from contextlib import nullcontext
from pathlib import Path
import platform
import time
from typing import Any

import cv2
import numpy as np
import torch
import yaml

from models import build_model


WINDOW_NAME = "FocusMamba Depth Demo"


def _resolve_device(requested: str) -> torch.device:
    key = requested.strip().lower()
    if key == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if key == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested device 'mps' but torch.backends.mps.is_available() is False.")
        return torch.device("mps")
    if key == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested device 'cuda' but CUDA is not available.")
        return torch.device("cuda")
    return torch.device(key)


def _resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    key = name.strip().lower()
    if key == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    if key in {"fp16", "float16", "half"}:
        return torch.float16
    if key in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype '{name}'. Use auto, float16, or float32.")


def _extract_model_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unexpected checkpoint format: {type(checkpoint)}")
    if "model" in checkpoint and isinstance(checkpoint["model"], dict):
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        cleaned[key.removeprefix("module.")] = value
    return cleaned


def _minimal_vda_config(
    checkpoint_path: Path,
    variant: str,
    stream_max_cache_len: int,
    stream_reset_interval: int,
) -> dict[str, Any]:
    return {
        "model": {
            "type": "video_depth_anything",
            "variant": variant,
            "num_frames": max(16, stream_max_cache_len),
            "mode": "streaming_emulated",
            "checkpoint_path": str(checkpoint_path),
            "strict_checkpoint": False,
            "stream_max_cache_len": stream_max_cache_len,
            "stream_reset_interval": stream_reset_interval,
            "state_gate_enabled": False,
            "prefilter_enabled": False,
        },
        "data": {
            "train_num_frames": 8,
            "val_num_frames": 16,
        },
        "loss": {
            "training_target": "metric",
        },
    }


def _load_model(
    *,
    checkpoint_path: Path,
    config_path: Path | None,
    variant: str,
    device: torch.device,
    dtype: torch.dtype,
    stream_max_cache_len: int,
    stream_reset_interval: int,
) -> tuple[torch.nn.Module, dict[str, Any], str]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if config_path is not None:
        cfg = yaml.safe_load(config_path.read_text()) or {}
        load_mode = "config+checkpoint"
    elif isinstance(checkpoint, dict) and isinstance(checkpoint.get("config"), dict):
        cfg = checkpoint["config"]
        load_mode = "embedded-config+checkpoint"
    else:
        cfg = _minimal_vda_config(
            checkpoint_path=checkpoint_path,
            variant=variant,
            stream_max_cache_len=stream_max_cache_len,
            stream_reset_interval=stream_reset_interval,
        )
        load_mode = "raw-vda-checkpoint"

    model = build_model(cfg)
    model_cfg = cfg.setdefault("model", {})
    if hasattr(model, "default_mode"):
        model.default_mode = "streaming_emulated"
        model_cfg["mode"] = "streaming_emulated"
    if hasattr(model, "stream_max_cache_len"):
        model.stream_max_cache_len = int(stream_max_cache_len)
        model_cfg["stream_max_cache_len"] = int(stream_max_cache_len)
    if hasattr(model, "stream_reset_interval"):
        model.stream_reset_interval = int(stream_reset_interval)
        model_cfg["stream_reset_interval"] = int(stream_reset_interval)

    if load_mode != "raw-vda-checkpoint":
        state_dict = _extract_model_state_dict(checkpoint)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"[demo] missing keys during load: {len(missing_keys)}")
        if unexpected_keys:
            print(f"[demo] unexpected keys during load: {len(unexpected_keys)}")

    model = model.to(device=device)
    if dtype != torch.float32:
        model = model.to(dtype=dtype)
    model.eval()
    return model, cfg, load_mode


def _autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype == torch.float16:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _parse_source(source: str) -> tuple[int | str, int]:
    if source == "webcam":
        if platform.system() == "Darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
            return 0, cv2.CAP_AVFOUNDATION
        return 0, cv2.CAP_ANY

    path = Path(source)
    if path.exists():
        return str(path), cv2.CAP_ANY

    if source.isdigit():
        index = int(source)
        if platform.system() == "Darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
            return index, cv2.CAP_AVFOUNDATION
        return index, cv2.CAP_ANY

    return source, cv2.CAP_ANY


def _open_capture(
    source: str,
    *,
    camera_width: int,
    camera_height: int,
    camera_fps: int,
) -> cv2.VideoCapture:
    parsed_source, backend = _parse_source(source)
    cap = cv2.VideoCapture(parsed_source, backend)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source '{source}'.")

    if isinstance(parsed_source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
        cap.set(cv2.CAP_PROP_FPS, camera_fps)
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def _resize_for_model(frame_bgr: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0:
        return frame_bgr
    height, width = frame_bgr.shape[:2]
    scale = min(1.0, float(max_side) / float(max(height, width)))
    if scale >= 0.999:
        return frame_bgr
    new_width = max(32, int(round(width * scale / 2.0) * 2))
    new_height = max(32, int(round(height * scale / 2.0) * 2))
    return cv2.resize(frame_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)


def _frame_to_tensor(frame_bgr: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).contiguous()
    tensor = tensor.to(device=device, dtype=dtype).div_(255.0)
    return tensor


def _colorize_depth(
    depth_map: np.ndarray,
    *,
    display_mode: str,
    range_state: dict[str, float | None],
    ema_decay: float,
) -> np.ndarray:
    values = depth_map.astype(np.float32, copy=False)
    if display_mode == "inverse":
        values = 1.0 / np.clip(values, 1e-6, None)

    finite = np.isfinite(values)
    if not np.any(finite):
        norm = np.zeros_like(values, dtype=np.float32)
    else:
        valid = values[finite]
        lo = float(np.percentile(valid, 2.0))
        hi = float(np.percentile(valid, 98.0))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(valid.min())
            hi = float(valid.max()) + 1e-6

        if range_state["lo"] is None:
            range_state["lo"] = lo
            range_state["hi"] = hi
        else:
            range_state["lo"] = ema_decay * float(range_state["lo"]) + (1.0 - ema_decay) * lo
            range_state["hi"] = ema_decay * float(range_state["hi"]) + (1.0 - ema_decay) * hi

        denom = max(float(range_state["hi"]) - float(range_state["lo"]), 1e-6)
        norm = np.clip((values - float(range_state["lo"])) / denom, 0.0, 1.0)

    colored = cv2.applyColorMap((norm * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    return colored


def _compose_view(
    frame_bgr: np.ndarray,
    depth_bgr: np.ndarray,
    *,
    view_mode: str,
    overlay_alpha: float,
) -> np.ndarray:
    if view_mode == "depth":
        return depth_bgr
    if view_mode == "overlay":
        return cv2.addWeighted(frame_bgr, 1.0 - overlay_alpha, depth_bgr, overlay_alpha, 0.0)
    return np.hstack([frame_bgr, depth_bgr])


def _draw_hud(
    frame: np.ndarray,
    *,
    fps: float,
    infer_ms: float,
    device: torch.device,
    input_shape: tuple[int, int],
    checkpoint_label: str,
    streaming: bool,
) -> np.ndarray:
    text_lines = [
        f"{checkpoint_label}",
        f"device={device.type}  streaming={'on' if streaming else 'off'}  input={input_shape[0]}x{input_shape[1]}",
        f"fps={fps:5.1f}  infer={infer_ms:5.1f} ms",
        "keys: q/esc quit, r reset state",
    ]
    y = 24
    for line in text_lines:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22
    return frame


def _extract_depth_tensor(outputs: Any) -> torch.Tensor:
    if isinstance(outputs, dict):
        if "depth" not in outputs:
            raise KeyError("Model output dict does not contain a 'depth' entry.")
        depth = outputs["depth"]
    else:
        depth = outputs
    if not torch.is_tensor(depth):
        raise TypeError(f"Expected tensor depth output, got {type(depth)}")
    return depth


def _depth_tensor_to_numpy(depth: torch.Tensor) -> np.ndarray:
    if depth.ndim == 5:
        return depth[0, 0, 0].detach().float().cpu().numpy()
    if depth.ndim == 4:
        return depth[0, 0].detach().float().cpu().numpy()
    raise ValueError(f"Unsupported depth tensor shape: {tuple(depth.shape)}")


def run_demo(
    checkpoint_path: str,
    *,
    source: str = "webcam",
    config_path: str | None = None,
    device: str = "auto",
    dtype: str = "auto",
    max_side: int = 320,
    camera_width: int = 640,
    camera_height: int = 360,
    camera_fps: int = 30,
    stream_max_cache_len: int = 16,
    stream_reset_interval: int = 0,
    variant: str = "small",
    view_mode: str = "side-by-side",
    overlay_alpha: float = 0.45,
    mirror: bool = False,
    auto_reset_threshold: float = 0.0,
    range_ema: float = 0.85,
    display_mode: str = "inverse",
) -> None:
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    config = None if config_path is None else Path(config_path).expanduser().resolve()
    if config is not None and not config.is_file():
        raise FileNotFoundError(f"Config not found: {config}")

    torch_device = _resolve_device(device)
    run_dtype = _resolve_dtype(dtype, torch_device)

    if torch_device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    cv2.setUseOptimized(True)

    model, cfg, load_mode = _load_model(
        checkpoint_path=checkpoint,
        config_path=config,
        variant=variant,
        device=torch_device,
        dtype=run_dtype,
        stream_max_cache_len=stream_max_cache_len,
        stream_reset_interval=stream_reset_interval,
    )
    streaming = hasattr(model, "stream_step") and hasattr(model, "reset_stream_state")
    stream_state = model.reset_stream_state(batch_size=1, device=torch_device) if streaming else None

    checkpoint_label = checkpoint.parent.name if checkpoint.parent != checkpoint.parent.parent else checkpoint.stem
    print(f"[demo] load_mode={load_mode}")
    print(f"[demo] device={torch_device} dtype={run_dtype}")
    print(f"[demo] source={source} max_side={max_side} view={view_mode}")
    if "model" in cfg:
        print(f"[demo] model.type={cfg['model'].get('type')} variant={cfg['model'].get('variant')}")

    cap = _open_capture(
        source,
        camera_width=camera_width,
        camera_height=camera_height,
        camera_fps=camera_fps,
    )

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    frame_times: deque[float] = deque(maxlen=30)
    infer_times_ms: deque[float] = deque(maxlen=30)
    range_state: dict[str, float | None] = {"lo": None, "hi": None}
    prev_reset_frame: np.ndarray | None = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if mirror:
                frame = cv2.flip(frame, 1)

            if auto_reset_threshold > 0.0 and streaming:
                reset_probe = cv2.resize(frame, (96, 54), interpolation=cv2.INTER_AREA)
                if prev_reset_frame is not None:
                    diff = np.mean(np.abs(reset_probe.astype(np.float32) - prev_reset_frame.astype(np.float32))) / 255.0
                    if diff > auto_reset_threshold:
                        stream_state = model.reset_stream_state(batch_size=1, device=torch_device)
                        print(f"[demo] auto-reset stream state (diff={diff:.3f})")
                prev_reset_frame = reset_probe

            input_frame = _resize_for_model(frame, max_side=max_side)
            input_tensor = _frame_to_tensor(input_frame, device=torch_device, dtype=run_dtype)

            start = time.perf_counter()
            with torch.inference_mode():
                with _autocast_context(torch_device, run_dtype):
                    if streaming:
                        outputs = model.stream_step(input_tensor, stream_state)
                        stream_state = outputs["stream_state"]
                    else:
                        outputs = model(input_tensor.unsqueeze(2))
            depth = _extract_depth_tensor(outputs)
            depth_map = _depth_tensor_to_numpy(depth)
            infer_ms = (time.perf_counter() - start) * 1000.0

            infer_times_ms.append(infer_ms)
            frame_times.append(time.perf_counter())
            fps = 0.0
            if len(frame_times) >= 2:
                fps = (len(frame_times) - 1) / max(frame_times[-1] - frame_times[0], 1e-6)

            depth_bgr = _colorize_depth(
                depth_map,
                display_mode=display_mode,
                range_state=range_state,
                ema_decay=range_ema,
            )
            depth_bgr = cv2.resize(depth_bgr, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
            composed = _compose_view(
                frame_bgr=frame,
                depth_bgr=depth_bgr,
                view_mode=view_mode,
                overlay_alpha=overlay_alpha,
            )
            avg_infer_ms = float(np.mean(infer_times_ms)) if infer_times_ms else infer_ms
            composed = _draw_hud(
                composed,
                fps=fps,
                infer_ms=avg_infer_ms,
                device=torch_device,
                input_shape=(input_frame.shape[1], input_frame.shape[0]),
                checkpoint_label=checkpoint_label,
                streaming=streaming,
            )
            cv2.imshow(WINDOW_NAME, composed)

            key = cv2.waitKey(1) & 0xFF
            if key in {27, ord("q")}:
                break
            if key == ord("r") and streaming:
                stream_state = model.reset_stream_state(batch_size=1, device=torch_device)
                print("[demo] manual stream-state reset")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time webcam/video depth demo")
    parser.add_argument("--checkpoint", type=str, required=True, help="Training checkpoint (best.pt/latest.pt) or raw VDA checkpoint.")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config. If omitted, the script uses the checkpoint's embedded config when available.")
    parser.add_argument("--source", type=str, default="webcam", help="'webcam', camera index like '0', or a video path.")
    parser.add_argument("--device", type=str, default="auto", help="auto, mps, cuda, or cpu.")
    parser.add_argument("--dtype", type=str, default="auto", help="auto, float16, or float32.")
    parser.add_argument("--variant", type=str, default="small", help="Fallback VDA variant when loading a raw base checkpoint.")
    parser.add_argument("--max-side", type=int, default=320, help="Maximum model input side length. Lower is faster.")
    parser.add_argument("--camera-width", type=int, default=640, help="Requested webcam capture width.")
    parser.add_argument("--camera-height", type=int, default=360, help="Requested webcam capture height.")
    parser.add_argument("--camera-fps", type=int, default=30, help="Requested webcam capture FPS.")
    parser.add_argument("--stream-max-cache-len", type=int, default=16, help="Streaming cache length override for compatible models.")
    parser.add_argument("--stream-reset-interval", type=int, default=0, help="Periodic state reset interval override. 0 disables resets.")
    parser.add_argument("--view-mode", choices=("side-by-side", "overlay", "depth"), default="side-by-side")
    parser.add_argument("--overlay-alpha", type=float, default=0.45, help="Blend factor for overlay view.")
    parser.add_argument("--display-mode", choices=("inverse", "depth"), default="inverse", help="Use inverse depth for more contrast in live scenes.")
    parser.add_argument("--mirror", action="store_true", help="Mirror the camera preview horizontally.")
    parser.add_argument("--auto-reset-threshold", type=float, default=0.0, help="Auto-reset stream state when frame-to-frame appearance changes sharply. 0 disables it.")
    parser.add_argument("--range-ema", type=float, default=0.85, help="EMA smoothing for display depth range. Higher is stabler, lower is more reactive.")
    args = parser.parse_args()

    run_demo(
        checkpoint_path=args.checkpoint,
        source=args.source,
        config_path=args.config,
        device=args.device,
        dtype=args.dtype,
        max_side=args.max_side,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        camera_fps=args.camera_fps,
        stream_max_cache_len=args.stream_max_cache_len,
        stream_reset_interval=args.stream_reset_interval,
        variant=args.variant,
        view_mode=args.view_mode,
        overlay_alpha=args.overlay_alpha,
        mirror=args.mirror,
        auto_reset_threshold=args.auto_reset_threshold,
        range_ema=args.range_ema,
        display_mode=args.display_mode,
    )


if __name__ == "__main__":
    main()
