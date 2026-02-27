"""
Teacher models package.

Provides frozen teacher wrappers for multi-teacher distillation.
All teachers take clean frames and produce metric depth pseudo-labels.
"""

from .teacher_base import TeacherBase
from .depth_anything_v3 import DepthAnythingV3Teacher
from .depth_pro import DepthProTeacher
from .metric3d_v2 import Metric3DV2Teacher
from .video_teacher import VideoDepthAnythingTeacher


TEACHER_REGISTRY = {
    "depth_anything_v3": DepthAnythingV3Teacher,
    "depth_pro": DepthProTeacher,
    "metric3d_v2": Metric3DV2Teacher,
    "video_depth_anything": VideoDepthAnythingTeacher,
}


def build_teacher(name: str, cfg: dict = None, **kwargs) -> TeacherBase:
    """Instantiate a teacher by name.

    Args:
        name: One of the registered teacher names.
        cfg: Optional config dict (e.g. from YAML). Distillation-specific keys
             (``name``, ``weight``, ``loss``, ``enabled``) are stripped out
             automatically; the rest are forwarded to the constructor.
        **kwargs: Additional overrides (take precedence over ``cfg``).

    Returns:
        TeacherBase instance.
    """
    if name not in TEACHER_REGISTRY:
        raise ValueError(
            f"Unknown teacher '{name}'. Available: {list(TEACHER_REGISTRY.keys())}"
        )
    # Keys used only by the training loop, never by teacher constructors.
    _SKIP_KEYS = {"name", "weight", "loss", "enabled"}
    constructor_kwargs = {}
    if cfg:
        constructor_kwargs = {k: v for k, v in cfg.items() if k not in _SKIP_KEYS}
    constructor_kwargs.update(kwargs)
    return TEACHER_REGISTRY[name](**constructor_kwargs)


__all__ = [
    "TeacherBase",
    "DepthAnythingV3Teacher",
    "DepthProTeacher",
    "Metric3DV2Teacher",
    "VideoDepthAnythingTeacher",
    "TEACHER_REGISTRY",
    "build_teacher",
]
