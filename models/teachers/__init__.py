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


def build_teacher(name: str, **kwargs) -> TeacherBase:
    """Instantiate a teacher by name.

    Args:
        name: One of the registered teacher names.
        **kwargs: Passed to the teacher constructor.

    Returns:
        TeacherBase instance.
    """
    if name not in TEACHER_REGISTRY:
        raise ValueError(
            f"Unknown teacher '{name}'. Available: {list(TEACHER_REGISTRY.keys())}"
        )
    return TEACHER_REGISTRY[name](**kwargs)


__all__ = [
    "TeacherBase",
    "DepthAnythingV3Teacher",
    "DepthProTeacher",
    "Metric3DV2Teacher",
    "VideoDepthAnythingTeacher",
    "TEACHER_REGISTRY",
    "build_teacher",
]
