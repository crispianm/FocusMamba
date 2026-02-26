"""Setup script for FocusMamba."""
from setuptools import setup, find_packages

setup(
    name="focusmamba",
    version="0.1.0",
    description="Degradation-Robust Metric Video Depth Estimation via Selective State Spaces",
    author="",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests", "tools", "genfocus_data", "checkpoints"]),
    install_requires=[
        "torch>=2.2",
        "torchvision>=0.17",
        "einops>=0.7",
        "numpy>=1.26",
        "pyyaml>=6.0",
        "tqdm>=4.66",
        "tensorboard>=2.15",
    ],
    extras_require={
        "dev": ["pytest>=7.4", "black", "ruff"],
        "export": ["onnx>=1.15", "onnxruntime>=1.17"],
        "eval": ["pandas>=2.1", "matplotlib>=3.8", "seaborn>=0.13"],
    },
    entry_points={
        "console_scripts": [
            "focusmamba-train=train:main",
            "focusmamba-demo=inference.demo:main",
            "focusmamba-benchmark=evaluation.benchmark:main",
        ],
    },
)
