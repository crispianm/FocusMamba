#!/usr/bin/env python3
"""CLI wrapper for generating the VDA bug-hunt suite."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.vda_bug_hunt_suite import main


if __name__ == "__main__":
    raise SystemExit(main(["generate", *sys.argv[1:]]))
