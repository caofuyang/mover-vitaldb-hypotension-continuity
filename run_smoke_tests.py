"""Run tests that require no patient-level or source-database files."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
environment = os.environ.copy()
python_path = [str(ROOT / "code"), str(ROOT / "tests")]
if environment.get("PYTHONPATH"):
    python_path.append(environment["PYTHONPATH"])
environment["PYTHONPATH"] = os.pathsep.join(python_path)
subprocess.run(
    [sys.executable, str(ROOT / "tests" / "run_all_method_tests.py")],
    check=True,
    cwd=ROOT,
    env=environment,
)
subprocess.run(
    [sys.executable, str(ROOT / "verify_release.py")],
    check=True,
    cwd=ROOT,
    env=environment,
)
