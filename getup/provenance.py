"""What a training run leaves next to its checkpoints, to reproduce and document it:
params/run_info.json (when, the exact command, the git state of this repo and of the folder
it runs from, i.e. unitree_rl_mjlab) and params/code/ (a copy of the code that ran)."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def git_state(folder: Path) -> dict | None:
    """Commit and whether there were uncommitted changes (None: not a git repository)."""
    try:
        head = subprocess.run(["git", "-C", str(folder), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10)
        if head.returncode != 0:
            return None
        status = subprocess.run(["git", "-C", str(folder), "status", "--porcelain"],
                                capture_output=True, text=True, timeout=10)
        return {"commit": head.stdout.strip(),
                "uncommitted_changes": bool(status.stdout.strip())}
    except (OSError, subprocess.SubprocessError):
        return None


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_provenance(log_dir: Path, repo: Path, code: list[Path], extra: dict | None = None,
                    data: list[Path] | None = None) -> None:
    """log_dir/params/run_info.json + log_dir/params/code/ (code: files or folders) +
    copies of the data files (e.g. the fallen-pose pool), with their md5 in run_info."""
    out = Path(log_dir) / "params"
    out.mkdir(parents=True, exist_ok=True)
    info = {"date": datetime.now().isoformat(timespec="seconds"),
            "command": [sys.executable] + sys.argv, "cwd": os.getcwd(),
            "git": git_state(Path(repo)), "cwd_git": git_state(Path.cwd()), **(extra or {})}
    for p in data or []:
        p = Path(p)
        if p.is_file():
            shutil.copy2(p, out / p.name)
            info.setdefault("data", {})[p.name] = {"from": str(p.resolve()), "md5": md5(p)}
    (out / "run_info.json").write_text(json.dumps(info, indent=1))
    dest = out / "code"
    for p in code:
        p = Path(p)
        if p.is_dir():
            shutil.copytree(p, dest / p.name, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        elif p.is_file():
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest / p.name)
