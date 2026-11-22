"""Fetch the Unitree Go2 reference model from MuJoCo Menagerie at a pinned commit (ADR 0002).

    python -m model.fetch_reference

Files land in model/reference/go2/ (git-ignored) and are verified against model/reference/go2.sha256.
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

from model.spec import REPO_ROOT

MENAGERIE_COMMIT = "370c43965323c0944b433b202aac5058b5f907a9"
BASE_URL = f"https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{MENAGERIE_COMMIT}/unitree_go2"
REF_DIR = REPO_ROOT / "model" / "reference"
GO2_DIR = REF_DIR / "go2"
MANIFEST = REF_DIR / "go2.sha256"


def _manifest() -> list[tuple[str, str]]:
    rows = []
    for line in MANIFEST.read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            rows.append((digest, name.strip()))
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def go2_scene_path(fetch: bool = True) -> Path:
    """Path to the verified Go2 scene.xml, fetching it first if needed."""
    if fetch:
        ensure_go2()
    return GO2_DIR / "scene.xml"


def ensure_go2() -> None:
    for digest, name in _manifest():
        dest = GO2_DIR / name
        if dest.exists() and _sha256(dest) == digest:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f"{BASE_URL}/{name}", timeout=60) as resp:
            data = resp.read()
        got = hashlib.sha256(data).hexdigest()
        if got != digest:
            raise RuntimeError(f"checksum mismatch for {name}: expected {digest}, got {got}")
        dest.write_bytes(data)


def main() -> int:
    ensure_go2()
    print(f"Go2 reference verified in {GO2_DIR} (menagerie {MENAGERIE_COMMIT[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
