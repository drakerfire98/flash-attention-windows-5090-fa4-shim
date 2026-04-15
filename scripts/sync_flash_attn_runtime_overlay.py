from __future__ import annotations

import shutil
import sys
from pathlib import Path

from patch_flash_attn_sm120_backward import ensure_patch_applied


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_source() -> Path:
    return _repo_root().parents[0] / "third_party" / "flash-attention-for-windows" / "flash_attn" / "cute" / "interface.py"


def _default_target() -> Path:
    return _repo_root() / "flash_attn_runtime" / "src" / "flash_attn" / "cute" / "interface.py"


def main() -> None:
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else _default_source()
    target = Path(sys.argv[2]) if len(sys.argv) > 2 else _default_target()

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    ensure_patch_applied(target, verbose=True)
    print(f"- synced_overlay_source={source}")
    print(f"- synced_overlay_target={target}")


if __name__ == "__main__":
    main()
