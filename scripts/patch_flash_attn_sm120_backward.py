from __future__ import annotations

import sys
from pathlib import Path


PATCH_MARKER = "    dQ_single_wg = False\n"
ANCHOR = (
    "    causal, local, window_size_left, window_size_right = _resolve_causal_local_window(\n"
    "        causal, window_size_left, window_size_right\n"
    "    )\n"
)
REPLACEMENT = ANCHOR + PATCH_MARKER


def main() -> None:
    default_target = (
        Path(__file__).resolve().parents[2]
        / "third_party"
        / "flash-attention-for-windows"
        / "flash_attn"
        / "cute"
        / "interface.py"
    )
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else default_target
    text = target.read_text(encoding="utf-8")

    if PATCH_MARKER in text:
        print(f"already patched: {target}")
        return

    if ANCHOR not in text:
        raise SystemExit(f"anchor not found in {target}")

    target.write_text(text.replace(ANCHOR, REPLACEMENT, 1), encoding="utf-8")
    print(f"patched {target}")


if __name__ == "__main__":
    main()
