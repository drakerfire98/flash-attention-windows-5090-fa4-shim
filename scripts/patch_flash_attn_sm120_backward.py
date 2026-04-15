from __future__ import annotations

import sys
from pathlib import Path


def _default_target() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "third_party"
        / "flash-attention-for-windows"
        / "flash_attn"
        / "cute"
        / "interface.py"
    )


def _apply_patch(text: str, old: str, new: str) -> tuple[str, bool]:
    if new in text:
        return text, False
    if old not in text:
        return text, False
    return text.replace(old, new, 1), True


def _collapse_consecutive_duplicates(text: str, needle: str) -> tuple[str, bool]:
    changed = False
    doubled = needle + needle
    while doubled in text:
        text = text.replace(doubled, needle)
        changed = True
    return text, changed


def main() -> None:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else _default_target()
    text = target.read_text(encoding="utf-8")
    original_text = text

    patches = (
        (
            "seed dQ_single_wg for the SM120 branch",
            (
                "    causal, local, window_size_left, window_size_right = _resolve_causal_local_window(\n"
                "        causal, window_size_left, window_size_right\n"
                "    )\n"
            ),
            (
                "    causal, local, window_size_left, window_size_right = _resolve_causal_local_window(\n"
                "        causal, window_size_left, window_size_right\n"
                "    )\n"
                "    dQ_single_wg = False\n"
            ),
        ),
        (
            "remove the SM120 forward block-sparsity guard",
            (
                "        elif arch // 10 == 12:\n"
                "            # SM120 (Blackwell GeForce / DGX Spark): uses SM80 MMA with SM120 SMEM capacity\n"
                "            assert not use_block_sparsity, \"Block sparsity not supported on SM 12.0\"\n"
                "            assert page_table is None, \"Paged KV not supported on SM 12.0 in this PR\"\n"
            ),
            (
                "        elif arch // 10 == 12:\n"
                "            # SM120 (Blackwell GeForce / DGX Spark): uses SM80 MMA with SM120 SMEM capacity\n"
                "            assert page_table is None, \"Paged KV not supported on SM 12.0 in this PR\"\n"
            ),
        ),
        (
            "remove the SM120 backward bridge-compat guards",
            (
                "        cluster_size = 1\n"
                "        use_2cta_instrs = False\n"
                "        num_threads = 128\n"
                "        assert not (block_sparse_tensors is not None), \"Block sparsity backward not supported on SM 12.0\"\n"
                "        assert score_mod is None and score_mod_bwd is None, \"score_mod backward not supported on SM 12.0\"\n"
                "        assert mask_mod is None, \"mask_mod backward not supported on SM 12.0\"\n"
                "        assert deterministic is False, \"deterministic backward not supported on SM 12.0\"\n"
            ),
            (
                "        cluster_size = 1\n"
                "        use_2cta_instrs = False\n"
                "        num_threads = 128\n"
                "        assert deterministic is False, \"deterministic backward not supported on SM 12.0\"\n"
            ),
        ),
    )

    applied_labels: list[str] = []
    text, changed = _collapse_consecutive_duplicates(text, "    dQ_single_wg = False\n")
    if changed:
        applied_labels.append("collapse duplicate dQ_single_wg lines")
    for label, old, new in patches:
        text, changed = _apply_patch(text, old, new)
        if changed:
            applied_labels.append(label)

    if text == original_text:
        print(f"already patched: {target}")
        return

    target.write_text(text, encoding="utf-8")
    print(f"patched {target}")
    for label in applied_labels:
        print(f"- {label}")


if __name__ == "__main__":
    main()
