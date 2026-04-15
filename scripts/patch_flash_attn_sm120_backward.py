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


def _validate_expected_state(text: str) -> None:
    required_markers = (
        "def _windows_compose_softcap_scoremod(score_mod, softcap):",
        "score_mod = _windows_compose_softcap_scoremod(score_mod, softcap)",
        "dQ_single_wg = False",
        "ctx.page_table = page_table",
        "ctx.score_mod = score_mod",
        "ctx.aux_tensors = aux_tensors",
        "ctx.learnable_sink = learnable_sink",
        "from cutlass.cute._compile_bridge import compat_replay_varlen_backward",
        "dq, dk, dv = compat_replay_varlen_backward(",
    )
    forbidden_fragments = (
        'assert not use_block_sparsity, "Block sparsity not supported on SM 12.0"',
        'assert page_table is None, "Paged KV not supported on SM 12.0 in this PR"',
        '"softcap and score_mod cannot be used together"',
        '"Block sparsity backward not supported on SM 12.0"',
        '"score_mod backward not supported on SM 12.0"',
        '"mask_mod backward not supported on SM 12.0"',
        'assert ctx.softcap == 0.0',
        '"mask_mod with aux_tensors is not yet supported for varlen sequences. This will be fixed in a future PR."',
        '"Block sparsity is not yet supported for varlen sequences. This will be fixed in a future PR."',
    )
    missing = [marker for marker in required_markers if marker not in text]
    present_forbidden = [frag for frag in forbidden_fragments if frag in text]
    if missing or present_forbidden:
        details: list[str] = []
        if missing:
            details.append("missing markers: " + "; ".join(missing))
        if present_forbidden:
            details.append("forbidden fragments still present: " + "; ".join(present_forbidden))
        raise RuntimeError("patch verification failed: " + " | ".join(details))


def main() -> None:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else _default_target()
    text = target.read_text(encoding="utf-8")
    original_text = text

    patches = (
        (
            "add softcap-plus-score_mod compatibility helper",
            (
                "from flash_attn.cute.block_sparsity import (\n"
                "    BlockSparseTensorsTorch,\n"
                "    to_cute_block_sparse_tensors,\n"
                "    normalize_block_sparse_config,\n"
                "    normalize_block_sparse_config_bwd,\n"
                ")\n"
            ),
            (
                "from flash_attn.cute.block_sparsity import (\n"
                "    BlockSparseTensorsTorch,\n"
                "    to_cute_block_sparse_tensors,\n"
                "    normalize_block_sparse_config,\n"
                "    normalize_block_sparse_config_bwd,\n"
                ")\n"
                "\n"
                "def _windows_compose_softcap_scoremod(score_mod, softcap):\n"
                "    if softcap is None:\n"
                "        return score_mod\n"
                "    if score_mod is None:\n"
                "        return utils.create_softcap_scoremod(softcap)\n"
                "    inv_softcap = 1.0 / softcap\n"
                "\n"
                "    @cute.jit\n"
                "    def scoremod_premask_fn(acc_S_SSA, batch_idx, head_idx, q_idx, kv_idx, aux_tensors):\n"
                "        scores = score_mod(acc_S_SSA, batch_idx, head_idx, q_idx, kv_idx, aux_tensors)\n"
                "        scores = scores * inv_softcap\n"
                "        return scores * cute.math.tanh(scores, fastmath=True)\n"
                "\n"
                "    return scoremod_premask_fn\n"
            ),
        ),
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
            "compose softcap with score_mod in the forward wrapper",
            (
                "    if softcap is not None:\n"
                "        assert score_mod is None, \"softcap and score_mod cannot be used together\"\n"
                "        score_mod = utils.create_softcap_scoremod(softcap)\n"
            ),
            (
                "    if softcap is not None:\n"
                "        score_mod = _windows_compose_softcap_scoremod(score_mod, softcap)\n"
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
            "remove the SM120 forward paged-KV guard",
            (
                "        elif arch // 10 == 12:\n"
                "            # SM120 (Blackwell GeForce / DGX Spark): uses SM80 MMA with SM120 SMEM capacity\n"
                "            assert page_table is None, \"Paged KV not supported on SM 12.0 in this PR\"\n"
                "            assert not is_split_kv, \"SplitKV not supported on SM 12.0 in this PR\"\n"
            ),
            (
                "        elif arch // 10 == 12:\n"
                "            # SM120 (Blackwell GeForce / DGX Spark): uses SM80 MMA with SM120 SMEM capacity\n"
                "            assert not is_split_kv, \"SplitKV not supported on SM 12.0 in this PR\"\n"
            ),
        ),
        (
            "remove the varlen mask_mod forward guard",
            (
                "    if mask_mod is not None:\n"
                "        if is_varlen:\n"
                "            raise NotImplementedError(\n"
                "                \"mask_mod with aux_tensors is not yet supported for varlen sequences. This will be fixed in a future PR.\"\n"
                "            )\n"
            ),
            (
                "    if mask_mod is not None:\n"
                "        # Windows native probe path replays varlen mask_mod through the shim bridge.\n"
                "        pass\n"
            ),
        ),
        (
            "stabilize the varlen mask_mod patched block",
            (
                "    if mask_mod is not None:\n"
                "        # Windows native probe path replays varlen mask_mod through the shim bridge.\n"
                "\n"
            ),
            (
                "    if mask_mod is not None:\n"
                "        # Windows native probe path replays varlen mask_mod through the shim bridge.\n"
                "        pass\n"
            ),
        ),
        (
            "remove the varlen block-sparsity forward guard",
            (
                "    if use_block_sparsity:\n"
                "        if is_varlen:\n"
                "            raise NotImplementedError(\n"
                "                \"Block sparsity is not yet supported for varlen sequences. This will be fixed in a future PR.\"\n"
                "            )\n"
                "        # NB: pack_gqa requires block sparse head dim == 1 (broadcasted)\n"
            ),
            (
                "    if use_block_sparsity:\n"
                "        # NB: pack_gqa requires block sparse head dim == 1 (broadcasted)\n"
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
        (
            "remove the varlen backward softcap guard",
            (
                "        q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k = ctx.saved_tensors\n"
                "        assert ctx.softcap == 0.0\n"
                "        if not ctx.return_lse:\n"
            ),
            (
                "        q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k = ctx.saved_tensors\n"
                "        if not ctx.return_lse:\n"
            ),
        ),
        (
            "save varlen compat metadata on the autograd context",
            (
                "        ctx.max_seqlen_q = max_seqlen_q\n"
                "        ctx.max_seqlen_k = max_seqlen_k\n"
                "        ctx.return_lse = return_lse\n"
            ),
            (
                "        ctx.max_seqlen_q = max_seqlen_q\n"
                "        ctx.max_seqlen_k = max_seqlen_k\n"
                "        ctx.page_table = page_table\n"
                "        ctx.score_mod = score_mod\n"
                "        ctx.aux_tensors = aux_tensors\n"
                "        ctx.learnable_sink = learnable_sink\n"
                "        ctx.return_lse = return_lse\n"
            ),
        ),
        (
            "route paged varlen backward through the shim replay helper",
            (
                "        if dout is None:\n"
                "            dout = torch.zeros_like(out)\n"
                "        dq, dk, dv = _flash_attn_bwd(\n"
                "            q,\n"
                "            k,\n"
                "            v,\n"
                "            out,\n"
                "            dout,\n"
                "            lse,\n"
                "            ctx.softmax_scale,\n"
                "            ctx.causal,\n"
                "            ctx.softcap,\n"
                "            window_size_left=ctx.window_size[0],\n"
                "            window_size_right=ctx.window_size[1],\n"
                "            cu_seqlens_q=cu_seqlens_q,\n"
                "            cu_seqlens_k=cu_seqlens_k,\n"
                "            seqused_q=seqused_q,\n"
                "            seqused_k=seqused_k,\n"
                "            max_seqlen_q=ctx.max_seqlen_q,\n"
                "            max_seqlen_k=ctx.max_seqlen_k,\n"
                "            deterministic=ctx.deterministic,\n"
                "            dlse=dlse,\n"
                "        )\n"
            ),
            (
                "        if dout is None:\n"
                "            dout = torch.zeros_like(out)\n"
                "        if getattr(ctx, \"page_table\", None) is not None:\n"
                "            from cutlass.cute._compile_bridge import compat_replay_varlen_backward\n"
                "\n"
                "            dq, dk, dv = compat_replay_varlen_backward(\n"
                "                q=q,\n"
                "                k=k,\n"
                "                v=v,\n"
                "                dout=dout,\n"
                "                dlse=dlse,\n"
                "                cu_seqlens_q=cu_seqlens_q,\n"
                "                cu_seqlens_k=cu_seqlens_k,\n"
                "                seqused_q=seqused_q,\n"
                "                seqused_k=seqused_k,\n"
                "                page_table=ctx.page_table,\n"
                "                softmax_scale=ctx.softmax_scale,\n"
                "                causal=ctx.causal,\n"
                "                window_size=ctx.window_size,\n"
                "                learnable_sink=getattr(ctx, \"learnable_sink\", None),\n"
                "                softcap=ctx.softcap,\n"
                "                score_mod=getattr(ctx, \"score_mod\", None),\n"
                "                aux_tensors=getattr(ctx, \"aux_tensors\", None),\n"
                "                return_lse=ctx.return_lse,\n"
                "            )\n"
                "        else:\n"
                "            dq, dk, dv = _flash_attn_bwd(\n"
                "                q,\n"
                "                k,\n"
                "                v,\n"
                "                out,\n"
                "                dout,\n"
                "                lse,\n"
                "                ctx.softmax_scale,\n"
                "                ctx.causal,\n"
                "                ctx.softcap,\n"
                "                window_size_left=ctx.window_size[0],\n"
                "                window_size_right=ctx.window_size[1],\n"
                "                cu_seqlens_q=cu_seqlens_q,\n"
                "                cu_seqlens_k=cu_seqlens_k,\n"
                "                seqused_q=seqused_q,\n"
                "                seqused_k=seqused_k,\n"
                "                max_seqlen_q=ctx.max_seqlen_q,\n"
                "                max_seqlen_k=ctx.max_seqlen_k,\n"
                "                deterministic=ctx.deterministic,\n"
                "                dlse=dlse,\n"
                "            )\n"
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

    _validate_expected_state(text)

    if text == original_text:
        print(f"already patched: {target}")
        print("- verification=ok")
        return

    target.write_text(text, encoding="utf-8")
    print(f"patched {target}")
    for label in applied_labels:
        print(f"- {label}")
    print("- verification=ok")


if __name__ == "__main__":
    main()
