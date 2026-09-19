#!/usr/bin/env python3
"""Independently re-derive the Qwen3 forward pass and check it against the golden.

Why this exists
---------------
Every downstream C++ test compares against data/golden. If that data is wrong,
every test is *consistently* wrong and the bug hides. So before trusting it we
re-derive the whole forward pass from the raw safetensors weights using plain
tensor math (no nn.Module, no model.forward) and assert it reproduces every
captured intermediate.

This doubles as the executable specification of the exact operator order that
the C++ implementation must mirror:

    embed -> input_layernorm -> q/k/v_proj
          -> q_norm/k_norm (per-head RMSNorm over head_dim, BEFORE RoPE)
          -> transpose to [B, H, S, D] -> RoPE (rotate_half / half-split)
          -> KV append -> GQA repeat -> scaled dot-product + causal mask
          -> softmax(fp32) @ v -> merge heads -> o_proj -> residual
          -> post_attention_layernorm -> gate/up -> silu(gate)*up -> down_proj
          -> residual -> final norm -> lm_head (= embed_tokens, tied)

Run:
    PYTHONPATH=/home/dr/mtk_toolkit_enhence/reference \\
    /home/dr/mtk_toolkit_enhence/.venv-ref/bin/python tools/check_oracle.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GOLDEN = ROOT / "data/golden"
DEFAULT_MODEL = "/home/dr/models/Qwen3-0.6B"

RESULTS: list[tuple[str, bool, str]] = []


def cmp(name: str, got: np.ndarray, want: np.ndarray, tol: float = 2e-4) -> bool:
    g = np.asarray(got, dtype=np.float64).ravel()
    w = np.asarray(want, dtype=np.float64).ravel()
    if g.shape != w.shape:
        RESULTS.append((name, False, f"shape {g.shape} vs {w.shape}"))
        return False
    denom = np.linalg.norm(g) * np.linalg.norm(w)
    cos = float(np.dot(g, w) / denom) if denom > 0 else 1.0
    maxabs = float(np.max(np.abs(g - w)))
    scale = float(np.max(np.abs(w))) + 1e-12
    ok = cos > 0.99999 and maxabs / scale < tol
    RESULTS.append(
        (name, ok, f"cos={cos:.8f} maxabs={maxabs:.3e} rel={maxabs / scale:.3e}")
    )
    return ok


def load_golden(golden_dir: Path) -> tuple[dict, dict[str, np.ndarray]]:
    manifest = json.loads((golden_dir / "manifest.json").read_text())
    arrays: dict[str, np.ndarray] = {}
    for name, info in manifest["arrays"].items():
        dtype = np.int32 if info["dtype"] == "i32" else np.float32
        flat = np.fromfile(golden_dir / info["file"], dtype=dtype)
        arrays[name] = flat.reshape(info["shape"])
    return manifest, arrays


def rms_norm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    xf = x.astype(np.float32)
    var = np.mean(xf * xf, axis=-1, keepdims=True)
    return (xf / np.sqrt(var + eps)) * w.astype(np.float32)


def rotate_half(x: np.ndarray) -> np.ndarray:
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def apply_rope(q, k, cos, sin):
    # cos/sin: [B, S, D] -> [B, 1, S, D]
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    q_out = q * cos + rotate_half(q) * sin
    k_out = k * cos + rotate_half(k) * sin
    return q_out, k_out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default=str(DEFAULT_GOLDEN))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prompt-index", type=int, default=0,
                    help="which prompt the op-level dumps belong to (0 = random ids)")
    args = ap.parse_args()

    golden_dir = Path(args.golden)
    manifest, G = load_golden(golden_dir)
    cfg = manifest["config"]
    eps = cfg["rms_norm_eps"]
    n_layers = cfg["num_hidden_layers"]
    n_heads = cfg["num_attention_heads"]
    n_kv = cfg["num_key_value_heads"]
    head_dim = cfg["head_dim"]
    n_rep = n_heads // n_kv
    scale = head_dim ** -0.5
    theta = cfg["rope_theta"]

    print(f"golden   : {golden_dir}")
    print(f"model    : {args.model}")
    print(f"reference: {manifest['reference_source']}")
    print(f"config   : {cfg}")

    from safetensors.torch import load_file

    sd = {k: v.float().numpy() for k, v in load_file(f"{args.model}/model.safetensors").items()}
    print(f"weights  : {len(sd)} tensors loaded")

    # ---- reproduce the prompt-0 op-level pass ----------------------------
    lengths = manifest["prompt_lengths"]
    pi = args.prompt_index
    offset = manifest["prompt_offsets"][pi]
    seq = lengths[pi]
    ids = G["input_ids"][offset : offset + seq].astype(np.int64)
    print(f"prompt {pi}: seq={seq} ids={ids.tolist()}")

    embed = sd["model.embed_tokens.weight"]
    h = embed[ids][None, :, :]  # [1, S, H]
    cmp("embed_out", h[0], G["embed_out"])

    # Layer 0, step by step.
    p = "model.layers.0."
    cmp("input_layernorm_in", h, G["input_layernorm_in"])

    x = rms_norm(h, sd[p + "input_layernorm.weight"], eps)
    cmp("input_layernorm_out", x, G["input_layernorm_out"])

    q = x @ sd[p + "self_attn.q_proj.weight"].T  # [1,S,2048]
    k = x @ sd[p + "self_attn.k_proj.weight"].T  # [1,S,1024]
    v = x @ sd[p + "self_attn.v_proj.weight"].T  # [1,S,1024]
    cmp("q_proj_out", q, G["q_proj_out"])
    cmp("k_proj_out", k, G["k_proj_out"])
    cmp("v_proj_out", v, G["v_proj_out"])

    qh = q.reshape(1, seq, n_heads, head_dim)
    qh = rms_norm(qh, sd[p + "self_attn.q_norm.weight"], eps)
    kh = k.reshape(1, seq, n_kv, head_dim)
    kh = rms_norm(kh, sd[p + "self_attn.k_norm.weight"], eps)
    cmp("q_norm_out", qh, G["q_norm_out"])
    cmp("k_norm_out", kh, G["k_norm_out"])

    qh = qh.transpose(0, 2, 1, 3)  # [1, H, S, D]
    kh = kh.transpose(0, 2, 1, 3)
    cmp("rope_q_in", qh, G["rope_q_in"])
    cmp("rope_k_in", kh, G["rope_k_in"])

    # RoPE frequencies.
    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    pos = np.arange(seq, dtype=np.float64)
    freqs = np.outer(pos, inv_freq)  # [S, D/2]
    emb = np.concatenate([freqs, freqs], axis=-1)  # [S, D]
    cos = np.cos(emb)[None, :, :].astype(np.float32)
    sin = np.sin(emb)[None, :, :].astype(np.float32)
    cmp("rope_cos", cos, G["rope_cos"])
    cmp("rope_sin", sin, G["rope_sin"])

    qr, kr = apply_rope(qh, kh, cos, sin)
    cmp("rope_q_out", qr, G["rope_q_out"])
    cmp("rope_k_out", kr, G["rope_k_out"])

    # GQA + causal attention.
    vr = v.reshape(1, seq, n_kv, head_dim).transpose(0, 2, 1, 3)  # [1, KV, S, D]
    kk = np.repeat(kr, n_rep, axis=1)  # [1, H, S, D]
    vv = np.repeat(vr, n_rep, axis=1)
    attn = (qr @ kk.transpose(0, 1, 3, 2)) * scale  # [1, H, S, S]
    mask = np.triu(np.full((seq, seq), -np.inf, dtype=np.float32), k=1)
    attn = attn + mask[None, None, :, :]
    attn = attn - attn.max(axis=-1, keepdims=True)
    ex = np.exp(attn)
    attn = (ex / ex.sum(axis=-1, keepdims=True)).astype(np.float32)
    ao = attn @ vv  # [1, H, S, D]
    ao = ao.transpose(0, 2, 1, 3).reshape(1, seq, n_heads * head_dim)
    cmp("attn_out", ao, G["o_proj_in"])

    o = ao @ sd[p + "self_attn.o_proj.weight"].T
    cmp("o_proj_out", o, G["o_proj_out"])

    h1 = h + o
    # `layer_hidden[i]` is the output of decoder layer i, i.e. AFTER the MLP
    # residual. The post-attention residual is only observable as the input to
    # post_attention_layernorm.
    cmp("layer0_post_attn_residual", h1, G["post_attention_layernorm_in"])

    y = rms_norm(h1, sd[p + "post_attention_layernorm.weight"], eps)
    cmp("post_attention_layernorm_out", y, G["post_attention_layernorm_out"])

    gate = y @ sd[p + "mlp.gate_proj.weight"].T
    up = y @ sd[p + "mlp.up_proj.weight"].T
    cmp("gate_proj_out", gate, G["gate_proj_out"])
    cmp("up_proj_out", up, G["up_proj_out"])

    sw = (gate / (1.0 + np.exp(-gate))) * up  # silu(gate) * up
    cmp("swiglu_out", sw, G["swiglu_out"])

    d = sw @ sd[p + "mlp.down_proj.weight"].T
    cmp("down_proj_out", d, G["down_proj_out"])
    cmp("layer0_out", h1 + d, G["layer_hidden"][0])

    # ---- full stack for the same prompt, no KV cache ---------------------
    print("\nfull forward (all 28 layers, prefill, no cache) ...")
    for li in range(n_layers):
        pl = f"model.layers.{li}."
        xn = rms_norm(h, sd[pl + "input_layernorm.weight"], eps)
        qn = (xn @ sd[pl + "self_attn.q_proj.weight"].T).reshape(1, seq, n_heads, head_dim)
        kn = (xn @ sd[pl + "self_attn.k_proj.weight"].T).reshape(1, seq, n_kv, head_dim)
        vn = (xn @ sd[pl + "self_attn.v_proj.weight"].T).reshape(1, seq, n_kv, head_dim)
        qn = rms_norm(qn, sd[pl + "self_attn.q_norm.weight"], eps).transpose(0, 2, 1, 3)
        kn = rms_norm(kn, sd[pl + "self_attn.k_norm.weight"], eps).transpose(0, 2, 1, 3)
        vn = vn.transpose(0, 2, 1, 3)
        qn, kn = apply_rope(qn, kn, cos, sin)
        an = (qn @ np.repeat(kn, n_rep, axis=1).transpose(0, 1, 3, 2)) * scale
        an = an + mask[None, None, :, :]
        an = an - an.max(axis=-1, keepdims=True)
        en = np.exp(an)
        an = (en / en.sum(axis=-1, keepdims=True)).astype(np.float32)
        on = (an @ np.repeat(vn, n_rep, axis=1)).transpose(0, 2, 1, 3).reshape(1, seq, -1)
        h = h + on @ sd[pl + "self_attn.o_proj.weight"].T
        yn = rms_norm(h, sd[pl + "post_attention_layernorm.weight"], eps)
        gn = yn @ sd[pl + "mlp.gate_proj.weight"].T
        un = yn @ sd[pl + "mlp.up_proj.weight"].T
        h = h + ((gn / (1.0 + np.exp(-gn))) * un) @ sd[pl + "mlp.down_proj.weight"].T
        cmp(f"layer_hidden[{li}]", h, G["layer_hidden"][li], tol=1e-3)

    fn = rms_norm(h, sd["model.norm.weight"], eps)
    cmp("final_norm", fn, G["final_norm"], tol=1e-3)

    logits = (fn @ sd["lm_head.weight"].T)[0]
    cmp("logits_prompt0", logits, G["logits"][offset : offset + seq], tol=1e-3)

    # ---- report ----------------------------------------------------------
    print()
    width = max(len(n) for n, _, _ in RESULTS)
    n_fail = 0
    for name, ok, detail in RESULTS:
        if not ok:
            n_fail += 1
        print(f"  [{'OK  ' if ok else 'FAIL'}] {name:<{width}}  {detail}")
    print()
    print(f"{len(RESULTS)} checks, {n_fail} failed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
