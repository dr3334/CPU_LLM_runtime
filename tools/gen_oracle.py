#!/usr/bin/env python3
"""Generate golden reference data for llmrt (Qwen3-0.6B, fp32).

The C++ runtime must reproduce the HuggingFace Qwen3 forward pass bit-for-bit
up to fp32 rounding. This script runs the *authoritative* reference and dumps
every intermediate the runtime needs to be checked against, as raw
little-endian binaries plus a JSON manifest that C++ can read with mmap and the
project's own JSON parser (no numpy/npz/zip on the C++ side).

Layout produced under --out (default: data/golden):

    manifest.json        array table: name -> {dtype, shape, file, ...}
    input_ids.i32.bin    [total_tokens]        all prompts concatenated
    logits.f32.bin       [total_tokens, vocab] fp32 logits, per-prompt offsets
    embed.f32.bin        [seq0, hidden]        embedding output, prompt 0
    final_norm.f32.bin   [seq0, hidden]        output of model.norm, prompt 0
    layer_hidden.f32.bin [28, seq0, hidden]    output of each decoder layer
    ops/<name>.f32.bin   per-op inputs/outputs for prompt 0, layer 0

Run it with the pinned reference environment:

    PYTHONPATH=/home/dr/mtk_toolkit_enhence/reference \\
    /home/dr/mtk_toolkit_enhence/.venv-ref/bin/python tools/gen_oracle.py

or, after `uv sync` in tools/:

    uv run --project tools tools/gen_oracle.py

Only the Python tooling may use torch/transformers: the C++ inference path
never links them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

DEFAULT_MODEL = "/home/dr/models/Qwen3-0.6B"
DEFAULT_REF_DIR = "/home/dr/mtk_toolkit_enhence/reference"
DEFAULT_OUT = "data/golden"

# Deterministic text prompts (zh + en). Prompt 0 is a seeded random-token
# prompt instead: random ids stress the op-level checks far harder than natural
# text, whose activations are comparatively well conditioned.
TEXT_PROMPTS = [
    "The capital of France is",
    "Explain in one sentence why the sky appears blue.",
    "1, 2, 3, 5, 8, 13,",
    "中国的首都是",
    "用一句话解释什么是机器学习。",
    "def fibonacci(n):",
]
RANDOM_TOKENS = 12
RANDOM_SEED = 42

# Greedy continuation capture. 16 steps is enough to catch a plumbing bug in the
# generation loop and to make "greedy token 100% identical" meaningful, without
# costing a full forward per step for thousands of steps.
GREEDY_STEPS = 16
# From generation_config.json: <|im_end|> and <|endoftext|>. Either ends the
# sequence, which is what Qwen3 was trained with.
EOS_TOKEN_IDS = [151645, 151643]


def import_reference(ref_dir: str):
    """Import the vendored (pinned) Qwen3 implementation, else stock transformers."""
    if ref_dir and ref_dir not in sys.path:
        sys.path.insert(0, ref_dir)
    try:
        from transformers_qwen3 import Qwen3Config, Qwen3ForCausalLM  # type: ignore

        return Qwen3Config, Qwen3ForCausalLM, f"vendored:{ref_dir}"
    except ImportError:
        from transformers import Qwen3Config, Qwen3ForCausalLM  # type: ignore

        return Qwen3Config, Qwen3ForCausalLM, "stock:transformers"


def import_modeling_module(ref_source: str):
    """Returns the module holding `apply_rotary_pos_emb` for the reference run."""
    if ref_source.startswith("vendored"):
        import transformers_qwen3.modeling_qwen3 as mq  # type: ignore

        return mq
    from transformers.models.qwen3 import modeling_qwen3 as mq  # type: ignore

    return mq


class Golden:
    """Collects named arrays and writes them as raw binaries + a manifest."""

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.ops_dir = out_dir / "ops"
        self.arrays: dict[str, dict] = {}

    def add_f32(self, name: str, tensor: torch.Tensor, subdir: str = "") -> None:
        arr = tensor.detach().to(torch.float32).contiguous().cpu().numpy()
        self._write(name, arr, "<f4", subdir)

    def add_i32(self, name: str, tensor: torch.Tensor, subdir: str = "") -> None:
        arr = tensor.detach().to(torch.int32).contiguous().cpu().numpy()
        self._write(name, arr, "<i4", subdir)

    def _write(self, name: str, arr: np.ndarray, dtype: str, subdir: str) -> None:
        arr = np.ascontiguousarray(arr, dtype=dtype)
        assert name not in self.arrays, f"duplicate array name: {name}"
        rel = f"{subdir}{name}.{ 'i32' if dtype == '<i4' else 'f32' }.bin"
        path = self.out_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        arr.tofile(path)
        self.arrays[name] = {
            "dtype": "i32" if dtype == "<i4" else "f32",
            "shape": list(arr.shape),
            "file": rel,
            "nbytes": int(arr.nbytes),
        }

    def add_meta(self, name: str, **extra) -> None:
        self.arrays[name].update(extra)

    def save_manifest(self, meta: dict) -> None:
        manifest = dict(meta)
        manifest["arrays"] = self.arrays
        (self.out_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False)
        )


def build_prompts(tokenizer) -> list[torch.Tensor]:
    prompts: list[torch.Tensor] = []

    gen = torch.Generator().manual_seed(RANDOM_SEED)
    prompts.append(
        torch.randint(0, 1000, (RANDOM_TOKENS,), generator=gen, dtype=torch.long)
    )
    for text in TEXT_PROMPTS:
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
        prompts.append(ids[0].to(torch.long))
    return prompts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--ref-dir", default=DEFAULT_REF_DIR)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    out_dir = (root / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    Qwen3Config, Qwen3ForCausalLM, ref_source = import_reference(args.ref_dir)
    print(f"reference      : {ref_source}")
    print(f"torch          : {torch.__version__}")
    print(f"model          : {args.model}")
    print(f"out            : {out_dir}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = Qwen3ForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        local_files_only=True,
    )
    model.eval()

    cfg = model.config
    print(
        f"config         : layers={cfg.num_hidden_layers} hidden={cfg.hidden_size} "
        f"heads={cfg.num_attention_heads} kv_heads={cfg.num_key_value_heads} "
        f"head_dim={cfg.head_dim} vocab={cfg.vocab_size}"
    )

    prompts = build_prompts(tokenizer)
    lengths = [int(p.numel()) for p in prompts]
    total = sum(lengths)
    print(f"prompts        : {len(prompts)} (total {total} tokens), lengths={lengths}")

    golden = Golden(out_dir)

    # ---- 1) logits for every prompt, concatenated, no padding -------------
    all_ids = torch.cat(prompts).to(torch.long)
    logits_chunks = []
    with torch.no_grad():
        for p in prompts:
            out = model(p.unsqueeze(0))
            logits_chunks.append(out.logits[0])
    logits = torch.cat(logits_chunks, dim=0)

    golden.add_i32("input_ids", all_ids)
    golden.add_f32("logits", logits)

    offsets, acc = [], 0
    for n in lengths:
        offsets.append(acc)
        acc += n
    golden.add_meta("input_ids", prompt_lengths=lengths, prompt_offsets=offsets)
    golden.add_meta("logits", prompt_lengths=lengths, prompt_offsets=offsets, vocab=int(cfg.vocab_size))

    # ---- 1b) greedy continuations, one token at a time --------------------
    #
    # This is what verifies the *generation loop* rather than just the argmax:
    # the runtime has to feed its own output back in and keep agreeing.
    #
    # Deliberately NOT model.generate(). That uses a KV cache, so its per-step
    # logits come out of a different computation path (cached attention versus a
    # full re-forward) and differ in the last bits. The runtime being tested has
    # no cache, so the reference is built the same way it runs -- otherwise the
    # comparison would be measuring the cache, not the loop.
    #
    # The cost is one full forward per generated token: 7 prompts x up to 16
    # steps. Same reason the runtime is slow without a cache, same magnitude.
    greedy: list[list[int]] = []
    greedy_lengths: list[int] = []
    with torch.no_grad():
        for p in prompts:
            ids = [int(t) for t in p]
            for _ in range(GREEDY_STEPS):
                out = model(torch.tensor([ids], dtype=torch.long))
                nxt = int(out.logits[0, -1].argmax())
                ids.append(nxt)
                if nxt in EOS_TOKEN_IDS:
                    break
            new = ids[len(p):]
            greedy_lengths.append(len(new))
            # Right-padded with -1; readers use greedy_lengths and never look
            # past the end of a sequence.
            greedy.append(new + [-1] * (GREEDY_STEPS - len(new)))

    golden.add_i32("greedy_ids", torch.tensor(greedy, dtype=torch.int32))
    golden.add_meta("greedy_ids", greedy_lengths=greedy_lengths, max_new=GREEDY_STEPS,
                    eos_token_ids=EOS_TOKEN_IDS)
    hits = sum(1 for n in greedy_lengths if n < GREEDY_STEPS)
    print(f"greedy         : {GREEDY_STEPS} steps max, lengths={greedy_lengths} "
          f"({hits} stopped early on eos)")

    # ---- 2) per-layer hidden states for prompt 0 -------------------------
    seq0 = prompts[0].unsqueeze(0)
    hiddens = []

    def grab_layer(mod, _inp, out):
        hiddens.append((out[0] if isinstance(out, tuple) else out).detach())

    handles = [layer.register_forward_hook(grab_layer) for layer in model.model.layers]
    with torch.no_grad():
        model(seq0)
    for h in handles:
        h.remove()

    golden.add_f32("layer_hidden", torch.stack(hiddens, dim=0))
    golden.add_f32("final_norm", model.model.norm(hiddens[-1]))
    golden.add_f32("embed_out", model.model.embed_tokens(seq0)[0])

    # ---- 3) per-op dumps for prompt 0, layer 0 ---------------------------
    cap = {}
    state = {"rope_call": 0}
    layer0 = model.model.layers[0]
    attn0 = layer0.self_attn
    mlp0 = layer0.mlp

    def hook_in_out(key):
        def h(_m, inputs, output):
            if inputs:
                cap[f"{key}_in"] = inputs[0].detach()
            cap[f"{key}_out"] = output.detach()
        return h

    def hook_out(key):
        def h(_m, _i, output):
            cap[key] = output.detach()
        return h

    mq = import_modeling_module(ref_source)
    orig_rope = mq.apply_rotary_pos_emb

    def patched_rope(q, k, cos, sin, *a, **kw):
        q_out, k_out = orig_rope(q, k, cos, sin, *a, **kw)
        if state["rope_call"] == 0:
            cap["rope_q_in"] = q.detach()
            cap["rope_k_in"] = k.detach()
            cap["rope_q_out"] = q_out.detach()
            cap["rope_k_out"] = k_out.detach()
            cap["rope_cos"] = cos.detach()
            cap["rope_sin"] = sin.detach()
        state["rope_call"] += 1
        return q_out, k_out

    hh = [
        layer0.input_layernorm.register_forward_hook(hook_in_out("input_layernorm")),
        attn0.q_proj.register_forward_hook(hook_in_out("q_proj")),
        attn0.k_proj.register_forward_hook(hook_in_out("k_proj")),
        attn0.v_proj.register_forward_hook(hook_in_out("v_proj")),
        attn0.q_norm.register_forward_hook(hook_out("q_norm_out")),
        attn0.k_norm.register_forward_hook(hook_out("k_norm_out")),
        attn0.o_proj.register_forward_hook(hook_in_out("o_proj")),
        layer0.post_attention_layernorm.register_forward_hook(
            hook_in_out("post_attention_layernorm")
        ),
        mlp0.gate_proj.register_forward_hook(hook_out("gate_proj_out")),
        mlp0.up_proj.register_forward_hook(hook_out("up_proj_out")),
        mlp0.down_proj.register_forward_hook(hook_in_out("down_proj")),
    ]
    mq.apply_rotary_pos_emb = patched_rope
    try:
        with torch.no_grad():
            model(seq0)
    finally:
        mq.apply_rotary_pos_emb = orig_rope
        for h in hh:
            h.remove()

    # SwiGLU output is exactly what down_proj consumed.
    cap["swiglu_out"] = cap["down_proj_in"]
    # Attention output before o_proj, after the head merge.
    cap["attn_out"] = cap["o_proj_in"]

    for key, tensor in sorted(cap.items()):
        golden.add_f32(key, tensor, subdir="ops/")

    golden.save_manifest(
        {
            "generated_by": "tools/gen_oracle.py",
            "model_dir": str(args.model),
            "reference_source": ref_source,
            "transformers": __import__("transformers").__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "compute_dtype": "float32",
            "attn_implementation": "eager",
            "tie_word_embeddings": bool(cfg.tie_word_embeddings),
            "config": {
                "num_hidden_layers": int(cfg.num_hidden_layers),
                "hidden_size": int(cfg.hidden_size),
                "num_attention_heads": int(cfg.num_attention_heads),
                "num_key_value_heads": int(cfg.num_key_value_heads),
                "head_dim": int(cfg.head_dim),
                "intermediate_size": int(cfg.intermediate_size),
                "vocab_size": int(cfg.vocab_size),
                "rope_theta": float(cfg.rope_theta),
                "rms_norm_eps": float(cfg.rms_norm_eps),
            },
            "text_prompts": TEXT_PROMPTS,
            "random_prompt": {"n_tokens": RANDOM_TOKENS, "seed": RANDOM_SEED, "vocab_hi": 1000},
            "prompt_offsets": offsets,
            "prompt_lengths": lengths,
            # Mirrored at the top level, not only on the array entry, so the C++
            # reader can use the same flat lookup as prompt_offsets. golden_ids
            # is right-padded with -1, so these say how much of each row is real.
            "greedy_lengths": greedy_lengths,
            "greedy_max_new": GREEDY_STEPS,
        }
    )

    total_mb = sum(v["nbytes"] for v in golden.arrays.values()) / 1e6
    print(f"wrote {len(golden.arrays)} arrays ({total_mb:.1f} MB) -> {out_dir}")
    for name in sorted(golden.arrays):
        info = golden.arrays[name]
        print(f"  {name:32} {info['dtype']:3} {info['shape']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
