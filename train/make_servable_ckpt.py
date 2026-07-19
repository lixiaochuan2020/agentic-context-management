"""Make a text-only Qwen3.5 SFT checkpoint servable by vLLM.

Background: vLLM only registers `Qwen3_5ForConditionalGeneration` (the full VLM)
for the qwen3_5 family — there is no text-only serving path. A checkpoint trained
from the `qwen3.5-9b-lm-only` base has model_type `qwen3_5_text` and lacks the
`model.visual.*` (vision tower) and `mtp.*` tensors, so vLLM rejects it.

The LM key layout is IDENTICAL between our SFT ckpt and the full base
(`model.language_model.*`, `lm_head.weight`), so we build a servable checkpoint
by starting from the full base (correct config / sharding / index / visual / mtp)
and OVERWRITING the LM tensors with our trained ones. Shape-preserving, additive.

Usage:
    python -m train.make_servable_ckpt \
        --sft  <ckpt>/checkpoint-XXX \
        --base Qwen/Qwen3.5-9B \
        --out  <ckpt>-servable
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file


def load_our_lm_tensors(sft_dir: Path) -> dict:
    """Load all tensors from the SFT ckpt (these are the LM weights to inject)."""
    files = [sft_dir / "model.safetensors"]
    if not files[0].is_file():
        files = sorted(sft_dir.glob("*.safetensors"))
    tensors: dict = {}
    for f in files:
        tensors.update(load_file(str(f)))
    return tensors


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sft", type=Path, required=True)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    our = load_our_lm_tensors(args.sft)
    print(f"[load] {len(our)} LM tensors from {args.sft}")

    # Start from a full copy of the base (config, index, tokenizer, preprocessor,
    # all shards incl. visual + mtp), then overwrite LM tensors shard-by-shard.
    if args.out.exists():
        raise SystemExit(f"refuse to overwrite existing {args.out}")
    print(f"[copy] base -> {args.out}")
    shutil.copytree(args.base, args.out, symlinks=False)  # resolves HF-cache symlinks

    shards = sorted(args.out.glob("*.safetensors"))
    placed = set()
    for shard in shards:
        with safe_open(str(shard), framework="pt") as h:
            keys = list(h.keys())
            sd = {k: h.get_tensor(k) for k in keys}
            meta = h.metadata() or {}
        n_repl = 0
        for k in keys:
            if k in our:
                if sd[k].shape != our[k].shape:
                    raise SystemExit(f"shape mismatch for {k}: base {sd[k].shape} vs sft {our[k].shape}")
                sd[k] = our[k].to(sd[k].dtype)
                placed.add(k)
                n_repl += 1
        save_file(sd, str(shard), metadata=meta)
        print(f"[shard] {shard.name}: replaced {n_repl}/{len(keys)} tensors")

    missing = set(our) - placed
    if missing:
        raise SystemExit(f"ERROR: {len(missing)} SFT tensors had no matching base key, e.g. {list(missing)[:5]}")
    print(f"[done] all {len(placed)} LM tensors injected; visual + mtp kept from base.")
    print(f"[done] servable checkpoint at {args.out}")
    # sanity: confirm config is the full VLM type
    cfg = json.loads((args.out / "config.json").read_text())
    print(f"[config] model_type={cfg.get('model_type')} architectures={cfg.get('architectures')}")


if __name__ == "__main__":
    main()
