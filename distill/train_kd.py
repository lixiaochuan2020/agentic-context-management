"""Offline on-policy distillation — Stage 2: KD-train the 9B student from cached teacher logprobs.

Consumes the .npz produced by distill/score_teacher_logprobs.py (input_ids, asst_pos, tk_ids,
tk_logprobs) and trains the student with top-K forward-KL at assistant positions only.

Memory trick: the student forward returns hidden states (L,H); we apply lm_head ONLY at the
assistant positions (gather hidden at asst_pos-1) → (M,V) logits, never the full (L,248K).

Parallelism: same stack as train/sft_hf.py (DeepSpeed ZeRO-3, optional Ulysses SP for long
sequences, VLM load + vision frozen). batch=1. SP×custom-loss is validated by a smoke first.

  python -m distill.train_kd --model <lm-only> --cache_dir <npz dir> --output_dir ... \
      --max_length 65536 --sp_size 1 [--max_samples 20]   # smoke
"""
from __future__ import annotations

import argparse
import glob
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from accelerate.utils import DeepSpeedSequenceParallelConfig, ParallelismConfig
from datasets import Dataset
from transformers import AutoConfig, AutoModelForImageTextToText, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from distill.kd_loss import topk_forward_kl_gathered

logger = logging.getLogger("train_kd")


def load_cache(cache_dir: str, max_length: int, max_samples: int = 0):
    files = sorted(glob.glob(f"{cache_dir}/*.npz"))
    rows, seen = [], 0
    for f in files:
        d = np.load(f)
        ids = d["input_ids"]
        seen += 1
        if len(ids) == 0 or len(ids) > max_length or len(d["asst_pos"]) == 0:
            continue   # filter FIRST (trajectories are long; smoke max_len would drop all if capped before filtering)
        rows.append({"input_ids": ids.tolist(), "asst_pos": d["asst_pos"].tolist(),
                     "tk_ids": d["tk_ids"].tolist(), "tk_logprobs": d["tk_logprobs"].astype("float32").tolist()})
        if max_samples and len(rows) >= max_samples:   # then cap to N that actually FIT
            break
    logger.info("loaded %d cached trajectories (<= %d tokens) from %d scanned", len(rows), max_length, seen)
    if len(rows) == 0:
        raise ValueError(f"0 trajectories <= {max_length} tokens in {cache_dir} — raise --max_length")
    return Dataset.from_list(rows)


def collate(features):
    f = features[0]  # batch=1
    return {
        "input_ids": torch.tensor([f["input_ids"]], dtype=torch.long),
        "attention_mask": torch.ones((1, len(f["input_ids"])), dtype=torch.long),
        "asst_pos": torch.tensor(f["asst_pos"], dtype=torch.long),
        "tk_ids": torch.tensor(f["tk_ids"], dtype=torch.long),
        "tk_logprobs": torch.tensor(f["tk_logprobs"], dtype=torch.float32),
    }


class KDTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        pos = inputs["asst_pos"]; tk_ids = inputs["tk_ids"]; tk_lp = inputs["tk_logprobs"]
        # logits_to_keep=1 → model computes its lm_head for just 1 position instead of the full
        # (1,L,248K) logits (~63GB at L≈128K) we never use; we take hidden_states + apply our own
        # lm_head at assistant positions below. output_hidden_states still returns all positions.
        out = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                    output_hidden_states=True, use_cache=False, logits_to_keep=1)
        h = out.hidden_states[-1][0]                       # (L, H)
        pred = (pos - 1).clamp(min=0).to(h.device)
        hp = h.index_select(0, pred)                       # (M, H) only assistant rows
        lm_head = model.get_output_embeddings()
        tk_ids = tk_ids.to(h.device); tk_lp = tk_lp.to(h.device)
        # Chunk the (M,V) student logits + full-vocab log_softmax over assistant positions so the
        # transient never exceeds CHUNK×V (was OOMing at M≈30K on DR9K); gradient-checkpoint each
        # chunk to bound backward memory too. Exact same mean top-20 KL over all M positions.
        M = hp.shape[0]; CHUNK = 2048
        def _chunk_loss(hp_c, id_c, lp_c):
            lg = lm_head(hp_c.to(lm_head.weight.dtype)).float()          # (m, V)
            return topk_forward_kl_gathered(lg, id_c, lp_c) * hp_c.shape[0]
        total = hp.new_zeros(())
        for c in range(0, M, CHUNK):
            total = total + checkpoint(_chunk_loss, hp[c:c+CHUNK], tk_ids[c:c+CHUNK], tk_lp[c:c+CHUNK],
                                       use_reentrant=False)
        loss = total / max(M, 1)
        if not hasattr(self, "_logged0"):
            self._logged0 = True
            logger.info("STEP0 GUARD: KD loss=%.4f  M=%d  finite=%s", loss.item(), len(pos), torch.isfinite(loss).item())
            assert torch.isfinite(loss) and loss.item() > 0, "KD loss not finite/positive — aborting"
        return (loss, out) if return_outputs else loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer", default=None,
                   help="tokenizer path; defaults to --model. Set separately when --model is a "
                        "trained ckpt dir that lacks tokenizer files (e.g. iter-2 init from iter-1).")
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_length", type=int, default=262144)
    p.add_argument("--max_samples", type=int, default=0, help=">0 → smoke on a subset")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--max_steps", type=int, default=-1, help=">0 → smoke (few steps)")
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--sp_size", type=int, default=1)
    p.add_argument("--dp_shard_size", type=int, default=8)
    p.add_argument("--save_steps", type=int, default=20)
    p.add_argument("--wandb_project", type=str, default=None, help="enable wandb logging under this project")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    use_wandb = bool(args.wandb_project)
    if use_wandb:
        import os
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    pcfg = None
    if args.sp_size > 1:
        pcfg = ParallelismConfig(sp_backend="deepspeed", sp_size=args.sp_size,
                                 dp_shard_size=args.dp_shard_size,
                                 sp_handler=DeepSpeedSequenceParallelConfig(
                                     sp_seq_length_is_variable=True, sp_attn_implementation="flash_attention_2"))

    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model, trust_remote_code=True)
    ds = load_cache(args.cache_dir, args.max_length, args.max_samples)

    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    is_vlm = hasattr(cfg, "vision_config") or hasattr(cfg, "text_config")
    Cls = AutoModelForImageTextToText if is_vlm else AutoModelForCausalLM
    model = Cls.from_pretrained(args.model, attn_implementation="flash_attention_2",
                                torch_dtype="bfloat16", trust_remote_code=True)
    if is_vlm:
        for n, pr in model.named_parameters():
            if not ("language_model" in n or "lm_head" in n):
                pr.requires_grad = False

    targs = TrainingArguments(
        output_dir=args.output_dir, parallelism_config=pcfg,
        per_device_train_batch_size=1, gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs, max_steps=args.max_steps,
        learning_rate=args.lr, warmup_ratio=0.05, lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": 0.1}, bf16=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1, save_steps=args.save_steps, save_strategy="steps",
        save_total_limit=3, save_only_model=True,
        report_to="wandb" if use_wandb else "none",
        run_name=Path(args.output_dir).name,
        remove_unused_columns=False, dataloader_num_workers=2,
    )
    trainer = KDTrainer(model=model, args=targs, train_dataset=ds, data_collator=collate)
    logger.info("Starting KD training; output -> %s", args.output_dir)
    trainer.train()
    trainer.save_model(args.output_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
