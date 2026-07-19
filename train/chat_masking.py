"""Assistant-token loss masking for Qwen3.5 ChatML, derived from delimiters.

The Qwen3.5 chat template ships with NO `{% generation %}` block, so
`apply_chat_template(return_assistant_tokens_mask=True)` returns an all-zero mask
(and `assistant_only_loss=True` then silently degrades to full-sequence loss —
training on the user question + retrieved documents, not just the model's output).

We instead mark assistant-generated tokens directly from the ChatML structure:
every token after `<|im_start|>assistant\\n` up to and including the closing
`<|im_end|>`. This is the single source of truth for both SFT label masking
(train/sft_hf.py) and OPD teacher scoring (distill/score_teacher_logprobs.py).
"""
from __future__ import annotations

IGNORE_INDEX = -100


def chatml_consts(tok) -> dict:
    """Token ids needed to find assistant spans (derived from the tokenizer, not hardcoded)."""
    return {"ims": tok.convert_tokens_to_ids("<|im_start|>"),
            "ime": tok.convert_tokens_to_ids("<|im_end|>"),
            "asst": tok.encode("assistant", add_special_tokens=False)[0],
            "nl": tok.encode("\n", add_special_tokens=False)[0]}


def assistant_positions(ids: list[int], c: dict) -> list[int]:
    """Indices of assistant-generated tokens: content after '<|im_start|>assistant\\n'
    through the closing '<|im_end|>' (inclusive)."""
    out, i, n = [], 0, len(ids)
    while i < n:
        if ids[i] == c["ims"] and i + 1 < n and ids[i + 1] == c["asst"]:
            j = i + 2
            while j < n and ids[j] != c["nl"]:
                j += 1
            j += 1
            while j < n and ids[j] != c["ime"]:
                out.append(j)
                j += 1
            if j < n:
                out.append(j)  # include the closing <|im_end|>
            i = j + 1
        else:
            i += 1
    return out


def build_labels(ids: list[int], c: dict, ignore_index: int = IGNORE_INDEX) -> list[int]:
    """Causal-LM labels masked to assistant tokens: ids at assistant positions, ignore_index elsewhere."""
    pos = set(assistant_positions(ids, c))
    return [ids[i] if i in pos else ignore_index for i in range(len(ids))]
