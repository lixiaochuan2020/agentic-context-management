"""Top-K forward-KL distillation loss for offline on-policy distillation.

At each assistant position we have the teacher's top-K (token_id, logprob) from the cache.
The student forward gives logits over the full vocab. Loss = KL(teacher_topk || student),
with the teacher distribution renormalized over its top-K and the student's logprobs gathered
at those same K token ids.

Alignment (the easy thing to get wrong): a causal LM's logits at position p-1 predict the
token at position p. The cached teacher dist at `pos` is the distribution FOR the token at
`pos` (vLLM prompt_logprobs[pos]). So student logits[pos-1] ↔ teacher topk at pos.
"""
from __future__ import annotations

import torch


def topk_forward_kl(logits: torch.Tensor, kd_pos: torch.Tensor,
                    tk_ids: torch.Tensor, tk_logprobs: torch.Tensor) -> torch.Tensor:
    """logits (L,V) student; kd_pos (M,) assistant token indices; tk_ids/tk_logprobs (M,K) teacher.
    Returns scalar mean KL(teacher_topk || student) over the M assistant positions."""
    pred = (kd_pos - 1).clamp(min=0)                  # student row predicting token at kd_pos
    s = logits[pred].log_softmax(dim=-1)              # (M, V)
    s_k = torch.gather(s, -1, tk_ids)                 # (M, K) student logprob at teacher's top-K
    t_lp = tk_logprobs.log_softmax(dim=-1)            # (M, K) teacher dist over its top-K (renormalized)
    t = t_lp.exp()
    kl = (t * (t_lp - s_k)).sum(dim=-1)               # (M,)
    return kl.mean()


def topk_forward_kl_gathered(logits_mv: torch.Tensor, tk_ids: torch.Tensor,
                             tk_logprobs: torch.Tensor) -> torch.Tensor:
    """Same loss, but logits_mv (M,V) are already the student logits at the M *predicting* rows
    (i.e. lm_head applied only at assistant positions → avoids materializing (L,248K) logits)."""
    s = logits_mv.log_softmax(dim=-1)
    s_k = torch.gather(s, -1, tk_ids)
    t_lp = tk_logprobs.log_softmax(dim=-1)
    t = t_lp.exp()
    return (t * (t_lp - s_k)).sum(dim=-1).mean()


def _selftest():
    torch.manual_seed(0)
    L, V, M, K = 12, 50, 4, 5
    kd_pos = torch.tensor([3, 5, 8, 11])
    tk_ids = torch.randint(0, V, (M, K))
    tk_logprobs = torch.randn(M, K)                   # arbitrary teacher top-K logits

    # Case 1: build student logits so the predicting rows put EXACTLY the teacher's renormalized
    # top-K mass on those K tokens (others -inf) → KL should be ~0.
    logits = torch.full((L, V), -1e9)
    t_lp = tk_logprobs.log_softmax(-1)
    for m, p in enumerate(kd_pos):
        logits[p - 1, tk_ids[m]] = t_lp[m]
    kl0 = topk_forward_kl(logits, kd_pos, tk_ids, tk_logprobs).item()

    # Case 2: random student → KL clearly positive
    kl1 = topk_forward_kl(torch.randn(L, V), kd_pos, tk_ids, tk_logprobs).item()
    print(f"matched-student KL={kl0:.6f} (want ~0)  |  random-student KL={kl1:.4f} (want >0)")
    assert abs(kl0) < 1e-4, kl0
    assert kl1 > 0.1, kl1
    print("kd_loss self-test PASSED")


if __name__ == "__main__":
    _selftest()
