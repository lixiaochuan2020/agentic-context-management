"""HistoryManager — message-list manager for the agent loop.

The runner is responsible for compressing ranges (manage_context) and
tracking summary boundaries; HistoryManager just stores the message list
and exposes range-based mutation primitives.

No automatic [Round N] tagging. No _idx field. Compression is by list
position (slice index), not by global round number.
"""
from __future__ import annotations

import copy


class HistoryManager:
    """Plain message-list container.

    - self.messages: live list, fed to the model
    - self.raw_messages: append-only mirror, never compressed (for traj dumps)
    """

    def __init__(self):
        self.messages: list[dict] = []
        self.raw_messages: list[dict] = []
        # Set by the runner before manage_context returns; consumed by the
        # token-trajectory bookkeeping after the next generate().
        self._tokens_before_compression: int | None = None

    # ── Append ──────────────────────────────────────────────

    def add_message(self, role: str, content: str) -> None:
        entry = {"role": role, "content": content}
        self.messages.append(entry)
        self.raw_messages.append(copy.deepcopy(entry))

    def add_message_dict(self, msg: dict) -> None:
        """Append a structured message (may carry tool_calls / tool_call_id)."""
        entry = dict(msg)  # shallow copy
        self.messages.append(entry)
        self.raw_messages.append(copy.deepcopy(entry))

    # ── Read ────────────────────────────────────────────────

    def get_messages_for_model(self, token_hint: int | None = None) -> list[dict]:
        """Return messages for model input.

        When token_hint is not None, append a "[CURRENT CONTEXT TOKEN: N]"
        marker to the *last tool message* (shallow-copied; self.messages is
        not mutated). The marker becomes part of the tool_response body the
        model reads, so it is a SIGNAL the agent consumes when deciding
        whether to compress context — not text the assistant should emit.
        When there is no tool message in the history yet (e.g. first turn,
        no tool call has been made), nothing is injected.
        """
        result = []
        for m in self.messages:
            out = {"role": m["role"], "content": m.get("content")}
            if "tool_calls" in m:
                out["tool_calls"] = m["tool_calls"]
            if "tool_call_id" in m:
                out["tool_call_id"] = m["tool_call_id"]
            # Only the deepseek-v4 client path sets this key (see client.py);
            # forwarding it here is a no-op for other models.
            if "reasoning_content" in m:
                out["reasoning_content"] = m["reasoning_content"]
            result.append(out)

        if token_hint is not None:
            for i in range(len(result) - 1, -1, -1):
                if result[i]["role"] == "tool":
                    base = result[i].get("content") or ""
                    suffix = f"\n\n[CURRENT CONTEXT TOKEN: {token_hint}]"
                    result[i] = dict(result[i])
                    result[i]["content"] = base + suffix
                    break

        return result

    def get_raw_history(self) -> list[dict]:
        """Return full append-only history (deep-cleaned of internal fields)."""
        result = []
        for m in self.raw_messages:
            out = {"role": m["role"], "content": m.get("content")}
            if "tool_calls" in m:
                out["tool_calls"] = m["tool_calls"]
            if "tool_call_id" in m:
                out["tool_call_id"] = m["tool_call_id"]
            result.append(out)
        return result

    # ── Compression ─────────────────────────────────────────

    def compress_range(self, start_pos: int, end_pos: int) -> list[dict]:
        """Remove self.messages[start_pos:end_pos] and return a deep copy.

        Half-open interval [start_pos, end_pos). Out-of-range or empty
        slices return [] without mutation.
        """
        if start_pos < 0:
            start_pos = 0
        if end_pos > len(self.messages):
            end_pos = len(self.messages)
        if start_pos >= end_pos:
            return []
        removed = copy.deepcopy(self.messages[start_pos:end_pos])
        del self.messages[start_pos:end_pos]
        return removed

    # ── Snapshot / Restore ─────────────────────────────────

    def snapshot(self) -> list[dict]:
        return copy.deepcopy(self.messages)

    def load_snapshot(self, snapshot: list[dict]) -> None:
        self.messages = copy.deepcopy(snapshot)
        self.raw_messages = copy.deepcopy(snapshot)

    @classmethod
    def from_saved_history(cls, history: list[dict],
                           raw_history: list[dict] | None = None) -> "HistoryManager":
        """Rebuild a HistoryManager from a previously saved message list.

        `history` populates `self.messages` (the live, compress-mutable view).
        `raw_history`, if given, populates `self.raw_messages` (the
        append-only archive) — used when resuming a rollout whose pre-loaded
        live messages have already had compression applied off-trajectory
        (e.g. teacher-guided iter rollouts) and we want the raw archive to
        retain the original pre-compress turns.
        """
        hm = cls()
        for m in history:
            hm.messages.append(dict(m))
        for m in (raw_history if raw_history is not None else history):
            hm.raw_messages.append(copy.deepcopy(m))
        return hm

    # ── Misc ────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.messages)

    def __repr__(self) -> str:
        return f"HistoryManager(messages={len(self.messages)})"
