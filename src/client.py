"""
Model clients: ABC interface + concrete implementations.

- BaseClient: abstract interface for runners
- LocalClient: local HuggingFace Qwen3 inference (lazy-loads torch/transformers)
- LiteLLMClient: online models via litellm (lazy-loads litellm)
"""
from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════
# BaseClient ABC
# ══════════════════════════════════════════════════════════

class BaseClient(ABC):
    """Abstract interface for model clients used by runners."""

    @abstractmethod
    def generate(
        self,
        messages: list[dict],
        tools: list | None = None,
        max_new_tokens: int = 2048,
    ) -> str:
        """Generate a response, return text content."""
        ...

    @abstractmethod
    def parse_tool_call(self, response_text: str) -> dict | None:
        """
        Parse tool call from the most recent generate().

        Returns:
            {"name": str, "arguments": dict} or None
        """
        ...

    @abstractmethod
    def count_tokens(
        self,
        messages: list[dict],
        tools: list | None = None,
    ) -> int:
        """Count tokens for messages (+ optional tools)."""
        ...

    @abstractmethod
    def build_assistant_message(self, response_text: str) -> dict:
        """
        Build assistant message dict.

        LocalClient: {"role": "assistant", "content": text}
        LiteLLMClient: {"role": "assistant", "content": text, "tool_calls": [...]}
        """
        ...

    @abstractmethod
    def build_tool_message(self, content: str) -> dict:
        """
        Build tool result message dict.

        LocalClient: {"role": "tool", "content": content}
        LiteLLMClient: {"role": "tool", "content": content, "tool_call_id": "..."}
        """
        ...


# ══════════════════════════════════════════════════════════
# LocalClient — local HuggingFace Qwen3 inference
# ══════════════════════════════════════════════════════════

class LocalClient(BaseClient):
    """Local HuggingFace model inference client.

    Heavy dependencies (torch, transformers) are imported lazily in __init__.
    """

    def __init__(self, model_path: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch

        logger.info("Loading tokenizer from %s ...", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        logger.info("Loading model from %s ...", model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        logger.info("Model loaded successfully on %s", self.model.device)

    def generate(
        self,
        messages: list[dict],
        tools: list | None = None,
        max_new_tokens: int = 2048,
    ) -> str:
        torch = self._torch

        encoded = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if isinstance(encoded, torch.Tensor):
            input_ids = encoded.to(self.model.device)
        else:
            input_ids = encoded["input_ids"].to(self.model.device)

        prompt_len = input_ids.shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        new_ids = output_ids[0][prompt_len:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)

    def count_tokens(
        self,
        messages: list[dict],
        tools: list | None = None,
    ) -> int:
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors=None,
        )
        if hasattr(encoded, "input_ids"):
            input_ids = encoded["input_ids"]
            if isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
                input_ids = input_ids[0]
        else:
            input_ids = encoded
        return len(input_ids)

    @staticmethod
    def parse_tool_call(response_text: str) -> dict | None:
        """Parse Qwen3 tool call: <tool_call>{"name":..., "arguments":...}</tool_call>"""
        match = re.search(
            r"<tool_call>(.*?)</tool_call>", response_text, re.DOTALL
        )
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                logger.warning(
                    "Failed to parse tool call JSON: %s",
                    match.group(1).strip()[:200],
                )
                return None
        return None

    def build_assistant_message(self, response_text: str) -> dict:
        return {"role": "assistant", "content": response_text}

    def build_tool_message(self, content: str) -> dict:
        return {"role": "tool", "content": content}


# ══════════════════════════════════════════════════════════
# LiteLLMClient — online models (Claude / GPT) via litellm
# ══════════════════════════════════════════════════════════

# Retry config
MAX_RETRIES = 5
RETRY_BASE_DELAY = 10
RETRY_MAX_DELAY = 120

# Custom pricing (per million tokens)
_CUSTOM_PRICING = {
    "gpt-5.5":      {"input": 5.00,  "cached_input": 0.50,  "output": 30.00},
    "gpt-5.4":      {"input": 2.50,  "cached_input": 0.25,  "output": 15.00},
    "gpt-5.4-mini": {"input": 0.75,  "cached_input": 0.075, "output": 4.50},
    "gpt-5.4-nano": {"input": 0.20,  "cached_input": 0.02,  "output": 1.25},
    "gpt-5.4-pro":  {"input": 30.00, "cached_input": None,  "output": 180.00},
}


class LiteLLMClient(BaseClient):
    """Online model client with native structured tool_calls.

    litellm is imported lazily in __init__.
    """

    def __init__(self, model: str, api_base: str | None = None):
        import litellm as _litellm
        _litellm.modify_params = True

        # Register custom pricing
        for _model_name, _prices in _CUSTOM_PRICING.items():
            _info = {
                "input_cost_per_token": _prices["input"] / 1_000_000,
                "output_cost_per_token": _prices["output"] / 1_000_000,
            }
            if _prices.get("cached_input") is not None:
                _info["cache_read_input_token_cost"] = _prices["cached_input"] / 1_000_000
            _litellm.register_model({_model_name: {"litellm_provider": "openai", **_info}})

        self._litellm = _litellm
        self.model = model
        # DeepSeek thinking-mode models (deepseek-v4*) require the prior
        # turn's reasoning_content to be echoed back on each request, or
        # the API rejects with 400. Scope this round-trip to that family
        # only — other providers/models are untouched.
        self._is_thinking_deepseek = "deepseek-v4" in model.lower()
        # Per-client api_base override. Both clients pass this explicitly:
        # the agent reads $AGENT_API_BASE, the summarizer reads $SUMMARIZER_API_BASE.
        # No implicit env fallback (no $OPENAI_BASE_URL coupling).
        self._api_base = api_base
        self._last_prompt_tokens: int | None = None
        self._last_tool_call: dict | None = None
        self._last_raw_tool_calls: list | None = None
        self._last_reasoning_content: str | None = None
        self._last_text: str = ""
        self._last_finish_reason: str | None = None
        self._session_cost: float = 0.0
        self._total_cost: float = 0.0
        self._last_completion_tokens: int = 0
        self._total_output_tokens: int = 0
        logger.info("LiteLLMClient initialized: model=%s api_base=%s", model, api_base or "<env default>")

    # ── Generation ───────────────────────────────────────

    def generate(
        self,
        messages: list[dict],
        tools: list | None = None,
        max_new_tokens: int = 2048,
        tool_choice: dict | str | None = None,
    ) -> str:
        prepared = self._prepare_messages(messages)

        kwargs = {
            "model": self.model,
            "messages": prepared,
            "max_tokens": max_new_tokens,
        }
        if self._api_base:
            kwargs["api_base"] = self._api_base
        if tools:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

        response = self._call_with_retry(**kwargs)

        if response.usage:
            self._last_prompt_tokens = response.usage.prompt_tokens
            ct = response.usage.completion_tokens or 0
            self._last_completion_tokens = ct
            self._total_output_tokens += ct

        call_cost = self._compute_cost(response)
        self._session_cost += call_cost
        self._total_cost += call_cost

        choice = response.choices[0]
        msg = choice.message

        finish_reason = getattr(choice, "finish_reason", None)
        self._last_finish_reason = finish_reason
        if finish_reason and finish_reason != "stop" and finish_reason != "tool_calls":
            logger.warning(
                "generate() finish_reason=%s (may indicate truncation)", finish_reason,
            )

        self._last_text = msg.content or ""
        self._last_reasoning_content = None
        if self._is_thinking_deepseek:
            rc = getattr(msg, "reasoning_content", None)
            if rc:
                self._last_reasoning_content = rc
        if msg.tool_calls:
            tc = msg.tool_calls[0]
            # vLLM's qwen3_xml parser occasionally emits malformed JSON in the
            # arguments string (e.g. trailing "Extra data"); don't kill the
            # whole rollout on that — keep `_last_tool_call` consistent and
            # let `parse_all_tool_calls()` below also drop malformed entries.
            try:
                _first_args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(
                    "generate(): first tool_call arguments not valid JSON "
                    "(%s); name=%s args[:200]=%r",
                    e, tc.function.name, str(tc.function.arguments)[:200],
                )
                _first_args = {}
            self._last_tool_call = {
                "name": tc.function.name,
                "arguments": _first_args,
            }
            self._last_raw_tool_calls = [
                {
                    "id": t.id,
                    "type": "function",
                    "function": {
                        "name": t.function.name,
                        "arguments": t.function.arguments,
                    },
                }
                for t in msg.tool_calls
            ]
        else:
            self._last_tool_call = None
            self._last_raw_tool_calls = None

        return self._last_text

    def _compute_cost(self, response) -> float:
        """Compute cost from token usage and _CUSTOM_PRICING, fallback to litellm."""
        if self.model in _CUSTOM_PRICING and response.usage:
            prices = _CUSTOM_PRICING[self.model]
            pt = response.usage.prompt_tokens or 0
            ct = response.usage.completion_tokens or 0
            cached = getattr(response.usage, "prompt_tokens_details", None)
            cached_tokens = getattr(cached, "cached_tokens", 0) if cached else 0
            non_cached = pt - (cached_tokens or 0)
            cost = non_cached * prices["input"] / 1_000_000
            if cached_tokens and prices.get("cached_input") is not None:
                cost += cached_tokens * prices["cached_input"] / 1_000_000
            cost += ct * prices["output"] / 1_000_000
            return cost
        try:
            return self._litellm.completion_cost(completion_response=response)
        except Exception:
            return 0.0

    @property
    def last_finish_reason(self) -> str | None:
        return self._last_finish_reason

    def get_and_reset_cost(self) -> float:
        """Return accumulated cost since last reset, then reset session counter."""
        cost = self._session_cost
        self._session_cost = 0.0
        return cost

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def total_output_tokens(self) -> int:
        return self._total_output_tokens

    def snapshot_output_tokens(self) -> int:
        """Return current total_output_tokens for delta computation by runners."""
        return self._total_output_tokens

    # ── Tool Call parsing ────────────────────────────────

    def parse_tool_call(self, response_text: str) -> dict | None:
        """Return cached first tool_call (does not parse text)."""
        return self._last_tool_call

    def parse_all_tool_calls(self) -> list[dict]:
        """Return all tool calls from the last generate(), each with id/name/arguments.

        Tool calls with malformed JSON arguments are skipped with a warning
        (the model occasionally emits invalid JSON, especially under strict
        prompt constraints). Returning a possibly-shorter list is preferable
        to crashing the whole rollout.
        """
        if not self._last_raw_tool_calls:
            return []
        result = []
        for tc in self._last_raw_tool_calls:
            args_str = tc["function"]["arguments"]
            try:
                args = json.loads(args_str)
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(
                    "parse_all_tool_calls: dropping tool_call with malformed args "
                    "(%s): name=%s args[:200]=%r",
                    e, tc["function"].get("name"), str(args_str)[:200],
                )
                continue
            result.append({
                "id": tc["id"],
                "name": tc["function"]["name"],
                "arguments": args,
            })
        return result

    def build_truncated_tool_pair(self, response_text: str) -> tuple[dict, dict]:
        """Build fake tool_call + error pair for truncated responses (finish_reason=length)."""
        import uuid
        fake_tc_id = f"tc_truncated_{uuid.uuid4().hex[:8]}"
        assistant_msg = {
            "role": "assistant",
            "content": response_text or None,
            "tool_calls": [{
                "id": fake_tc_id,
                "type": "function",
                "function": {
                    "name": "search",
                    "arguments": json.dumps({"query": "[truncated]"}),
                },
            }],
        }
        if self._is_thinking_deepseek and self._last_reasoning_content:
            assistant_msg["reasoning_content"] = self._last_reasoning_content
        tool_msg = {
            "role": "tool",
            "content": "Error: your previous tool call was truncated due to output length limit. "
                       "Please make a shorter, more focused search query, or provide your answer directly.",
            "tool_call_id": fake_tc_id,
        }
        return assistant_msg, tool_msg

    # ── Token counting ───────────────────────────────────

    def count_tokens(
        self,
        messages: list[dict],
        tools: list | None = None,
    ) -> int:
        try:
            return self._litellm.token_counter(model=self.model, messages=messages)
        except Exception:
            if self._last_prompt_tokens is not None:
                return self._last_prompt_tokens
            return 0

    # ── Message building ─────────────────────────────────

    def build_assistant_message(self, response_text: str) -> dict:
        msg: dict = {"role": "assistant", "content": response_text or ""}
        if self._last_raw_tool_calls:
            msg["tool_calls"] = self._last_raw_tool_calls
        if self._is_thinking_deepseek and self._last_reasoning_content:
            msg["reasoning_content"] = self._last_reasoning_content
        return msg

    def build_tool_message(self, content: str, tool_call_id: str | None = None) -> dict:
        if tool_call_id is None:
            tool_call_id = "tc_unknown"
            if self._last_raw_tool_calls:
                tool_call_id = self._last_raw_tool_calls[0]["id"]
        return {
            "role": "tool",
            "content": content,
            "tool_call_id": tool_call_id,
        }

    # ── Retry logic ──────────────────────────────────────

    def _call_with_retry(self, **kwargs) -> object:
        litellm = self._litellm
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return litellm.completion(**kwargs)
            except (litellm.RateLimitError, litellm.ServiceUnavailableError, litellm.APIConnectionError,
                    litellm.InternalServerError, litellm.Timeout) as e:
                if attempt == MAX_RETRIES:
                    logger.error("Transient error: max retries (%d) exhausted", MAX_RETRIES)
                    raise
                delay = min(RETRY_BASE_DELAY * (2 ** (attempt - 1)), RETRY_MAX_DELAY)
                logger.warning(
                    "Transient error (attempt %d/%d), retrying in %ds: %s",
                    attempt, MAX_RETRIES, delay, e,
                )
                time.sleep(delay)
        raise RuntimeError("Retry loop exited unexpectedly")

    # ── Bedrock message preparation ──────────────────────

    @staticmethod
    def _prepare_messages(messages: list[dict]) -> list[dict]:
        """Fix Bedrock Converse API requirements: strict alternation, orphaned tool_use/result pairs."""
        prepared = [dict(m) for m in messages]

        for _round in range(10):
            changed = False

            # Strip orphaned tool_calls (no matching tool_result in next message)
            for i, m in enumerate(prepared):
                tcs = m.get("tool_calls")
                if not tcs:
                    continue
                next_tcids = set()
                if i + 1 < len(prepared):
                    tcid = prepared[i + 1].get("tool_call_id")
                    if tcid:
                        next_tcids.add(tcid)
                surviving = [tc for tc in tcs if tc.get("id") in next_tcids]
                if len(surviving) < len(tcs):
                    changed = True
                    if surviving:
                        m["tool_calls"] = surviving
                    else:
                        m.pop("tool_calls", None)

            # Strip orphaned tool_results (no matching tool_call in previous message)
            for i, m in enumerate(prepared):
                tcid = m.get("tool_call_id")
                if not tcid:
                    continue
                matched = False
                if i > 0:
                    for tc in prepared[i - 1].get("tool_calls", []):
                        if tc.get("id") == tcid:
                            matched = True
                            break
                if not matched:
                    changed = True
                    m.pop("tool_call_id", None)
                    m["role"] = "user"

            # Merge consecutive same-role messages
            merged: list[dict] = []
            for msg in prepared:
                eff_role = "user" if msg["role"] == "tool" else msg["role"]

                if not merged:
                    merged.append(dict(msg))
                    continue

                prev = merged[-1]
                prev_eff = "user" if prev["role"] == "tool" else prev["role"]

                if eff_role == prev_eff and eff_role == "user":
                    prev_c = prev.get("content") or ""
                    cur_c = msg.get("content") or ""
                    prev["content"] = (prev_c + "\n\n" + cur_c).strip() if prev_c else cur_c
                    if msg.get("tool_call_id"):
                        prev["tool_call_id"] = msg["tool_call_id"]
                        prev["role"] = "tool"
                elif eff_role == prev_eff and eff_role == "assistant":
                    prev_c = prev.get("content") or ""
                    cur_c = msg.get("content") or ""
                    if cur_c:
                        prev["content"] = (prev_c + "\n" + cur_c).strip() if prev_c else cur_c
                    cur_tcs = msg.get("tool_calls")
                    if cur_tcs:
                        prev["tool_calls"] = (prev.get("tool_calls") or []) + cur_tcs
                else:
                    merged.append(dict(msg))

            if len(merged) != len(prepared):
                changed = True
            prepared = merged

            if not changed:
                break

        return prepared


# ── Backward compatibility aliases ───────────────────────
ModelClient = LocalClient
