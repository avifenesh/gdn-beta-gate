"""Shared pieces of step 7 (length-cost GRPO) and its readout: prompts, grading, vendor sampling, and capped
thinking generation with budget forcing.

Thinking generation (rollouts and readout alike): the chat template with enable_thinking=True ends the prompt with
"<think>\\n". Phase 1 samples up to `cap` tokens. A sequence that closed its thinking (</think>) and ended
(<|im_end|>) inside the cap is final. A sequence still thinking at the cap gets "\\n</think>\\n\\n" appended (forced,
not sampled: excluded from the policy loss) and phase 2 samples up to `answer_max` answer tokens. Reasoning tokens =
tokens before </think>. Sampling follows the Qwen3.5 card for thinking mode, text tasks: temperature 1.0, top_p 0.95,
top_k 20, presence_penalty 1.5 (applied here as a logits processor over the completion's own tokens).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field

import torch

SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
VENDOR = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5}


def boxed(text: str) -> str | None:
    """Content of the last \\boxed{...} (balanced braces)."""
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth, k = 0, j
    while k < len(text):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1 : k]
        k += 1
    return None


def gold_of(row: dict) -> str:
    if "answer" in row:
        return row["answer"]
    g = boxed(row["solution"])
    assert g is not None, row["id"]
    return g


def correct(pred_text: str, gold: str) -> bool:
    b = boxed(pred_text)
    if b is None:
        return False
    try:
        from math_verify import parse, verify

        return bool(verify(parse(f"${gold}$"), parse(f"\\boxed{{{b}}}")))
    except Exception:  # noqa: BLE001  (fallback: normalised string match)
        norm = lambda s: re.sub(r"\\left|\\right|\\!|\s|\\dfrac|\\tfrac", lambda m: "\\frac" if "frac" in m.group() else "", s).rstrip(".")  # noqa: E731
        return norm(b) == norm(gold)


class PresencePenalty:
    """OpenAI-style presence penalty: subtract `p` from the logit of every token already in the completion."""

    def __init__(self, p: float, prompt_len: int, prior: list[list[int]] | None = None):
        """prior: per row, completion tokens that sit in the context (a forced continuation's earlier reasoning), so
        the penalty covers the whole completion as a serving engine's would, not just this call's new tokens."""
        self.p, self.L, self.prior = p, prompt_len, prior
        self.base = None

    def __call__(self, input_ids, scores):
        if not self.p:
            return scores
        if self.base is None:
            self.base = torch.zeros_like(scores, dtype=torch.bool)
            for k, pr in enumerate(self.prior or []):
                if pr:
                    self.base[k, torch.tensor(pr, device=scores.device)] = True
        mask = self.base.clone()
        if input_ids.shape[1] > self.L:
            mask.scatter_(1, input_ids[:, self.L :], True)
        return scores - self.p * mask.to(scores.dtype)


@dataclass
class Rollout:
    prompt: list[int]
    think: list[int]                      # sampled reasoning tokens (phase 1, up to </think> exclusive)
    tail: list[int] = field(default_factory=list)   # tokens after the reasoning, sampled or forced
    tail_sampled: list[bool] = field(default_factory=list)
    text: str = ""
    forced: bool = False
    reasoning_tokens: int = 0

    @property
    def completion(self):
        return self.think + self.tail

    @property
    def sampled_mask(self):
        return [True] * len(self.think) + self.tail_sampled


class Thinker:
    def __init__(self, tok):
        self.tok = tok
        self.end_think = tok.convert_tokens_to_ids("</think>")
        self.im_end = tok.convert_tokens_to_ids("<|im_end|>")
        self.force = tok.encode("\n</think>\n\n", add_special_tokens=False)
        # <|im_end|>, the tokenizer eos, and <|endoftext|> (the text config's eos) all end a completion
        self.stops = sorted({self.im_end, tok.eos_token_id, tok.convert_tokens_to_ids("<|endoftext|>")})
        assert self.end_think in self.force

    def prompt(self, problem: str) -> list[int]:
        msgs = [{"role": "user", "content": problem + SUFFIX}]
        return self.tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=True, tokenize=True,
                                            return_dict=False)

    @torch.no_grad()
    def _sample(self, model, prompts, max_new, dev, sampling, greedy=False, prior=None):
        from transformers import LogitsProcessorList

        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        L = max(len(p) for p in prompts)
        ids = torch.full((len(prompts), L), pad, dtype=torch.long)
        att = torch.zeros_like(ids)
        for k, p in enumerate(prompts):
            ids[k, L - len(p):] = torch.tensor(p)
            att[k, L - len(p):] = 1
        kw = dict(max_new_tokens=max_new, eos_token_id=self.stops, pad_token_id=pad)
        if greedy:
            kw["do_sample"] = False
        else:
            kw.update(do_sample=True, temperature=sampling["temperature"], top_p=sampling["top_p"], top_k=sampling["top_k"],
                      logits_processor=LogitsProcessorList([PresencePenalty(sampling["presence_penalty"], L, prior)]))
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            out = model.generate(input_ids=ids.to(dev), attention_mask=att.to(dev), **kw)
        res = []
        for k in range(len(prompts)):
            g = out[k, L:].tolist()
            # trim padding after the stop token (generate pads finished rows)
            for i, t in enumerate(g):
                if t in self.stops:
                    g = g[: i + 1]
                    break
            res.append(g)
        return res

    def generate(self, model, prompts, cap, answer_max, dev, sampling=VENDOR, greedy=False, max_batch=64):
        """Capped thinking with budget forcing. Returns one Rollout per prompt (prompts may repeat)."""
        outs = []
        for s in range(0, len(prompts), max_batch):
            ps = prompts[s : s + max_batch]
            g1 = self._sample(model, ps, cap, dev, sampling, greedy)
            rolls, need = [], []
            for p, g in zip(ps, g1):
                if self.end_think in g:
                    i = g.index(self.end_think)
                    r = Rollout(p, g[:i], g[i:], [True] * (len(g) - i))
                    if g[-1] not in self.stops:
                        need.append(len(rolls))  # thinking closed but the answer ran into the cap: finish it
                else:
                    if g and g[-1] in self.stops:
                        g = g[:-1]  # stopped inside its thinking (a silent stop): drop the stop, force the close
                    r = Rollout(p, g, list(self.force), [False] * len(self.force), forced=True)
                    need.append(len(rolls))
                r.reasoning_tokens = len(r.think)
                rolls.append(r)
            if need:
                g2 = self._sample(model, [rolls[i].prompt + rolls[i].completion for i in need], answer_max, dev,
                                  sampling, greedy, prior=[rolls[i].completion for i in need])
                for i, g in zip(need, g2):
                    rolls[i].tail += g
                    rolls[i].tail_sampled += [True] * len(g)
            for r in rolls:
                r.text = self.tok.decode(r.tail, skip_special_tokens=True)
            outs += rolls
        return outs


def length_reward(correct_flags, lengths, alpha):
    """Correctness-gated length penalty (Arora and Zanette 2025, arXiv 2502.04463): within a prompt's group,
    r = 1[correct] * (1 - alpha * sigmoid((len - mean_c) / std_c)), mean and std over the group's correct samples.
    Incorrect samples get 0 whatever their length."""
    cl = [l for c, l in zip(correct_flags, lengths) if c]
    if not cl:
        return [0.0] * len(lengths)
    mu = sum(cl) / len(cl)
    sd = math.sqrt(sum((l - mu) ** 2 for l in cl) / len(cl))
    out = []
    for c, l in zip(correct_flags, lengths):
        if not c:
            out.append(0.0)
        else:
            z = (l - mu) / sd if sd > 0 else 0.0
            out.append(1.0 - alpha * (1.0 / (1.0 + math.exp(-z))))
    return out


def load_jsonl(p):
    return [json.loads(l) for l in open(p)]
