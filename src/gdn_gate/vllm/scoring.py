"""Continuation log-likelihoods on vLLM 0.31 without a full-prompt log-softmax.

prompt_logprobs=k computes a full-vocabulary log-softmax at EVERY prompt row; MMLU-Pro cloze has 10.2M prompt tokens
but only 1.2M option tokens, so the first step-2 run spent >30 min inside one part (workstation, 2026-10-08). vLLM 0.31's
`prompt_logprob_token_ids` + `prompt_logprob_start` score a fixed id set (here: the option's own distinct ids, at most
89 <= 128) only at rows >= start, each normalised over the full vocabulary (compute_token_logprobs on the row's
logits), which is what HF does at the option positions.
"""
from __future__ import annotations


def cont_logprobs(llm, ctxs, conts, SamplingParams, TokensPrompt):
    """For each (context ids, continuation ids): the logprob of every continuation token given everything before it."""
    reqs, params, idsets = [], [], []
    for c, o in zip(ctxs, conts):
        ids = sorted(set(o))
        idsets.append(ids)
        reqs.append(TokensPrompt(prompt_token_ids=list(c) + list(o)))
        params.append(SamplingParams(max_tokens=1, temperature=0.0, detokenize=False,
                                     prompt_logprob_token_ids=ids, prompt_logprob_start=len(c) - 1))
    outs = llm.generate(reqs, params, use_tqdm=False)
    res = []
    for o, cont, ids, out in zip(conts, conts, idsets, outs):
        S = out.prompt_token_id_logprobs
        if S is None or len(S) != len(cont):
            raise RuntimeError(f"prompt_token_id_logprobs rows {None if S is None else len(S)} != {len(cont)}")
        col = {t: j for j, t in enumerate(ids)}
        res.append([float(S[i][col[t]]) for i, t in enumerate(cont)])
    return res
