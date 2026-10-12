"""P9: beta-range arms for Olmo-Hybrid in vLLM 0.31.0, no kernel source change.

vLLM computes the GDN gates in vllm/model_executor/layers/mamba/gdn/olmo_gdn_linear_attn.py:
    g, beta = fused_olmo_hybrid_gdn_gating(self.A_log, a, b, self.dt_bias, self.allow_neg_eigval)
with beta = sigmoid(b) (x2 when linear_allow_neg_eigval) in fp32. install(arm) replaces that module-level function
with a wrapper, so every GDN layer's call goes through it:
  stock   the original call, unchanged
  half    the original kernel with allow_neg_eigval=False: beta = sigmoid(b), what the config flag off gives
  erase   the original call, then beta.clamp_(max=1.0): min(2 sigmoid(b), 1), the erase-only clamp
  erase_g0..erase_g3  the erase clamp in one depth quarter of the GDN layers only (GDN layers 0-5, 6-11, 12-17, 18-23
          in forward order; 2026-10-09 localization of the counting cost). A layer is identified by its A_log
          parameter: the k-th distinct A_log.data_ptr() seen is GDN layer k (the profile run calls every layer in
          depth order, and CUDA-graph capture passes the same parameters). LAYERS records the order.
The wrapper must run in the engine process: callers set VLLM_ENABLE_V1_MULTIPROCESSING=0 before importing vllm
(install() refuses otherwise).

Loader fix (2026-10-07): vLLM 0.31.0 cannot load any Olmo-Hybrid checkpoint (Base and Think-SFT ship the same 523
tensor names). OlmoHybridModel.hf_to_vllm_mapper maps {q,k,v}_conv1d to (conv1d, 0|1|2); WeightsMapper.apply passes
that shard id as a tensor attribute (`loaded_weight.shard_id`), which the linear layers' load_weights read, but the
fused conv1d parameter is loaded through AutoWeightsLoader._load_param, which calls weight_loader(param, w) with no
shard id, and _make_fused_conv1d_weight_loader indexes dims[None] -> TypeError. patch_conv1d_loader() wraps that
factory (module-level name, looked up at layer construction) so the loader takes the shard id from the tensor
attribute when the argument is None, and fails loudly if both are missing. LOADER counts the calls (expect 3 per GDN
layer, 72 for the 24-layer model). No site-packages file is edited. Engagement: CALLS counts wrapper calls (eager and CUDA-graph capture), BETA_MAX keeps
the largest beta returned while STATS is on (eager runs only; graph replays do not enter Python).

Position encoding (2026-10-08): vLLM 0.31.0 runs Think-SFT as NoPE (its `rope_parameters: null`), which is correct;
transformers 5.17 ran it with RoPE, the whole 0.16 vLLM-HF p9 gap. Fixed on the HF side (patch_olmo.fix_position_encoding).
"""
from __future__ import annotations

import os

ARMS = ("stock", "half", "erase", "erase_g0", "erase_g1", "erase_g2", "erase_g3")
GROUP = 6  # GDN layers per depth quarter (24 GDN layers)
LAYERS = {}  # A_log.data_ptr() -> GDN layer index, in first-seen order
CALLS = {"n": 0}
STATS = {"on": False, "beta_max": 0.0, "gt1": 0, "n": 0}
_STATE = {"arm": None, "orig": None, "conv_factory": None}
LOADER = {"n": 0, "shards": {}}


def patch_conv1d_loader(G):
    """Make the fused conv1d weight loader read the stacked shard id from the tensor attribute. Idempotent."""
    if _STATE["conv_factory"] is not None:
        return
    orig_factory = G._make_fused_conv1d_weight_loader
    _STATE["conv_factory"] = orig_factory

    def factory(dims, tp_size, tp_rank):
        inner = orig_factory(dims, tp_size, tp_rank)

        def weight_loader(param, loaded_weight, loaded_shard_id=None):
            if loaded_shard_id is None:
                loaded_shard_id = getattr(loaded_weight, "shard_id", None)
            if not isinstance(loaded_shard_id, int):
                raise ValueError(f"fused conv1d loader: no integer shard id ({loaded_shard_id!r})")
            LOADER["n"] += 1
            LOADER["shards"][loaded_shard_id] = LOADER["shards"].get(loaded_shard_id, 0) + 1
            return inner(param, loaded_weight, loaded_shard_id)

        return weight_loader

    G._make_fused_conv1d_weight_loader = factory


def install(arm: str):
    assert arm in ARMS, arm
    assert os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") == "0", "the wrapper must live in the engine process"
    import vllm
    from vllm.model_executor.layers.mamba.gdn import olmo_gdn_linear_attn as G

    assert vllm.__version__ == "0.31.0", vllm.__version__
    patch_conv1d_loader(G)
    if _STATE["orig"] is None:
        _STATE["orig"] = G.fused_olmo_hybrid_gdn_gating
    orig = _STATE["orig"]

    def gating(A_log, a, b, dt_bias, allow_neg_eigval=False, *args, **kw):
        assert allow_neg_eigval, "the arms are defined relative to the shipped (0, 2) model"
        CALLS["n"] += 1
        if _STATE["arm"] == "half":
            g, beta = orig(A_log, a, b, dt_bias, False, *args, **kw)
        else:
            g, beta = orig(A_log, a, b, dt_bias, True, *args, **kw)
            if _STATE["arm"] == "erase":
                beta.clamp_(max=1.0)
            elif _STATE["arm"].startswith("erase_g"):
                k = LAYERS.setdefault(A_log.data_ptr(), len(LAYERS))
                assert k < 4 * GROUP, f"more GDN layers than expected: {k + 1}"
                if k // GROUP == int(_STATE["arm"][-1]):
                    beta.clamp_(max=1.0)
        if STATS["on"] and not _capturing():
            STATS["beta_max"] = max(STATS["beta_max"], float(beta.max()))
            STATS["gt1"] += int((beta > 1.0).sum())
            STATS["n"] += beta.numel()
            k = LAYERS.setdefault(A_log.data_ptr(), len(LAYERS))
            pl = STATS.setdefault("per_layer", {}).setdefault(k, [0, 0])
            pl[0] += int((beta > 1.0).sum())
            pl[1] += beta.numel()
        return g, beta

    _STATE["arm"] = arm
    G.fused_olmo_hybrid_gdn_gating = gating
    patch_state_cache(G)
    return gating


# Recurrent-state cache (2026-10-08): fp32, as gate_vllm.engine_kwargs does for Qwen3.5; vLLM's default "auto" keeps
# the GDN state in the model dtype (bf16) across decode steps, and the think battery decodes up to 32k tokens. The
# fp32 delta-rule experiment of the same day was negative (FINDINGS 18:50: 0.013 vs a 0.16 vLLM-HF gap) and is gone.
def patch_state_cache(G=None):
    if _STATE.get("cache_patched"):
        return
    import vllm

    orig_init = vllm.LLM.__init__

    def init(self, *a, **kw):
        kw.setdefault("mamba_ssm_cache_dtype", "float32")
        # cloze scoring asks prompt_logprob_token_ids for an option's distinct ids (vllm/scoring.py); vLLM's default
        # max_logprobs 20 refused 26 on the first Olmo mmlu run (2026-10-08 21:13)
        kw["max_logprobs"] = max(int(kw.get("max_logprobs") or 0), 256)
        return orig_init(self, *a, **kw)

    vllm.LLM.__init__ = init
    _STATE["cache_patched"] = True


def _capturing() -> bool:
    import torch

    return torch.cuda.is_current_stream_capturing()
