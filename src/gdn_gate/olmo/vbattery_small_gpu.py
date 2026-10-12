#!/usr/bin/env python3
"""Small-GPU entry point for ../vllm/vbattery_olmo.py (unchanged scorers, tasks and receipt layout).

The 7B Olmo-Hybrid on the 23.9 GiB workstation has fewer Mamba cache blocks than vbattery_olmo's fixed max_num_seqs=128
(GDN needs one block per running sequence; see gate_vllm.py), so this wrapper overrides the engine's max_num_seqs
(and optionally max_model_len) from the environment: OLMO_VLLM_MAX_NUM_SEQS, OLMO_VLLM_MAX_MODEL_LEN. It sets the
same per-arm VLLM_CACHE_ROOT vbattery_olmo sets, before vLLM is imported, and records the overrides in the arm's
receipt.json (engine_overrides). Arguments are vbattery_olmo.py's.
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
arm = sys.argv[sys.argv.index("--arm") + 1] if "--arm" in sys.argv else "stock"
os.environ["VLLM_CACHE_ROOT"] = str(Path.home() / ".cache/gdn_gate/vllm/vllm_cache" / f"olmo_{arm}")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
sys.path.insert(0, str(HERE.parent / "vllm"))
import vbattery_olmo as VB  # noqa: E402

OV = {}
if os.environ.get("OLMO_VLLM_MAX_NUM_SEQS"):
    OV["max_num_seqs"] = int(os.environ["OLMO_VLLM_MAX_NUM_SEQS"])
if os.environ.get("OLMO_VLLM_MAX_MODEL_LEN"):
    OV["max_model_len"] = int(os.environ["OLMO_VLLM_MAX_MODEL_LEN"])


def _patch():
    import lm_eval.models.vllm_causallms as LV

    orig = LV.VLLM.__init__

    def init(self, *a, **kw):
        kw.update(OV)
        return orig(self, *a, **kw)

    LV.VLLM.__init__ = init


_orig_install = VB.vllm_arm.install


def install(a):  # vbattery_olmo.main calls vllm_arm.install(arm) before it imports lm_eval's vLLM class
    r = _orig_install(a)
    _patch()
    return r


VB.vllm_arm.install = install

# WG_DUMP_GEN=PATH (diagnostic, 2026-10-09 fda divergence): record the first generate call's first 4 prompts
# (token ids) and sampling params exactly as lm-eval hands them to vLLM.
if os.environ.get("WG_DUMP_GEN"):
    import vllm as _v

    _orig_gen = _v.LLM.generate

    def _gen(self, prompts, sampling_params=None, *a, **kw):
        p = os.environ.pop("WG_DUMP_GEN", None)
        if p:
            sps = sampling_params if isinstance(sampling_params, list) else [sampling_params]
            json.dump({"prompts": [list(x["prompt_token_ids"]) if isinstance(x, dict) else str(x)[:2000] for x in list(prompts)[:4]],
                       "sampling": [repr(s) for s in sps[:4]], "n": len(prompts)}, open(p, "w"))
        return _orig_gen(self, prompts, sampling_params, *a, **kw)

    _v.LLM.generate = _gen
if __name__ == "__main__":
    VB.main()
    out = None
    if "--out" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--out") + 1]) / arm / "receipt.json"
    if out and out.exists():
        r = json.load(open(out))
        r["engine_overrides"] = OV
        r["entry"] = "olmo/vbattery_small_gpu.py"
        json.dump(r, open(out, "w"), indent=1)
