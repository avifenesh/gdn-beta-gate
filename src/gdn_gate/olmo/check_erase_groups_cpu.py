#!/usr/bin/env python3
"""CPU check for vllm_arm's erase_g0..g3 arms (2026-10-09): a fake gating function stands in for vLLM's fused kernel;
24 fake GDN layers (distinct A_log tensors) are called in depth order twice (profile run, then a forward). Each group
arm must clamp beta at 1 in exactly its 6 layers and leave the other 18 bitwise unchanged; stock and erase keep their
behavior; a 25th layer fails loudly. Run in the vLLM venv (imports vllm on CPU)."""
import os
import sys

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402
from vllm.model_executor.layers.mamba.gdn import olmo_gdn_linear_attn as G  # noqa: E402

import vllm_arm  # noqa: E402


def fake(A_log, a, b, dt_bias, allow_neg_eigval=False, *args, **kw):
    beta = torch.sigmoid(b.float()) * (2.0 if allow_neg_eigval else 1.0)
    return a.float(), beta


G.fused_olmo_hybrid_gdn_gating = fake
vllm_arm._STATE["orig"] = None
torch.manual_seed(0)
A = [torch.randn(4) for _ in range(24)]
b = torch.full((8, 4), 3.0)  # 2 sigmoid(3) = 1.905 > 1 everywhere
a = torch.zeros(8, 4)
ok = True
for arm in vllm_arm.ARMS:
    vllm_arm.LAYERS.clear()
    fn = vllm_arm.install(arm)
    for rep in range(2):
        clamped = []
        for k in range(24):
            _, beta = fn(A[k], a, b, None, True)
            clamped.append(bool((beta <= 1.0).all()))
    if arm == "stock":
        want = [False] * 24
    elif arm == "half":
        want = [True] * 24
    elif arm == "erase":
        want = [True] * 24
    else:
        g = int(arm[-1])
        want = [k // 6 == g for k in range(24)]
    good = clamped == want
    ok &= good
    print(arm, "OK" if good else f"FAIL {clamped}")
vllm_arm.LAYERS.clear()
fn = vllm_arm.install("erase_g1")
try:
    for k in range(25):
        fn(A[k] if k < 24 else torch.randn(4), a, b, None, True)
    print("red arm FAIL: 25th layer accepted"); ok = False
except AssertionError as e:
    print("red arm OK:", e)
print("ALL OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
