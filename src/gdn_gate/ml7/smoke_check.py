#!/usr/bin/env python3
"""Plugin-path smoke verdict for readout_vllm.py on a synthetic checkpoint (synth_ckpt.py): the unmerged LoRA written
on every arm (72 modules = 18 GDN layers x 4 projections) and the engine/trainer parity check passed, gate written and
engaged on wide (beta > 1 seen) and wide1 (beta never above 1), no gate on ctrl.
Usage: smoke_check.py DIR OUT.json  (DIR holds smoke_wide.json, smoke_wide1.json, smoke_ctrl.json)"""
import json
import sys
from pathlib import Path

d, out = Path(sys.argv[1]), sys.argv[2]
res, ok = {}, True
for arm in ("wide", "wide1", "ctrl"):
    f = d / f"smoke_{arm}.json"
    if not f.exists():
        res[arm] = {"pass": False, "why": "missing"}
        ok = False
        continue
    j = json.loads(f.read_text())
    par, calls, st = j.get("parity") or {}, j["gate_calls"], j["gate_stats"] or {}
    ck = par.get("sync") or {}
    checks = {"lora_unmerged_72": ck.get("lora_modules") == 72, "parity": bool(par.get("pass") and par.get("start_pass")),
              "rows": j["summary"]["n_rows"] == 8,
              "prior_presence_engaged": j["prior_presence"]["row_steps"] > 0 or j["summary"]["per_cap"]["256"]["forced_frac"] == 0}
    if arm == "ctrl":
        checks.update(engine_stock=j["engine"] == "stock", no_gate=ck.get("gate_layers") == 0 and calls["gate"] == 0)
    else:
        checks.update(gate_layers_18=ck.get("gate_layers") == 18, gate_called=calls["gate"] > 0)
        if arm == "wide":
            checks["beta_above_1_seen"] = st.get("beta_gt1", 0) > 0
        else:
            checks["beta_never_above_1"] = st.get("beta_n", 0) > 0 and st.get("beta_gt1", 1) == 0 and st.get("beta_max", 9) <= 1.0
    res[arm] = {"checks": checks, "pass": all(checks.values()), "acc": {c: v["acc"] for c, v in j["summary"]["per_cap"].items()},
                "gate_stats": st, "gate_calls": calls}
    ok &= res[arm]["pass"]
res["verdict"] = "PASS" if ok else "FAIL"
Path(out).write_text(json.dumps(res, indent=1))
print(json.dumps({k: (v["pass"] if isinstance(v, dict) else v) for k, v in res.items()}))
sys.exit(0 if ok else 1)
