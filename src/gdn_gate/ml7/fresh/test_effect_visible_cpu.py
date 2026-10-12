#!/usr/bin/env python3
"""test_effect_visible_cpu.py: CPU test of the 2026-10-12 readout self-test amendment (owner approval: effect_visible 4.0
only for fresh-set readouts of the Qwen3.5-4B checkpoints ctrl_s1_cap512_r_box and wide_s2_cap512_r_box, step 150).

  E1  the two listed (4B checkpoint, test_fresh.jsonl) pairs get 4.0
  E2  red arms keep 5.0: the same checkpoints on any other data file (val, test_gsm8k, test_math500), the 2B checkpoints
      of the same arm and seed (ml7/grpo, r_gb10 and r_box), other 4B checkpoints on the fresh file, step-125 files
  E3  the visibility check flips only by the bar: with the measured ctrl_1 numbers (effect 0.0405, prefill floor mean
      0.008371) lora_visible is False at 5.0 and True at 4.0, as train_grpo_vllm.parity_check computes it
  E4  an agreement failure still fails under the override: train_grpo_vllm.agrees does not read effect_visible, and an
      effect with corr 0.5 (or slope 0.5) fails at 4.0 exactly as at 5.0
  E5  ledger: code_ledger.py accept walks the amended readout_vllm.py back to the 4B touch-2 lock sha256 9ef24b62...
Usage: python test_effect_visible_cpu.py (exit 0 = all pass)"""
import inspect
import os
import subprocess
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
ML7 = HERE.parent
sys.path.insert(0, str(ML7))
import readout_vllm as R  # noqa: E402

B = os.environ.get("WG_RUNS", "runs") + "/"
FRESH = B + "ml7/data/test_fresh.jsonl"
FAILS = []


def check(name, ok, why=""):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f": {why}"))
    if not ok:
        FAILS.append(name)


def main() -> int:
    c1 = B + "ml7_4b/grpo/ctrl_s1_cap512_r_box/ckpt_step150.pt"
    w2 = B + "ml7_4b/grpo/wide_s2_cap512_r_box/ckpt_step150.pt"
    check("E1 4B ctrl s1 on the fresh file -> 4.0", R.effect_visible_of(c1, FRESH) == 4.0)
    check("E1 4B wide s2 on the fresh file -> 4.0", R.effect_visible_of(w2, FRESH) == 4.0)
    reds = [(c1, B + "ml7/data/val.jsonl"), (c1, B + "ml7/data/test_gsm8k.jsonl"), (w2, B + "ml7/data/test_math500.jsonl"),
            (B + "ml7/grpo/ctrl_s1_cap512_r_gb10/ckpt_step150.pt", FRESH), (B + "ml7/grpo/wide_s2_cap512_r_gb10/ckpt_step150.pt", FRESH),
            (B + "ml7/grpo/ctrl_s4_cap512_r_box/ckpt_step150.pt", FRESH), (B + "ml7_4b/grpo/ctrl_s0_cap512_r_box/ckpt_step150.pt", FRESH),
            (B + "ml7_4b/grpo/wide1_s2_cap512_r_box/ckpt_step150.pt", FRESH), (B + "ml7_4b/grpo/ctrlpm_s1_cap512_r_box/ckpt_step150.pt", FRESH),
            (B + "ml7_4b/grpo/wide_s2_cap512_r_box/ckpt_step125.pt", FRESH), ("none", FRESH)]
    bad = [(c, d) for c, d in reds if R.effect_visible_of(c, d) != 5.0]
    check("E2 red arms keep 5.0 (other data, 2B, other 4B checkpoints, other steps)", not bad, bad)
    # E3/E4: the check formulas of train_grpo_vllm.parity_check (source asserted, then evaluated)
    import importlib.util  # noqa: PLC0415

    src = (ML7 / "train_grpo_vllm.py").read_text()
    check("E3 parity_check visibility formula as tested",
          'visible = a.effect_visible * max(res["floor_prefill"]["mean"], a.parity_floor)' in src
          and '"lora_visible": res["lora_effect"]["effect_mean"] >= visible' in src
          and '"lora_effect_agrees": agrees(res["lora_effect"], a)' in src)
    floor, eff = 0.008371146013894092, 0.04051765211266248
    vis = lambda x: eff >= x * max(floor, 1e-3)  # noqa: E731
    check("E3 measured ctrl_1: not visible at 5.0, visible at 4.0", (not vis(5.0)) and vis(4.0), (vis(5.0), vis(4.0)))
    spec = importlib.util.spec_from_file_location("tgv_src", ML7 / "train_grpo_vllm.py")
    tree = compile(src, "train_grpo_vllm.py", "exec")
    g: dict = {"__name__": "tgv_probe"}
    # evaluate only the agrees() definition (the module imports torch / vllm at call time, not needed here)
    import ast  # noqa: PLC0415

    mod = ast.parse(src)
    fn = next(n for n in mod.body if isinstance(n, ast.FunctionDef) and n.name == "agrees")
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "agrees", "exec"), g)
    agrees = g["agrees"]
    check("E4 agrees() does not read effect_visible", "effect_visible" not in inspect.getsource(agrees) if False else "effect_visible" not in ast.get_source_segment(src, fn))
    for vx in (5.0, 4.0):
        a = types.SimpleNamespace(effect_corr=0.9, effect_slope_tol=0.1, effect_visible=vx)
        good = {"slope": 0.95, "corr": 0.96, "effect_mean": eff, "engine_effect_mean": 0.0398, "err_mean": 0.011}
        badc = dict(good, corr=0.5)
        bads = dict(good, slope=0.5)
        check(f"E4 effect_visible {vx}: good agreement passes, corr 0.5 and slope 0.5 fail",
              agrees(good, a) and not agrees(badc, a) and not agrees(bads, a), (agrees(good, a), agrees(badc, a), agrees(bads, a)))
    p = subprocess.run([sys.executable, str(ML7 / "code_ledger.py"), "accept", "--file", str(ML7 / "readout_vllm.py"), "--want",
                        "9ef24b622b545ab0899ce78878f34cf857af8c125dfb3c05fd4a7d6514b34282"], capture_output=True, text=True)
    check("E5 ledger walks the amended file back to the touch-2 lock", p.returncode == 0, p.stdout + p.stderr)
    del spec, tree
    print(f"\n{'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
