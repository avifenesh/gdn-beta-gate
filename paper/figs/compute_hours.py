"""Logged GPU time per device class for the compute statement. CPU only.

Sums (a) the last elapsed_s, or the sum of step_s, of every training log under the runs tree and (b) timing.wall_s
of every readout JSON under results and the runs tree. The device comes from the file's own env.gpu when
it records one, else from the run directory's tag (gb10 / box / rig / spark). Jobs that left no timing log (HF
scoring, Olmo lesions, lm-eval batteries, stopped pilots without a log) are not counted, so owned-hardware totals are
lower bounds. Rented totals in the paper come from box hours, not from this script.
The run trees are not released: in this repository the script documents how compute_hours.json was made (WG_RUNS points it at a run tree).
Writes paper/data/compute_hours.json. Run: python3 paper/figs/compute_hours.py
"""
import json
import os
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
RUNS = Path(os.environ.get("WG_RUNS", "runs"))
# Forgetting-recipe runs (Section E): the rented box (4B and 9B LoRA and Gate(0,2)) and the RTX 5090 learning-rate pilots.
RETRO = RUNS / "retrofit"
OUT = ROOT / "paper/data/compute_hours.json"
# pilots, cap pilots, warm-ups and smoke runs (a subset of "train", reported separately)
PILOT = re.compile(r"pilot|cap1024|cap2048|_fmt_|smoke")


def dev_of(gpu: str | None, path: Path) -> str:
    s = (gpu or "").lower()
    p = str(path).lower()
    if "gb10" in s:
        return "GB10"
    if "5090" in s:
        return "RTX 5090 Laptop"
    if "pro 6000" in s:
        return "RTX PRO 6000"
    if "gb10" in p or "/spark/" in p:
        return "GB10"
    if "_box" in p or "/box" in p or "box2" in p:
        return "RTX PRO 6000"
    if "/rig/" in p or "_rig" in p:
        return "RTX 5090 Laptop"
    return "unknown"


def env_gpu(d: dict) -> str | None:
    for k in ("env", "meta"):
        e = d.get(k)
        if isinstance(e, dict):
            if isinstance(e.get("gpu"), str):
                return e["gpu"]
            ee = e.get("env")
            if isinstance(ee, dict) and isinstance(ee.get("gpu"), str):
                return ee["gpu"]
    return None


def train_seconds(path: Path) -> float:
    if path.suffix == ".jsonl":
        rows = []
        for line in open(path):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    else:
        rows = json.load(open(path))
        if not isinstance(rows, list):
            return 0.0
    if any("step_s" in r for r in rows):
        return float(sum(r.get("step_s", 0.0) for r in rows))
    el = [r["elapsed_s"] for r in rows if "elapsed_s" in r]
    return float(max(el)) if el else 0.0


# Rented boxes: (lane phase, GPUs, hours, the FINDINGS.md string the hours come from). Hours are box time, so they
# include staging, idle time and stopped or repeated runs.
RENTED = [
    ("Retrofit cloud 8x, 2026-09-29 21:09 to 09-30 07:31 Israel", 8, 10 + 22 / 60, "21:09 to 07:31 IDT"),
    ("Phase 1-2 cloud 8x, 01:40-11:08 Israel", 8, (11 + 8 / 60) - (1 + 40 / 60), "on-demand 05:29-11:08 about <price>"),
    ("Phase 3 cloud 8x, 13:29-23:45 Israel", 8, 10.3, "about 10.3 h x <price>"),
    ("Phase 4 4x", 4, 3.66, "~3.66 h at <price>"),
    ("Phase 5 cloud 4x spot", 4, 4.9, "~4.9 h x <price>"),
    ("Phase 5 cloud 8x spot", 8, 2.2, "~2.2 h x <price>"),
]


def main():
    tot = defaultdict(float)
    n = defaultdict(int)
    for p in list(RUNS.rglob("train_log.jsonl")) + list(RUNS.rglob("train_log.json")):
        gpu = None
        for mf in ("meta.json", "result.json"):
            m = p.parent / mf
            if m.exists():
                try:
                    gpu = env_gpu(json.load(open(m))) or gpu
                except Exception:  # noqa: BLE001
                    pass
        d = dev_of(gpu, p)
        sec = train_seconds(p)
        tot[("train", d)] += sec
        n[("train", d)] += 1
        if PILOT.search(str(p.parent)):
            tot[("train_pilot", d)] += sec
            n[("train_pilot", d)] += 1
    # The retrofit box: every train log and the general battery timings of each run (analysis_root holds copies).
    for p in sorted((RETRO / "box").rglob("train_log.json")):
        if "analysis_root" in p.parts:
            continue
        sec = train_seconds(p)
        tot[("train", "RTX PRO 6000")] += sec
        n[("train", "RTX PRO 6000")] += 1
    for p in sorted((RETRO / "box").rglob("general.json")):
        if "analysis_root" in p.parts:
            continue
        g = json.load(open(p))
        tot[("readout", "RTX PRO 6000")] += float(g.get("mmlu_seconds", 0.0)) + float(g.get("humaneval_seconds", 0.0))
        n[("readout", "RTX PRO 6000")] += 1
    # The retrofit learning-rate pilots on the RTX 5090 Laptop: a separate key, outside the device total.
    for sub in ("sweep", "long", "diag"):
        for p in sorted((RETRO / sub).rglob("train_log.json")):
            tot[("retrofit_pilot", "RTX 5090 Laptop")] += train_seconds(p)
            n[("retrofit_pilot", "RTX 5090 Laptop")] += 1
    seen = set()
    for base in (RES, RUNS):
        for p in base.rglob("*.json"):
            if p.name.startswith("analysis") or p.stat().st_size < 2000:
                continue
            try:
                d = json.load(open(p))
            except Exception:  # noqa: BLE001
                continue
            if not isinstance(d, dict) or not str(d.get("schema", "")).startswith("wg-readout"):
                continue
            w = (d.get("timing") or {}).get("wall_s")
            if not w:
                continue
            key = (p.name, round(float(w), 1))
            if key in seen:  # the same readout copied into two trees
                continue
            seen.add(key)
            dv = dev_of(env_gpu(d), p)
            tot[("readout", dv)] += float(w)
            n[("readout", dv)] += 1
    out = {"note": __doc__.strip().splitlines()[0], "hours": {}, "files": {}}
    for (kind, dv), s in sorted(tot.items()):
        out["hours"].setdefault(dv, {})[kind] = s / 3600
        out["files"].setdefault(dv, {})[kind] = n[(kind, dv)]
    for dv, h in out["hours"].items():
        h["total"] = sum(v for k, v in h.items() if k in ("train", "readout"))
    out["rented"] = {"boxes": [{"what": w, "gpus": g, "hours": h, "gpu_hours": g * h, "findings_string": f}
                               for w, g, h, f in RENTED],
                     "gpu_hours_total": sum(g * h for _, g, h, _ in RENTED)}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
