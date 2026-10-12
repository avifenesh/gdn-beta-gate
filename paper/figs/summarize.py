"""Print a compact summary of every step 12/14 analysis receipt used by the paper (CPU only)."""
import json, sys
from pathlib import Path
R = Path(__file__).resolve().parents[2] / "results"
FILES = sys.argv[1:] or [
    "ml7/step14/analysis_gsm8k.json", "ml7/step14/analysis_math500.json",
    "ml7/step14_ext/analysis_gsm8k.json", "ml7/step14_ext/analysis_math500.json",
    "ml7_4b/step14/analysis_gsm8k.json", "ml7_4b/step14/analysis_math500.json",
    "ml7_4b/step14_ctrlpm/analysis_gsm8k.json", "ml7_4b/step14_ctrlpm/analysis_math500.json",
]
for f in FILES:
    d = json.load(open(R / f))
    print("=====", f, "n_items", d["n_items"], "caps", d["caps"])
    for name in ("baseline", "warmup"):
        b = d[name]
        print(f"  {name:9s} area {b['norm_auc_log2']:.4f} nat {b['natural_len']:.0f} cut {b['cut_off_share_at_max']:.3f} acc {b['acc']}")
    for arm, a in d["arms"].items():
        print(f"  {arm:9s} seeds {a['seeds']} area {a['norm_auc_log2']['seed_mean']:.4f} {[round(x,4) for x in a['norm_auc_log2']['per_seed']]}"
              f" nat {a['natural_len']['seed_mean']:.0f} {[round(x) for x in a['natural_len']['per_seed']]}"
              f" cut {a['cut_off_share_at_max']['seed_mean']:.3f} {a['cut_off_share_at_max']['per_seed']}"
              f" acc8k {a['acc']['8192']['seed_mean']:.3f} acc256 {a['acc']['256']['seed_mean']:.3f}")
    for blk in ("effects", "length_effects"):
        for c, e in d.get(blk, {}).items():
            print(f"  {blk[:6]} {c:22s} {e['seed_mean_diff']:+.4f} seeds {[round(x,4) for x in e['per_seed_diff']]} ci [{e['ci95'][0]:+.4f},{e['ci95'][1]:+.4f}] {e['verdict']}")
    for h, v in d.get("preregistered_hypotheses", {}).items():
        print("  ", h, v.get("verdict"))
