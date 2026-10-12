#!/usr/bin/env python3
"""fresh_table.py RES_DIR: markdown tables from the fresh-set analyses (analysis_fresh_4b.json, analysis_fresh_2b.json):
per arm seed means (area, natural length, accuracy at 8,192, cut-off share), the hypotheses, the reported contrasts and
the per-source descriptive areas. Prints to stdout (for FINDINGS.md)."""
import json
import sys
from pathlib import Path


def f(x, nd=3):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def main():
    R = Path(sys.argv[1])
    for rec in ("4b", "2b"):
        p = R / f"analysis_fresh_{rec}.json"
        if not p.exists():
            continue
        a = json.loads(p.read_text())
        print(f"\n{rec.upper()} (seeds {a['fresh']['seeds']}, {a['n_items']} items x {a['samples_per_item']} samples)\n")
        print("| arm | area | area per seed | natural length | length per seed | acc 8,192 | cut-off at 8,192 |")
        print("|---|---|---|---|---|---|---|")
        b = a["baseline"]
        print(f"| untouched | {f(b['norm_auc_log2'])} | - | {b['natural_len']:.0f} | - | {f(b['acc']['8192'])} | {f(b['cut_off_share_at_max'])} |")
        if a.get("warmup"):
            w = a["warmup"]
            print(f"| warm-up | {f(w['norm_auc_log2'])} | - | {w['natural_len']:.0f} | - | {f(w['acc']['8192'])} | {f(w['cut_off_share_at_max'])} |")
        for arm, v in a["arms"].items():
            ps = " / ".join(f"{x:.3f}" for x in v["norm_auc_log2"]["per_seed"])
            pl = " / ".join(f"{x:.0f}" for x in v["natural_len"]["per_seed"])
            print(f"| {arm} | {f(v['norm_auc_log2']['seed_mean'])} | {ps} | {v['natural_len']['seed_mean']:.0f} | {pl} | "
                  f"{f(v['acc']['8192']['seed_mean'])} | {f(v['cut_off_share_at_max']['seed_mean'])} |")
        print()
        for h in ("H5", "H6"):
            r = a["preregistered_hypotheses"][h]
            ps = ", ".join(f"{x:+.4f}" if abs(x) < 10 else f"{x:+.0f}" for x in r.get("per_seed_diff", []))
            ci = r.get("ci95") or [None, None]
            print(f"- {h} ({r['role']}): {r['contrast']} {r['metric']} = {r.get('seed_mean_diff'):+.4f} "
                  f"(seeds {ps}; CI [{ci[0]:.4f}, {ci[1]:.4f}]): {r['verdict']}")
        print("\nContrasts (rule verdicts):")
        for blk, name in (("effects", "area"), ("length_effects", "natural length")):
            for k, e in a[blk].items():
                print(f"- {name} {k}: {e['seed_mean_diff']:+.4f} CI [{e['ci95'][0]:.4f}, {e['ci95'][1]:.4f}] "
                      f"seeds {[round(x, 4) for x in e['per_seed_diff']]}: {e['verdict']}")
        print("\nPer source (descriptive, area / natural length seed means):")
        for src, v in a["per_source_descriptive"].items():
            parts = [f"untouched {f(v['baseline_norm_auc_log2'])}"]
            parts += [f"{arm} {f(x['norm_auc_log2']['seed_mean'])} / {x['natural_len']['seed_mean']:.0f}" for arm, x in v["arms"].items()]
            print(f"- {src} ({v['n_items']} items): " + "; ".join(parts))


if __name__ == "__main__":
    main()
