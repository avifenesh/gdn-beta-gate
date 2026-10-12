"""Check every row of paper/CLAIMS.md against its receipt, then check every number in paper/main.tex against the rows.

Row kinds. json: value at the path, times scale, within tol of expect. text: the string occurs in the file.
textfull: the whole file, stripped of surrounding whitespace, equals the string (for one-value files).

main.tex scan. Every number in the document body (main text, tables, captions and appendices; comments, citation
keys, labels, references, URLs, typewriter names and table layout arguments removed) must match a verified value:
  - a json row's expect, within max(tol, half a unit of the last shown digit), read either as is or as a percent
    of a fraction (48 matches 0.48);
  - a number inside a text or textfull row's verified string;
  - or an entry of ALLOW below (years, budgets, model sizes, small integers used as counts or indices).
Signs are ignored (a difference written $-0.002$ matches expect -0.002 or 0.002).
Exit status 1 if any row fails or any main.tex number is unmatched. Run: python3 paper/figs/check_claims.py
"""
import gzip
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CL = ROOT / "paper/CLAIMS.md"
TEX = ROOT / "paper/main.tex"
SHORT = {
    "A4M": "results/ml7_4b/step14_ctrlpm/analysis_math500.json",
    "A4G": "results/ml7_4b/step14_ctrlpm/analysis_gsm8k.json",
    "A2M": "results/ml7/step14/analysis_math500.json",
    "A2G": "results/ml7/step14/analysis_gsm8k.json",
    "AXM": "results/ml7/step14_ext/analysis_math500.json",
    "AXG": "results/ml7/step14_ext/analysis_gsm8k.json",
    "OCP": "results/olmo/vllm2/counting_paired.json",
    "OLOC": "results/olmo/vllm2/counting_localization.json",
    "ORP": "results/olmo/vllm2/ruler_paired.json",
    "ONLL": "results/olmo/lesion_nll_base_summary.json",
    "VC32": "results/vcount/step6_32k_paired.json",
    "W1": "results/w1main/compare.json",
    "D": "paper/data/derived.json",
    "X": "paper/data/extra.json",
    "CH": "paper/data/compute_hours.json",
    "F4": "results/fresh/analysis_fresh_4b.json",
    "F2": "results/fresh/analysis_fresh_2b.json",
    "FFIN": "results/fresh/final.json",
    "PL2": "ml_pipeline/PIPELINE.md",
    "PL4": "ml_pipeline_4b/PIPELINE.md",
    "LEDG": "src/gdn_gate/ml7/amendments/ledger.json",
    "FIND": "results/FINDINGS.md",
    "G12": "results/mech/g1g2_summary.json",
    "GS": "results/mech/gate_stats_summary/gate_stats_summary_fixed.json",
    "TS": "results/mech/text_stats.json",
    "IM": "results/mech/item_match.json",
    "TJ": "results/mech/traj.json",
    "WS": "results/mech/w_stats.json",
}

# Numbers allowed without a row, each with its reason.
ALLOW = {
    # years and dates of named contests, releases and the census (dates of records are in text rows)
    **{y: "year" for y in (2021, 2022, 2023, 2024, 2025, 2026, 2027)},
    # reasoning budgets of the readout (set by design, Section 3) and the budget/context labels 1k..32k
    **{b: "budget" for b in (256, 512, 1024, 2048, 4096, 8192)},
    # small integers used as counts, seeds, indices, arm-name ceilings, degrees of freedom and equation terms
    **{i: "small integer" for i in range(0, 11)},
    13: "x_6 = log2 8192 in the area equation",
    12: "AMC 12A / 12B contest names",
    20: "rank of LoRA-r20 (arm name)",
    5090: "GPU model name (RTX 5090)",
    6000: "GPU model name (RTX PRO 6000)",
}
# number followed by these units/suffixes is a model size or a context/budget label, not a measured value
SIZE_SUFFIX = re.compile(r"^(?:\s?(?:B|M)\b|k\b|-?Base\b)")


def resolve(f):
    f = SHORT.get(f, f)
    p = Path(f).expanduser()
    return p if p.is_absolute() else ROOT / p


def read_text_any(p):
    """Released readouts are gzipped (results/**/readout_*.json.gz): read FILE or FILE.gz."""
    if not p.exists() and Path(str(p) + ".gz").exists():
        return gzip.open(str(p) + ".gz", "rt").read()
    return p.read_text()


def get(obj, path):
    for k in path.split("/"):
        obj = obj[int(k)] if isinstance(obj, list) else obj[k]
    return obj


NUM = re.compile(r"(?<![A-Za-z_\\\d.])(\d{1,3}(?:(?:,|\{,\})\d{3})+|\d+)(\.\d+)?")


def numbers_in(text):
    out = []
    for m in NUM.finditer(text):
        ip, fp = m.group(1).replace("{,}", "").replace(",", ""), m.group(2) or ""
        out.append((float(ip + fp), len(fp) - 1 if fp else 0, m.start(), m.end()))
    return out


def check_rows():
    cache, rows, bad = {}, [], []
    for line in CL.read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 8 or cells[2] not in ("json", "text", "textfull"):
            # A line that starts like a ledger row but does not parse is a broken row, not a skip.
            if re.match(r"^\|\s*[A-Z][A-Za-z]*\d+[a-z]?\s*\|", line):
                bad.append(f"malformed row ({len(cells)} cells): {line[:120]}")
            continue
        rid, text, kind, f, path, scale, expect, tol = cells
        p = resolve(f)
        try:
            if kind == "text":
                ok = path in read_text_any(p)
                got = "found" if ok else "missing"
            elif kind == "textfull":
                ok = read_text_any(p).strip() == path
                got = "equal" if ok else "differs"
            else:
                if p not in cache:
                    cache[p] = json.loads(read_text_any(p))
                v = float(get(cache[p], path)) * float(scale)
                ok = abs(v - float(expect)) <= float(tol) + 1e-12
                got = f"{v:.6g}"
        except Exception as e:  # noqa: BLE001
            ok, got = False, f"error {type(e).__name__}: {e}"
        rows.append((rid, kind, path, expect, tol, ok))
        if not ok:
            bad.append(f"{rid}: {text} | {f} {path} -> {got} (expect {expect})")
    return rows, bad


def tex_body():
    s = TEX.read_text()
    s = s[s.index(r"\begin{document}"):]
    s = re.sub(r"(?<!\\)%.*", "", s)
    s = s.replace(r"\bibliography{references}", "")
    s = s.replace(r"\times", " x ")
    for cmd in ("cite", "cref", "Cref", "ref", "label", "url", "nolinkurl", "texttt", "includegraphics", "begin",
                "end", "cmidrule", "setlength", "bibliographystyle", "multicolumn", "section", "subsection",
                "paragraph", "emph"):
        if cmd in ("multicolumn",):
            s = re.sub(r"\\multicolumn\{\d+\}\{[^}]*\}", "", s)
            continue
        if cmd in ("section", "subsection", "paragraph", "emph"):
            continue
        s = re.sub(r"\\" + cmd + r"(?:\[[^\]]*\])?(?:\([^)]*\))?\{[^{}]*\}(?:\{[^{}]*\})?", " ", s)
    s = re.sub(r"\\begin\{tabular\}\{[^}]*\}", " ", s)
    s = re.sub(r"\[[a-z]+\]", " ", s)  # float placement [t]
    s = re.sub(r"\[leftmargin=[^\]]*\]", " ", s)
    s = re.sub(r"\\(?:gone|gtwo)\b", " ", s)
    s = re.sub(r"[a-z]_\{?\d+\}?", " ", s)  # math subscripts x_1, x_{j+1}
    s = re.sub(r"\^\{?-?\d+\}?", " ", s)  # exponents 10^{-4}
    return s


def verified(rows):
    vals = []
    for rid, kind, path, expect, tol, ok in rows:
        if not ok:
            continue
        if kind == "json":
            vals.append((abs(float(expect)), float(tol), rid))
        else:
            for v, d, _, _ in numbers_in(path):
                vals.append((v, 0.0, rid))
    return vals


def matches(x, d, vals):
    half = 0.5 * 10 ** (-d)
    for v, tol, rid in vals:
        if abs(x - v) <= max(tol, half) + 1e-9:
            return rid
        if v <= 1.0 and abs(x / 100 - v) <= max(tol, half / 100) + 1e-9:
            return rid + "%"
    return None


def main():
    rows, bad = check_rows()
    print(f"{len(rows)} rows checked, {len(bad)} failed")
    for b in bad:
        print("FAIL", b)
    vals = verified(rows)
    if not TEX.exists():
        print("paper/main.tex is not in this repository: tex scan skipped")
        sys.exit(1 if bad else 0)
    body = tex_body()
    n, unmatched, allowed = 0, [], 0
    for x, d, a, b in numbers_in(body):
        n += 1
        if SIZE_SUFFIX.match(body[b:b + 6]):
            allowed += 1
            continue
        if d == 0 and x in ALLOW:
            allowed += 1
            continue
        if matches(x, d, vals) is None:
            ctx = body[max(0, a - 50):b + 30].replace("\n", " ")
            unmatched.append(f"{x:g} ... {ctx}")
    print(f"main.tex: {n} numbers scanned, {n - allowed - len(unmatched)} matched to rows, "
          f"{allowed} allowlisted, {len(unmatched)} unmatched")
    for u in unmatched:
        print("UNMATCHED", u)
    sys.exit(1 if bad or unmatched else 0)


if __name__ == "__main__":
    main()
