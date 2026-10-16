"""Tier A of tests/run_cpu_checks.sh as pytest tests: every receipt the paper reads, rebuilt or checked from released
data on CPU. Run: uv run pytest tests/ (a few minutes)."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    r = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r.stdout


def test_claims():
    assert "0 failed" in run("paper/figs/check_claims.py")


def test_environment_summary():
    run("scripts/collect_env.py", "--check")


def test_rl_analyses_rerun():
    out = run("scripts/rerun_analysis.py")
    assert out.count("OK") == 16, out


def test_ruler_pairing_rerun():
    assert run("scripts/rerun_ruler_pairing.py").count("OK") == 4


def test_mechanism_summaries_rerun():
    assert run("scripts/rerun_mech.py").count("OK") == 4


def test_olmo_nll_summary(tmp_path):
    run("src/gdn_gate/olmo/analyze_nll.py", "--run", "results/olmo", "--out", tmp_path / "nll.json")
    assert json.loads((tmp_path / "nll.json").read_text()) == json.loads(
        (ROOT / "results/olmo/lesion_nll_base_summary.json").read_text())


def test_derive_is_reproducible(tmp_path):
    keep = tmp_path / "derived.json"
    shutil.copyfile(ROOT / "paper/data/derived.json", keep)
    try:
        run("paper/figs/derive.py")
        assert (ROOT / "paper/data/derived.json").read_bytes() == keep.read_bytes()
    finally:
        shutil.copyfile(keep, ROOT / "paper/data/derived.json")


def test_extra_stats_are_reproducible(tmp_path):
    keep = tmp_path / "extra.json"
    shutil.copyfile(ROOT / "paper/data/extra.json", keep)
    try:
        run("paper/figs/stats_extra.py")
        assert (ROOT / "paper/data/extra.json").read_bytes() == keep.read_bytes()
    finally:
        shutil.copyfile(keep, ROOT / "paper/data/extra.json")
