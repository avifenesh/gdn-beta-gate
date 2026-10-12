#!/usr/bin/env python3
"""Fail closed unless rl_common.correct grades through math_verify on this venv.

rl_common.correct falls back to a normalized string match when math_verify cannot be imported, and swallows the
error, so a missing package silently marks '18.00', '$18', '18 dollars' and '1,000' wrong (ml_pipeline step 4,
grader_check.json). This check must fail when math_verify is absent (red arm) and pass when it is present.
Usage: preflight_grader.py BROAD_DIR
"""
import sys

sys.path.insert(0, sys.argv[1])
import rl_common as R  # noqa: E402

import math_verify  # noqa: E402,F401  (ImportError here = FAIL)

accept = [("\\boxed{\\$18}", "18"), ("\\boxed{18.00}", "18"), ("\\boxed{1,000}", "1000"), ("\\boxed{18}", "18"),
          ("\\boxed{\\dfrac{14}{3}}", "\\frac{14}{3}")]
reject = [("\\boxed{19}", "18"), ("18", "18"), ("\\boxed{}", "18"), ("\\boxed{18} then \\boxed{19}", "18")]
bad = [a for a in accept if not R.correct(*a)] + [r for r in reject if R.correct(*r)]
if bad:
    sys.exit(f"preflight_grader: FAIL {bad}")
print("preflight_grader: OK (math_verify", getattr(math_verify, "__version__", "?"), ")")
