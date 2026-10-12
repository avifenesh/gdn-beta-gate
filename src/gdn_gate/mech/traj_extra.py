"""traj_extra.py MECH_DIR: two derived views of MECH_DIR/traj.json (written by traj.py), saved as MECH_DIR/traj_extra.json.

natural_close_rate_by_window: per run, 1 - mean forced share over the step windows 0-10, 10-25, 25-50, 50-75, 75-100
and 100-150 (the share of rollouts that closed their reasoning inside the 512-token training cap).
cum_length_signal_groups_to_reach_natural_close: per run and threshold 0.05 / 0.1 / 0.2, the first step at which one
minus the 10-step rolling forced share reaches the threshold, and the number of length-signal groups (groups whose correct
samples differ in length) summed over steps 0 to that step.

Reconstructed for the release from the lab's inline computation; its output equals the lab's traj_extra.json."""
import json
import sys
from pathlib import Path

import numpy as np

WINDOWS = [(0, 10), (10, 25), (25, 50), (50, 75), (75, 100), (100, 150)]
THRESHOLDS = (0.05, 0.1, 0.2)


def roll(x, k=10):
    x = np.asarray(x, float)
    return np.array([x[max(0, i - k + 1): i + 1].mean() for i in range(len(x))])


def main():
    mech = Path(sys.argv[1] if len(sys.argv) > 1 else "results/mech")
    series = json.load(open(mech / "traj.json"))["series"]
    win = {f"{a}-{b}": {run: float(1 - np.mean(s["forced_frac"][a:b])) for run, s in series.items()} for a, b in WINDOWS}
    cum = {}
    for run, s in series.items():
        nat = 1 - roll(s["forced_frac"])
        cum[run] = {}
        for thr in THRESHOLDS:
            idx = np.where(nat >= thr)[0]
            if len(idx):
                i = int(idx[0])
                cum[run][str(thr)] = {"step": i, "cum_groups": int(sum(s["len_groups"][: i + 1]))}
            else:
                cum[run][str(thr)] = {"step": None, "cum_groups": None}
    out = {"natural_close_rate_by_window": win, "cum_length_signal_groups_to_reach_natural_close": cum}
    (mech / "traj_extra.json").write_text(json.dumps(out, indent=1))
    print("wrote", mech / "traj_extra.json")


if __name__ == "__main__":
    main()
