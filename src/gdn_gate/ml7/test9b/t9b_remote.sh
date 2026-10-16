#!/usr/bin/env bash
# t9b_remote.sh list: runs on a Spark unit (synced with the ml7 tree). One line per job dir of the 9B AIME 2025 read
# under ~/gdn-gate/jobs: NAME LABEL STATUS CMD_SHA256. STATUS is the job's rc when it ended, "running" while its GPU
# telemetry file (written every 10 s by spark_job.sh) is under 10 minutes old, else "stale" (the job died unrecorded).
set -u
case "${1:-}" in
  list)
    cd "$HOME/gdn-gate/jobs" 2>/dev/null || exit 0
    for d in t9b-aime2025-*; do
      [ -d "$d" ] || continue
      lab=${d#t9b-aime2025-}; lab=${lab%-*}
      if [ -e "$d/rc" ]; then st=$(cat "$d/rc")
      elif [ -n "$(find "$d/gpu.csv" -mmin -10 2>/dev/null)" ]; then st=running
      else st=stale; fi
      cs=$(sha256sum < "$d/cmd" 2>/dev/null | cut -c1-64)
      echo "$d $lab ${st:-none} ${cs:-none}"
    done;;
  *) echo "usage: t9b_remote.sh list" >&2; exit 2;;
esac
