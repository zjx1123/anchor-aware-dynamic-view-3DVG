#!/bin/bash
# Usage: sh status_daemon.sh [scanrefer|nr3d]

cd "$(dirname "$0")"

DATASET="${1:-scanrefer}"
PID_FILE="../logs/pids/${DATASET}.pid"

if [ ! -f "$PID_FILE" ]; then
  echo "[$DATASET] not running (no pid file)"
  exit 0
fi

PID="$(cat "$PID_FILE")"
if kill -0 "$PID" 2>/dev/null; then
  echo "[$DATASET] running (PID=$PID)"
  echo "--- recent nohup logs ---"
  ls -lt ../logs/pids/${DATASET}_nohup_*.log 2>/dev/null | head -3
  echo "--- recent python full logs ---"
  ls -lt ../logs/${DATASET}*_full_*.log ../logs/scanrefer_full_*.log ../logs/visprog_scanrefer_*.log 2>/dev/null | head -5
else
  echo "[$DATASET] pid file exists but process dead (PID=$PID)"
  rm -f "$PID_FILE"
fi
