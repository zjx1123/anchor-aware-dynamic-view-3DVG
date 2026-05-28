#!/bin/bash
# Usage: sh stop_daemon.sh [scanrefer|nr3d]

cd "$(dirname "$0")"

DATASET="${1:-scanrefer}"
PID_FILE="../logs/pids/${DATASET}.pid"

if [ ! -f "$PID_FILE" ]; then
  echo "[$DATASET] not running"
  exit 0
fi

PID="$(cat "$PID_FILE")"
if kill -0 "$PID" 2>/dev/null; then
  # 终止整个进程树（sh + python 子进程）
  pkill -P "$PID" 2>/dev/null || true
  kill "$PID" 2>/dev/null || true
  sleep 1
  kill -9 "$PID" 2>/dev/null || true
  pkill -9 -P "$PID" 2>/dev/null || true
  echo "[$DATASET] stopped (PID=$PID)"
else
  echo "[$DATASET] process already exited (PID=$PID)"
fi
rm -f "$PID_FILE"
