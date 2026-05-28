#!/bin/bash
# 后台运行评估，本地断网/关闭 SSH 后进程仍继续。
# Usage:
#   sh run_daemon.sh scanrefer
#   sh run_daemon.sh nr3d

set -e
cd "$(dirname "$0")"

DATASET="${1:-scanrefer}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
PID_DIR="../logs/pids"
mkdir -p "$PID_DIR" ../logs

case "$DATASET" in
  scanrefer)
    RUN_SCRIPT="run_script.sh"
    ;;
  nr3d)
    RUN_SCRIPT="run_script_nr3d.sh"
    ;;
  *)
    echo "Usage: sh run_daemon.sh [scanrefer|nr3d]"
    exit 1
    ;;
esac

PID_FILE="${PID_DIR}/${DATASET}.pid"
NOHUP_LOG="${PID_DIR}/${DATASET}_nohup_${TIMESTAMP}.log"

if [ -f "$PID_FILE" ]; then
  OLD_PID="$(cat "$PID_FILE")"
  if kill -0 "$OLD_PID" 2>/dev/null; then
    echo "[$DATASET] already running (PID=$OLD_PID)"
    echo "  stop: sh stop_daemon.sh $DATASET"
    echo "  status: sh status_daemon.sh $DATASET"
    exit 1
  fi
  rm -f "$PID_FILE"
fi

# nohup: 忽略 SIGHUP，SSH 断开不会终止任务
nohup sh "$RUN_SCRIPT" >"$NOHUP_LOG" 2>&1 &
echo $! >"$PID_FILE"

echo "[$DATASET] started in background (immune to SSH disconnect)"
echo "  PID: $(cat "$PID_FILE")"
echo "  nohup wrapper log: $NOHUP_LOG"
echo "  python logs: ../logs/scanrefer_full_* / visprog_scanrefer_*  (or nr3d_*)"
echo ""
echo "  tail -f $NOHUP_LOG"
echo "  sh status_daemon.sh $DATASET"
echo "  sh stop_daemon.sh $DATASET"
