#!/usr/bin/env bash
# Everything: starts the ACE-Step server (AI edits), then the home page with the
# chord finder, the splitter and the mixer on one port (mixpipe.cli ui, :8760).
# Ctrl+C stops both (an ACE-Step that was already running is left alone).
#
#   scripts/studio.sh [ui flags]       e.g. --port 8770 --no-browser
#   make studio ARGS="--no-browser"
#
# ACESTEP_DIR    the ACE-Step 1.5 checkout (default ../ACE-Step-1.5; make install-acestep)
# ACESTEP_PORT   its port (default 8001)
# Logs: out/logs/acestep.log

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACESTEP_DIR="${ACESTEP_DIR:-$ROOT/../ACE-Step-1.5}"
ACESTEP_PORT="${ACESTEP_PORT:-8001}"
export ACESTEP_URL="http://127.0.0.1:$ACESTEP_PORT"
LOG="$ROOT/out/logs/acestep.log"
PY="$ROOT/.venv/bin/python"
[[ -x "$PY" ]] || PY=python3
ACE_PID=""

acestep_up() { curl -s -m 2 "$ACESTEP_URL/health" 2>/dev/null | grep -q '"status":"ok"'; }

stop() {
    if [[ -n "$ACE_PID" ]] && kill -0 "$ACE_PID" 2>/dev/null; then
        echo "stopping ACE-Step ..."
        pkill -TERM -P "$ACE_PID" 2>/dev/null || true     # uv's python child
        kill -TERM "$ACE_PID" 2>/dev/null || true
        wait "$ACE_PID" 2>/dev/null || true
    fi
}
trap stop EXIT
trap 'exit 0' INT TERM                  # Ctrl+C is how you stop it, not an error

# ---- ACE-Step ------------------------------------------------------------
if acestep_up; then
    echo "ACE-Step is already running at $ACESTEP_URL"
elif [[ ! -d "$ACESTEP_DIR" ]]; then
    echo "ACE-Step isn't installed at $ACESTEP_DIR."
    echo "  make install-acestep   (about 11 GB with the model)"
    echo "Starting without AI edits."
elif ! command -v uv >/dev/null; then
    echo "ACE-Step needs uv (brew install uv). Starting the mixer without AI edits."
else
    mkdir -p "$(dirname "$LOG")"
    echo "starting ACE-Step on $ACESTEP_URL (log: out/logs/acestep.log) ..."
    (cd "$ACESTEP_DIR" && ACESTEP_LM_BACKEND=mlx ACESTEP_INIT_LLM=false TOKENIZERS_PARALLELISM=false \
        exec uv run acestep-api --host 127.0.0.1 --port "$ACESTEP_PORT") >"$LOG" 2>&1 &
    ACE_PID=$!
    for _ in $(seq 1 90); do
        acestep_up && break
        if ! kill -0 "$ACE_PID" 2>/dev/null; then
            echo "ACE-Step exited; last lines of the log:"
            tail -n 15 "$LOG"
            echo "Starting without AI edits."
            ACE_PID=""
            break
        fi
        sleep 2
    done
    if [[ -n "$ACE_PID" ]]; then
        if acestep_up; then
            echo "ACE-Step is ready"
        else
            echo "ACE-Step is still starting; the AI edit card shows when it's ready"
        fi
    fi
fi

# ---- the tools -------------------------------------------------------------
echo "starting the chord finder, splitter and mixer ..."
cd "$ROOT"
PYTHONPATH=src "$PY" -m mixpipe.cli ui "$@"
