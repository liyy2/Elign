#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: dl_monitor_loop.sh [--interval <seconds>] [--state-dir <dir>] [--once]

Continuously monitors training logs, sends to Codex for analysis.
Exits when max calls exceeded.

Environment:
  CODEX_CMD        Codex CLI binary (default: codex)
  CODEX_ARGS       Extra args for Codex CLI (default: --dangerously-bypass-approvals-and-sandbox)
  MAX_CALLS        Max Codex calls before exit (default: 100)
  HEARTBEAT_SEC    Force prompt interval even with no new logs (default: 1800)
  MASTER_PROMPT    Path to master_prompt.md (default: <script_dir>/master_prompt.md)
  WANDB_API_KEY    Weights & Biases API key for experiment tracking
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

INTERVAL="600"
ONCE="0"
STATE_DIR="${HOME}/.codexmon"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --interval) INTERVAL="${2:-}"; shift 2 ;;
    --state-dir) STATE_DIR="${2:-}"; shift 2 ;;
    --once)     ONCE="1"; shift ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 2
      ;;
  esac
done

mkdir -p "$STATE_DIR"

STATE="$STATE_DIR/state.json"
REPORT="$STATE_DIR/report.json"
ALERTS_LOG="$STATE_DIR/alerts.log"

CODEX_CMD="${CODEX_CMD:-codex}"
CODEX_ARGS="${CODEX_ARGS:---dangerously-bypass-approvals-and-sandbox}"
MAX_CALLS="${MAX_CALLS:-400}"
HEARTBEAT_SEC="${HEARTBEAT_SEC:-1800}"
SLACK_WEBHOOK="${SLACK_WEBHOOK:-}"
SLACK_BOT_TOKEN="${SLACK_BOT_TOKEN:-}"
SLACK_CHANNEL="${SLACK_CHANNEL:-}"

# Weights & Biases API key for experiment tracking:
# set `WANDB_API_KEY` in your environment or run `wandb login` (do not hardcode keys here).

# Master prompt file (experiment-specific goals)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MASTER_PROMPT="${MASTER_PROMPT:-$SCRIPT_DIR/master_prompt.md}"

TMP_DIR="$(mktemp -d /tmp/codexmon.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

# Initialize state file if missing
init_state() {
  [[ -f "$STATE" ]] && return
  cat > "$STATE" <<'JSON'
{"last_summary":"","last_alert_level":"none","last_prompt_ts":0,"call_count":0}
JSON
}

read_state() {
  python3 -c "import json; print(json.load(open('$STATE')).get('$1', ''))"
}

write_state() {
  python3 - "$STATE" "$1" "$2" "$3" "$4" <<'PY'
import json, sys
f, summary, alert, ts, calls = sys.argv[1:]
s = json.load(open(f))
s.update({"last_summary": summary, "last_alert_level": alert,
          "last_prompt_ts": int(ts), "call_count": int(calls)})
json.dump(s, open(f, "w"))
PY
}

build_prompt() {
  local prior_summary="$1" prior_alert="$2" call_count="$3"

  # Load master prompt if exists
  local master_content=""
  if [[ -f "$MASTER_PROMPT" ]]; then
    master_content="$(cat "$MASTER_PROMPT")"
  fi

  # Read feedback from Slack channel
  local feedback=""
  feedback="$(read_slack_feedback)"

  cat > "$TMP_DIR/prompt.txt" <<EOF
$master_content

${feedback:+## User Feedback
$feedback

}---

Return exactly one line of JSON:
{"alert_level":"none|info|warning|critical","summary":"...","recommendation":"what to do next or try in next run"}

Prior summary: $prior_summary
Prior alert: $prior_alert
Call count: $call_count / $MAX_CALLS
EOF
}

run_codex() {
  if ! command -v "$CODEX_CMD" >/dev/null 2>&1; then
    echo "Codex not found: $CODEX_CMD" >&2
    exit 1
  fi
  read -r -a args <<< "$CODEX_ARGS"
  # Stream to terminal AND capture to file (unbuffered)
  stdbuf -oL "$CODEX_CMD" exec "${args[@]}" "$(cat "$TMP_DIR/prompt.txt")" 2>&1 | tee "$TMP_DIR/response.txt" || true
}

parse_json() {
  python3 - "$1" <<'PY'
import json, sys
for line in open(sys.argv[1]):
    line = line.strip()
    # Skip template lines
    if "none|info|warning|critical" in line:
        continue
    if line.startswith("{") and line.endswith("}"):
        try:
            obj = json.loads(line)
            if "alert_level" in obj and "summary" in obj:
                print(line)
                sys.exit(0)
        except: pass
print('{"alert_level":"info","summary":"(no valid JSON)","recommendation":""}')
PY
}

get_json_field() {
  python3 -c "import json; print(json.loads(open('$REPORT').readline()).get('$1',''))"
}

send_slack() {
  local message="$1"
  # Use bot token to post to DM/channel
  if [[ -n "$SLACK_BOT_TOKEN" && -n "$SLACK_CHANNEL" ]]; then
    curl -s -X POST -H "Authorization: Bearer $SLACK_BOT_TOKEN" \
      -H 'Content-type: application/json' \
      --data "{\"channel\":\"$SLACK_CHANNEL\",\"text\":\"$message\"}" \
      "https://slack.com/api/chat.postMessage" >/dev/null 2>&1 || true
  fi
}

# Read recent messages from Slack channel (feedback from user)
read_slack_feedback() {
  if [[ -z "$SLACK_BOT_TOKEN" || -z "$SLACK_CHANNEL" ]]; then
    echo ""
    return
  fi
  
  local last_ts_file="$STATE_DIR/slack_last_ts"
  local oldest="0"
  [[ -f "$last_ts_file" ]] && oldest="$(cat "$last_ts_file")"
  
  # Fetch messages newer than last check
  local response
  response=$(curl -s -H "Authorization: Bearer $SLACK_BOT_TOKEN" \
    "https://slack.com/api/conversations.history?channel=$SLACK_CHANNEL&oldest=$oldest&limit=10" 2>/dev/null)
  
  # Extract messages and update timestamp
  python3 - "$response" "$last_ts_file" <<'PY'
import json, sys
try:
    data = json.loads(sys.argv[1])
    if data.get("ok") and data.get("messages"):
        msgs = [m.get("text","") for m in data["messages"] if m.get("subtype") is None]
        if msgs:
            print("\n".join(reversed(msgs)))
        # Save latest timestamp
        latest = max(float(m.get("ts", "0")) for m in data["messages"])
        if latest > 0:
            with open(sys.argv[2], "w") as f:
                f.write(str(latest))
except: pass
PY
}

# ─────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────
init_state

while true; do
  last_summary="$(read_state last_summary)"
  last_alert="$(read_state last_alert_level)"
  last_prompt_ts="$(read_state last_prompt_ts)"
  call_count="$(read_state call_count)"

  # Bail if max calls exceeded
  if (( call_count >= MAX_CALLS )); then
    echo "Max Codex calls ($MAX_CALLS) reached. Exiting."
    send_slack "🛑 Monitor stopped: max calls ($MAX_CALLS) reached"
    exit 0
  fi

  now_ts="$(date +%s)"

  # Build prompt and call Codex
  build_prompt "$last_summary" "$last_alert" "$call_count"
  
  # Log prompt to Slack
  prompt_preview="$(head -c 500 "$TMP_DIR/prompt.txt" | tr '\n' ' ')"
  send_slack "🚀 [Call #$((call_count + 1))] Starting Codex...\n📝 Prompt: ${prompt_preview}..."
  
  run_codex
  parse_json "$TMP_DIR/response.txt" > "$REPORT"
  call_count=$(( call_count + 1 ))

  # Extract response fields
  alert_level="$(get_json_field alert_level)"
  summary="$(get_json_field summary)"
  recommendation="$(get_json_field recommendation)"

  # Log alerts and recommendations
  if [[ "$alert_level" == "warning" || "$alert_level" == "critical" ]]; then
    echo "[$(date)] $alert_level: $summary" >> "$ALERTS_LOG"
    [[ -n "$recommendation" ]] && echo "  -> $recommendation" >> "$ALERTS_LOG"
  fi

  # Always log recommendation to a separate file
  if [[ -n "$recommendation" ]]; then
    echo "[$(date)] $recommendation" >> "$STATE_DIR/recommendations.log"
  fi

  # Send Slack notification
  send_slack "[Monitor #$call_count] $alert_level: $summary\n💡 $recommendation"

  # Update state
  write_state "$summary" "$alert_level" "$now_ts" "$call_count"

  # Single iteration mode
  [[ "$ONCE" == "1" ]] && exit 0

  sleep "$INTERVAL"
done
