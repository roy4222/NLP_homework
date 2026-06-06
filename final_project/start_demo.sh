#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-5000}"
WEB_HOST="${WEB_HOST:-127.0.0.1}"
WEB_PORT="${WEB_PORT:-3000}"
NODE_DIR="${NODE_DIR:-$HOME/.local/node/node-v22.14.0-linux-x64/bin}"

if [[ -d "$NODE_DIR" ]]; then
  export PATH="$NODE_DIR:$HOME/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
fi

cleanup() {
  echo
  echo "Stopping demo services..."
  if [[ -n "${API_PID:-}" ]] && kill -0 "$API_PID" 2>/dev/null; then
    kill "$API_PID" 2>/dev/null || true
  fi
  if [[ -n "${WEB_PID:-}" ]] && kill -0 "$WEB_PID" 2>/dev/null; then
    kill "$WEB_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

need_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing command: $1" >&2
    exit 1
  fi
}

wait_for_url() {
  local url="$1"
  local name="$2"
  local max_attempts="${3:-30}"
  for _ in $(seq 1 "$max_attempts"); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "$name is ready: $url"
      return 0
    fi
    sleep 1
  done
  echo "$name did not become ready: $url" >&2
  return 1
}

port_in_use() {
  python3 - "$WEB_HOST" "$1" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(0.4)
    sys.exit(0 if sock.connect_ex((host, port)) == 0 else 1)
PY
}

find_free_port() {
  local port="$1"
  while port_in_use "$port"; do
    port=$((port + 1))
  done
  echo "$port"
}

need_command uv
need_command npm
need_command curl
need_command python3

cd "$ROOT_DIR"

WEB_PORT="$(find_free_port "$WEB_PORT")"
: > /tmp/nlp_final_flask.log
: > /tmp/nlp_final_next.log

echo "Starting Flask API on http://$API_HOST:$API_PORT"
uv run python api/app.py > /tmp/nlp_final_flask.log 2>&1 &
API_PID="$!"
wait_for_url "http://$API_HOST:$API_PORT/api/health" "Flask API"

cd "$ROOT_DIR/web"
if [[ ! -d node_modules ]]; then
  echo "Installing frontend dependencies..."
  npm ci
fi

echo "Starting Next.js frontend on http://$WEB_HOST:$WEB_PORT"
npm run dev -- --hostname "$WEB_HOST" -p "$WEB_PORT" > /tmp/nlp_final_next.log 2>&1 &
WEB_PID="$!"
wait_for_url "http://$WEB_HOST:$WEB_PORT" "Next.js frontend" 45

echo
echo "Demo is ready."
echo "Frontend: http://$WEB_HOST:$WEB_PORT"
echo "API health: http://$API_HOST:$API_PORT/api/health"
echo
echo "If Next.js reports a different port, check: /tmp/nlp_final_next.log"
echo "Flask log: /tmp/nlp_final_flask.log"
echo "Next log: /tmp/nlp_final_next.log"
echo
echo "Press Ctrl+C to stop both services."

wait "$API_PID" "$WEB_PID"
