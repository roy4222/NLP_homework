#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEB_HOST="${WEB_HOST:-0.0.0.0}"
WEB_PORT="${WEB_PORT:-3001}"
NODE_DIR="${NODE_DIR:-$HOME/.local/node/node-v22.14.0-linux-x64/bin}"

if [[ -d "$NODE_DIR" ]]; then
  export PATH="$NODE_DIR:$HOME/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
fi

cd "$ROOT_DIR/web"

npm run build

cd "$ROOT_DIR/web/out"
exec python3 -m http.server "$WEB_PORT" --bind "$WEB_HOST"
