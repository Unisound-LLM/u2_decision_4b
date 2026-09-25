#!/usr/bin/env bash
# 三条 DEMO 的 curl 调用（需服务已启动在 BASE）
# 用法：
#   bash curl_demo.sh
#   BASE=http://127.0.0.1:8000 bash curl_demo.sh deepseek
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
BASE="${BASE:-http://127.0.0.1:8000}"
CASE="${1:-all}"

call() {
  local name="$1"
  local file="$DIR/curl_payload_${name}.json"
  echo "=== POST $BASE/v1/systemone  ($name) ==="
  curl -sS "$BASE/v1/systemone" \
    -H 'Content-Type: application/json' \
    -d @"$file"
  echo
  echo
}

# ---- 也可直接复制下面三条（相对 online_server 目录）----
# safety:
# curl -s http://127.0.0.1:8000/v1/systemone -H 'Content-Type: application/json' \
#   -d @curl_payload_safety.json
#
# deepseek:
# curl -s http://127.0.0.1:8000/v1/systemone -H 'Content-Type: application/json' \
#   -d @curl_payload_deepseek.json
#
# routing:
# curl -s http://127.0.0.1:8000/v1/systemone -H 'Content-Type: application/json' \
#   -d @curl_payload_routing.json

case "$CASE" in
  all) call safety; call deepseek; call routing ;;
  safety|deepseek|routing) call "$CASE" ;;
  *) echo "usage: $0 [all|safety|deepseek|routing]"; exit 1 ;;
esac
