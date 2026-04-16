#!/usr/bin/env bash
set -e

echo "Starting backend with PORT=${PORT:-10000}..."

PORT_VALUE="${PORT:-10000}"

exec uvicorn main:app --host 0.0.0.0 --port "$PORT_VALUE"

