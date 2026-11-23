#!/usr/bin/env bash
set -euo pipefail

if (( $# != 0 )); then
    echo "Finetune reads config.toml; run without CLI arguments." >&2
    exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
exec python -m src.zalo_ai.finetune
