#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
find . -type d -name "__pycache__" -prune -exec rm -rf {} +

echo "Removed .pytest_cache and __pycache__ directories."
