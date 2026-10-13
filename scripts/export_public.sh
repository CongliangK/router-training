#!/usr/bin/env bash
# Build a clean, publishable copy of this repository at ../<name>-public.
#
# Why this exists: the private git history contains real task prompts and
# labels. Pushing this repo or any branch of it would publish that history.
# The public release must therefore be a fresh git repo built only from
# allow-listed files.
#
# Usage: bash scripts/export_public.sh [target-dir]   (default: sibling dir)
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DST="${1:-$SRC/../router-training-public}"

if [ -e "$DST/.git" ]; then
  echo "Refusing to overwrite existing git repo: $DST" >&2
  exit 1
fi

# Allow-list: code, public docs, public data examples, metadata.
# Everything else (real data, labeling batches, artifacts, logs, weights,
# environment records, snapshot manifests) stays private.
rsync -a --delete \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude '*.safetensors' --exclude '*.bin' --exclude '*.pt' --exclude '*.pth' --exclude '*.ckpt' \
  --exclude 'data/*.jsonl' \
  --exclude 'data/labeling/batch-*.jsonl' \
  --exclude 'data/labeling/labels-*.jsonl' \
  --exclude 'data/synth' \
  --exclude 'artifacts/**/*.jsonl' \
  --exclude 'artifacts/real-history-v3c/full-checkpoint' \
  --exclude 'logs' \
  --exclude 'observations' \
  --exclude 'environment.json' \
  --exclude 'snapshot-manifest.json' \
  "$SRC"/ "$DST"/

# Guard: fail loudly if any real (non-example) data file slipped through.
if compgen -G "$DST/data/*.jsonl" > /dev/null; then
  echo "ERROR: real data files present in export:" >&2
  ls "$DST"/data/*.jsonl >&2
  exit 1
fi
if find "$DST/artifacts" -name '*.jsonl' 2>/dev/null | grep -q .; then
  echo "ERROR: prediction files present in export." >&2
  exit 1
fi
if grep -rq "liangruochong\|/home/USER/文档" "$DST" --include='*.md' --include='*.py' --include='*.json'; then
  echo "WARNING: personal identifiers found in exported text files (review above hits)." >&2
fi

# Sanitize embedded local paths (model-manifest.json is copied into report.json).
find "$DST" -name '*.json' -type f -exec sed -i \
  -e 's#/home/[^/" ]*/文档/01_活跃项目/deepseek/router-training#/repo#g' \
  -e 's#/home/[^/" ]*/文档/01_活跃项目/deepseek/user-modes/router-framework#/upstream-router-framework#g' \
  {} +

echo "Public export created at: $DST"
echo "Remaining steps: review, git init, commit, push. See docs/RELEASE-CHECKLIST.md"
