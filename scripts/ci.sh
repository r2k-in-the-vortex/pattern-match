#!/usr/bin/env bash
# Local mirror of the CI checks in README §10.5.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

echo "==> Lean: build"
(cd lean && lake build)

echo "==> Lean: no sorry in core modules"
"$root/scripts/check-sorry.sh"

echo "==> Lean: golden vectors are current"
tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT
(cd lean && lake exe golden) > "$tmp"
if ! diff -u "$root/testdata/golden/vectors.json" "$tmp"; then
  echo "golden vectors are stale; run scripts/gen-golden.sh" >&2
  exit 1
fi

echo "==> Rust: format"
cargo fmt --all -- --check

echo "==> Rust: clippy"
cargo clippy --workspace --all-targets -- -D warnings

echo "==> Rust: tests"
cargo test --workspace

echo "CI checks passed."
