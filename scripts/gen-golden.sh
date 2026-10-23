#!/usr/bin/env bash
# Regenerate testdata/golden/vectors.json from the Lean specification.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

(cd "$root/lean" && lake build golden)
(cd "$root/lean" && lake exe golden) > "$root/testdata/golden/vectors.json"
echo "wrote testdata/golden/vectors.json"
