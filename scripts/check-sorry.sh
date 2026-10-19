#!/usr/bin/env bash
# Fail if a core specification module contains the `sorry` token in code
# (README §10.5). Lean comments — including nested `/- -/` blocks — are
# stripped first, so prose mentioning `sorry` does not trip the gate.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root/lean"

files=(
  PatternMatch/Geometry.lean
  PatternMatch/DLT.lean
  PatternMatch/RANSAC.lean
)

missing=0
for file in "${files[@]}"; do
  if [[ ! -f "$file" ]]; then
    echo "check-sorry: missing $file" >&2
    missing=1
  fi
done
if [[ $missing -ne 0 ]]; then
  exit 1
fi

matches=$(
  awk '
    FNR == 1 { depth = 0 }
    {
      line = $0
      code = ""
      i = 1
      n = length(line)
      while (i <= n) {
        two = substr(line, i, 2)
        if (depth == 0 && two == "--") break
        if (two == "/-") { depth++; i += 2; continue }
        if (depth > 0 && two == "-/") { depth--; i += 2; continue }
        if (depth == 0) code = code substr(line, i, 1)
        i++
      }
      if (code ~ /(^|[^A-Za-z0-9_])sorry([^A-Za-z0-9_]|$)/) {
        printf "%s:%d: %s\n", FILENAME, FNR, code
      }
    }
  ' "${files[@]}"
)

if [[ -n "$matches" ]]; then
  echo "check-sorry: found 'sorry' in core modules:" >&2
  echo "$matches" >&2
  exit 1
fi

echo "check-sorry: clean"
