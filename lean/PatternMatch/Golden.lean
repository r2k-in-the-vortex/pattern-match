/-!
# Golden vectors

Executable definitions used by `Main.lean` to emit
`testdata/golden/vectors.json`. M1 adds geometry vectors; M4 adds DLT and
RANSAC vectors. The Rust side consumes the file in
`crates/pm-core/tests/golden.rs`.
-/

namespace PatternMatch.Golden

/-- Identifier of the golden-vector file format. -/
def formatId : String := "pattern-match-golden"

/-- Version of the golden-vector file format. -/
def formatVersion : Nat := 1

/-- Render pre-serialized vector objects as the golden-vector JSON document. -/
def renderJson (vectors : List String) : String :=
  "{\"format\":\"" ++ formatId ++ "\",\"version\":" ++ toString formatVersion ++
    ",\"vectors\":[" ++ String.intercalate "," vectors ++ "]}"

/-- The M0 golden-vector document: valid JSON containing no vectors yet. -/
def emptyJson : String :=
  renderJson []

end PatternMatch.Golden
