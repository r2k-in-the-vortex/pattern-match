import PatternMatch.Golden

/-- Emit the golden-vector JSON document on stdout. -/
def main : IO Unit :=
  IO.println PatternMatch.Golden.allJson
