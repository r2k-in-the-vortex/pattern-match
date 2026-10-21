/-!
# Golden vectors

Executable definitions used by `Main.lean` to emit
`testdata/golden/vectors.json`. M1 adds geometry vectors; M4 adds DLT and
RANSAC vectors. The Rust side consumes the file in
`crates/pm-core/tests/golden.rs`.

All numeric leaves are encoded as exact rational strings `"numerator/denominator"`
so the committed file is reproducible and the Rust comparisons do not depend on
floating-point printing. The formulas below mirror the authoritative
definitions in `PatternMatch.Geometry`; each vector names its theorem ID.
This module deliberately imports no Mathlib so that `lake exe golden` stays a
small, fast build.
-/

namespace PatternMatch.Golden

/-- Identifier of the golden-vector file format. -/
def formatId : String := "pattern-match-golden"

/-- Version of the golden-vector file format. -/
def formatVersion : Nat := 1

/-! ## Exact rational helpers -/

abbrev Q := Rat
abbrev P3 := Fin 3 → Q
abbrev M3 := Fin 3 → Fin 3 → Q

/-- A three-element vector. -/
def vec3 {α : Type} (a b c : α) : Fin 3 → α
  | ⟨0, _⟩ => a
  | ⟨1, _⟩ => b
  | _ => c

/-- A homogeneous point. -/
def p3 (a b c : Q) : P3 := vec3 a b c

/-- A 3×3 matrix given row by row. -/
def m3 (a b c d e f g h i : Q) : M3 :=
  vec3 (vec3 a b c) (vec3 d e f) (vec3 g h i)

def matVec (A : M3) (p : P3) : P3 :=
  fun i => A i 0 * p 0 + A i 1 * p 1 + A i 2 * p 2

def matMul (A B : M3) : M3 :=
  fun i j => A i 0 * B 0 j + A i 1 * B 1 j + A i 2 * B 2 j

def normalizeP (p : P3) : P3 := fun i => p i / p 2

def det3 (A : M3) : Q :=
  A 0 0 * (A 1 1 * A 2 2 - A 1 2 * A 2 1) -
    A 0 1 * (A 1 0 * A 2 2 - A 1 2 * A 2 0) +
    A 0 2 * (A 1 0 * A 2 1 - A 1 1 * A 2 0)

/-- Inverse of a non-singular 3×3 matrix (adjugate / determinant). -/
def inverse3 (A : M3) : M3 :=
  let d := det3 A
  let c00 := A 1 1 * A 2 2 - A 1 2 * A 2 1
  let c01 := -(A 1 0 * A 2 2 - A 1 2 * A 2 0)
  let c02 := A 1 0 * A 2 1 - A 1 1 * A 2 0
  let c10 := -(A 0 1 * A 2 2 - A 0 2 * A 2 1)
  let c11 := A 0 0 * A 2 2 - A 0 2 * A 2 0
  let c12 := -(A 0 0 * A 2 1 - A 0 1 * A 2 0)
  let c20 := A 0 1 * A 1 2 - A 0 2 * A 1 1
  let c21 := -(A 0 0 * A 1 2 - A 0 2 * A 1 0)
  let c22 := A 0 0 * A 1 1 - A 0 1 * A 1 0
  m3 (c00 / d) (c10 / d) (c20 / d)
     (c01 / d) (c11 / d) (c21 / d)
     (c02 / d) (c12 / d) (c22 / d)

def crossRatio (x₁ x₂ x₃ x₄ : Q) : Q :=
  ((x₁ - x₃) * (x₂ - x₄)) / ((x₁ - x₄) * (x₂ - x₃))

def mobius (a b c d x : Q) : Q := (a * x + b) / (c * x + d)

/-! ## Similarity with angles that are exact multiples of 90° -/

/-- A similarity whose angle is `quarterTurns * 90°`. -/
structure Sim where
  s : Q
  quarterTurns : Int
  tx : Q
  ty : Q

def cosQuarter (k : Int) : Q :=
  match k % 4 with
  | 0 => 1
  | 1 => 0
  | 2 => -1
  | _ => 0

def sinQuarter (k : Int) : Q :=
  match k % 4 with
  | 0 => 0
  | 1 => 1
  | 2 => 0
  | _ => -1

def simMatrix (g : Sim) : M3 :=
  let c := cosQuarter g.quarterTurns
  let s := sinQuarter g.quarterTurns
  m3 (g.s * c) (-(g.s * s)) g.tx
     (g.s * s) (g.s * c) g.ty
     0 0 1

def simComp (g h : Sim) : Sim :=
  let c := cosQuarter g.quarterTurns
  let s := sinQuarter g.quarterTurns
  { s := g.s * h.s
    quarterTurns := g.quarterTurns + h.quarterTurns
    tx := g.tx + g.s * (c * h.tx - s * h.ty)
    ty := g.ty + g.s * (s * h.tx + c * h.ty) }

def jacEntry (H : M3) (i j : Fin 3) : Q :=
  let w := H 2 2
  let qi := H i 2 / w
  (H i j - qi * H 2 j) / w

/-! ## JSON rendering -/

def ratJson (q : Q) : String :=
  "\"" ++ toString q.num ++ "/" ++ toString q.den ++ "\""

def ratListJson (xs : List Q) : String :=
  "[" ++ String.intercalate "," (xs.map ratJson) ++ "]"

def p3Json (p : P3) : String :=
  ratListJson [p 0, p 1, p 2]

def m3Json (A : M3) : String :=
  ratListJson [A 0 0, A 0 1, A 0 2,
               A 1 0, A 1 1, A 1 2,
               A 2 0, A 2 1, A 2 2]

def strJson (s : String) : String := "\"" ++ s ++ "\""

def obj (fields : List (String × String)) : String :=
  "{" ++ String.intercalate "," (fields.map fun (k, v) => strJson k ++ ":" ++ v) ++ "}"

/-! ## M1 vectors -/

-- Shared test data.

def hA : M3 := m3 1 2 3 0 1 4 5 6 1

def hB : M3 := m3 2 0 (-1) 1 3 2 0 1 1

def pTest : P3 := p3 2 (-1) 3

/-- G1: `(H₁H₂)·p = H₁·(H₂·p)` (the Rust test checks both sides). -/
def g1Vector : String :=
  obj [
    ("kind", strJson "homography_action"),
    ("id", strJson "G1-action-composition"),
    ("theorem", strJson "G1"),
    ("h1", m3Json hA),
    ("h2", m3Json hB),
    ("p", p3Json pTest),
    ("expected", p3Json (matVec (matMul hA hB) pTest))
  ]

/-- G2: scaling by a non-zero `c` leaves the affine action unchanged. -/
def g2Vector : String :=
  obj [
    ("kind", strJson "homography_scaling"),
    ("id", strJson "G2-projective-scale"),
    ("theorem", strJson "G2"),
    ("h", m3Json hA),
    ("c", ratJson (-2)),
    ("p", p3Json pTest),
    ("expected", p3Json (normalizeP (matVec hA pTest)))
  ]

/-- G3: the inverse undoes the action; the inverse matrix is supplied too. -/
def g3Vector : String :=
  obj [
    ("kind", strJson "homography_inverse"),
    ("id", strJson "G3-inverse"),
    ("theorem", strJson "G3"),
    ("h", m3Json hA),
    ("h_inv", m3Json (inverse3 hA)),
    ("p", p3Json pTest),
    ("expected", p3Json (normalizeP pTest))
  ]

def simFirst : Sim := { s := 2, quarterTurns := 1, tx := 3, ty := -1 }
def simSecond : Sim := { s := (1 : Q) / 2, quarterTurns := 2, tx := 4, ty := 2 }

/-- G4: embedding is a homomorphism; expected parameters and matrix. -/
def g4Vector : String :=
  let g := simFirst
  let h := simSecond
  let c := simComp g h
  obj [
    ("kind", strJson "similarity_compose"),
    ("id", strJson "G4-similarity-composition"),
    ("theorem", strJson "G4"),
    ("first", obj [
      ("scale", ratJson g.s),
      ("theta_deg", ratJson ((g.quarterTurns * 90 : Int) : Q)),
      ("tx", ratJson g.tx),
      ("ty", ratJson g.ty)]),
    ("second", obj [
      ("scale", ratJson h.s),
      ("theta_deg", ratJson ((h.quarterTurns * 90 : Int) : Q)),
      ("tx", ratJson h.tx),
      ("ty", ratJson h.ty)]),
    ("expected", obj [
      ("scale", ratJson c.s),
      ("theta_deg", ratJson ((c.quarterTurns * 90 : Int) : Q)),
      ("tx", ratJson c.tx),
      ("ty", ratJson c.ty)]),
    ("expected_matrix", m3Json (simMatrix c))
  ]

/-- G5: the block `[[a,-b],[b,a]]` decomposes as `(s, cos, sin)`. -/
def g5Vector (id : String) (a b s : Q) : String :=
  obj [
    ("kind", strJson "similarity_decomposition"),
    ("id", strJson id),
    ("theorem", strJson "G5"),
    ("a", ratJson a),
    ("b", ratJson b),
    ("expected_scale", ratJson s),
    ("expected_cos", ratJson (a / s)),
    ("expected_sin", ratJson (b / s))
  ]

/-- G6: cross-ratio is invariant under a non-degenerate Möbius map. -/
def g6Vector (id : String) (x₁ x₂ x₃ x₄ : Q) (a b c d : Q) : String :=
  obj [
    ("kind", strJson "cross_ratio"),
    ("id", strJson id),
    ("theorem", strJson "G6"),
    ("x", ratListJson [x₁, x₂, x₃, x₄]),
    ("mobius", ratListJson [a, b, c, d]),
    ("expected", ratJson (crossRatio x₁ x₂ x₃ x₄)),
    ("expected_transformed",
      ratJson (crossRatio
        (mobius a b c d x₁) (mobius a b c d x₂)
        (mobius a b c d x₃) (mobius a b c d x₄)))
  ]

/-- G7: collinear points stay collinear under a homography. -/
def g7Vector : String :=
  let p := p3 0 0 1
  let q := p3 1 1 1
  let r := p3 2 2 1
  obj [
    ("kind", strJson "collinearity"),
    ("id", strJson "G7-collinearity"),
    ("theorem", strJson "G7"),
    ("h", m3Json hA),
    ("points", "[" ++ String.intercalate ","
      [p3Json p, p3Json q, p3Json r] ++ "]"),
    ("expected_det", ratJson 0)
  ]

/-- G8: origin and local Jacobian of the affine action. -/
def g8GeneralVector : String :=
  let h := m3 2 1 3 1 4 5 (1 / 2) (1 / 4) 2
  obj [
    ("kind", strJson "jacobian"),
    ("id", strJson "G8-jacobian-general"),
    ("theorem", strJson "G8"),
    ("h", m3Json h),
    ("expected_origin", ratListJson [h 0 2 / h 2 2, h 1 2 / h 2 2]),
    ("expected_jacobian", ratListJson
      [jacEntry h 0 0, jacEntry h 0 1, jacEntry h 1 0, jacEntry h 1 1])
  ]

/-- G8 on a similarity embedding: the local pose is exactly `(t, θ, s)`. -/
def g8SimilarityVector : String :=
  let g : Sim := { s := 2, quarterTurns := 1, tx := 3, ty := -1 }
  let h := simMatrix g
  obj [
    ("kind", strJson "jacobian"),
    ("id", strJson "G8-jacobian-similarity"),
    ("theorem", strJson "G8"),
    ("h", m3Json h),
    ("expected_origin", ratListJson [h 0 2 / h 2 2, h 1 2 / h 2 2]),
    ("expected_jacobian", ratListJson
      [jacEntry h 0 0, jacEntry h 0 1, jacEntry h 1 0, jacEntry h 1 1]),
    ("expected_angle_deg", ratJson 90),
    ("expected_scale", ratJson 2)
  ]

/-- All M1 golden vectors, in a stable order. -/
def geometryVectors : List String :=
  [g1Vector,
   g2Vector,
   g3Vector,
   g4Vector,
   g5Vector "G5-decomposition-3-4" 3 4 5,
   g5Vector "G5-decomposition-0-neg2" 0 (-2) 2,
   g6Vector "G6-cross-ratio" 1 2 4 8 2 1 1 3,
   g6Vector "G6-cross-ratio-degenerate" 2 5 2 9 2 1 1 3,
   g7Vector,
   g8GeneralVector,
   g8SimilarityVector]

/-! ## M4 DLT vectors -/

/-- An affine correspondence with exact rational coordinates. -/
structure CorrQ where
  x : Q
  y : Q
  u : Q
  v : Q

def srcQ (c : CorrQ) : P3 := p3 c.x c.y 1

def dstQ (c : CorrQ) : P3 := p3 c.u c.v 1

/-- The matrix whose columns are three homogeneous points. -/
def colsQ (p q r : P3) : M3 := fun i j => vec3 p q r j i

/-- Coordinates of a point in the projective basis given by `P`. -/
def coordsQ (P : M3) (x : P3) : P3 := matVec (inverse3 P) x

def diagQ (a b c : Q) : M3 := m3 a 0 0 0 b 0 0 0 c

/-- Apply a homography in affine coordinates. -/
def applyH (H : M3) (p : P3) : P3 := normalizeP (matVec H p)

/-- Build a correspondence from a source point and a homography. -/
def corrFromH (H : M3) (x y : Q) : CorrQ :=
  let q := applyH H (p3 x y 1)
  { x := x, y := y, u := q 0, v := q 1 }

/-- The DLT solution from the projective-basis construction (D3). -/
def dltSolution (c0 c1 c2 c3 : CorrQ) : M3 :=
  let p0 := srcQ c0
  let p1 := srcQ c1
  let p2 := srcQ c2
  let p3v := srcQ c3
  let q0 := dstQ c0
  let q1 := dstQ c1
  let q2 := dstQ c2
  let q3 := dstQ c3
  let P := colsQ p0 p1 p2
  let T := colsQ q0 q1 q2
  let a := coordsQ P p3v
  let b := coordsQ T q3
  let D := diagQ (b 0 / a 0) (b 1 / a 1) (b 2 / a 2)
  matMul (matMul T D) (inverse3 P)

/-- Normalize a matrix by `h33`. -/
def normalizeMat (H : M3) : M3 := fun i j => H i j / H 2 2

def corrJson (c : CorrQ) : String :=
  obj [
    ("x", ratJson c.x),
    ("y", ratJson c.y),
    ("u", ratJson c.u),
    ("v", ratJson c.v)
  ]

def corrListJson (cs : List CorrQ) : String :=
  "[" ++ String.intercalate "," (cs.map corrJson) ++ "]"

def natJson (n : Nat) : String := toString n

def boolJson (b : Bool) : String := if b then "true" else "false"

/-- A general-position DLT vector: four correspondences from a known
homography, plus the normalized DLT solution. -/
def dltGeneralVector (id thm : String) (H : M3) (pts : List (Q × Q)) : String :=
  match pts with
  | [(x0, y0), (x1, y1), (x2, y2), (x3, y3)] =>
      let cs := [corrFromH H x0 y0, corrFromH H x1 y1,
                 corrFromH H x2 y2, corrFromH H x3 y3]
      match cs with
      | [c0, c1, c2, c3] =>
          let solved := normalizeMat (dltSolution c0 c1 c2 c3)
          obj [
            ("kind", strJson "dlt_solution"),
            ("id", strJson id),
            ("theorem", strJson thm),
            ("correspondences", corrListJson cs),
            ("expected", m3Json solved),
            ("expected_det_nonzero", boolJson (decide (det3 solved ≠ 0)))
          ]
      | _ =>
          obj [("kind", strJson "dlt_solution"), ("id", strJson id),
               ("theorem", strJson thm), ("correspondences", "[]"),
               ("expected", m3Json (m3 0 0 0 0 0 0 0 0 0)),
               ("expected_det_nonzero", boolJson false)]
  | _ => obj [("kind", strJson "dlt_solution"), ("id", strJson id),
              ("theorem", strJson thm), ("correspondences", "[]"),
              ("expected", m3Json (m3 0 0 0 0 0 0 0 0 0)),
              ("expected_det_nonzero", boolJson false)]

/-- A collinear-source DLT vector: the implementation must reject it. -/
def dltDegenerateVector (id : String) (cs : List CorrQ) : String :=
  obj [
    ("kind", strJson "dlt_degenerate"),
    ("id", strJson id),
    ("theorem", strJson "D4"),
    ("correspondences", corrListJson cs),
    ("expected_rejected", boolJson true)
  ]

def dltVectors : List String :=
  let H := m3 2 0 1 0 3 (-2) (1 / 10) (1 / 20) 1
  let Minv := m3 (1 / 1000) 0 (-10) 0 (1 / 1000) (-20) 0 0 1
  let Hlarge := matMul H Minv
  [dltGeneralVector "D3-general" "D3" H [(0, 0), (4, 0), (0, 3), (2, 2)],
   dltGeneralVector "D5-large-coordinates" "D5" Hlarge
     [(10000, 20000), (14000, 20000), (10000, 23000), (12000, 22000)],
   dltDegenerateVector "D4-collinear-sources" [
     { x := 0, y := 0, u := 0, v := 0 },
     { x := 1, y := 0, u := 2, v := 1 },
     { x := 2, y := 0, u := 4, v := 2 },
     { x := 0, y := 1, u := 1, v := 5 }]]

/-! ## M4 RANSAC vectors -/

/-- Binomial coefficient over the naturals. -/
def chooseNat : Nat → Nat → Nat
  | _, 0 => 1
  | 0, _ + 1 => 0
  | n + 1, k + 1 => chooseNat n k + chooseNat n (k + 1)

/-- Exact all-inlier sampling probability. -/
def sampleProb (n I k : Nat) : Q :=
  (chooseNat I k : Q) / (chooseNat n k : Q)

/-- Exact conservative bound `(I/n)^k`. -/
def boundProb (n I k : Nat) : Q := ((I : Q) / (n : Q)) ^ k

/-- Exact success probability after `N` iterations. -/
def successProb (q : Q) (N : Nat) : Q := 1 - (1 - q) ^ N

/-- Exact required iteration count, capped by a fuel argument. -/
def requiredItersAux (P q : Q) : Nat → Nat → Nat
  | 0, N => N
  | fuel + 1, N =>
      if successProb q N ≥ P then N else requiredItersAux P q fuel (N + 1)

def requiredIters (P q : Q) (cap : Nat) : Nat := requiredItersAux P q cap 0

def ransacProbabilityVector (id : String) (n I k : Nat) : String :=
  obj [
    ("kind", strJson "ransac_probability"),
    ("id", strJson id),
    ("theorem", strJson "R1"),
    ("n", natJson n),
    ("I", natJson I),
    ("k", natJson k),
    ("expected_favorable", natJson (chooseNat I k)),
    ("expected_total", natJson (chooseNat n k)),
    ("expected_q", ratJson (sampleProb n I k)),
    ("expected_bound", ratJson (boundProb n I k))
  ]

def ransacSuccessVector (id : String) (q : Q) (N : Nat) : String :=
  obj [
    ("kind", strJson "ransac_success"),
    ("id", strJson id),
    ("theorem", strJson "R3"),
    ("q", ratJson q),
    ("N", natJson N),
    ("expected", ratJson (successProb q N))
  ]

def ransacIterationsVector (id : String) (P q : Q) (cap : Nat) : String :=
  obj [
    ("kind", strJson "ransac_iterations"),
    ("id", strJson id),
    ("theorem", strJson "R5"),
    ("P", ratJson P),
    ("q", ratJson q),
    ("cap", natJson cap),
    ("expected", natJson (requiredIters P q cap))
  ]

/-- R2: the conservative bound on the all-inlier probability. -/
def ransacBoundVector (id : String) (n I k : Nat) : String :=
  obj [
    ("kind", strJson "ransac_bound"),
    ("id", strJson id),
    ("theorem", strJson "R2"),
    ("n", natJson n),
    ("I", natJson I),
    ("k", natJson k),
    ("expected_q", ratJson (sampleProb n I k)),
    ("expected_bound", ratJson (boundProb n I k))
  ]

/-- R4: the three monotonicity directions. -/
def ransacMonotonicityVector (id : String) (q : Q) (N_low N_high : Nat)
    (n I_low I_high k_low k_high : Nat) : String :=
  obj [
    ("kind", strJson "ransac_monotonicity"),
    ("id", strJson id),
    ("theorem", strJson "R4"),
    ("q", ratJson q),
    ("N_low", natJson N_low),
    ("N_high", natJson N_high),
    ("n", natJson n),
    ("I_low", natJson I_low),
    ("I_high", natJson I_high),
    ("k_low", natJson k_low),
    ("k_high", natJson k_high),
    ("expected_monotone", boolJson true)
  ]

def ransacVectors : List String :=
  [ransacProbabilityVector "R1-100-40-4" 100 40 4,
   ransacProbabilityVector "R1-20-7-3" 20 7 3,
   ransacProbabilityVector "R1-10-4-2" 10 4 2,
   ransacBoundVector "R2-100-40-4" 100 40 4,
   ransacBoundVector "R2-10-4-2" 10 4 2,
   ransacSuccessVector "R3-half-5" (1 / 2) 5,
   ransacSuccessVector "R3-nine-tenths-3" (9 / 10) 3,
   ransacMonotonicityVector "R4-half" (1 / 2) 2 5 100 20 40 2 4,
   ransacIterationsVector "R5-99-50" (99 / 100) (1 / 2) 1000,
   ransacIterationsVector "R5-999-50" (999 / 1000) (1 / 2) 1000,
   ransacIterationsVector "R5-99-90" (99 / 100) (9 / 10) 1000]

/-- Render pre-serialized vector objects as the golden-vector JSON document. -/
def renderJson (vectors : List String) : String :=
  "{\"format\":\"" ++ formatId ++ "\",\"version\":" ++ toString formatVersion ++
    ",\"vectors\":[" ++ String.intercalate "," vectors ++ "]}"

/-- The golden-vector document (geometry, DLT, RANSAC). -/
def allJson : String := renderJson (geometryVectors ++ dltVectors ++ ransacVectors)

end PatternMatch.Golden
