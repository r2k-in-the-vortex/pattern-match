module

import PatternMatch.Geometry
import Mathlib.Basic.Real.Basic
import Mathlib.LinearAlgebra.FiniteDimensional.Lemmas
import Mathlib.LinearAlgebra.Matrix.Determinant.Basic
import Mathlib.LinearAlgebra.Matrix.Diagonal
import Mathlib.LinearAlgebra.Matrix.NonsingularInverse
import Mathlib.LinearAlgebra.Matrix.ToLin

/-!
# DLT specification (M4)

Formal statements and proofs from README §10.3:

* the DLT constraint rows and `A · vec(H) = 0` (D1);
* rank–nullity for four correspondences (D2);
* uniqueness from general position (D3);
* collinear degeneracy (D4);
* Hartley normalization equivalence (D5).

D6 (noisy least-squares optimality) is a documented follow-on and is outside the
no-`sorry` gate until promoted.
-/

namespace PatternMatch.DLT

open Matrix
open scoped Matrix

/-- A homogeneous point in the projective plane. -/
abbrev Point : Type := Fin 3 → ℝ

/-- A projective homography. -/
abbrev Mat : Type := Matrix (Fin 3) (Fin 3) ℝ

/-- A homography flattened in row-major order, matching the DLT parameter order. -/
abbrev HVec : Type := Fin 9 → ℝ

/-- An affine correspondence `(x, y) ↔ (u, v)`. -/
structure Corr where
  /-- Source x. -/
  x : ℝ
  /-- Source y. -/
  y : ℝ
  /-- Target u. -/
  u : ℝ
  /-- Target v. -/
  v : ℝ

/-- The source affine point `(x, y, 1)`. -/
def src (c : Corr) : Point := ![c.x, c.y, 1]

/-- The target affine point `(u, v, 1)`. -/
def dst (c : Corr) : Point := ![c.u, c.v, 1]

/-- Row-major flattening of a homography. -/
def vecH (H : Mat) : HVec :=
  ![H 0 0, H 0 1, H 0 2, H 1 0, H 1 1, H 1 2, H 2 0, H 2 1, H 2 2]

/-- Inverse of row-major flattening. -/
def matOfVec (v : HVec) : Mat :=
  !![v 0, v 1, v 2; v 3, v 4, v 5; v 6, v 7, v 8]

/-- First DLT row for a correspondence:
`[-x, -y, -1, 0, 0, 0, ux, uy, u]`. -/
def dltRow1 (c : Corr) : HVec :=
  ![-c.x, -c.y, -1, 0, 0, 0, c.u * c.x, c.u * c.y, c.u]

/-- Second DLT row for a correspondence:
`[0, 0, 0, -x, -y, -1, vx, vy, v]`. -/
def dltRow2 (c : Corr) : HVec :=
  ![0, 0, 0, -c.x, -c.y, -1, c.v * c.x, c.v * c.y, c.v]

/-- The `8 × 9` DLT system for four correspondences.  Rows `2i` and `2i+1` are
the two equations of correspondence `i`. -/
def dltSystem4 (cs : Fin 4 → Corr) : Matrix (Fin 8) (Fin 9) ℝ :=
  Matrix.of
    ![dltRow1 (cs 0), dltRow2 (cs 0),
      dltRow1 (cs 1), dltRow2 (cs 1),
      dltRow1 (cs 2), dltRow2 (cs 2),
      dltRow1 (cs 3), dltRow2 (cs 3)]

/-- The DLT equations for one correspondence, written in point coordinates.  The
third homogeneous coordinate of the source is `1`, so these are exactly the two
DLT rows: `(H p)₀ = u (H p)₂` and `(H p)₁ = v (H p)₂`. -/
def Satisfies (H : Mat) (c : Corr) : Prop :=
  (H *ᵥ src c) 0 = c.u * (H *ᵥ src c) 2 ∧
    (H *ᵥ src c) 1 = c.v * (H *ᵥ src c) 2

/-- A homography satisfies a family of correspondences. -/
def SatisfiesAll (H : Mat) (cs : Fin 4 → Corr) : Prop := ∀ i, Satisfies H (cs i)

/-- The matrix whose columns are the points of `p`. -/
def colsOf (p : Fin 3 → Point) : Mat := fun i j => p j i

/-- The matrix whose columns are three homogeneous points. -/
def cols3 (p q r : Point) : Mat := colsOf ![p, q, r]

/-- The determinant of `cols3` is zero exactly when the three projective points
are collinear. -/
def NoThreeCollinear (p0 p1 p2 p3 : Point) : Prop :=
  (cols3 p0 p1 p2).det ≠ 0 ∧ (cols3 p0 p1 p3).det ≠ 0 ∧
    (cols3 p0 p2 p3).det ≠ 0 ∧ (cols3 p1 p2 p3).det ≠ 0

/-! ## Small helper lemmas -/

/-- Row extraction from the transposed family matrix. -/
@[simp] theorem colsOf_transpose_apply (p : Fin 3 → Point) (i : Fin 3) :
    (colsOf p)ᵀ i = p i := by
  ext j
  rfl

/-- Row extraction from the transposed column matrix. -/
@[simp] theorem cols3_transpose_apply (p q r : Point) (i : Fin 3) :
    (cols3 p q r)ᵀ i = ![p, q, r] i := by
  ext j
  rfl

/-- Linear combination of the columns of `cols3 p q r`. -/
theorem cols3_mulVec (p q r : Point) (a b c : ℝ) :
    cols3 p q r *ᵥ ![a, b, c] = a • p + b • q + c • r := by
  ext i
  simp [cols3, colsOf, Matrix.mulVec, dotProduct, Fin.sum_univ_three,
    vecHead, vecTail]
  ring

/-- If the third column is a linear combination of the first two, the determinant
vanishes. -/
theorem det_cols3_eq_zero_of_combination (p q r : Point) (a b : ℝ)
    (h : r = a • p + b • q) :
    (cols3 p q r).det = 0 := by
  apply Matrix.det_eq_zero_of_not_linearIndependent_cols
  rw [Fintype.not_linearIndependent_iff]
  refine ⟨![a, b, -1], ?_, ⟨2, by simp⟩⟩
  have hsum : a • p + b • q + (-1 : ℝ) • r = 0 := by
    rw [h]
    module
  simpa [Fin.sum_univ_three] using hsum

/-- The two DLT rows are equivalent to `H p = w • q` for some scalar `w`. -/
theorem satisfies_iff (H : Mat) (c : Corr) :
    Satisfies H c ↔ ∃ w : ℝ, H *ᵥ src c = w • dst c := by
  constructor
  · intro h
    refine ⟨(H *ᵥ src c) 2, ?_⟩
    ext i
    fin_cases i
    · simpa [Satisfies, dst, mul_comm] using h.1
    · simpa [Satisfies, dst, mul_comm] using h.2
    · simp [dst]
  · rintro ⟨w, hw⟩
    constructor
    · have h0 := congrFun hw 0
      have h2 := congrFun hw 2
      simp [dst] at h0 h2
      rw [h0, h2]
      ring
    · have h1 := congrFun hw 1
      have h2 := congrFun hw 2
      simp [dst] at h1 h2
      rw [h1, h2]
      ring

/-- The first DLT row annihilates `vecH H` exactly when `H` maps the source of
`c` to a scalar multiple of its target. -/
theorem row1_dot_vecH (c : Corr) (H : Mat) :
    (∑ j, dltRow1 c j * vecH H j) =
      -((H *ᵥ src c) 0) + c.u * (H *ᵥ src c) 2 := by
  simp [dltRow1, vecH, src, Matrix.mulVec, Fin.sum_univ_three,
    Fin.sum_univ_succ, vecHead, vecTail]
  ring

/-- The second DLT row annihilates `vecH H` exactly when `H` maps the source of
`c` to a scalar multiple of its target. -/
theorem row2_dot_vecH (c : Corr) (H : Mat) :
    (∑ j, dltRow2 c j * vecH H j) =
      -((H *ᵥ src c) 1) + c.v * (H *ᵥ src c) 2 := by
  simp [dltRow2, vecH, src, Matrix.mulVec, Fin.sum_univ_three,
    Fin.sum_univ_succ, vecHead, vecTail]
  ring

/-! ## D1: exact correspondences satisfy the DLT rows -/

/-- **D1.** If `H` maps the source of a correspondence to a scalar multiple of
its target, both DLT rows annihilate `vecH H`. -/
theorem dltRows_eq_zero_of_satisfies (c : Corr) (H : Mat) (h : Satisfies H c) :
    (∑ j, dltRow1 c j * vecH H j) = 0 ∧
      (∑ j, dltRow2 c j * vecH H j) = 0 := by
  constructor
  · rw [row1_dot_vecH]
    rw [h.1]
    ring
  · rw [row2_dot_vecH]
    rw [h.2]
    ring

/-- **D1.** In matrix form: exact correspondences put `vecH H` in the kernel of
the DLT system. -/
theorem D1 (H : Mat) (cs : Fin 4 → Corr) (h : SatisfiesAll H cs) :
    dltSystem4 cs *ᵥ vecH H = 0 := by
  ext i
  fin_cases i
  · simpa [dltSystem4, Matrix.mulVec, dotProduct] using
      (dltRows_eq_zero_of_satisfies (cs 0) H (h 0)).1
  · simpa [dltSystem4, Matrix.mulVec, dotProduct] using
      (dltRows_eq_zero_of_satisfies (cs 0) H (h 0)).2
  · simpa [dltSystem4, Matrix.mulVec, dotProduct] using
      (dltRows_eq_zero_of_satisfies (cs 1) H (h 1)).1
  · simpa [dltSystem4, Matrix.mulVec, dotProduct] using
      (dltRows_eq_zero_of_satisfies (cs 1) H (h 1)).2
  · simpa [dltSystem4, Matrix.mulVec, dotProduct] using
      (dltRows_eq_zero_of_satisfies (cs 2) H (h 2)).1
  · simpa [dltSystem4, Matrix.mulVec, dotProduct] using
      (dltRows_eq_zero_of_satisfies (cs 2) H (h 2)).2
  · simpa [dltSystem4, Matrix.mulVec, dotProduct] using
      (dltRows_eq_zero_of_satisfies (cs 3) H (h 3)).1
  · simpa [dltSystem4, Matrix.mulVec, dotProduct] using
      (dltRows_eq_zero_of_satisfies (cs 3) H (h 3)).2

/-! ## D2: the four-point system always has a non-trivial kernel -/

/-- **D2.** The coefficient matrix for four correspondences is `8 × 9`, so its
kernel is non-trivial: there is a non-zero flattened matrix satisfying all eight
equations. -/
theorem D2 (cs : Fin 4 → Corr) :
    ∃ v : HVec, v ≠ 0 ∧ dltSystem4 cs *ᵥ v = 0 := by
  let A := dltSystem4 cs
  let f : HVec →ₗ[ℝ] (Fin 8 → ℝ) := A.mulVecLin
  have hdim : Module.finrank ℝ (Fin 8 → ℝ) < Module.finrank ℝ HVec := by
    simp [HVec]
  have hker : LinearMap.ker f ≠ ⊥ := LinearMap.ker_ne_bot_of_finrank_lt hdim
  rw [Submodule.ne_bot_iff] at hker
  obtain ⟨v, hvker, hvne⟩ := hker
  exact ⟨v, hvne, by simpa [f, A, Matrix.mulVecLin_apply] using hvker⟩

/-! ## D3: uniqueness from general position -/

/-- A diagonal matrix of a 3-vector. -/
def diag3 (w : Point) : Mat := Matrix.diagonal w

/-- Linear combination of the columns of `colsOf p`. -/
theorem colsOf_mulVec (p : Fin 3 → Point) (v : Point) :
    colsOf p *ᵥ v = v 0 • p 0 + v 1 • p 1 + v 2 • p 2 := by
  ext i
  simp [colsOf, Matrix.mulVec, dotProduct, Fin.sum_univ_three,
    vecHead, vecTail]
  ring

/-- Coordinates of a point in the projective basis given by three points. -/
noncomputable def coords (p : Fin 3 → Point) (x : Point) : Point :=
  (colsOf p)⁻¹ *ᵥ x

/-- Multiplying the basis matrix by the coordinates recovers the point. -/
theorem colsOf_mulVec_coords (p : Fin 3 → Point) (x : Point)
    (hP : (colsOf p).det ≠ 0) :
    colsOf p *ᵥ coords p x = x := by
  unfold coords
  rw [Matrix.mulVec_mulVec]
  rw [Matrix.mul_nonsing_inv _ (isUnit_iff_ne_zero.mpr hP)]
  simp

/-- A point is the corresponding linear combination of the basis points. -/
theorem coords_combination (p : Fin 3 → Point) (x : Point)
    (hP : (colsOf p).det ≠ 0) :
    x = coords p x 0 • p 0 + coords p x 1 • p 1 + coords p x 2 • p 2 := by
  rw [← colsOf_mulVec, colsOf_mulVec_coords p x hP]

/-- The first three points of a quadruple, as a 3-family. -/
def prefix3 (p : Fin 4 → Point) : Fin 3 → Point := ![p 0, p 1, p 2]

/-- Casting an index into the quadruple agrees with the prefix. -/
theorem prefix3_castSucc (p : Fin 4 → Point) (j : Fin 3) :
    p j.castSucc = prefix3 p j := by
  fin_cases j <;> rfl

/-- `cols3` is `colsOf` on the prefix of four points. -/
theorem cols3_eq_colsOf (p : Fin 4 → Point) :
    cols3 (p 0) (p 1) (p 2) = colsOf (prefix3 p) := rfl

/-- In a general-position quadruple, the coordinates of the fourth point in the
basis of the first three are all non-zero. -/
theorem coords_ne_zero_of_noThreeCollinear
    (p : Fin 4 → Point) (h : NoThreeCollinear (p 0) (p 1) (p 2) (p 3))
    (i : Fin 3) :
    coords (prefix3 p) (p 3) i ≠ 0 := by
  have hP : (colsOf (prefix3 p)).det ≠ 0 := h.1
  have hcomb : p 3 = coords (prefix3 p) (p 3) 0 • p 0 +
      coords (prefix3 p) (p 3) 1 • p 1 +
      coords (prefix3 p) (p 3) 2 • p 2 := by
    simpa [prefix3] using coords_combination (prefix3 p) (p 3) hP
  fin_cases i
  · intro h0
    have h0' : coords (prefix3 p) (p 3) 0 = 0 := by simpa using h0
    have hzero : (cols3 (p 1) (p 2) (p 3)).det = 0 := by
      apply det_cols3_eq_zero_of_combination (p 1) (p 2) (p 3)
        (coords (prefix3 p) (p 3) 1)
        (coords (prefix3 p) (p 3) 2)
      calc p 3 = coords (prefix3 p) (p 3) 0 • p 0 +
            coords (prefix3 p) (p 3) 1 • p 1 +
            coords (prefix3 p) (p 3) 2 • p 2 := hcomb
        _ = coords (prefix3 p) (p 3) 1 • p 1 +
            coords (prefix3 p) (p 3) 2 • p 2 := by rw [h0']; simp
    exact h.2.2.2 hzero
  · intro h1
    have h1' : coords (prefix3 p) (p 3) 1 = 0 := by simpa using h1
    have hzero : (cols3 (p 0) (p 2) (p 3)).det = 0 := by
      apply det_cols3_eq_zero_of_combination (p 0) (p 2) (p 3)
        (coords (prefix3 p) (p 3) 0)
        (coords (prefix3 p) (p 3) 2)
      calc p 3 = coords (prefix3 p) (p 3) 0 • p 0 +
            coords (prefix3 p) (p 3) 1 • p 1 +
            coords (prefix3 p) (p 3) 2 • p 2 := hcomb
        _ = coords (prefix3 p) (p 3) 0 • p 0 +
            coords (prefix3 p) (p 3) 2 • p 2 := by rw [h1']; simp
    exact h.2.2.1 hzero
  · intro h2
    have h2' : coords (prefix3 p) (p 3) 2 = 0 := by simpa using h2
    have hzero : (cols3 (p 0) (p 1) (p 3)).det = 0 := by
      apply det_cols3_eq_zero_of_combination (p 0) (p 1) (p 3)
        (coords (prefix3 p) (p 3) 0)
        (coords (prefix3 p) (p 3) 1)
      calc p 3 = coords (prefix3 p) (p 3) 0 • p 0 +
            coords (prefix3 p) (p 3) 1 • p 1 +
            coords (prefix3 p) (p 3) 2 • p 2 := hcomb
        _ = coords (prefix3 p) (p 3) 0 • p 0 +
            coords (prefix3 p) (p 3) 1 • p 1 := by rw [h2']; simp
    exact h.2.1 hzero

/-- The determinant of a diagonal matrix with non-zero diagonal is non-zero. -/
theorem det_diag3_ne_zero (w : Point) (hw : ∀ i, w i ≠ 0) :
    (diag3 w).det ≠ 0 := by
  rw [diag3, Matrix.det_diagonal]
  exact Finset.prod_ne_zero_iff.mpr (fun i _ => hw i)

/-- Applying `Q * D * P⁻¹` to `P * v` collapses to `Q * (D * v)`. -/
theorem mul_inv_mulVec (P Q D : Mat) (v : Point) (hP : P.det ≠ 0) :
    (Q * D * P⁻¹) *ᵥ (P *ᵥ v) = Q *ᵥ (D *ᵥ v) := by
  rw [Matrix.mulVec_mulVec, Matrix.mul_assoc,
    Matrix.nonsing_inv_mul _ (isUnit_iff_ne_zero.mpr hP), Matrix.mul_one,
    Matrix.mulVec_mulVec]

/-- If `H` sends the three basis points to `w j • q j`, it is the product of
the target basis, the diagonal scale matrix, and the inverse source basis. -/
theorem representation (p q : Fin 3 → Point) (H : Mat) (w : Point)
    (hP : (colsOf p).det ≠ 0)
    (hw : ∀ j, H *ᵥ p j = w j • q j) :
    H = colsOf q * diag3 w * (colsOf p)⁻¹ := by
  have hHP : H * colsOf p = colsOf q * diag3 w := by
    ext i j
    have h := congrFun (hw j) i
    simp [colsOf, diag3, Matrix.mul_apply, Matrix.diagonal, Matrix.mulVec,
      dotProduct, Fin.sum_univ_three, Pi.smul_apply, smul_eq_mul,
      vecHead, vecTail] at h ⊢
    simpa [mul_comm, mul_left_comm, mul_assoc] using h
  calc H = H * colsOf p * (colsOf p)⁻¹ :=
        (Matrix.mul_nonsing_inv_cancel_right (A := colsOf p) H
          (isUnit_iff_ne_zero.mpr hP)).symm
    _ = (colsOf q * diag3 w) * (colsOf p)⁻¹ := by rw [hHP]

/-- The kernel of the DLT system is the set of homographies satisfying the four
correspondences. -/
theorem satisfies_of_rows_eq_zero (c : Corr) (H : Mat)
    (h1 : ∑ j, dltRow1 c j * vecH H j = 0)
    (h2 : ∑ j, dltRow2 c j * vecH H j = 0) :
    Satisfies H c := by
  constructor
  · rw [row1_dot_vecH] at h1
    linarith
  · rw [row2_dot_vecH] at h2
    linarith

/-- Membership in the DLT kernel is equivalent to satisfying every
correspondence. -/
theorem satisfiesAll_of_dltSystem4 (cs : Fin 4 → Corr) (H : Mat)
    (h : dltSystem4 cs *ᵥ vecH H = 0) :
    SatisfiesAll H cs := by
  have h0 := congrFun h 0
  have h1 := congrFun h 1
  have h2 := congrFun h 2
  have h3 := congrFun h 3
  have h4 := congrFun h 4
  have h5 := congrFun h 5
  have h6 := congrFun h 6
  have h7 := congrFun h 7
  simp [dltSystem4, Matrix.mulVec, dotProduct] at h0 h1 h2 h3 h4 h5 h6 h7
  intro i
  fin_cases i
  · exact satisfies_of_rows_eq_zero (cs 0) H h0 h1
  · exact satisfies_of_rows_eq_zero (cs 1) H h2 h3
  · exact satisfies_of_rows_eq_zero (cs 2) H h4 h5
  · exact satisfies_of_rows_eq_zero (cs 3) H h6 h7

/-- **D3 (core).** In general position, the homographies sending four points to
four projectively corresponding points form a one-dimensional space spanned by
an invertible solution. -/
theorem exists_unique_solution (p q : Fin 4 → Point)
    (hsrc : NoThreeCollinear (p 0) (p 1) (p 2) (p 3))
    (hdst : NoThreeCollinear (q 0) (q 1) (q 2) (q 3)) :
    ∃ H0 : Mat, H0.det ≠ 0 ∧
      (∀ i, ∃ w : ℝ, H0 *ᵥ p i = w • q i) ∧
      ∀ H : Mat, (∀ i, ∃ w : ℝ, H *ᵥ p i = w • q i) →
        ∃ lam : ℝ, H = lam • H0 := by
  let p3 : Fin 3 → Point := prefix3 p
  let q3 : Fin 3 → Point := prefix3 q
  have hP : (colsOf p3).det ≠ 0 := hsrc.1
  have hQ : (colsOf q3).det ≠ 0 := hdst.1
  let α : Point := coords p3 (p 3)
  let β : Point := coords q3 (q 3)
  have hα : ∀ i, α i ≠ 0 := fun i =>
    coords_ne_zero_of_noThreeCollinear p hsrc i
  have hβ : ∀ i, β i ≠ 0 := fun i =>
    coords_ne_zero_of_noThreeCollinear q hdst i
  let w0 : Point := fun i => β i / α i
  have hw0 : ∀ i, w0 i * α i = β i := by
    intro i
    exact div_mul_cancel₀ (β i) (hα i)
  let H0 : Mat := colsOf q3 * diag3 w0 * (colsOf p3)⁻¹
  have hD0 : (diag3 w0).det ≠ 0 :=
    det_diag3_ne_zero w0 (fun i => div_ne_zero (hβ i) (hα i))
  have hH0det : H0.det ≠ 0 := by
    have hH0P : H0 * colsOf p3 = colsOf q3 * diag3 w0 := by
      rw [show H0 = colsOf q3 * diag3 w0 * (colsOf p3)⁻¹ from rfl,
        Matrix.mul_assoc,
        Matrix.nonsing_inv_mul _ (isUnit_iff_ne_zero.mpr hP),
        Matrix.mul_one]
    have hprod : (colsOf q3 * diag3 w0).det ≠ 0 := by
      rw [Matrix.det_mul]
      exact mul_ne_zero hQ hD0
    intro h
    apply hprod
    rw [← hH0P, Matrix.det_mul, h, zero_mul]
  have hPj : ∀ j : Fin 3, colsOf p3 *ᵥ Pi.single j (1 : ℝ) = p3 j := by
    intro j
    rw [colsOf_mulVec]
    fin_cases j <;> simp
  have hQj : ∀ j : Fin 3, colsOf q3 *ᵥ Pi.single j (1 : ℝ) = q3 j := by
    intro j
    rw [colsOf_mulVec]
    fin_cases j <;> simp
  have hH0basis : ∀ j : Fin 3, H0 *ᵥ p3 j = w0 j • q3 j := by
    intro j
    calc H0 *ᵥ p3 j
        = (colsOf q3 * diag3 w0 * (colsOf p3)⁻¹) *ᵥ
            (colsOf p3 *ᵥ Pi.single j (1 : ℝ)) := by
            rw [show H0 = colsOf q3 * diag3 w0 * (colsOf p3)⁻¹ from rfl,
              ← hPj j]
      _ = colsOf q3 *ᵥ (diag3 w0 *ᵥ Pi.single j (1 : ℝ)) := by
            rw [mul_inv_mulVec _ _ _ _ hP]
      _ = colsOf q3 *ᵥ (w0 j • Pi.single j (1 : ℝ)) := by
            rw [diag3, Matrix.diagonal_mulVec_single, mul_one]
            congr 1
            simpa using (Pi.single_smul' j (w0 j) (1 : ℝ))
      _ = w0 j • (colsOf q3 *ᵥ Pi.single j (1 : ℝ)) := by
            rw [Matrix.mulVec_smul]
      _ = w0 j • q3 j := by rw [hQj j]
  have hH0p3 : H0 *ᵥ p 3 = (1 : ℝ) • q 3 := by
    have hαeq : colsOf p3 *ᵥ α = p 3 := colsOf_mulVec_coords p3 (p 3) hP
    have hβeq : colsOf q3 *ᵥ β = q 3 := colsOf_mulVec_coords q3 (q 3) hQ
    calc H0 *ᵥ p 3
        = (colsOf q3 * diag3 w0 * (colsOf p3)⁻¹) *ᵥ
            (colsOf p3 *ᵥ α) := by
            rw [show H0 = colsOf q3 * diag3 w0 * (colsOf p3)⁻¹ from rfl,
              ← hαeq]
      _ = colsOf q3 *ᵥ (diag3 w0 *ᵥ α) := by
            rw [mul_inv_mulVec _ _ _ _ hP]
      _ = colsOf q3 *ᵥ β := by
            congr 1
            ext i
            rw [diag3, Matrix.mulVec_diagonal, hw0 i]
      _ = (1 : ℝ) • q 3 := by
            rw [hβeq]
            simp
  have hH0sat : ∀ i, ∃ w : ℝ, H0 *ᵥ p i = w • q i := by
    intro i
    fin_cases i
    · exact ⟨w0 0, by simpa [p3, q3, prefix3] using hH0basis 0⟩
    · exact ⟨w0 1, by simpa [p3, q3, prefix3] using hH0basis 1⟩
    · exact ⟨w0 2, by simpa [p3, q3, prefix3] using hH0basis 2⟩
    · exact ⟨1, hH0p3⟩
  refine ⟨H0, hH0det, hH0sat, ?_⟩
  intro H hH
  choose w hw using fun j : Fin 3 => hH j.castSucc
  obtain ⟨w3, hw3⟩ := hH 3
  have hrepr : H = colsOf q3 * diag3 w * (colsOf p3)⁻¹ :=
    representation p3 q3 H w hP (fun j => by
      simpa [p3, q3, prefix3_castSucc] using hw j)
  have hαeq : colsOf p3 *ᵥ α = p 3 := colsOf_mulVec_coords p3 (p 3) hP
  have hβeq : colsOf q3 *ᵥ β = q 3 := colsOf_mulVec_coords q3 (q 3) hQ
  have hleft : colsOf q3 *ᵥ (diag3 w *ᵥ α) = w3 • (colsOf q3 *ᵥ β) := by
    calc colsOf q3 *ᵥ (diag3 w *ᵥ α)
        = (colsOf q3 * diag3 w * (colsOf p3)⁻¹) *ᵥ
            (colsOf p3 *ᵥ α) := by
            rw [mul_inv_mulVec _ _ _ _ hP]
      _ = H *ᵥ p 3 := by rw [← hrepr, hαeq]
      _ = w3 • q 3 := hw3
      _ = w3 • (colsOf q3 *ᵥ β) := by rw [hβeq]
  have hQinj : Function.Injective (colsOf q3).mulVec :=
    Matrix.mulVec_injective_iff_isUnit.mpr
      ((Matrix.isUnit_iff_isUnit_det _).mpr (isUnit_iff_ne_zero.mpr hQ))
  have heq : diag3 w *ᵥ α = w3 • β := by
    apply hQinj
    rw [← Matrix.mulVec_smul] at hleft
    exact hleft
  have hrel : ∀ i, w i = w3 * w0 i := by
    intro i
    have hi := congrFun heq i
    rw [diag3, Matrix.mulVec_diagonal] at hi
    simp only [Pi.smul_apply, smul_eq_mul] at hi
    rw [← hw0 i] at hi
    have hi' : w i * α i = (w3 * w0 i) * α i := by
      calc w i * α i = w3 * (w0 i * α i) := hi
        _ = (w3 * w0 i) * α i := by ring
    exact mul_right_cancel₀ (hα i) hi'
  have hdiag : diag3 w = w3 • diag3 w0 := by
    ext i j
    by_cases hij : i = j
    · subst hij
      simp [diag3, Matrix.diagonal, hrel i]
    · simp [diag3, Matrix.diagonal, hij]
  refine ⟨w3, ?_⟩
  calc H = colsOf q3 * diag3 w * (colsOf p3)⁻¹ := hrepr
    _ = colsOf q3 * (w3 • diag3 w0) * (colsOf p3)⁻¹ := by rw [hdiag]
    _ = w3 • (colsOf q3 * diag3 w0 * (colsOf p3)⁻¹) := by
          rw [Matrix.mul_smul, Matrix.smul_mul]
    _ = w3 • H0 := by
          rw [show H0 = colsOf q3 * diag3 w0 * (colsOf p3)⁻¹ from rfl]

/-! ## Flattening lemmas and the D3 kernel/rank corollaries -/

/-- Re-flattening a matrix is the identity. -/
@[simp] theorem matOfVec_vecH (H : Mat) : matOfVec (vecH H) = H := by
  ext i j
  fin_cases i <;> fin_cases j <;> rfl

/-- Reconstructing from a flattened matrix is the identity. -/
@[simp] theorem vecH_matOfVec (v : HVec) : vecH (matOfVec v) = v := by
  ext i
  fin_cases i <;> rfl

/-- Flattening commutes with scalar multiplication. -/
theorem vecH_smul (c : ℝ) (H : Mat) : vecH (c • H) = c • vecH H := by
  ext i
  fin_cases i <;> simp [vecH, Pi.smul_apply, smul_eq_mul]

/-- **D3.** For four correspondences in general position, the DLT kernel is the
line spanned by an invertible solution. -/
theorem D3 (cs : Fin 4 → Corr)
    (hsrc : NoThreeCollinear (src (cs 0)) (src (cs 1)) (src (cs 2)) (src (cs 3)))
    (hdst : NoThreeCollinear (dst (cs 0)) (dst (cs 1)) (dst (cs 2)) (dst (cs 3))) :
    ∃ H0 : Mat, H0.det ≠ 0 ∧
      dltSystem4 cs *ᵥ vecH H0 = 0 ∧
      ∀ H, dltSystem4 cs *ᵥ vecH H = 0 → ∃ lam : ℝ, H = lam • H0 := by
  obtain ⟨H0, hdet, hsat, huniq⟩ :=
    exists_unique_solution (fun i => src (cs i)) (fun i => dst (cs i)) hsrc hdst
  refine ⟨H0, hdet, ?_, ?_⟩
  · apply D1
    intro i
    exact (satisfies_iff H0 (cs i)).mpr (hsat i)
  · intro H hH
    apply huniq H
    intro i
    exact (satisfies_iff H (cs i)).mp (satisfiesAll_of_dltSystem4 cs H hH i)

/-- **D3.** The DLT kernel has dimension one in general position. -/
theorem D3_kernel_finrank (cs : Fin 4 → Corr)
    (hsrc : NoThreeCollinear (src (cs 0)) (src (cs 1)) (src (cs 2)) (src (cs 3)))
    (hdst : NoThreeCollinear (dst (cs 0)) (dst (cs 1)) (dst (cs 2)) (dst (cs 3))) :
    Module.finrank ℝ (LinearMap.ker (dltSystem4 cs).mulVecLin) = 1 := by
  obtain ⟨H0, hdet, hker, huniq⟩ := D3 cs hsrc hdst
  have hH0ne : vecH H0 ≠ 0 := by
    intro hzero
    have hinj : Function.Injective vecH := by
      intro a b hab
      calc a = matOfVec (vecH a) := (matOfVec_vecH a).symm
        _ = matOfVec (vecH b) := by rw [hab]
        _ = b := matOfVec_vecH b
    have hmat : H0 = 0 := by
      apply hinj
      rw [hzero]
      ext i
      fin_cases i <;> simp [vecH]
    rw [hmat] at hdet
    simp at hdet
  have hker_eq : LinearMap.ker (dltSystem4 cs).mulVecLin = ℝ ∙ vecH H0 := by
    apply le_antisymm
    · intro v hv
      rw [Submodule.mem_span_singleton]
      have hv0 : dltSystem4 cs *ᵥ v = 0 := by
        simpa [Matrix.mulVecLin_apply] using hv
      have hv0' : dltSystem4 cs *ᵥ vecH (matOfVec v) = 0 := by
        simpa using hv0
      obtain ⟨lam, hlam⟩ := huniq (matOfVec v) hv0'
      refine ⟨lam, ?_⟩
      calc lam • vecH H0 = vecH (lam • H0) := (vecH_smul lam H0).symm
        _ = vecH (matOfVec v) := by rw [hlam]
        _ = v := vecH_matOfVec v
    · rw [Submodule.span_le]
      intro v hv
      rw [Set.mem_singleton_iff] at hv
      subst hv
      change dltSystem4 cs *ᵥ vecH H0 = 0
      exact hker
  rw [hker_eq]
  exact finrank_span_singleton hH0ne

/-- **D3.** The DLT coefficient matrix has rank eight in general position. -/
theorem D3_rank (cs : Fin 4 → Corr)
    (hsrc : NoThreeCollinear (src (cs 0)) (src (cs 1)) (src (cs 2)) (src (cs 3)))
    (hdst : NoThreeCollinear (dst (cs 0)) (dst (cs 1)) (dst (cs 2)) (dst (cs 3))) :
    Module.finrank ℝ (LinearMap.range (dltSystem4 cs).mulVecLin) = 8 := by
  have hk := D3_kernel_finrank cs hsrc hdst
  have hrank := LinearMap.finrank_range_add_finrank_ker (dltSystem4 cs).mulVecLin
  have hdim : Module.finrank ℝ HVec = 9 := Module.finrank_fin_fun ℝ
  rw [hdim, hk] at hrank
  omega

/-! ## D4: collinear degeneracy -/

/-- The rank-one matrix `u ⊗ φ`. -/
noncomputable def rankOne (u : Point) (φ : Point →ₗ[ℝ] ℝ) : Mat :=
  Matrix.of fun i j => u i * φ (Pi.single j (1 : ℝ))

/-- A rank-one matrix acts as `x ↦ φ x • u`. -/
theorem rankOne_mulVec (u : Point) (φ : Point →ₗ[ℝ] ℝ) (x : Point) :
    rankOne u φ *ᵥ x = φ x • u := by
  have hsingle : ∀ j : Fin 3,
      (fun k => if j = k then (1 : ℝ) else 0) = Pi.single j 1 := by
    intro j
    funext k
    rw [Pi.single_apply]
    by_cases h : k = j
    · subst h
      simp
    · have h' : ¬ j = k := fun hjk => h hjk.symm
      simp [h, h']
  ext i
  rw [LinearMap.pi_apply_eq_sum_univ φ x]
  simp only [rankOne, Matrix.mulVec, dotProduct, Finset.mul_sum, hsingle,
    Matrix.of_apply, Pi.smul_apply, smul_eq_mul]
  rw [Finset.sum_mul]
  apply Finset.sum_congr rfl
  intro j _
  ring

/-- A rank-one matrix with non-zero factors is non-zero. -/
theorem rankOne_ne_zero (u : Point) (φ : Point →ₗ[ℝ] ℝ)
    (hu : u ≠ 0) (hφ : φ ≠ 0) :
    rankOne u φ ≠ 0 := by
  intro hzero
  apply hφ
  apply LinearMap.ext
  intro x
  have hzero' : rankOne u φ *ᵥ x = 0 := by rw [hzero]; simp
  rw [rankOne_mulVec] at hzero'
  exact (smul_eq_zero.mp hzero').resolve_right hu

/-- **D4.** If three source points are collinear, a consistent configuration
does not determine a unique homography: there is a solution not proportional
to the given invertible solution. -/
theorem collinear_degenerate
    (p q : Fin 4 → Point) (H0 : Mat)
    (hH0 : ∀ i, ∃ w : ℝ, H0 *ᵥ p i = w • q i)
    (hH0det : H0.det ≠ 0)
    (hp0 : p 0 ≠ 0) (hq3 : q 3 ≠ 0)
    (hcoll : ¬ LinearIndependent ℝ (fun j : Fin 3 => p j.castSucc)) :
    ∃ H1 : Mat, (∀ i, ∃ w : ℝ, H1 *ᵥ p i = w • q i) ∧
      ¬ ∃ c : ℝ, H1 = c • H0 := by
  let p3 : Fin 3 → Point := fun j => p j.castSucc
  let P : Mat := colsOf p3
  have hPdet : P.det = 0 := by
    apply Matrix.det_eq_zero_of_not_linearIndependent_cols
    simpa [P, p3] using hcoll
  have hrows : ¬ LinearIndependent ℝ (fun i => P i) := by
    intro hlin
    have hunit : IsUnit P := Matrix.linearIndependent_row_iff_isUnit.mp hlin
    have hdetunit : IsUnit P.det := (Matrix.isUnit_iff_isUnit_det P).mp hunit
    rw [isUnit_iff_ne_zero] at hdetunit
    exact hdetunit hPdet
  obtain ⟨a, ha_sum, ⟨k, hak⟩⟩ := Fintype.not_linearIndependent_iff.mp hrows
  let φ : Point →ₗ[ℝ] ℝ := ∑ i, a i • LinearMap.proj i
  have hφp : ∀ j : Fin 3, φ (p3 j) = 0 := by
    intro j
    have h := congrFun ha_sum j
    simpa [φ, P, colsOf, Finset.sum_apply, LinearMap.proj_apply] using h
  have hφne : φ ≠ 0 := by
    intro hzero
    have hk : φ (Pi.single k (1 : ℝ)) = 0 := by rw [hzero]; simp
    have hval : φ (Pi.single k (1 : ℝ)) = a k := by
      simp [φ, Pi.single_apply, Finset.sum_ite_eq', Finset.mem_univ]
    rw [hval] at hk
    exact hak hk
  have hKne : rankOne (q 3) φ ≠ 0 := rankOne_ne_zero (q 3) φ hq3 hφne
  let K : Mat := rankOne (q 3) φ
  let H1 : Mat := H0 + K
  have hKpi : ∀ j : Fin 3, K *ᵥ p3 j = 0 := by
    intro j
    rw [show K = rankOne (q 3) φ from rfl, rankOne_mulVec, hφp j, zero_smul]
  have hH1sat : ∀ i, ∃ w : ℝ, H1 *ᵥ p i = w • q i := by
    intro i
    fin_cases i
    · obtain ⟨w, hw⟩ := hH0 0
      refine ⟨w, ?_⟩
      change (H0 + K) *ᵥ p 0 = w • q 0
      rw [Matrix.add_mulVec, show K *ᵥ p 0 = 0 from hKpi 0,
        add_zero]
      exact hw
    · obtain ⟨w, hw⟩ := hH0 1
      refine ⟨w, ?_⟩
      change (H0 + K) *ᵥ p 1 = w • q 1
      rw [Matrix.add_mulVec, show K *ᵥ p 1 = 0 from hKpi 1,
        add_zero]
      exact hw
    · obtain ⟨w, hw⟩ := hH0 2
      refine ⟨w, ?_⟩
      change (H0 + K) *ᵥ p 2 = w • q 2
      rw [Matrix.add_mulVec, show K *ᵥ p 2 = 0 from hKpi 2,
        add_zero]
      exact hw
    · obtain ⟨w3, hw3⟩ := hH0 3
      refine ⟨w3 + φ (p 3), ?_⟩
      change (H0 + K) *ᵥ p 3 = (w3 + φ (p 3)) • q 3
      rw [Matrix.add_mulVec, show K *ᵥ p 3 = φ (p 3) • q 3 from by
        rw [show K = rankOne (q 3) φ from rfl, rankOne_mulVec], hw3, add_smul]
  refine ⟨H1, hH1sat, ?_⟩
  rintro ⟨c, hc⟩
  have hHK : H0 + K = c • H0 := by
    simpa [H1] using hc
  have hKrel : K = (c - 1) • H0 := by
    calc K = (H0 + K) - H0 := by abel
      _ = c • H0 - H0 := by rw [hHK]
      _ = (c - 1) • H0 := by rw [sub_smul, one_smul]
  have hH0p0ne : H0 *ᵥ p 0 ≠ 0 := by
    intro h
    apply hp0
    have hinj : Function.Injective H0.mulVec :=
      Matrix.mulVec_injective_iff_isUnit.mpr
        ((Matrix.isUnit_iff_isUnit_det _).mpr (isUnit_iff_ne_zero.mpr hH0det))
    exact hinj (by simpa using h)
  have hc1 : c - 1 = 0 := by
    have hKp0 : K *ᵥ p 0 = 0 := hKpi 0
    rw [hKrel, Matrix.smul_mulVec] at hKp0
    rcases smul_eq_zero.mp hKp0 with h | h
    · exact h
    · exact absurd h hH0p0ne
  have hc_eq : c = 1 := sub_eq_zero.mp hc1
  have hKzero : K = 0 := by
    rw [hKrel, hc_eq]
    simp
  exact hKne hKzero

/-- **D4.** In DLT form: collinear source points make the system degenerate for
consistent data. -/
theorem D4 (cs : Fin 4 → Corr) (H0 : Mat)
    (hH0 : dltSystem4 cs *ᵥ vecH H0 = 0)
    (hH0det : H0.det ≠ 0)
    (hcoll : ¬ LinearIndependent ℝ
      (fun j : Fin 3 => src (cs j.castSucc))) :
    ∃ H1 : Mat, dltSystem4 cs *ᵥ vecH H1 = 0 ∧
      ¬ ∃ c : ℝ, H1 = c • H0 := by
  have hsat : ∀ i, ∃ w : ℝ, H0 *ᵥ src (cs i) = w • dst (cs i) := by
    intro i
    exact (satisfies_iff H0 (cs i)).mp (satisfiesAll_of_dltSystem4 cs H0 hH0 i)
  have hp0 : src (cs 0) ≠ 0 := by
    intro h
    have := congrFun h 2
    simp [src] at this
  have hq3 : dst (cs 3) ≠ 0 := by
    intro h
    have := congrFun h 2
    simp [dst] at this
  obtain ⟨H1, hH1, hindep⟩ :=
    collinear_degenerate (fun i => src (cs i)) (fun i => dst (cs i)) H0 hsat hH0det
      hp0 hq3 hcoll
  refine ⟨H1, ?_, hindep⟩
  apply D1
  intro i
  exact (satisfies_iff H1 (cs i)).mpr (hH1 i)

/-! ## D5: Hartley normalization -/

/-- **D5.** Hartley normalization is equivalent: a homography solving the
normalized correspondence system, conjugated by the normalization transforms,
solves the original system, and conversely. -/
theorem D5 (Hn Tsrc Tdst : Mat) (_hTsrc : Tsrc.det ≠ 0) (hTdst : Tdst.det ≠ 0)
    (p q : Fin 4 → Point) :
    (∀ i, ∃ w : ℝ, Hn *ᵥ (Tsrc *ᵥ p i) = w • (Tdst *ᵥ q i)) ↔
      (∀ i, ∃ w : ℝ, (Tdst⁻¹ * Hn * Tsrc) *ᵥ p i = w • q i) := by
  constructor
  · intro h i
    obtain ⟨w, hw⟩ := h i
    refine ⟨w, ?_⟩
    have h' := congrArg (fun v => Tdst⁻¹ *ᵥ v) hw
    simpa [Matrix.mulVec_mulVec, Matrix.mulVec_smul, Matrix.mul_assoc,
      Matrix.nonsing_inv_mul _ (isUnit_iff_ne_zero.mpr hTdst)] using h'
  · intro h i
    obtain ⟨w, hw⟩ := h i
    refine ⟨w, ?_⟩
    have h' := congrArg (fun v => Tdst *ᵥ v) hw
    have hmul : Tdst *ᵥ ((Tdst⁻¹ * Hn * Tsrc) *ᵥ p i) =
        Hn *ᵥ (Tsrc *ᵥ p i) := by
      rw [Matrix.mulVec_mulVec]
      have hassoc : Tdst * (Tdst⁻¹ * Hn * Tsrc) = Hn * Tsrc := by
        calc Tdst * (Tdst⁻¹ * Hn * Tsrc)
            = (Tdst * (Tdst⁻¹ * Hn)) * Tsrc := by rw [← Matrix.mul_assoc]
          _ = ((Tdst * Tdst⁻¹) * Hn) * Tsrc := by rw [← Matrix.mul_assoc]
          _ = (1 * Hn) * Tsrc := by
                rw [Matrix.mul_nonsing_inv _ (isUnit_iff_ne_zero.mpr hTdst)]
          _ = Hn * Tsrc := by rw [Matrix.one_mul]
      rw [hassoc, Matrix.mulVec_mulVec]
    rw [hmul] at h'
    simpa [Matrix.mulVec_smul] using h'

end PatternMatch.DLT
