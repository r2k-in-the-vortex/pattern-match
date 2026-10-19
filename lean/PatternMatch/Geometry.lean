module

import Mathlib.Analysis.Calculus.Deriv.Inv
import Mathlib.Analysis.Calculus.FDeriv.Mul
import Mathlib.Analysis.Calculus.FDeriv.Pi
import Mathlib.Analysis.Complex.Trigonometric
import Mathlib.Analysis.Normed.Operator.Mul
import Mathlib.Analysis.SpecialFunctions.Complex.Arg
import Mathlib.Analysis.SpecialFunctions.Complex.Log
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Data.Matrix.Mul
import Mathlib.LinearAlgebra.Matrix.Determinant.Basic
import Mathlib.LinearAlgebra.Matrix.NonsingularInverse
import Mathlib.LinearAlgebra.Matrix.Notation
import Mathlib.Topology.Algebra.Module.ContinuousLinearMap.PiProd

/-!
# Geometry specification (M1)

Formal statements and proofs from README §10.2:

* homogeneous points and homographies with affine normalization;
* similarity parameters, embedding, and composition (G1–G5);
* the cross-ratio and its projective invariance (G6);
* preservation of collinearity (G7);
* the local Jacobian at the model origin (G8).

Committed core modules contain no `sorry` (checked by `scripts/check-sorry.sh`).
-/

namespace PatternMatch.Geometry

open Matrix
open scoped Matrix BigOperators

/-! ## Homogeneous points and homographies -/

/-- A homogeneous point in the projective plane, as a 3-vector over `ℝ`. -/
abbrev HPoint : Type := Fin 3 → ℝ

/-- A projective homography, as a 3×3 real matrix acting on homogeneous points. -/
abbrev HMatrix : Type := Matrix (Fin 3) (Fin 3) ℝ

/-- Raw linear action of a homography on a homogeneous point.  The result is a
homogeneous representative; affine normalization is `normalize`. -/
def act (H : HMatrix) (p : HPoint) : HPoint := H *ᵥ p

/-- Affine normalization: divide by the homogeneous coordinate.  The value at a
point with zero homogeneous coordinate is not meaningful; every theorem below
states the non-zero hypothesis explicitly. -/
noncomputable def normalize (p : HPoint) : HPoint := fun i => p i / p 2

/-! ## G1: homography action is composable -/

/-- **G1.** The action of a product is the composition of the actions. -/
theorem act_comp (H₁ H₂ : HMatrix) (p : HPoint) :
    act (H₁ * H₂) p = act H₁ (act H₂ p) := by
  simp [act]

/-! ## G2: projective scaling is invisible -/

/-- Scaling a homography scales the raw action. -/
theorem act_smul (c : ℝ) (H : HMatrix) (p : HPoint) :
    act (c • H) p = c • act H p := by
  simpa [act] using Matrix.smul_mulVec c H p

/-- The action is linear in the point. -/
theorem act_smul_point (c : ℝ) (H : HMatrix) (p : HPoint) :
    act H (c • p) = c • act H p := by
  simpa [act] using Matrix.mulVec_smul H c p

/-- Scaling a homogeneous point does not change its affine normalization. -/
theorem normalize_smul (c : ℝ) (hc : c ≠ 0) (p : HPoint) (hp : p 2 ≠ 0) :
    normalize (c • p) = normalize p := by
  ext i
  simp only [normalize, Pi.smul_apply, smul_eq_mul]
  field_simp [hc, hp]

/-- **G2.** A non-zero projective scale factor leaves the affine action unchanged. -/
theorem normalize_act_smul (c : ℝ) (hc : c ≠ 0) (H : HMatrix) (p : HPoint)
    (hp : (act H p) 2 ≠ 0) :
    normalize (act (c • H) p) = normalize (act H p) := by
  rw [act_smul]
  exact normalize_smul c hc (act H p) hp

/-! ## G3: inversion -/

/-- The inverse homography undoes the raw action. -/
theorem act_inv_act (H : HMatrix) (hH : H.det ≠ 0) (p : HPoint) :
    act H⁻¹ (act H p) = p := by
  rw [← act_comp, Matrix.nonsing_inv_mul H (isUnit_iff_ne_zero.mpr hH)]
  simp [act]

/-- **G3.** On affine points, applying `H` and then `H⁻¹` recovers the original
affine point. -/
theorem normalize_act_inv (H : HMatrix) (hH : H.det ≠ 0) (p : HPoint)
    (hp : p 2 ≠ 0) (hHp : (act H p) 2 ≠ 0) :
    normalize (act H⁻¹ (normalize (act H p))) = normalize p := by
  have hscale : normalize (act H p) = (act H p 2)⁻¹ • act H p := by
    ext i
    simp [normalize, Pi.smul_apply, smul_eq_mul, div_eq_mul_inv, mul_comm]
  rw [hscale]
  rw [act_smul_point (act H p 2)⁻¹ H⁻¹ (act H p)]
  rw [act_inv_act H hH p]
  exact normalize_smul (act H p 2)⁻¹ (inv_ne_zero hHp) p hp

/-! ## G4: similarities form a subgroup -/

/-- Parameters of a similarity `q = t + s R(θ) p` (README §4). -/
@[ext]
structure Similarity where
  /-- Uniform scale, `1` at the taught size. -/
  scale : ℝ
  /-- Rotation angle in radians; positive appears clockwise with `y` down. -/
  theta : ℝ
  /-- Image position of the model zero. -/
  tx : ℝ
  /-- Image y of the model zero. -/
  ty : ℝ

namespace Similarity

/-- The 3×3 homography embedding of a similarity (README §4). -/
noncomputable def toMatrix (g : Similarity) : HMatrix :=
  !![g.scale * Real.cos g.theta, -g.scale * Real.sin g.theta, g.tx;
     g.scale * Real.sin g.theta, g.scale * Real.cos g.theta, g.ty;
     0, 0, 1]

/-- Group composition: `g.comp h` applies `h` first and then `g`
(README G4: `(s₁s₂, θ₁+θ₂, t₁+s₁R(θ₁)t₂)`). -/
noncomputable def comp (g h : Similarity) : Similarity where
  scale := g.scale * h.scale
  theta := g.theta + h.theta
  tx := g.tx + g.scale * (Real.cos g.theta * h.tx - Real.sin g.theta * h.ty)
  ty := g.ty + g.scale * (Real.sin g.theta * h.tx + Real.cos g.theta * h.ty)

/-- Identity similarity. -/
def id : Similarity where
  scale := 1
  theta := 0
  tx := 0
  ty := 0

/-- Inverse similarity: scale `1/s`, angle `-θ`, and the translation that
returns the zero to the origin. -/
noncomputable def inv (g : Similarity) : Similarity where
  scale := g.scale⁻¹
  theta := -g.theta
  tx := -(g.scale⁻¹) * (Real.cos g.theta * g.tx + Real.sin g.theta * g.ty)
  ty := -(g.scale⁻¹) * (-Real.sin g.theta * g.tx + Real.cos g.theta * g.ty)

/-- **G4a.** The embedding is a group homomorphism. -/
theorem toMatrix_comp (g h : Similarity) :
    (g.comp h).toMatrix = g.toMatrix * h.toMatrix := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [comp, toMatrix, Matrix.mul_apply, Fin.sum_univ_three, Real.cos_add, Real.sin_add] <;>
    ring

/-- **G4b.** The identity similarity embeds as the identity matrix. -/
theorem toMatrix_id : (id : Similarity).toMatrix = 1 := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [id, toMatrix]

/-- **G4c.** Similarity composition is associative. -/
theorem comp_assoc (a b c : Similarity) :
    a.comp (b.comp c) = (a.comp b).comp c := by
  ext <;> simp [comp, Real.cos_add, Real.sin_add] <;> ring

/-- **G4d.** A similarity with non-zero scale has a two-sided inverse. -/
theorem comp_inv (g : Similarity) (hs : g.scale ≠ 0) :
    g.comp g.inv = id := by
  ext
  · simp [comp, inv, id, hs]
  · simp [comp, inv, id]
  · simp only [comp, inv, id]
    field_simp [hs]
    ring_nf
    rw [show g.tx - g.tx * Real.cos g.theta ^ 2 - g.tx * Real.sin g.theta ^ 2 =
        g.tx * (1 - (Real.sin g.theta ^ 2 + Real.cos g.theta ^ 2)) by ring]
    rw [Real.sin_sq_add_cos_sq]
    ring
  · simp only [comp, inv, id]
    field_simp [hs]
    ring_nf
    rw [show g.ty - g.ty * Real.sin g.theta ^ 2 - g.ty * Real.cos g.theta ^ 2 =
        g.ty * (1 - (Real.sin g.theta ^ 2 + Real.cos g.theta ^ 2)) by ring]
    rw [Real.sin_sq_add_cos_sq]
    ring

/-- **G4e.** The inverse similarity embeds as the inverse matrix. -/
theorem toMatrix_inv (g : Similarity) (hs : g.scale ≠ 0) :
    g.inv.toMatrix = g.toMatrix⁻¹ := by
  have hproduct : g.toMatrix * g.inv.toMatrix = 1 := by
    rw [← toMatrix_comp, comp_inv g hs, toMatrix_id]
  exact (Matrix.inv_eq_right_inv hproduct).symm

end Similarity

/-! ## G5: similarity decomposition -/

/-- **G5a.** A non-zero similarity block
`[[a, -b], [b, a]]` decomposes with `s = √(a²+b²) > 0` and an angle `θ` whose
cosine and sine are `a/s` and `b/s`. -/
theorem exists_cos_sin_of_ne_zero (a b : ℝ) (h : a ≠ 0 ∨ b ≠ 0) :
    let s := Real.sqrt (a ^ 2 + b ^ 2)
    0 < s ∧
      Real.cos (Complex.arg (Complex.mk a b)) = a / s ∧
      Real.sin (Complex.arg (Complex.mk a b)) = b / s := by
  dsimp only
  have hpos : 0 < a ^ 2 + b ^ 2 := by
    rcases h with ha | hb
    · have : 0 < a ^ 2 := sq_pos_of_ne_zero ha
      nlinarith [sq_nonneg b]
    · have : 0 < b ^ 2 := sq_pos_of_ne_zero hb
      nlinarith [sq_nonneg a]
  have hs : 0 < Real.sqrt (a ^ 2 + b ^ 2) := Real.sqrt_pos.mpr hpos
  have hz : Complex.mk a b ≠ 0 := by
    intro hzero
    have ha : a = 0 := congrArg Complex.re hzero
    have hb : b = 0 := congrArg Complex.im hzero
    rcases h with h | h
    · exact h ha
    · exact h hb
  have hnorm : ‖Complex.mk a b‖ = Real.sqrt (a ^ 2 + b ^ 2) := by
    rw [Complex.norm_def, Complex.normSq_mk]
    congr 1
    ring
  refine ⟨hs, ?_, ?_⟩
  · rw [Complex.cos_arg hz, hnorm]
  · rw [Complex.sin_arg, hnorm]

/-- **G5b.** The decomposition angle is unique up to multiples of `2π`. -/
theorem angle_unique_mod_two_pi {θ₁ θ₂ : ℝ}
    (hc : Real.cos θ₁ = Real.cos θ₂) (hs : Real.sin θ₁ = Real.sin θ₂) :
    ∃ k : ℤ, θ₁ = θ₂ + k * (2 * Real.pi) := by
  have hexp : Complex.exp ((θ₁ : ℂ) * Complex.I) = Complex.exp ((θ₂ : ℂ) * Complex.I) := by
    rw [Complex.exp_ofReal_mul_I, Complex.exp_ofReal_mul_I, hc, hs]
  rcases Complex.exp_eq_exp_iff_exists_int.mp hexp with ⟨k, hk⟩
  refine ⟨k, ?_⟩
  have him := congrArg Complex.im hk
  simpa [Complex.mul_im, Complex.ofReal_re, Complex.ofReal_im] using him

/-! ## G6: cross-ratio invariance -/

/-- Cross-ratio of four points on a projective line. -/
noncomputable def crossRatio (x₁ x₂ x₃ x₄ : ℝ) : ℝ :=
  ((x₁ - x₃) * (x₂ - x₄)) / ((x₁ - x₄) * (x₂ - x₃))

/-- A Möbius (projective) transformation of the line. -/
noncomputable def mobius (a b c d x : ℝ) : ℝ :=
  (a * x + b) / (c * x + d)

/-- Difference identity for a Möbius transformation. -/
theorem mobius_sub (a b c d x y : ℝ)
    (hx : c * x + d ≠ 0) (hy : c * y + d ≠ 0) :
    mobius a b c d x - mobius a b c d y =
      (a * d - b * c) * (x - y) / ((c * x + d) * (c * y + d)) := by
  unfold mobius
  rw [div_sub_div (a * x + b) (a * y + b) hx hy]
  congr 1
  ring

/-- **G6.** The cross-ratio is invariant under non-degenerate Möbius
transformations wherever the involved denominators are non-zero. -/
theorem crossRatio_mobius (a b c d x₁ x₂ x₃ x₄ : ℝ)
    (hdet : a * d - b * c ≠ 0)
    (h₁ : c * x₁ + d ≠ 0) (h₂ : c * x₂ + d ≠ 0)
    (h₃ : c * x₃ + d ≠ 0) (h₄ : c * x₄ + d ≠ 0)
    (h₁₄ : x₁ - x₄ ≠ 0) (h₂₃ : x₂ - x₃ ≠ 0) :
    crossRatio (mobius a b c d x₁) (mobius a b c d x₂)
        (mobius a b c d x₃) (mobius a b c d x₄) =
      crossRatio x₁ x₂ x₃ x₄ := by
  have h₁₃ := mobius_sub a b c d x₁ x₃ h₁ h₃
  have h₂₄ := mobius_sub a b c d x₂ x₄ h₂ h₄
  have h₁₄' := mobius_sub a b c d x₁ x₄ h₁ h₄
  have h₂₃' := mobius_sub a b c d x₂ x₃ h₂ h₃
  unfold crossRatio
  rw [h₁₃, h₂₄, h₁₄', h₂₃']
  field_simp [hdet, h₁, h₂, h₃, h₄, h₁₄, h₂₃]

/-! ## G7: homographies preserve collinearity -/

/-- The matrix whose columns are three homogeneous points. -/
def cols (p q r : HPoint) : HMatrix :=
  (Matrix.of ![p, q, r])ᵀ

/-- Multiplying by `H` acts on the columns of the point matrix. -/
theorem cols_act (H : HMatrix) (p q r : HPoint) :
    cols (act H p) (act H q) (act H r) = H * cols p q r := by
  ext i j
  fin_cases j <;>
    simp [cols, act, Matrix.mul_apply, Matrix.mulVec_apply_eq_sum, Fin.sum_univ_three]

/-- **G7.** A homography maps collinear points to collinear points. -/
theorem det_cols_collinear (H : HMatrix) (p q r : HPoint)
    (h : (cols p q r).det = 0) :
    (cols (act H p) (act H q) (act H r)).det = 0 := by
  rw [cols_act, Matrix.det_mul, h, mul_zero]

/-! ## G8: local Jacobian at the model origin -/

/-- The linear functional `p ↦ ∑ⱼ Hᵢⱼ pⱼ` on the first two coordinates. -/
noncomputable def numCLM (H : HMatrix) (i : Fin 3) : (Fin 2 → ℝ) →L[ℝ] ℝ :=
  ∑ j : Fin 2, H i (Fin.castSucc j) • ContinuousLinearMap.proj j

/-- One coordinate of the affine action, written with `p`'s two coordinates. -/
noncomputable def affineCoord (H : HMatrix) (i : Fin 3) : (Fin 2 → ℝ) → ℝ :=
  fun p => (numCLM H i p + H i 2) / (numCLM H 2 p + H 2 2)

/-- The image of the model origin under `H` (README §4, `qₓ, q_y`). -/
noncomputable def originOf (H : HMatrix) : Fin 2 → ℝ :=
  ![H 0 2 / H 2 2, H 1 2 / H 2 2]

/-- The local Jacobian of the affine action at the origin (README G8/§4). -/
noncomputable def jacobianAtOrigin (H : HMatrix) : Matrix (Fin 2) (Fin 2) ℝ :=
  let qx := H 0 2 / H 2 2
  let qy := H 1 2 / H 2 2
  !![(H 0 0 - qx * H 2 0) / H 2 2, (H 0 1 - qx * H 2 1) / H 2 2;
     (H 1 0 - qy * H 2 0) / H 2 2, (H 1 1 - qy * H 2 1) / H 2 2]

/-- The affine action as a map of the two image coordinates. -/
noncomputable def affineMap (H : HMatrix) (p : Fin 2 → ℝ) : Fin 2 → ℝ :=
  fun i => affineCoord H (Fin.castSucc i) p

/-- The Jacobian at the origin as a continuous linear map. -/
noncomputable def jacobianCLM (H : HMatrix) : (Fin 2 → ℝ) →L[ℝ] (Fin 2 → ℝ) :=
  ContinuousLinearMap.pi fun i =>
    (1 / H 2 2) •
      (numCLM H (Fin.castSucc i) - (H (Fin.castSucc i) 2 / H 2 2) • numCLM H 2)

/-- **G8a.** Each coordinate of the rational affine action has the README's
local Jacobian row as its derivative at the origin. -/
theorem hasFDerivAt_affineCoord (H : HMatrix) (i : Fin 3) (hw : H 2 2 ≠ 0) :
    HasFDerivAt (affineCoord H i)
      ((1 / H 2 2) • (numCLM H i - (H i 2 / H 2 2) • numCLM H 2)) 0 := by
  have hn : HasFDerivAt (fun p : Fin 2 → ℝ => numCLM H i p + H i 2) (numCLM H i) 0 :=
    (numCLM H i).hasFDerivAt.add_const (H i 2)
  have hd : HasFDerivAt (fun p : Fin 2 → ℝ => numCLM H 2 p + H 2 2) (numCLM H 2) 0 :=
    (numCLM H 2).hasFDerivAt.add_const (H 2 2)
  have hg0 : numCLM H 2 0 + H 2 2 = H 2 2 := by
    simp [numCLM]
  have hinv : HasFDerivAt (fun p : Fin 2 → ℝ => (numCLM H 2 p + H 2 2)⁻¹)
      ((-ContinuousLinearMap.mulLeftRight ℝ ℝ (H 2 2)⁻¹ (H 2 2)⁻¹).comp (numCLM H 2)) 0 := by
    have hf' : HasFDerivAt Inv.inv
        (-ContinuousLinearMap.mulLeftRight ℝ ℝ (H 2 2)⁻¹ (H 2 2)⁻¹)
        (numCLM H 2 0 + H 2 2) := by
      simpa only [hg0] using (hasFDerivAt_inv' (𝕜 := ℝ) (R := ℝ) hw)
    simpa only [Function.comp_def] using hf'.comp 0 hd
  have hmul := hn.mul hinv
  convert hmul using 1
  · ext p
    simp [affineCoord, div_eq_mul_inv]
  · ext v
    simp [_root_.add_apply, _root_.smul_apply, ContinuousLinearMap.comp_apply,
      _root_.neg_apply, ContinuousLinearMap.mulLeftRight_apply, smul_eq_mul]
    field_simp [hw]
    ring

/-- **G8b.** The full affine action has the README's local Jacobian at the
origin. -/
theorem hasFDerivAt_affineMap (H : HMatrix) (hw : H 2 2 ≠ 0) :
    HasFDerivAt (affineMap H) (jacobianCLM H) 0 := by
  unfold affineMap jacobianCLM
  rw [hasFDerivAt_pi]
  intro i
  exact hasFDerivAt_affineCoord H (Fin.castSucc i) hw

/-- The continuous Jacobian of G8b is the matrix action of `jacobianAtOrigin`. -/
theorem jacobianCLM_apply (H : HMatrix) (hw : H 2 2 ≠ 0) (v : Fin 2 → ℝ) :
    jacobianCLM H v = jacobianAtOrigin H *ᵥ v := by
  ext i
  fin_cases i <;>
    simp [jacobianCLM, jacobianAtOrigin, ContinuousLinearMap.pi_apply, numCLM,
      Fin.sum_univ_two, Matrix.mulVec, dotProduct, div_eq_mul_inv] <;>
    field_simp [hw] <;>
    ring

end PatternMatch.Geometry
