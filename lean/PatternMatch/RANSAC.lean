module

import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Defs
import Mathlib.Algebra.Order.BigOperators.GroupWithZero.Finset
import Mathlib.Algebra.Order.Floor.Semiring
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Basic.Real.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Choose.Bounds
import Mathlib.Data.Nat.Factorial.BigOperators
import Mathlib.Data.Rat.Cast.Order

/-!
# RANSAC specification (M4)

Formal statements and proofs from README §10.4:

* the all-inlier sampling probability (R1);
* the conservative bound `C(I,k)/C(n,k) ≤ (I/n)^k` (R2);
* the success probability after `N` iterations (R3);
* monotonicity in `N`, `I`, and sample size `k` (R4);
* the required-iteration count guarantee (R5).

Probabilities are exact rationals.  The product Bernoulli model used for R3 is
the finite space `Fin N → Bool` with weights `q` and `1 - q`; this is precisely
the independence assumption in the README.
-/

namespace PatternMatch.RANSAC

open scoped BigOperators

/-- **R1 / R2.** Probability that a uniformly chosen `k`-subset of `n`
correspondences lies entirely inside an inlier set of size `I`. -/
def inlierSampleProbability (n I k : ℕ) : ℚ :=
  (Nat.choose I k : ℚ) / (Nat.choose n k : ℚ)

/-- Success probability after `N` independent trials, each succeeding with
probability `q`. -/
def successProbability (q : ℚ) (N : ℕ) : ℚ :=
  1 - (1 - q) ^ N

/-! ## R1: the all-inlier sample count -/

/-- **R1.** Among all `k`-subsets of an `n`-element universe, exactly `C(I,k)`
are contained in an inlier set of size `I`. -/
theorem R1_card {α : Type*} [DecidableEq α] [Fintype α]
    (inliers : Finset α) (k : ℕ) :
    (((Finset.univ : Finset α).powersetCard k).filter
      (fun s => s ⊆ inliers)).card = Nat.choose inliers.card k := by
  have h : ((Finset.univ : Finset α).powersetCard k).filter
      (fun s => s ⊆ inliers) = inliers.powersetCard k := by
    ext s
    simp [Finset.mem_powersetCard, and_comm]
  rw [h, Finset.card_powersetCard]

/-- **R1.** The probability of an all-inlier sample is the ratio of the
favorable count to the total count, `C(I,k) / C(n,k)`. -/
theorem R1_probability {α : Type*} [DecidableEq α] [Fintype α]
    (inliers : Finset α) (k : ℕ) (_hk : k ≤ Fintype.card α) :
    ((((Finset.univ : Finset α).powersetCard k).filter
        (fun s => s ⊆ inliers)).card : ℚ) /
        (((Finset.univ : Finset α).powersetCard k).card : ℚ) =
      inlierSampleProbability (Fintype.card α) inliers.card k := by
  rw [R1_card, Finset.card_powersetCard]
  rfl

/-! ## R2: the conservative bound -/

/-- Product form of the all-inlier probability. -/
theorem inlierSampleProbability_eq_prod (n I k : ℕ) (hk : k ≤ n) :
    inlierSampleProbability n I k =
      ∏ i ∈ Finset.range k,
        (((I - i : ℕ) : ℚ) / ((n - i : ℕ) : ℚ)) := by
  unfold inlierSampleProbability
  have hkfact : ((k.factorial : ℕ) : ℚ) ≠ 0 := by
    exact_mod_cast (Nat.factorial_ne_zero k)
  have hprodI : (Nat.descFactorial I k : ℚ) =
      ∏ i ∈ Finset.range k, ((I - i : ℕ) : ℚ) := by
    exact_mod_cast Nat.descFactorial_eq_prod_range I k
  have hprodN : (Nat.descFactorial n k : ℚ) =
      ∏ i ∈ Finset.range k, ((n - i : ℕ) : ℚ) := by
    exact_mod_cast Nat.descFactorial_eq_prod_range n k
  have hchooseI : (Nat.choose I k : ℚ) =
      (∏ i ∈ Finset.range k, ((I - i : ℕ) : ℚ)) / ((k.factorial : ℕ) : ℚ) := by
    rw [eq_div_iff hkfact]
    rw [mul_comm, ← hprodI]
    exact_mod_cast (Nat.descFactorial_eq_factorial_mul_choose I k).symm
  have hchooseN : (Nat.choose n k : ℚ) =
      (∏ i ∈ Finset.range k, ((n - i : ℕ) : ℚ)) / ((k.factorial : ℕ) : ℚ) := by
    rw [eq_div_iff hkfact]
    rw [mul_comm, ← hprodN]
    exact_mod_cast (Nat.descFactorial_eq_factorial_mul_choose n k).symm
  rw [hchooseI, hchooseN]
  rw [div_div_div_cancel_right₀ hkfact]
  rw [← Finset.prod_div_distrib]

/-- **R2.** For `0 < I ≤ n` and `1 ≤ k ≤ n`,
`C(I,k)/C(n,k) ≤ (I/n)^k`. -/
theorem R2 (n I k : ℕ) (hI0 : 0 < I) (hIn : I ≤ n)
    (_hk : 1 ≤ k) (hkn : k ≤ n) :
    inlierSampleProbability n I k ≤ ((I : ℚ) / (n : ℚ)) ^ k := by
  rw [inlierSampleProbability_eq_prod n I k hkn]
  have hnpos : (0 : ℚ) < (n : ℚ) := by
    exact_mod_cast (lt_of_lt_of_le hI0 hIn)
  have hterm : ∀ i ∈ Finset.range k,
      (((I - i : ℕ) : ℚ) / ((n - i : ℕ) : ℚ)) ≤ (I : ℚ) / (n : ℚ) := by
    intro i hi
    have hik : i < k := Finset.mem_range.mp hi
    have hin : i < n := lt_of_lt_of_le hik hkn
    have hden : (0 : ℚ) < ((n - i : ℕ) : ℚ) := by
      exact_mod_cast Nat.sub_pos_of_lt hin
    rw [div_le_iff₀ hden]
    by_cases hiI : i ≤ I
    · have hcastI : ((I - i : ℕ) : ℚ) = (I : ℚ) - i := by
        rw [Nat.cast_sub hiI]
      have hcastN : ((n - i : ℕ) : ℚ) = (n : ℚ) - i := by
        rw [Nat.cast_sub (le_of_lt hin)]
      rw [hcastI, hcastN]
      have hIn' : (I : ℚ) ≤ n := by exact_mod_cast hIn
      have hi' : (0 : ℚ) ≤ i := by positivity
      have hn' : (0 : ℚ) ≤ n := le_of_lt hnpos
      field_simp
      nlinarith
    · have hzero : ((I - i : ℕ) : ℚ) = 0 := by
        rw [Nat.sub_eq_zero_of_le (le_of_lt (lt_of_not_ge hiI))]
        simp
      rw [hzero]
      positivity
  calc ∏ i ∈ Finset.range k,
        (((I - i : ℕ) : ℚ) / ((n - i : ℕ) : ℚ))
      ≤ ∏ i ∈ Finset.range k, ((I : ℚ) / (n : ℚ)) := by
        apply Finset.prod_le_prod₀
        · intro i hi
          positivity
        · exact hterm
    _ = ((I : ℚ) / (n : ℚ)) ^ k := by
        rw [Finset.prod_const, Finset.card_range, div_pow]

/-! ## R3: success after N independent iterations -/

/-- Weight of one sequence of `N` Bernoulli trials with success probability
`q`. -/
def trialMass (q : ℚ) {N : ℕ} (ω : Fin N → Bool) : ℚ :=
  ∏ i, if ω i then q else 1 - q

/-- Probability of a decidable event in the product Bernoulli model. -/
noncomputable def trialProb (q : ℚ) {N : ℕ}
    (p : (Fin N → Bool) → Prop) : ℚ := by
  classical
  exact ∑ ω, if p ω then trialMass q ω else 0

/-- The total mass of the product Bernoulli model is one. -/
theorem trialMass_total (q : ℚ) (N : ℕ) :
    (∑ ω : Fin N → Bool, trialMass q ω) = 1 := by
  unfold trialMass
  have h := (Finset.prod_univ_sum (t := fun _ : Fin N => (Finset.univ : Finset Bool))
    (f := fun _ b => if b then q else 1 - q)).symm
  simp only [Fintype.piFinset_univ] at h
  rw [h]
  simp

/-- The probability of an event plus its complement is one. -/
theorem trialProb_add_compl (q : ℚ) {N : ℕ} (p : (Fin N → Bool) → Prop) :
    trialProb q p + trialProb q (fun ω => ¬ p ω) = 1 := by
  classical
  unfold trialProb
  rw [← Finset.sum_add_distrib]
  trans (∑ ω : Fin N → Bool, trialMass q ω)
  · apply Finset.sum_congr rfl
    intro ω _
    by_cases h : p ω <;> simp [h]
  · exact trialMass_total q N

/-- The probability that all `N` trials fail is `(1 - q)^N`. -/
theorem trialProb_no_success (q : ℚ) (N : ℕ) :
    trialProb q (fun ω : Fin N → Bool => ∀ i, ω i = false) = (1 - q) ^ N := by
  classical
  unfold trialProb
  rw [Finset.sum_eq_single (fun _ : Fin N => false)]
  · simp [trialMass]
  · intro ω _ hω
    have hneq : ¬ ∀ i, ω i = false := by
      intro h
      exact hω (funext h)
    simp [hneq]
  · intro hnot
    exact absurd (Finset.mem_univ _) hnot

/-- **R3.** After `N` independent iterations, the probability of at least one
success is `1 - (1 - q)^N`. -/
theorem R3 (q : ℚ) (N : ℕ) :
    trialProb q (fun ω : Fin N → Bool => ∃ i, ω i = true) =
      1 - (1 - q) ^ N := by
  classical
  have hcompl : (fun ω : Fin N → Bool => ¬ ∃ i, ω i = true) =
      (fun ω => ∀ i, ω i = false) := by
    funext ω
    simp
  have htotal := trialProb_add_compl q
    (fun ω : Fin N → Bool => ∃ i, ω i = true)
  rw [hcompl, trialProb_no_success] at htotal
  linarith

/-! ## R4: monotonicity -/

/-- **R4.** Success is monotone non-decreasing in the iteration count. -/
theorem R4_mono_N (q : ℚ) (hq0 : 0 ≤ q) (hq1 : q ≤ 1)
    {N M : ℕ} (hNM : N ≤ M) :
    successProbability q N ≤ successProbability q M := by
  unfold successProbability
  have hbase0 : 0 ≤ 1 - q := by linarith
  have hbase1 : 1 - q ≤ 1 := by linarith
  have hpow : (1 - q) ^ M ≤ (1 - q) ^ N :=
    pow_le_pow_of_le_one hbase0 hbase1 hNM
  linarith

/-- **R4.** Success is monotone non-decreasing in the per-trial success
probability. -/
theorem R4_mono_q {q u : ℚ} (hqu : q ≤ u) (hu1 : u ≤ 1) (N : ℕ) :
    successProbability q N ≤ successProbability u N := by
  unfold successProbability
  have h0 : 0 ≤ 1 - u := by linarith
  have h1 : 1 - u ≤ 1 - q := by linarith
  have hpow : (1 - u) ^ N ≤ (1 - q) ^ N :=
    pow_le_pow_left₀ h0 h1 N
  linarith

/-- **R4.** The all-inlier probability is monotone non-decreasing in the inlier
count. -/
theorem R4_mono_I (n k : ℕ) {I I' : ℕ} (hI : I ≤ I') (_hI' : I' ≤ n)
    (hk : k ≤ n) :
    inlierSampleProbability n I k ≤ inlierSampleProbability n I' k := by
  unfold inlierSampleProbability
  have hden : (0 : ℚ) < (Nat.choose n k : ℚ) := by
    exact_mod_cast Nat.choose_pos hk
  rw [div_le_div_iff_of_pos_right hden]
  exact_mod_cast Nat.choose_le_choose k hI

/-- One sampling step: increasing the sample size by one cannot increase the
all-inlier probability. -/
theorem inlierSampleProbability_succ_le (n I k : ℕ) (hIn : I ≤ n)
    (hk : k + 1 ≤ n) :
    inlierSampleProbability n I (k + 1) ≤
      inlierSampleProbability n I k := by
  rw [inlierSampleProbability_eq_prod n I (k + 1) hk,
    inlierSampleProbability_eq_prod n I k (Nat.le_of_succ_le hk)]
  rw [Finset.prod_range_succ]
  have hnonneg : 0 ≤ ∏ i ∈ Finset.range k,
      (((I - i : ℕ) : ℚ) / ((n - i : ℕ) : ℚ)) := by
    apply Finset.prod_nonneg
    intro i _
    positivity
  have hfactor : (((I - k : ℕ) : ℚ) / ((n - k : ℕ) : ℚ)) ≤ 1 := by
    have hden : (0 : ℚ) < ((n - k : ℕ) : ℚ) := by
      exact_mod_cast Nat.sub_pos_of_lt (Nat.lt_of_succ_le hk)
    rw [div_le_one hden]
    exact_mod_cast Nat.sub_le_sub_right hIn k
  calc (∏ i ∈ Finset.range k,
        (((I - i : ℕ) : ℚ) / ((n - i : ℕ) : ℚ))) *
        (((I - k : ℕ) : ℚ) / ((n - k : ℕ) : ℚ))
      ≤ (∏ i ∈ Finset.range k,
          (((I - i : ℕ) : ℚ) / ((n - i : ℕ) : ℚ))) * 1 := by
        gcongr
    _ = ∏ i ∈ Finset.range k,
        (((I - i : ℕ) : ℚ) / ((n - i : ℕ) : ℚ)) := by ring

/-- **R4.** The all-inlier probability is monotone non-increasing in the sample
size. -/
theorem R4_anti_k (n I : ℕ) (hIn : I ≤ n) {k k' : ℕ} (hkk : k ≤ k')
    (hk' : k' ≤ n) :
    inlierSampleProbability n I k' ≤ inlierSampleProbability n I k := by
  induction k' with
  | zero =>
      have hk0 : k = 0 := Nat.eq_zero_of_le_zero hkk
      subst hk0
      simp
  | succ k' ih =>
      by_cases hle : k ≤ k'
      · exact le_trans
          (inlierSampleProbability_succ_le n I k' hIn hk')
          (ih hle (Nat.le_of_succ_le hk'))
      · have hk_eq : k = k' + 1 := by omega
        subst hk_eq
        simp

/-- The all-inlier probability is at most one when `I ≤ n` and `k ≤ n`. -/
theorem inlierSampleProbability_le_one (n I k : ℕ) (hI : I ≤ n) (hk : k ≤ n) :
    inlierSampleProbability n I k ≤ 1 := by
  unfold inlierSampleProbability
  have hden : (0 : ℚ) < (Nat.choose n k : ℚ) := by
    exact_mod_cast Nat.choose_pos hk
  rw [div_le_one hden]
  exact_mod_cast Nat.choose_le_choose k hI

/-- **R4.** Success is monotone non-decreasing in the inlier count. -/
theorem R4_success_mono_I (n k : ℕ) {I I' : ℕ} (hI : I ≤ I') (hI' : I' ≤ n)
    (hk : k ≤ n) (N : ℕ) :
    successProbability (inlierSampleProbability n I k) N ≤
      successProbability (inlierSampleProbability n I' k) N := by
  apply R4_mono_q (R4_mono_I n k hI hI' hk)
  exact inlierSampleProbability_le_one n I' k hI' hk

/-- **R4.** Success is monotone non-increasing in the sample size. -/
theorem R4_success_anti_k (n I : ℕ) (hI : I ≤ n) {k k' : ℕ} (hkk : k ≤ k')
    (hk' : k' ≤ n) (N : ℕ) :
    successProbability (inlierSampleProbability n I k') N ≤
      successProbability (inlierSampleProbability n I k) N := by
  apply R4_mono_q (R4_anti_k n I hI hkk hk')
  exact inlierSampleProbability_le_one n I k hI (Nat.le_trans hkk hk')

/-! ## R5: required iterations -/

/-- The number of iterations required to reach success probability `P`, from a
per-sample success probability `q`. -/
noncomputable def requiredIterations (P q : ℝ) : ℕ :=
  Nat.ceil (Real.log (1 - P) / Real.log (1 - q))

/-- **R5.** If the iteration count is at least
`⌈log(1-P) / log(1-q)⌉`, the success probability is at least `P`. -/
theorem R5 (P q : ℝ) (_hP0 : 0 < P) (hP1 : P < 1)
    (hq0 : 0 < q) (hq1 : q < 1) (N : ℕ)
    (hN : requiredIterations P q ≤ N) :
    P ≤ 1 - (1 - q) ^ N := by
  have hPpos : 0 < 1 - P := by linarith
  have hqpos : 0 < 1 - q := by linarith
  have hqlt : 1 - q < 1 := by linarith
  have hlogq : Real.log (1 - q) < 0 := Real.log_neg hqpos hqlt
  have hratio : Real.log (1 - P) / Real.log (1 - q) ≤ (N : ℝ) := by
    simpa using (Nat.ceil_le.mp hN)
  have hmul : (N : ℝ) * Real.log (1 - q) ≤ Real.log (1 - P) :=
    (div_le_iff_of_neg hlogq).mp hratio
  have hpow : (1 - q) ^ N ≤ 1 - P := by
    calc (1 - q) ^ N
        = Real.exp (Real.log ((1 - q) ^ N)) :=
            (Real.exp_log (pow_pos hqpos N)).symm
      _ = Real.exp ((N : ℝ) * Real.log (1 - q)) := by rw [Real.log_pow]
      _ ≤ Real.exp (Real.log (1 - P)) := Real.exp_le_exp.mpr hmul
      _ = 1 - P := Real.exp_log hPpos
  linarith

end PatternMatch.RANSAC
