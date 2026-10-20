//! Similarity and homography algebra: embedding, composition, pose extraction,
//! and the projective invariants proved in Lean (README §4, §10.2).
//!
//! The Lean specification lives in `lean/PatternMatch/Geometry.lean`.
//! Theorem references:
//!
//! * G1 — [`Homography::compose`] is matrix multiplication and
//!   [`Similarity::compose`] is associative (`act_comp`);
//! * G2 — [`Homography::act_homogeneous`] is invariant under non-zero scaling
//!   of the matrix (`normalize_act_smul`);
//! * G3 — [`Homography::try_inverse`] undoes the action (`normalize_act_inv`);
//! * G4 — [`Similarity::embed`] is a group homomorphism (`toMatrix_comp`);
//! * G5 — [`similarity_from_block`] decomposes a similarity block
//!   (`exists_cos_sin_of_ne_zero`);
//! * G6 — [`cross_ratio`] is invariant under [`mobius`] (`crossRatio_mobius`);
//! * G7 — [`det_points`] vanishes after applying a homography whenever it
//!   vanishes before (`det_cols_collinear`);
//! * G8 — [`Homography::pose_at_origin`] extracts position, angle, and scale
//!   from the local Jacobian at the model origin (`hasFDerivAt_affineMap`).
//!
//! DLT arrives in M4.

use nalgebra::{Matrix2, Matrix3, Vector2, Vector3};
use serde::{Deserialize, Serialize};

/// Numerical tolerance for treating a homogeneous coordinate or determinant as
/// zero.  The exact statements in the Lean specification use `≠ 0`; the Rust
/// implementation needs a finite threshold because inputs are floats.
pub const NUMERIC_EPS: f64 = 1e-12;

/// Convert an image point to a homogeneous point with `w = 1`.
#[must_use]
pub fn homogenize(p: Vector2<f64>) -> Vector3<f64> {
    Vector3::new(p.x, p.y, 1.0)
}

/// Affine-normalize a homogeneous point, returning `None` for `w ≈ 0`.
#[must_use]
pub fn affine_normalize(p: Vector3<f64>) -> Option<Vector2<f64>> {
    if p.z.abs() <= NUMERIC_EPS {
        None
    } else {
        Some(Vector2::new(p.x / p.z, p.y / p.z))
    }
}

/// A similarity pose `q = t + s · R(θ) · p` (README §4).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Similarity {
    /// Uniform scale relative to the taught size.
    pub scale: f64,
    /// Rotation angle in radians. Positive appears clockwise with `y` down.
    pub theta: f64,
    /// Image x of the model zero.
    pub tx: f64,
    /// Image y of the model zero.
    pub ty: f64,
}

impl Similarity {
    /// The identity transform.
    pub const IDENTITY: Self = Self {
        scale: 1.0,
        theta: 0.0,
        tx: 0.0,
        ty: 0.0,
    };

    /// Construct a similarity from a reported pose.
    #[must_use]
    pub fn from_pose(x: f64, y: f64, theta: f64, scale: f64) -> Self {
        Self {
            scale,
            theta,
            tx: x,
            ty: y,
        }
    }

    /// The 2×2 rotation matrix `R(θ)`.
    #[must_use]
    pub fn rotation(theta: f64) -> Matrix2<f64> {
        let (sin, cos) = theta.sin_cos();
        Matrix2::new(cos, -sin, sin, cos)
    }

    /// Whether every parameter is finite.
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.scale.is_finite()
            && self.theta.is_finite()
            && self.tx.is_finite()
            && self.ty.is_finite()
    }

    /// Transform a point from model coordinates to image coordinates (G4).
    #[must_use]
    pub fn transform_point(&self, p: Vector2<f64>) -> Vector2<f64> {
        let rotated = Self::rotation(self.theta) * p;
        Vector2::new(
            self.tx + self.scale * rotated.x,
            self.ty + self.scale * rotated.y,
        )
    }

    /// Transform a direction (no translation).
    #[must_use]
    pub fn transform_vector(&self, v: Vector2<f64>) -> Vector2<f64> {
        Self::rotation(self.theta) * v * self.scale
    }

    /// Compose two similarities: `self` applied after `rhs` (G4).
    #[must_use]
    pub fn compose(&self, rhs: &Self) -> Self {
        let scaled = Self::rotation(self.theta) * Vector2::new(rhs.tx, rhs.ty) * self.scale;
        Self {
            scale: self.scale * rhs.scale,
            theta: self.theta + rhs.theta,
            tx: self.tx + scaled.x,
            ty: self.ty + scaled.y,
        }
    }

    /// Inverse similarity, or `None` when the scale is (numerically) zero.
    #[must_use]
    pub fn try_inverse(&self) -> Option<Self> {
        if !self.is_finite() || self.scale.abs() <= NUMERIC_EPS {
            return None;
        }
        let inv_scale = 1.0 / self.scale;
        // t' = -(1/s) R(-θ) t.
        let rotated = Self::rotation(-self.theta) * Vector2::new(self.tx, self.ty);
        Some(Self {
            scale: inv_scale,
            theta: -self.theta,
            tx: -inv_scale * rotated.x,
            ty: -inv_scale * rotated.y,
        })
    }

    /// Embed the similarity as a 3×3 homography (README §4).
    #[must_use]
    pub fn embed(&self) -> Homography {
        let (sin, cos) = self.theta.sin_cos();
        let mut m = Matrix3::zeros();
        m[(0, 0)] = self.scale * cos;
        m[(0, 1)] = -self.scale * sin;
        m[(0, 2)] = self.tx;
        m[(1, 0)] = self.scale * sin;
        m[(1, 1)] = self.scale * cos;
        m[(1, 2)] = self.ty;
        m[(2, 2)] = 1.0;
        Homography(m)
    }
}

/// A 3×3 projective homography mapping model to image homogeneous points.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Homography(pub Matrix3<f64>);

impl Homography {
    /// The identity homography.
    pub const IDENTITY: Self = Self(Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0));

    /// Wrap a raw matrix.
    #[must_use]
    pub fn from_matrix(m: Matrix3<f64>) -> Self {
        Self(m)
    }

    /// The underlying matrix.
    #[must_use]
    pub fn matrix(&self) -> &Matrix3<f64> {
        &self.0
    }

    /// Determinant of the homography matrix.
    #[must_use]
    pub fn determinant(&self) -> f64 {
        self.0.determinant()
    }

    /// Raw homogeneous action `H · p` (G1).
    #[must_use]
    pub fn act_homogeneous(&self, p: Vector3<f64>) -> Vector3<f64> {
        self.0 * p
    }

    /// Affine action, returning `None` when the result is at infinity (G2).
    #[must_use]
    pub fn try_act_point(&self, p: Vector2<f64>) -> Option<Vector2<f64>> {
        affine_normalize(self.act_homogeneous(homogenize(p)))
    }

    /// Matrix product: `self` applied after `rhs` (G1).
    #[must_use]
    pub fn compose(&self, rhs: &Self) -> Self {
        Self(self.0 * rhs.0)
    }

    /// Inverse homography, or `None` when the matrix is (numerically) singular (G3).
    #[must_use]
    pub fn try_inverse(&self) -> Option<Self> {
        let det = self.determinant();
        if !det.is_finite() || det.abs() <= NUMERIC_EPS {
            return None;
        }
        self.0.try_inverse().map(Self)
    }

    /// Normalize the matrix for output (README §4): divide by `h33` when it is
    /// non-negligible, otherwise by the Frobenius norm.
    #[must_use]
    pub fn normalized_for_output(&self) -> Matrix3<f64> {
        let h33 = self.0[(2, 2)];
        if h33.abs() > NUMERIC_EPS {
            self.0 / h33
        } else {
            let norm = self.0.norm();
            if norm > NUMERIC_EPS {
                self.0 / norm
            } else {
                self.0
            }
        }
    }

    /// Image position of the model origin (README §4, `qx, qy`).
    #[must_use]
    pub fn origin(&self) -> Option<Vector2<f64>> {
        let w = self.0[(2, 2)];
        if w.abs() <= NUMERIC_EPS {
            None
        } else {
            Some(Vector2::new(self.0[(0, 2)] / w, self.0[(1, 2)] / w))
        }
    }

    /// The local Jacobian of the affine action at the model origin (G8).
    #[must_use]
    pub fn jacobian_at_origin(&self) -> Option<Matrix2<f64>> {
        let w = self.0[(2, 2)];
        if w.abs() <= NUMERIC_EPS {
            return None;
        }
        let qx = self.0[(0, 2)] / w;
        let qy = self.0[(1, 2)] / w;
        let mut j = Matrix2::zeros();
        j[(0, 0)] = (self.0[(0, 0)] - qx * self.0[(2, 0)]) / w;
        j[(0, 1)] = (self.0[(0, 1)] - qx * self.0[(2, 1)]) / w;
        j[(1, 0)] = (self.0[(1, 0)] - qy * self.0[(2, 0)]) / w;
        j[(1, 1)] = (self.0[(1, 1)] - qy * self.0[(2, 1)]) / w;
        Some(j)
    }

    /// Report the local pose at the origin (README §4): position `q`, angle
    /// `atan2(J10, J00)`, scale `sqrt(|det J|)`. Returns `None` for a
    /// numerically singular Jacobian.
    #[must_use]
    pub fn pose_at_origin(&self) -> Option<PoseEstimate> {
        let origin = self.origin()?;
        let j = self.jacobian_at_origin()?;
        let det = j.determinant();
        if !det.is_finite() || det.abs() <= NUMERIC_EPS {
            return None;
        }
        Some(PoseEstimate {
            x: origin.x,
            y: origin.y,
            angle: j[(1, 0)].atan2(j[(0, 0)]),
            scale: det.abs().sqrt(),
        })
    }
}

/// A pose reported at the model zero (README §4, §9.9).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PoseEstimate {
    /// Image x of the model zero.
    pub x: f64,
    /// Image y of the model zero.
    pub y: f64,
    /// Local rotation angle in radians.
    pub angle: f64,
    /// Local uniform scale.
    pub scale: f64,
}

/// Decompose a similarity block `[[a, -b], [b, a]]` (G5).
///
/// Returns `(scale, theta)` with `scale = √(a²+b²)` and `theta = atan2(b, a)`.
/// Returns `None` when the block is zero.
#[must_use]
pub fn similarity_from_block(a: f64, b: f64) -> Option<(f64, f64)> {
    let scale = a.hypot(b);
    if !scale.is_finite() || scale <= NUMERIC_EPS {
        None
    } else {
        Some((scale, b.atan2(a)))
    }
}

/// Cross-ratio of four points on a projective line (G6).
#[must_use]
pub fn cross_ratio(x1: f64, x2: f64, x3: f64, x4: f64) -> f64 {
    ((x1 - x3) * (x2 - x4)) / ((x1 - x4) * (x2 - x3))
}

/// Möbius transformation `x ↦ (a·x + b) / (c·x + d)` (G6).
#[must_use]
pub fn mobius(a: f64, b: f64, c: f64, d: f64, x: f64) -> f64 {
    (a * x + b) / (c * x + d)
}

/// Determinant of the matrix whose columns are the three points (G7).
///
/// The determinant is zero exactly when the points are collinear in the
/// projective plane.
#[must_use]
pub fn det_points(p: Vector3<f64>, q: Vector3<f64>, r: Vector3<f64>) -> f64 {
    Matrix3::from_columns(&[p, q, r]).determinant()
}

/// A rotated rectangle in image coordinates (README §4).
///
/// The rectangle is stored as center, width, height, and angle in radians.  The
/// local x-axis is the rectangle's width direction.  Positive angles appear
/// clockwise on screen because image `y` points down.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RotatedRect {
    /// Center x.
    pub cx: f32,
    /// Center y.
    pub cy: f32,
    /// Full width along the local x-axis.
    pub width: f32,
    /// Full height along the local y-axis.
    pub height: f32,
    /// Rotation angle in radians.
    pub angle: f32,
}

impl RotatedRect {
    /// Construct a rectangle.
    #[must_use]
    pub fn new(cx: f32, cy: f32, width: f32, height: f32, angle: f32) -> Self {
        Self {
            cx,
            cy,
            width,
            height,
            angle,
        }
    }

    /// Whether the rectangle has finite, positive dimensions.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.cx.is_finite()
            && self.cy.is_finite()
            && self.width.is_finite()
            && self.height.is_finite()
            && self.angle.is_finite()
            && self.width > 0.0
            && self.height > 0.0
    }

    /// The four corners in image coordinates, counter-clockwise in the local
    /// frame: `(-w/2,-h/2)`, `(w/2,-h/2)`, `(w/2,h/2)`, `(-w/2,h/2)`.
    #[must_use]
    pub fn corners(&self) -> [[f32; 2]; 4] {
        let (sin, cos) = self.angle.sin_cos();
        let hw = self.width * 0.5;
        let hh = self.height * 0.5;
        let local = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)];
        local.map(|(x, y)| [self.cx + cos * x - sin * y, self.cy + sin * x + cos * y])
    }

    /// Map a point from the rectangle's local frame to image coordinates.
    #[must_use]
    pub fn local_to_image(&self, p: [f32; 2]) -> [f32; 2] {
        let (sin, cos) = self.angle.sin_cos();
        [
            self.cx + cos * p[0] - sin * p[1],
            self.cy + sin * p[0] + cos * p[1],
        ]
    }

    /// Map an image point into the rectangle's local frame.
    #[must_use]
    pub fn image_to_local(&self, p: [f32; 2]) -> [f32; 2] {
        let (sin, cos) = self.angle.sin_cos();
        let dx = p[0] - self.cx;
        let dy = p[1] - self.cy;
        [cos * dx + sin * dy, -sin * dx + cos * dy]
    }

    /// Whether an image point lies inside the rectangle (inclusive boundary).
    #[must_use]
    pub fn contains(&self, x: f32, y: f32) -> bool {
        let local = self.image_to_local([x, y]);
        local[0].abs() <= self.width * 0.5 && local[1].abs() <= self.height * 0.5
    }

    /// Axis-aligned image bounds `[min_x, min_y, max_x, max_y]`.
    #[must_use]
    pub fn bounds(&self) -> [f32; 4] {
        let corners = self.corners();
        let mut bounds = [
            f32::INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
        ];
        for corner in corners {
            bounds[0] = bounds[0].min(corner[0]);
            bounds[1] = bounds[1].min(corner[1]);
            bounds[2] = bounds[2].max(corner[0]);
            bounds[3] = bounds[3].max(corner[1]);
        }
        bounds
    }
}

/// The model zero frame: origin and x-axis direction in image coordinates
/// (README §4).  `p_model = R(-angle) · (p_image - origin)`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ZeroFrame {
    /// Origin in image coordinates.
    pub origin: [f32; 2],
    /// Model x-axis angle in image coordinates, in radians.
    pub angle: f32,
}

impl ZeroFrame {
    /// Zero at the image origin with the image axes.
    #[must_use]
    pub fn identity() -> Self {
        Self {
            origin: [0.0, 0.0],
            angle: 0.0,
        }
    }

    /// Whether the frame is finite.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.origin[0].is_finite() && self.origin[1].is_finite() && self.angle.is_finite()
    }

    /// Image point to model coordinates.
    #[must_use]
    pub fn to_model(&self, p: [f32; 2]) -> [f32; 2] {
        let (sin, cos) = self.angle.sin_cos();
        let dx = p[0] - self.origin[0];
        let dy = p[1] - self.origin[1];
        [cos * dx + sin * dy, -sin * dx + cos * dy]
    }

    /// Model point to image coordinates.
    #[must_use]
    pub fn to_image(&self, p: [f32; 2]) -> [f32; 2] {
        let (sin, cos) = self.angle.sin_cos();
        [
            self.origin[0] + cos * p[0] - sin * p[1],
            self.origin[1] + sin * p[0] + cos * p[1],
        ]
    }

    /// Rotate a direction from image to model coordinates.
    #[must_use]
    pub fn vector_to_model(&self, v: [f32; 2]) -> [f32; 2] {
        let (sin, cos) = self.angle.sin_cos();
        [cos * v[0] + sin * v[1], -sin * v[0] + cos * v[1]]
    }

    /// Rotate a direction from model to image coordinates.
    #[must_use]
    pub fn vector_to_image(&self, v: [f32; 2]) -> [f32; 2] {
        let (sin, cos) = self.angle.sin_cos();
        [cos * v[0] - sin * v[1], sin * v[0] + cos * v[1]]
    }

    /// Map an image-frame angle to the model frame.
    #[must_use]
    pub fn angle_to_model(&self, angle: f32) -> f32 {
        angle - self.angle
    }

    /// Map a model-frame angle to the image frame.
    #[must_use]
    pub fn angle_to_image(&self, angle: f32) -> f32 {
        angle + self.angle
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-9;

    fn assert_close(actual: f64, expected: f64, context: &str) {
        let scale = expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= TOL * scale,
            "{context}: actual {actual}, expected {expected}"
        );
    }

    fn assert_vec_close(actual: Vector2<f64>, expected: Vector2<f64>, context: &str) {
        assert_close(actual.x, expected.x, context);
        assert_close(actual.y, expected.y, context);
    }

    fn assert_mat_close(actual: &Matrix3<f64>, expected: &Matrix3<f64>, context: &str) {
        for i in 0..3 {
            for j in 0..3 {
                assert_close(actual[(i, j)], expected[(i, j)], context);
            }
        }
    }

    fn sample_similarities() -> Vec<Similarity> {
        vec![
            Similarity::IDENTITY,
            Similarity::from_pose(0.0, 0.0, 0.0, 1.0),
            Similarity::from_pose(12.5, -7.25, 0.37, 1.4),
            Similarity::from_pose(-3.0, 8.0, -2.1, 0.5),
            Similarity::from_pose(100.0, -50.0, std::f64::consts::PI, 2.0),
        ]
    }

    #[test]
    fn similarity_embed_matches_definition() {
        let s = Similarity::from_pose(3.0, -2.0, std::f64::consts::FRAC_PI_2, 2.0);
        let embedded = s.embed();
        let m = embedded.matrix();
        assert_close(m[(0, 0)], 0.0, "h00");
        assert_close(m[(0, 1)], -2.0, "h01");
        assert_close(m[(0, 2)], 3.0, "h02");
        assert_close(m[(1, 0)], 2.0, "h10");
        assert_close(m[(1, 1)], 0.0, "h11");
        assert_close(m[(1, 2)], -2.0, "h12");
        assert_close(m[(2, 0)], 0.0, "h20");
        assert_close(m[(2, 1)], 0.0, "h21");
        assert_close(m[(2, 2)], 1.0, "h22");
    }

    #[test]
    fn similarity_composition_is_matrix_product_g4() {
        let sims = sample_similarities();
        for a in &sims {
            for b in &sims {
                let composed = a.compose(b);
                assert_mat_close(
                    composed.embed().matrix(),
                    &(a.embed().matrix() * b.embed().matrix()),
                    "G4 embed(comp) = embed * embed",
                );
                let p = Vector2::new(1.75, -4.5);
                assert_vec_close(
                    composed.transform_point(p),
                    a.transform_point(b.transform_point(p)),
                    "G4 action composes",
                );
            }
        }
    }

    #[test]
    fn similarity_composition_is_associative_g4() {
        let sims = sample_similarities();
        for a in &sims {
            for b in &sims {
                for c in &sims {
                    let left = a.compose(&b.compose(c));
                    let right = a.compose(b).compose(c);
                    assert_close(left.scale, right.scale, "G4 assoc scale");
                    assert_close(left.theta, right.theta, "G4 assoc theta");
                    assert_close(left.tx, right.tx, "G4 assoc tx");
                    assert_close(left.ty, right.ty, "G4 assoc ty");
                }
            }
        }
    }

    #[test]
    fn similarity_inverse_roundtrip_g3_g4() {
        for s in sample_similarities() {
            let inv = s.try_inverse().expect("sample similarities are invertible");
            let p = Vector2::new(2.5, -1.25);
            assert_vec_close(
                inv.transform_point(s.transform_point(p)),
                p,
                "G3 point roundtrip",
            );
            assert_vec_close(
                s.transform_point(inv.transform_point(p)),
                p,
                "G3 inverse roundtrip",
            );
            let product = s.compose(&inv);
            assert_close(product.scale, 1.0, "G4 inverse scale");
            assert_close(product.tx, 0.0, "G4 inverse tx");
            assert_close(product.ty, 0.0, "G4 inverse ty");
        }
        assert!(
            Similarity::from_pose(0.0, 0.0, 0.0, 0.0)
                .try_inverse()
                .is_none()
        );
    }

    #[test]
    fn homography_inverse_roundtrip_g3() {
        let h = Homography::from_matrix(Matrix3::new(1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 1.0));
        let inv = h.try_inverse().expect("det = 2");
        let p = Vector3::new(2.0, -1.0, 3.0);
        let roundtrip = inv.act_homogeneous(h.act_homogeneous(p));
        assert_close(roundtrip.x, p.x, "G3 raw x");
        assert_close(roundtrip.y, p.y, "G3 raw y");
        assert_close(roundtrip.z, p.z, "G3 raw w");
        assert_mat_close(
            &(h.matrix() * inv.matrix()),
            &Matrix3::identity(),
            "G3 H * Hinv = I",
        );
    }

    #[test]
    fn singular_homography_is_rejected_g3() {
        let h = Homography::from_matrix(Matrix3::new(1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 0.0, 0.0, 1.0));
        assert!(h.try_inverse().is_none());
    }

    #[test]
    fn projective_scaling_is_invisible_g2() {
        let h = Homography::from_matrix(Matrix3::new(1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 1.0));
        let scaled = Homography::from_matrix(h.matrix() * -2.5);
        let p = Vector2::new(2.0, -1.0);
        assert_vec_close(
            scaled.try_act_point(p).unwrap(),
            h.try_act_point(p).unwrap(),
            "G2 affine action invariant",
        );
    }

    #[test]
    fn pose_extraction_roundtrip_g8() {
        for s in sample_similarities() {
            let pose = s.embed().pose_at_origin().unwrap();
            assert_close(pose.x, s.tx, "G8 pose x");
            assert_close(pose.y, s.ty, "G8 pose y");
            assert_close(pose.scale, s.scale, "G8 pose scale");
            let d = (pose.angle - s.theta).rem_euclid(std::f64::consts::TAU);
            let d = d.min(std::f64::consts::TAU - d);
            assert!(
                d <= 1e-9,
                "G8 pose angle: actual {}, expected {}",
                pose.angle,
                s.theta
            );
        }
    }

    #[test]
    fn pose_extraction_general_homography_g8() {
        // General H: J must equal the README's local Jacobian formula.
        let h = Homography::from_matrix(Matrix3::new(2.0, 1.0, 3.0, 1.0, 4.0, 5.0, 0.5, 0.25, 2.0));
        let origin = h.origin().unwrap();
        assert_close(origin.x, 1.5, "G8 origin x");
        assert_close(origin.y, 2.5, "G8 origin y");
        let j = h.jacobian_at_origin().unwrap();
        assert_close(j[(0, 0)], 5.0 / 8.0, "G8 J00");
        assert_close(j[(0, 1)], 5.0 / 16.0, "G8 J01");
        assert_close(j[(1, 0)], -1.0 / 8.0, "G8 J10");
        assert_close(j[(1, 1)], 27.0 / 16.0, "G8 J11");
        // The Jacobian is the derivative of the affine action at the origin:
        // a finite-difference check against the rational map.  Forward
        // differences are first-order accurate, so use a looser tolerance.
        let f = |p: Vector2<f64>| h.try_act_point(p).unwrap();
        let eps = 1e-6;
        let px = f(Vector2::new(eps, 0.0)) - f(Vector2::new(0.0, 0.0));
        let py = f(Vector2::new(0.0, eps)) - f(Vector2::new(0.0, 0.0));
        let fd_close = |actual: f64, expected: f64| (actual - expected).abs() <= 1e-5;
        assert!(fd_close(px.x / eps, j[(0, 0)]), "G8 derivative J00");
        assert!(fd_close(px.y / eps, j[(1, 0)]), "G8 derivative J10");
        assert!(fd_close(py.x / eps, j[(0, 1)]), "G8 derivative J01");
        assert!(fd_close(py.y / eps, j[(1, 1)]), "G8 derivative J11");
    }

    #[test]
    fn similarity_block_decomposition_g5() {
        let (scale, theta) = similarity_from_block(3.0, 4.0).unwrap();
        assert_close(scale, 5.0, "G5 scale");
        assert_close(theta.cos(), 3.0 / 5.0, "G5 cos");
        assert_close(theta.sin(), 4.0 / 5.0, "G5 sin");
        let (scale, theta) = similarity_from_block(0.0, -2.0).unwrap();
        assert_close(scale, 2.0, "G5 scale");
        assert_close(theta.cos(), 0.0, "G5 cos");
        assert_close(theta.sin(), -1.0, "G5 sin");
        assert!(similarity_from_block(0.0, 0.0).is_none());
    }

    #[test]
    fn cross_ratio_mobius_invariance_g6() {
        let x = [1.0, 2.0, 4.0, 8.0];
        let cr = cross_ratio(x[0], x[1], x[2], x[3]);
        let transformed = x.map(|v| mobius(2.0, 1.0, 1.0, 3.0, v));
        let cr_transformed = cross_ratio(
            transformed[0],
            transformed[1],
            transformed[2],
            transformed[3],
        );
        assert_close(cr, 9.0 / 7.0, "G6 cross-ratio");
        assert_close(cr_transformed, cr, "G6 invariance");
    }

    #[test]
    fn homography_preserves_collinearity_g7() {
        let h = Homography::from_matrix(Matrix3::new(1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 1.0));
        let p = Vector3::new(0.0, 0.0, 1.0);
        let q = Vector3::new(1.0, 1.0, 1.0);
        let r = Vector3::new(2.0, 2.0, 1.0);
        assert_close(det_points(p, q, r), 0.0, "G7 collinear input");
        assert_close(
            det_points(
                h.act_homogeneous(p),
                h.act_homogeneous(q),
                h.act_homogeneous(r),
            ),
            0.0,
            "G7 collinearity preserved",
        );
        let non_collinear = Vector3::new(2.0, 0.0, 1.0);
        assert!(det_points(p, q, non_collinear).abs() > 1e-9);
        assert!(
            det_points(
                h.act_homogeneous(p),
                h.act_homogeneous(q),
                h.act_homogeneous(non_collinear)
            )
            .abs()
                > 1e-9
        );
    }

    #[test]
    fn output_normalization_uses_h33_then_norm() {
        let h = Homography::from_matrix(Matrix3::new(2.0, 0.0, 4.0, 0.0, 2.0, 6.0, 0.0, 0.0, 2.0));
        let n = h.normalized_for_output();
        assert_close(n[(0, 2)], 2.0, "h02 / h33");
        assert_close(n[(2, 2)], 1.0, "h33 / h33");

        let at_infinity =
            Homography::from_matrix(Matrix3::new(2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0));
        let n = at_infinity.normalized_for_output();
        assert_close(n.norm(), 1.0, "Frobenius normalization");
    }
}

#[cfg(test)]
mod teach_geometry_tests {
    use super::*;

    fn assert_close(actual: f32, expected: f32) {
        assert!((actual - expected).abs() < 1e-5, "{actual} vs {expected}");
    }

    #[test]
    fn rotated_rect_contains_points_in_local_frame() {
        let rect = RotatedRect::new(10.0, 20.0, 8.0, 4.0, std::f32::consts::FRAC_PI_2);
        assert!(rect.contains(10.0, 20.0));
        assert!(rect.contains(12.0, 20.0));
        assert!(rect.contains(8.0, 20.0));
        assert!(!rect.contains(16.0, 20.0));
        assert!(!rect.contains(10.0, 26.0));
        assert!(rect.is_valid());
        let bounds = rect.bounds();
        assert!((bounds[0] - 8.0).abs() < 1e-4);
        assert!((bounds[2] - 12.0).abs() < 1e-4);
        assert!((bounds[1] - 16.0).abs() < 1e-4);
        assert!((bounds[3] - 24.0).abs() < 1e-4);
    }

    #[test]
    fn rotated_rect_local_roundtrip() {
        let rect = RotatedRect::new(-3.0, 7.5, 11.0, 5.5, 0.37);
        for p in [[0.0, 0.0], [4.5, -2.0], [-1.0, 2.75]] {
            let image = rect.local_to_image(p);
            let local = rect.image_to_local(image);
            assert_close(local[0], p[0]);
            assert_close(local[1], p[1]);
        }
        let corners = rect.corners();
        for corner in corners {
            assert!(rect.contains(corner[0], corner[1]));
        }
    }

    #[test]
    fn zero_frame_roundtrip() {
        let zero = ZeroFrame {
            origin: [12.5, -4.0],
            angle: 0.62,
        };
        let image = [30.0, 18.0];
        let model = zero.to_model(image);
        let back = zero.to_image(model);
        assert_close(back[0], image[0]);
        assert_close(back[1], image[1]);

        let direction = [0.8f32, -0.6];
        let model_v = zero.vector_to_model(direction);
        let back_v = zero.vector_to_image(model_v);
        assert_close(back_v[0], direction[0]);
        assert_close(back_v[1], direction[1]);
        assert_close(model_v[0].hypot(model_v[1]), 1.0);
    }
}
