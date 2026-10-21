//! Deterministic synthetic benchmark (README §13.3).
//!
//! A taught L-bracket is rendered into test scenes with known similarity
//! poses.  The benchmark reports detection rate, position/angle/scale error,
//! false detections, and Hough-only versus hybrid recall.  The tilt scenes
//! render the same bracket through a known perspective homography and report
//! recall, ROI-corner reprojection error, and the fraction of correct
//! detections that accept an estimated homography (README §13.4).

use nalgebra::Matrix3;
use pm_core::{
    Detection, FeatureParams, GrayF32, Homography, LoadedImage, Model, RotatedRect, SearchDefaults,
    SearchParams, Similarity, TeachSettings, ZeroFrame, detect,
};

const SCENE_WIDTH: usize = 256;
const SCENE_HEIGHT: usize = 224;
const BACKGROUND: f32 = 0.12;
const SOURCE_WIDTH: usize = 128;
const SOURCE_HEIGHT: usize = 128;
const ZERO: [f32; 2] = [64.0, 64.0];

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    fn range(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.unit()
    }
}

fn point_in_polygon(x: f32, y: f32, polygon: &[[f32; 2]]) -> bool {
    let mut inside = false;
    let mut previous = polygon.len() - 1;
    for current in 0..polygon.len() {
        let (xi, yi) = (polygon[current][0], polygon[current][1]);
        let (xj, yj) = (polygon[previous][0], polygon[previous][1]);
        if ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
        previous = current;
    }
    inside
}

fn source_gray() -> GrayF32 {
    let polygon = [
        [38.0, 32.0],
        [86.0, 32.0],
        [86.0, 52.0],
        [58.0, 52.0],
        [58.0, 94.0],
        [38.0, 94.0],
    ];
    let hole = (72.0f32, 44.0f32, 6.0f32);
    let mut data = vec![BACKGROUND; SOURCE_WIDTH * SOURCE_HEIGHT];
    for y in 0..SOURCE_HEIGHT {
        for x in 0..SOURCE_WIDTH {
            let samples = 3;
            let mut covered = 0;
            for sy in 0..samples {
                for sx in 0..samples {
                    let fx = x as f32 + (sx as f32 + 0.5) / samples as f32;
                    let fy = y as f32 + (sy as f32 + 0.5) / samples as f32;
                    if point_in_polygon(fx, fy, &polygon)
                        && (fx - hole.0).hypot(fy - hole.1) > hole.2
                    {
                        covered += 1;
                    }
                }
            }
            let coverage = covered as f32 / (samples * samples) as f32;
            data[y * SOURCE_WIDTH + x] = BACKGROUND + coverage * 0.78;
        }
    }
    GrayF32::new(SOURCE_WIDTH, SOURCE_HEIGHT, data).unwrap()
}

fn teach_model() -> Model {
    let gray = source_gray();
    let display = gray.display_rgba();
    let loaded = LoadedImage { gray, display };
    let settings = TeachSettings {
        name: "bracket".to_owned(),
        roi: RotatedRect::new(ZERO[0], ZERO[1], 128.0, 128.0, 0.0),
        zero: ZeroFrame {
            origin: ZERO,
            angle: 0.0,
        },
        params: FeatureParams::default(),
        mask: Vec::new(),
        search_defaults: SearchDefaults::default(),
    };
    Model::build(&loaded, &settings).expect("model")
}

fn render_scene(
    source: &GrayF32,
    poses: &[pm_core::geometry::Similarity],
    contrast: f32,
    brightness: f32,
) -> GrayF32 {
    let mut data = vec![BACKGROUND; SCENE_WIDTH * SCENE_HEIGHT];
    for pose in poses {
        let (sin, cos) = (pose.theta as f32).sin_cos();
        let inverse_scale = 1.0 / pose.scale as f32;
        for y in 0..SCENE_HEIGHT {
            for x in 0..SCENE_WIDTH {
                let mut sum = 0.0;
                let mut object = false;
                for sy in 0..6 {
                    for sx in 0..6 {
                        let qx = x as f32 + (sx as f32 - 2.5) / 6.0;
                        let qy = y as f32 + (sy as f32 - 2.5) / 6.0;
                        let dx = qx - pose.tx as f32;
                        let dy = qy - pose.ty as f32;
                        let px = (cos * dx + sin * dy) * inverse_scale + ZERO[0];
                        let py = (-sin * dx + cos * dy) * inverse_scale + ZERO[1];
                        let value = source.sample_bilinear(px, py);
                        sum += value;
                        object |= value > 0.3;
                    }
                }
                if object {
                    data[y * SCENE_WIDTH + x] = (sum / 36.0).max(BACKGROUND);
                }
            }
        }
    }
    for value in &mut data {
        *value = ((*value - 0.5) * contrast + 0.5 + brightness).clamp(0.0, 1.0);
    }
    GrayF32::new(SCENE_WIDTH, SCENE_HEIGHT, data).unwrap()
}

fn add_clutter(image: &mut GrayF32, rng: &mut SplitMix64, count: usize) {
    for _ in 0..count {
        let x0 = rng.range(0.0, SCENE_WIDTH as f32);
        let y0 = rng.range(0.0, SCENE_HEIGHT as f32);
        let length = rng.range(8.0, 45.0);
        let angle = rng.range(0.0, std::f32::consts::TAU);
        let value = rng.range(0.18, 0.35);
        let thickness = if rng.unit() < 0.5 { 1 } else { 2 };
        let dx = angle.cos();
        let dy = angle.sin();
        for step in 0..length as i32 {
            for offset in -(thickness / 2)..=(thickness / 2) {
                let x = (x0 + step as f32 * dx - offset as f32 * dy).round() as i32;
                let y = (y0 + step as f32 * dy + offset as f32 * dx).round() as i32;
                if x >= 0 && y >= 0 && (x as usize) < SCENE_WIDTH && (y as usize) < SCENE_HEIGHT {
                    image.data[y as usize * SCENE_WIDTH + x as usize] = value;
                }
            }
        }
    }
}

fn occlude(
    image: &mut GrayF32,
    source: &GrayF32,
    pose: &pm_core::geometry::Similarity,
    rng: &mut SplitMix64,
) {
    // Estimate the object mask in scene coordinates, then erase a rectangular
    // patch covering roughly 30% of the object's pixels.
    let (sin, cos) = (pose.theta as f32).sin_cos();
    let inverse_scale = 1.0 / pose.scale as f32;
    let mut mask_pixels = Vec::new();
    for y in 0..SCENE_HEIGHT {
        for x in 0..SCENE_WIDTH {
            let dx = x as f32 - pose.tx as f32;
            let dy = y as f32 - pose.ty as f32;
            let px = (cos * dx + sin * dy) * inverse_scale + ZERO[0];
            let py = (-sin * dx + cos * dy) * inverse_scale + ZERO[1];
            if source.sample_bilinear(px, py) > 0.3 {
                mask_pixels.push((x, y));
            }
        }
    }
    if mask_pixels.is_empty() {
        return;
    }
    let target = (mask_pixels.len() as f32 * 0.3) as usize;
    let aspect = rng.range(0.6, 1.6);
    let width = ((target as f32 * aspect).sqrt() as usize).max(4);
    let height = ((target as f32 / aspect).sqrt() as usize).max(4);
    let (center_x, center_y) = mask_pixels[rng.next_u64() as usize % mask_pixels.len()];
    let x0 = center_x.saturating_sub(width / 2);
    let y0 = center_y.saturating_sub(height / 2);
    for y in y0..(y0 + height).min(SCENE_HEIGHT) {
        for x in x0..(x0 + width).min(SCENE_WIDTH) {
            image.data[y * SCENE_WIDTH + x] = BACKGROUND;
        }
    }
}

fn random_pose(rng: &mut SplitMix64, border_clip: bool) -> pm_core::geometry::Similarity {
    let theta = rng.range(-std::f32::consts::PI, std::f32::consts::PI);
    let scale = rng.range(0.5, 2.0);
    let extent = 40.0 * scale;
    let margin = extent + 3.0;
    let mut tx = rng.range(margin, SCENE_WIDTH as f32 - margin);
    let mut ty = rng.range(margin, SCENE_HEIGHT as f32 - margin);
    if border_clip {
        let side = (rng.next_u64() % 4) as usize;
        match side {
            0 => tx = rng.range(extent * 0.25, extent * 0.75),
            1 => tx = SCENE_WIDTH as f32 - rng.range(extent * 0.25, extent * 0.75),
            2 => ty = rng.range(extent * 0.25, extent * 0.75),
            _ => ty = SCENE_HEIGHT as f32 - rng.range(extent * 0.25, extent * 0.75),
        }
    }
    pm_core::geometry::Similarity::from_pose(tx as f64, ty as f64, theta as f64, scale as f64)
}

fn random_tilt_base(rng: &mut SplitMix64) -> Similarity {
    let theta = rng.range(-std::f32::consts::PI, std::f32::consts::PI);
    let scale = rng.range(0.7, 1.6);
    let margin = 50.0 * scale + 3.0;
    let tx = rng.range(margin, SCENE_WIDTH as f32 - margin);
    let ty = rng.range(margin, SCENE_HEIGHT as f32 - margin);
    Similarity::from_pose(tx as f64, ty as f64, theta as f64, scale as f64)
}

/// A physically derived plane tilt about the model origin.
///
/// The camera looks down `z`; the plane basis is `R_x(tilt_x) R_y(tilt_y)`
/// applied to the model axes and translated to distance `d`.  The focal length
/// is chosen so the local scale at the model origin equals the base similarity
/// scale.  Both tilt components are at most the requested magnitude.
fn tilt_homography(
    rng: &mut SplitMix64,
    base: &Similarity,
    min_deg: f32,
    max_deg: f32,
) -> Homography {
    let magnitude = rng.range(min_deg, max_deg).to_radians();
    let azimuth = rng.range(0.0, std::f32::consts::TAU);
    let tilt_x = magnitude * azimuth.sin();
    let tilt_y = magnitude * azimuth.cos();
    let distance = rng.range(500.0, 800.0);
    let cos_x = tilt_x.cos();
    let cos_y = tilt_y.cos();
    let focal = distance / (cos_x * cos_y).sqrt().max(1e-3);
    let tilt = Matrix3::new(
        f64::from(focal * cos_y),
        0.0,
        0.0,
        f64::from(focal * tilt_x.sin() * tilt_y.sin()),
        f64::from(focal * cos_x),
        0.0,
        f64::from(-cos_x * tilt_y.sin()),
        f64::from(tilt_x.sin()),
        f64::from(distance),
    );
    base.embed().compose(&Homography::from_matrix(tilt))
}

fn render_tilted_scene(source: &GrayF32, homography: &Homography) -> GrayF32 {
    let inverse = homography.try_inverse().expect("truth is invertible");
    let samples = 4usize;
    let mut data = vec![BACKGROUND; SCENE_WIDTH * SCENE_HEIGHT];
    for y in 0..SCENE_HEIGHT {
        for x in 0..SCENE_WIDTH {
            let mut sum = 0.0f32;
            let mut object = false;
            for sy in 0..samples {
                for sx in 0..samples {
                    let qx = x as f32 + (sx as f32 + 0.5) / samples as f32 - 0.5;
                    let qy = y as f32 + (sy as f32 + 0.5) / samples as f32 - 0.5;
                    let Some(p) =
                        inverse.try_act_point(nalgebra::Vector2::new(f64::from(qx), f64::from(qy)))
                    else {
                        continue;
                    };
                    let value = source.sample_bilinear(p.x as f32 + ZERO[0], p.y as f32 + ZERO[1]);
                    sum += value;
                    object |= value > 0.3;
                }
            }
            if object {
                data[y * SCENE_WIDTH + x] = (sum / (samples * samples) as f32).max(BACKGROUND);
            }
        }
    }
    GrayF32::new(SCENE_WIDTH, SCENE_HEIGHT, data).unwrap()
}

fn pose_matches(detection: &Detection, truth: &Similarity) -> bool {
    let distance = (detection.pose.tx - truth.tx).hypot(detection.pose.ty - truth.ty);
    let angle = (detection.pose.theta - truth.theta).rem_euclid(std::f64::consts::TAU);
    let angle = angle.min(std::f64::consts::TAU - angle);
    let scale_ratio = (detection.pose.scale / truth.scale).max(truth.scale / detection.pose.scale);
    distance < 4.0 && angle < 10.0f64.to_radians() && scale_ratio < 1.1
}

fn roi_reprojection_error(
    detection: &Detection,
    truth: &Homography,
    roi_model: &[[f32; 2]; 4],
) -> f32 {
    let detected = Homography::from_array(detection.homography);
    roi_model
        .iter()
        .map(|corner| {
            let p = nalgebra::Vector2::new(f64::from(corner[0]), f64::from(corner[1]));
            let expected = truth.try_act_point(p);
            let actual = detected.try_act_point(p);
            match (expected, actual) {
                (Some(expected), Some(actual)) => (expected - actual).norm() as f32,
                _ => f32::INFINITY,
            }
        })
        .fold(0.0f32, f32::max)
}

#[derive(Default)]
struct Metrics {
    ground_truth: usize,
    true_positives: usize,
    false_positives: usize,
    images: usize,
    homography_accepted: usize,
    fallback_reason_missing: usize,
    position_errors: Vec<f32>,
    angle_errors: Vec<f32>,
    scale_errors: Vec<f32>,
}

impl Metrics {
    fn add(&mut self, truths: &[pm_core::geometry::Similarity], detections: &[Detection]) {
        self.images += 1;
        self.ground_truth += truths.len();
        let mut used = vec![false; detections.len()];
        for truth in truths {
            if let Some((index, detection)) = detections
                .iter()
                .enumerate()
                .filter(|(index, _)| !used[*index])
                .find(|(_, detection)| pose_matches(detection, truth))
            {
                used[index] = true;
                self.true_positives += 1;
                if detection.homography_estimated {
                    self.homography_accepted += 1;
                } else if detection.diagnostics.fallback_reason.is_none() {
                    self.fallback_reason_missing += 1;
                }
                self.position_errors.push(
                    ((detection.pose.tx - truth.tx).hypot(detection.pose.ty - truth.ty)) as f32,
                );
                let angle = (detection.pose.theta - truth.theta).rem_euclid(std::f64::consts::TAU);
                let angle = angle.min(std::f64::consts::TAU - angle);
                self.angle_errors.push(angle.to_degrees() as f32);
                self.scale_errors
                    .push(((detection.pose.scale / truth.scale - 1.0).abs() * 100.0) as f32);
            }
        }
        self.false_positives += used.iter().filter(|used| !**used).count();
    }

    fn recall(&self) -> f32 {
        if self.ground_truth == 0 {
            1.0
        } else {
            self.true_positives as f32 / self.ground_truth as f32
        }
    }

    fn false_positives_per_image(&self) -> f32 {
        self.false_positives as f32 / self.images.max(1) as f32
    }

    fn max_or_zero(values: &[f32]) -> f32 {
        values.iter().copied().fold(0.0, f32::max)
    }

    fn report(&self, name: &str) {
        println!(
            "{name}: recall {:.3} ({}/{}), FP/image {:.3}, accepted H {}, missing fallback reason {}, \
             max pos {:.3} px, max angle {:.3}°, max scale {:.3}%",
            self.recall(),
            self.true_positives,
            self.ground_truth,
            self.false_positives_per_image(),
            self.homography_accepted,
            self.fallback_reason_missing,
            Self::max_or_zero(&self.position_errors),
            Self::max_or_zero(&self.angle_errors),
            Self::max_or_zero(&self.scale_errors)
        );
    }
}

fn run_scenes(
    model: &Model,
    params: &SearchParams,
    seed: u64,
    count: usize,
    make_scene: impl Fn(&mut SplitMix64, usize) -> (GrayF32, Vec<pm_core::geometry::Similarity>),
) -> Metrics {
    let mut metrics = Metrics::default();
    for index in 0..count {
        let mut rng = SplitMix64::new(seed.wrapping_add(index as u64 * 7919));
        let (scene, truths) = make_scene(&mut rng, index);
        let result = detect(model, &scene, params).expect("detect");
        metrics.add(&truths, &result.detections);
    }
    metrics
}

#[test]
#[ignore]
fn diagnose_clean_homography() {
    let model = teach_model();
    let source = model.source.as_ref().expect("embedded source").gray.clone();
    let params = SearchParams::from(model.search_defaults);
    for index in 0..16 {
        let mut rng = SplitMix64::new(1000 + index * 7919);
        let truth = random_pose(&mut rng, false);
        let scene = render_scene(
            &source,
            &[truth],
            rng.range(0.9, 1.1),
            rng.range(-0.03, 0.03),
        );
        let result = detect(&model, &scene, &params).expect("detect");
        for detection in &result.detections {
            let matched = pose_matches(detection, &truth);
            println!(
                "scene {index}: match {matched}, score {:.3}, est {}, reason {:?}, cond {:?}, \
                 corr {}, inl {}, corner corr {}, corner inl {}, pose x {:.1} y {:.1} theta {:.3} \
                 scale {:.3}, truth x {:.1} y {:.1} theta {:.3} scale {:.3}",
                detection.score,
                detection.homography_estimated,
                detection.diagnostics.fallback_reason,
                detection.diagnostics.homography_condition,
                detection.diagnostics.correspondences,
                detection.diagnostics.inliers,
                detection.diagnostics.corner_correspondences,
                detection.diagnostics.corner_inliers,
                detection.pose.tx,
                detection.pose.ty,
                detection.pose.theta,
                detection.pose.scale,
                truth.tx,
                truth.ty,
                truth.theta,
                truth.scale,
            );
        }
    }
}

#[test]
#[ignore]
fn diagnose_clean_errors() {
    let model = teach_model();
    let source = model.source.as_ref().expect("embedded source").gray.clone();
    let params = SearchParams::from(model.search_defaults);
    for index in 0..16 {
        let mut rng = SplitMix64::new(1000 + index * 7919);
        let truth = random_pose(&mut rng, false);
        let scene = render_scene(
            &source,
            &[truth],
            rng.range(0.9, 1.1),
            rng.range(-0.03, 0.03),
        );
        let result = detect(&model, &scene, &params).expect("detect");
        if let Some(detection) = result
            .detections
            .iter()
            .find(|detection| pose_matches(detection, &truth))
        {
            let position = (detection.pose.tx - truth.tx).hypot(detection.pose.ty - truth.ty);
            let angle = (detection.pose.theta - truth.theta).rem_euclid(std::f64::consts::TAU);
            let angle = angle.min(std::f64::consts::TAU - angle);
            let _ = (position, angle);
        }
    }
}

#[test]
fn synthetic_similarity_benchmark() {
    let model = teach_model();
    let source = model.source.as_ref().expect("embedded source").gray.clone();
    let params = SearchParams::from(model.search_defaults);
    let mut hybrid_clean = Metrics::default();
    for index in 0..16 {
        let mut rng = SplitMix64::new(1000 + index * 7919);
        let truth = random_pose(&mut rng, false);
        let scene = render_scene(
            &source,
            &[truth],
            rng.range(0.9, 1.1),
            rng.range(-0.03, 0.03),
        );
        let result = detect(&model, &scene, &params).expect("detect");
        hybrid_clean.add(&[truth], &result.detections);
    }
    hybrid_clean.report("clean similarity hybrid");
    assert_eq!(
        hybrid_clean.fallback_reason_missing, 0,
        "fallback reason missing on a clean-similarity detection"
    );

    let mut hough_clean = Metrics::default();
    let hough_params = SearchParams {
        hypothesis_sources: pm_core::model::HypothesisSources::Hough,
        ..params
    };
    for index in 0..8 {
        let mut rng = SplitMix64::new(1000 + index * 7919);
        let truth = random_pose(&mut rng, false);
        let scene = render_scene(&source, &[truth], 1.0, 0.0);
        let result = detect(&model, &scene, &hough_params).expect("detect");
        hough_clean.add(&[truth], &result.detections);
    }
    hough_clean.report("clean similarity Hough-only");

    let low_contrast = run_scenes(&model, &params, 1500, 8, |rng, _| {
        let truth = random_pose(rng, false);
        let scene = render_scene(&source, &[truth], 0.5, 0.0);
        (scene, vec![truth])
    });
    low_contrast.report("low contrast hybrid");
    let low_contrast_hough = run_scenes(&model, &hough_params, 1500, 8, |rng, _| {
        let truth = random_pose(rng, false);
        let scene = render_scene(&source, &[truth], 0.5, 0.0);
        (scene, vec![truth])
    });
    low_contrast_hough.report("low contrast Hough-only");

    let occluded = run_scenes(&model, &params, 2000, 10, |rng, _| {
        let truth = random_pose(rng, false);
        let mut scene = render_scene(&source, &[truth], 1.0, 0.0);
        occlude(&mut scene, &source, &truth, rng);
        (scene, vec![truth])
    });
    occluded.report("30% occlusion");

    let clutter = run_scenes(&model, &params, 3000, 8, |rng, _| {
        let mut scene = GrayF32::filled(SCENE_WIDTH, SCENE_HEIGHT, BACKGROUND);
        add_clutter(&mut scene, rng, 25);
        (scene, Vec::new())
    });
    clutter.report("clutter-only");

    let clutter_with_object = run_scenes(&model, &params, 4000, 6, |rng, _| {
        let truth = random_pose(rng, false);
        let mut scene = render_scene(&source, &[truth], 1.0, 0.0);
        add_clutter(&mut scene, rng, 20);
        (scene, vec![truth])
    });
    clutter_with_object.report("clutter + object");

    let border = run_scenes(&model, &params, 5000, 6, |rng, _| {
        let truth = random_pose(rng, true);
        let scene = render_scene(&source, &[truth], 1.0, 0.0);
        (scene, vec![truth])
    });
    border.report("border-clipped");

    let multi = run_scenes(&model, &params, 6000, 4, |rng, _| {
        let first = random_pose(rng, false);
        let mut second = random_pose(rng, false);
        while (first.tx - second.tx).abs() < 70.0 && (first.ty - second.ty).abs() < 70.0 {
            second = random_pose(rng, false);
        }
        let scene = render_scene(&source, &[first, second], 1.0, 0.0);
        (scene, vec![first, second])
    });
    multi.report("multi-instance");

    println!(
        "Hough-only recall {:.3}, hybrid recall {:.3}",
        hough_clean.recall(),
        hybrid_clean.recall()
    );
    assert!(
        hybrid_clean.recall() >= 0.99,
        "clean similarity recall {:.3}",
        hybrid_clean.recall()
    );
    assert!(
        hybrid_clean.false_positives_per_image() <= 0.2,
        "clean FP/image {:.3}",
        hybrid_clean.false_positives_per_image()
    );
    assert!(
        hybrid_clean.recall() + 1e-6 >= hough_clean.recall(),
        "hybrid recall {:.3} below Hough-only {:.3}",
        hybrid_clean.recall(),
        hough_clean.recall()
    );
    assert!(
        Metrics::max_or_zero(&hybrid_clean.position_errors) <= 0.3,
        "position error"
    );
    assert!(
        Metrics::max_or_zero(&hybrid_clean.angle_errors) <= 0.2,
        "angle error"
    );
    assert!(
        Metrics::max_or_zero(&hybrid_clean.scale_errors) <= 0.3,
        "scale error"
    );
    assert!(
        low_contrast.recall() >= 0.85,
        "low-contrast recall {:.3} (Hough-only {:.3})",
        low_contrast.recall(),
        low_contrast_hough.recall()
    );
    assert!(
        occluded.recall() >= 0.9,
        "occlusion recall {:.3}",
        occluded.recall()
    );
    assert!(
        clutter.false_positives_per_image() <= 0.2,
        "clutter FP/image {:.3}",
        clutter.false_positives_per_image()
    );
    assert!(
        clutter_with_object.recall() >= 0.9,
        "clutter object recall {:.3}",
        clutter_with_object.recall()
    );
    assert!(
        border.recall() >= 0.8,
        "border recall {:.3}",
        border.recall()
    );
    assert!(multi.recall() >= 0.9, "multi recall {:.3}", multi.recall());
}

#[test]
fn synthetic_tilt_benchmark() {
    let model = teach_model();
    let source = model.source.as_ref().expect("embedded source").gray.clone();
    let params = SearchParams::from(model.search_defaults);
    let roi_model: [[f32; 2]; 4] = model
        .roi
        .corners()
        .map(|corner| model.zero.to_model(corner));
    let mut scenes = 0usize;
    let mut correct = 0usize;
    let mut accepted = 0usize;
    let mut fallback = 0usize;
    let mut worst_reprojection = 0.0f32;
    let mut worst_fallback_reprojection = 0.0f32;

    for index in 0..12 {
        let mut rng = SplitMix64::new(9000 + index as u64 * 7919);
        let base = random_tilt_base(&mut rng);
        let truth = tilt_homography(&mut rng, &base, 8.0, 20.0);
        let local = truth.pose_at_origin().expect("truth pose");
        let truth_pose = Similarity::from_pose(local.x, local.y, local.angle, local.scale);
        let scene = render_tilted_scene(&source, &truth);
        let result = detect(&model, &scene, &params).expect("detect");
        scenes += 1;
        let Some(detection) = result
            .detections
            .iter()
            .find(|detection| pose_matches(detection, &truth_pose))
        else {
            continue;
        };
        correct += 1;
        let reprojection = roi_reprojection_error(detection, &truth, &roi_model);
        if detection.homography_estimated {
            accepted += 1;
            worst_reprojection = worst_reprojection.max(reprojection);
        } else {
            fallback += 1;
            worst_fallback_reprojection = worst_fallback_reprojection.max(reprojection);
        }
    }

    let recall = correct as f32 / scenes as f32;
    let accepted_fraction = accepted as f32 / correct.max(1) as f32;
    println!(
        "tilt ≤20°: recall {recall:.3} ({correct}/{scenes}), accepted H {accepted_fraction:.3} \
         ({accepted}/{correct}), fallbacks {fallback}, max ROI reprojection {worst_reprojection:.3} px \
         (fallback worst {worst_fallback_reprojection:.3} px)"
    );
    assert!(recall >= 0.9, "tilt recall {recall:.3}");
    assert!(
        accepted_fraction >= 0.8,
        "accepted homography fraction {accepted_fraction:.3}"
    );
    assert!(
        worst_reprojection <= 1.5,
        "ROI-corner reprojection {worst_reprojection:.3} px"
    );
}
