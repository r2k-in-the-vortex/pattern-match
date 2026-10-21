//! Deterministic synthetic benchmark (README §13.3).
//!
//! A taught L-bracket is rendered into test scenes with known similarity
//! poses.  The benchmark reports detection rate, position/angle/scale error,
//! false detections, and Hough-only versus hybrid recall.  The tilt scenes
//! render the same bracket through a known perspective homography and report
//! recall, ROI-corner reprojection error, and the fraction of correct
//! detections that accept an estimated homography (README §13.4).
//!
//! Scene generation and metric accumulation live in [`pm_core::synthetic`];
//! this test fixes the scenario seeds and counts so regressions are comparable
//! across runs, and CLI `bench synthetic` shares the same generator.

use pm_core::synthetic::{BenchmarkMetrics, SplitMix64, SyntheticRenderer};
use pm_core::{
    FeatureParams, GrayF32, LoadedImage, Model, RotatedRect, SearchDefaults, SearchParams,
    TeachSettings, ZeroFrame, detect,
};

const SOURCE_WIDTH: usize = 128;
const SOURCE_HEIGHT: usize = 128;
const ZERO: [f32; 2] = [64.0, 64.0];
const BACKGROUND: f32 = 0.12;

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

fn renderer(model: &Model) -> SyntheticRenderer<'_> {
    // Keep the historical L-bracket margin so benchmark numbers stay
    // comparable with earlier milestones; the derived radius is close.
    SyntheticRenderer::new(model)
        .expect("embedded source")
        .with_extent(40.0)
        .with_foreground_threshold(0.18)
}

fn report(name: &str, metrics: &BenchmarkMetrics) {
    println!(
        "{name}: recall {:.3} ({}/{}), FP/image {:.3}, accepted H {}, missing fallback reason {}, \
         max pos {:.3} px, max angle {:.3}°, max scale {:.3}%",
        metrics.recall(),
        metrics.true_positives,
        metrics.ground_truth,
        metrics.false_positives_per_image(),
        metrics.homography_accepted,
        metrics.fallback_reason_missing,
        metrics.max_position_error(),
        metrics.max_angle_error(),
        metrics.max_scale_error()
    );
}

#[test]
fn synthetic_similarity_benchmark() {
    let model = teach_model();
    let renderer = renderer(&model);
    let params = SearchParams::from(model.search_defaults);
    let hough_params = SearchParams {
        hypothesis_sources: pm_core::model::HypothesisSources::Hough,
        ..params
    };

    let mut hybrid_clean = BenchmarkMetrics::default();
    for index in 0..16 {
        let mut rng = SplitMix64::new(1000 + index as u64 * 7919);
        let truth = renderer.random_pose(&mut rng, false);
        let scene = renderer.render(&[truth], rng.range(0.9, 1.1), rng.range(-0.03, 0.03));
        let result = detect(&model, &scene, &params).expect("detect");
        hybrid_clean.add_similarity(&[truth], &result.detections);
    }
    report("clean similarity hybrid", &hybrid_clean);
    assert_eq!(
        hybrid_clean.fallback_reason_missing, 0,
        "fallback reason missing on a clean-similarity detection"
    );

    let mut hough_clean = BenchmarkMetrics::default();
    for index in 0..8 {
        let mut rng = SplitMix64::new(1000 + index as u64 * 7919);
        let truth = renderer.random_pose(&mut rng, false);
        let scene = renderer.render(&[truth], 1.0, 0.0);
        let result = detect(&model, &scene, &hough_params).expect("detect");
        hough_clean.add_similarity(&[truth], &result.detections);
    }
    report("clean similarity Hough-only", &hough_clean);

    let mut low_contrast = BenchmarkMetrics::default();
    let mut low_contrast_hough = BenchmarkMetrics::default();
    for index in 0..8 {
        let mut rng = SplitMix64::new(1500 + index as u64 * 7919);
        let truth = renderer.random_pose(&mut rng, false);
        let scene = renderer.render(&[truth], 0.5, 0.0);
        let result = detect(&model, &scene, &params).expect("detect");
        low_contrast.add_similarity(&[truth], &result.detections);
        let hough_result = detect(&model, &scene, &hough_params).expect("detect");
        low_contrast_hough.add_similarity(&[truth], &hough_result.detections);
    }
    report("low contrast hybrid", &low_contrast);
    report("low contrast Hough-only", &low_contrast_hough);

    let mut occluded = BenchmarkMetrics::default();
    for index in 0..10 {
        let mut rng = SplitMix64::new(2000 + index as u64 * 7919);
        let truth = renderer.random_pose(&mut rng, false);
        let mut scene = renderer.render(&[truth], 1.0, 0.0);
        renderer.occlude(&mut scene, &truth, &mut rng);
        let result = detect(&model, &scene, &params).expect("detect");
        occluded.add_similarity(&[truth], &result.detections);
    }
    report("30% occlusion", &occluded);

    let (width, height) = renderer.dimensions();
    let mut clutter = BenchmarkMetrics::default();
    for index in 0..8 {
        let mut rng = SplitMix64::new(3000 + index as u64 * 7919);
        let mut scene = GrayF32::filled(width, height, renderer.background());
        renderer.add_clutter(&mut scene, &mut rng, 25);
        let result = detect(&model, &scene, &params).expect("detect");
        clutter.add_similarity(&[], &result.detections);
    }
    report("clutter-only", &clutter);

    let mut clutter_with_object = BenchmarkMetrics::default();
    for index in 0..6 {
        let mut rng = SplitMix64::new(4000 + index as u64 * 7919);
        let truth = renderer.random_pose(&mut rng, false);
        let mut scene = renderer.render(&[truth], 1.0, 0.0);
        renderer.add_clutter(&mut scene, &mut rng, 20);
        let result = detect(&model, &scene, &params).expect("detect");
        clutter_with_object.add_similarity(&[truth], &result.detections);
    }
    report("clutter + object", &clutter_with_object);

    let mut border = BenchmarkMetrics::default();
    for index in 0..6 {
        let mut rng = SplitMix64::new(5000 + index as u64 * 7919);
        let truth = renderer.random_pose(&mut rng, true);
        let scene = renderer.render(&[truth], 1.0, 0.0);
        let result = detect(&model, &scene, &params).expect("detect");
        border.add_similarity(&[truth], &result.detections);
    }
    report("border-clipped", &border);

    let mut multi = BenchmarkMetrics::default();
    for index in 0..4 {
        let mut rng = SplitMix64::new(6000 + index as u64 * 7919);
        let first = renderer.random_pose(&mut rng, false);
        let mut second = renderer.random_pose(&mut rng, false);
        while (first.tx - second.tx).abs() < 70.0 && (first.ty - second.ty).abs() < 70.0 {
            second = renderer.random_pose(&mut rng, false);
        }
        let scene = renderer.render(&[first, second], 1.0, 0.0);
        let result = detect(&model, &scene, &params).expect("detect");
        multi.add_similarity(&[first, second], &result.detections);
    }
    report("multi-instance", &multi);

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
        hybrid_clean.max_position_error() <= 0.3,
        "position error {:.3}",
        hybrid_clean.max_position_error()
    );
    assert!(
        hybrid_clean.max_angle_error() <= 0.2,
        "angle error {:.3}",
        hybrid_clean.max_angle_error()
    );
    assert!(
        hybrid_clean.max_scale_error() <= 0.3,
        "scale error {:.3}",
        hybrid_clean.max_scale_error()
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
    let renderer = renderer(&model);
    let params = SearchParams::from(model.search_defaults);
    let roi_model: [[f32; 2]; 4] = model
        .roi
        .corners()
        .map(|corner| model.zero.to_model(corner));
    let mut metrics = BenchmarkMetrics::default();

    for index in 0..12 {
        let mut rng = SplitMix64::new(9000 + index as u64 * 7919);
        let base = renderer.random_tilt_base(&mut rng);
        let truth = renderer.tilt_homography(&mut rng, &base, 8.0, 20.0);
        let local = truth.pose_at_origin().expect("truth pose");
        let truth_pose =
            pm_core::geometry::Similarity::from_pose(local.x, local.y, local.angle, local.scale);
        let scene = renderer.render_tilted(&truth);
        let result = detect(&model, &scene, &params).expect("detect");
        metrics.add_tilt(&truth_pose, &truth, &roi_model, &result.detections);
    }

    let scenario = metrics.report("tilt");
    let correctness = metrics.true_positives as f32 / metrics.cases as f32;
    let accepted_fraction =
        metrics.homography_accepted as f32 / metrics.true_positives.max(1) as f32;
    println!(
        "tilt ≤20°: recall {correctness:.3} ({}/{}), accepted H {accepted_fraction:.3} \
         ({}/{}), fallbacks {}, max ROI reprojection {:.3} px (fallback worst {:.3} px)",
        metrics.true_positives,
        metrics.cases,
        metrics.homography_accepted,
        metrics.true_positives,
        metrics.homography_fallback,
        scenario.max_roi_reprojection_px.unwrap_or(0.0),
        scenario.max_fallback_roi_reprojection_px.unwrap_or(0.0)
    );
    assert!(correctness >= 0.9, "tilt recall {correctness:.3}");
    assert!(
        accepted_fraction >= 0.8,
        "accepted homography fraction {accepted_fraction:.3}"
    );
    assert!(
        scenario.max_roi_reprojection_px.unwrap_or(0.0) <= 1.5,
        "ROI-corner reprojection {:.3} px",
        scenario.max_roi_reprojection_px.unwrap_or(0.0)
    );
}
