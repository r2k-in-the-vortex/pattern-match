//! Temporary fine-grained performance profile for `pm-core` detection.
//!
//! Run with:
//! ```text
//! cargo run --release -j 1 -p pm-core --example profile -- \
//!     models/Bracket.pmodel testdata/synthetic/Untitled.png
//! ```

use std::time::Instant;

use pm_core::corner::detect_corners;
use pm_core::descriptor::describe_corners;
use pm_core::edge::{DistanceField, extract_edges};
use pm_core::gradient::GradientField;
use pm_core::search::{
    TestPyramid, cluster_hypotheses, descriptor_hypotheses, hough_hypotheses, model_roi_diagonal,
    segment_hypotheses, select_coarse_level,
};
use pm_core::{Model, Pyramid, SearchParams, detect, load_image};

fn ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn detect_ms(label: &str, model: &Model, image: &pm_core::GrayF32, params: &SearchParams) {
    let t = Instant::now();
    let result = detect(model, image, params).expect("detect");
    eprintln!(
        "  {label}: total {:.1} ms (preprocess {:.1}, features {:.1}, search {:.1}, refine {:.1}), \
         {} gen / {} scored / {} det",
        ms(t),
        result.timings.preprocess_ms,
        result.timings.features_ms,
        result.timings.search_ms,
        result.timings.refine_ms,
        result.hypotheses_generated,
        result.hypotheses_scored,
        result.detections.len()
    );
}

fn main() {
    let mut args = std::env::args().skip(1);
    let model_path = args
        .next()
        .unwrap_or_else(|| "models/Bracket.pmodel".into());
    let image_path = args
        .next()
        .unwrap_or_else(|| "testdata/synthetic/Untitled.png".into());

    let t = Instant::now();
    let model = Model::load(&model_path).expect("load model");
    eprintln!("load model: {:.1} ms", ms(t));

    let t = Instant::now();
    let image = load_image(&image_path).expect("load image");
    eprintln!("load image (decode+gray): {:.1} ms", ms(t));
    eprintln!(
        "image {}x{}, model edges {}, corners {}, segments {}, pairs {}",
        image.gray.width,
        image.gray.height,
        model.features.edge_count(),
        model.features.corners.len(),
        model.features.segments.len(),
        model.segment_pairs.len()
    );

    let params = SearchParams::from(model.search_defaults);

    // Feature pipeline, component by component.
    let t = Instant::now();
    let base = image.gray.gaussian_blur(model.params.blur_sigma);
    eprintln!("  blur sigma {}: {:.1} ms", model.params.blur_sigma, ms(t));

    let t = Instant::now();
    let pyramid = Pyramid::build(&base, 3);
    eprintln!(
        "  pyramid build ({} levels): {:.1} ms",
        pyramid.len(),
        ms(t)
    );

    let mut built_levels = Vec::new();
    for level in 0..pyramid.len() {
        let gray = pyramid.level_unchecked(level).clone();
        let t = Instant::now();
        let gradient = GradientField::compute(&gray);
        let g = ms(t);
        let t = Instant::now();
        let edges = extract_edges(&gradient, &model.params.edge);
        let e = ms(t);
        let t = Instant::now();
        let _distance = DistanceField::from_edges(&edges, gray.width, gray.height);
        let d = ms(t);
        let t = Instant::now();
        let corners = detect_corners(&gradient, &model.params.corner);
        let c = ms(t);
        let t = Instant::now();
        let segments = pm_core::segment::fit_segments(&edges, &model.params.segment);
        let s = ms(t);
        eprintln!(
            "  L{level} {}x{} gradient {:.1}, edges {:.1} ({} pts, {:?} chain lengths), EDT {:.1}, corners {:.1} ({}), segments {:.1} ({}): {:.1} ms",
            gray.width,
            gray.height,
            g,
            e,
            edges.points.len(),
            edges
                .chains
                .iter()
                .map(|c| c.points.len())
                .collect::<Vec<_>>(),
            d,
            c,
            corners.len(),
            s,
            segments.len(),
            g + e + d + c + s,
        );
        // Descriptor computation cost for this level's corners.
        let t = Instant::now();
        let desc = describe_corners(&corners, &edges);
        eprintln!(
            "      describe_corners: {:.1} ms ({} corners x {} edges, dim {})",
            ms(t),
            corners.len(),
            edges.points.len(),
            desc.first().map_or(0, |d| d.values.len())
        );
        built_levels.push((edges, corners));
    }
    drop(built_levels);

    let t = Instant::now();
    let test = TestPyramid::build(&image.gray, 3, &model.params);
    eprintln!(
        "TestPyramid::build total: {:.1} ms (preprocess {:.1}, features {:.1})",
        ms(t),
        test.preprocess_ms,
        test.features_ms
    );

    let coarse = select_coarse_level(&model, &params, test.levels.len());
    eprintln!(
        "coarse level: {coarse} ({} levels), model pts L{coarse} {}, test pts {}",
        test.levels.len(),
        model.features_at_level(coarse).edges.points.len(),
        test.levels[coarse].edges.points.len()
    );
    let level = &test.levels[coarse];
    let model_points = &model.features_at_level(coarse).edges.points;
    let max_radius = model_points
        .iter()
        .map(|p| (p.x * p.x + p.y * p.y).sqrt())
        .fold(0.0f32, f32::max);
    let margin = (max_radius * params.scale_max.max(1.0)).ceil() as i32
        + (params.orientation_gate * 2.0).ceil() as i32;
    let acc_w = level.gray.width as i32 + 2 * margin;
    let acc_h = level.gray.height as i32 + 2 * margin;
    eprintln!(
        "  hough geometry: max model radius {max_radius:.1}, margin {margin}, accumulator {acc_w}x{acc_h} x {} scales = {:.1} M cells = {:.1} MB per rotation",
        params.scale_bins,
        (acc_w as f64 * acc_h as f64 * params.scale_bins as f64) / 1e6,
        (acc_w as f64 * acc_h as f64 * params.scale_bins as f64 * 4.0) / 1e6
    );

    let t = Instant::now();
    let hough = hough_hypotheses(&model, level, coarse, &params);
    eprintln!("hough: {:.1} ms ({} hypotheses)", ms(t), hough.len());

    let t = Instant::now();
    let desc = descriptor_hypotheses(&model, &test.levels, coarse, &params);
    eprintln!("descriptors: {:.1} ms ({} hypotheses)", ms(t), desc.len());

    let t = Instant::now();
    let seg = segment_hypotheses(&model, &test.levels, coarse, &params);
    eprintln!("segments: {:.1} ms ({} hypotheses)", ms(t), seg.len());

    let mut all = Vec::new();
    all.extend(hough);
    all.extend(desc);
    all.extend(seg);
    let diag = model_roi_diagonal(&model);
    let t = Instant::now();
    let clustered = cluster_hypotheses(all, diag, &params);
    eprintln!("cluster: {:.1} ms (-> {})", ms(t), clustered.len());
    drop(clustered);

    let t = Instant::now();
    let result = detect(&model, &image.gray, &params).expect("detect");
    eprintln!(
        "detect total: {:.1} ms (preprocess {:.1}, features {:.1}, search {:.1}, refine {:.1}), \
         {} hypotheses generated, {} scored",
        ms(t),
        result.timings.preprocess_ms,
        result.timings.features_ms,
        result.timings.search_ms,
        result.timings.refine_ms,
        result.hypotheses_generated,
        result.hypotheses_scored
    );
    eprintln!("detections: {}", result.detections.len());
    for detection in &result.detections {
        eprintln!(
            "det tx={:.4} ty={:.4} theta={:.6} scale={:.6} score={:.4} hom={}",
            detection.pose.tx,
            detection.pose.ty,
            detection.pose.theta,
            detection.pose.scale,
            detection.score,
            detection.homography_estimated
        );
    }

    // Parameter sweeps to attribute Hough cost.
    eprintln!("Hough parameter sweep:");
    for (rot_bins, scale_bins, test_cap, model_cap) in [
        (60, 12, 1500, 400),
        (1, 12, 1500, 400),
        (60, 1, 1500, 400),
        (60, 12, 500, 400),
        (60, 12, 1500, 100),
        (1, 1, 1500, 400),
    ] {
        let mut p = params;
        p.rotation_bins = rot_bins;
        p.scale_bins = scale_bins;
        p.max_test_hough_points = test_cap;
        p.max_model_hough_points = model_cap;
        let t = Instant::now();
        let h = hough_hypotheses(&model, level, coarse, &p);
        eprintln!(
            "  rot {rot_bins:2} scale {scale_bins:2} test {test_cap:4} model {model_cap:3}: {:7.1} ms ({} hyp)",
            ms(t),
            h.len()
        );
    }

    eprintln!("Refine parameter sweep:");
    detect_ms("baseline", &model, &image.gray, &params);
    let mut no_h = params;
    no_h.homography.enabled = false;
    detect_ms("homography disabled", &model, &image.gray, &no_h);
    let mut one = params;
    one.max_hypotheses_scored = 1;
    detect_ms("1 hypothesis", &model, &image.gray, &one);
    let mut no_corners = params;
    no_corners.hypothesis_sources = pm_core::HypothesisSources::Hough;
    detect_ms("hough only", &model, &image.gray, &no_corners);
}
