//! Regression for the six-instance bracket scene.
//!
//! `models/Bracket.pmodel` embeds its 1024×1024 source image, which contains
//! six copies of the taught bracket at different rotations/translations.  The
//! model ROI covers only the central instance.  With the default per-image
//! verification budget of 24 hypotheses, duplicate coarse seeds for the five
//! strongest instances used to crowd out the sixth before it was refined.  The
//! diversity pass in `rank_hypotheses` keeps the budget on distinct poses, so
//! all six must be found.

use std::path::Path;

use pm_core::{Model, SearchParams, detect};

#[test]
fn all_six_bracket_instances_are_detected() {
    let model_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/Bracket.pmodel");
    let model = Model::load(&model_path).expect("load models/Bracket.pmodel fixture");
    let source = model
        .source
        .clone()
        .expect("fixture model must embed its source image");
    let params = SearchParams::from(model.search_defaults);

    let result = detect(&model, &source.gray, &params).expect("detect");
    let expected = [
        (479.76, 556.34),
        (520.89, 873.21),
        (742.74, 164.71),
        (174.17, 553.26),
        (324.80, 220.76),
        (779.57, 346.35),
    ];
    assert_eq!(
        result.detections.len(),
        expected.len(),
        "detections: {:?}",
        result
            .detections
            .iter()
            .map(|detection| (detection.pose.tx, detection.pose.ty))
            .collect::<Vec<_>>()
    );
    for (x, y) in expected {
        assert!(
            result
                .detections
                .iter()
                .any(|detection| (detection.pose.tx - x).abs() < 2.0
                    && (detection.pose.ty - y).abs() < 2.0),
            "missing instance near ({x}, {y})"
        );
    }
}
