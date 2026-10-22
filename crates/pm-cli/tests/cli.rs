//! End-to-end tests for the `pm-cli` command surface.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use pm_core::{
    FeatureParams, GrayF32, LoadedImage, Model, ResultsFile, RotatedRect, SearchDefaults,
    SearchParams, Similarity, TeachSettings, ZeroFrame, detect,
};

const ZERO: [f32; 2] = [62.0, 63.0];
const BACKGROUND: f32 = 0.12;

struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!("pm-cli-test-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).expect("create temp dir");
        Self { path }
    }

    fn join(&self, relative: &str) -> PathBuf {
        self.path.join(relative)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
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

fn source_image() -> GrayF32 {
    let width = 128;
    let height = 128;
    let polygon = [
        [38.0, 32.0],
        [86.0, 32.0],
        [86.0, 52.0],
        [58.0, 52.0],
        [58.0, 94.0],
        [38.0, 94.0],
    ];
    let hole = (72.0f32, 44.0f32, 6.0f32);
    let mut data = vec![BACKGROUND; width * height];
    for y in 0..height {
        for x in 0..width {
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
            data[y * width + x] = BACKGROUND + covered as f32 / 9.0 * 0.78;
        }
    }
    GrayF32::new(width, height, data).unwrap()
}

fn teach(dir: &TempDir) -> PathBuf {
    let gray = source_image();
    let display = gray.display_rgba();
    let loaded = LoadedImage { gray, display };
    let settings = TeachSettings {
        name: "cli-test".to_owned(),
        roi: RotatedRect::new(ZERO[0], ZERO[1], 88.0, 88.0, 0.0),
        zero: ZeroFrame {
            origin: ZERO,
            angle: 0.0,
        },
        params: FeatureParams::default(),
        mask: Vec::new(),
        search_defaults: SearchDefaults::default(),
    };
    let model = Model::build(&loaded, &settings).expect("build model");
    let path = dir.join("model.pmodel");
    model.save(&path).expect("save model");
    path
}

fn render_scene(source: &GrayF32, pose: &Similarity, width: usize, height: usize) -> GrayF32 {
    let mut data = vec![BACKGROUND; width * height];
    let (sin, cos) = (pose.theta as f32).sin_cos();
    let inverse_scale = 1.0 / pose.scale as f32;
    for y in 0..height {
        for x in 0..width {
            let dx = x as f32 - pose.tx as f32;
            let dy = y as f32 - pose.ty as f32;
            let px = (cos * dx + sin * dy) * inverse_scale + ZERO[0];
            let py = (-sin * dx + cos * dy) * inverse_scale + ZERO[1];
            let value = source.sample_bilinear(px, py);
            if value > 0.3 {
                data[y * width + x] = value;
            }
        }
    }
    GrayF32::new(width, height, data).unwrap()
}

fn save_png(gray: &GrayF32, path: &Path) {
    gray.display_rgba().save(path).expect("save PNG");
}

fn assert_success(output: &Output, context: &str) {
    assert!(
        output.status.success(),
        "{context}: status {:?}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn load_results(path: &Path) -> ResultsFile {
    let json = fs::read_to_string(path).expect("read result JSON");
    ResultsFile::from_json(&json).expect("parse result JSON")
}

#[test]
fn run_writes_deterministic_json_csv_and_overlay() {
    let dir = TempDir::new("run");
    let model = teach(&dir);
    let scene = render_scene(
        &source_image(),
        &Similarity::from_pose(140.0, 120.0, 0.6, 1.2),
        256,
        224,
    );
    let image = dir.join("scene.png");
    save_png(&scene, &image);

    let first = dir.join("first.json");
    let first_csv = dir.join("first.csv");
    let overlay = dir.join("overlay.png");
    let second = dir.join("second.json");

    let output = Command::new(env!("CARGO_BIN_EXE_pm-cli"))
        .arg("run")
        .arg("--model")
        .arg(&model)
        .arg("--image")
        .arg(&image)
        .arg("--output")
        .arg(&first)
        .arg("--csv")
        .arg(&first_csv)
        .arg("--overlay")
        .arg(&overlay)
        .arg("--timings")
        .output()
        .expect("run pm-cli");
    assert_success(&output, "first run");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("detection(s)"), "stdout: {stdout}");
    assert!(stdout.contains("timings:"), "stdout: {stdout}");

    let output = Command::new(env!("CARGO_BIN_EXE_pm-cli"))
        .arg("run")
        .arg("--model")
        .arg(&model)
        .arg("--image")
        .arg(&image)
        .arg("--output")
        .arg(&second)
        .output()
        .expect("run pm-cli");
    assert_success(&output, "second run");

    let first_doc = load_results(&first);
    let second_doc = load_results(&second);
    assert_eq!(first_doc.model, "cli-test");
    assert_eq!((first_doc.image.width, first_doc.image.height), (256, 224));
    assert!(!first_doc.detections.is_empty(), "no detections");
    assert_eq!(first_doc.detections, second_doc.detections);
    assert_eq!(first_doc.parameters, second_doc.parameters);
    assert!(
        first_doc.detections[0].score >= first_doc.parameters.score_threshold,
        "score below threshold"
    );

    // The CLI must reproduce the library result for the same model, image, and
    // parameters (README §14, M7 exit criteria).  Compare the serialized
    // schema, which omits the internal corner diagnostics.
    let loaded_model = Model::load(&model).expect("reload model");
    let loaded_image = pm_core::load_image(&image).expect("reload image");
    let params = SearchParams::from(loaded_model.search_defaults);
    let library_result =
        detect(&loaded_model, &loaded_image.gray, &params).expect("library detect");
    let library_document = ResultsFile::new(
        &loaded_model.name,
        loaded_image.gray.width,
        loaded_image.gray.height,
        &params,
        &library_result,
    );
    let cli_json: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&first).expect("read first result"))
            .expect("first result JSON");
    let library_json: serde_json::Value =
        serde_json::from_str(&library_document.to_json().expect("library JSON"))
            .expect("library result JSON");
    assert_eq!(cli_json["detections"], library_json["detections"]);
    assert_eq!(cli_json["parameters"], library_json["parameters"]);

    let csv = fs::read_to_string(&first_csv).expect("read CSV");
    let lines: Vec<&str> = csv.lines().collect();
    assert_eq!(lines.len(), first_doc.detections.len() + 1);
    assert!(lines[0].starts_with("id,seed_source,x,y,angle_deg,scale,h00"));
    assert!(lines[0].ends_with("inliers,correspondences"));

    assert!(overlay.exists(), "overlay not written");
    let overlay_image = pm_core::load_image(&overlay).expect("decode overlay");
    assert_eq!(
        (
            overlay_image.gray.width as u32,
            overlay_image.gray.height as u32
        ),
        (256, 224)
    );
}

#[test]
fn params_override_merges_and_rejects_unknown_keys() {
    let dir = TempDir::new("params");
    let model = teach(&dir);
    let scene = render_scene(
        &source_image(),
        &Similarity::from_pose(140.0, 120.0, 0.6, 1.2),
        256,
        224,
    );
    let image = dir.join("scene.png");
    save_png(&scene, &image);

    let params = dir.join("params.json");
    fs::write(&params, r#"{"score_threshold": 0.99, "max_detections": 1}"#).unwrap();
    let output_path = dir.join("output.json");
    let output = Command::new(env!("CARGO_BIN_EXE_pm-cli"))
        .arg("run")
        .arg("--model")
        .arg(&model)
        .arg("--image")
        .arg(&image)
        .arg("--output")
        .arg(&output_path)
        .arg("--params")
        .arg(&params)
        .output()
        .expect("run pm-cli");
    assert_success(&output, "run with params");
    let document = load_results(&output_path);
    assert_eq!(document.parameters.score_threshold, 0.99);
    assert_eq!(document.parameters.max_detections, 1);
    assert_eq!(document.parameters.rotation_bins, 60, "defaults preserved");
    assert_eq!(
        document.parameters.orientation_gate, 0.08_f32,
        "fast gate by default"
    );

    let accuracy_output = dir.join("high-accuracy.json");
    let output = Command::new(env!("CARGO_BIN_EXE_pm-cli"))
        .arg("run")
        .arg("--model")
        .arg(&model)
        .arg("--image")
        .arg(&image)
        .arg("--output")
        .arg(&accuracy_output)
        .arg("--high-accuracy")
        .output()
        .expect("run pm-cli");
    assert_success(&output, "run with high accuracy");
    let document = load_results(&accuracy_output);
    assert_eq!(
        document.parameters.orientation_gate, 0.2_f32,
        "--high-accuracy selects the wide gate"
    );

    let bad = dir.join("bad.json");
    fs::write(&bad, r#"{"scroe_threshold": 0.5}"#).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_pm-cli"))
        .arg("run")
        .arg("--model")
        .arg(&model)
        .arg("--image")
        .arg(&image)
        .arg("--output")
        .arg(dir.join("bad-output.json"))
        .arg("--params")
        .arg(&bad)
        .output()
        .expect("run pm-cli");
    assert!(!output.status.success(), "unknown key accepted");
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("unknown parameter"),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn fail_on_none_reports_an_empty_result() {
    let dir = TempDir::new("fail-on-none");
    let model = teach(&dir);
    let blank = GrayF32::filled(256, 224, BACKGROUND);
    let image = dir.join("blank.png");
    save_png(&blank, &image);
    let output_path = dir.join("blank.json");

    let output = Command::new(env!("CARGO_BIN_EXE_pm-cli"))
        .arg("run")
        .arg("--model")
        .arg(&model)
        .arg("--image")
        .arg(&image)
        .arg("--output")
        .arg(&output_path)
        .arg("--fail-on-none")
        .output()
        .expect("run pm-cli");
    assert!(!output.status.success(), "empty result exited zero");
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("no detections"),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let document = load_results(&output_path);
    assert!(document.detections.is_empty());
}

#[test]
fn batch_mirrors_directories_and_writes_a_combined_csv() {
    let dir = TempDir::new("batch");
    let model = teach(&dir);
    let source = source_image();
    let first_scene = render_scene(
        &source,
        &Similarity::from_pose(140.0, 120.0, 0.6, 1.2),
        256,
        224,
    );
    let second_scene = render_scene(
        &source,
        &Similarity::from_pose(180.0, 130.0, -0.5, 0.9),
        256,
        224,
    );
    fs::create_dir_all(dir.join("images/sub")).unwrap();
    save_png(&first_scene, &dir.join("images/a.png"));
    save_png(&second_scene, &dir.join("images/sub/b.png"));
    fs::write(dir.join("images/notes.txt"), "not an image").unwrap();

    let output_dir = dir.join("results");
    let overlay_dir = dir.join("overlays");
    let csv = dir.join("batch.csv");
    let output = Command::new(env!("CARGO_BIN_EXE_pm-cli"))
        .arg("batch")
        .arg("--model")
        .arg(&model)
        .arg("--input")
        .arg(dir.join("images"))
        .arg("--output-dir")
        .arg(&output_dir)
        .arg("--recursive")
        .arg("--csv")
        .arg(&csv)
        .arg("--overlay-dir")
        .arg(&overlay_dir)
        .output()
        .expect("run pm-cli");
    assert_success(&output, "batch");

    let first = load_results(&output_dir.join("a.json"));
    let second = load_results(&output_dir.join("sub/b.json"));
    assert!(!first.detections.is_empty(), "first batch image missed");
    assert!(!second.detections.is_empty(), "second batch image missed");
    assert!(output_dir.join("sub/b.json").exists());
    assert!(
        !output_dir.join("notes.json").exists(),
        "non-image processed"
    );
    assert!(overlay_dir.join("a.overlay.png").exists());
    assert!(overlay_dir.join("sub/b.overlay.png").exists());

    let rows = fs::read_to_string(&csv).expect("read batch CSV");
    let lines: Vec<&str> = rows.lines().collect();
    assert_eq!(
        lines.len(),
        first.detections.len() + second.detections.len() + 1
    );
    assert!(lines[0].starts_with("image,id,seed_source,"));
    assert!(lines.iter().any(|line| line.starts_with("a.png,")));
    assert!(lines.iter().any(|line| line.starts_with("sub/b.png,")));

    let shallow_dir = dir.join("shallow");
    let output = Command::new(env!("CARGO_BIN_EXE_pm-cli"))
        .arg("batch")
        .arg("--model")
        .arg(&model)
        .arg("--input")
        .arg(dir.join("images"))
        .arg("--output-dir")
        .arg(&shallow_dir)
        .output()
        .expect("run pm-cli");
    assert_success(&output, "non-recursive batch");
    assert!(shallow_dir.join("a.json").exists());
    assert!(!shallow_dir.join("sub/b.json").exists());
}

#[test]
fn inspect_and_synthetic_benchmark_emit_machine_readable_reports() {
    let dir = TempDir::new("inspect-bench");
    let model = teach(&dir);

    let output = Command::new(env!("CARGO_BIN_EXE_pm-cli"))
        .arg("model")
        .arg("inspect")
        .arg("--model")
        .arg(&model)
        .arg("--json")
        .output()
        .expect("run pm-cli");
    assert_success(&output, "inspect");
    let summary: serde_json::Value = serde_json::from_slice(&output.stdout).expect("inspect JSON");
    assert_eq!(summary["name"], "cli-test");
    assert_eq!(summary["format"], "pattern-match-model");
    assert!(summary["features"]["edges"].as_u64().unwrap() > 0);

    let report_path = dir.join("report.json");
    let output = Command::new(env!("CARGO_BIN_EXE_pm-cli"))
        .arg("bench")
        .arg("synthetic")
        .arg("--model")
        .arg(&model)
        .arg("--cases")
        .arg("1")
        .arg("--seed")
        .arg("7")
        .arg("--report")
        .arg(&report_path)
        .output()
        .expect("run pm-cli");
    assert_success(&output, "bench synthetic");
    let report: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&report_path).expect("read report"))
            .expect("report JSON");
    assert_eq!(report["format"], "pattern-match-benchmark");
    assert_eq!(report["version"], 1);
    assert_eq!(report["seed"], 7);
    assert_eq!(report["cases"], 1);
    let scenarios = report["scenarios"].as_array().expect("scenarios");
    assert_eq!(scenarios.len(), 10);
    assert!(
        scenarios
            .iter()
            .all(|scenario| scenario["recall"].is_number()),
        "scenario recall missing"
    );
}
