//! Still-image run mode (README §11.3).
//!
//! Run state is deliberately separate from teach state: a loaded model and its
//! teach image stay resident while a test image is opened, detected on a worker
//! thread, inspected as data, and rendered as an overlay.  The worker uses the
//! same [`pm_core::detect_with_progress`] entry point and result/overlay
//! constructors as `pm-cli run`, so exported values match the CLI for the same
//! image, parameters, and seed.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use eframe::egui::{self, Color32, FontId, Pos2, Rect, Sense, Stroke};
use pm_core::{
    DetectStage, DetectionOverlayOptions, Homography, HypothesisSources, LoadedImage, Model,
    ProgressSink, ResultsFile, RgbaImage, RotatedRect, SearchDefaults, SearchParams, SeedSource,
    detect_with_progress, detections_to_csv, load_image, model_roi_in_model_coords,
    render_detection_overlay,
};

use crate::{
    RoiDrag, draw_rect_editor, edit_rotated_rect, image_to_screen, screen_to_image,
    texture_from_rgba,
};

/// Tool active in the run canvas.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RunTool {
    /// Select detections and edit the search region.
    Select,
    /// Draw or edit the optional rotated search region.
    Region,
}

impl RunTool {
    fn hint(self) -> &'static str {
        match self {
            Self::Select => "Select: click a detection to highlight it.",
            Self::Region => {
                "Search region: drag to create, drag inside to move, corners resize, handle rotates."
            }
        }
    }
}

/// A completed run, ready for display and export.
#[derive(Clone, Debug)]
pub(crate) struct RunOutput {
    /// Versioned result document, identical in shape to the CLI output.
    pub document: ResultsFile,
    /// Composite of the display image and all detection geometry, drawn with
    /// [`render_detection_overlay`] exactly like `pm-cli run --overlay`.
    pub overlay: RgbaImage,
}

/// Messages sent from the detection worker.
enum RunMessage {
    Progress { stage: DetectStage, fraction: f32 },
    Finished(Box<Result<RunOutput, String>>),
}

/// Handle to the running detection worker.
struct RunJob {
    receiver: Receiver<RunMessage>,
    cancel: Arc<AtomicBool>,
    started: Instant,
}

/// [`ProgressSink`] forwarding worker progress over a channel.
struct JobProgress {
    sender: Mutex<mpsc::Sender<RunMessage>>,
    cancel: Arc<AtomicBool>,
}

impl ProgressSink for JobProgress {
    fn progress(&self, stage: DetectStage, fraction: f32) {
        if let Ok(sender) = self.sender.lock() {
            let _ = sender.send(RunMessage::Progress { stage, fraction });
        }
    }

    fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }
}

/// Separate still-image run state.
pub(crate) struct RunState {
    image: Option<LoadedImage>,
    image_texture: Option<egui::TextureHandle>,
    image_path: Option<PathBuf>,
    /// Search parameters; initialized from the model defaults.
    pub params: SearchParams,
    defaults: SearchDefaults,
    /// Optional rotated search region in test-image coordinates.
    pub search_region: Option<RotatedRect>,
    /// Active canvas tool.
    pub tool: RunTool,
    drag: Option<RoiDrag>,
    /// Completed run, if any.
    result: Option<RunOutput>,
    detections_texture: Option<egui::TextureHandle>,
    /// Index into `result.document.detections`.
    pub selected: Option<usize>,
    job: Option<RunJob>,
    /// Current reported stage while running.
    pub stage: Option<DetectStage>,
    /// Whole-run progress in `[0, 1]`.
    pub fraction: f32,
    /// Whether the user requested cancellation.
    pub cancelled: bool,
    /// Whether the detection composite is drawn on the canvas.
    pub show_detections: bool,
    /// Wall time of the last completed run.
    pub last_elapsed_ms: f64,
}

impl Default for RunState {
    fn default() -> Self {
        Self {
            image: None,
            image_texture: None,
            image_path: None,
            params: SearchParams::default(),
            defaults: SearchDefaults::default(),
            search_region: None,
            tool: RunTool::Select,
            drag: None,
            result: None,
            detections_texture: None,
            selected: None,
            job: None,
            stage: None,
            fraction: 0.0,
            cancelled: false,
            show_detections: true,
            last_elapsed_ms: 0.0,
        }
    }
}

impl RunState {
    /// Whether a test image is loaded.
    pub fn has_image(&self) -> bool {
        self.image.is_some()
    }

    /// Whether a completed result is available.
    pub fn has_result(&self) -> bool {
        self.result.is_some()
    }

    /// Whether a worker is running.
    pub fn is_running(&self) -> bool {
        self.job.is_some()
    }

    /// Number of detections in the last result.
    pub fn detection_count(&self) -> usize {
        self.result
            .as_ref()
            .map_or(0, |result| result.document.detections.len())
    }

    /// Last result document, if any.
    pub fn document(&self) -> Option<&ResultsFile> {
        self.result.as_ref().map(|result| &result.document)
    }

    /// Cancel the worker and forget its in-flight output.
    fn abort(&mut self) {
        if let Some(job) = &self.job {
            job.cancel.store(true, Ordering::Relaxed);
        }
        self.job = None;
        self.stage = None;
        self.fraction = 0.0;
        self.cancelled = false;
    }

    fn clear_result(&mut self) {
        self.result = None;
        self.detections_texture = None;
        self.selected = None;
        self.last_elapsed_ms = 0.0;
    }

    /// Reset when no model is loaded.
    pub fn clear_model(&mut self) {
        self.abort();
        self.clear_result();
    }

    /// Initialize parameters from a newly opened or saved model.
    pub fn configure_for_model(&mut self, defaults: SearchDefaults) {
        self.abort();
        self.clear_result();
        self.defaults = defaults;
        self.params = SearchParams::from(defaults);
        self.search_region = None;
    }

    /// Replace the test image.
    pub fn set_image(&mut self, ctx: &egui::Context, image: LoadedImage, path: Option<PathBuf>) {
        self.abort();
        self.clear_result();
        self.image_texture = Some(texture_from_rgba(ctx, "test-image", &image.display));
        self.image = Some(image);
        self.image_path = path;
        self.search_region = None;
        self.drag = None;
    }

    /// Load a test image from disk.
    pub fn load_image(&mut self, ctx: &egui::Context, path: PathBuf) -> Result<(), String> {
        match load_image(&path) {
            Ok(image) => {
                self.set_image(ctx, image, Some(path));
                Ok(())
            }
            Err(error) => Err(format!("Failed to load {}: {error}", path.display())),
        }
    }

    /// Ask the worker to stop.
    pub fn cancel(&mut self) {
        if let Some(job) = &self.job {
            job.cancel.store(true, Ordering::Relaxed);
            self.cancelled = true;
        }
    }

    /// Start detection on a worker thread.
    pub fn start(&mut self, ctx: &egui::Context, model: &Model) -> Result<(), String> {
        if self.job.is_some() {
            return Err("A detection run is already in progress.".to_owned());
        }
        let Some(image) = self.image.clone() else {
            return Err("Open a test image before running.".to_owned());
        };
        let mut params = self.params;
        params.search_region = self.search_region;
        validate_params(&params)?;

        let model = model.clone();
        let display = image.display.clone();
        let (width, height) = (image.gray.width, image.gray.height);
        let (sender, receiver) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let progress = JobProgress {
            sender: Mutex::new(sender.clone()),
            cancel: Arc::clone(&cancel),
        };
        let repaint = ctx.clone();
        std::thread::Builder::new()
            .name("pm-detect".to_owned())
            .spawn(move || {
                let outcome =
                    detect_with_progress(&model, &image.gray, &params, &progress).map(|result| {
                        let document =
                            ResultsFile::new(model.name.clone(), width, height, &params, &result);
                        let roi_model = model_roi_in_model_coords(&model);
                        let overlay = render_detection_overlay(
                            &display,
                            &roi_model,
                            &model.features.edges.points,
                            &result.detections,
                            &DetectionOverlayOptions::default(),
                        );
                        RunOutput { document, overlay }
                    });
                let _ = sender.send(RunMessage::Finished(Box::new(
                    outcome.map_err(|error| error.to_string()),
                )));
                repaint.request_repaint();
            })
            .map_err(|error| format!("Failed to start detection worker: {error}"))?;

        self.clear_result();
        self.stage = Some(DetectStage::Preprocess);
        self.fraction = 0.0;
        self.cancelled = false;
        self.job = Some(RunJob {
            receiver,
            cancel,
            started: Instant::now(),
        });
        Ok(())
    }

    /// Drain worker messages.  Returns a status message when the run finishes.
    pub fn poll(&mut self, ctx: &egui::Context) -> Option<String> {
        self.job.as_ref()?;
        loop {
            let message = self
                .job
                .as_ref()
                .map(|job| job.receiver.try_recv())
                .unwrap_or(Err(TryRecvError::Empty));
            match message {
                Ok(RunMessage::Progress { stage, fraction }) => {
                    self.stage = Some(stage);
                    // Parallel stages finish out of order; keep the bar monotone.
                    self.fraction = self.fraction.max(fraction.clamp(0.0, 1.0));
                }
                Ok(RunMessage::Finished(outcome)) => {
                    let was_cancelled = self.cancelled;
                    self.job = None;
                    self.stage = None;
                    self.fraction = 0.0;
                    self.cancelled = false;
                    return Some(match *outcome {
                        Ok(output) => {
                            self.detections_texture =
                                Some(texture_from_rgba(ctx, "detections", &output.overlay));
                            self.last_elapsed_ms = output.document.timings_ms.total;
                            let count = output.document.detections.len();
                            self.selected = (count > 0).then_some(0);
                            self.result = Some(output);
                            format!(
                                "Run complete: {count} detection(s) in {:.1} ms.",
                                self.last_elapsed_ms
                            )
                        }
                        Err(_) if was_cancelled => "Run cancelled.".to_owned(),
                        Err(error) => format!("Detection failed: {error}"),
                    });
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    let was_cancelled = self.cancelled;
                    self.job = None;
                    self.stage = None;
                    self.fraction = 0.0;
                    self.cancelled = false;
                    return Some(if was_cancelled {
                        "Run cancelled.".to_owned()
                    } else {
                        "Detection worker stopped unexpectedly.".to_owned()
                    });
                }
            }
        }
        if self.job.is_some() {
            ctx.request_repaint_after(Duration::from_millis(50));
        }
        None
    }

    /// Run-mode left panel.  Returns `true` when the open-image dialog should
    /// be shown.
    pub fn left_panel(&mut self, ui: &mut egui::Ui, has_model: bool) -> bool {
        let mut open_requested = false;
        ui.heading("Test image");
        if let Some(path) = &self.image_path {
            ui.label(path.file_name().map_or_else(
                || path.display().to_string(),
                |name| name.to_string_lossy().into_owned(),
            ));
        } else if self.image.is_some() {
            ui.label("In-memory test image");
        } else {
            ui.label("No test image loaded.");
        }
        if let Some(image) = &self.image {
            ui.label(format!("{} × {} px", image.gray.width, image.gray.height));
        }
        if ui.button("Open test image…").clicked() {
            open_requested = true;
        }
        ui.separator();
        ui.heading("Tools");
        ui.horizontal_wrapped(|ui| {
            ui.selectable_value(&mut self.tool, RunTool::Select, "Select");
            ui.selectable_value(&mut self.tool, RunTool::Region, "Search region");
        });
        ui.label(self.tool.hint());
        ui.separator();
        self.region_fields(ui);
        ui.separator();
        ui.heading("Run");
        if let Some(stage) = self.stage {
            ui.label(format!("Stage: {}", stage.as_str()));
            ui.add(egui::ProgressBar::new(self.fraction).show_percentage());
            if let Some(job) = &self.job {
                ui.label(format!(
                    "Elapsed {:.0} ms",
                    job.started.elapsed().as_secs_f64() * 1000.0
                ));
            }
        } else if !has_model {
            ui.label("Open a model to run detection.");
        } else if self.has_result() {
            ui.label(format!(
                "{} detection(s); last run {:.1} ms.",
                self.detection_count(),
                self.last_elapsed_ms
            ));
        } else {
            ui.label("Press Run in the toolbar.");
        }
        open_requested
    }

    fn region_fields(&mut self, ui: &mut egui::Ui) {
        ui.label("Search region");
        let mut region = self.search_region;
        let mut changed = false;
        if let Some(rect) = &mut region {
            egui::Grid::new("search_region_grid")
                .num_columns(2)
                .spacing([8.0, 4.0])
                .show(ui, |ui| {
                    ui.label("Center x");
                    changed |= ui
                        .add(egui::DragValue::new(&mut rect.cx).speed(0.5))
                        .changed();
                    ui.end_row();
                    ui.label("Center y");
                    changed |= ui
                        .add(egui::DragValue::new(&mut rect.cy).speed(0.5))
                        .changed();
                    ui.end_row();
                    ui.label("Width");
                    changed |= ui
                        .add(
                            egui::DragValue::new(&mut rect.width)
                                .speed(0.5)
                                .range(crate::MIN_ROI_SIZE..=f32::MAX),
                        )
                        .changed();
                    ui.end_row();
                    ui.label("Height");
                    changed |= ui
                        .add(
                            egui::DragValue::new(&mut rect.height)
                                .speed(0.5)
                                .range(crate::MIN_ROI_SIZE..=f32::MAX),
                        )
                        .changed();
                    ui.end_row();
                    ui.label("Angle");
                    let mut degrees = rect.angle.to_degrees();
                    if ui
                        .add(egui::DragValue::new(&mut degrees).speed(1.0).suffix("°"))
                        .changed()
                    {
                        rect.angle = crate::wrap_angle(degrees.to_radians());
                        changed = true;
                    }
                    ui.end_row();
                });
            if ui.button("Clear search region").clicked() {
                region = None;
                changed = true;
            }
        } else {
            ui.label("Whole image (no region).");
        }
        if changed {
            self.search_region = region;
            self.clear_result();
        }
    }

    /// Run-mode right panel: search parameters, result table, and details.
    pub fn right_panel(&mut self, ui: &mut egui::Ui, has_model: bool, running: bool) {
        ui.heading("Search parameters");
        if !has_model {
            ui.colored_label(ui.visuals().warn_fg_color, "No model is loaded.");
        }
        ui.add_enabled_ui(!running, |ui| {
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(has_model, egui::Button::new("Reset to model defaults"))
                    .clicked()
                {
                    self.params = SearchParams::from(self.defaults);
                    self.clear_result();
                }
            });
            Self::search_parameter_ui(ui, &mut self.params);
        });
        ui.separator();
        self.results_ui(ui);
    }

    /// Search-parameter controls shared by still-image and live run modes.
    pub(crate) fn search_parameter_ui(ui: &mut egui::Ui, params: &mut SearchParams) {
        ui.label("Rotation");
        ui.horizontal(|ui| {
            ui.add(
                egui::DragValue::new(&mut params.rotation_min_deg)
                    .speed(1.0)
                    .range(-360.0..=360.0)
                    .suffix("°"),
            );
            ui.label("to");
            ui.add(
                egui::DragValue::new(&mut params.rotation_max_deg)
                    .speed(1.0)
                    .range(-360.0..=360.0)
                    .suffix("°"),
            );
        });
        ui.label("Scale");
        ui.horizontal(|ui| {
            ui.add(
                egui::DragValue::new(&mut params.scale_min)
                    .speed(0.01)
                    .range(0.05..=4.0),
            );
            ui.label("to");
            ui.add(
                egui::DragValue::new(&mut params.scale_max)
                    .speed(0.01)
                    .range(0.05..=4.0),
            );
        });
        let mut high_accuracy =
            params.orientation_gate >= SearchParams::HIGH_ACCURACY_ORIENTATION_GATE;
        if ui
            .checkbox(&mut high_accuracy, "High accuracy")
            .on_hover_text(
                "Wider Hough orientation gate (0.20 rad): slower, with slightly better \
                 recall on low-contrast and cluttered scenes",
            )
            .changed()
        {
            params.orientation_gate = if high_accuracy {
                SearchParams::HIGH_ACCURACY_ORIENTATION_GATE
            } else {
                SearchParams::FAST_ORIENTATION_GATE
            };
        }
        ui.add(egui::Slider::new(&mut params.score_threshold, 0.0..=1.0).text("Score threshold"));
        if ui
            .add(egui::Slider::new(&mut params.coverage_min, 0.0..=1.0).text("Coverage minimum"))
            .changed()
        {
            params.score.coverage_min = params.coverage_min;
        }
        ui.add(egui::DragValue::new(&mut params.max_detections).range(1..=100));
        if ui
            .checkbox(&mut params.polarity_enabled, "Require polarity match")
            .changed()
        {
            params.score.polarity_enabled = params.polarity_enabled;
        }

        let mut sources = params.hypothesis_sources;
        egui::ComboBox::from_label("Hypothesis sources")
            .selected_text(sources_label(sources))
            .show_ui(ui, |ui| {
                ui.selectable_value(
                    &mut sources,
                    HypothesisSources::Both,
                    "Hough + descriptors/segments",
                );
                ui.selectable_value(&mut sources, HypothesisSources::Hough, "Hough only");
                ui.selectable_value(
                    &mut sources,
                    HypothesisSources::Descriptors,
                    "Descriptors/segments only",
                );
            });
        params.hypothesis_sources = sources;

        let mut auto_level = params.coarse_level.is_none();
        if ui
            .checkbox(&mut auto_level, "Automatic search level")
            .changed()
        {
            params.coarse_level = if auto_level { None } else { Some(1) };
        }
        if let Some(level) = &mut params.coarse_level {
            ui.horizontal(|ui| {
                ui.label("Coarse level");
                ui.add(egui::DragValue::new(level).range(0..=3));
            });
        }

        ui.horizontal(|ui| {
            ui.label("Model corner cap");
            ui.add(egui::DragValue::new(&mut params.model_corner_cap).range(0..=2000));
        });
        ui.horizontal(|ui| {
            ui.label("Test corner cap");
            ui.add(egui::DragValue::new(&mut params.test_corner_cap).range(0..=4000));
        });

        ui.add(
            egui::Slider::new(&mut params.homography.ransac.confidence, 0.5..=0.999)
                .text("RANSAC confidence"),
        );
        ui.horizontal(|ui| {
            ui.label("RANSAC seed");
            ui.add(egui::DragValue::new(&mut params.homography.ransac.seed));
        });
    }

    fn results_ui(&mut self, ui: &mut egui::Ui) {
        let Some(result) = &self.result else {
            ui.heading("Results");
            ui.label("No run yet.");
            return;
        };
        let document = &result.document;
        let count = document.detections.len();
        ui.heading(format!(
            "Results ({count} detection{})",
            if count == 1 { "" } else { "s" }
        ));
        ui.label(format!(
            "Total {:.1} ms · preprocess {:.1} · features {:.1} · search {:.1} · refine {:.1}",
            document.timings_ms.total,
            document.timings_ms.preprocess,
            document.timings_ms.features,
            document.timings_ms.search,
            document.timings_ms.refine
        ));
        if count == 0 {
            ui.label("No detections above the threshold.");
            return;
        }

        let mut selected = self.selected;
        egui::ScrollArea::horizontal().show(ui, |ui| {
            egui::Grid::new("run_results")
                .num_columns(9)
                .striped(true)
                .spacing([8.0, 3.0])
                .show(ui, |ui| {
                    for header in [
                        "#", "x", "y", "angle°", "scale", "score", "conf", "seed", "H",
                    ] {
                        ui.strong(header);
                    }
                    ui.end_row();
                    for (index, record) in document.detections.iter().enumerate() {
                        if ui
                            .selectable_label(selected == Some(index), record.id.to_string())
                            .clicked()
                        {
                            selected = Some(index);
                        }
                        ui.label(format!("{:.1}", record.pose.x));
                        ui.label(format!("{:.1}", record.pose.y));
                        ui.label(format!("{:.2}", record.pose.angle_deg));
                        ui.label(format!("{:.3}", record.pose.scale));
                        ui.label(format!("{:.3}", record.score));
                        ui.label(format!("{:.3}", record.confidence));
                        ui.label(seed_label(record.diagnostics.seed_source));
                        ui.label(if record.homography_estimated {
                            "yes"
                        } else {
                            "sim"
                        });
                        ui.end_row();
                    }
                });
        });
        self.selected = selected;

        if let Some(index) = self.selected
            && let Some(record) = document.detections.get(index)
        {
            ui.separator();
            ui.heading(format!("Detection {}", record.id));
            egui::Grid::new("detection_details")
                .num_columns(2)
                .spacing([8.0, 3.0])
                .show(ui, |ui| {
                    ui.label("Position");
                    ui.monospace(format!("({:.3}, {:.3}) px", record.pose.x, record.pose.y));
                    ui.end_row();
                    ui.label("Angle");
                    ui.monospace(format!("{:.3}°", record.pose.angle_deg));
                    ui.end_row();
                    ui.label("Scale");
                    ui.monospace(format!("{:.4}", record.pose.scale));
                    ui.end_row();
                    ui.label("Score / confidence");
                    ui.monospace(format!("{:.3} / {:.3}", record.score, record.confidence));
                    ui.end_row();
                    ui.label("Seed source");
                    ui.monospace(seed_label(record.diagnostics.seed_source));
                    ui.end_row();
                    ui.label("Coverage");
                    ui.monospace(format!("{:.3}", record.diagnostics.coverage));
                    ui.end_row();
                    ui.label("Matched fraction");
                    ui.monospace(format!("{:.3}", record.diagnostics.matched_fraction));
                    ui.end_row();
                    ui.label("Mean edge distance");
                    ui.monospace(format!("{:.3} px", record.diagnostics.mean_distance_px));
                    ui.end_row();
                    ui.label("Orientation agreement");
                    ui.monospace(format!(
                        "{:.3}",
                        record.diagnostics.mean_orientation_agreement
                    ));
                    ui.end_row();
                    ui.label("Correspondences / inliers");
                    ui.monospace(format!(
                        "{} / {}",
                        record.diagnostics.correspondences, record.diagnostics.inliers
                    ));
                    ui.end_row();
                    ui.label("RANSAC iterations");
                    ui.monospace(record.diagnostics.ransac_iterations.to_string());
                    ui.end_row();
                    ui.label("Homography");
                    ui.monospace(if record.homography_estimated {
                        "estimated"
                    } else {
                        "similarity fallback"
                    });
                    ui.end_row();
                    ui.label("Condition");
                    ui.monospace(format!("{:?}", record.diagnostics.homography_condition));
                    ui.end_row();
                    ui.label("Ambiguity");
                    ui.monospace(format!("{:.3}", record.diagnostics.ambiguity));
                    ui.end_row();
                    ui.label("Fallback reason");
                    ui.monospace(
                        record
                            .diagnostics
                            .fallback_reason
                            .clone()
                            .unwrap_or_else(|| "none".to_owned()),
                    );
                    ui.end_row();
                });
            ui.label("Homography (model → image)");
            for row in &record.homography {
                ui.monospace(format!(
                    "{:>12.6} {:>12.6} {:>12.6}",
                    row[0], row[1], row[2]
                ));
            }
        }
    }

    /// Draw the run canvas.  Returns the cursor position in image coordinates.
    pub fn draw_canvas(&mut self, ui: &mut egui::Ui, model: Option<&Model>) -> Option<(f32, f32)> {
        let Some(image) = &self.image else {
            ui.centered_and_justified(|ui| {
                ui.label("Open a test image to run detection.");
            });
            return None;
        };
        let (image_width, image_height) = (image.display.width(), image.display.height());
        let texture = if self.show_detections && self.detections_texture.is_some() {
            self.detections_texture.as_ref()
        } else {
            self.image_texture.as_ref()
        }
        .map(|texture| texture.id());

        let available = ui.available_rect_before_wrap();
        let image_size = egui::vec2(image_width as f32, image_height as f32);
        let scale = (available.width() / image_size.x)
            .min(available.height() / image_size.y)
            .max(1e-3);
        let rect = Rect::from_center_size(available.center(), image_size * scale);
        let response = ui.interact(
            rect,
            ui.scope_id().with("run_canvas"),
            Sense::click_and_drag(),
        );
        let pointer = response
            .interact_pointer_pos()
            .map(|pos| screen_to_image(pos, rect, scale));
        let hover = response
            .hover_pos()
            .map(|pos| screen_to_image(pos, rect, scale));

        match self.tool {
            RunTool::Region => {
                if edit_rotated_rect(
                    rect,
                    scale,
                    &response,
                    pointer,
                    &mut self.search_region,
                    &mut self.drag,
                ) {
                    self.clear_result();
                }
            }
            RunTool::Select => {
                if response.clicked()
                    && let (Some(point), Some(model)) = (pointer, model)
                {
                    self.select_at(point, model, scale);
                }
            }
        }

        let painter = ui.painter();
        let uv = Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
        if let Some(texture) = texture {
            painter.image(texture, rect, uv, Color32::WHITE);
        }
        if let Some(region) = self.search_region {
            draw_rect_editor(
                painter,
                rect,
                scale,
                &region,
                Color32::from_rgb(90, 170, 255),
            );
        }
        if self.show_detections
            && let (Some(result), Some(model), Some(index)) = (&self.result, model, self.selected)
            && let Some(record) = result.document.detections.get(index)
        {
            draw_selected_detection(painter, rect, scale, model, index, record);
        }
        painter.text(
            rect.left_top() + egui::vec2(8.0, 8.0),
            egui::Align2::LEFT_TOP,
            self.tool.hint(),
            FontId::proportional(12.0),
            Color32::from_rgba_unmultiplied(20, 20, 20, 190),
        );

        hover.map(|point| (point[0], point[1]))
    }

    fn select_at(&mut self, point: [f32; 2], model: &Model, scale: f32) {
        let Some(result) = &self.result else {
            return;
        };
        let roi_model = model_roi_in_model_coords(model);
        let mut best: Option<(f32, usize)> = None;
        for (index, record) in result.document.detections.iter().enumerate() {
            let homography = Homography::from_array(record.homography);
            let polygon: Vec<[f32; 2]> = roi_model
                .iter()
                .filter_map(|&corner| homography.try_act_array(corner))
                .collect();
            if polygon.len() == roi_model.len() && point_in_polygon(point, &polygon) {
                self.selected = Some(index);
                return;
            }
            if let Some(zero) = homography.try_act_array([0.0, 0.0]) {
                let distance = (zero[0] - point[0]).hypot(zero[1] - point[1]);
                if distance <= 16.0 / scale.max(1e-3)
                    && best.as_ref().is_none_or(|(best, _)| distance < *best)
                {
                    best = Some((distance, index));
                }
            }
        }
        self.selected = best.map(|(_, index)| index);
    }

    /// Write the result JSON (README §12.3).
    pub fn export_json(&self, path: &Path) -> Result<(), String> {
        let result = self
            .result
            .as_ref()
            .ok_or_else(|| "No results to export.".to_owned())?;
        let json = result
            .document
            .to_json()
            .map_err(|error| error.to_string())?;
        write_text(path, &json)
    }

    /// Write the detection CSV.
    pub fn export_csv(&self, path: &Path) -> Result<(), String> {
        let result = self
            .result
            .as_ref()
            .ok_or_else(|| "No results to export.".to_owned())?;
        write_text(path, &detections_to_csv(&result.document.detections))
    }

    /// Write the annotated overlay PNG.
    pub fn save_overlay_png(&self, path: &Path) -> Result<(), String> {
        let result = self
            .result
            .as_ref()
            .ok_or_else(|| "No results to export.".to_owned())?;
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)
                .map_err(|error| format!("Create {}: {error}", parent.display()))?;
        }
        result
            .overlay
            .save(path)
            .map_err(|error| format!("Write overlay {}: {error}", path.display()))
    }
}

fn validate_params(params: &SearchParams) -> Result<(), String> {
    if !params.rotation_min_deg.is_finite()
        || !params.rotation_max_deg.is_finite()
        || params.rotation_min_deg > params.rotation_max_deg
    {
        return Err("Rotation range must be finite with minimum not above maximum.".to_owned());
    }
    if !params.scale_min.is_finite()
        || !params.scale_max.is_finite()
        || params.scale_min <= 0.0
        || params.scale_min > params.scale_max
    {
        return Err("Scale range must be positive with minimum not above maximum.".to_owned());
    }
    if let Some(region) = params.search_region
        && !region.is_valid()
    {
        return Err("Search region must have finite center/angle and positive size.".to_owned());
    }
    Ok(())
}

fn write_text(path: &Path, text: &str) -> Result<(), String> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("Create {}: {error}", parent.display()))?;
    }
    std::fs::write(path, text).map_err(|error| format!("Write {}: {error}", path.display()))
}

fn sources_label(sources: HypothesisSources) -> &'static str {
    match sources {
        HypothesisSources::Both => "Hough + descriptors",
        HypothesisSources::Hough => "Hough only",
        HypothesisSources::Descriptors => "Descriptors only",
    }
}

fn seed_label(source: SeedSource) -> &'static str {
    match source {
        SeedSource::Hough => "hough",
        SeedSource::CornerDescriptor => "corner",
        SeedSource::SegmentPair => "segment",
    }
}

fn point_in_polygon(point: [f32; 2], polygon: &[[f32; 2]]) -> bool {
    let mut inside = false;
    let mut previous = polygon.len() - 1;
    for current in 0..polygon.len() {
        let (xi, yi) = (polygon[current][0], polygon[current][1]);
        let (xj, yj) = (polygon[previous][0], polygon[previous][1]);
        if (yi > point[1]) != (yj > point[1])
            && point[0] < (xj - xi) * (point[1] - yi) / (yj - yi) + xi
        {
            inside = !inside;
        }
        previous = current;
    }
    inside
}

fn draw_selected_detection(
    painter: &egui::Painter,
    rect: Rect,
    scale: f32,
    model: &Model,
    index: usize,
    record: &pm_core::DetectionRecord,
) {
    let homography = Homography::from_array(record.homography);
    let roi_model = model_roi_in_model_coords(model);
    // Keep the detection's own outline hue and lighten it so the selection is
    // still visible over light image areas.
    let base = pm_core::DETECTION_COLORS[index % pm_core::DETECTION_COLORS.len()];
    let color = Color32::from_rgb(
        lighten_channel(base[0]),
        lighten_channel(base[1]),
        lighten_channel(base[2]),
    );
    let corners: Vec<Pos2> = roi_model
        .iter()
        .filter_map(|&corner| homography.try_act_array(corner))
        .map(|point| image_to_screen(point, rect, scale))
        .collect();
    if corners.len() == 4 {
        for index in 0..4 {
            painter.line_segment(
                [corners[index], corners[(index + 1) % 4]],
                Stroke::new(3.0, color),
            );
        }
        for corner in &corners {
            painter.circle_filled(*corner, 3.5, color);
        }
    }
    let axis = roi_axis_length(&roi_model);
    if let Some(origin) = homography.try_act_array([0.0, 0.0]) {
        let origin = image_to_screen(origin, rect, scale);
        for axis_point in [[axis, 0.0], [0.0, axis]] {
            if let Some(end) = homography.try_act_array(axis_point) {
                painter.line_segment(
                    [origin, image_to_screen(end, rect, scale)],
                    Stroke::new(3.0, color),
                );
            }
        }
        painter.circle_filled(origin, 4.0, color);
        painter.text(
            origin + egui::vec2(8.0, -8.0),
            egui::Align2::LEFT_BOTTOM,
            format!("#{} {:.3}", record.id, record.confidence),
            FontId::proportional(13.0),
            color,
        );
    }
}

/// Blend one 8-bit color channel a third of the way toward white.
fn lighten_channel(channel: u8) -> u8 {
    (f32::from(channel) + (255.0 - f32::from(channel)) / 3.0).round() as u8
}

fn roi_axis_length(roi_model: &[[f32; 2]; 4]) -> f32 {
    let width = (roi_model[1][0] - roi_model[0][0]).hypot(roi_model[1][1] - roi_model[0][1]);
    let height = (roi_model[3][0] - roi_model[0][0]).hypot(roi_model[3][1] - roi_model[0][1]);
    width.min(height) * 0.5
}
