//! Live webcam mode (README §5.4 and §11.4).
//!
//! Capture and detection each run on their own thread.  The capture thread
//! publishes only the newest frame; the detection thread always consumes the
//! latest frame and republishes its result, so slow detection drops frames
//! instead of queuing them.  The UI thread never blocks on either: it polls
//! shared state, uploads textures, and draws.
//!
//! Live results use the same [`pm_core::detect_with_progress`] entry point and
//! overlay renderer as still-image run mode, so a frozen webcam frame taught
//! in the UI behaves exactly like the same pixels opened from disk.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use eframe::egui::{self, Color32, FontId, Rect, Sense};
use pm_core::{
    CaptureConfig, CaptureDevice, DetectStage, DetectionOverlayOptions, Frame, FrameSource,
    LoadedImage, Model, ProgressSink, RgbaImage, SearchDefaults, SearchParams, camera_available,
    detect_with_progress, list_cameras, model_roi_in_model_coords, open_webcam,
    render_detection_overlay,
};

use crate::run::RunState;
use crate::screen_to_image;

/// A completed live detection, ready for display.
struct LiveOutput {
    document: pm_core::ResultsFile,
    overlay: RgbaImage,
    /// Frame the result belongs to (kept so a freeze captures the displayed image).
    frame: Arc<Frame>,
    completed_at: Instant,
}

impl LiveOutput {
    fn latency_ms(&self) -> f32 {
        self.completed_at
            .saturating_duration_since(self.frame.captured_at)
            .as_secs_f32()
            * 1000.0
    }
}

/// State shared by the UI and the two worker threads.
struct Shared {
    latest_frame: Mutex<Option<Arc<Frame>>>,
    frame_ready: Condvar,
    latest_output: Mutex<Option<LiveOutput>>,
    /// Progress of the frame currently being detected.
    progress: Mutex<Option<(DetectStage, f32)>>,
    /// Parameters may be edited while live; the detector snapshots them per frame.
    params: Mutex<SearchParams>,
    description: Mutex<Option<String>>,
    capture_count: AtomicU64,
    output_count: AtomicU64,
    error: Mutex<Option<String>>,
    stop: AtomicBool,
}

impl Shared {
    fn new(params: SearchParams) -> Self {
        Self {
            latest_frame: Mutex::new(None),
            frame_ready: Condvar::new(),
            latest_output: Mutex::new(None),
            progress: Mutex::new(None),
            params: Mutex::new(params),
            description: Mutex::new(None),
            capture_count: AtomicU64::new(0),
            output_count: AtomicU64::new(0),
            error: Mutex::new(None),
            stop: AtomicBool::new(false),
        }
    }

    fn request_stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
        self.frame_ready.notify_all();
    }

    fn fail(&self, message: String) {
        let mut error = self.error.lock().unwrap_or_else(PoisonError::into_inner);
        if error.is_none() {
            *error = Some(message);
        }
        drop(error);
        self.request_stop();
    }
}

/// Forwards detection progress to the UI without blocking the worker.
struct LiveProgress {
    shared: Arc<Shared>,
}

impl ProgressSink for LiveProgress {
    fn progress(&self, stage: DetectStage, fraction: f32) {
        if let Ok(mut progress) = self.shared.progress.lock() {
            *progress = Some((stage, fraction));
        }
    }

    fn cancelled(&self) -> bool {
        self.shared.stop.load(Ordering::Relaxed)
    }
}

struct LiveSession {
    shared: Arc<Shared>,
    capture: Option<JoinHandle<()>>,
    detector: Option<JoinHandle<()>>,
}

/// A frozen display frame and the image it came from.
struct FrozenFrame {
    texture: egui::TextureHandle,
    frame: Arc<Frame>,
}

/// Smoothed frame-rate estimate.
#[derive(Default)]
struct Rate {
    count: u64,
    at: Option<Instant>,
    value: f32,
}

impl Rate {
    fn update(&mut self, count: u64, now: Instant) {
        let Some(at) = self.at else {
            self.count = count;
            self.at = Some(now);
            return;
        };
        let seconds = now.saturating_duration_since(at).as_secs_f32();
        if seconds < 0.25 {
            return;
        }
        let instant = count.saturating_sub(self.count) as f32 / seconds;
        self.value = if self.value <= 0.0 {
            instant
        } else {
            self.value * 0.6 + instant * 0.4
        };
        self.count = count;
        self.at = Some(now);
    }
}

/// Live camera state, independent of teach and still-image run state.
pub(crate) struct LiveState {
    devices: Vec<CaptureDevice>,
    device_choice: Option<usize>,
    devices_loaded: bool,
    config: CaptureConfig,
    session: Option<LiveSession>,
    params: SearchParams,
    defaults: SearchDefaults,
    description: Option<String>,
    preview_frame: Option<Arc<Frame>>,
    preview_sequence: u64,
    preview_texture: Option<egui::TextureHandle>,
    output: Option<LiveOutput>,
    detections_texture: Option<egui::TextureHandle>,
    frozen: Option<FrozenFrame>,
    pub show_detections: bool,
    capture_rate: Rate,
    output_rate: Rate,
    latency_ms: f32,
    stage: Option<(DetectStage, f32)>,
    error: Option<String>,
}

impl Drop for LiveState {
    fn drop(&mut self) {
        if let Some(session) = &self.session {
            session.shared.request_stop();
        }
    }
}

impl Default for LiveState {
    fn default() -> Self {
        Self {
            devices: Vec::new(),
            device_choice: None,
            devices_loaded: false,
            config: CaptureConfig::default(),
            session: None,
            params: SearchParams::default(),
            defaults: SearchDefaults::default(),
            description: None,
            preview_frame: None,
            preview_sequence: 0,
            preview_texture: None,
            output: None,
            detections_texture: None,
            frozen: None,
            show_detections: true,
            capture_rate: Rate::default(),
            output_rate: Rate::default(),
            latency_ms: 0.0,
            stage: None,
            error: None,
        }
    }
}

impl LiveState {
    /// Whether capture and detection threads are running.
    pub fn is_running(&self) -> bool {
        self.session.is_some()
    }

    /// Whether a frame is available to display or teach from.
    pub fn has_frame(&self) -> bool {
        self.frozen.is_some() || self.preview_frame.is_some()
    }

    /// Whether any frame has completed detection.
    pub fn has_output(&self) -> bool {
        self.output.is_some()
    }

    /// Whether the preview is frozen.
    pub fn is_frozen(&self) -> bool {
        self.frozen.is_some()
    }

    /// Smoothed captured-frame rate.
    pub fn capture_fps(&self) -> f32 {
        self.capture_rate.value
    }

    /// Smoothed processed-frame rate.
    pub fn processed_fps(&self) -> f32 {
        self.output_rate.value
    }

    /// Capture-to-result latency of the latest processed frame.
    pub fn latency_ms(&self) -> f32 {
        self.latency_ms
    }

    /// Number of detections in the latest processed frame.
    pub fn detection_count(&self) -> usize {
        self.output
            .as_ref()
            .map_or(0, |output| output.document.detections.len())
    }

    /// Last reported detection stage and its fraction.
    pub fn stage(&self) -> Option<(DetectStage, f32)> {
        self.stage
    }

    fn selected_device(&self) -> Option<&CaptureDevice> {
        self.device_choice.and_then(|index| self.devices.get(index))
    }

    pub fn can_start(&self) -> bool {
        camera_available()
            && !self.is_running()
            && self.selected_device().is_some()
            && self.config.is_valid()
    }

    /// Enumerate cameras once, on first use.
    fn ensure_devices(&mut self) {
        if self.devices_loaded {
            return;
        }
        self.devices_loaded = true;
        self.reload_devices();
    }

    fn reload_devices(&mut self) {
        match list_cameras() {
            Ok(devices) => {
                self.error = None;
                self.devices = devices;
                if self.devices.is_empty() {
                    self.device_choice = None;
                    self.error = Some("No cameras found.".to_owned());
                } else if self
                    .device_choice
                    .is_none_or(|index| index >= self.devices.len())
                {
                    self.device_choice = Some(0);
                }
            }
            Err(error) => {
                self.devices.clear();
                self.device_choice = None;
                self.error = Some(error.to_string());
            }
        }
    }

    /// Start capture (and detection when a model is loaded).
    pub fn start(&mut self, ctx: &egui::Context, model: Option<Model>) -> Result<(), String> {
        if self.session.is_some() {
            return Err("The camera is already running.".to_owned());
        }
        if !camera_available() {
            return Err("This build has no camera backend.".to_owned());
        }
        if !self.config.is_valid() {
            return Err("Camera width, height, and fps must be non-zero.".to_owned());
        }
        let device = self
            .selected_device()
            .cloned()
            .ok_or_else(|| "Select a camera first.".to_owned())?;
        let config = self.config;
        self.start_with(ctx, model, move || open_webcam(&device, config))
    }

    /// Start from a caller-provided source factory.  Used by tests to drive
    /// live mode without hardware; `open` runs on the capture thread.
    fn start_with<F>(
        &mut self,
        ctx: &egui::Context,
        model: Option<Model>,
        open: F,
    ) -> Result<(), String>
    where
        F: FnOnce() -> pm_core::Result<Box<dyn FrameSource>> + Send + 'static,
    {
        let shared = Arc::new(Shared::new(self.params));
        self.preview_frame = None;
        self.preview_sequence = 0;
        self.preview_texture = None;
        self.output = None;
        self.detections_texture = None;
        self.frozen = None;
        self.description = None;
        self.error = None;
        self.capture_rate = Rate::default();
        self.output_rate = Rate::default();
        self.latency_ms = 0.0;
        self.stage = None;

        let capture_shared = Arc::clone(&shared);
        let capture_ctx = ctx.clone();
        let capture = thread::Builder::new()
            .name("pm-capture".to_owned())
            .spawn(move || capture_loop(open, capture_shared, capture_ctx))
            .map_err(|error| format!("Failed to start capture thread: {error}"))?;

        let detector = match model {
            Some(model) => {
                let detect_shared = Arc::clone(&shared);
                let detect_ctx = ctx.clone();
                match thread::Builder::new()
                    .name("pm-live-detect".to_owned())
                    .spawn(move || detection_loop(model, detect_shared, detect_ctx))
                {
                    Ok(handle) => Some(handle),
                    Err(error) => {
                        shared.request_stop();
                        let _ = capture.join();
                        return Err(format!("Failed to start detection thread: {error}"));
                    }
                }
            }
            None => None,
        };

        self.session = Some(LiveSession {
            shared,
            capture: Some(capture),
            detector,
        });
        Ok(())
    }

    /// Signal both workers and wait for them to release the camera.
    pub fn stop(&mut self) {
        let Some(mut session) = self.session.take() else {
            return;
        };
        session.shared.request_stop();
        if let Some(handle) = session.capture.take() {
            join_briefly(handle, Duration::from_millis(500));
        }
        if let Some(handle) = session.detector.take() {
            join_briefly(handle, Duration::from_millis(500));
        }
        self.stage = None;
        self.capture_rate = Rate::default();
        self.output_rate = Rate::default();
    }

    /// Reset when the active model changes.  A running session is stopped so
    /// the next start detects with the new model.
    pub fn configure_for_model(&mut self, defaults: SearchDefaults) {
        self.stop();
        self.defaults = defaults;
        self.params = SearchParams::from(defaults);
        self.clear_output();
    }

    /// Reset when the model is cleared by loading a new teach image.
    pub fn clear_model(&mut self) {
        self.stop();
        self.clear_output();
    }

    fn clear_output(&mut self) {
        self.output = None;
        self.detections_texture = None;
        self.preview_frame = None;
        self.preview_sequence = 0;
        self.preview_texture = None;
        self.frozen = None;
        self.description = None;
        self.latency_ms = 0.0;
        self.capture_rate = Rate::default();
        self.output_rate = Rate::default();
        self.stage = None;
        self.error = None;
    }

    /// Freeze the currently displayed image (including detections when shown).
    pub fn freeze(&mut self) -> bool {
        if self.frozen.is_some() {
            return true;
        }
        let (frame, texture) = if self.show_detections && self.output.is_some() {
            (
                self.output.as_ref().map(|output| Arc::clone(&output.frame)),
                self.detections_texture.clone(),
            )
        } else {
            (self.preview_frame.clone(), self.preview_texture.clone())
        };
        let (Some(frame), Some(texture)) = (frame, texture) else {
            return false;
        };
        self.frozen = Some(FrozenFrame { texture, frame });
        true
    }

    /// Resume live display updates.
    pub fn unfreeze(&mut self) {
        self.frozen = None;
    }

    /// Image corresponding to the current (frozen or live) display.
    pub fn snapshot_frame(&self) -> Option<LoadedImage> {
        let frame = self
            .frozen
            .as_ref()
            .map(|frozen| &frozen.frame)
            .or(self.preview_frame.as_ref())?;
        Some(frame.image.clone())
    }

    /// Drain worker state and maintain textures.  Returns a status message
    /// when the session stops.
    pub fn poll(&mut self, ctx: &egui::Context) -> Option<String> {
        let shared = Arc::clone(&self.session.as_ref()?.shared);

        if self.frozen.is_none() {
            let frame = shared
                .latest_frame
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .clone();
            if let Some(frame) = frame.filter(|frame| frame.sequence != self.preview_sequence) {
                update_texture(
                    ctx,
                    &mut self.preview_texture,
                    "live-preview",
                    &frame.image.display,
                );
                self.preview_sequence = frame.sequence;
                self.preview_frame = Some(frame);
            }
        }

        if self.frozen.is_none() {
            let output = shared
                .latest_output
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .take();
            if let Some(output) = output {
                update_texture(
                    ctx,
                    &mut self.detections_texture,
                    "live-detections",
                    &output.overlay,
                );
                self.latency_ms = output.latency_ms();
                self.output = Some(output);
            }
        }

        let description = shared
            .description
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        if description.is_some() {
            self.description = description;
        }
        self.stage = *shared
            .progress
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let now = Instant::now();
        self.capture_rate
            .update(shared.capture_count.load(Ordering::Relaxed), now);
        self.output_rate
            .update(shared.output_count.load(Ordering::Relaxed), now);

        let error = shared
            .error
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        let finished = {
            let session = self.session.as_ref().expect("session exists");
            session.capture.as_ref().is_none_or(JoinHandle::is_finished)
                && session
                    .detector
                    .as_ref()
                    .is_none_or(JoinHandle::is_finished)
        };

        if let Some(error) = error {
            self.stop();
            self.error = Some(error.clone());
            return Some(error);
        }
        if finished {
            self.stop();
            return Some("Camera stopped.".to_owned());
        }

        ctx.request_repaint_after(Duration::from_millis(33));
        None
    }

    /// Camera controls and live statistics.  Returns `true` when Start was
    /// clicked; the caller supplies the context and model.
    pub fn left_panel(&mut self, ui: &mut egui::Ui, has_model: bool) -> bool {
        let mut start_requested = false;
        ui.heading("Camera");
        if !camera_available() {
            ui.colored_label(
                ui.visuals().warn_fg_color,
                "This build has no camera backend.",
            );
            return false;
        }
        self.ensure_devices();
        let running = self.is_running();
        ui.add_enabled_ui(!running, |ui| {
            ui.horizontal(|ui| {
                let selected = self
                    .selected_device()
                    .map_or_else(|| "Select camera".to_owned(), CaptureDevice::label);
                egui::ComboBox::from_id_salt("live_camera_device")
                    .selected_text(selected)
                    .show_ui(ui, |ui| {
                        for (index, device) in self.devices.iter().enumerate() {
                            ui.selectable_value(
                                &mut self.device_choice,
                                Some(index),
                                device.label(),
                            );
                        }
                    });
                if ui.button("Refresh").clicked() {
                    self.reload_devices();
                }
            });
            ui.horizontal(|ui| {
                ui.label("Width");
                ui.add(
                    egui::DragValue::new(&mut self.config.width)
                        .range(160..=4096)
                        .speed(1.0),
                );
                ui.label("Height");
                ui.add(
                    egui::DragValue::new(&mut self.config.height)
                        .range(120..=2160)
                        .speed(1.0),
                );
                ui.label("FPS");
                ui.add(
                    egui::DragValue::new(&mut self.config.fps)
                        .range(1..=120)
                        .speed(1.0),
                );
            });
        });
        if running {
            if ui.button("Stop camera").clicked() {
                self.stop();
            }
        } else if ui
            .add_enabled(self.can_start(), egui::Button::new("Start camera"))
            .clicked()
        {
            start_requested = true;
        }
        if let Some(description) = &self.description {
            ui.label(description);
        }
        if let Some(error) = &self.error {
            ui.colored_label(ui.visuals().warn_fg_color, error);
        }

        ui.separator();
        ui.heading("Live");
        ui.label(format!(
            "Capture {:.1} fps · processed {:.1} fps",
            self.capture_rate.value, self.output_rate.value
        ));
        if let Some((stage, fraction)) = self.stage
            && running
        {
            ui.label(format!("{} {:.0}%", stage.as_str(), fraction * 100.0));
            ui.add(egui::ProgressBar::new(fraction).show_percentage());
        }
        if self.has_output() {
            ui.label(format!(
                "{} detection(s) · latency {:.0} ms",
                self.detection_count(),
                self.latency_ms
            ));
        } else if running {
            ui.label("Waiting for the first processed frame.");
        }
        if self.is_frozen() {
            ui.colored_label(ui.visuals().warn_fg_color, "Display frozen");
        }
        ui.checkbox(&mut self.show_detections, "Show detections");
        if running && !has_model {
            ui.label("Load a model to detect; preview is still available.");
        }
        ui.separator();
        ui.heading("Freeze");
        ui.horizontal(|ui| {
            if self.is_frozen() {
                if ui.button("Resume").clicked() {
                    self.unfreeze();
                }
            } else if ui
                .add_enabled(self.has_frame(), egui::Button::new("Freeze frame"))
                .clicked()
            {
                self.freeze();
            }
        });
        ui.label("Use the toolbar to teach from or test the frozen frame.");
        start_requested
    }

    /// Live search parameters and the latest detection summary.
    pub fn right_panel(&mut self, ui: &mut egui::Ui, has_model: bool) {
        ui.heading("Live search parameters");
        if !has_model {
            ui.colored_label(ui.visuals().warn_fg_color, "No model is loaded.");
        }
        if ui
            .add_enabled(has_model, egui::Button::new("Reset to model defaults"))
            .clicked()
        {
            self.params = SearchParams::from(self.defaults);
        }
        RunState::search_parameter_ui(ui, &mut self.params);
        self.push_params();

        ui.separator();
        ui.heading("Latest detection");
        let Some(output) = &self.output else {
            ui.label("No processed frame yet.");
            return;
        };
        ui.label(format!(
            "Frame {} · detection {:.1} ms · latency {:.0} ms",
            output.frame.sequence, output.document.timings_ms.total, self.latency_ms
        ));
        let count = output.document.detections.len();
        ui.label(format!(
            "{count} detection{}",
            if count == 1 { "" } else { "s" }
        ));
        if count == 0 {
            return;
        }
        egui::ScrollArea::horizontal().show(ui, |ui| {
            egui::Grid::new("live_results")
                .num_columns(8)
                .striped(true)
                .spacing([8.0, 3.0])
                .show(ui, |ui| {
                    for header in ["#", "x", "y", "angle°", "scale", "score", "conf", "H"] {
                        ui.strong(header);
                    }
                    ui.end_row();
                    for record in &output.document.detections {
                        ui.label(record.id.to_string());
                        ui.label(format!("{:.1}", record.pose.x));
                        ui.label(format!("{:.1}", record.pose.y));
                        ui.label(format!("{:.2}", record.pose.angle_deg));
                        ui.label(format!("{:.3}", record.pose.scale));
                        ui.label(format!("{:.3}", record.score));
                        ui.label(format!("{:.3}", record.confidence));
                        ui.label(if record.homography_estimated {
                            "yes"
                        } else {
                            "sim"
                        });
                        ui.end_row();
                    }
                });
        });
    }

    fn push_params(&self) {
        if let Some(session) = &self.session {
            let mut params = self.params;
            params.search_region = None;
            if let Ok(mut shared) = session.shared.params.lock() {
                *shared = params;
            }
        }
    }

    /// Draw the live image.  Returns the cursor position in image coordinates.
    pub fn draw_canvas(&mut self, ui: &mut egui::Ui) -> Option<(f32, f32)> {
        let handle = if let Some(frozen) = &self.frozen {
            Some(&frozen.texture)
        } else if self.show_detections && self.detections_texture.is_some() {
            self.detections_texture.as_ref()
        } else {
            self.preview_texture.as_ref()
        };
        let Some(handle) = handle else {
            ui.centered_and_justified(|ui| {
                ui.label(if self.is_running() {
                    "Waiting for the camera…"
                } else {
                    "Start the camera to preview, freeze, and teach from it."
                });
            });
            return None;
        };
        let [width, height] = handle.size();
        let image_size = egui::vec2(width as f32, height as f32);
        let available = ui.available_rect_before_wrap();
        let scale = (available.width() / image_size.x)
            .min(available.height() / image_size.y)
            .max(1e-3);
        let rect = Rect::from_center_size(available.center(), image_size * scale);
        let response = ui.interact(rect, ui.scope_id().with("live_canvas"), Sense::hover());
        let hover = response
            .hover_pos()
            .map(|pos| screen_to_image(pos, rect, scale));

        let painter = ui.painter();
        let uv = Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
        painter.image(handle.id(), rect, uv, Color32::WHITE);

        let state = if self.frozen.is_some() {
            "FROZEN"
        } else if self.is_running() {
            "LIVE"
        } else {
            "STOPPED"
        };
        let mut caption = format!(
            "{state} · {:.1} fps capture · {:.1} fps processed · {:.0} ms",
            self.capture_rate.value, self.output_rate.value, self.latency_ms
        );
        if self.has_output() {
            caption.push_str(&format!(" · {} detection(s)", self.detection_count()));
        }
        // A translucent backing keeps the caption legible over any scene.
        let galley = painter.layout_no_wrap(
            caption,
            FontId::proportional(13.0),
            Color32::from_rgba_unmultiplied(240, 240, 240, 240),
        );
        let position = rect.left_top() + egui::vec2(10.0, 9.0);
        painter.rect_filled(
            Rect::from_min_size(
                position - egui::vec2(5.0, 4.0),
                galley.size() + egui::vec2(10.0, 8.0),
            ),
            4.0,
            Color32::from_black_alpha(130),
        );
        painter.galley(position, galley, Color32::WHITE);
        hover.map(|point| (point[0], point[1]))
    }
}

/// Wait briefly for a worker to notice cancellation.  If it does not (for
/// example, a camera read is stuck), the handle is dropped and the thread is
/// detached rather than freezing the UI.
fn join_briefly(handle: JoinHandle<()>, timeout: Duration) {
    let deadline = Instant::now() + timeout;
    while !handle.is_finished() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(2));
    }
    if handle.is_finished() {
        let _ = handle.join();
    }
}

fn capture_loop<F>(open: F, shared: Arc<Shared>, ctx: egui::Context)
where
    F: FnOnce() -> pm_core::Result<Box<dyn FrameSource>>,
{
    let mut source = match open() {
        Ok(source) => source,
        Err(error) => {
            shared.fail(format!("Failed to open camera: {error}"));
            ctx.request_repaint();
            return;
        }
    };
    *shared
        .description
        .lock()
        .unwrap_or_else(PoisonError::into_inner) = Some(source.description());
    ctx.request_repaint();

    while !shared.stop.load(Ordering::Relaxed) {
        match source.next_frame() {
            Ok(Some(frame)) => {
                {
                    let mut slot = shared
                        .latest_frame
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner);
                    *slot = Some(Arc::new(frame));
                }
                shared.capture_count.fetch_add(1, Ordering::Relaxed);
                shared.frame_ready.notify_one();
                ctx.request_repaint();
            }
            Ok(None) => break,
            Err(error) => {
                shared.fail(format!("Camera stopped: {error}"));
                break;
            }
        }
    }
    let _ = source.stop();
    shared.request_stop();
    ctx.request_repaint();
}

fn detection_loop(model: Model, shared: Arc<Shared>, ctx: egui::Context) {
    let mut last_sequence = 0_u64;
    loop {
        if shared.stop.load(Ordering::Relaxed) {
            break;
        }
        let frame = {
            let mut slot = shared
                .latest_frame
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            loop {
                if shared.stop.load(Ordering::Relaxed) {
                    return;
                }
                if let Some(frame) = slot
                    .as_ref()
                    .filter(|frame| frame.sequence != last_sequence)
                    .map(Arc::clone)
                {
                    break frame;
                }
                let (guard, _timeout) = shared
                    .frame_ready
                    .wait_timeout(slot, Duration::from_millis(100))
                    .unwrap_or_else(PoisonError::into_inner);
                slot = guard;
            }
        };
        last_sequence = frame.sequence;
        let params = *shared.params.lock().unwrap_or_else(PoisonError::into_inner);
        let progress = LiveProgress {
            shared: Arc::clone(&shared),
        };
        match detect_with_progress(&model, &frame.image.gray, &params, &progress) {
            Ok(detection) => {
                let document = pm_core::ResultsFile::new(
                    model.name.clone(),
                    frame.image.gray.width,
                    frame.image.gray.height,
                    &params,
                    &detection,
                );
                let roi_model = model_roi_in_model_coords(&model);
                let overlay = render_detection_overlay(
                    &frame.image.display,
                    &roi_model,
                    &model.features.edges.points,
                    &detection.detections,
                    &DetectionOverlayOptions::default(),
                );
                let output = LiveOutput {
                    document,
                    overlay,
                    frame,
                    completed_at: Instant::now(),
                };
                *shared
                    .latest_output
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner) = Some(output);
                *shared
                    .progress
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner) = None;
                shared.output_count.fetch_add(1, Ordering::Relaxed);
                ctx.request_repaint();
            }
            Err(pm_core::Error::Cancelled) => break,
            Err(error) => {
                shared.fail(format!("Detection failed: {error}"));
                break;
            }
        }
    }
}

fn update_texture(
    ctx: &egui::Context,
    slot: &mut Option<egui::TextureHandle>,
    name: &str,
    image: &RgbaImage,
) {
    let size = [image.width() as usize, image.height() as usize];
    let color = egui::ColorImage::from_rgba_unmultiplied(size, image.as_raw());
    if let Some(handle) = slot
        && handle.size() == size
    {
        handle.set(color, egui::TextureOptions::LINEAR);
        return;
    }
    *slot = Some(ctx.load_texture(name, color, egui::TextureOptions::LINEAR));
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use pm_core::{
        FeatureParams, GrayF32, HypothesisSources, Model, RotatedRect, SearchParams, TeachSettings,
        ZeroFrame,
    };

    use super::*;

    fn test_image(width: usize, height: usize) -> LoadedImage {
        let mut gray = GrayF32::filled(width, height, 0.15);
        for y in height / 4..height * 3 / 4 {
            for x in width / 4..width * 3 / 4 {
                gray.data[y * width + x] = 0.85;
            }
        }
        LoadedImage {
            display: gray.display_rgba(),
            gray,
        }
    }

    struct TestSource {
        image: LoadedImage,
        sequence: u64,
        delay: Duration,
    }

    impl FrameSource for TestSource {
        fn description(&self) -> String {
            "test source 96×72".to_owned()
        }

        fn next_frame(&mut self) -> pm_core::Result<Option<Frame>> {
            std::thread::sleep(self.delay);
            self.sequence += 1;
            Ok(Some(Frame {
                image: self.image.clone(),
                sequence: self.sequence,
                captured_at: Instant::now(),
            }))
        }
    }

    fn tiny_params(defaults: SearchDefaults) -> SearchParams {
        SearchParams {
            rotation_min_deg: -10.0,
            rotation_max_deg: 10.0,
            rotation_bins: 3,
            scale_min: 0.96,
            scale_max: 1.04,
            scale_bins: 3,
            max_model_hough_points: 128,
            max_test_hough_points: 128,
            hypothesis_cap: 20,
            max_hypotheses_scored: 4,
            hypothesis_sources: HypothesisSources::Hough,
            score_threshold: 0.2,
            coverage_min: 0.2,
            ..SearchParams::from(defaults)
        }
    }

    fn wait_until(
        live: &mut LiveState,
        ctx: &egui::Context,
        deadline: Instant,
        mut condition: impl FnMut(&LiveState) -> bool,
    ) {
        while !condition(live) {
            let _ = live.poll(ctx);
            assert!(Instant::now() < deadline, "live test timed out");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn live_capture_detection_and_freeze() {
        let image = test_image(96, 72);
        let model = Model::build(
            &image,
            &TeachSettings {
                name: "live-test".to_owned(),
                roi: RotatedRect::new(48.0, 36.0, 60.0, 44.0, 0.0),
                zero: ZeroFrame {
                    origin: [48.0, 36.0],
                    angle: 0.0,
                },
                params: FeatureParams::default(),
                mask: Vec::new(),
                search_defaults: SearchDefaults::default(),
            },
        )
        .expect("build model");

        let ctx = egui::Context::default();
        let mut live = LiveState::default();
        live.params = tiny_params(model.search_defaults);
        let source_image = image.clone();
        live.start_with(&ctx, Some(model), move || {
            Ok(Box::new(TestSource {
                image: source_image,
                sequence: 0,
                delay: Duration::from_millis(20),
            }))
        })
        .expect("start live");

        let deadline = Instant::now() + Duration::from_secs(90);
        wait_until(&mut live, &ctx, deadline, |live| {
            live.has_frame() && live.capture_fps() > 0.0
        });
        wait_until(&mut live, &ctx, deadline, LiveState::has_output);
        assert!(
            live.detection_count() > 0,
            "the taught image should match itself"
        );

        assert!(live.freeze(), "a frame is available to freeze");
        assert!(live.is_frozen());
        let frozen = live.snapshot_frame().expect("frozen image");
        assert_eq!((frozen.gray.width, frozen.gray.height), (96, 72));
        live.unfreeze();
        assert!(!live.is_frozen());

        live.stop();
        assert!(!live.is_running());
    }

    /// End-to-end hardware check: teach from a real webcam frame, then detect
    /// the same scene through the live pipeline.  Run with
    /// `cargo test -p pm-ui -- --ignored webcam_frame`.
    #[cfg(feature = "camera")]
    #[test]
    #[ignore = "requires a connected camera"]
    #[allow(clippy::field_reassign_with_default)]
    fn webcam_frame_can_be_taught_and_detected_live() {
        let devices = list_cameras().expect("enumerate cameras");
        assert!(!devices.is_empty(), "no cameras found");
        let device = devices[0].clone();
        let config = CaptureConfig::default();
        let mut source = open_webcam(&device, config).expect("open camera");
        let frame = source
            .next_frame()
            .expect("read frame")
            .expect("frame available");
        let _ = source.stop();
        drop(source);
        let loaded = frame.image;
        let (width, height) = (loaded.gray.width, loaded.gray.height);
        let model = Model::build(
            &loaded,
            &TeachSettings {
                name: "webcam-live".to_owned(),
                roi: RotatedRect::new(
                    width as f32 / 2.0,
                    height as f32 / 2.0,
                    width as f32 * 0.8,
                    height as f32 * 0.8,
                    0.0,
                ),
                zero: ZeroFrame {
                    origin: [width as f32 / 2.0, height as f32 / 2.0],
                    angle: 0.0,
                },
                params: FeatureParams::default(),
                mask: Vec::new(),
                search_defaults: SearchDefaults::default(),
            },
        )
        .expect("build a model from the camera frame");

        let ctx = egui::Context::default();
        let mut live = LiveState::default();
        live.params = tiny_params(model.search_defaults);
        live.start_with(&ctx, Some(model), move || open_webcam(&device, config))
            .expect("start live");
        let deadline = Instant::now() + Duration::from_secs(120);
        wait_until(&mut live, &ctx, deadline, |live| {
            live.has_output() && live.detection_count() > 0
        });
        assert!(live.freeze(), "a live frame is available to freeze");
        let frozen = live.snapshot_frame().expect("frozen camera frame");
        assert_eq!((frozen.gray.width, frozen.gray.height), (width, height));
        live.stop();
    }
}
