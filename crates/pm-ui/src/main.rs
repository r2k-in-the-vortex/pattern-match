//! Pattern Match desktop application.
//!
//! M3 adds teach mode: a rotated ROI, zero frame, mask tools with undo/redo,
//! model save/load, and the trainable model summary.  M7.5 adds still-image run
//! mode with a separate test image, worker-thread detection, results inspection,
//! and JSON/CSV/overlay export.  M8 adds live webcam mode with device and mode
//! selection, freeze-frame teaching, and frame-dropping live detection.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod live;
mod run;

use std::path::PathBuf;
use std::time::{Duration, Instant};

use eframe::egui::{self, Color32, Pos2, Rect, Sense, Stroke};
use live::LiveState;
use pm_core::{
    FeatureParams, FeatureSet, GrayF32, LoadedImage, MaskBounds, MaskPrimitive, MaskRaster, Model,
    OverlayOptions, RgbaImage, RotatedRect, SearchDefaults, TeachSettings, ZeroFrame,
    extract_features, filter_features, load_image, mask_bounds_for, render_overlay,
};
use run::RunState;

const MIN_ROI_SIZE: f32 = 4.0;
pub(crate) const HANDLE_HIT_RADIUS: f32 = 10.0;
const MASK_HISTORY_LIMIT: usize = 64;

/// Top-level application mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    /// Teach a model from the source image.
    Teach,
    /// Run the loaded model against a still test image.
    Run,
    /// Preview and detect on a live camera stream.
    Live,
}

fn main() -> eframe::Result {
    let startup_image = std::env::args_os().nth(1).map(PathBuf::from);
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_title("Pattern Match"),
        ..Default::default()
    };

    eframe::run_native(
        "Pattern Match",
        options,
        Box::new(move |cc| {
            let mut app = PatternMatchApp::default();
            if let Some(path) = startup_image {
                app.open_path(&cc.egui_ctx, path);
            }
            Ok(Box::new(app))
        }),
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tool {
    Select,
    Roi,
    Zero,
    PolygonMask,
    BrushMask,
}

impl Tool {
    fn hint(self) -> &'static str {
        match self {
            Self::Select => "Select: pan not required; choose a teaching tool.",
            Self::Roi => {
                "ROI: drag to create, drag inside to move, corners resize, handle rotates."
            }
            Self::Zero => "Zero: click to place the origin, drag the axis handle to rotate.",
            Self::PolygonMask => {
                "Mask polygon: click vertices; double-click or Enter closes; Esc cancels."
            }
            Self::BrushMask => "Mask brush: drag to paint; choose Add or Erase in the panel.",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum RoiDrag {
    Create { start: [f32; 2] },
    Move { last: [f32; 2] },
    Resize { anchor: [f32; 2] },
    Rotate,
}

struct PatternMatchApp {
    mode: Mode,
    run: RunState,
    live: LiveState,
    loaded: Option<LoadedImage>,
    source_available: bool,
    features: Option<FeatureSet>,
    image_texture: Option<egui::TextureHandle>,
    overlay_texture: Option<egui::TextureHandle>,
    params: FeatureParams,
    overlay: OverlayOptions,
    params_dirty: bool,
    params_changed_at: Option<Instant>,
    extraction_ms: f64,
    status: String,
    image_path: Option<PathBuf>,
    model_path: Option<PathBuf>,
    cursor_px: Option<(f32, f32)>,

    tool: Tool,
    roi: Option<RotatedRect>,
    zero: Option<ZeroFrame>,
    roi_drag: Option<RoiDrag>,
    zero_rotating: bool,
    mask_primitives: Vec<MaskPrimitive>,
    mask_add: bool,
    brush_radius: f32,
    brush_stroke: Option<Vec<[f32; 2]>>,
    polygon_points: Vec<[f32; 2]>,
    show_masked: bool,
    undo_stack: Vec<Vec<MaskPrimitive>>,
    redo_stack: Vec<Vec<MaskPrimitive>>,
    model: Option<Model>,
    model_name: String,
    model_dirty: bool,
    search_defaults: SearchDefaults,
    preview_counts: (usize, usize, usize, usize),
    teach_changed: bool,
}

impl Default for PatternMatchApp {
    fn default() -> Self {
        Self {
            mode: Mode::Teach,
            run: RunState::default(),
            live: LiveState::default(),
            loaded: None,
            source_available: false,
            features: None,
            image_texture: None,
            overlay_texture: None,
            params: FeatureParams::default(),
            overlay: OverlayOptions::default(),
            params_dirty: false,
            params_changed_at: None,
            extraction_ms: 0.0,
            status: "Drag an image here or use Open image.".to_owned(),
            image_path: None,
            model_path: None,
            cursor_px: None,
            tool: Tool::Select,
            roi: None,
            zero: None,
            roi_drag: None,
            zero_rotating: false,
            mask_primitives: Vec::new(),
            mask_add: true,
            brush_radius: 12.0,
            brush_stroke: None,
            polygon_points: Vec::new(),
            show_masked: false,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            model: None,
            model_name: "model".to_owned(),
            model_dirty: false,
            search_defaults: SearchDefaults::default(),
            preview_counts: (0, 0, 0, 0),
            teach_changed: false,
        }
    }
}

impl PatternMatchApp {
    fn open_path(&mut self, ctx: &egui::Context, path: PathBuf) {
        if path
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("pmodel"))
        {
            self.open_model(ctx, path);
        } else if self.mode == Mode::Run {
            self.open_test_image(ctx, path);
        } else {
            self.mode = Mode::Teach;
            self.load(ctx, path);
        }
    }

    fn open_test_image(&mut self, ctx: &egui::Context, path: PathBuf) {
        match self.run.load_image(ctx, path.clone()) {
            Ok(()) => {
                self.status = format!("Test image loaded: {}", path.display());
            }
            Err(error) => {
                self.status = error;
            }
        }
    }

    fn load(&mut self, ctx: &egui::Context, path: PathBuf) {
        match load_image(&path) {
            Ok(loaded) => {
                let name = path.file_stem().map_or_else(
                    || "model".to_owned(),
                    |stem| stem.to_string_lossy().into_owned(),
                );
                self.set_source_image(ctx, loaded, name, Some(path));
                self.status = "Image loaded. Draw an ROI to begin teaching.".to_owned();
            }
            Err(error) => {
                self.status = format!("Failed to load {}: {error}", path.display());
            }
        }
    }

    /// Replace the teach source image, resetting teach and run state.  The
    /// image may come from disk, an embedded model source, or a live frame.
    fn set_source_image(
        &mut self,
        ctx: &egui::Context,
        loaded: LoadedImage,
        model_name: String,
        path: Option<PathBuf>,
    ) {
        self.image_texture = Some(texture_from_rgba(ctx, "source", &loaded.display));
        self.loaded = Some(loaded);
        self.source_available = true;
        self.model_name = model_name;
        self.image_path = path;
        self.model_path = None;
        self.model = None;
        self.model_dirty = false;
        self.run.clear_model();
        self.live.clear_model();
        self.roi = None;
        self.zero = None;
        self.mask_primitives.clear();
        self.undo_stack.clear();
        self.redo_stack.clear();
        self.polygon_points.clear();
        self.brush_stroke = None;
        self.roi_drag = None;
        self.zero_rotating = false;
        self.show_masked = false;
        self.tool = Tool::Select;
        self.params_dirty = false;
        self.params_changed_at = None;
        self.re_extract(ctx);
    }

    fn re_extract(&mut self, ctx: &egui::Context) {
        let Some(loaded) = &self.loaded else {
            self.features = None;
            self.overlay_texture = None;
            return;
        };
        let start = Instant::now();
        let features = extract_features(&loaded.gray, &self.params);
        self.extraction_ms = start.elapsed().as_secs_f64() * 1000.0;
        self.features = Some(features);
        self.params_dirty = false;
        self.params_changed_at = None;
        if self.model.is_some() {
            self.model_dirty = true;
        }
        self.refresh_preview_overlay(ctx);
    }

    fn refresh_preview_overlay(&mut self, ctx: &egui::Context) {
        self.teach_changed = false;
        let (Some(features), Some(loaded)) = (&self.features, &self.loaded) else {
            self.overlay_texture = None;
            self.preview_counts = (0, 0, 0, 0);
            return;
        };
        let mut preview = features.clone();
        if let Some(roi) = self.roi {
            preview = filter_features(&preview, |x, y| roi.contains(x, y));
        }
        if !self.show_masked
            && let (Some(zero), false) = (self.zero, self.mask_primitives.is_empty())
        {
            let bounds = self.mask_raster_bounds();
            let raster = MaskRaster::rasterize(&self.mask_primitives, bounds, 1.0);
            preview = filter_features(&preview, |x, y| !raster.is_excluded(zero.to_model([x, y])));
        }
        self.preview_counts = (
            preview.edge_count(),
            preview.chain_count(),
            preview.corner_count(),
            preview.segment_count(),
        );
        let overlay = render_overlay(
            loaded.display.width(),
            loaded.display.height(),
            &preview,
            &self.overlay,
        );
        self.overlay_texture = Some(texture_from_rgba(ctx, "overlay", &overlay));
    }

    fn mask_raster_bounds(&self) -> MaskBounds {
        let roi = self.roi.unwrap_or_else(|| {
            let (width, height) = self
                .loaded
                .as_ref()
                .map_or((1, 1), |loaded| (loaded.gray.width, loaded.gray.height));
            RotatedRect::new(
                width as f32 * 0.5,
                height as f32 * 0.5,
                width as f32,
                height as f32,
                0.0,
            )
        });
        let zero = self.zero.unwrap_or_else(ZeroFrame::identity);
        mask_bounds_for(&roi, &zero, &self.mask_primitives)
    }

    fn handle_drop(&mut self, ctx: &egui::Context) {
        let dropped = ctx.input(|input| {
            input
                .raw
                .dropped_files
                .first()
                .map(|file| file.path().to_path_buf())
        });
        if let Some(path) = dropped {
            self.open_path(ctx, path);
        }
    }

    fn mark_teach_changed(&mut self) {
        self.teach_changed = true;
        if self.model.is_some() {
            self.model_dirty = true;
        }
    }

    fn push_mask_primitive(&mut self, primitive: MaskPrimitive) {
        self.undo_stack.push(self.mask_primitives.clone());
        if self.undo_stack.len() > MASK_HISTORY_LIMIT {
            self.undo_stack.remove(0);
        }
        self.redo_stack.clear();
        self.mask_primitives.push(primitive);
        self.mark_teach_changed();
    }

    fn undo_mask(&mut self) {
        if let Some(previous) = self.undo_stack.pop() {
            self.redo_stack
                .push(std::mem::replace(&mut self.mask_primitives, previous));
            self.mark_teach_changed();
        }
    }

    fn redo_mask(&mut self) {
        if let Some(next) = self.redo_stack.pop() {
            self.undo_stack
                .push(std::mem::replace(&mut self.mask_primitives, next));
            self.mark_teach_changed();
        }
    }

    fn clear_mask(&mut self) {
        if !self.mask_primitives.is_empty() {
            self.undo_stack.push(self.mask_primitives.clone());
            self.redo_stack.clear();
            self.mask_primitives.clear();
            self.mark_teach_changed();
        }
    }

    fn finish_polygon(&mut self) {
        if self.polygon_points.len() >= 3 {
            if let Some(zero) = self.zero {
                let points = self
                    .polygon_points
                    .iter()
                    .map(|&point| zero.to_model(point))
                    .collect();
                self.push_mask_primitive(MaskPrimitive::Polygon {
                    points,
                    add: self.mask_add,
                });
            } else {
                self.status = "Place the zero frame before masking.".to_owned();
            }
        }
        self.polygon_points.clear();
    }

    fn save_model(&mut self, path: PathBuf) {
        let (Some(loaded), Some(roi), Some(zero)) = (&self.loaded, self.roi, self.zero) else {
            self.status = "Draw an ROI and place the zero frame before saving.".to_owned();
            return;
        };
        if !self.source_available {
            self.status = "This model has no embedded source image to rebuild from.".to_owned();
            return;
        }
        let settings = TeachSettings {
            name: self.model_name.clone(),
            roi,
            zero,
            params: self.params,
            mask: self.mask_primitives.clone(),
            search_defaults: self.search_defaults,
        };
        match Model::build(loaded, &settings) {
            Ok(model) => match model.save(&path) {
                Ok(()) => {
                    let defaults = model.search_defaults;
                    self.model_name = model.name.clone();
                    self.status = format!(
                        "Saved {}: {} edges, {} corners, {} segments, {:.1}% masked.",
                        path.display(),
                        model.features.edge_count(),
                        model.features.corner_count(),
                        model.features.segment_count(),
                        model.excluded_edge_weight_fraction * 100.0
                    );
                    self.model = Some(model);
                    self.model_path = Some(path);
                    self.model_dirty = false;
                    self.run.configure_for_model(defaults);
                    self.live.configure_for_model(defaults);
                }
                Err(error) => {
                    self.status = format!("Failed to save model: {error}");
                }
            },
            Err(error) => {
                self.status = format!("Failed to build model: {error}");
            }
        }
    }

    fn open_model(&mut self, ctx: &egui::Context, path: PathBuf) {
        match Model::load(&path) {
            Ok(model) => {
                self.params = model.params;
                self.roi = Some(model.roi);
                self.zero = Some(model.zero);
                self.mask_primitives = model.mask_primitives.clone();
                self.search_defaults = model.search_defaults;
                self.model_name = model.name.clone();
                self.undo_stack.clear();
                self.redo_stack.clear();
                self.polygon_points.clear();
                self.brush_stroke = None;
                self.roi_drag = None;
                self.zero_rotating = false;
                self.show_masked = false;
                self.tool = Tool::Select;
                self.params_dirty = false;
                self.params_changed_at = None;

                let has_source = model.source.is_some();
                self.source_available = has_source;
                if let Some(source) = model.source.clone() {
                    self.image_texture = Some(texture_from_rgba(ctx, "source", &source.display));
                    self.loaded = Some(source);
                    self.image_path = None;
                } else {
                    let (width, height) = model.source_size;
                    let gray = GrayF32::filled(width.max(1), height.max(1), 0.5);
                    self.image_texture =
                        Some(texture_from_rgba(ctx, "source", &gray.display_rgba()));
                    self.loaded = Some(LoadedImage {
                        display: gray.display_rgba(),
                        gray,
                    });
                    self.image_path = None;
                }
                self.features = Some(model.features_in_image_coords());
                if has_source {
                    // Re-extract the full teach preview from the embedded source;
                    // the saved model remains authoritative.
                    let start = Instant::now();
                    if let Some(loaded) = &self.loaded {
                        let features = extract_features(&loaded.gray, &self.params);
                        self.features = Some(features);
                        self.extraction_ms = start.elapsed().as_secs_f64() * 1000.0;
                    }
                }
                self.model_dirty = false;
                self.model_path = Some(path.clone());
                let defaults = model.search_defaults;
                self.model = Some(model);
                self.run.configure_for_model(defaults);
                self.live.configure_for_model(defaults);
                self.refresh_preview_overlay(ctx);
                self.status = if has_source {
                    format!("Opened {}.", path.display())
                } else {
                    format!("Opened {} (no embedded source).", path.display())
                };
            }
            Err(error) => {
                self.status = format!("Failed to open {}: {error}", path.display());
            }
        }
    }

    fn update_roi_fields(&mut self, ui: &mut egui::Ui) {
        let mut roi = self.roi;
        let mut changed = false;
        if let Some(rect) = &mut roi {
            egui::Grid::new("roi_grid")
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
                                .range(MIN_ROI_SIZE..=f32::MAX),
                        )
                        .changed();
                    ui.end_row();
                    ui.label("Height");
                    changed |= ui
                        .add(
                            egui::DragValue::new(&mut rect.height)
                                .speed(0.5)
                                .range(MIN_ROI_SIZE..=f32::MAX),
                        )
                        .changed();
                    ui.end_row();
                    ui.label("Angle");
                    let mut degrees = rect.angle.to_degrees();
                    if ui
                        .add(egui::DragValue::new(&mut degrees).speed(1.0).suffix("°"))
                        .changed()
                    {
                        rect.angle = wrap_angle(degrees.to_radians());
                        changed = true;
                    }
                    ui.end_row();
                });
        } else {
            ui.label("No ROI. Use the ROI tool or set the full image.");
            if ui.button("Use full image").clicked()
                && let Some(loaded) = &self.loaded
            {
                roi = Some(RotatedRect::new(
                    loaded.gray.width as f32 * 0.5,
                    loaded.gray.height as f32 * 0.5,
                    loaded.gray.width as f32,
                    loaded.gray.height as f32,
                    0.0,
                ));
                changed = true;
            }
        }
        if changed {
            self.roi = roi;
            self.mark_teach_changed();
        }
    }

    fn update_zero_fields(&mut self, ui: &mut egui::Ui) {
        let mut zero = self.zero;
        let mut changed = false;
        if let Some(frame) = &mut zero {
            egui::Grid::new("zero_grid")
                .num_columns(2)
                .spacing([8.0, 4.0])
                .show(ui, |ui| {
                    ui.label("Origin x");
                    changed |= ui
                        .add(egui::DragValue::new(&mut frame.origin[0]).speed(0.5))
                        .changed();
                    ui.end_row();
                    ui.label("Origin y");
                    changed |= ui
                        .add(egui::DragValue::new(&mut frame.origin[1]).speed(0.5))
                        .changed();
                    ui.end_row();
                    ui.label("Angle");
                    let mut degrees = frame.angle.to_degrees();
                    if ui
                        .add(egui::DragValue::new(&mut degrees).speed(1.0).suffix("°"))
                        .changed()
                    {
                        frame.angle = wrap_angle(degrees.to_radians());
                        changed = true;
                    }
                    ui.end_row();
                });
        } else {
            ui.label("No zero frame. Click with the Zero tool.");
        }
        if changed {
            self.zero = zero;
            self.mark_teach_changed();
        }
    }

    fn update_mask_panel(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.radio_value(&mut self.mask_add, true, "Add");
            ui.radio_value(&mut self.mask_add, false, "Erase");
        });
        ui.add(egui::Slider::new(&mut self.brush_radius, 1.0..=80.0).text("Brush radius"));
        ui.horizontal(|ui| {
            if ui
                .add_enabled(!self.undo_stack.is_empty(), egui::Button::new("Undo"))
                .clicked()
            {
                self.undo_mask();
            }
            if ui
                .add_enabled(!self.redo_stack.is_empty(), egui::Button::new("Redo"))
                .clicked()
            {
                self.redo_mask();
            }
            if ui
                .add_enabled(!self.mask_primitives.is_empty(), egui::Button::new("Clear"))
                .clicked()
            {
                self.clear_mask();
            }
        });
        if ui.checkbox(&mut self.show_masked, "Show masked").changed() {
            self.mark_teach_changed();
        }
        ui.label(format!("Mask primitives: {}", self.mask_primitives.len()));
    }

    fn draw_canvas(&mut self, ui: &mut egui::Ui) -> Option<(f32, f32)> {
        let Some(loaded) = &self.loaded else {
            ui.centered_and_justified(|ui| {
                ui.label("Drag an image here or use Open image…");
            });
            return None;
        };
        let (image_width, image_height) = (loaded.display.width(), loaded.display.height());
        let image_texture = self.image_texture.as_ref().map(|texture| texture.id());
        let overlay_texture = self.overlay_texture.as_ref().map(|texture| texture.id());

        let available = ui.available_rect_before_wrap();
        let image_size = egui::vec2(image_width as f32, image_height as f32);
        let scale = (available.width() / image_size.x)
            .min(available.height() / image_size.y)
            .max(1e-3);
        let rect = Rect::from_center_size(available.center(), image_size * scale);
        let response = ui.interact(rect, ui.scope_id().with("canvas"), Sense::click_and_drag());
        let pointer = response
            .interact_pointer_pos()
            .map(|pos| screen_to_image(pos, rect, scale));
        let hover = response
            .hover_pos()
            .map(|pos| screen_to_image(pos, rect, scale));

        if !ui.ctx().egui_wants_keyboard_input() {
            if ui.input(|input| input.key_pressed(egui::Key::Enter))
                && !self.polygon_points.is_empty()
            {
                self.finish_polygon();
            }
            if ui.input(|input| input.key_pressed(egui::Key::Escape)) {
                self.polygon_points.clear();
                self.brush_stroke = None;
            }
            if ui.input(|input| input.modifiers.command && input.key_pressed(egui::Key::Z)) {
                self.undo_mask();
            }
            if ui.input(|input| input.modifiers.command && input.key_pressed(egui::Key::Y)) {
                self.redo_mask();
            }
        }

        match self.tool {
            Tool::Roi => self.handle_roi(rect, scale, &response, pointer),
            Tool::Zero => self.handle_zero(rect, scale, &response, pointer),
            Tool::PolygonMask => self.handle_polygon(&response, pointer),
            Tool::BrushMask => self.handle_brush(rect, scale, &response, pointer),
            Tool::Select => {}
        }

        let painter = ui.painter();
        let uv = Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
        if let Some(texture) = image_texture {
            painter.image(texture, rect, uv, Color32::WHITE);
        }
        if let Some(texture) = overlay_texture {
            painter.image(texture, rect, uv, Color32::WHITE);
        }
        if let Some(zero) = self.zero {
            draw_mask_overlay(
                painter,
                rect,
                scale,
                &self.mask_primitives,
                &zero,
                self.mask_add,
                self.brush_stroke.as_deref(),
                self.brush_radius,
                &self.polygon_points,
            );
            draw_zero(painter, rect, scale, &zero);
        }
        if let Some(roi) = self.roi {
            draw_roi(painter, rect, scale, &roi);
        }
        painter.text(
            rect.left_top() + egui::vec2(8.0, 8.0),
            egui::Align2::LEFT_TOP,
            self.tool.hint(),
            egui::FontId::proportional(12.0),
            Color32::from_rgba_unmultiplied(20, 20, 20, 190),
        );

        hover.map(|point| (point[0], point[1]))
    }

    fn handle_roi(
        &mut self,
        rect: Rect,
        scale: f32,
        response: &egui::Response,
        pointer: Option<[f32; 2]>,
    ) {
        if edit_rotated_rect(
            rect,
            scale,
            response,
            pointer,
            &mut self.roi,
            &mut self.roi_drag,
        ) {
            self.mark_teach_changed();
        }
    }

    fn handle_zero(
        &mut self,
        rect: Rect,
        scale: f32,
        response: &egui::Response,
        pointer: Option<[f32; 2]>,
    ) {
        let Some(point) = pointer else {
            if response.drag_stopped() {
                self.zero_rotating = false;
            }
            return;
        };
        let screen = response
            .interact_pointer_pos()
            .expect("pointer position exists");

        if response.drag_started() {
            if let Some(zero) = self.zero {
                if image_to_screen(zero_handle_image(&zero), rect, scale).distance(screen)
                    <= HANDLE_HIT_RADIUS
                {
                    self.zero_rotating = true;
                } else {
                    self.zero = Some(ZeroFrame {
                        origin: point,
                        angle: zero.angle,
                    });
                    self.zero_rotating = false;
                }
            } else {
                self.zero = Some(ZeroFrame {
                    origin: point,
                    angle: 0.0,
                });
                self.zero_rotating = false;
            }
            self.mark_teach_changed();
        }

        if response.dragged()
            && self.zero_rotating
            && let Some(zero) = &mut self.zero
        {
            zero.angle = wrap_angle((point[1] - zero.origin[1]).atan2(point[0] - zero.origin[0]));
            self.mark_teach_changed();
        }

        if response.clicked() {
            self.zero = Some(ZeroFrame {
                origin: point,
                angle: self.zero.map_or(0.0, |zero| zero.angle),
            });
            self.mark_teach_changed();
        }

        if response.drag_stopped() {
            self.zero_rotating = false;
        }
    }

    fn handle_polygon(&mut self, response: &egui::Response, pointer: Option<[f32; 2]>) {
        if self.zero.is_none() {
            if response.clicked() || response.drag_started() {
                self.status = "Place the zero frame before masking.".to_owned();
            }
            return;
        }
        if response.secondary_clicked() {
            self.polygon_points.clear();
            return;
        }
        if response.double_clicked() {
            if self.polygon_points.len() >= 3 {
                self.finish_polygon();
            }
            return;
        }
        if response.clicked()
            && let Some(point) = pointer
        {
            self.polygon_points.push(point);
        }
    }

    fn handle_brush(
        &mut self,
        _rect: Rect,
        _scale: f32,
        response: &egui::Response,
        pointer: Option<[f32; 2]>,
    ) {
        if self.zero.is_none() {
            if response.clicked() || response.drag_started() {
                self.status = "Place the zero frame before masking.".to_owned();
            }
            return;
        }
        let Some(point) = pointer else {
            return;
        };

        if response.drag_started() {
            self.brush_stroke = Some(vec![point]);
        }
        if response.dragged()
            && let Some(stroke) = &mut self.brush_stroke
        {
            let threshold = (self.brush_radius * 0.5).max(1.0);
            if let Some(last) = stroke.last() {
                if (point[0] - last[0]).hypot(point[1] - last[1]) >= threshold {
                    stroke.push(point);
                }
            } else {
                stroke.push(point);
            }
        }
        if response.drag_stopped()
            && let Some(stroke) = self.brush_stroke.take()
        {
            self.finish_brush(stroke);
        }
        if response.clicked() {
            self.finish_brush(vec![point]);
        }
    }

    fn finish_brush(&mut self, stroke: Vec<[f32; 2]>) {
        if stroke.is_empty() {
            return;
        }
        if let Some(zero) = self.zero {
            let points = stroke.iter().map(|&point| zero.to_model(point)).collect();
            self.push_mask_primitive(MaskPrimitive::Brush {
                points,
                radius: self.brush_radius,
                add: self.mask_add,
            });
        }
    }
}

impl eframe::App for PatternMatchApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.handle_drop(&ctx);
        if let Some(status) = self.run.poll(&ctx) {
            self.status = status;
        }
        if let Some(status) = self.live.poll(&ctx) {
            self.status = status;
        }

        let mut live_start_requested = false;
        let mut live_stop_requested = false;
        let mut live_freeze_toggle = false;
        let mut live_teach_requested = false;
        let mut live_test_requested = false;
        let mut open_image_requested = false;
        let mut open_model_requested = false;
        let mut save_model_requested = false;
        let mut re_extract_requested = false;
        let mut overlay_changed = false;
        let mut open_test_image_requested = false;
        let mut run_requested = false;
        let mut cancel_requested = false;
        let mut export_json_requested = false;
        let mut export_csv_requested = false;
        let mut save_overlay_requested = false;

        egui::Panel::top("toolbar").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Pattern Match");
                ui.separator();
                ui.selectable_value(&mut self.mode, Mode::Teach, "Teach");
                ui.selectable_value(&mut self.mode, Mode::Run, "Run");
                ui.selectable_value(&mut self.mode, Mode::Live, "Live");
                ui.separator();
                match self.mode {
                    Mode::Teach => {
                        if ui.button("Open image…").clicked() {
                            open_image_requested = true;
                        }
                        if ui
                            .add_enabled(self.loaded.is_some(), egui::Button::new("Re-extract"))
                            .clicked()
                        {
                            re_extract_requested = true;
                        }
                        ui.separator();
                        if ui.button("Open model…").clicked() {
                            open_model_requested = true;
                        }
                        let can_save = self.loaded.is_some()
                            && self.roi.is_some()
                            && self.zero.is_some()
                            && self.source_available;
                        if ui
                            .add_enabled(can_save, egui::Button::new("Save model…"))
                            .clicked()
                        {
                            save_model_requested = true;
                        }
                        ui.separator();
                        let mut layers_changed = false;
                        layers_changed |=
                            ui.checkbox(&mut self.overlay.draw_edges, "Edges").changed();
                        layers_changed |= ui
                            .checkbox(&mut self.overlay.draw_corners, "Corners")
                            .changed();
                        layers_changed |= ui
                            .checkbox(&mut self.overlay.draw_segments, "Segments")
                            .changed();
                        overlay_changed = layers_changed;
                    }
                    Mode::Run => {
                        if ui.button("Open test image…").clicked() {
                            open_test_image_requested = true;
                        }
                        ui.separator();
                        if self.run.is_running() {
                            if ui.button("Cancel").clicked() {
                                cancel_requested = true;
                            }
                        } else {
                            let can_run = self.model.is_some() && self.run.has_image();
                            if ui.add_enabled(can_run, egui::Button::new("Run")).clicked() {
                                run_requested = true;
                            }
                        }
                        ui.separator();
                        if ui.button("Open model…").clicked() {
                            open_model_requested = true;
                        }
                        let has_result = self.run.has_result();
                        if ui
                            .add_enabled(has_result, egui::Button::new("Export JSON…"))
                            .clicked()
                        {
                            export_json_requested = true;
                        }
                        if ui
                            .add_enabled(has_result, egui::Button::new("Export CSV…"))
                            .clicked()
                        {
                            export_csv_requested = true;
                        }
                        if ui
                            .add_enabled(has_result, egui::Button::new("Save overlay…"))
                            .clicked()
                        {
                            save_overlay_requested = true;
                        }
                        ui.separator();
                        ui.checkbox(&mut self.run.show_detections, "Detections");
                    }
                    Mode::Live => {
                        if ui.button("Open model…").clicked() {
                            open_model_requested = true;
                        }
                        ui.separator();
                        if self.live.is_running() {
                            if ui.button("Stop camera").clicked() {
                                live_stop_requested = true;
                            }
                        } else if ui
                            .add_enabled(self.live.can_start(), egui::Button::new("Start camera"))
                            .clicked()
                        {
                            live_start_requested = true;
                        }
                        if self.live.is_frozen() {
                            if ui.button("Resume").clicked() {
                                live_freeze_toggle = true;
                            }
                        } else if ui
                            .add_enabled(self.live.has_frame(), egui::Button::new("Freeze"))
                            .clicked()
                        {
                            live_freeze_toggle = true;
                        }
                        if ui
                            .add_enabled(
                                self.live.has_frame(),
                                egui::Button::new("Teach from frame"),
                            )
                            .clicked()
                        {
                            live_teach_requested = true;
                        }
                        if ui
                            .add_enabled(
                                self.live.has_frame(),
                                egui::Button::new("Use as test image"),
                            )
                            .clicked()
                        {
                            live_test_requested = true;
                        }
                        ui.separator();
                        ui.checkbox(&mut self.live.show_detections, "Detections");
                    }
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(format!("pm-core {}", pm_core::VERSION));
                });
            });
        });

        egui::Panel::left("model_panel")
            .resizable(true)
            .default_size(250.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| match self.mode {
                    Mode::Teach => {
                        ui.heading("Image");
                        if let Some(path) = &self.image_path {
                            ui.label(path.file_name().map_or_else(
                                || path.display().to_string(),
                                |name| name.to_string_lossy().into_owned(),
                            ));
                        } else if let Some(loaded) = &self.loaded {
                            ui.label("Embedded source");
                            ui.label(format!("{} × {} px", loaded.gray.width, loaded.gray.height));
                        } else {
                            ui.label("No image loaded.");
                        }
                        if let Some(path) = &self.model_path {
                            ui.label(format!(
                                "Model: {}",
                                path.file_name().map_or_else(
                                    || path.display().to_string(),
                                    |name| name.to_string_lossy().into_owned(),
                                )
                            ));
                        }
                        ui.separator();
                        ui.heading("Tools");
                        let mut tool = self.tool;
                        ui.horizontal_wrapped(|ui| {
                            ui.selectable_value(&mut tool, Tool::Select, "Select");
                            ui.selectable_value(&mut tool, Tool::Roi, "ROI");
                            ui.selectable_value(&mut tool, Tool::Zero, "Zero");
                            ui.selectable_value(&mut tool, Tool::PolygonMask, "Polygon");
                            ui.selectable_value(&mut tool, Tool::BrushMask, "Brush");
                        });
                        if tool != self.tool {
                            self.tool = tool;
                            self.roi_drag = None;
                            self.zero_rotating = false;
                            self.brush_stroke = None;
                            self.polygon_points.clear();
                        }
                        ui.label(self.tool.hint());
                        ui.separator();
                        ui.heading("Mask");
                        self.update_mask_panel(ui);
                        ui.separator();
                        ui.heading("Features");
                        let (edges, chains, corners, segments) = self.preview_counts;
                        ui.label(format!("Edge points: {edges}"));
                        ui.label(format!("Chains: {chains}"));
                        ui.label(format!("Corners: {corners}"));
                        ui.label(format!("Segments: {segments}"));
                        ui.separator();
                        ui.heading("Orientation legend");
                        draw_orientation_legend(ui);
                    }
                    Mode::Run => {
                        let has_model = self.model.is_some();
                        if self.run.left_panel(ui, has_model) {
                            open_test_image_requested = true;
                        }
                        if self.loaded.is_some() {
                            ui.separator();
                            if ui.button("Use teach image").clicked()
                                && let Some(loaded) = self.loaded.clone()
                            {
                                self.run.set_image(&ctx, loaded, None);
                                self.status = "Teach image copied to the test image.".to_owned();
                            }
                        }
                    }
                    Mode::Live => {
                        if self.live.left_panel(ui, self.model.is_some()) {
                            live_start_requested = true;
                        }
                    }
                });
            });

        egui::Panel::right("parameters_panel")
            .resizable(true)
            .default_size(340.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| match self.mode {
                    Mode::Teach => {
                        ui.heading("Teach");
                        ui.label("ROI");
                        self.update_roi_fields(ui);
                        ui.separator();
                        ui.label("Zero frame");
                        self.update_zero_fields(ui);
                        ui.separator();
                        ui.label("Model");
                        ui.horizontal(|ui| {
                            ui.label("Name");
                            if ui.text_edit_singleline(&mut self.model_name).changed()
                                && self.model.is_some()
                            {
                                self.model_dirty = true;
                            }
                        });
                        if let Some(model) = &self.model {
                            let (edges, chains, corners, segments, descriptors, pairs) =
                                model.counts();
                            ui.label(format!("Edges: {edges}, chains: {chains}"));
                            ui.label(format!("Corners: {corners}, descriptors: {descriptors}"));
                            ui.label(format!("Segments: {segments}, pairs: {pairs}"));
                            ui.label(format!(
                                "Masked edge weight: {:.1}%",
                                model.excluded_edge_weight_fraction * 100.0
                            ));
                            if self.model_dirty {
                                ui.colored_label(
                                    ui.visuals().warn_fg_color,
                                    "Modified since last save",
                                );
                            }
                        }
                        ui.separator();
                        ui.heading("Parameters");
                        let mut changed = false;
                        changed |= ui
                            .add(
                                egui::Slider::new(&mut self.params.blur_sigma, 0.0..=3.0)
                                    .text("Blur σ"),
                            )
                            .changed();
                        ui.separator();
                        ui.label("Edges");
                        changed |= ui
                            .add(
                                egui::Slider::new(
                                    &mut self.params.edge.high_percentile,
                                    0.5..=0.99,
                                )
                                .text("Adaptive percentile"),
                            )
                            .changed();
                        changed |= ui
                            .add(
                                egui::Slider::new(&mut self.params.edge.high_floor, 0.01..=0.4)
                                    .text("High threshold floor"),
                            )
                            .changed();
                        changed |= ui
                            .add(
                                egui::Slider::new(&mut self.params.edge.low_ratio, 0.1..=0.9)
                                    .text("Low / high ratio"),
                            )
                            .changed();
                        changed |= ui
                            .add(
                                egui::Slider::new(&mut self.params.edge.min_chain_length, 1..=20)
                                    .text("Min chain length"),
                            )
                            .changed();
                        changed |= ui
                            .checkbox(&mut self.params.edge.subpixel, "Subpixel edges")
                            .changed();
                        ui.separator();
                        ui.label("Corners");
                        changed |= ui
                            .checkbox(&mut self.params.detect_corners, "Detect corners")
                            .changed();
                        changed |= ui
                            .add(
                                egui::Slider::new(&mut self.params.corner.quality, 0.001..=0.1)
                                    .logarithmic(true)
                                    .text("Quality"),
                            )
                            .changed();
                        changed |= ui
                            .add(
                                egui::Slider::new(&mut self.params.corner.min_distance, 1..=20)
                                    .text("Min distance"),
                            )
                            .changed();
                        changed |= ui
                            .add(
                                egui::Slider::new(&mut self.params.corner.window_sigma, 0.5..=4.0)
                                    .text("Window σ"),
                            )
                            .changed();
                        ui.separator();
                        ui.label("Segments");
                        changed |= ui
                            .checkbox(&mut self.params.fit_segments, "Fit segments")
                            .changed();
                        changed |= ui
                            .add(
                                egui::Slider::new(&mut self.params.segment.min_length, 2.0..=100.0)
                                    .text("Min length"),
                            )
                            .changed();
                        changed |= ui
                            .add(
                                egui::Slider::new(
                                    &mut self.params.segment.max_perpendicular_error,
                                    0.1..=3.0,
                                )
                                .text("Max deviation"),
                            )
                            .changed();
                        if changed {
                            self.params_dirty = true;
                            self.params_changed_at = Some(Instant::now());
                        }
                        ui.separator();
                        if ui
                            .add_enabled(
                                self.loaded.is_some(),
                                egui::Button::new("Apply and re-extract"),
                            )
                            .clicked()
                        {
                            re_extract_requested = true;
                        }
                        if self.params_dirty {
                            ui.colored_label(ui.visuals().warn_fg_color, "Parameters changed");
                        }
                    }
                    Mode::Run => {
                        let has_model = self.model.is_some();
                        let running = self.run.is_running();
                        self.run.right_panel(ui, has_model, running);
                    }
                    Mode::Live => self.live.right_panel(ui, self.model.is_some()),
                });
            });

        egui::Panel::bottom("status_bar").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(&self.status);
                ui.separator();
                if let Some((x, y)) = self.cursor_px {
                    ui.label(format!("({x:.1}, {y:.1}) px"));
                    ui.separator();
                }
                match self.mode {
                    Mode::Teach => {
                        ui.label(format!("features {:.1} ms", self.extraction_ms));
                    }
                    Mode::Run => {
                        if let Some(stage) = self.run.stage {
                            ui.label(format!(
                                "{} {:.0}%",
                                stage.as_str(),
                                self.run.fraction * 100.0
                            ));
                            ui.separator();
                        }
                        if self.run.last_elapsed_ms > 0.0 {
                            ui.label(format!("run {:.1} ms", self.run.last_elapsed_ms));
                        } else {
                            ui.label(format!("{} detection(s)", self.run.detection_count()));
                        }
                    }
                    Mode::Live => {
                        ui.label(format!(
                            "capture {:.1} fps · processed {:.1} fps · {:.0} ms",
                            self.live.capture_fps(),
                            self.live.processed_fps(),
                            self.live.latency_ms()
                        ));
                        ui.separator();
                        if let Some((stage, fraction)) = self.live.stage() {
                            ui.label(format!("{} {:.0}%", stage.as_str(), fraction * 100.0));
                        } else {
                            ui.label(format!("{} detection(s)", self.live.detection_count()));
                        }
                    }
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(format!("pm-ui {}", env!("CARGO_PKG_VERSION")));
                });
            });
        });

        let cursor = egui::CentralPanel::default()
            .show(ui, |ui| match self.mode {
                Mode::Teach => self.draw_canvas(ui),
                Mode::Run => self.run.draw_canvas(ui, self.model.as_ref()),
                Mode::Live => self.live.draw_canvas(ui),
            })
            .inner;
        self.cursor_px = cursor;

        if open_image_requested
            && let Some(path) = rfd::FileDialog::new()
                .add_filter("Images", &["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
                .pick_file()
        {
            self.load(&ctx, path);
        }
        if open_model_requested
            && let Some(path) = rfd::FileDialog::new()
                .add_filter("Pattern Match model", &["pmodel"])
                .pick_file()
        {
            self.open_model(&ctx, path);
        }
        if save_model_requested
            && let Some(path) = rfd::FileDialog::new()
                .add_filter("Pattern Match model", &["pmodel"])
                .set_file_name(format!("{}.pmodel", self.model_name))
                .save_file()
        {
            let path = if path.extension().is_some() {
                path
            } else {
                path.with_extension("pmodel")
            };
            self.save_model(path);
        }
        if open_test_image_requested
            && let Some(path) = rfd::FileDialog::new()
                .add_filter("Images", &["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
                .pick_file()
        {
            self.open_test_image(&ctx, path);
        }
        if run_requested {
            let outcome = match self.model.as_ref() {
                Some(model) => self.run.start(&ctx, model),
                None => Err("Open a model before running.".to_owned()),
            };
            match outcome {
                Ok(()) => self.status = "Detection started.".to_owned(),
                Err(error) => self.status = error,
            }
        }
        if cancel_requested {
            self.run.cancel();
            self.status = "Cancelling detection…".to_owned();
        }

        if live_start_requested {
            match self.live.start(&ctx, self.model.clone()) {
                Ok(()) => self.status = "Camera starting…".to_owned(),
                Err(error) => self.status = error,
            }
        }
        if live_stop_requested {
            self.live.stop();
            self.status = "Camera stopped.".to_owned();
        }
        if live_freeze_toggle {
            if self.live.is_frozen() {
                self.live.unfreeze();
            } else {
                self.live.freeze();
            }
        }
        if live_teach_requested && let Some(frame) = self.live.snapshot_frame() {
            self.set_source_image(&ctx, frame, "camera".to_owned(), None);
            self.mode = Mode::Teach;
            self.status =
                "Camera frame loaded for teaching. Draw an ROI and zero frame.".to_owned();
        }
        if live_test_requested && let Some(frame) = self.live.snapshot_frame() {
            self.run.set_image(&ctx, frame, None);
            self.live.stop();
            self.mode = Mode::Run;
            self.status = "Camera frame copied to the test image.".to_owned();
        }
        if self.mode != Mode::Live {
            // Release the device when the user leaves Live mode.
            self.live.stop();
        }

        let export_stem = self
            .run
            .document()
            .map_or_else(|| "results".to_owned(), |document| document.model.clone());
        if export_json_requested
            && let Some(path) = rfd::FileDialog::new()
                .add_filter("JSON", &["json"])
                .set_file_name(format!("{export_stem}-results.json"))
                .save_file()
        {
            let path = ensure_extension(path, "json");
            self.status = match self.run.export_json(&path) {
                Ok(()) => format!("Wrote {}", path.display()),
                Err(error) => error,
            };
        }
        if export_csv_requested
            && let Some(path) = rfd::FileDialog::new()
                .add_filter("CSV", &["csv"])
                .set_file_name(format!("{export_stem}-results.csv"))
                .save_file()
        {
            let path = ensure_extension(path, "csv");
            self.status = match self.run.export_csv(&path) {
                Ok(()) => format!("Wrote {}", path.display()),
                Err(error) => error,
            };
        }
        if save_overlay_requested
            && let Some(path) = rfd::FileDialog::new()
                .add_filter("PNG", &["png"])
                .set_file_name(format!("{export_stem}-overlay.png"))
                .save_file()
        {
            let path = ensure_extension(path, "png");
            self.status = match self.run.save_overlay_png(&path) {
                Ok(()) => format!("Wrote {}", path.display()),
                Err(error) => error,
            };
        }
        let mut live_extract = false;
        if self.params_dirty
            && let Some(changed_at) = self.params_changed_at
        {
            let delay = Duration::from_millis(250);
            let elapsed = changed_at.elapsed();
            if elapsed >= delay {
                live_extract = true;
            } else {
                ctx.request_repaint_after(delay.saturating_sub(elapsed));
            }
        }
        if re_extract_requested || live_extract {
            self.re_extract(&ctx);
        } else if overlay_changed || self.teach_changed {
            self.refresh_preview_overlay(&ctx);
        }
    }
}

fn ensure_extension(path: PathBuf, extension: &str) -> PathBuf {
    if path.extension().is_some() {
        path
    } else {
        path.with_extension(extension)
    }
}

pub(crate) fn wrap_angle(angle: f32) -> f32 {
    let wrapped =
        (angle + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI;
    if wrapped <= -std::f32::consts::PI {
        wrapped + std::f32::consts::TAU
    } else {
        wrapped
    }
}

pub(crate) fn image_to_screen(p: [f32; 2], rect: Rect, scale: f32) -> Pos2 {
    rect.min + egui::vec2(p[0] * scale, p[1] * scale)
}

pub(crate) fn screen_to_image(pos: Pos2, rect: Rect, scale: f32) -> [f32; 2] {
    [(pos.x - rect.left()) / scale, (pos.y - rect.top()) / scale]
}

fn rotation_handle_image(roi: &RotatedRect) -> [f32; 2] {
    let (sin, cos) = roi.angle.sin_cos();
    let v = [-sin, cos];
    let top = [
        roi.cx - v[0] * roi.height * 0.5,
        roi.cy - v[1] * roi.height * 0.5,
    ];
    let offset = 26.0f32.max(roi.width.min(roi.height) * 0.6);
    [top[0] - v[0] * offset, top[1] - v[1] * offset]
}

fn zero_handle_image(zero: &ZeroFrame) -> [f32; 2] {
    let (sin, cos) = zero.angle.sin_cos();
    [zero.origin[0] + cos * 40.0, zero.origin[1] + sin * 40.0]
}

fn draw_roi(painter: &egui::Painter, rect: Rect, scale: f32, roi: &RotatedRect) {
    draw_rect_editor(painter, rect, scale, roi, Color32::from_rgb(255, 200, 60));
}

/// Draw a rotated rectangle with corner and rotation handles.
pub(crate) fn draw_rect_editor(
    painter: &egui::Painter,
    rect: Rect,
    scale: f32,
    roi: &RotatedRect,
    color: Color32,
) {
    let corners = roi.corners();
    let screen: Vec<Pos2> = corners
        .iter()
        .map(|&corner| image_to_screen(corner, rect, scale))
        .collect();
    for index in 0..4 {
        painter.line_segment(
            [screen[index], screen[(index + 1) % 4]],
            Stroke::new(1.5, color),
        );
        painter.rect_filled(
            Rect::from_center_size(screen[index], egui::vec2(7.0, 7.0)),
            1.0,
            color,
        );
    }
    let top_center = image_to_screen(
        [
            (corners[0][0] + corners[1][0]) * 0.5,
            (corners[0][1] + corners[1][1]) * 0.5,
        ],
        rect,
        scale,
    );
    let rotation = image_to_screen(rotation_handle_image(roi), rect, scale);
    painter.line_segment([top_center, rotation], Stroke::new(1.0, color));
    painter.circle_filled(rotation, 4.0, color);
}

/// Edit a rotated rectangle with drag, resize, and rotation handles at
/// `MIN_ROI_SIZE` granularity.  Returns whether the geometry changed.
pub(crate) fn edit_rotated_rect(
    rect: Rect,
    scale: f32,
    response: &egui::Response,
    pointer: Option<[f32; 2]>,
    target: &mut Option<RotatedRect>,
    drag: &mut Option<RoiDrag>,
) -> bool {
    let Some(point) = pointer else {
        if response.drag_stopped() {
            *drag = None;
        }
        return false;
    };
    let screen = response
        .interact_pointer_pos()
        .expect("pointer position exists");
    let mut changed = false;

    if response.drag_started() {
        if let Some(roi) = *target {
            let corners = roi
                .corners()
                .map(|corner| image_to_screen(corner, rect, scale));
            if let Some(index) = corners
                .iter()
                .position(|corner| corner.distance(screen) <= HANDLE_HIT_RADIUS)
            {
                *drag = Some(RoiDrag::Resize {
                    anchor: roi.corners()[(index + 2) % 4],
                });
            } else if image_to_screen(rotation_handle_image(&roi), rect, scale).distance(screen)
                <= HANDLE_HIT_RADIUS
            {
                *drag = Some(RoiDrag::Rotate);
            } else if roi.contains(point[0], point[1]) {
                *drag = Some(RoiDrag::Move { last: point });
            } else {
                *drag = Some(RoiDrag::Create { start: point });
                *target = Some(RotatedRect::new(
                    point[0],
                    point[1],
                    MIN_ROI_SIZE,
                    MIN_ROI_SIZE,
                    0.0,
                ));
                changed = true;
            }
        } else {
            *drag = Some(RoiDrag::Create { start: point });
            *target = Some(RotatedRect::new(
                point[0],
                point[1],
                MIN_ROI_SIZE,
                MIN_ROI_SIZE,
                0.0,
            ));
            changed = true;
        }
    }

    if response.dragged() {
        match *drag {
            Some(RoiDrag::Create { start }) => {
                let width = (point[0] - start[0]).abs().max(MIN_ROI_SIZE);
                let height = (point[1] - start[1]).abs().max(MIN_ROI_SIZE);
                *target = Some(RotatedRect::new(
                    (start[0] + point[0]) * 0.5,
                    (start[1] + point[1]) * 0.5,
                    width,
                    height,
                    0.0,
                ));
                changed = true;
            }
            Some(RoiDrag::Move { last }) => {
                if let Some(roi) = target {
                    roi.cx += point[0] - last[0];
                    roi.cy += point[1] - last[1];
                    changed = true;
                }
                *drag = Some(RoiDrag::Move { last: point });
            }
            Some(RoiDrag::Resize { anchor }) => {
                if let Some(roi) = target {
                    let (sin, cos) = roi.angle.sin_cos();
                    let u = [cos, sin];
                    let v = [-sin, cos];
                    let d = [point[0] - anchor[0], point[1] - anchor[1]];
                    let du = d[0] * u[0] + d[1] * u[1];
                    let dv = d[0] * v[0] + d[1] * v[1];
                    roi.width = du.abs().max(MIN_ROI_SIZE);
                    roi.height = dv.abs().max(MIN_ROI_SIZE);
                    roi.cx = anchor[0] + u[0] * du * 0.5 + v[0] * dv * 0.5;
                    roi.cy = anchor[1] + u[1] * du * 0.5 + v[1] * dv * 0.5;
                    changed = true;
                }
            }
            Some(RoiDrag::Rotate) => {
                if let Some(roi) = target {
                    roi.angle = wrap_angle(
                        (point[1] - roi.cy).atan2(point[0] - roi.cx) + std::f32::consts::FRAC_PI_2,
                    );
                    changed = true;
                }
            }
            None => {}
        }
    }

    if response.drag_stopped() {
        if matches!(*drag, Some(RoiDrag::Create { .. }))
            && let Some(roi) = *target
            && (roi.width < MIN_ROI_SIZE || roi.height < MIN_ROI_SIZE)
        {
            *target = None;
            changed = true;
        }
        *drag = None;
    }
    changed
}

fn draw_zero(painter: &egui::Painter, rect: Rect, scale: f32, zero: &ZeroFrame) {
    let color = Color32::from_rgb(80, 220, 120);
    let origin = image_to_screen(zero.origin, rect, scale);
    painter.line_segment(
        [origin - egui::vec2(9.0, 0.0), origin + egui::vec2(9.0, 0.0)],
        Stroke::new(1.5, color),
    );
    painter.line_segment(
        [origin - egui::vec2(0.0, 9.0), origin + egui::vec2(0.0, 9.0)],
        Stroke::new(1.5, color),
    );
    let handle = image_to_screen(zero_handle_image(zero), rect, scale);
    painter.line_segment([origin, handle], Stroke::new(1.5, color));
    painter.circle_filled(handle, 4.0, color);
    // Arrow head at the x-axis handle.
    let direction = (handle - origin).normalized();
    let normal = egui::vec2(-direction.y, direction.x);
    painter.line_segment(
        [handle, handle - direction * 9.0 + normal * 4.0],
        Stroke::new(1.5, color),
    );
    painter.line_segment(
        [handle, handle - direction * 9.0 - normal * 4.0],
        Stroke::new(1.5, color),
    );
}

#[allow(clippy::too_many_arguments)]
fn draw_mask_overlay(
    painter: &egui::Painter,
    rect: Rect,
    scale: f32,
    primitives: &[MaskPrimitive],
    zero: &ZeroFrame,
    current_add: bool,
    brush_stroke: Option<&[[f32; 2]]>,
    brush_radius: f32,
    polygon_points: &[[f32; 2]],
) {
    for primitive in primitives {
        draw_mask_primitive(painter, rect, scale, zero, primitive);
    }
    if !polygon_points.is_empty() {
        let color = mask_color(current_add, 190);
        let points: Vec<Pos2> = polygon_points
            .iter()
            .map(|&point| image_to_screen(point, rect, scale))
            .collect();
        for pair in points.windows(2) {
            painter.line_segment([pair[0], pair[1]], Stroke::new(1.5, color));
        }
        for point in &points {
            painter.circle_filled(*point, 3.0, color);
        }
        if let Some(first) = points.first()
            && let Some(last) = points.last()
            && points.len() >= 3
        {
            painter.line_segment([*last, *first], Stroke::new(1.0, color.gamma_multiply(0.5)));
        }
    }
    if let Some(stroke) = brush_stroke {
        draw_brush_stroke(
            painter,
            rect,
            scale,
            stroke,
            brush_radius,
            mask_color(current_add, 160),
        );
    }
}

fn draw_mask_primitive(
    painter: &egui::Painter,
    rect: Rect,
    scale: f32,
    zero: &ZeroFrame,
    primitive: &MaskPrimitive,
) {
    let color = mask_color(primitive.is_add(), 180);
    let fill = mask_color(primitive.is_add(), 45);
    match primitive {
        MaskPrimitive::Polygon { points, .. } => {
            let screen: Vec<Pos2> = points
                .iter()
                .map(|&point| image_to_screen(zero.to_image(point), rect, scale))
                .collect();
            if screen.len() >= 3 {
                painter.add(egui::Shape::convex_polygon(
                    screen.clone(),
                    fill,
                    Stroke::new(1.2, color),
                ));
                painter.add(egui::Shape::closed_line(screen, Stroke::new(1.2, color)));
            }
        }
        MaskPrimitive::Brush { points, radius, .. } => {
            let image_points: Vec<[f32; 2]> =
                points.iter().map(|&point| zero.to_image(point)).collect();
            draw_brush_stroke(painter, rect, scale, &image_points, *radius, color);
        }
    }
}

fn draw_brush_stroke(
    painter: &egui::Painter,
    rect: Rect,
    scale: f32,
    points: &[[f32; 2]],
    radius: f32,
    color: Color32,
) {
    let screen: Vec<Pos2> = points
        .iter()
        .map(|&point| image_to_screen(point, rect, scale))
        .collect();
    let width = (radius * 2.0 * scale).max(1.0);
    if screen.len() == 1 {
        painter.circle_filled(screen[0], width * 0.5, color);
    }
    for pair in screen.windows(2) {
        painter.line_segment([pair[0], pair[1]], Stroke::new(width, color));
        painter.circle_filled(pair[1], width * 0.5, color);
    }
    if let Some(first) = screen.first() {
        painter.circle_filled(*first, width * 0.5, color);
    }
}

fn mask_color(add: bool, alpha: u8) -> Color32 {
    if add {
        Color32::from_rgba_unmultiplied(230, 60, 60, alpha)
    } else {
        Color32::from_rgba_unmultiplied(60, 120, 230, alpha)
    }
}

pub(crate) fn texture_from_rgba(
    ctx: &egui::Context,
    name: &str,
    image: &RgbaImage,
) -> egui::TextureHandle {
    let size = [image.width() as usize, image.height() as usize];
    let color = egui::ColorImage::from_rgba_unmultiplied(size, image.as_raw());
    ctx.load_texture(name, color, egui::TextureOptions::LINEAR)
}

fn draw_orientation_legend(ui: &mut egui::Ui) {
    let width = ui.available_width();
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, 14.0), egui::Sense::hover());
    let painter = ui.painter();
    let steps = 48;
    for step in 0..steps {
        let t = step as f32 / steps as f32;
        let color = pm_core::overlay::orientation_color(t * std::f32::consts::PI);
        let x0 = rect.left() + rect.width() * t;
        let x1 = rect.left() + rect.width() * (step + 1) as f32 / steps as f32;
        painter.rect_filled(
            Rect::from_min_max(
                egui::pos2(x0, rect.top()),
                egui::pos2(x1 + 0.5, rect.bottom()),
            ),
            0.0,
            Color32::from_rgb(color[0], color[1], color[2]),
        );
    }
    ui.horizontal(|ui| {
        ui.label("0°");
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label("180°");
        });
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn write_test_image() -> PathBuf {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let (width, height) = (96, 72);
        let mut data = vec![20u8; width * height];
        for y in 14..58 {
            for x in 14..82 {
                data[y * width + x] = 220;
            }
        }
        let image = image::GrayImage::from_raw(width as u32, height as u32, data).unwrap();
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("pm-ui-test-{}-{id}.png", std::process::id()));
        image.save(&path).unwrap();
        path
    }

    #[test]
    fn loading_runs_pipeline_and_builds_textures() {
        let path = write_test_image();
        let ctx = egui::Context::default();
        let mut app = PatternMatchApp::default();
        app.load(&ctx, path.clone());
        std::fs::remove_file(&path).ok();

        assert!(app.features.is_some());
        assert!(app.image_texture.is_some());
        assert!(app.overlay_texture.is_some());
        let features = app.features.as_ref().unwrap();
        assert!(features.edge_count() > 0);
        assert!(features.corner_count() >= 4);
        assert!(features.segment_count() >= 4);
        assert!(app.extraction_ms >= 0.0);
    }

    #[test]
    fn teach_save_and_reopen_roundtrip() {
        let image_path = write_test_image();
        let model_path =
            std::env::temp_dir().join(format!("pm-ui-m3-test-{}.pmodel", std::process::id()));
        let ctx = egui::Context::default();
        let mut app = PatternMatchApp::default();
        app.load(&ctx, image_path.clone());
        app.roi = Some(RotatedRect::new(48.0, 36.0, 70.0, 50.0, 0.15));
        app.zero = Some(ZeroFrame {
            origin: [48.0, 36.0],
            angle: 0.15,
        });
        app.mask_primitives = vec![MaskPrimitive::Polygon {
            points: vec![[-10.0, -12.0], [4.0, -12.0], [4.0, -4.0], [-10.0, -4.0]],
            add: true,
        }];
        app.model_name = "ui-test".to_owned();
        app.save_model(model_path.clone());
        std::fs::remove_file(&image_path).ok();
        assert!(app.model.is_some());
        let saved = app.model.as_ref().unwrap();
        let saved_features = saved.features.clone();
        let saved_mask = saved.mask.excluded.clone();

        app.open_model(&ctx, model_path.clone());
        std::fs::remove_file(&model_path).ok();
        assert_eq!(app.model_name, "ui-test");
        assert!(app.roi.is_some());
        assert!(app.zero.is_some());
        assert_eq!(app.mask_primitives.len(), 1);
        assert_eq!(app.model.as_ref().unwrap().features, saved_features);
        assert_eq!(app.model.as_ref().unwrap().mask.excluded, saved_mask);
        assert!(app.features.is_some());
        assert!(app.overlay_texture.is_some());
    }

    #[test]
    fn run_mode_matches_direct_detection_and_exports() {
        use pm_core::{DETECTION_CSV_HEADER, HypothesisSources, ResultsFile, SearchParams, detect};

        let image_path = write_test_image();
        let model_path =
            std::env::temp_dir().join(format!("pm-ui-m75-{}.pmodel", std::process::id()));
        let ctx = egui::Context::default();
        let mut app = PatternMatchApp::default();
        app.load(&ctx, image_path.clone());
        app.roi = Some(RotatedRect::new(48.0, 36.0, 70.0, 50.0, 0.0));
        app.zero = Some(ZeroFrame {
            origin: [48.0, 36.0],
            angle: 0.0,
        });
        app.model_name = "run-test".to_owned();
        app.save_model(model_path.clone());
        let model = app.model.clone().expect("model built");
        assert_eq!(
            app.run.params,
            SearchParams::from(model.search_defaults),
            "run parameters start from the model defaults"
        );

        // A tiny search range centred on the true pose keeps the unit test
        // fast while exercising the full worker/result/export path.
        app.run.params = SearchParams {
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
            ..SearchParams::from(model.search_defaults)
        };
        let loaded = app.loaded.clone().expect("teach image");
        app.run.set_image(&ctx, loaded.clone(), None);
        app.run.start(&ctx, &model).expect("start run");
        let deadline = Instant::now() + Duration::from_secs(120);
        while app.run.is_running() {
            let _ = app.run.poll(&ctx);
            assert!(Instant::now() < deadline, "run timed out");
            std::thread::sleep(Duration::from_millis(5));
        }
        let document = app.run.document().expect("run result").clone();
        assert!(
            !document.detections.is_empty(),
            "the taught image should match itself"
        );

        let mut params = app.run.params;
        params.search_region = app.run.search_region;
        let direct = detect(&model, &loaded.gray, &params).expect("direct detect");
        let expected = ResultsFile::new(
            model.name.clone(),
            loaded.gray.width,
            loaded.gray.height,
            &params,
            &direct,
        );
        assert_eq!(
            document.detections, expected.detections,
            "UI detections must match the library/CLI path"
        );

        let stem = format!("pm-ui-m75-{}", std::process::id());
        let json_path = std::env::temp_dir().join(format!("{stem}.json"));
        let csv_path = std::env::temp_dir().join(format!("{stem}.csv"));
        let png_path = std::env::temp_dir().join(format!("{stem}.png"));
        app.run.export_json(&json_path).expect("json export");
        app.run.export_csv(&csv_path).expect("csv export");
        app.run.save_overlay_png(&png_path).expect("overlay export");

        let json = std::fs::read_to_string(&json_path).expect("read json");
        assert_eq!(
            json,
            document.to_json().expect("serialize document"),
            "the exported JSON is the versioned result document"
        );
        let parsed = ResultsFile::from_json(&json).expect("parse json");
        assert_eq!(parsed.model, document.model);
        assert_eq!(parsed.detections.len(), document.detections.len());
        let csv = std::fs::read_to_string(&csv_path).expect("read csv");
        assert!(csv.starts_with(DETECTION_CSV_HEADER));
        assert_eq!(csv.lines().count(), document.detections.len() + 1);
        let overlay = image::open(&png_path).expect("open overlay");
        assert_eq!(overlay.width(), loaded.display.width());
        assert_eq!(overlay.height(), loaded.display.height());

        for path in [image_path, model_path, json_path, csv_path, png_path] {
            let _ = std::fs::remove_file(path);
        }
    }

    #[test]
    fn camera_frame_can_become_the_teach_source() {
        let (width, height) = (96, 72);
        let mut data = vec![0.12_f32; width * height];
        for y in 14..58 {
            for x in 14..82 {
                data[y * width + x] = 0.86;
            }
        }
        let gray = GrayF32::new(width, height, data).expect("test image");
        let frame = LoadedImage {
            display: gray.display_rgba(),
            gray,
        };

        let ctx = egui::Context::default();
        let mut app = PatternMatchApp::default();
        app.set_source_image(&ctx, frame, "camera".to_owned(), None);
        assert!(app.loaded.is_some());
        assert!(app.image_path.is_none());
        assert_eq!(app.model_name, "camera");
        assert!(app.features.as_ref().is_some_and(|f| f.edge_count() > 0));
        assert!(app.image_texture.is_some());
        assert!(app.overlay_texture.is_some());
        assert!(app.model.is_none(), "a new source clears the old model");

        app.roi = Some(RotatedRect::new(48.0, 36.0, 70.0, 50.0, 0.0));
        app.zero = Some(ZeroFrame {
            origin: [48.0, 36.0],
            angle: 0.0,
        });
        let path =
            std::env::temp_dir().join(format!("pm-ui-camera-teach-{}.pmodel", std::process::id()));
        app.save_model(path.clone());
        assert!(app.model.is_some(), "the in-memory frame can be taught");
        assert_eq!(app.model_name, "camera");
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn polygon_finish_uses_model_coordinates() {
        let mut app = PatternMatchApp {
            zero: Some(ZeroFrame {
                origin: [20.0, 10.0],
                angle: 0.0,
            }),
            polygon_points: vec![[20.0, 10.0], [24.0, 10.0], [24.0, 14.0]],
            ..PatternMatchApp::default()
        };
        app.finish_polygon();
        assert_eq!(app.mask_primitives.len(), 1);
        assert_eq!(
            app.mask_primitives[0].points()[0],
            [0.0, 0.0],
            "image point at the zero origin maps to model origin"
        );
        assert!(!app.undo_stack.is_empty());
        app.undo_mask();
        assert!(app.mask_primitives.is_empty());
        app.redo_mask();
        assert_eq!(app.mask_primitives.len(), 1);
    }
}
