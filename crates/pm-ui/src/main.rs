//! Pattern Match desktop application.
//!
//! M2 wires image loading, the default feature pipeline, and the feature
//! overlay.  Teaching tools (ROI, zero frame, masks) arrive in M3.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::path::PathBuf;
use std::time::Instant;

use eframe::egui;
use pm_core::{
    FeatureParams, FeatureSet, LoadedImage, OverlayOptions, RgbaImage, extract_features,
    load_image, render_overlay,
};

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
                app.load(&cc.egui_ctx, path);
            }
            Ok(Box::new(app))
        }),
    )
}

struct PatternMatchApp {
    loaded: Option<LoadedImage>,
    features: Option<FeatureSet>,
    image_texture: Option<egui::TextureHandle>,
    overlay_texture: Option<egui::TextureHandle>,
    params: FeatureParams,
    overlay: OverlayOptions,
    params_dirty: bool,
    extraction_ms: f64,
    status: String,
    image_path: Option<PathBuf>,
    cursor_px: Option<(f32, f32)>,
}

impl Default for PatternMatchApp {
    fn default() -> Self {
        Self {
            loaded: None,
            features: None,
            image_texture: None,
            overlay_texture: None,
            params: FeatureParams::default(),
            overlay: OverlayOptions::default(),
            params_dirty: false,
            extraction_ms: 0.0,
            status: "Drag an image here or use Open image.".to_owned(),
            image_path: None,
            cursor_px: None,
        }
    }
}

impl PatternMatchApp {
    fn load(&mut self, ctx: &egui::Context, path: PathBuf) {
        match load_image(&path) {
            Ok(loaded) => {
                self.image_texture = Some(texture_from_rgba(ctx, "source", &loaded.display));
                self.loaded = Some(loaded);
                self.image_path = Some(path);
                self.params_dirty = false;
                self.re_extract(ctx);
                self.status = "Image loaded.".to_owned();
            }
            Err(error) => {
                self.status = format!("Failed to load {}: {error}", path.display());
            }
        }
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
        self.update_overlay(ctx);
    }

    fn update_overlay(&mut self, ctx: &egui::Context) {
        let (Some(features), Some(loaded)) = (&self.features, &self.loaded) else {
            self.overlay_texture = None;
            return;
        };
        let overlay = render_overlay(
            loaded.display.width(),
            loaded.display.height(),
            features,
            &self.overlay,
        );
        self.overlay_texture = Some(texture_from_rgba(ctx, "overlay", &overlay));
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
            self.load(ctx, path);
        }
    }

    fn feature_counts(&self) -> (usize, usize, usize) {
        self.features.as_ref().map_or((0, 0, 0), |features| {
            (
                features.edge_count(),
                features.corner_count(),
                features.segment_count(),
            )
        })
    }
}

impl eframe::App for PatternMatchApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.handle_drop(&ctx);

        let mut open_requested = false;
        let mut re_extract_requested = false;
        let mut overlay_changed = false;

        egui::Panel::top("toolbar").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Pattern Match");
                ui.separator();
                if ui.button("Open image…").clicked() {
                    open_requested = true;
                }
                if ui
                    .add_enabled(self.loaded.is_some(), egui::Button::new("Re-extract"))
                    .clicked()
                {
                    re_extract_requested = true;
                }
                ui.separator();
                let mut layers_changed = false;
                layers_changed |= ui.checkbox(&mut self.overlay.draw_edges, "Edges").changed();
                layers_changed |= ui
                    .checkbox(&mut self.overlay.draw_corners, "Corners")
                    .changed();
                layers_changed |= ui
                    .checkbox(&mut self.overlay.draw_segments, "Segments")
                    .changed();
                overlay_changed = layers_changed;
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(format!("pm-core {}", pm_core::VERSION));
                });
            });
        });

        egui::Panel::left("model_panel")
            .resizable(true)
            .default_size(230.0)
            .show(ui, |ui| {
                ui.heading("Image");
                if let Some(path) = &self.image_path {
                    ui.label(path.file_name().map_or_else(
                        || path.display().to_string(),
                        |name| name.to_string_lossy().into_owned(),
                    ));
                } else {
                    ui.label("No image loaded.");
                }
                if let Some(loaded) = &self.loaded {
                    ui.label(format!("{} × {} px", loaded.gray.width, loaded.gray.height));
                }
                ui.separator();
                ui.heading("Features");
                let (edges, corners, segments) = self.feature_counts();
                ui.label(format!("Edge points: {edges}"));
                ui.label(format!(
                    "Chains: {}",
                    self.features.as_ref().map_or(0, FeatureSet::chain_count)
                ));
                ui.label(format!("Corners: {corners}"));
                ui.label(format!("Segments: {segments}"));
                ui.separator();
                ui.heading("Tools");
                ui.label("ROI, zero frame, and mask tools arrive in M3.");
                ui.separator();
                ui.heading("Orientation legend");
                draw_orientation_legend(ui);
            });

        egui::Panel::right("parameters_panel")
            .resizable(true)
            .default_size(280.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
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
                            egui::Slider::new(&mut self.params.edge.high_percentile, 0.5..=0.99)
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
                ui.label(format!("features {:.1} ms", self.extraction_ms));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(format!("pm-ui {}", env!("CARGO_PKG_VERSION")));
                });
            });
        });

        let cursor = egui::CentralPanel::default()
            .show(ui, |ui| {
                let Some(loaded) = &self.loaded else {
                    ui.centered_and_justified(|ui| {
                        ui.label("Drag an image here or use Open image…");
                    });
                    return None;
                };
                let available = ui.available_rect_before_wrap();
                let image_size = egui::vec2(
                    loaded.display.width() as f32,
                    loaded.display.height() as f32,
                );
                let scale = (available.width() / image_size.x)
                    .min(available.height() / image_size.y)
                    .max(1e-3);
                let rect = egui::Rect::from_center_size(available.center(), image_size * scale);
                let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
                let painter = ui.painter();
                if let Some(texture) = &self.image_texture {
                    painter.image(texture.id(), rect, uv, egui::Color32::WHITE);
                }
                if let Some(texture) = &self.overlay_texture {
                    painter.image(texture.id(), rect, uv, egui::Color32::WHITE);
                }
                let hover = ui.ctx().input(|input| input.pointer.hover_pos());
                hover
                    .filter(|pos| rect.contains(*pos))
                    .map(|pos| ((pos.x - rect.left()) / scale, (pos.y - rect.top()) / scale))
            })
            .inner;
        self.cursor_px = cursor;

        if open_requested
            && let Some(path) = rfd::FileDialog::new()
                .add_filter("Images", &["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
                .pick_file()
        {
            self.load(&ctx, path);
        }
        if re_extract_requested {
            self.re_extract(&ctx);
        } else if overlay_changed {
            self.update_overlay(&ctx);
        }
    }
}

fn texture_from_rgba(ctx: &egui::Context, name: &str, image: &RgbaImage) -> egui::TextureHandle {
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
            egui::Rect::from_min_max(
                egui::pos2(x0, rect.top()),
                egui::pos2(x1 + 0.5, rect.bottom()),
            ),
            0.0,
            egui::Color32::from_rgb(color[0], color[1], color[2]),
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

    #[test]
    fn loading_runs_pipeline_and_builds_textures() {
        let (width, height) = (64, 48);
        let mut data = vec![20u8; width * height];
        for y in 10..height - 10 {
            for x in 10..width - 10 {
                data[y * width + x] = 220;
            }
        }
        let image = image::GrayImage::from_raw(width as u32, height as u32, data).unwrap();
        let path = std::env::temp_dir().join(format!("pm-ui-load-test-{}.png", std::process::id()));
        image.save(&path).unwrap();

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
}
