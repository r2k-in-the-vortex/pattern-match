//! Pattern Match desktop application.
//!
//! M0 scaffold: the shell layout is in place, but no image, teaching, or
//! detection functionality is wired up yet (milestones M2–M6).

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use eframe::egui;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_title("Pattern Match"),
        ..Default::default()
    };

    eframe::run_native(
        "Pattern Match",
        options,
        Box::new(|_cc| Ok(Box::new(PatternMatchApp))),
    )
}

struct PatternMatchApp;

impl eframe::App for PatternMatchApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::Panel::top("toolbar").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Pattern Match");
                ui.separator();
                ui.label(format!("pm-core {}", pm_core::VERSION));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label("M0 scaffold");
                });
            });
        });

        egui::Panel::left("model_panel")
            .resizable(true)
            .default_size(220.0)
            .show(ui, |ui| {
                ui.heading("Model");
                ui.label("No image loaded.");
                ui.separator();
                ui.heading("Tools");
                ui.label("Teaching tools arrive in M3.");
            });

        egui::Panel::right("parameters_panel")
            .resizable(true)
            .default_size(260.0)
            .show(ui, |ui| {
                ui.heading("Parameters");
                ui.label("Search parameters arrive in M5.");
                ui.separator();
                ui.heading("Results");
                ui.label("No detections.");
            });

        egui::Panel::bottom("status_bar").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label("Ready");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(format!("pm-ui {}", env!("CARGO_PKG_VERSION")));
                });
            });
        });

        egui::CentralPanel::default().show(ui, |ui| {
            ui.centered_and_justified(|ui| {
                ui.label("Canvas — image loading arrives in M2.");
            });
        });
    }
}
