use egui::{Color32, RichText, Slider, SliderClamping};
use egui_plot::{
    AxisHints, Bar, BarChart, GridMark, HLine, HPlacement, HoverPosition, Legend, Line, LineStyle,
    Plot, PlotPoints,
};
use std::collections::VecDeque;

use crate::helpers::is_meter_overload;
use crate::multimeter::MeterMode;

// Configuration for graph settings
#[derive(Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GraphConfig {
    pub num_bins: usize, // Number of bins for histogram, 0 for auto
    pub max_bins: usize, // Maximum number of bins for slider
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            num_bins: 0,   // 0 means auto
            max_bins: 100, // Default maximum bins
        }
    }
}

type PlotRun = Vec<[f64; 2]>;

/// Upper bound on points drawn per frame; larger windows are min/max decimated.
const MAX_DRAW_POINTS: usize = 6000;

/// Split samples into measurement runs and overload runs.
/// OL points use the last (or first) valid Y so they stay in axis range; they
/// are never NaN (egui panics on NaN paths).
fn split_meas_and_ol(xs: &[f64], ys: &[f64], mode: MeterMode) -> (Vec<PlotRun>, Vec<PlotRun>) {
    let fallback = ys
        .iter()
        .copied()
        .find(|y| y.is_finite() && !is_meter_overload(*y, mode))
        .unwrap_or(0.0);
    let mut meas_runs: Vec<PlotRun> = Vec::new();
    let mut ol_runs: Vec<PlotRun> = Vec::new();
    let mut meas: PlotRun = Vec::new();
    let mut ol: PlotRun = Vec::new();
    let mut last_valid = fallback;
    for (&x, &y) in xs.iter().zip(ys) {
        if !y.is_finite() {
            if !meas.is_empty() {
                meas_runs.push(std::mem::take(&mut meas));
            }
            if !ol.is_empty() {
                ol_runs.push(std::mem::take(&mut ol));
            }
        } else if is_meter_overload(y, mode) {
            if !meas.is_empty() {
                meas_runs.push(std::mem::take(&mut meas));
            }
            ol.push([x, last_valid]);
        } else {
            if !ol.is_empty() {
                ol_runs.push(std::mem::take(&mut ol));
            }
            last_valid = y;
            meas.push([x, y]);
        }
    }
    if !meas.is_empty() {
        meas_runs.push(meas);
    }
    if !ol.is_empty() {
        ol_runs.push(ol);
    }
    (meas_runs, ol_runs)
}

/// One screen-space segment per OL run so dash length is even in pixels.
/// Per-sample vertices would restart the dash pattern on serial jitter.
fn flatten_ol_run(run: &[[f64; 2]], half_gap: f64) -> PlotRun {
    match run {
        [] => Vec::new(),
        [p] => vec![[p[0] - half_gap, p[1]], [p[0] + half_gap, p[1]]],
        [first, .., last] => vec![*first, *last],
    }
}

/// View state of the DMM line graph. Not persisted: times restart with the app.
#[derive(Debug)]
pub struct GraphView {
    /// Right edge tracks the newest sample.
    pub follow: bool,
    /// Visible time window, seconds.
    pub x_span: f64,
    /// Left edge when not following.
    pub x_min: f64,
    /// Y range tracks the visible data.
    pub auto_y: bool,
    pub y_min: f64,
    pub y_max: f64,
    /// Auto Y has been fitted to data at least once (first fit jumps, later ones ease).
    y_seeded: bool,
    /// Result of the last save, shown next to the buttons.
    pub status: String,
    /// Pending background save; resolves to a status message.
    save_rx: Option<std::sync::mpsc::Receiver<String>>,
}

impl Default for GraphView {
    fn default() -> Self {
        Self {
            follow: true,
            x_span: 30.0,
            x_min: 0.0,
            auto_y: true,
            y_min: -1.0,
            y_max: 1.0,
            y_seeded: false,
            status: String::new(),
            save_rx: None,
        }
    }
}

impl GraphView {
    fn zoom_x(&mut self, factor: f64) {
        let span = (self.x_span * factor).clamp(1e-3, 1e8);
        if !self.follow {
            self.x_min += (self.x_span - span) / 2.0;
        }
        self.x_span = span;
    }

    fn zoom_y(&mut self, factor: f64) {
        let centre = (self.y_min + self.y_max) / 2.0;
        let half = ((self.y_max - self.y_min) / 2.0 * factor).max(1e-12);
        self.auto_y = false;
        self.y_min = centre - half;
        self.y_max = centre + half;
    }
}

/// Newest-aligned view of the time and value buffers (the value buffer can be
/// longer, e.g. after a PSU session; the tail always pairs up).
struct Series<'a> {
    times: &'a VecDeque<f64>,
    values: &'a VecDeque<f64>,
    off_t: usize,
    off_v: usize,
    len: usize,
}

impl<'a> Series<'a> {
    fn new(times: &'a VecDeque<f64>, values: &'a VecDeque<f64>) -> Self {
        let len = times.len().min(values.len());
        Self {
            times,
            values,
            off_t: times.len() - len,
            off_v: values.len() - len,
            len,
        }
    }

    fn t(&self, i: usize) -> f64 {
        self.times[self.off_t + i]
    }

    fn v(&self, i: usize) -> f64 {
        self.values[self.off_v + i]
    }

    /// Sample index range covering `[lo, hi]`, padded by one sample each side.
    fn window(&self, lo: f64, hi: f64) -> (usize, usize) {
        let first = self
            .times
            .partition_point(|&t| t < lo)
            .saturating_sub(self.off_t)
            .saturating_sub(1);
        let last = (self
            .times
            .partition_point(|&t| t <= hi)
            .saturating_sub(self.off_t)
            + 1)
        .min(self.len);
        (first.min(last), last)
    }

    /// Samples in `[i0, i1)`. When there are more than `max_points`, each
    /// `bucket_dt`-second bucket keeps its min, max and first overload sample.
    /// Buckets are aligned to absolute time so the trace does not shimmer as
    /// the window slides.
    fn decimated(
        &self,
        i0: usize,
        i1: usize,
        mode: MeterMode,
        max_points: usize,
        bucket_dt: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        if i1 - i0 <= max_points || bucket_dt <= 0.0 {
            for i in i0..i1 {
                xs.push(self.t(i));
                ys.push(self.v(i));
            }
            return (xs, ys);
        }
        let mut flush = |picks: &mut [Option<usize>; 3]| {
            let mut idx: Vec<usize> = picks.iter().flatten().copied().collect();
            idx.sort_unstable();
            idx.dedup();
            for i in idx {
                xs.push(self.t(i));
                ys.push(self.v(i));
            }
            *picks = [None; 3];
        };
        // [min, max, first overload] of the current bucket
        let mut picks: [Option<usize>; 3] = [None; 3];
        let mut key = i64::MIN;
        for i in i0..i1 {
            let k = (self.t(i) / bucket_dt).floor() as i64;
            if k != key {
                flush(&mut picks);
                key = k;
            }
            let y = self.v(i);
            if !y.is_finite() || is_meter_overload(y, mode) {
                picks[2].get_or_insert(i);
                continue;
            }
            if picks[0].is_none_or(|j| y < self.v(j)) {
                picks[0] = Some(i);
            }
            if picks[1].is_none_or(|j| y > self.v(j)) {
                picks[1] = Some(i);
            }
        }
        flush(&mut picks);
        (xs, ys)
    }
}

/// Round up to 1, 2 or 5 times a power of ten.
fn nice_step(x: f64) -> f64 {
    if !(x.is_finite() && x > 0.0) {
        return 0.0;
    }
    let mag = 10f64.powf(x.log10().floor());
    [1.0, 2.0, 5.0, 10.0]
        .into_iter()
        .map(|m| m * mag)
        .find(|s| *s >= x)
        .unwrap_or(10.0 * mag)
}

/// Padded min/max of the plottable values, or `None` when there are none.
fn auto_y_range(ys: &[f64], mode: MeterMode) -> Option<(f64, f64)> {
    let (lo, hi) = ys
        .iter()
        .copied()
        .filter(|y| y.is_finite() && !is_meter_overload(*y, mode))
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), y| {
            (lo.min(y), hi.max(y))
        });
    if lo > hi {
        return None;
    }
    // Padded, and never thinner than 0.2 % of the level (or 1e-6), so meter noise
    // on a steady signal is not blown up to full height.
    let centre = (lo + hi) / 2.0;
    let half = ((hi - lo) * 0.55).max(centre.abs() * 0.001).max(5e-7);
    Some((centre - half, centre + half))
}

/// Clock time for an axis tick, with the precision the visible span needs.
fn format_clock(t: f64, wall0: f64, span: f64) -> String {
    use chrono::TimeZone;
    let ms = ((wall0 + t) * 1000.0).round() as i64;
    let Some(dt) = chrono::Local.timestamp_millis_opt(ms).single() else {
        return format!("{t:.1}");
    };
    let clock = dt.format("%H:%M:%S").to_string();
    let millis = ms.rem_euclid(1000);
    if span < 2.0 {
        format!("{clock}.{millis:03}")
    } else if span < 20.0 {
        format!("{clock}.{}", millis / 100)
    } else {
        clock
    }
}

fn unix_now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64())
}

/// Write `(time, value)` rows as CSV. Returns the number of rows written.
#[cfg(not(target_arch = "wasm32"))]
fn save_graph_csv(
    path: &std::path::Path,
    rows: &[(f64, f64)],
    unit: &str,
    mode: MeterMode,
    wall0: f64,
) -> Result<usize, String> {
    let mut writer = csv::WriterBuilder::new()
        .from_path(path)
        .map_err(|e| e.to_string())?;
    writer
        .write_record(["timestamp", "seconds", "value", "unit"])
        .map_err(|e| e.to_string())?;
    for &(t, y) in rows {
        let stamp = chrono::DateTime::from_timestamp_millis(((wall0 + t) * 1000.0).round() as i64)
            .map(|d| d.to_rfc3339())
            .unwrap_or_default();
        let value = if is_meter_overload(y, mode) {
            "OL".to_owned()
        } else {
            y.to_string()
        };
        writer
            .write_record([stamp, format!("{t:.3}"), value, unit.to_owned()])
            .map_err(|e| e.to_string())?;
    }
    writer.flush().map_err(|e| e.to_string())?;
    Ok(rows.len())
}

/// Ask for a path and write the rows off the UI thread, so sampling and
/// repaint carry on while the dialog is open. The receiver gets a status line
/// (empty when the dialog was cancelled).
#[cfg(not(target_arch = "wasm32"))]
fn spawn_csv_save(
    ctx: egui::Context,
    rows: Vec<(f64, f64)>,
    unit: String,
    mode: MeterMode,
    wall0: f64,
) -> std::sync::mpsc::Receiver<String> {
    let (tx, rx) = std::sync::mpsc::channel();
    tokio::spawn(async move {
        let picked = rfd::AsyncFileDialog::new()
            .add_filter("CSV", &["csv"])
            .set_file_name("graph.csv")
            .save_file()
            .await;
        let msg = match picked {
            None => String::new(),
            Some(file) => {
                let path = file.path().to_path_buf();
                let done = tokio::task::spawn_blocking(move || {
                    save_graph_csv(&path, &rows, &unit, mode, wall0)
                })
                .await;
                match done {
                    Ok(Ok(n)) => format!("Saved {n} samples"),
                    Ok(Err(e)) => format!("Save failed: {e}"),
                    Err(e) => format!("Save failed: {e}"),
                }
            }
        };
        let _ = tx.send(msg);
        ctx.request_repaint();
    });
    rx
}

#[allow(clippy::too_many_arguments)]
pub fn show_line_graph(
    ui: &mut egui::Ui,
    values: &mut VecDeque<f64>,
    times: &mut VecDeque<f64>,
    view: &mut GraphView,
    reverse_graph: bool,
    graph_line_color: Color32,
    mem_depth: &mut usize,
    graph_update_interval_ms: &mut u64,
    reverse_graph_mut: &mut bool,
    mem_depth_max: usize,
    graph_update_interval_max: u64,
    curr_unit: &str,
    metermode: MeterMode,
) {
    // Prefixed unit text uses the mode's base unit; `curr_unit` can already carry a prefix.
    let base_unit: String = if metermode == MeterMode::Temp && !curr_unit.is_empty() {
        curr_unit.to_owned()
    } else {
        metermode.default_unit().to_owned()
    };
    let si = crate::helpers::mode_takes_si_prefix(&metermode);
    if let Some(rx) = &view.save_rx
        && let Ok(msg) = rx.try_recv()
    {
        view.status = msg;
        view.save_rx = None;
    }
    let (now, dt) = ui.ctx().input(|i| (i.time, f64::from(i.stable_dt)));
    let wall0 = unix_now() - now;

    let series = Series::new(times, values);
    let t_last = if series.len > 0 {
        series.t(series.len - 1)
    } else {
        0.0
    };
    let t_first = if series.len > 0 { series.t(0) } else { 0.0 };
    if series.len == 0 {
        view.y_seeded = false;
    }

    // Window to draw. Following tracks the clock so it scrolls smoothly.
    let span = view.x_span.max(1e-3);
    let (x_lo, x_hi) = if view.follow {
        (now - span * 0.99, now + span * 0.01)
    } else {
        (view.x_min, view.x_min + span)
    };
    let (i0, i1) = series.window(x_lo, x_hi);
    let bucket_dt = nice_step(span / (MAX_DRAW_POINTS / 3) as f64);
    let (xs, ys) = series.decimated(i0, i1, metermode, MAX_DRAW_POINTS, bucket_dt);

    // Y range: auto eases towards the data (grows at once, shrinks gradually).
    if view.auto_y
        && let Some((tl, th)) = auto_y_range(&ys, metermode)
    {
        if view.y_seeded {
            let k = 1.0 - (-dt * 8.0).exp();
            view.y_min = if tl < view.y_min {
                tl
            } else {
                view.y_min + (tl - view.y_min) * k
            };
            view.y_max = if th > view.y_max {
                th
            } else {
                view.y_max + (th - view.y_max) * k
            };
        } else {
            view.y_min = tl;
            view.y_max = th;
            view.y_seeded = true;
        }
    }
    if view.y_max - view.y_min < 1e-12 {
        view.y_max = view.y_min + 1e-12;
    }
    let (y_lo, y_hi) = (view.y_min, view.y_max);

    // Is anything of the trace on screen?
    let plottable = |y: &f64| y.is_finite() && !is_meter_overload(*y, metermode);
    let on_screen = ys.iter().any(|y| plottable(y) && *y >= y_lo && *y <= y_hi);
    let lost = !on_screen && ys.iter().any(plottable);
    let no_data_in_view = series.len > 0 && xs.is_empty();

    let gap = if xs.len() > 1 {
        (xs[xs.len() - 1] - xs[0]) / xs.len() as f64 * 0.4
    } else {
        span * 0.005
    };
    let (meas_runs, ol_runs) = split_meas_and_ol(&xs, &ys, metermode);

    let mut reset = false;
    ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
        // Bottom up: buffer row is lowest, view row above it.
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().slider_width = 120.0;
            ui.label("Buffer");
            ui.add(
                Slider::new(mem_depth, 10..=mem_depth_max)
                    .text("samples")
                    .logarithmic(true)
                    .clamping(SliderClamping::Always),
            );
            ui.add(
                Slider::new(graph_update_interval_ms, 10..=graph_update_interval_max)
                    .text("ms/sample")
                    .step_by(10.0)
                    .clamping(SliderClamping::Always),
            );
            reset |= ui.button("Clear").clicked();
            ui.checkbox(reverse_graph_mut, "Reverse time");
        });
        ui.horizontal_wrapped(|ui| {
            // Time
            ui.label("Time");
            if ui
                .selectable_label(view.follow, "Live")
                .on_hover_text("Follow the present (the right edge)")
                .clicked()
            {
                view.follow = true;
            }
            ui.add_sized(
                [72.0, 20.0],
                egui::DragValue::new(&mut view.x_span)
                    .speed(span * 0.01)
                    .range(1e-3..=1e8)
                    .custom_formatter(|n, _| crate::helpers::format_si(n, "s"))
                    .custom_parser(|s| crate::helpers::parse_si(s, "s")),
            )
            .on_hover_text("Visible time span");
            if ui
                .small_button("−")
                .on_hover_text("Zoom time out")
                .clicked()
            {
                view.zoom_x(2.0);
            }
            if ui.small_button("+").on_hover_text("Zoom time in").clicked() {
                view.zoom_x(0.5);
            }
            ui.separator();
            // Value
            ui.label("Value");
            if ui
                .selectable_label(view.auto_y, "Auto")
                .on_hover_text("Fit the value axis to the visible data")
                .clicked()
            {
                view.auto_y = true;
            }
            let speed = (view.y_max - view.y_min).abs() * 0.005;
            for (v, hint) in [(&mut view.y_min, "Bottom"), (&mut view.y_max, "Top")] {
                let unit_f = base_unit.clone();
                let unit_p = base_unit.clone();
                let r = ui.add_sized(
                    [92.0, 20.0],
                    egui::DragValue::new(v)
                        .speed(speed)
                        .custom_formatter(move |n, _| {
                            if si {
                                crate::helpers::format_si(n, &unit_f)
                            } else {
                                format!("{n:.3} {unit_f}")
                            }
                        })
                        .custom_parser(move |s| crate::helpers::parse_si(s, &unit_p)),
                );
                if r.changed() {
                    view.auto_y = false;
                }
                r.on_hover_text(hint);
            }
            if ui
                .small_button("−")
                .on_hover_text("Zoom value out")
                .clicked()
            {
                view.zoom_y(2.0);
            }
            if ui
                .small_button("+")
                .on_hover_text("Zoom value in")
                .clicked()
            {
                view.zoom_y(0.5);
            }
            ui.separator();
            if ui
                .button("Fit all")
                .on_hover_text("Show the whole buffer")
                .clicked()
                && series.len > 0
            {
                view.follow = false;
                view.x_min = t_first;
                view.x_span = (t_last - t_first).max(1e-3);
                view.auto_y = true;
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                let can_save = series.len > 0 && view.save_rx.is_none();
                ui.add_enabled_ui(can_save, |ui| {
                    ui.menu_button("Save…", |ui| {
                        for (label, whole) in [
                            ("Whole buffer (CSV)", true),
                            ("Visible window (CSV)", false),
                        ] {
                            if ui.button(label).clicked() {
                                ui.close();
                                let (a, b) = if whole {
                                    (0, series.len)
                                } else {
                                    series.window(x_lo, x_hi)
                                };
                                let rows: Vec<(f64, f64)> =
                                    (a..b).map(|i| (series.t(i), series.v(i))).collect();
                                view.save_rx = Some(spawn_csv_save(
                                    ui.ctx().clone(),
                                    rows,
                                    base_unit.clone(),
                                    metermode,
                                    wall0,
                                ));
                                view.status = "Saving…".to_owned();
                            }
                        }
                    });
                });
            }
            ui.label("?").on_hover_text(
                "Wheel: zoom time\nShift+wheel: scroll time\nCtrl+wheel: zoom value\n\
                 Drag: scroll time (and value when Auto is off)\nDrag an axis: zoom just that axis",
            );
            if !view.status.is_empty() {
                ui.weak(&view.status);
            }
        });
        ui.separator();

        let hover_unit = base_unit.clone();
        let plot = Plot::new("graph")
            .legend(Legend::default().text_style(egui::TextStyle::Monospace))
            .y_axis_min_width(4.0)
            .y_axis_label(curr_unit)
            .x_axis_label("Time")
            .invert_x(reverse_graph)
            // Wheel and box-zoom are handled below so a stray scroll cannot lose the trace.
            .allow_scroll(false)
            .allow_zoom(false)
            .allow_boxed_zoom(false)
            .allow_double_click_reset(false)
            .allow_drag([true, !view.auto_y])
            .x_axis_formatter(move |mark, range| {
                format_clock(mark.value, wall0, range.end() - range.start())
            })
            .y_axis_formatter(move |mark, range| {
                crate::helpers::format_si_tick(mark.value, range.end() - range.start(), si)
            })
            .label_formatter(move |pos| {
                let (y, x) = match pos {
                    HoverPosition::NearDataPoint { position, .. } => (position.y, position.x),
                    HoverPosition::Elsewhere { position } => (position.y, position.x),
                };
                let value = if si {
                    crate::helpers::format_si(y, &hover_unit)
                } else {
                    format!("{y:.3} {hover_unit}")
                };
                Some(format!("{value}\n{}", format_clock(x, wall0, 1.0)))
            })
            .show_axes(true)
            .show_grid(true);

        let response = plot.show(ui, |plot_ui| {
            plot_ui.set_plot_bounds(egui_plot::PlotBounds::from_min_max(
                [x_lo, y_lo],
                [x_hi, y_hi],
            ));
            plot_ui.set_auto_bounds([false, false]);
            for (i, run) in meas_runs.into_iter().enumerate() {
                let name = if i == 0 { curr_unit } else { "" };
                plot_ui.line(
                    Line::new(format!("meas{i}"), run)
                        .name(name)
                        .stroke(egui::Stroke::new(2.0, graph_line_color)),
                );
            }
            for (i, run) in ol_runs.into_iter().enumerate() {
                let name = if i == 0 { "OVERLOAD" } else { "" };
                plot_ui.line(
                    Line::new(format!("ol{i}"), flatten_ol_run(&run, gap))
                        .name(name)
                        .stroke(egui::Stroke::new(1.5, Color32::from_rgb(220, 50, 50)))
                        .style(LineStyle::Dashed { length: 3.0 }),
                );
            }
            let hint = if no_data_in_view {
                Some("No samples in this time window - press Live or Fit all")
            } else if lost {
                Some("Trace is outside the value range - press Auto")
            } else {
                None
            };
            if let Some(hint) = hint {
                plot_ui.text(
                    egui_plot::Text::new(
                        "hint",
                        egui_plot::PlotPoint::new((x_lo + x_hi) / 2.0, (y_lo + y_hi) / 2.0),
                        RichText::new(hint)
                            .color(Color32::from_rgb(230, 160, 40))
                            .size(15.0),
                    )
                    .name(""),
                );
            }
        });

        // Pull drag and axis-zoom changes back into the view state.
        let bounds = response.transform.bounds();
        let (nx_lo, nx_hi) = (bounds.min()[0], bounds.max()[0]);
        let (ny_lo, ny_hi) = (bounds.min()[1], bounds.max()[1]);
        let tol_x = (x_hi - x_lo).abs() * 1e-4;
        let tol_y = (y_hi - y_lo).abs() * 1e-4;
        if (nx_lo - x_lo).abs() > tol_x || (nx_hi - x_hi).abs() > tol_x {
            view.x_span = (nx_hi - nx_lo).max(1e-3);
            view.x_min = nx_lo;
            // Scrolling back to the present resumes following.
            view.follow = nx_hi >= now - view.x_span * 0.005;
        }
        if (ny_lo - y_lo).abs() > tol_y || (ny_hi - y_hi).abs() > tol_y {
            view.auto_y = false;
            view.y_min = ny_lo;
            view.y_max = ny_hi;
        }

        // Wheel: zoom time / scroll time / zoom value, around the cursor.
        if response.response.hovered() {
            let (scroll, zoom, cursor) =
                ui.input(|i| (i.smooth_scroll_delta, i.zoom_delta(), i.pointer.hover_pos()));
            let at = cursor.map(|p| response.transform.value_from_position(p));
            if scroll.y != 0.0 {
                let f = (-f64::from(scroll.y) * 0.004).exp();
                if view.follow {
                    view.zoom_x(f);
                } else if let Some(at) = at {
                    let new = (view.x_span * f).clamp(1e-3, 1e8);
                    view.x_min = at.x - (at.x - view.x_min) * new / view.x_span;
                    view.x_span = new;
                }
            }
            if scroll.x != 0.0 {
                let per_px = view.x_span / f64::from(response.response.rect.width().max(1.0));
                let dir = if reverse_graph { 1.0 } else { -1.0 };
                if view.follow {
                    view.x_min = now - view.x_span * 0.99;
                    view.follow = false;
                }
                view.x_min += dir * f64::from(scroll.x) * per_px;
                if view.x_min + view.x_span >= now - view.x_span * 0.005 {
                    view.follow = true;
                }
            }
            if zoom != 1.0 {
                let f = 1.0 / f64::from(zoom);
                let anchor = at.map_or((view.y_min + view.y_max) / 2.0, |p| p.y);
                view.auto_y = false;
                view.y_min = anchor - (anchor - view.y_min) * f;
                view.y_max = anchor + (view.y_max - anchor) * f;
            }
        }
    });
    if reset {
        values.clear();
        times.clear();
    }
}

#[derive(Clone, Copy)]
pub struct PsuGraphStyle {
    pub set_v: f64,
    pub set_i: f64,
    pub set_p: f64,
    pub color_v: Color32,
    pub color_i: Color32,
    pub color_p: Color32,
}

pub struct PsuGraph<'a> {
    pub volt: &'a mut VecDeque<f64>,
    pub curr: &'a mut VecDeque<f64>,
    pub power: &'a mut VecDeque<f64>,
    pub set_v: f64,
    pub set_i: f64,
    pub set_p: f64,
    pub color_v: Color32,
    pub color_i: Color32,
    pub color_p: Color32,
}

/// Place `set` at this fraction of plot height when it is the scale driver.
/// V / I / P use slightly different values so the three set-lines separate
/// without lying: a 5 V trace still meets the 5 V line.
const V_SET_AT: f64 = 1.00;
const I_SET_AT: f64 = 0.96;
const P_SET_AT: f64 = 0.92;

fn axis_span(set: f64, values: &VecDeque<f64>, set_at: f64) -> f64 {
    let data_max = values.iter().copied().fold(0.0_f64, f64::max);
    let set_at = set_at.max(1e-3);
    (set / set_at).max(data_max).max(1e-6)
}

fn scaled_points(values: &VecDeque<f64>, reverse: bool, span: f64) -> PlotPoints<'static> {
    let span = span.max(1e-9);
    let mut points: Vec<f64> = values.iter().map(|v| *v / span).collect();
    if reverse {
        points.reverse();
    }
    PlotPoints::from_ys_f64(&points)
}

fn psu_hline(name: &str, y: f64, color: Color32, style: LineStyle) -> HLine {
    HLine::new(name, y)
        .stroke(egui::Stroke::new(1.5, color))
        .style(style)
}

fn format_axis_tick(span: f64, mark: GridMark) -> String {
    let value = mark.value * span;
    let step = mark.step_size * span;
    let decimals = if step > 0.0 {
        (-step.log10().round() as i32).max(0) as usize
    } else {
        3
    };
    format!("{value:.decimals$}")
}

fn qty_axis(label: &str, color: Color32, span: f64, placement: HPlacement) -> AxisHints<'static> {
    AxisHints::new_y()
        .label(RichText::new(label).color(color))
        .placement(placement)
        .min_thickness(40.0)
        .tick_label_color(color)
        .formatter(move |mark, _| format_axis_tick(span, mark))
}

/// One plot: V / I / P overlay, each with its own Y axis (normalized to setpoint).
#[allow(clippy::too_many_arguments)]
pub fn show_psu_graphs(
    ui: &mut egui::Ui,
    data: PsuGraph<'_>,
    reverse_graph: bool,
    mem_depth: &mut usize,
    graph_update_interval_ms: &mut u64,
    reverse_graph_mut: &mut bool,
    mem_depth_max: usize,
    graph_update_interval_max: u64,
) {
    ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.add(
                Slider::new(mem_depth, 10..=mem_depth_max)
                    .text("Memory Depth")
                    .logarithmic(true)
                    .clamping(SliderClamping::Always),
            );
            ui.add(
                Slider::new(graph_update_interval_ms, 10..=graph_update_interval_max)
                    .text("Update Interval (ms)")
                    .step_by(10.0)
                    .clamping(SliderClamping::Always),
            );
            if ui.button("Reset Graph").clicked() {
                data.volt.clear();
                data.curr.clear();
                data.power.clear();
            }
            ui.checkbox(reverse_graph_mut, "Reverse Graph (most recent on left)");
        });
        ui.label("Graph Adjustments");
        ui.separator();

        let v_span = axis_span(data.set_v, data.volt, V_SET_AT);
        let i_span = axis_span(data.set_i, data.curr, I_SET_AT);
        let p_span = axis_span(data.set_p, data.power, P_SET_AT);

        let y_axes = vec![
            qty_axis("V", data.color_v, v_span, HPlacement::Left),
            qty_axis("A", data.color_i, i_span, HPlacement::Right),
            qty_axis("W", data.color_p, p_span, HPlacement::Right),
        ];

        Plot::new("psu_graph")
            .legend(Legend::default().text_style(egui::TextStyle::Monospace))
            .custom_y_axes(y_axes)
            .x_axis_label("Samples")
            .show_axes(true)
            .show_grid(true)
            .label_formatter(move |pos| match pos {
                HoverPosition::NearDataPoint {
                    plot_name,
                    position,
                    ..
                } => {
                    let y = match *plot_name {
                        "V" => format!("{:.3} V", position.y * v_span),
                        "A" => format!("{:.3} A", position.y * i_span),
                        "W" => format!("{:.3} W", position.y * p_span),
                        "Set V" => format!("Set {:.3} V", data.set_v),
                        "Set I" => format!("Set {:.3} A", data.set_i),
                        "Set P" => format!("Set {:.3} W", data.set_p),
                        other => other.to_owned(),
                    };
                    Some(format!("{y}\nsample {:.0}", position.x))
                }
                HoverPosition::Elsewhere { position } => Some(format!(
                    "{:.3} V\n{:.3} A\n{:.3} W",
                    position.y * v_span,
                    position.y * i_span,
                    position.y * p_span
                )),
            })
            .show(ui, |plot_ui| {
                let new_bounds =
                    egui_plot::PlotBounds::from_min_max([0.0, 0.0], [*mem_depth as f64, 1.08]);
                plot_ui.set_plot_bounds(new_bounds);
                plot_ui.set_auto_bounds([false, false]);
                plot_ui.line(
                    Line::new("V", scaled_points(data.volt, reverse_graph, v_span))
                        .stroke(egui::Stroke::new(2.0, data.color_v)),
                );
                plot_ui.line(
                    Line::new("A", scaled_points(data.curr, reverse_graph, i_span))
                        .stroke(egui::Stroke::new(2.0, data.color_i)),
                );
                plot_ui.line(
                    Line::new("W", scaled_points(data.power, reverse_graph, p_span))
                        .stroke(egui::Stroke::new(2.0, data.color_p)),
                );
                plot_ui.hline(psu_hline(
                    "Set V",
                    data.set_v / v_span,
                    data.color_v,
                    LineStyle::dashed_dense(),
                ));
                plot_ui.hline(psu_hline(
                    "Set I",
                    data.set_i / i_span,
                    data.color_i,
                    LineStyle::dashed_loose(),
                ));
                plot_ui.hline(psu_hline(
                    "Set P",
                    data.set_p / p_span,
                    data.color_p,
                    LineStyle::dotted_dense(),
                ));
            });
    });
}

#[allow(clippy::too_many_arguments)]
pub fn show_histogram(
    ui: &mut egui::Ui,
    hist_values: &mut VecDeque<f64>,
    curr_meas: f64,
    metermode: MeterMode,
    graph_config: &mut GraphConfig,
    hist_bar_color: Color32,
    hist_collect_active: &mut bool,
    hist_collect_interval_ms: &mut u64,
    hist_mem_depth: &mut usize,
    hist_mem_depth_max: usize,
) {
    let hist_unit = metermode.default_unit();
    let hist_si = crate::helpers::mode_takes_si_prefix(&metermode);
    let fmt_val = |v: f64| {
        if hist_si {
            crate::helpers::format_si(v, hist_unit)
        } else {
            format!("{v:.3} {hist_unit}")
        }
    };
    // Format the latest measurement for display
    let (_formatted_value, display_unit) = crate::helpers::format_measurement(
        curr_meas,
        10,
        1_000_000.0,
        0.0001,
        &metermode,
        false,
        None,
    );

    // Create bar chart data
    let hist_values_vec: Vec<f64> = hist_values
        .iter()
        .copied()
        .filter(|y| y.is_finite() && !is_meter_overload(*y, metermode))
        .collect();
    let (bar_chart, max_count, _num_bins, hist_bin_width, hist_range_start, hist_range_end) =
        if hist_values_vec.is_empty() {
            (
                BarChart::new("Histogram (0 values, bin width: 0)".to_string(), vec![]),
                0.0,
                0,
                0.0,
                0.0,
                0.0,
            )
        } else {
            // Calculate min and max for binning
            let (min, max) = hist_values_vec
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &x| {
                    (min.min(x), max.max(x))
                });
            // Ensure valid range, handle single-value case
            let range_width = if min == max {
                if min == 0.0 {
                    1.0 // Avoid zero range for zero values
                } else {
                    min.abs() * 0.1 // 10% of value for single value
                }
            } else {
                max - min
            };
            let range_start = if min == max {
                min - range_width / 2.0
            } else {
                min
            };
            let range_end = range_start + range_width;

            // Determine number of bins
            let num_bins = if graph_config.num_bins == 0 {
                // Auto-bin using square root rule, capped at max_bins
                let sqrt_bins = (hist_values_vec.len() as f64).sqrt().ceil() as usize;
                sqrt_bins.min(graph_config.max_bins).max(1) // Ensure at least one bin
            } else {
                graph_config.num_bins.max(1) // Ensure at least one bin
            };

            // Calculate bin width in data units
            let bin_width = range_width / num_bins as f64;

            // Create bins
            let mut counts = vec![0; num_bins];
            for &value in &hist_values_vec {
                if value >= range_start && value <= range_end {
                    let bin_index = ((value - range_start) / bin_width).floor() as usize;
                    let bin_index = bin_index.min(num_bins - 1); // Clamp to last bin
                    counts[bin_index] += 1;
                }
            }

            // Compute max_count separately
            let max_count = *counts.iter().max().unwrap_or(&0) as f64;

            // Format bin width for legend
            let (formatted_bin_width, bin_width_unit) = crate::helpers::format_measurement(
                bin_width,
                10,
                1_000_000.0,
                0.0001,
                &metermode,
                false,
                None,
            );
            let chart_name = format!(
                "  Samples: {}\nBin Width: {} {}\n      Min: {}\n      Max: {}",
                hist_values_vec.len(),
                formatted_bin_width.trim_start(),
                bin_width_unit,
                fmt_val(min),
                fmt_val(max)
            );

            // Create bars in normalized canvas coordinates (0 to num_bins)
            let display_bar_width = 1.0; // Width of 1.0 in normalized units
            let bars: Vec<Bar> = counts
                .into_iter()
                .enumerate()
                .map(|(i, count)| {
                    let count_f64 = count as f64;
                    // Center the bar at i + 0.5 in normalized coordinates
                    let bar_center = i as f64 + 0.5;
                    // Directly initialize stroke based on theme
                    let stroke = if ui.ctx().theme().default_visuals().dark_mode {
                        egui::Stroke::new(0.5, Color32::from_rgb(255, 255, 255))
                    } else {
                        egui::Stroke::new(0.5, Color32::from_rgb(0, 0, 0))
                    };
                    Bar::new(bar_center, count_f64)
                        .width(display_bar_width * 0.95) // Slight gap between bars
                        .fill(hist_bar_color)
                        .stroke(stroke)
                })
                .collect();

            // Define element formatter for hover tooltip
            let formatter = Box::new(move |bar: &Bar, _chart: &BarChart| {
                // Calculate bin index from bar center (subtract 0.5 to get zero-based index)
                let bin_index = (bar.argument - 0.5).floor() as usize;
                // Calculate bin range
                let bin_start = range_start + bin_index as f64 * bin_width;
                let bin_end = bin_start + bin_width;
                // Format bin start and end using the same formatting as measurements
                let (formatted_start, _) = crate::helpers::format_measurement(
                    bin_start,
                    10,
                    1_000_000.0,
                    0.0001,
                    &metermode,
                    false,
                    None,
                );
                let (formatted_end, _) = crate::helpers::format_measurement(
                    bin_end,
                    10,
                    1_000_000.0,
                    0.0001,
                    &metermode,
                    false,
                    None,
                );
                // Sample count is the bar's value (height)
                let sample_count = bar.value as usize;
                format!(
                    "Bin Range: {} to {} {}\nSamples: {}",
                    formatted_start.trim_start(),
                    formatted_end.trim_start(),
                    display_unit,
                    sample_count
                )
            });

            (
                BarChart::new(chart_name, bars)
                    .color(hist_bar_color)
                    .element_formatter(formatter),
                max_count,
                num_bins,
                bin_width,
                range_start,
                range_end,
            )
        };

    // Use bottom-up layout to place controls at bottom and plot above
    ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
        // Diagnostic labels (bottom to top due to bottom_up layout)
        // if num_bins > 0 {
        //     let bin_ranges: Vec<String> = (0..num_bins)
        //         .map(|i| {
        //             let bin_start = range_start + i as f64 * bin_width;
        //             let bin_end = bin_start + bin_width;
        //             format!("Bin {}: {:.2} to {:.2}", i, bin_start, bin_end)
        //         })
        //         .collect();
        //     ui.label(format!("Bin ranges: {:?}", bin_ranges));
        // }
        // ui.label(format!("Max count: {}", max_count));
        // ui.label(format!("Data range: {:.2} to {:.2}", min, max));
        // ui.label(format!("Bin width (data units): {:.6}", bin_width));
        // ui.label(format!("Number of bins: {}", num_bins));
        // ui.separator();

        ui.horizontal_wrapped(|ui| {
            // Histogram memory depth slider
            ui.add(
                Slider::new(hist_mem_depth, 100..=hist_mem_depth_max)
                    .text("Memory Depth")
                    .step_by(100.0)
                    .clamping(SliderClamping::Always),
            );

            // Reset button
            if ui.button("Reset Histogram").clicked() {
                hist_values.clear();
            }

            // Start/Stop collection button
            if ui
                .button(if *hist_collect_active {
                    "Stop Collection"
                } else {
                    "Start Collection"
                })
                .clicked()
            {
                *hist_collect_active = !*hist_collect_active;
            }

            // Number of bins slider
            let num_bins_label = if graph_config.num_bins == 0 {
                "Bins: Auto".to_string()
            } else {
                format!("Bins: {}", graph_config.num_bins)
            };
            ui.add(
                Slider::new(&mut graph_config.num_bins, 0..=graph_config.max_bins)
                    .text(num_bins_label)
                    .step_by(1.0)
                    .clamping(SliderClamping::Always),
            );
            let mut interval_str = hist_collect_interval_ms.to_string();

            // Collection interval
            if ui
                .add(
                    egui::TextEdit::singleline(&mut interval_str)
                        .desired_width(100.0)
                        .hint_text("Collection Interval (ms)"),
                )
                .changed()
            {
                if let Ok(new_interval) = interval_str.parse::<u64>() {
                    if new_interval > 0 {
                        *hist_collect_interval_ms = new_interval;
                    }
                }
            }
            ui.label("Collection Interval (ms)");
        });
        ui.label("Histogram Adjustments");
        ui.separator();

        // Plot the histogram above controls, taking remaining space
        let plot = Plot::new("histogram")
            .show_axes(true)
            .show_grid(true)
            .y_axis_label("Count")
            .x_axis_label(format!("Value ({hist_unit})"))
            .x_axis_formatter(move |mark, _| {
                // Bars sit at bin index + 0.5; label them with the value they cover.
                let value = hist_range_start + mark.value * hist_bin_width;
                crate::helpers::format_si_tick(value, hist_range_end - hist_range_start, hist_si)
            })
            .allow_scroll(false) // Prevent scrolling to keep bins stable
            .default_y_bounds(-0.1, 1.0)
            .include_y(max_count * 1.2)
            .legend(
                Legend::default()
                    .position(egui_plot::Corner::RightTop)
                    .text_style(egui::TextStyle::Monospace),
            );

        plot.show(ui, |plot_ui| {
            // Auto-scale x, do y manually to leave space for legend
            plot_ui.set_auto_bounds([true, true]);
            plot_ui.bar_chart(bar_chart);
        });
    });
}

impl super::MyApp {
    // Update histogram buffer with new measurement
    pub fn update_histogram(&mut self, meas: f64) {
        if meas.is_finite() && !is_meter_overload(meas, self.metermode) && self.hist_collect_active
        {
            let current_time = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64();
            let hist_interval = self.hist_collect_interval_ms as f64 / 1000.0; // Convert ms to seconds
            if current_time - self.last_hist_collect_time >= hist_interval {
                self.hist_values.push_back(meas);
                // Respect hist_mem_depth for histogram
                while self.hist_values.len() > self.hist_mem_depth {
                    self.hist_values.pop_front();
                }
                self.last_hist_collect_time = current_time;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xs(ys: &[f64]) -> Vec<f64> {
        (0..ys.len()).map(|i| i as f64).collect()
    }

    #[test]
    fn ol_run_sits_in_the_gap_at_last_valid_y() {
        let ys = [1.0, 2.0, 9.9e31, 9.9e31, 3.0];
        let (meas, ol) = split_meas_and_ol(&xs(&ys), &ys, MeterMode::Res);
        assert_eq!(meas.len(), 2);
        assert_eq!(meas[0], [[0.0, 1.0], [1.0, 2.0]]);
        assert_eq!(meas[1], [[4.0, 3.0]]);
        assert_eq!(ol.len(), 1);
        assert_eq!(ol[0], [[2.0, 2.0], [3.0, 2.0]]);
    }

    #[test]
    fn all_ol_keeps_scrolling_at_zero() {
        let ys = [9.9e31, 9.9e31];
        let (meas, ol) = split_meas_and_ol(&xs(&ys), &ys, MeterMode::Adc);
        assert!(meas.is_empty());
        assert_eq!(ol, vec![vec![[0.0, 0.0], [1.0, 0.0]]]);
    }

    #[test]
    fn one_gigahertz_is_not_an_ol_run() {
        let ys = [1e9];
        let (meas, ol) = split_meas_and_ol(&xs(&ys), &ys, MeterMode::Freq);
        assert_eq!(meas, vec![vec![[0.0, 1e9]]]);
        assert!(ol.is_empty());
    }

    #[test]
    fn ol_overlay_is_one_segment_not_per_sample() {
        let ys = [1.0, 9.9e31, 9.9e31, 9.9e31, 2.0];
        let (_, ol) = split_meas_and_ol(&xs(&ys), &ys, MeterMode::Res);
        assert_eq!(ol[0].len(), 3);
        assert_eq!(flatten_ol_run(&ol[0], 0.4), [[1.0, 1.0], [3.0, 1.0]]);
    }
}

#[cfg(test)]
mod view_tests {
    use super::*;

    fn buffers(n: usize) -> (VecDeque<f64>, VecDeque<f64>) {
        let times = (0..n).map(|i| i as f64 * 0.1).collect();
        let values = (0..n).map(|i| (i % 7) as f64).collect();
        (times, values)
    }

    #[test]
    fn window_selects_padded_range() {
        let (t, v) = buffers(100);
        let s = Series::new(&t, &v);
        let (i0, i1) = s.window(2.0, 3.0);
        assert!(s.t(i0) <= 2.0 && s.t(i1 - 1) >= 3.0);
        assert!(i1 - i0 <= 13);
    }

    #[test]
    fn series_pairs_up_from_the_newest_end() {
        let (mut t, mut v) = buffers(10);
        for _ in 0..5 {
            v.push_front(99.0); // stale extra values at the old end
        }
        t.pop_front();
        let s = Series::new(&t, &v);
        assert_eq!(s.len, 9);
        assert_eq!(s.t(8), 9.0 * 0.1);
        assert_eq!(s.v(8), (9 % 7) as f64);
    }

    #[test]
    fn decimation_keeps_extremes_and_overload() {
        let n = 30_000;
        let times: VecDeque<f64> = (0..n).map(|i| i as f64).collect();
        let mut values: VecDeque<f64> = (0..n).map(|i| (i % 100) as f64).collect();
        values[12_345] = 500.0;
        values[20_000] = 9.9e31;
        let s = Series::new(&times, &values);
        let (xs, ys) = s.decimated(0, n, MeterMode::Vdc, 3000, 10.0);
        assert!(xs.len() <= 9000);
        assert!(ys.contains(&500.0));
        assert!(ys.contains(&9.9e31));
        assert!(xs.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn clock_ticks_format_at_every_precision() {
        for span in [0.01, 1.0, 10.0, 600.0] {
            let s = format_clock(12.3456, 1_700_000_000.0, span);
            assert!(s.len() >= 8 && s.contains(':'), "{s}");
        }
    }

    #[test]
    fn auto_range_ignores_overload_and_pads() {
        let (lo, hi) = auto_y_range(&[1.0, 3.0, 9.9e31], MeterMode::Vdc).unwrap();
        assert!(lo < 1.0 && hi > 3.0 && hi < 4.0);
        assert!(auto_y_range(&[9.9e31], MeterMode::Vdc).is_none());
        let (lo, hi) = auto_y_range(&[2.0, 2.0], MeterMode::Vdc).unwrap();
        assert!(lo < 2.0 && hi > 2.0);
    }
}

#[cfg(test)]
mod frame_tests {
    use super::*;

    /// Run the real graph for many frames with live data, wheel and drag input.
    #[test]
    fn graph_survives_live_frames_and_input() {
        let ctx = egui::Context::default();
        let mut values: VecDeque<f64> = VecDeque::new();
        let mut times: VecDeque<f64> = VecDeque::new();
        let mut view = GraphView::default();
        let (mut depth, mut interval, mut reverse) = (50_000usize, 20u64, false);
        let pos = egui::pos2(300.0, 200.0);
        let mut span_before = 0.0;
        for frame in 0..600 {
            let t = frame as f64 * 0.02;
            let mut input = egui::RawInput {
                time: Some(t),
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(800.0, 500.0),
                )),
                ..Default::default()
            };
            input.events.push(egui::Event::PointerMoved(pos));
            match frame {
                100..=130 => input.events.push(egui::Event::MouseWheel {
                    unit: egui::MouseWheelUnit::Point,
                    delta: egui::vec2(0.0, 20.0),
                    modifiers: egui::Modifiers::NONE,
                    phase: egui::TouchPhase::Move,
                }),
                200..=230 => input.events.push(egui::Event::MouseWheel {
                    unit: egui::MouseWheelUnit::Point,
                    delta: egui::vec2(15.0, 0.0),
                    modifiers: egui::Modifiers::SHIFT,
                    phase: egui::TouchPhase::Move,
                }),
                300..=320 => input.events.push(egui::Event::MouseWheel {
                    unit: egui::MouseWheelUnit::Point,
                    delta: egui::vec2(0.0, 20.0),
                    modifiers: egui::Modifiers::CTRL,
                    phase: egui::TouchPhase::Move,
                }),
                _ => {}
            }
            if frame == 400 {
                // overload run plus a gap in the data
                for _ in 0..5 {
                    values.push_back(9.9e31);
                    times.push_back(t);
                }
            }
            if frame == 450 {
                view.follow = false;
                view.x_min = t + 1000.0; // scrolled away: no samples in view
            }
            if frame == 500 {
                view.zoom_x(1e-6);
                view.zoom_y(1e-6);
            }
            values.push_back((t * 3.0).sin() * 1e-3);
            times.push_back(t);
            let mut out = ctx.run_ui(input, |ui| {
                egui::CentralPanel::default().show(ui, |ui| {
                    show_line_graph(
                        ui,
                        &mut values,
                        &mut times,
                        &mut view,
                        reverse,
                        Color32::GREEN,
                        &mut depth,
                        &mut interval,
                        &mut reverse,
                        1_000_000,
                        1000,
                        "VDC",
                        MeterMode::Vdc,
                    );
                });
            });
            out.textures_delta.clear();
            match frame {
                99 => span_before = view.x_span,
                140 => {
                    assert!(view.x_span < span_before, "wheel should zoom time in");
                    assert!(view.follow, "zooming while live keeps following");
                    assert!(view.auto_y);
                }
                235 => assert!(!view.follow, "shift+wheel scrolls back in time"),
                330 => assert!(!view.auto_y, "ctrl+wheel zooms value manually"),
                _ => {}
            }
            if frame == 350 {
                reverse = true;
            }
        }
        assert!(view.y_max > view.y_min);
        assert!(view.x_span > 0.0);
    }
}
