use crate::multimeter::MeterMode;

/// Compact XDM1041/1241 `MEAS?` open-line value, and the Victor OL flag.
/// This is *not* a universal ceiling: 1 GΩ / 1 GHz are valid on other meters.
pub const METER_OVERLOAD_VALUE: f64 = 1e9;

/// Firmware sentinels such as XDM1051 `~1e31` / Keysight `9.9e37`.
/// Below this, GΩ and GHz readings must still graph as numbers.
pub const SCPI_OVERLOAD_MAGNITUDE: f64 = 1e20;

/// Open/OL: huge SCPI sentinels in any mode, or the 1041 `1e9` flag in ohms-family modes.
pub fn is_meter_overload(value: f64, mode: MeterMode) -> bool {
    if !value.is_finite() {
        return true;
    }
    let mag = value.abs();
    if mag >= SCPI_OVERLOAD_MAGNITUDE {
        return true;
    }
    mag == METER_OVERLOAD_VALUE
        && matches!(
            mode,
            MeterMode::Diod | MeterMode::Cont | MeterMode::Res | MeterMode::Fres
        )
}

/// SI prefixes by decimal exponent, pico to tera.
const SI_PREFIXES: [(i32, &str); 9] = [
    (-12, "p"),
    (-9, "n"),
    (-6, "μ"),
    (-3, "m"),
    (0, ""),
    (3, "k"),
    (6, "M"),
    (9, "G"),
    (12, "T"),
];

/// Modes whose unit takes an SI prefix. Duty (%) and temperature do not.
pub fn mode_takes_si_prefix(mode: &MeterMode) -> bool {
    !matches!(mode, MeterMode::Duty | MeterMode::Temp)
}

fn si_exponent(value: f64) -> i32 {
    if !value.is_finite() || value == 0.0 {
        return 0;
    }
    let exp = (value.abs().log10().floor() as i32).div_euclid(3) * 3;
    exp.clamp(-12, 12)
}

fn si_prefix_for(exp: i32) -> &'static str {
    SI_PREFIXES
        .iter()
        .find(|(e, _)| *e == exp)
        .map_or("", |(_, p)| p)
}

/// Scale `value` into 1..1000 and return it with its SI prefix.
/// Zero and non-finite values stay unscaled.
pub fn si_scale(value: f64) -> (f64, &'static str) {
    let exp = si_exponent(value);
    (value / 10f64.powi(exp), si_prefix_for(exp))
}

/// Compact SI number with unit, e.g. `1.5 mV`. Trailing zeros are trimmed.
pub fn format_si(value: f64, unit: &str) -> String {
    format_si_range(value, f64::INFINITY, unit, true)
}

/// Like `format_si`, with enough decimals to tell apart values `range / 50` apart.
fn format_si_range(value: f64, range: f64, unit: &str, si: bool) -> String {
    if !value.is_finite() {
        return value.to_string();
    }
    if value == 0.0 {
        return if unit.is_empty() {
            "0".to_owned()
        } else {
            format!("0 {unit}")
        };
    }
    let exp = if si { si_exponent(value) } else { 0 };
    let scaled = value / 10f64.powi(exp);
    let decimals = if range.is_finite() && range > 0.0 {
        let step = range / 10f64.powi(exp) / 50.0;
        ((-step.log10()).ceil() as i32).clamp(0, 9) as usize
    } else {
        5
    };
    let mut text = format!("{scaled:.decimals$}");
    if text.contains('.') {
        text.truncate(text.trim_end_matches('0').trim_end_matches('.').len());
    }
    let prefix = si_prefix_for(exp);
    if unit.is_empty() {
        format!("{text}{prefix}")
    } else {
        format!("{text} {prefix}{unit}")
    }
}

/// Axis tick label: SI prefix on the number, no unit (the axis label carries it).
pub fn format_si_tick(value: f64, range: f64, si: bool) -> String {
    format_si_range(value, range, "", si)
}

/// Parse `5m`, `250u`, `1.5k`, `2 mV` (unit letters after the prefix are ignored).
pub fn parse_si(text: &str, unit: &str) -> Option<f64> {
    let t = text.trim().replace(['µ', 'μ'], "u");
    let t = t.strip_suffix(unit).unwrap_or(&t).trim();
    if let Ok(v) = t.parse::<f64>() {
        return Some(v);
    }
    let last = t.chars().last()?;
    let exp = match last {
        'p' => -12,
        'n' => -9,
        'u' => -6,
        'm' => -3,
        'k' | 'K' => 3,
        'M' => 6,
        'G' => 9,
        'T' => 12,
        _ => return None,
    };
    let num: f64 = t[..t.len() - last.len_utf8()].trim().parse().ok()?;
    Some(num * 10f64.powi(exp))
}

pub fn format_measurement(
    value: f64,
    max_digits: usize,
    sci_threshold_high: f64,
    sci_threshold_low: f64,
    meter_mode: &MeterMode,
    auto_scale_units: bool,
    lcd_override: Option<(&str, &str)>,
) -> (String, String) {
    if value.is_nan() {
        return ("    NaN".to_string(), "".to_string());
    }

    if is_meter_overload(value, *meter_mode) {
        return ("OVERLOAD".to_string(), "".to_string());
    }

    // Victor 6000-count meters: show wire-decoded LCD text (4 digits), unit from annunciator.
    if let Some((lcd, unit)) = lcd_override {
        if !lcd.is_empty() {
            return (
                format!("{:>width$}", lcd, width = max_digits),
                unit.to_string(),
            );
        }
    }

    let abs_value = value.abs();
    let mut display_value = value;
    let mut display_unit = meter_mode.default_unit().to_string();

    // Adjust value and unit to an SI prefix (mV, kOhm, uF, ...) so the value reads 1..1000.
    if auto_scale_units && mode_takes_si_prefix(meter_mode) && abs_value > 0.0 {
        let (scaled, prefix) = si_scale(value);
        display_value = scaled;
        display_unit = format!("{prefix}{display_unit}");
    }

    let abs_display_value = display_value.abs();

    // Format the value
    let formatted_value = if abs_display_value >= sci_threshold_high
        || (abs_display_value < sci_threshold_low && abs_display_value > 0.0)
    {
        format!("{:>width$.3e}", display_value, width = max_digits)
    } else {
        let precision = if abs_display_value >= 1000.0 {
            2
        } else if abs_display_value >= 100.0 {
            3
        } else if abs_display_value >= 10.0 {
            4
        } else {
            5
        };
        format!("{:>width$.*}", precision, display_value, width = max_digits)
    };

    (formatted_value, display_unit)
}

pub fn powered_by(ui: &mut egui::Ui) {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 0.0;
        ui.label("Powered by ");
        ui.hyperlink_to("egui", "https://github.com/emilk/egui");
        ui.label(", ");
        ui.hyperlink_to(
            "eframe",
            "https://github.com/emilk/egui/tree/master/crates/eframe",
        );
        ui.label(", ");
        ui.hyperlink_to("B612 Font", "https://b612-font.com/");
        ui.label(" and ");
        ui.hyperlink_to(
            "TheHWCave",
            "https://github.com/TheHWcave/OWON-XDM1041/tree/main",
        );
        ui.label(".");
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multimeter::MeterMode;

    #[test]
    fn xdm1041_1e9_is_ol_only_in_ohms_family() {
        assert!(is_meter_overload(1e9, MeterMode::Res));
        assert!(is_meter_overload(1e9, MeterMode::Cont));
        assert!(is_meter_overload(1e9, MeterMode::Diod));
        assert!(!is_meter_overload(1e9, MeterMode::Freq));
        assert!(!is_meter_overload(1e9, MeterMode::Vdc));
        assert!(!is_meter_overload(1e10, MeterMode::Res));
        assert!(!is_meter_overload(50e6, MeterMode::Res));
    }

    #[test]
    fn huge_scpi_sentinels_are_ol_in_every_mode() {
        for mode in MeterMode::ALL {
            assert!(is_meter_overload(9.9e31, mode));
            assert!(is_meter_overload(-9.9e37, mode));
            assert!(is_meter_overload(f64::INFINITY, mode));
            let (text, unit) = format_measurement(9.9e31, 10, 1e6, 1e-6, &mode, true, None);
            assert_eq!(text, "OVERLOAD");
            assert_eq!(unit, "");
        }
    }

    #[test]
    fn readout_uses_si_prefixes() {
        let f = |v: f64, m: MeterMode| {
            let (t, u) = format_measurement(v, 10, 1e15, 1e-15, &m, true, None);
            (t.trim().to_owned(), u)
        };
        assert_eq!(f(0.0123, MeterMode::Vdc), ("12.3000".into(), "mVDC".into()));
        assert_eq!(
            f(0.04403, MeterMode::Adc),
            ("44.0300".into(), "mADC".into())
        );
        assert_eq!(f(4.7e-6, MeterMode::Adc), ("4.70000".into(), "μADC".into()));
        assert_eq!(f(2.5e-9, MeterMode::Per), ("2.50000".into(), "ns".into()));
        assert_eq!(f(1.5e6, MeterMode::Freq), ("1.50000".into(), "MHz".into()));
        assert_eq!(f(4700.0, MeterMode::Res), ("4.70000".into(), "kOhm".into()));
        assert_eq!(f(3.3, MeterMode::Vdc), ("3.30000".into(), "VDC".into()));
        assert_eq!(f(50.0, MeterMode::Duty), ("50.0000".into(), "%".into()));
    }

    #[test]
    fn si_tick_and_parse() {
        assert_eq!(format_si_tick(0.0, 1.0, true), "0");
        assert_eq!(format_si_tick(0.0005, 0.01, true), "500μ");
        assert_eq!(format_si_tick(1500.0, 5000.0, true), "1.5k");
        assert_eq!(format_si(0.0015, "V"), "1.5 mV");
        assert_eq!(format_si_tick(0.25, 1.0, false), "0.25");
        assert_eq!(parse_si("250u", "V"), Some(250e-6));
        assert_eq!(parse_si("2 mV", "V"), Some(2e-3));
        assert_eq!(parse_si("1.5k", ""), Some(1500.0));
        assert_eq!(parse_si("abc", ""), None);
    }

    #[test]
    fn gigahertz_is_not_overload() {
        let (text, unit) = format_measurement(1e9, 10, 1e12, 1e-6, &MeterMode::Freq, true, None);
        assert_ne!(text, "OVERLOAD");
        assert_eq!(unit, "GHz");
    }
}
