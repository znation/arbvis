use image::Rgb;

/// Map a byte to a color based on value range.
///
/// Color scheme from
/// https://stairwell.com/resources/hilbert-curves-visualizing-binary-files-with-color-and-patterns/
pub fn byte_to_pixel(v: u8) -> Rgb<u8> {
    match v {
        0 => Rgb([0, 0, 0]),
        0xFF => Rgb([255, 255, 255]),
        b @ 0x01..=0x1F => {
            let value = ((b - 0x01) as u32 * 255 / (0x1F - 0x01)) as u8;
            Rgb([0, value, 0])
        }
        b @ 0x20..=0x7E => {
            let value = ((b - 0x20) as u32 * 255 / (0x7E - 0x20)) as u8;
            Rgb([0, 0, value])
        }
        b => {
            let value = ((b - 0x7F) as u32 * 255 / (0xFE - 0x7F)) as u8;
            Rgb([value, 0, 0])
        }
    }
}

/// Pre-computed 256-entry color lookup table.
pub fn build_pixel_lut() -> [Rgb<u8>; 256] {
    let mut lut = [Rgb([0u8, 0, 0]); 256];
    for (i, entry) in lut.iter_mut().enumerate() {
        *entry = byte_to_pixel(i as u8);
    }
    lut
}

/// Diff color LUT: linearly scales black (0) → magenta (255).
pub fn build_diff_pixel_lut() -> [Rgb<u8>; 256] {
    let mut lut = [Rgb([0u8, 0, 0]); 256];
    for (i, entry) in lut.iter_mut().enumerate() {
        *entry = Rgb([i as u8, 0, i as u8]);
    }
    lut
}

/// Per-tensor diff LUT encoding file position as hue and diff magnitude as brightness.
///
/// `position` in [0.0, 1.0]: 0.0 = cyan (file start), 1.0 = magenta (file end).
/// Hue H = 180° + position × 120°, analytically converted to full-brightness RGB:
///   H ∈ [180°, 240°]: (0,  255×(240−H)/60,  255)   cyan → blue
///   H ∈ [240°, 300°]: (255×(H−240)/60,  0,  255)   blue → magenta
/// Brightness scales linearly with diff byte d: d=0 → black, d=255 → full hue.
pub fn build_diff_position_lut(position: f32) -> [Rgb<u8>; 256] {
    let t = position.clamp(0.0, 1.0);
    let h = 180.0_f32 + t * 120.0;
    let (r_full, g_full, b_full): (f32, f32, f32) = if h <= 240.0 {
        (0.0, 255.0 * (240.0 - h) / 60.0, 255.0)
    } else {
        (255.0 * (h - 240.0) / 60.0, 0.0, 255.0)
    };
    let mut lut = [Rgb([0u8, 0, 0]); 256];
    for (i, entry) in lut.iter_mut().enumerate() {
        let d = i as f32 / 255.0;
        *entry = Rgb([
            (r_full * d).round() as u8,
            (g_full * d).round() as u8,
            (b_full * d).round() as u8,
        ]);
    }
    lut
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_is_black() {
        assert_eq!(byte_to_pixel(0x00), Rgb([0, 0, 0]));
    }

    #[test]
    fn ff_is_white() {
        assert_eq!(byte_to_pixel(0xFF), Rgb([255, 255, 255]));
    }

    #[test]
    fn printable_ascii_is_blue_only() {
        let c = byte_to_pixel(b'A');
        assert_eq!(c[0], 0);
        assert_eq!(c[1], 0);
        assert!(c[2] > 0);
    }

    #[test]
    fn control_is_green_only() {
        let c = byte_to_pixel(0x10);
        assert_eq!(c[0], 0);
        assert!(c[1] > 0);
        assert_eq!(c[2], 0);
    }

    #[test]
    fn high_byte_is_red_only() {
        let c = byte_to_pixel(0x80);
        assert!(c[0] > 0);
        assert_eq!(c[1], 0);
        assert_eq!(c[2], 0);
    }

    #[test]
    fn lut_has_256_entries() {
        let lut = build_pixel_lut();
        assert_eq!(lut.len(), 256);
        assert_eq!(lut[0], Rgb([0, 0, 0]));
        assert_eq!(lut[255], Rgb([255, 255, 255]));
    }

    #[test]
    fn diff_position_lut_zero_is_always_black() {
        assert_eq!(build_diff_position_lut(0.0)[0], Rgb([0, 0, 0]));
        assert_eq!(build_diff_position_lut(0.5)[0], Rgb([0, 0, 0]));
        assert_eq!(build_diff_position_lut(1.0)[0], Rgb([0, 0, 0]));
    }

    #[test]
    fn diff_position_lut_cyan_at_start() {
        let lut = build_diff_position_lut(0.0);
        assert_eq!(lut[255], Rgb([0, 255, 255]));
    }

    #[test]
    fn diff_position_lut_blue_at_mid() {
        let lut = build_diff_position_lut(0.5);
        assert_eq!(lut[255], Rgb([0, 0, 255]));
    }

    #[test]
    fn diff_position_lut_magenta_at_end() {
        let lut = build_diff_position_lut(1.0);
        assert_eq!(lut[255], Rgb([255, 0, 255]));
    }
}