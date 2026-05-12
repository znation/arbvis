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

/// Signed diff LUT used for all diffs (safetensors per-tensor and plain binary).
///
/// Encoding:
///   127         → no change     → black
///   128..=254   → value grew    → green, brightness = (v − 127) / 127
///   0..=126     → value shrank  → red,   brightness = (127 − v) / 127
///   255         → non-finite    → white
pub fn build_diff_signed_lut() -> [Rgb<u8>; 256] {
    let mut lut = [Rgb([0u8, 0, 0]); 256];
    for (i, entry) in lut.iter_mut().enumerate() {
        *entry = match i {
            127 => Rgb([0, 0, 0]),
            255 => Rgb([255, 255, 255]),
            128..=254 => {
                let b = ((i - 127) as f32 / 127.0 * 255.0).round() as u8;
                Rgb([0, b, 0])   // green: value increased
            }
            _ => {
                let b = ((127 - i) as f32 / 127.0 * 255.0).round() as u8;
                Rgb([b, 0, 0])   // red: value decreased
            }
        };
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

}