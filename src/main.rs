use std::io::{self, Read};

use fast_hilbert::h2xy;
use image::{DynamicImage, GenericImage, Rgba};
use show_image::create_window;

#[show_image::main]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // create input image (same size as output)
    // for now, truncate at output size using raw bytes
    let mut input = Vec::<u8>::new();
    let mut count = 0;
    for possible_value in io::stdin().bytes() {
        if count > 10000000 {
            break;
        }
        match possible_value {
            Ok(v) => input.push(v),
            Err(e) => return Err(Box::new(e)),
        }
        count += 1;
    }

    let w = f64::sqrt(input.len() as f64) as usize;
    let h = w;

    // create output image
    let mut img = DynamicImage::new_rgb8(w as u32, h as u32);
    for i in 0..input.len() {
        let mut pixel = Rgba::<u8>([0, 0, 0, 255]);

        // color scheme from
        // https://stairwell.com/resources/hilbert-curves-visualizing-binary-files-with-color-and-patterns/
        if input[i] == 0 {
            // null (0) bytes
            // do nothing; default color is black
        }
        else if input[i] == 0xFF {
            // highest bytes - white
            pixel = Rgba([255, 255, 255, 255]);
        }
        else if input[i] <= 0x1F {
            // low bytes/control characters
            let value = ((input[i] as f32 - 0x01 as f32) / (0x1F as f32 - 0x01 as f32)) * 255.0;
            pixel = Rgba::<u8>([0, value as u8, 0, 255]);
        }
        else if input[i] <= 0x7E {
            // ascii printable chars
            let value = ((input[i] as f32 - 0x20 as f32) / (0x7E as f32 - 0x20 as f32)) * 255.0;
            pixel = Rgba::<u8>([0, 0, value as u8, 255]);
        }
        else {
            // higher bytes
            let value = ((input[i] as f32 - 0x7F as f32) / (0xFE as f32 - 0x7F as f32)) * 255.0;
            pixel = Rgba::<u8>([value as u8, 0, 0, 255]);
        }
        let (x, y): (u32, u32) = h2xy(i as u64, 1);
        if x as usize >= w || y as usize >= h {
            // out of bounds
            // TODO: line up image size to fit hilbert curve
            break
        }
        img.put_pixel(x as u32, y as u32, pixel);
    }

     // Create a window with default options and display the image.
    let window = create_window("image", Default::default())?;
    window.set_image("image-001", img)?;
    window.wait_until_destroyed()?;
    Ok(())
}
