use std::io::{self, Read};

use image::{DynamicImage, GenericImage, Rgba};
use show_image::create_window;

#[show_image::main]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let w: usize = 1280;
    let h: usize = 720;

    // create input image (same size as output)
    // for now, truncate at output size using raw bytes
    let mut input = Vec::<u8>::new();
    input.reserve(w as usize * h as usize);
    let mut count = 0;
    for possible_value in io::stdin().bytes() {
        if count >= w * h {
            break;
        }
        match possible_value {
            Ok(v) => input.push(v),
            Err(e) => return Err(Box::new(e)),
        }
        count += 1;
    }

    // create output image
    let mut img = DynamicImage::new_rgb8(w as u32, h as u32);
    for x in 0..w {
        for y in 0..h {
            let mut pixel = Rgba::<u8>([0, 0, 0, 255]);

            // color scheme from
            // https://stairwell.com/resources/hilbert-curves-visualizing-binary-files-with-color-and-patterns/
            let i = (x * y) % input.len();
            if input[i] == 0 {
                // null (0) bytes
                // do nothing; default color is black
            }
            else if input[i] <= 0x1F {
                // low bytes/control characters
                pixel = Rgba::<u8>([0, 255, 0, 255]);
            }
            else if input[i] <= 0x7E {
                // ascii printable chars
                pixel = Rgba::<u8>([0, 0, 255, 255]);
            }
            else {
                // high bytes
                pixel = Rgba::<u8>([255, 0, 0, 255]);
            }
            
            img.put_pixel(x as u32, y as u32, pixel);
        }
    }

     // Create a window with default options and display the image.
    let window = create_window("image", Default::default())?;
    window.set_image("image-001", img)?;
    window.wait_until_destroyed()?;
    Ok(())
}
