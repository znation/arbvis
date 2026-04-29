use std::io::{self, Read};

use fast_hilbert::h2xy;
use image::{DynamicImage, GenericImage, Rgba};
use show_image::create_window;

#[show_image::main]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    const W: u32 = 400;
    const H: u32 = 400;
    const LEN: usize = W as usize * H as usize;
    let mut hilbert_cache: [Option<(u32, u32)>; LEN] = [None; LEN];
    
    // create output image
    let mut img = DynamicImage::new_rgb8(W, H);

     // Create a window with default options and display the image.
    let mut window = create_window("image", Default::default())?;

    // create input image (same size as output)
    // for now, truncate at output size using raw bytes
    let mut count = 0;
    for possible_value in io::stdin().bytes() {
        if count >= LEN {
            window.set_image("image-001", img)?;
            img = DynamicImage::new_rgb8(W, H);
            window = create_window("image", Default::default())?;
            break;
        }
        match possible_value {
            Ok(v) => {
                let mut pixel = Rgba::<u8>([0, 0, 0, 255]);

                // color scheme from
                // https://stairwell.com/resources/hilbert-curves-visualizing-binary-files-with-color-and-patterns/
                if v == 0 {
                    // null (0) bytes
                    // do nothing; default color is black
                }
                else if v == 0xFF {
                    // highest bytes - white
                    pixel = Rgba([255, 255, 255, 255]);
                }
                else if v <= 0x1F {
                    // low bytes/control characters
                    let value = ((v as f32 - 0x01 as f32) / (0x1F as f32 - 0x01 as f32)) * 255.0;
                    pixel = Rgba::<u8>([0, value as u8, 0, 255]);
                }
                else if v <= 0x7E {
                    // ascii printable chars
                    let value = ((v as f32 - 0x20 as f32) / (0x7E as f32 - 0x20 as f32)) * 255.0;
                    pixel = Rgba::<u8>([0, 0, value as u8, 255]);
                }
                else {
                    // higher bytes
                    let value = ((v as f32 - 0x7F as f32) / (0xFE as f32 - 0x7F as f32)) * 255.0;
                    pixel = Rgba::<u8>([value as u8, 0, 0, 255]);
                }

                if hilbert_cache[count] == None {
                    hilbert_cache[count] = Some(h2xy(count as u64, 1));
                }
                let (x, y): (u32, u32) = hilbert_cache[count].unwrap();


                if x >= W || y >= H {
                    // out of bounds
                    // TODO: line up image size to fit hilbert curve
                    continue
                }
                img.put_pixel(x as u32, y as u32, pixel);

                if count % 1000 == 0 {
                    window.set_image("image-001", img.clone())?;
                }
            },
            Err(e) => return Err(Box::new(e)),
        }
        count += 1;
    }

    window.set_image("image-001", img)?;
    window.wait_until_destroyed()?;
    Ok(())
}
