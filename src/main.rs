use image::{DynamicImage, GenericImage, Rgba};
use show_image::create_window;

#[show_image::main]
fn main() -> Result<(), Box<dyn std::error::Error>> {

    // output
    let w= 1280;
    let h = 720;
    let mut img = DynamicImage::new_rgb8(w, h);
    for x in 0..w {
        for y in 0..h {
            let r = 0;
            let g = 0;
            let b = ((x as f32 * y as f32) / (w as f32 * h as f32)) * 255.0;
            let pixel = Rgba::<u8>([r, g, b as u8, 255]);
            img.put_pixel(x, y, pixel);
        }
    }

     // Create a window with default options and display the image.
    let window = create_window("image", Default::default())?;
    window.set_image("image-001", img)?;
    window.wait_until_destroyed()?;
    Ok(())
}
