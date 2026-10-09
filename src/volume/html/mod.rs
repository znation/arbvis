//! Standalone Three.js viewer for the 3D (`--3d`) bundle.
//!
//! The 3D analog of [`crate::tiled::html`]. Like the Leaflet viewer it is a
//! self-contained `index.html` built as a string and loads its rendering
//! library (here Three.js) from a CDN via an ES-module import map, so it
//! deploys to an HF Space with no build step. At runtime it fetches
//! `meta.json` and `volume.bin` (written alongside it by
//! [`crate::volume::render_volume`]).

use crate::registry::Branding;

mod template;

use template::TEMPLATE;

/// Build the 3D viewer HTML. Branding/title/inputs are injected as a JSON
/// config blob; everything else (grid extent, LUT) is read from `meta.json`
/// at runtime.
pub fn build_volume_html(title: &str, inputs: &[String], branding: &Branding) -> String {
    let config = serde_json::json!({
        "title": title,
        "brandName": branding.name,
        "repoUrl": branding.repo_url,
        "inputs": inputs,
    });
    // `</` would prematurely close the inline <script>; neutralize it.
    let config = config.to_string().replace("</", "<\\/");
    TEMPLATE.replace("__CONFIG_JSON__", &config)
}

#[cfg(test)]
mod template_tests {
    use super::TEMPLATE;

    #[test]
    fn template_survived_the_module_split_intact() {
        assert!(TEMPLATE.starts_with("<!DOCTYPE html>\n<html lang=\"en\">"));
        assert!(TEMPLATE.contains("__CONFIG_JSON__"));
        assert!(TEMPLATE.trim_end().ends_with("</html>"));
        assert!(!TEMPLATE.contains("</script><script>"));
    }
}
