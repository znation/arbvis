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
    // Escape every `<` in the JSON blob, not just `</`: `<!--` followed by
    // `<script>` puts the HTML parser into the script-data-double-escaped
    // state, in which the template's own closing `</script>` no longer closes
    // the element (page break). `\u003C` is a valid JSON escape that JS string
    // literals decode back to `<`, so no `<` sequence from the config survives
    // into the inline script.
    let config = config.to_string().replace('<', "\\u003C");
    TEMPLATE.replace("__CONFIG_JSON__", &config)
}

#[cfg(test)]
mod build_tests {
    use super::*;

    /// The template embeds the config as `const CFG = <json>;` on one line;
    /// pull the JSON back out and parse it.
    fn config_blob(html: &str) -> serde_json::Value {
        let marker = "const CFG = ";
        let start = html.find(marker).unwrap_or_else(|| {
            panic!("template no longer declares `const CFG = __CONFIG_JSON__;`")
        }) + marker.len();
        let line = &html[start..html[start..].find('\n').expect("CFG line unterminated") + start];
        let line = line.strip_suffix(';').unwrap_or(line);
        serde_json::from_str(line).expect("injected config blob must be valid JSON")
    }

    #[test]
    fn config_carries_title_inputs_and_branding() {
        let branding = Branding::new("mwv", "https://example.com/mwv");
        let html = build_volume_html(
            "my title",
            &["a.bin".to_string(), "b.bin".to_string()],
            &branding,
        );
        let cfg = config_blob(&html);
        assert_eq!(cfg["title"], "my title");
        assert_eq!(cfg["brandName"], "mwv");
        assert_eq!(cfg["repoUrl"], "https://example.com/mwv");
        assert_eq!(cfg["inputs"], serde_json::json!(["a.bin", "b.bin"]));
    }

    #[test]
    fn script_closing_sequences_in_config_are_neutralized() {
        // A raw `<` inside the injected JSON would be dangerous beyond `</`:
        // `</script>` closes the inline script, and `<!--` + `<script>` flips
        // the parser into the double-escaped state that swallows the
        // template's own closing tag. Every `<` must be escaped to `\u003C`.
        let html = build_volume_html(
            "<!--<script></script><script>alert(1)</script>",
            &[],
            &Branding::default(),
        );
        let marker = "const CFG = ";
        let start = html.find(marker).unwrap() + marker.len();
        let cfg_line = &html[start..html[start..].find('\n').unwrap() + start];
        assert!(
            !cfg_line.contains('<'),
            "no raw `<` may survive in the config line: {cfg_line}"
        );
        assert!(cfg_line.contains("\\u003C"));
        // The document itself still parses down to the template's own ending.
        assert!(html.trim_end().ends_with("</html>"));
    }

    #[test]
    fn default_branding_fills_both_fields() {
        let html = build_volume_html("t", &[], &Branding::default());
        let cfg = config_blob(&html);
        assert_eq!(cfg["brandName"], "arbvis");
        assert_eq!(cfg["repoUrl"], "https://github.com/znation/arbvis");
        assert_eq!(cfg["title"], "t");
        assert_eq!(cfg["inputs"], serde_json::json!([]));
    }
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
