//! `--regen-html` support: rebuild `index.html` for an existing 2D tile
//! bundle without re-rendering tiles. Split out of `tiled/mod.rs` so that
//! module stays focused on tile rendering/pipelining.

use std::path::Path;

use anyhow::Context;

use crate::registry::Branding;
use crate::tiled::html;
use crate::tiled::leaf::TILE;

/// Return the extension of the first regular file found in `dir`.
fn sniff_ext_in(dir: &std::path::Path) -> Option<String> {
    std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .find_map(|e| {
            e.path()
                .extension()
                .and_then(|s| s.to_str())
                // Only plain alphanumeric extensions (png/avif) are accepted:
                // the sniffed extension is interpolated into single-quoted JS
                // string literals in the generated viewer (getTileUrl), so a
                // hostile bundle tile named e.g. `0_0.png';alert(1);'` must
                // not supply its whole suffix as the "extension". Non-alnum
                // candidates are skipped so the caller's fallback applies.
                .filter(|s| !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric()))
                .map(|s| s.to_string())
        })
}

/// Return the extension of any tile under `tiles/{zoom}/<x>/<y>.<ext>`.
fn sniff_ext_for_zoom(tiles_dir: &std::path::Path, zoom: u32) -> Option<String> {
    let zoom_dir = tiles_dir.join(format!("{zoom}"));
    let x_dir = std::fs::read_dir(&zoom_dir)
        .ok()?
        .filter_map(|e| e.ok())
        .find(|e| e.path().is_dir())?;
    sniff_ext_in(&x_dir.path())
}

/// Parse one `labels.json` file entry back into a [`html::FileEntity`].
fn file_entity_from_json(v: &serde_json::Value) -> html::FileEntity {
    let name = v["name"].as_str().unwrap_or("").to_string();
    let pixel_x = v["x"].as_u64().unwrap_or(0) as u32;
    let pixel_y = v["y"].as_u64().unwrap_or(0) as u32;
    let hue = v["hue"].as_u64().unwrap_or(0) as u16;
    let byte_size = v["size"].as_u64().unwrap_or(0);
    let bbox = {
        if let Some(b) = v["bbox"].as_array() {
            let g = |i: usize| b.get(i).and_then(|x| x.as_u64()).unwrap_or(0) as u32;
            (g(0), g(1), g(2), g(3))
        } else {
            (0, 0, 0, 0)
        }
    };
    let segments = if let Some(segs) = v["segs"].as_array() {
        segs.iter()
            .filter_map(|s| {
                let arr = s.as_array()?;
                let g = |i: usize| arr.get(i)?.as_u64().map(|x| x as u32);
                Some((g(0)?, g(1)?, g(2)?, g(3)?))
            })
            .collect()
    } else {
        vec![]
    };
    html::FileEntity {
        name,
        pixel_x,
        pixel_y,
        hue,
        byte_size,
        bbox,
        segments,
    }
}

/// Rebuild a multi-scene viewer from the scene-keyed `labels.json` shape
/// (`{ "scenes": [...] }`). All geometry is persisted per scene, so unlike the
/// single-scene path this needs no tile-directory scan. Returns `Ok(false)`
/// when `labels.json` isn't the scenes shape, so the caller falls through to
/// the legacy single-pyramid regen.
fn regen_html_multi(
    tile_dir: &Path,
    parsed: &serde_json::Value,
    branding: &Branding,
    title: Option<&str>,
) -> anyhow::Result<bool> {
    let Some(scenes_json) = parsed.get("scenes").and_then(|v| v.as_array()) else {
        return Ok(false);
    };
    let u32_field =
        |s: &serde_json::Value, k: &str| s.get(k).and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let str_field = |s: &serde_json::Value, k: &str, dflt: &str| {
        s.get(k)
            .and_then(|v| v.as_str())
            .unwrap_or(dflt)
            .to_string()
    };
    let scenes: Vec<html::SceneView> = scenes_json
        .iter()
        .map(|s| {
            let entities = s
                .get("files")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().map(file_entity_from_json).collect())
                .unwrap_or_default();
            let key = s.get("key").and_then(|v| v.as_str()).unwrap_or("");
            html::SceneView {
                key: (!key.is_empty()).then(|| key.to_string()),
                label: str_field(s, "label", ""),
                order: u32_field(s, "order"),
                world_w: u32_field(s, "world_w"),
                world_h: u32_field(s, "world_h"),
                max_zoom: u32_field(s, "max_zoom"),
                detail_depth: u32_field(s, "detail_depth"),
                height: u32_field(s, "height"),
                width: u32_field(s, "width"),
                leaf_ext: str_field(s, "leaf_ext", "png"),
                pyramid_ext: str_field(s, "pyramid_ext", "png"),
                entities,
            }
        })
        .collect();
    html::write_leaflet_html_multi(
        tile_dir,
        &scenes,
        title.unwrap_or(branding.name.as_ref()),
        &[],
        branding,
    )?;
    log::info!(
        "Regenerated index.html ({} scenes) in {}",
        scenes.len(),
        tile_dir.display()
    );
    Ok(true)
}

/// Regenerate `index.html` for an existing tiles directory without re-rendering tiles.
pub fn regen_html(tile_dir: &Path, branding: &Branding, title: Option<&str>) -> anyhow::Result<()> {
    // An explicit `--title` overrides the branding default; without it the
    // regen keeps the tool name (the original render's title is not persisted
    // in a 2D bundle's labels.json, so there is nothing to restore here).
    let title = title.unwrap_or(branding.name.as_ref());
    let tiles_dir = tile_dir.join("tiles");

    // Read labels.json first. Newer outputs persist `max_zoom`/`detail_depth` so
    // we can tell the dense overview levels apart from the sparse variable-depth
    // detail levels — without that, the deepest *detail* zoom dir would be
    // mistaken for the overview leaf and corrupt every derived dimension.
    let labels_path = tile_dir.join("labels.json");
    let json_str = std::fs::read_to_string(&labels_path).with_context(|| {
        format!(
            "cannot read {} (--regen-html expects a viewer bundle directory \
             rendered with --out; a --3d bundle is regenerated only with --3d)",
            labels_path.display()
        )
    })?;
    let parsed: serde_json::Value = serde_json::from_str(&json_str)?;
    // Multi-scene outputs carry per-scene geometry in labels.json and live
    // under `tiles/<key>/…`, so they regenerate without any dir scan.
    if regen_html_multi(tile_dir, &parsed, branding, Some(title))? {
        return Ok(());
    }
    let detail_depth = parsed
        .get("detail_depth")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    let values: Vec<serde_json::Value> = match &parsed {
        serde_json::Value::Array(a) => a.clone(),
        serde_json::Value::Object(o) => o
            .get("files")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default(),
        _ => anyhow::bail!("labels.json: unexpected JSON shape (expected array or object)"),
    };

    // Overview leaf zoom: prefer the persisted value; else fall back to the
    // deepest zoom dir minus any detail levels (and minus 0 for legacy outputs
    // that predate the persisted fields).
    let deepest = std::fs::read_dir(&tiles_dir)
        .with_context(|| format!("cannot read {}", tiles_dir.display()))?
        .filter_map(|e| e.ok())
        .filter_map(|e| e.file_name().to_string_lossy().parse::<u32>().ok())
        .max()
        .ok_or_else(|| anyhow::anyhow!("no zoom levels found in {}", tiles_dir.display()))?;
    let max_zoom = parsed
        .get("max_zoom")
        .and_then(|v| v.as_u64())
        .map(|m| m as u32)
        .unwrap_or_else(|| deepest.saturating_sub(detail_depth));

    let zoom_dir = tiles_dir.join(format!("{max_zoom}"));
    let width_tiles = std::fs::read_dir(&zoom_dir)
        .with_context(|| format!("cannot read {}", zoom_dir.display()))?
        .filter(|e| e.as_ref().map(|e| e.path().is_dir()).unwrap_or(false))
        .count() as u32;
    let first_x = std::fs::read_dir(&zoom_dir)?
        .filter_map(|e| e.ok())
        .find(|e| e.path().is_dir())
        .ok_or_else(|| anyhow::anyhow!("no x-dirs found at zoom {max_zoom}"))?;
    let first_x_path = first_x.path();
    let height_tiles = std::fs::read_dir(&first_x_path)?.count() as u32;
    // Sniff the tile extensions from existing files. Leaf zoom (max_zoom) and
    // the pyramid levels can use different formats — e.g. indexed-PNG leaves
    // with lossy-AVIF pyramid — so we sniff each one independently.
    let leaf_ext = sniff_ext_in(&first_x_path).unwrap_or_else(|| "png".to_string());
    let pyramid_ext = if max_zoom > 0 {
        sniff_ext_for_zoom(&tiles_dir, max_zoom - 1).unwrap_or_else(|| leaf_ext.clone())
    } else {
        leaf_ext.clone()
    };
    let height = height_tiles * TILE;
    // Unified Hilbert / arch bounds formula. At leaf zoom `max_zoom` the canvas
    // covers `width_tiles × height_tiles` tiles; the leaflet view at zoom 0
    // covers `width_tiles / 2^max_zoom × height_tiles / 2^max_zoom` tiles —
    // exactly one of which is 1 by construction (Hilbert: kh == max_zoom + 8;
    // arch: max_zoom = log2(min(w_p2, h_p2))). Multiplying back up by TILE
    // gives geo extents. For square Hilbert this collapses to `world_h = TILE,
    // world_w = TILE * 2^(kw-kh)` — the historical formula.
    // `max_zoom` is read back from persisted state (labels.json, or a zoom
    // dir name when that field is absent), so it is not trusted: an out-of-
    // range value would make the shift below overflow and panic the whole
    // regen run. Fail with a descriptive error instead.
    let two_pow_mz = 1u32.checked_shl(max_zoom).ok_or_else(|| {
        anyhow::anyhow!(
            "labels.json max_zoom {max_zoom} is out of range (0–31); \
             the bundle's labels.json or zoom directories look corrupt"
        )
    })?;
    let world_w = (width_tiles / two_pow_mz.max(1)).max(1) * TILE;
    let world_h = (height_tiles / two_pow_mz.max(1)).max(1) * TILE;

    let entities: Vec<html::FileEntity> = values.iter().map(file_entity_from_json).collect();

    let width = width_tiles * TILE;
    html::write_leaflet_html(
        tile_dir,
        world_w,
        world_h,
        max_zoom,
        detail_depth,
        height,
        width,
        TILE,
        &entities,
        title,
        &[],
        &leaf_ext,
        &pyramid_ext,
        branding,
    )?;
    log::info!(
        "Regenerated index.html in {} (zoom 0–{max_zoom}, +{detail_depth} detail, {width_tiles}×{height_tiles} tiles, height={height})",
        tile_dir.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A corrupt labels.json (`max_zoom` out of the representable shift
    /// range) with a matching zoom directory must produce a descriptive
    /// error, not a shift-overflow panic that kills the run.
    #[test]
    fn regen_html_errors_on_out_of_range_max_zoom() {
        let dir = tempfile::tempdir().unwrap();
        let zoom_dir = dir.path().join("tiles").join("34").join("0");
        std::fs::create_dir_all(&zoom_dir).unwrap();
        std::fs::write(
            dir.path().join("labels.json"),
            r#"{"max_zoom": 34, "files": []}"#,
        )
        .unwrap();
        let branding = crate::registry::Branding::default();
        let err = regen_html(dir.path(), &branding, None).unwrap_err();
        assert!(
            err.to_string().contains("max_zoom 34 is out of range"),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    fn regen_html_errors_when_labels_json_missing() {
        let dir = tempfile::tempdir().unwrap();
        let branding = crate::registry::Branding::default();
        let err = regen_html(dir.path(), &branding, None).unwrap_err();
        assert!(
            err.to_string()
                .contains("--regen-html expects a viewer bundle"),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    fn file_entity_from_json_parses_fields_and_defaults() {
        let full: serde_json::Value = serde_json::json!({
            "name": "model.bin",
            "x": 7,
            "y": 9,
            "hue": 120,
            "size": 4096,
            "bbox": [1, 2, 3, 4],
            "segs": [[0, 1, 2, 3], null, [4, 5, 6, 7], [8]]
        });
        let e = file_entity_from_json(&full);
        assert_eq!(e.name, "model.bin");
        assert_eq!(e.pixel_x, 7);
        assert_eq!(e.pixel_y, 9);
        assert_eq!(e.hue, 120);
        assert_eq!(e.byte_size, 4096);
        assert_eq!(e.bbox, (1, 2, 3, 4));
        // Malformed segment entries (null, too-short) are dropped, not fatal.
        assert_eq!(e.segments, vec![(0, 1, 2, 3), (4, 5, 6, 7)]);

        let empty = file_entity_from_json(&serde_json::json!({}));
        assert_eq!(empty.name, "");
        assert_eq!(empty.pixel_x, 0);
        assert_eq!(empty.pixel_y, 0);
        assert_eq!(empty.hue, 0);
        assert_eq!(empty.byte_size, 0);
        assert_eq!(empty.bbox, (0, 0, 0, 0));
        assert!(empty.segments.is_empty());
    }

    #[test]
    fn regen_html_multi_scene_shape_rewrites_index_html() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("labels.json"),
            serde_json::json!({
                "scenes": [{
                    "key": "base",
                    "label": "base scene",
                    "order": 1,
                    "world_w": 256,
                    "world_h": 256,
                    "max_zoom": 2,
                    "detail_depth": 0,
                    "height": 256,
                    "width": 256,
                    "leaf_ext": "png",
                    "pyramid_ext": "png",
                    "files": [{"name": "a.bin", "x": 0, "y": 0, "size": 4}]
                }]
            })
            .to_string(),
        )
        .unwrap();
        let branding = crate::registry::Branding::default();
        regen_html(dir.path(), &branding, None).unwrap();
        let index = std::fs::read_to_string(dir.path().join("index.html")).unwrap();
        assert!(
            index.contains("base scene"),
            "scene label must reach the HTML"
        );
    }

    /// An explicit `--title` must reach the regenerated multi-scene HTML;
    /// without it the branding name is used (a 2D bundle's labels.json does
    /// not persist the original title).
    #[test]
    fn regen_html_honors_title_override() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("labels.json"),
            serde_json::json!({
                "scenes": [{
                    "key": "base", "label": "base scene", "order": 1,
                    "world_w": 256, "world_h": 256, "max_zoom": 2,
                    "detail_depth": 0, "height": 256, "width": 256,
                    "leaf_ext": "png", "pyramid_ext": "png",
                    "files": []
                }]
            })
            .to_string(),
        )
        .unwrap();
        let branding = crate::registry::Branding::default();
        regen_html(dir.path(), &branding, Some("custom title")).unwrap();
        let index = std::fs::read_to_string(dir.path().join("index.html")).unwrap();
        assert!(
            index.contains("custom title"),
            "--title override must reach the regenerated HTML"
        );
    }

    #[test]
    fn regen_hostile_tile_filename_extension_is_not_injected_into_the_viewer() {
        let dir = tempfile::tempdir().unwrap();
        // A hostile bundle names its only tile so that the raw suffix after
        // the last dot is a JS string-breakout payload. The sniffer must
        // reject it and fall back to a safe extension rather than
        // interpolating it into the viewer's single-quoted JS literals.
        let leaf = dir.path().join("tiles").join("0").join("0");
        std::fs::create_dir_all(&leaf).unwrap();
        std::fs::write(leaf.join("0.png';alert(1);'"), b"x").unwrap();
        let labels: serde_json::Value = serde_json::json!([
            {"name": "a.bin", "x": 0, "y": 0, "size": 4}
        ]);
        std::fs::write(dir.path().join("labels.json"), labels.to_string()).unwrap();
        let branding = crate::registry::Branding::default();
        regen_html(dir.path(), &branding, None).unwrap();
        let index = std::fs::read_to_string(dir.path().join("index.html")).unwrap();
        assert!(
            !index.contains("alert(1)"),
            "hostile tile-filename suffix must not reach the generated viewer JS"
        );
        // The sniffer fell back to `png`, which still reaches the template.
        assert!(
            index.contains("? 'png'"),
            "fallback extension must reach the HTML"
        );
    }

    #[test]
    fn regen_html_legacy_array_shape_sniffs_tile_layout() {
        let dir = tempfile::tempdir().unwrap();
        // One tile at zoom 0 under tiles/0/0/, in png format, plus a coarser
        // pyramid level in a different format to exercise independent sniffing.
        let leaf = dir.path().join("tiles").join("0").join("0");
        std::fs::create_dir_all(&leaf).unwrap();
        std::fs::write(leaf.join("0.png"), b"x").unwrap();
        let labels: serde_json::Value = serde_json::json!([
            {"name": "a.bin", "x": 0, "y": 0, "size": 4}
        ]);
        std::fs::write(dir.path().join("labels.json"), labels.to_string()).unwrap();
        let branding = crate::registry::Branding::default();
        regen_html(dir.path(), &branding, None).unwrap();
        let index = std::fs::read_to_string(dir.path().join("index.html")).unwrap();
        // The sniffed leaf extension must reach the generated viewer config.
        assert!(index.contains("png"), "leaf extension must reach the HTML");
        // World size for a single tile at zoom 0 is one TILE-sized canvas.
        assert!(
            index.contains("WORLD_W = 512"),
            "world extents must reach the HTML"
        );
    }

    /// The object shape (`{"files": [...], "max_zoom": M, "detail_depth": D}`)
    /// regen path must use the persisted `max_zoom`/`detail_depth` rather than
    /// sniffing them from directory names, and must sniff the leaf and pyramid
    /// extensions independently (mixed png leaves / avif pyramid).
    #[test]
    fn regen_html_object_shape_uses_persisted_zoom_fields_and_sniffs_both_exts() {
        let dir = tempfile::tempdir().unwrap();
        // Leaf zoom 1 (persisted) with png tiles; pyramid zoom 0 in avif.
        let leaf = dir.path().join("tiles").join("1").join("0");
        std::fs::create_dir_all(&leaf).unwrap();
        std::fs::write(leaf.join("0.png"), b"x").unwrap();
        let pyramid = dir.path().join("tiles").join("0").join("0");
        std::fs::create_dir_all(&pyramid).unwrap();
        std::fs::write(pyramid.join("0.avif"), b"x").unwrap();
        std::fs::write(
            dir.path().join("labels.json"),
            serde_json::json!({
                "files": [{"name": "a.bin", "x": 0, "y": 0, "size": 4}],
                "max_zoom": 1,
                "detail_depth": 1
            })
            .to_string(),
        )
        .unwrap();
        let branding = crate::registry::Branding::default();
        regen_html(dir.path(), &branding, None).unwrap();
        let index = std::fs::read_to_string(dir.path().join("index.html")).unwrap();
        // Leaf and pyramid extensions are interpolated independently.
        assert!(
            index.contains("? 'png' : 'avif'"),
            "mixed leaf/png + pyramid/avif extensions must reach the viewer: {index}"
        );
        // Persisted max_zoom wins over the deepest zoom dir minus detail_depth.
        assert!(
            index.contains("var MAX_ZOOM = 1;"),
            "persisted max_zoom must reach the viewer"
        );
        // detail_depth > 0 emits the variable-depth detail layer.
        assert!(
            index.contains("DetailTileLayer"),
            "detail_depth > 0 must emit the detail tile layer"
        );
        // Viewer headroom: max_zoom + detail_depth + 3.
        assert!(
            index.contains("maxZoom: 5"),
            "viewer maxZoom must include detail headroom"
        );
    }

    /// A labels.json that is neither an array, an object with `files`, nor a
    /// scenes object is rejected with a descriptive error instead of
    /// regenerating a broken viewer.
    #[test]
    fn regen_html_rejects_scalar_labels_json_shape() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("labels.json"), "42").unwrap();
        let branding = crate::registry::Branding::default();
        let err = regen_html(dir.path(), &branding, None).unwrap_err();
        assert!(
            err.to_string().contains("unexpected JSON shape"),
            "unexpected error: {err:#}"
        );
    }
}
