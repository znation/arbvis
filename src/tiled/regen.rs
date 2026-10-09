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
    html::write_leaflet_html_multi(tile_dir, &scenes, &branding.name, &[], branding)?;
    log::info!(
        "Regenerated index.html ({} scenes) in {}",
        scenes.len(),
        tile_dir.display()
    );
    Ok(true)
}

/// Regenerate `index.html` for an existing tiles directory without re-rendering tiles.
pub fn regen_html(tile_dir: &Path, branding: &Branding) -> anyhow::Result<()> {
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
    if regen_html_multi(tile_dir, &parsed, branding)? {
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
    let two_pow_mz = 1u32 << max_zoom;
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
        &branding.name,
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
