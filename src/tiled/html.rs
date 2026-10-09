//! Viewer `index.html` generation: embeds entity metadata, tile-URL template,
//! and branding into the Leaflet viewer template (single- and multi-scene).

use anyhow::{bail, Context};
use std::path::{Path, PathBuf};

use crate::registry::Branding;

/// Metadata for one file entity in the Leaflet viewer.
pub struct FileEntity {
    /// Display name shown on the viewer's label.
    pub name: String,
    /// Label anchor x in canvas pixels (rects centroid; curve midpoint when
    /// the rects are empty).
    pub pixel_x: u32,
    /// Label anchor y in canvas pixels (file-rect centroid).
    pub pixel_y: u32,
    /// Stable label hue derived from the name (`name_hue`, 0..360).
    pub hue: u16,
    /// Bytes this entity covers in the source stream.
    pub byte_size: u64,
    /// Bounding box over all the entity's rects, `(x0, y0, x1, y1)` in canvas
    /// pixels; `(0, 0, 0, 0)` for an entity with no rects.
    pub bbox: (u32, u32, u32, u32),
    /// Disjoint sub-rects the entity covers (canvas pixels) — a file whose
    /// Hilbert image is split by its neighbors yields several segments.
    pub segments: Vec<(u32, u32, u32, u32)>,
}

/// Generate HTML viewer and labels JSON as byte vectors without writing to disk.
///
/// `leaf_ext` and `pyramid_ext` are the file extensions for the tile format
/// emitted at the deepest zoom vs. the downsampled levels. They may differ:
/// e.g. leaves can be lossless AVIF while pyramid levels are lossy AVIF, or
/// leaves indexed-palette PNG while pyramid is AVIF — Leaflet's tileLayer URL
/// template uses a custom `getTileUrl` to switch on `z`.
pub fn generate_leaflet_content(
    world_w: u32,
    world_h: u32,
    max_zoom: u32,
    detail_depth: u32,
    height: u32,
    width: u32,
    tile_size: u32,
    entities: &[FileEntity],
    title: &str,
    inputs: &[String],
    leaf_ext: &str,
    pyramid_ext: &str,
    branding: &Branding,
) -> (Vec<u8>, Vec<u8>) {
    let entities_json = build_labels_json(entities, max_zoom, detail_depth);
    let html = build_html(
        world_w,
        world_h,
        max_zoom,
        detail_depth,
        height,
        width,
        tile_size,
        title,
        inputs,
        leaf_ext,
        pyramid_ext,
        branding,
    );
    (html.into_bytes(), entities_json.into_bytes())
}

/// Write Leaflet.js viewer HTML and entity labels JSON to `dir`.
pub fn write_leaflet_html(
    dir: &Path,
    world_w: u32,
    world_h: u32,
    max_zoom: u32,
    detail_depth: u32,
    height: u32,
    width: u32,
    tile_size: u32,
    entities: &[FileEntity],
    title: &str,
    inputs: &[String],
    leaf_ext: &str,
    pyramid_ext: &str,
    branding: &Branding,
) -> anyhow::Result<()> {
    let entities_json = build_labels_json(entities, max_zoom, detail_depth);
    let html = build_html(
        world_w,
        world_h,
        max_zoom,
        detail_depth,
        height,
        width,
        tile_size,
        title,
        inputs,
        leaf_ext,
        pyramid_ext,
        branding,
    );
    write_viewer_pair(dir, html.as_bytes(), entities_json.as_bytes())
}

/// Write the viewer's `index.html` and `labels.json` as a pair.
///
/// Both artifacts are staged to `<name>.part` siblings first, then renamed
/// into place, so a process killed mid-write (or an ENOSPC partway through)
/// leaves the previous complete pair instead of a truncated file that the
/// generated viewer serves as if complete. If either target path cannot be
/// replaced (it exists as a directory), the write fails before anything is
/// staged or sealed, leaving the previous pair untouched.
fn write_viewer_pair(dir: &Path, html: &[u8], labels: &[u8]) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir)
        .with_context(|| format!("creating viewer dir {}", dir.display()))?;
    let index = dir.join("index.html");
    let labels_path = dir.join("labels.json");
    for target in [&index, &labels_path] {
        if target.is_dir() {
            bail!(
                "cannot write viewer artifact {}: path exists as a directory",
                target.display()
            );
        }
    }
    let part = |p: &Path| -> PathBuf {
        let mut name = p.file_name().map(|n| n.to_os_string()).unwrap_or_default();
        name.push(".part");
        p.with_file_name(name)
    };
    let index_part = part(&index);
    let labels_part = part(&labels_path);

    let res: anyhow::Result<()> = (|| {
        std::fs::write(&index_part, html)
            .with_context(|| format!("staging {}", index_part.display()))?;
        std::fs::write(&labels_part, labels)
            .with_context(|| format!("staging {}", labels_part.display()))?;
        std::fs::rename(&index_part, &index)
            .with_context(|| format!("sealing {}", index.display()))?;
        std::fs::rename(&labels_part, &labels_path)
            .with_context(|| format!("sealing {}", labels_path.display()))?;
        Ok(())
    })();
    if res.is_err() {
        // Best effort: don't leave stale staging files behind.
        let _ = std::fs::remove_file(&index_part);
        let _ = std::fs::remove_file(&labels_part);
    }
    res
}

fn entities_to_json(entities: &[FileEntity]) -> String {
    let entries: Vec<String> = entities
        .iter()
        .map(|e| {
            let escaped = e
                .name
                .replace('\\', "\\\\")
                .replace('"', "\\\"")
                .replace("</", "<\\/");
            let segs: Vec<String> = e
                .segments
                .iter()
                .map(|&(x0, y0, x1, y1)| format!("[{},{},{},{}]", x0, y0, x1, y1))
                .collect();
            format!(
                "{{\"name\":\"{}\",\"x\":{},\"y\":{},\"hue\":{},\"size\":{},\"bbox\":[{}, {}, {}, {}],\"segs\":[{}]}}",
                escaped,
                e.pixel_x,
                e.pixel_y,
                e.hue,
                e.byte_size,
                e.bbox.0, e.bbox.1, e.bbox.2, e.bbox.3,
                segs.join(",")
            )
        })
        .collect();
    format!("[{}]", entries.join(","))
}

/// Schema: `{ "files": [...], "max_zoom": M, "detail_depth": D }`. The
/// `max_zoom`/`detail_depth` fields let [`crate::tiled::regen_html`] tell the
/// dense overview levels apart from the sparse variable-depth detail levels
/// (which otherwise look like extra zoom dirs and corrupt the derived
/// geometry). The old bare-array and `{files}`-only schemas are still readable
/// by the regen path (it falls back to detail_depth = 0).
fn build_labels_json(entities: &[FileEntity], max_zoom: u32, detail_depth: u32) -> String {
    format!(
        "{{\"files\":{},\"max_zoom\":{max_zoom},\"detail_depth\":{detail_depth}}}",
        entities_to_json(entities)
    )
}

fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// The leaflet attribution control's HTML: a link to the branded repo. Shared
/// by the single- and multi-scene viewers.
fn attribution_html(branding: &Branding) -> String {
    format!(
        "<a href=\"{}\">{}</a>",
        escape_html(&branding.repo_url),
        escape_html(&branding.name)
    )
}

fn build_info_html(title: &str, inputs: &[String], branding: &Branding) -> String {
    let title_html = escape_html(title);
    let repo_url = escape_html(&branding.repo_url);
    let sources_html = if inputs.is_empty() {
        String::new()
    } else {
        let items: Vec<String> = inputs
            .iter()
            .map(|s| {
                let display = escape_html(s);
                if let Some(url) = crate::hf_url::web_url(s) {
                    // The URL is built from the raw input string, so it can
                    // contain `"` or `>` — escape it too, or it breaks out of
                    // the href attribute and injects HTML into the viewer.
                    let url = escape_html(&url);
                    format!("<a href=\"{url}\" target=\"_blank\" rel=\"noopener\">{display}</a>")
                } else {
                    format!("<span>{display}</span>")
                }
            })
            .collect();
        format!("<div id=\"arbvis-sources\">{}</div>", items.join(", "))
    };
    format!(
        "<div id=\"arbvis-info\"><div id=\"arbvis-title\"><a href=\"{repo_url}\" target=\"_blank\" rel=\"noopener\">{title_html}</a></div>{sources_html}</div>"
    )
}

fn build_html(
    world_w: u32,
    world_h: u32,
    max_zoom: u32,
    detail_depth: u32,
    height: u32,
    width: u32,
    tile_size: u32,
    title: &str,
    inputs: &[String],
    leaf_ext: &str,
    pyramid_ext: &str,
    branding: &Branding,
) -> String {
    let info_html = build_info_html(title, inputs, branding);
    let attribution = attribution_html(branding);
    // Real tiles exist up to `max_zoom + detail_depth`; allow 3 more zoom
    // levels of CSS upsampling past that (the historical "+3" headroom).
    let viewer_max_zoom = max_zoom + detail_depth + 3;
    // Variable-depth detail layer: a second tile layer carrying source-resolution
    // tiles over shrunk tensors at zooms `max_zoom+1 ..= max_zoom+detail_depth`.
    // Missing (sparse) tiles fall through to the base layer's upsample via a
    // transparent `errorTileUrl`. Empty when nothing was shrunk.
    let detail_layer_js = if detail_depth > 0 {
        format!(
            r#"
    var DetailTileLayer = L.TileLayer.extend({{
      getTileUrl: function(coords) {{
        return 'tiles/' + coords.z + '/' + coords.x + '/' + coords.y + '.{leaf_ext}';
      }}
    }});
    new DetailTileLayer('', {{
      tileSize: {tile_size},
      minNativeZoom: {detail_min},
      maxNativeZoom: {detail_max},
      minZoom: {detail_min},
      bounds: [[-{world_h}, 0], [0, {world_w}]],
      noWrap: true,
      errorTileUrl: 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7',
    }}).addTo(map);
"#,
            leaf_ext = leaf_ext,
            tile_size = tile_size,
            detail_min = max_zoom + 1,
            detail_max = max_zoom + detail_depth,
            world_h = world_h,
            world_w = world_w,
        )
    } else {
        String::new()
    };
    // For non-square canvases the pyramid bottoms out with the smaller axis at
    // 1 tile and the larger axis at `aspect_max/aspect_min` tiles, so even at
    // Leaflet's zoom 0 we can't see the whole thing. Let the viewer keep
    // shrinking past the pyramid root by one zoom level per 2× aspect skew.
    // `minNativeZoom: 0` on the tile layer keeps tile fetches valid — Leaflet
    // CSS-scales the zoom-0 tiles for negative zooms.
    let aspect_max = world_w.max(world_h);
    let aspect_min = world_w.min(world_h).max(1);
    let viewer_min_zoom = -((aspect_max as f64 / aspect_min as f64).log2().ceil() as i32);
    format!(
        r#"<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{title_escaped}</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
        integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
        crossorigin=""/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
          integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo="
          crossorigin=""></script>
  <style>
    html, body, #map {{ height: 100%; margin: 0; padding: 0; }}
    .leaflet-right .leaflet-control {{ margin-right: 10px; }}
    .leaflet-control-attribution {{ box-sizing: border-box; }}
    /* One pixel is one byte/element. Past the deepest native zoom Leaflet
       CSS-upscales the leaf tiles; the default bilinear smoothing turns crisp
       per-element cells into a blurry wash. Force nearest-neighbour so zoomed-in
       tiles stay sharp (and honest about the underlying resolution). */
    .leaflet-tile {{ image-rendering: crisp-edges; image-rendering: pixelated; }}
    .file-label {{
      background: rgba(0,0,0,0.65);
      color: #ccc;
      padding: 2px 5px;
      font: 11px/1.4 monospace;
      white-space: nowrap;
      border-radius: 2px;
      pointer-events: none;
    }}
    #arbvis-info {{
      position: absolute;
      top: 10px;
      left: 50px;
      z-index: 1000;
      background: rgba(0,0,0,0.65);
      color: #ccc;
      padding: 6px 10px;
      font: 12px/1.5 monospace;
      border-radius: 3px;
      max-width: 60vw;
      pointer-events: none;
    }}
    #arbvis-info a {{ pointer-events: auto; color: #7af; text-decoration: none; }}
    #arbvis-info a:hover {{ text-decoration: underline; }}
    #arbvis-title {{ font-weight: bold; font-size: 13px; margin-bottom: 2px; }}
    #arbvis-title a {{ color: inherit; opacity: 0.7; }}
    #arbvis-title a:hover {{ opacity: 1; text-decoration: none; }}
    #arbvis-sources {{ font-size: 11px; color: #aaa; }}
  </style>
</head>
<body>
  {info_html}
  <div id="map"></div>
  <script>
    var map = L.map('map', {{
      crs: L.CRS.Simple,
      minZoom: {viewer_min_zoom},
      maxZoom: {viewer_max_zoom},
      preferCanvas: true,
    }});
    var ArbvisTileLayer = L.TileLayer.extend({{
      getTileUrl: function(coords) {{
        var ext = coords.z >= {max_zoom} ? '{leaf_ext}' : '{pyramid_ext}';
        return 'tiles/' + coords.z + '/' + coords.x + '/' + coords.y + '.' + ext;
      }}
    }});
    new ArbvisTileLayer('', {{
      tileSize: {tile_size},
      // GridLayer defaults `minZoom` to 0 and blanks the layer (no tiles
      // rendered) whenever the *map* zoom rounds below it. For non-square
      // canvases the viewer's min zoom is negative (`viewer_min_zoom`), so the
      // user can zoom out past the pyramid root to fit the whole canvas — but
      // with the default the base layer would go empty there, leaving only the
      // label overlay on a blank background. Pin the layer's `minZoom` to the
      // viewer min so it keeps rendering; `minNativeZoom: 0` still clamps the
      // actual tile fetches to zoom 0 and CSS-scales them down.
      minZoom: {viewer_min_zoom},
      minNativeZoom: 0,
      maxNativeZoom: {max_zoom},
      bounds: [[-{world_h}, 0], [0, {world_w}]],
      noWrap: true,
      attribution: '{attribution}'
    }}).addTo(map);
{detail_layer_js}
    map.fitBounds([[-{world_h}, 0], [0, {world_w}]]);
    // `viewer_min_zoom` lets the user zoom out past the pyramid root for
    // non-square canvases (so the whole canvas fits when the viewer aspect
    // doesn't match the viewport's), but using that as the *initial* zoom
    // makes tall/wide layouts (e.g. a wide multi-panel MoE-summary canvas in
    // a square-ish viewport) load as a tiny thin strip in a sea of empty space — the user
    // has to manually zoom in one or more levels before they see content.
    // Clamp the initial zoom at 0 (the pyramid root) so they land on a
    // usable view. Zooming out past 0 is still available manually.
    if (map.getZoom() < 0) {{
      map.setZoom(0);
    }}

    var HEIGHT = {height};
    var WIDTH = {width};
    var WORLD_W = {world_w};
    var WORLD_H = {world_h};
    var MAX_ZOOM = {max_zoom};

    var activeOverlays = L.layerGroup().addTo(map);

    function updateLabels(labels) {{
      var bounds = map.getBounds();
      var sw = bounds.getSouthWest();
      var ne = bounds.getNorthEast();
      // Geo↔pixel conversion factors. WORLD_W geo units span WIDTH canvas px
      // (and likewise for height), so canvas_x = lng * WIDTH / WORLD_W and
      // canvas_y = -lat * HEIGHT / WORLD_H. Hilbert canvases have
      // WORLD_W/WIDTH == WORLD_H/HEIGHT (uniform scaling) but arch canvases
      // can be non-square, so the two axes need separate ratios.
      var minX = sw.lng * WIDTH / WORLD_W;
      var minY = -ne.lat * HEIGHT / WORLD_H;
      var maxX = ne.lng * WIDTH / WORLD_W;
      var maxY = -sw.lat * HEIGHT / WORLD_H;

      var visible = [];
      for (var i = 0; i < labels.length; i++) {{
        var l = labels[i];
        var b = l.bbox;
        if (b[0] < maxX && b[2] > minX && b[1] < maxY && b[3] > minY) {{
          visible.push(l);
        }}
      }}

      visible.sort(function(a, b) {{ return b.size - a.size; }});
      if (visible.length > 1000) {{
        visible.length = 1000;
      }}

      activeOverlays.clearLayers();

      var placed = [];

      for (var i = 0; i < visible.length; i++) {{
        var l = visible[i];
        if (l.segs && l.segs.length > 0) {{
          // Viewport pixels per canvas pixel at the current zoom. At the leaf
          // zoom (MAX_ZOOM) a tile is rendered 1:1, so scale = 1; each level
          // out halves it. Independent of canvas aspect.
          var scale = Math.pow(2, map.getZoom() - MAX_ZOOM);
          var minWorld = 2 / scale;
          var ll = l.segs
            .filter(function(s) {{
              var len = Math.max(Math.abs(s[2] - s[0]), Math.abs(s[3] - s[1]));
              return len >= minWorld;
            }})
            .map(function(s) {{
              return [
                [-(s[1] / HEIGHT) * WORLD_H, (s[0] / WIDTH) * WORLD_W],
                [-(s[3] / HEIGHT) * WORLD_H, (s[2] / WIDTH) * WORLD_W],
              ];
            }});
          activeOverlays.addLayer(L.polyline(ll, {{
            color: 'hsl(' + l.hue + ',70%,60%)',
            weight: i < 3 ? 2 : 1,
            opacity: 0.9,
            fill: false,
            interactive: false,
          }}));
        }}
        var lat = -(l.y / HEIGHT) * WORLD_H;
        var lng =  (l.x / WIDTH) * WORLD_W;
        var pt = map.latLngToContainerPoint([lat, lng]);
        var tw = l.name.length * 7 + 12;
        var th = 22;
        var vw = map.getSize().x;
        var vh = map.getSize().y;
        var lx = Math.max(0, Math.min(pt.x - tw/2, vw - tw));
        var ly = Math.max(0, Math.min(pt.y - th/2, vh - th));
        var lb = {{ x: lx, y: ly, w: tw, h: th }};
        var overlaps = false;
        for (var j = 0; j < placed.length; j++) {{
          var p = placed[j];
          if (lb.x < p.x + p.w && lb.x + lb.w > p.x &&
              lb.y < p.y + p.h && lb.y + lb.h > p.y) {{
            overlaps = true;
            break;
          }}
        }}
        if (!overlaps) {{
          placed.push(lb);
          // Small dot at the true centroid anchors the label visually when
          // the label is clamped away from the centroid to stay on-screen.
          activeOverlays.addLayer(L.circleMarker([lat, lng], {{
            radius: 3,
            color: 'hsl(' + l.hue + ',70%,60%)',
            fillColor: 'hsl(' + l.hue + ',70%,60%)',
            fillOpacity: 1,
            weight: 0,
            interactive: false,
          }}));
          activeOverlays.addLayer(L.marker([lat, lng], {{
            icon: L.divIcon({{
              className: 'file-label',
              html: escHtml(l.name),
              iconSize: [tw, th],
              iconAnchor: [pt.x - lx, pt.y - ly]
            }}),
            interactive: false
          }}));
        }}
      }}
    }}

    // Entity names come from the visualized files (e.g. filenames listed in a
    // hostile Hub repo) and Leaflet's divIcon injects its `html:` value as
    // innerHTML — escape every name before it reaches one.
    var escHtml = function (s) {{
      return String(s).replace(/[&<>"']/g, function (c) {{
        return {{ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }}[c];
      }});
    }};
    fetch('labels.json')
      .then(function(r) {{ return r.json(); }})
      .then(function(data) {{
        // New schema: {{ files: [...] }}. Legacy schema: bare array of file entities.
        var files = Array.isArray(data) ? data : (data.files || []);
        function redraw() {{
          updateLabels(files);
        }}
        redraw();
        map.on('zoomend moveend', redraw);
      }});
  </script>
</body>
</html>"#,
        title_escaped = escape_html(title),
        info_html = info_html,
        attribution = attribution,
        max_zoom = max_zoom,
        detail_layer_js = detail_layer_js,
        viewer_max_zoom = viewer_max_zoom,
        viewer_min_zoom = viewer_min_zoom,
        world_w = world_w,
        world_h = world_h,
        height = height,
        width = width,
        leaf_ext = leaf_ext,
        pyramid_ext = pyramid_ext,
    )
}

// ===========================================================================
// Multi-scene viewer
//
// A *scene* is one independent tile pyramid under `tiles/<key>/`. When a render
// produces more than one (e.g. `modelweightvis --moe` → "summary" + "cka"), the
// viewer registers one Leaflet base layer per scene and a `L.control.layers`
// switcher ("tabs"). The single-scene path above is left untouched, so ordinary
// renders stay byte-for-byte identical.
// ===========================================================================

/// Per-scene geometry + entities handed to the multi-scene HTML/labels builder.
/// One is produced per tile pyramid by the tiler ([`crate::tiled::run_tiles`]).
pub struct SceneView {
    /// `Some(key)` → tiles live under `tiles/<key>/`; `None` → legacy `tiles/`
    /// (only used for the lone implicit default scene).
    pub key: Option<String>,
    /// Human-readable scene name shown in the viewer's tab / layer switcher.
    pub label: String,
    /// Tab ordering; the lowest-`order` scene is the default-active layer.
    pub order: u32,
    /// Leaflet world width at zoom 0.
    pub world_w: u32,
    /// Leaflet world height at zoom 0 (one `TILE` for the collapsed axis).
    pub world_h: u32,
    /// Deepest pyramid zoom level the tile grid serves.
    pub max_zoom: u32,
    /// Extra sparse detail zoom levels past `max_zoom` (0 for Hilbert).
    pub detail_depth: u32,
    /// Canvas height in pixels (power of two for Hilbert).
    pub height: u32,
    /// Canvas width in pixels (power of two for Hilbert).
    pub width: u32,
    /// File extension of the leaf tiles (`leaf_format.extension()`).
    pub leaf_ext: String,
    /// File extension of the downsampled pyramid tiles.
    pub pyramid_ext: String,
    /// Overlay entities written to this scene's `labels.json`.
    pub entities: Vec<FileEntity>,
}

/// Quote + escape a string for embedding in JSON / JS source. Neutralizes
/// `</` so a hostile value like `</script>` cannot prematurely close an
/// inline `<script>` block (same treatment as the 3D viewer's config JSON);
/// `<\/` is a valid JSON/JS escape for `/`.
fn json_str(s: &str) -> String {
    format!(
        "\"{}\"",
        s.replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace("</", "<\\/")
    )
}

/// Scene-keyed labels JSON:
/// `{ "scenes": [ { key, label, order, world_w, world_h, width, height,
/// max_zoom, detail_depth, files: [...] }, ... ] }`. Per-scene geometry is
/// persisted so [`crate::tiled::regen_html`] can rebuild the viewer without
/// scanning the (now per-scene) tile directories.
/// Comma-separated `name:value` serialization of the ten per-scene fields
/// shared by the labels-JSON object and the viewer's JS literal. Keys are
/// double-quoted when `quoted_keys` (JSON); bare otherwise (JS literal).
/// String values go through [`json_str`]; numeric values render bare.
fn scene_fields(s: &SceneView, quoted_keys: bool) -> String {
    let key = |name: &str| {
        if quoted_keys {
            format!("\"{name}\"")
        } else {
            name.to_string()
        }
    };
    [
        ("key", json_str(s.key.as_deref().unwrap_or(""))),
        ("label", json_str(&s.label)),
        ("world_w", s.world_w.to_string()),
        ("world_h", s.world_h.to_string()),
        ("width", s.width.to_string()),
        ("height", s.height.to_string()),
        ("max_zoom", s.max_zoom.to_string()),
        ("detail_depth", s.detail_depth.to_string()),
        ("leaf_ext", json_str(&s.leaf_ext)),
        ("pyramid_ext", json_str(&s.pyramid_ext)),
    ]
    .map(|(name, value)| format!("{}:{}", key(name), value))
    .join(",")
}

fn build_labels_json_scenes(scenes: &[SceneView]) -> String {
    let arr: Vec<String> = scenes
        .iter()
        .map(|s| {
            format!(
                "{{{fields},\"order\":{order},\"files\":{files}}}",
                fields = scene_fields(s, true),
                // Keep `order` adjacent to the identifying fields for humans;
                // JSON is parsed by key (regen.rs), so placement is free.
                order = s.order,
                files = entities_to_json(&s.entities),
            )
        })
        .collect();
    format!("{{\"scenes\":[{}]}}", arr.join(","))
}

/// JS array literal of per-scene descriptors for the viewer.
fn scenes_js_literal(scenes: &[SceneView]) -> String {
    let items: Vec<String> = scenes
        .iter()
        .map(|s| format!("{{{}}}", scene_fields(s, false)))
        .collect();
    format!("[{}]", items.join(","))
}

/// Build the multi-scene viewer HTML. Scenes must be pre-sorted by `order`
/// (the first is the default-active layer).
fn build_html_multi(
    scenes: &[SceneView],
    title: &str,
    inputs: &[String],
    branding: &Branding,
) -> String {
    let info_html = build_info_html(title, inputs, branding);
    // Map zoom envelope spanning every scene's pyramid + detail + upsample
    // headroom (the historical "+3"), and the most-negative aspect-fit zoom.
    let viewer_max_zoom = scenes
        .iter()
        .map(|s| s.max_zoom + s.detail_depth + 3)
        .max()
        .unwrap_or(3);
    let viewer_min_zoom = scenes
        .iter()
        .map(|s| {
            let aspect_max = s.world_w.max(s.world_h);
            let aspect_min = s.world_w.min(s.world_h).max(1);
            -((aspect_max as f64 / aspect_min as f64).log2().ceil() as i32)
        })
        .min()
        .unwrap_or(0);

    // Build via token replacement rather than `format!` — the Leaflet JS is
    // dense with literal `{`/`}` that `format!` would force us to double.
    TEMPLATE_MULTI
        .replace("/*__INFO__*/", &info_html)
        .replace("/*__SCENES__*/", &scenes_js_literal(scenes))
        .replace("/*__TILE__*/", &TILE_SIZE.to_string())
        .replace("/*__VMIN__*/", &viewer_min_zoom.to_string())
        .replace("/*__VMAX__*/", &viewer_max_zoom.to_string())
        .replace("__ATTRIBUTION__", &attribution_html(branding))
        .replace("__TITLE_ESCAPED__", &escape_html(title))
}

/// Tile edge length used by the viewer; matches [`crate::tiled::leaf::TILE`].
const TILE_SIZE: u32 = crate::tiled::leaf::TILE;

const TEMPLATE_MULTI: &str = r#"<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>__TITLE_ESCAPED__</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
        integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
        crossorigin=""/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
          integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo="
          crossorigin=""></script>
  <style>
    html, body, #map { height: 100%; margin: 0; padding: 0; }
    .leaflet-right .leaflet-control { margin-right: 10px; }
    .leaflet-control-attribution { box-sizing: border-box; }
    .leaflet-control-layers { font: 12px/1.4 monospace; }
    /* One pixel is one byte/element. Past the deepest native zoom Leaflet
       CSS-upscales the leaf tiles; the default bilinear smoothing turns crisp
       per-element cells into a blurry wash. Force nearest-neighbour so zoomed-in
       tiles stay sharp (and honest about the underlying resolution). */
    .leaflet-tile { image-rendering: crisp-edges; image-rendering: pixelated; }
    .file-label {
      background: rgba(0,0,0,0.65);
      color: #ccc;
      padding: 2px 5px;
      font: 11px/1.4 monospace;
      white-space: nowrap;
      border-radius: 2px;
      pointer-events: none;
    }
    #arbvis-info {
      position: absolute;
      top: 10px;
      left: 50px;
      z-index: 1000;
      background: rgba(0,0,0,0.65);
      color: #ccc;
      padding: 6px 10px;
      font: 12px/1.5 monospace;
      border-radius: 3px;
      max-width: 60vw;
      pointer-events: none;
    }
    #arbvis-info a { pointer-events: auto; color: #7af; text-decoration: none; }
    #arbvis-info a:hover { text-decoration: underline; }
    #arbvis-title { font-weight: bold; font-size: 13px; margin-bottom: 2px; }
    #arbvis-title a { color: inherit; opacity: 0.7; }
    #arbvis-title a:hover { opacity: 1; text-decoration: none; }
    #arbvis-sources { font-size: 11px; color: #aaa; }
  </style>
</head>
<body>
  /*__INFO__*/
  <div id="map"></div>
  <script>
    var SCENES = /*__SCENES__*/;
    var TILE = /*__TILE__*/;
    var TRANSPARENT = 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7';

    // Scene labels and entity names come from the visualized files (e.g. a
    // labels.json carried in a hostile Hub repo, or plugin-supplied scene
    // tags) and reach HTML sinks: Leaflet's divIcon and Control.Layers both
    // inject their text as innerHTML — escape every label before it reaches
    // one. Defined before the layers control is built so it is initialized
    // when first called.
    var escHtml = function (s) {
      return String(s).replace(/[&<>"']/g, function (c) {
        return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
      });
    };

    var map = L.map('map', {
      crs: L.CRS.Simple,
      minZoom: /*__VMIN__*/,
      maxZoom: /*__VMAX__*/,
      preferCanvas: true,
    });

    function sceneBounds(s) { return [[-s.world_h, 0], [0, s.world_w]]; }

    function makeBaseLayer(s) {
      var Base = L.TileLayer.extend({
        getTileUrl: function(c) {
          var ext = c.z >= s.max_zoom ? s.leaf_ext : s.pyramid_ext;
          return 'tiles/' + s.key + '/' + c.z + '/' + c.x + '/' + c.y + '.' + ext;
        }
      });
      var grp = L.layerGroup();
      grp.addLayer(new Base('', {
        tileSize: TILE,
        // Pin to the map's min zoom (the most-zoomed-out across all scenes) so a
        // non-square scene keeps rendering when the user zooms past its pyramid
        // root, instead of GridLayer's default `minZoom: 0` blanking the layer.
        // `minNativeZoom: 0` still clamps fetches to zoom 0 and CSS-scales them.
        minZoom: /*__VMIN__*/,
        minNativeZoom: 0,
        maxNativeZoom: s.max_zoom,
        bounds: sceneBounds(s),
        noWrap: true,
        attribution: '__ATTRIBUTION__'
      }));
      if (s.detail_depth > 0) {
        var Detail = L.TileLayer.extend({
          getTileUrl: function(c) {
            return 'tiles/' + s.key + '/' + c.z + '/' + c.x + '/' + c.y + '.' + s.leaf_ext;
          }
        });
        grp.addLayer(new Detail('', {
          tileSize: TILE,
          minNativeZoom: s.max_zoom + 1,
          maxNativeZoom: s.max_zoom + s.detail_depth,
          minZoom: s.max_zoom + 1,
          bounds: sceneBounds(s),
          noWrap: true,
          errorTileUrl: TRANSPARENT,
        }));
      }
      return grp;
    }

    var baseLayers = {};
    var layerToScene = [];
    for (var i = 0; i < SCENES.length; i++) {
      var grp = makeBaseLayer(SCENES[i]);
      baseLayers[escHtml(SCENES[i].label)] = grp;
      layerToScene.push({ layer: grp, scene: SCENES[i] });
    }

    var activeScene = SCENES[0];
    baseLayers[escHtml(activeScene.label)].addTo(map);
    L.control.layers(baseLayers, null, { collapsed: false }).addTo(map);

    function fitScene(s) {
      map.fitBounds(sceneBounds(s));
      if (map.getZoom() < 0) { map.setZoom(0); }
    }
    fitScene(activeScene);

    var activeOverlays = L.layerGroup().addTo(map);
    var filesByKey = {};

    function updateLabels() {
      var s = activeScene;
      var WIDTH = s.width, HEIGHT = s.height, WORLD_W = s.world_w, WORLD_H = s.world_h, MAX_ZOOM = s.max_zoom;
      var labels = filesByKey[s.key] || [];

      var bounds = map.getBounds();
      var sw = bounds.getSouthWest();
      var ne = bounds.getNorthEast();
      var minX = sw.lng * WIDTH / WORLD_W;
      var minY = -ne.lat * HEIGHT / WORLD_H;
      var maxX = ne.lng * WIDTH / WORLD_W;
      var maxY = -sw.lat * HEIGHT / WORLD_H;

      var visible = [];
      for (var i = 0; i < labels.length; i++) {
        var l = labels[i];
        var b = l.bbox;
        if (b[0] < maxX && b[2] > minX && b[1] < maxY && b[3] > minY) {
          visible.push(l);
        }
      }
      visible.sort(function(a, b) { return b.size - a.size; });
      if (visible.length > 1000) { visible.length = 1000; }

      activeOverlays.clearLayers();
      var placed = [];

      for (var i = 0; i < visible.length; i++) {
        var l = visible[i];
        if (l.segs && l.segs.length > 0) {
          var scale = Math.pow(2, map.getZoom() - MAX_ZOOM);
          var minWorld = 2 / scale;
          var ll = l.segs
            .filter(function(seg) {
              var len = Math.max(Math.abs(seg[2] - seg[0]), Math.abs(seg[3] - seg[1]));
              return len >= minWorld;
            })
            .map(function(seg) {
              return [
                [-(seg[1] / HEIGHT) * WORLD_H, (seg[0] / WIDTH) * WORLD_W],
                [-(seg[3] / HEIGHT) * WORLD_H, (seg[2] / WIDTH) * WORLD_W],
              ];
            });
          activeOverlays.addLayer(L.polyline(ll, {
            color: 'hsl(' + l.hue + ',70%,60%)',
            weight: i < 3 ? 2 : 1,
            opacity: 0.9,
            fill: false,
            interactive: false,
          }));
        }
        var lat = -(l.y / HEIGHT) * WORLD_H;
        var lng = (l.x / WIDTH) * WORLD_W;
        var pt = map.latLngToContainerPoint([lat, lng]);
        var tw = l.name.length * 7 + 12;
        var th = 22;
        var vw = map.getSize().x;
        var vh = map.getSize().y;
        var lx = Math.max(0, Math.min(pt.x - tw / 2, vw - tw));
        var ly = Math.max(0, Math.min(pt.y - th / 2, vh - th));
        var lb = { x: lx, y: ly, w: tw, h: th };
        var overlaps = false;
        for (var j = 0; j < placed.length; j++) {
          var p = placed[j];
          if (lb.x < p.x + p.w && lb.x + lb.w > p.x &&
              lb.y < p.y + p.h && lb.y + lb.h > p.y) {
            overlaps = true;
            break;
          }
        }
        if (!overlaps) {
          placed.push(lb);
          activeOverlays.addLayer(L.circleMarker([lat, lng], {
            radius: 3,
            color: 'hsl(' + l.hue + ',70%,60%)',
            fillColor: 'hsl(' + l.hue + ',70%,60%)',
            fillOpacity: 1,
            weight: 0,
            interactive: false,
          }));
          activeOverlays.addLayer(L.marker([lat, lng], {
            icon: L.divIcon({
              className: 'file-label',
              html: escHtml(l.name),
              iconSize: [tw, th],
              iconAnchor: [pt.x - lx, pt.y - ly]
            }),
            interactive: false
          }));
        }
      }
    }

    map.on('baselayerchange', function(e) {
      for (var i = 0; i < layerToScene.length; i++) {
        if (layerToScene[i].layer === e.layer) {
          activeScene = layerToScene[i].scene;
          break;
        }
      }
      fitScene(activeScene);
      updateLabels();
    });

    fetch('labels.json')
      .then(function(r) { return r.json(); })
      .then(function(data) {
        var scenes = data.scenes || [];
        for (var i = 0; i < scenes.length; i++) {
          filesByKey[scenes[i].key] = scenes[i].files || [];
        }
        updateLabels();
        map.on('zoomend moveend', updateLabels);
      });
  </script>
</body>
</html>"#;

/// Write the multi-scene Leaflet viewer + scene-keyed labels JSON to `dir`.
pub fn write_leaflet_html_multi(
    dir: &Path,
    scenes: &[SceneView],
    title: &str,
    inputs: &[String],
    branding: &Branding,
) -> anyhow::Result<()> {
    write_viewer_pair(
        dir,
        build_html_multi(scenes, title, inputs, branding).as_bytes(),
        build_labels_json_scenes(scenes).as_bytes(),
    )
}

/// Multi-scene equivalent of [`generate_leaflet_content`] for the streaming
/// path: returns `(index.html bytes, labels.json bytes)` without touching disk.
pub fn generate_leaflet_content_multi(
    scenes: &[SceneView],
    title: &str,
    inputs: &[String],
    branding: &Branding,
) -> (Vec<u8>, Vec<u8>) {
    (
        build_html_multi(scenes, title, inputs, branding).into_bytes(),
        build_labels_json_scenes(scenes).into_bytes(),
    )
}

#[cfg(test)]
mod tests {
    use super::{
        build_html, build_html_multi, build_info_html, build_labels_json_scenes, json_str, scenes_js_literal, scene_fields, Branding, FileEntity, SceneView,
    };
    fn scene(key: &str, world_w: u32, world_h: u32) -> SceneView {
        SceneView {
            key: Some(key.to_string()),
            label: key.to_string(),
            order: 0,
            world_w,
            world_h,
            max_zoom: 2,
            detail_depth: 0,
            height: world_h,
            width: world_w,
            leaf_ext: "png".to_string(),
            pyramid_ext: "avif".to_string(),
            entities: Vec::new(),
        }
    }

    /// The labels JSON and the inline JS literal are written by the same
    /// [`scene_fields`] serializer, so a bug there corrupts both consumers:
    /// `regen.rs` parses the JSON (double-quoted keys required) and the viewer
    /// evals the JS literal (bare identifiers). Both string-valued fields and
    /// hostile content must round-trip identically in each mode.
    #[test]
    fn scene_fields_json_and_js_literals_agree_field_for_field() {
        let sv = scene("k", 10, 20);
        let json = scene_fields(&sv, true);
        let js = scene_fields(&sv, false);
        for name in [
            "key", "label", "world_w", "world_h", "width", "height", "max_zoom", "detail_depth",
            "leaf_ext", "pyramid_ext",
        ] {
            assert!(json.contains(&format!("\"{name}\":")), "JSON key: {json}");
            assert!(js.contains(&format!("{name}:")), "JS key: {js}");
        }
        // Geometry and extensions must be identical in both encodings.
        assert!(json.contains("\"world_w\":10") && js.contains("world_w:10"));
        assert!(json.contains("\"leaf_ext\":\"png\"") && js.contains("leaf_ext:\"png\""));
    }

    /// A scene with no key (the legacy lone-pyramid layout, tagged scenes are
    /// the only other producer) serializes as the empty string in both the
    /// labels JSON and the JS literal — regen.rs keys scenes by this value.
    #[test]
    fn scene_fields_keyless_scene_serializes_empty_key() {
        let mut sv = scene("ignored", 10, 20);
        sv.key = None;
        let json = scene_fields(&sv, true);
        let js = scene_fields(&sv, false);
        assert!(json.starts_with("\"key\":\"\","), "{json}");
        assert!(js.starts_with("key:\"\","), "{js}");
    }

    /// `json_str` is the sink for every scene string that lands inside a JS
    /// string literal in the inline `var SCENES` block — a `</script>` in a
    /// label would terminate the inline script; a stray quote or backslash
    /// would corrupt the surrounding literal.
    #[test]
    fn json_str_escapes_quotes_backslashes_and_script_closers() {
        assert_eq!(json_str("plain"), "\"plain\"");
        assert_eq!(json_str("a\\b"), "\"a\\\\b\"");
        assert_eq!(json_str("say \"hi\""), "\"say \\\"hi\\\"\"");
        assert_eq!(json_str("</script>"), "\"<\\/script>\"");
        assert_eq!(json_str("</div>"), "\"<\\/div>\"");
    }

    /// The multi-scene labels JSON must parse as JSON (regen.rs reads it back)
    /// and carry each scene's order and entity list, with hostile entity names
    /// surviving the JSON round-trip byte-for-byte.
    #[test]
    fn multi_scene_labels_json_round_trips_through_serde() {
        let mut sv = scene("k", 10, 20);
        sv.order = 3;
        sv.entities = vec![FileEntity {
            name: "a\"b</script>c\\d".to_string(),
            pixel_x: 1,
            pixel_y: 2,
            hue: 180,
            byte_size: 7,
            bbox: (0, 0, 3, 4),
            segments: vec![(0, 0, 1, 1), (2, 2, 3, 3)],
        }];
        let json = build_labels_json_scenes(&[sv]);
        let v: serde_json::Value = serde_json::from_str(&json).expect("labels JSON parses");
        let s = &v["scenes"][0];
        assert_eq!(s["key"], "k");
        assert_eq!(s["order"], 3);
        assert_eq!(s["max_zoom"], 2);
        let e = &s["files"][0];
        assert_eq!(e["name"], "a\"b</script>c\\d");
        assert_eq!(e["segs"], serde_json::json!([[0, 0, 1, 1], [2, 2, 3, 3]]));
    }

    /// The inline JS literal is the viewer's scene table; unlike the JSON it
    /// uses bare identifier keys, and it must contain no top-level quotes on
    /// key names (the viewer indexes `SCENES[i].key` etc.).
    #[test]
    fn scenes_js_literal_uses_bare_keys_and_single_object_per_scene() {
        let scenes = [scene("a", 10, 20), scene("b", 30, 40)];
        let js = scenes_js_literal(&scenes);
        assert!(js.starts_with("[") && js.ends_with("]"));
        assert_eq!(js.matches("{key:").count(), 2);
        assert!(!js.contains("{\"key\""), "keys must be bare: {js}");
        assert!(js.contains("{key:\"a\",label:\"a\",world_w:10"), "{js}");
        assert!(js.contains("{key:\"b\",label:\"b\",world_w:30"), "{js}");
    }

    /// Scene labels reach Leaflet's `L.control.layers(baseLayers, …)`, which
    /// renders each layer's name as innerHTML — a hostile label (from a
    /// labels.json carried in an untrusted tiles bundle via `--regen-html`,
    /// or from a plugin's `SceneTag.label`) would inject markup into the
    /// viewer page. Every use of a scene label as a base-layer key must be
    /// wrapped in `escHtml` at the sink.
    #[test]
    fn scene_labels_are_html_escaped_before_leaflet_layers_control() {
        let evil = "<img src=x onerror=alert(1)>";
        let mut sv = scene(evil, 560, 64);
        sv.label = evil.to_string();
        let html = build_html_multi(&[sv], "t", &[], &Branding::default());
        // The label is interpolated into the inline `var SCENES` JSON, so its
        // raw text appears there inside a JS string — but the layers-control
        // sink must only ever see it wrapped in escHtml.
        assert!(
            html.contains("baseLayers[escHtml(SCENES[i].label)]"),
            "per-scene layer keys must be escHtml-wrapped: {html}"
        );
        assert!(
            html.contains("baseLayers[escHtml(activeScene.label)].addTo(map)"),
            "the default scene's layer key must be escHtml-wrapped: {html}"
        );
        assert!(
            !html.contains("baseLayers[SCENES[i].label]")
                && !html.contains("baseLayers[activeScene.label].addTo"),
            "no unwrapped scene label may reach the innerHTML sink: {html}"
        );
    }

    /// Multi-scene parity with [`tall_canvas_tile_layer_renders_below_pyramid_root`]:
    /// the tab switcher's base layers must carry the (global) map min zoom so a
    /// tall scene still renders when zoomed past its pyramid root, and the
    /// crisp-upscale rule must be present. With a tall 8:1 scene the global
    /// `viewer_min_zoom` is -4, used by the map + each scene's base layer.
    #[test]
    fn multi_scene_viewer_pins_minzoom_and_upscales_crisply() {
        let scenes = [scene("summary", 560, 64), scene("cka", 366, 2949)];
        let html = build_html_multi(&scenes, "t", &[], &Branding::default());
        let n = html.matches("minZoom: -4,").count();
        assert!(
            n >= 2,
            "expected the map and the per-scene base layer to set minZoom: -4, found {n}",
        );
        assert!(
            html.contains("image-rendering: pixelated"),
            "leaf tiles must upscale crisply past max_zoom",
        );
    }

    /// Entity names derive from the visualized files' filenames (repo-fetched
    /// names for `hf://` inputs) and Leaflet's divIcon injects its `html:`
    /// value as innerHTML — so the emitted viewer must route every label
    /// through the `escHtml` helper rather than interpolate the raw name.
    #[test]
    fn entity_labels_are_html_escaped_in_both_viewers() {
        let html = build_html(
            256,
            256,
            2,
            0,
            256,
            256,
            256,
            "t",
            &[],
            "png",
            "avif",
            &Branding::default(),
        );
        assert!(
            html.contains("var escHtml = function"),
            "single-scene viewer must define the escHtml helper"
        );
        assert!(
            html.contains("html: escHtml(l.name)"),
            "single-scene viewer must escape entity names before divIcon html"
        );
        assert!(
            !html.contains("html: l.name"),
            "no raw entity-name interpolation may remain in the single-scene viewer"
        );

        let scenes = [scene("s", 256, 256)];
        let multi = build_html_multi(&scenes, "t", &[], &Branding::default());
        assert!(
            multi.contains("var escHtml = function"),
            "multi-scene viewer must define the escHtml helper"
        );
        assert!(
            multi.contains("html: escHtml(l.name)"),
            "multi-scene viewer must escape entity names before divIcon html"
        );
        assert!(
            !multi.contains("html: l.name"),
            "no raw entity-name interpolation may remain in the multi-scene viewer"
        );
    }

    use super::write_leaflet_html_multi;
    use std::fs;
    use tempfile::tempdir;

    /// A failure while writing the viewer pair (here: `index.html` exists as a
    /// directory, so it cannot be replaced) must fail loudly without replacing
    /// labels.json — the previous pair stays intact, with no .part residue.
    #[test]
    fn failed_viewer_write_leaves_previous_pair_intact() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("labels.json"), b"OLD").unwrap();
        fs::create_dir(dir.path().join("index.html")).unwrap();
        let scenes = [scene("s", 256, 256)];
        let res = write_leaflet_html_multi(dir.path(), &scenes, "t", &[], &Branding::default());
        assert!(res.is_err(), "write into a directory path must fail loudly");
        assert_eq!(fs::read(dir.path().join("labels.json")).unwrap(), b"OLD");
        assert!(!dir.path().join("labels.json.part").exists());
        assert!(!dir.path().join("index.html.part").exists());
    }

    /// Non-square canvas (8:1 tall) → the viewer allows zooming out past the
    /// pyramid root (`viewer_min_zoom < 0`). The base tile layer must carry the
    /// same `minZoom` as the map, otherwise GridLayer's default `minZoom: 0`
    /// blanks the layer at negative zooms and the fully-zoomed-out view shows
    /// the label overlay over an empty background (no tiles). Also assert the
    /// crisp-upscaling rule that keeps zoomed-in leaf tiles sharp.
    #[test]
    fn tall_canvas_tile_layer_renders_below_pyramid_root() {
        // world_h/world_w = 2949/366 ≈ 8.06 → viewer_min_zoom = -ceil(log2) = -4.
        let html = build_html(
            366,
            2949,
            2,
            0,
            11796,
            1464,
            512,
            "t",
            &[],
            "png",
            "avif",
            &Branding::default(),
        );
        // One `minZoom: -4,` for the map options, one for the tile layer. The
        // detail layer (absent here, detail_depth = 0) would use its own value.
        let n = html.matches("minZoom: -4,").count();
        assert!(
            n >= 2,
            "expected both the map and the base tile layer to set minZoom: -4, found {n} occurrence(s)",
        );
        assert!(
            html.contains("image-rendering: pixelated"),
            "leaf tiles must upscale crisply past max_zoom",
        );
    }

    /// Custom branding must rebrand both the info-panel title link and the
    /// leaflet attribution, in both the single- and multi-scene viewers, with
    /// no leftover default arbvis URL.
    #[test]
    fn branding_overrides_repo_link_and_attribution() {
        let branding = Branding::new(
            "modelweightvis",
            "https://github.com/znation/modelweightvis",
        );

        let single = build_html(
            560,
            64,
            2,
            0,
            64,
            560,
            512,
            "t",
            &[],
            "png",
            "avif",
            &branding,
        );
        let multi = build_html_multi(&[scene("summary", 560, 64)], "t", &[], &branding);

        for html in [&single, &multi] {
            assert!(
                html.contains("https://github.com/znation/modelweightvis"),
                "branded repo URL must appear",
            );
            assert!(
                !html.contains("github.com/znation/arbvis"),
                "no default arbvis URL should remain",
            );
            assert!(
                html.contains(">modelweightvis</a>"),
                "attribution must use the branded name",
            );
        }
    }

    /// A scene label or key containing `</script>` must not break out of the
    /// inline `var SCENES = …` script block. Labels/keys can originate in
    /// data files (labels.json via regen) or from names inside untrusted
    /// input files via plugin scene tags, so the JSON string escaping used
    /// there must neutralize `</` (the 3D viewer does the same for its
    /// config JSON).
    #[test]
    fn hostile_scene_label_cannot_close_inline_script() {
        let evil = "</script><script>alert(1)</script>";
        let html = build_html_multi(
            &[SceneView {
                key: Some(evil.to_string()),
                label: evil.to_string(),
                order: 0,
                world_w: 560,
                world_h: 64,
                max_zoom: 2,
                detail_depth: 0,
                height: 64,
                width: 560,
                leaf_ext: "png".to_string(),
                pyramid_ext: "avif".to_string(),
                entities: Vec::new(),
            }],
            "t",
            &[],
            &Branding::default(),
        );
        let benign = build_html_multi(&[scene("ok", 560, 64)], "t", &[], &Branding::default());
        assert_eq!(
            html.matches("</script>").count(),
            benign.matches("</script>").count(),
            "the hostile label must not add closing </script> tags beyond the template's own: {html}"
        );
        assert!(
            html.contains("<\\/script>"),
            "`</` must be neutralized to `<\\/`: {html}"
        );
    }

    /// An hf:// input string containing `"` must not break out of the href
    /// attribute in the sources panel - the URL is built from the raw input,
    /// so it must be attribute-escaped before interpolation.
    #[test]
    fn hf_source_url_is_attribute_escaped() {
        let html = build_info_html(
            "t",
            &["hf://a\"b\"c/x\"><script>alert(1)</script>".to_string()],
            &Branding::default(),
        );
        assert!(
            !html.contains("\"><script>"),
            "raw `\"><script>` must not survive in the href context: {html}"
        );
        assert!(
            html.contains("&quot;"),
            "quotes in the URL must be escaped: {html}"
        );
    }
}
