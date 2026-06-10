use std::path::Path;

/// Metadata for one file entity in the Leaflet viewer.
pub struct FileEntity {
    pub name: String,
    pub pixel_x: u32,
    pub pixel_y: u32,
    pub hue: u16,
    pub byte_size: u64,
    pub bbox: (u32, u32, u32, u32),
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
) -> anyhow::Result<()> {
    let entities_json = build_labels_json(entities, max_zoom, detail_depth);
    std::fs::write(dir.join("labels.json"), &entities_json)?;

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
    );
    std::fs::write(dir.join("index.html"), html)?;
    Ok(())
}

fn entities_to_json(entities: &[FileEntity]) -> String {
    let entries: Vec<String> = entities
        .iter()
        .map(|e| {
            let escaped = e.name.replace('\\', "\\\\").replace('"', "\\\"");
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

/// Convert an `hf://` path to its huggingface.co web URL, or return `None` for
/// non-hf paths.
///
/// hf://[type/]owner/repo[@rev]/path → https://huggingface.co/[type/]owner/repo/blob/rev/path
/// Trailing `/` in path → use `tree` instead of `blob`.
fn hf_url_to_web(s: &str) -> Option<String> {
    let rest = s.strip_prefix("hf://")?;
    let segs: Vec<&str> = rest.split('/').collect();
    if segs.len() < 2 {
        return None;
    }

    let (type_prefix, segs) = match segs[0] {
        "datasets" | "spaces" => (Some(segs[0]), &segs[1..]),
        "models" => (None, &segs[1..]),
        // Xet buckets and bare owner/repo/... (model default)
        _ => (None, &segs[..]),
    };

    if segs.len() < 2 {
        return None;
    }

    let owner = segs[0];
    let (repo, rev) = if let Some(at) = segs[1].find('@') {
        (&segs[1][..at], &segs[1][at + 1..])
    } else {
        (segs[1], "main")
    };

    let path_parts = &segs[2..];
    let base = if let Some(tp) = type_prefix {
        format!("https://huggingface.co/{tp}/{owner}/{repo}")
    } else {
        format!("https://huggingface.co/{owner}/{repo}")
    };

    if path_parts.is_empty() {
        return Some(base);
    }

    let path = path_parts.join("/");
    let verb = if rest.ends_with('/') { "tree" } else { "blob" };
    Some(format!("{base}/{verb}/{rev}/{path}"))
}

fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn build_info_html(title: &str, inputs: &[String]) -> String {
    let title_html = escape_html(title);
    let sources_html = if inputs.is_empty() {
        String::new()
    } else {
        let items: Vec<String> = inputs
            .iter()
            .map(|s| {
                let display = escape_html(s);
                if let Some(url) = hf_url_to_web(s) {
                    format!("<a href=\"{url}\" target=\"_blank\" rel=\"noopener\">{display}</a>")
                } else {
                    format!("<span>{display}</span>")
                }
            })
            .collect();
        format!("<div id=\"arbvis-sources\">{}</div>", items.join(", "))
    };
    format!(
        "<div id=\"arbvis-info\"><div id=\"arbvis-title\"><a href=\"https://github.com/znation/arbvis\" target=\"_blank\" rel=\"noopener\">{title_html}</a></div>{sources_html}</div>"
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
) -> String {
    let info_html = build_info_html(title, inputs);
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
      attribution: '<a href="https://github.com/znation/arbvis">arbvis</a>'
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
              html: l.name,
              iconSize: [tw, th],
              iconAnchor: [pt.x - lx, pt.y - ly]
            }}),
            interactive: false
          }}));
        }}
      }}
    }}

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
    pub label: String,
    pub order: u32,
    pub world_w: u32,
    pub world_h: u32,
    pub max_zoom: u32,
    pub detail_depth: u32,
    pub height: u32,
    pub width: u32,
    pub leaf_ext: String,
    pub pyramid_ext: String,
    pub entities: Vec<FileEntity>,
}

/// Quote + escape a string for embedding in JSON / JS source.
fn json_str(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

/// Scene-keyed labels JSON:
/// `{ "scenes": [ { key, label, order, world_w, world_h, width, height,
/// max_zoom, detail_depth, files: [...] }, ... ] }`. Per-scene geometry is
/// persisted so [`crate::tiled::regen_html`] can rebuild the viewer without
/// scanning the (now per-scene) tile directories.
fn build_labels_json_scenes(scenes: &[SceneView]) -> String {
    let arr: Vec<String> = scenes
        .iter()
        .map(|s| {
            format!(
                "{{\"key\":{key},\"label\":{label},\"order\":{order},\"world_w\":{ww},\"world_h\":{wh},\"width\":{w},\"height\":{h},\"max_zoom\":{mz},\"detail_depth\":{dd},\"leaf_ext\":{le},\"pyramid_ext\":{pe},\"files\":{files}}}",
                key = json_str(s.key.as_deref().unwrap_or("")),
                label = json_str(&s.label),
                order = s.order,
                ww = s.world_w,
                wh = s.world_h,
                w = s.width,
                h = s.height,
                mz = s.max_zoom,
                dd = s.detail_depth,
                le = json_str(&s.leaf_ext),
                pe = json_str(&s.pyramid_ext),
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
        .map(|s| {
            format!(
                "{{key:{key},label:{label},world_w:{ww},world_h:{wh},width:{w},height:{h},max_zoom:{mz},detail_depth:{dd},leaf_ext:{le},pyramid_ext:{pe}}}",
                key = json_str(s.key.as_deref().unwrap_or("")),
                label = json_str(&s.label),
                ww = s.world_w,
                wh = s.world_h,
                w = s.width,
                h = s.height,
                mz = s.max_zoom,
                dd = s.detail_depth,
                le = json_str(&s.leaf_ext),
                pe = json_str(&s.pyramid_ext),
            )
        })
        .collect();
    format!("[{}]", items.join(","))
}

/// Build the multi-scene viewer HTML. Scenes must be pre-sorted by `order`
/// (the first is the default-active layer).
fn build_html_multi(scenes: &[SceneView], title: &str, inputs: &[String]) -> String {
    let info_html = build_info_html(title, inputs);
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
        minNativeZoom: 0,
        maxNativeZoom: s.max_zoom,
        bounds: sceneBounds(s),
        noWrap: true,
        attribution: '<a href="https://github.com/znation/arbvis">arbvis</a>'
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
      baseLayers[SCENES[i].label] = grp;
      layerToScene.push({ layer: grp, scene: SCENES[i] });
    }

    var activeScene = SCENES[0];
    baseLayers[activeScene.label].addTo(map);
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
              html: l.name,
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
) -> anyhow::Result<()> {
    std::fs::write(dir.join("labels.json"), build_labels_json_scenes(scenes))?;
    std::fs::write(
        dir.join("index.html"),
        build_html_multi(scenes, title, inputs),
    )?;
    Ok(())
}

/// Multi-scene equivalent of [`generate_leaflet_content`] for the streaming
/// path: returns `(index.html bytes, labels.json bytes)` without touching disk.
pub fn generate_leaflet_content_multi(
    scenes: &[SceneView],
    title: &str,
    inputs: &[String],
) -> (Vec<u8>, Vec<u8>) {
    (
        build_html_multi(scenes, title, inputs).into_bytes(),
        build_labels_json_scenes(scenes).into_bytes(),
    )
}

#[cfg(test)]
mod tests {
    use super::{build_html, hf_url_to_web};

    /// Non-square canvas (8:1 tall) → the viewer allows zooming out past the
    /// pyramid root (`viewer_min_zoom < 0`). The base tile layer must carry the
    /// same `minZoom` as the map, otherwise GridLayer's default `minZoom: 0`
    /// blanks the layer at negative zooms and the fully-zoomed-out view shows
    /// the label overlay over an empty background (no tiles). Also assert the
    /// crisp-upscaling rule that keeps zoomed-in leaf tiles sharp.
    #[test]
    fn tall_canvas_tile_layer_renders_below_pyramid_root() {
        // world_h/world_w = 2949/366 ≈ 8.06 → viewer_min_zoom = -ceil(log2) = -4.
        let html = build_html(366, 2949, 2, 0, 11796, 1464, 512, "t", &[], "png", "avif");
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

    #[test]
    fn bare_model_repo() {
        assert_eq!(
            hf_url_to_web("hf://owner/repo"),
            Some("https://huggingface.co/owner/repo".to_string())
        );
    }

    #[test]
    fn bare_dataset_repo() {
        assert_eq!(
            hf_url_to_web("hf://datasets/owner/repo"),
            Some("https://huggingface.co/datasets/owner/repo".to_string())
        );
    }

    #[test]
    fn file_in_model_repo() {
        assert_eq!(
            hf_url_to_web("hf://owner/repo/model.safetensors"),
            Some("https://huggingface.co/owner/repo/blob/main/model.safetensors".to_string())
        );
    }

    #[test]
    fn file_in_dataset_repo() {
        assert_eq!(
            hf_url_to_web("hf://datasets/owner/repo/data.safetensors"),
            Some(
                "https://huggingface.co/datasets/owner/repo/blob/main/data.safetensors".to_string()
            )
        );
    }

    #[test]
    fn non_hf_url_returns_none() {
        assert_eq!(hf_url_to_web("/local/path/file.safetensors"), None);
        assert_eq!(hf_url_to_web("hf://owner"), None);
    }
}
