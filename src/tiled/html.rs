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
    height: u32,
    width: u32,
    tile_size: u32,
    entities: &[FileEntity],
    title: &str,
    inputs: &[String],
    leaf_ext: &str,
    pyramid_ext: &str,
) -> (Vec<u8>, Vec<u8>) {
    let entities_json = build_labels_json(entities);
    let html = build_html(
        world_w,
        world_h,
        max_zoom,
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
    height: u32,
    width: u32,
    tile_size: u32,
    entities: &[FileEntity],
    title: &str,
    inputs: &[String],
    leaf_ext: &str,
    pyramid_ext: &str,
) -> anyhow::Result<()> {
    let entities_json = build_labels_json(entities);
    std::fs::write(dir.join("labels.json"), &entities_json)?;

    let html = build_html(
        world_w,
        world_h,
        max_zoom,
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

/// Schema: `{ "files": [...] }`. The old bare-array schema is still readable
/// by the regen path.
fn build_labels_json(entities: &[FileEntity]) -> String {
    format!("{{\"files\":{}}}", entities_to_json(entities))
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
    height: u32,
    width: u32,
    tile_size: u32,
    title: &str,
    inputs: &[String],
    leaf_ext: &str,
    pyramid_ext: &str,
) -> String {
    let info_html = build_info_html(title, inputs);
    let viewer_max_zoom = max_zoom + 3;
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
      minZoom: 0,
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
      maxNativeZoom: {max_zoom},
      bounds: [[-{world_h}, 0], [0, {world_w}]],
      noWrap: true,
      attribution: '<a href="https://github.com/znation/arbvis">arbvis</a>'
    }}).addTo(map);
    map.fitBounds([[-{world_h}, 0], [0, {world_w}]]);

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
        viewer_max_zoom = viewer_max_zoom,
        world_w = world_w,
        world_h = world_h,
        height = height,
        width = width,
        leaf_ext = leaf_ext,
        pyramid_ext = pyramid_ext,
    )
}

#[cfg(test)]
mod tests {
    use super::hf_url_to_web;

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
