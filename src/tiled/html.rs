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

/// Write Leaflet.js viewer HTML and entity labels JSON to `dir`.
pub fn write_leaflet_html(
    dir: &Path,
    world_w: u32,
    max_zoom: u32,
    height: u32,
    entities: &[FileEntity],
) -> anyhow::Result<()> {
    // Write entity data to a separate file so the HTML loads instantly and the
    // browser can fetch/process labels asynchronously after the map is visible.
    let entities_json: String = {
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
    };
    std::fs::write(dir.join("labels.json"), &entities_json)?;

    let html = format!(
        r#"<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>arbvis tiled</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
        integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
        crossorigin=""/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
          integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo="
          crossorigin=""></script>
  <style>
    html, body, #map {{ height: 100%; margin: 0; padding: 0; }}
    .file-label {{
      background: rgba(0,0,0,0.65);
      color: #ccc;
      padding: 2px 5px;
      font: 11px/1.4 monospace;
      white-space: nowrap;
      border-radius: 2px;
      pointer-events: none;
      transform: translate(-50%, -50%);
    }}
  </style>
</head>
<body>
  <div id="map"></div>
  <script>
    var map = L.map('map', {{
      crs: L.CRS.Simple,
      minZoom: 0,
      maxZoom: {max_zoom},
      preferCanvas: true,
    }});
    L.tileLayer('tiles/{{z}}/{{x}}/{{y}}.png', {{
      tileSize: 256,
      bounds: [[-256, 0], [0, {world_w}]],
      noWrap: true,
      attribution: 'arbvis'
    }}).addTo(map);
    map.fitBounds([[-256, 0], [0, {world_w}]]);

    var HEIGHT = {height};

    var activeOverlays = L.layerGroup().addTo(map);

    function updateLabels(labels) {{
      var bounds = map.getBounds();
      var sw = bounds.getSouthWest();
      var ne = bounds.getNorthEast();
      var minX = sw.lng * HEIGHT / 256;
      var minY = -ne.lat * HEIGHT / 256;
      var maxX = ne.lng * HEIGHT / 256;
      var maxY = -sw.lat * HEIGHT / 256;

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
          var ll = l.segs.map(function(s) {{
            return [
              [-(s[1] / HEIGHT) * 256, (s[0] / HEIGHT) * 256],
              [-(s[3] / HEIGHT) * 256, (s[2] / HEIGHT) * 256],
            ];
          }});
          activeOverlays.addLayer(L.polyline(ll, {{
            color: 'hsl(' + l.hue + ',70%,60%)',
            weight: 1,
            opacity: 0.9,
            fill: false,
            interactive: false,
          }}));
        }}
        var lat = -(l.y / HEIGHT) * 256;
        var lng =  (l.x / HEIGHT) * 256;
        var pt = map.latLngToContainerPoint([lat, lng]);
        var tw = l.name.length * 7 + 12;
        var th = 22;
        var lb = {{ x: pt.x - tw/2, y: pt.y - th/2, w: tw, h: th }};
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
          activeOverlays.addLayer(L.marker([lat, lng], {{
            icon: L.divIcon({{
              className: 'file-label',
              html: l.name,
              iconSize: [0, 0],
              iconAnchor: [0, 0]
            }}),
            interactive: false
          }}));
        }}
      }}
    }}

    fetch('labels.json')
      .then(function(r) {{ return r.json(); }})
      .then(function(labels) {{
        updateLabels(labels);
        map.on('zoomend moveend', function() {{ updateLabels(labels); }});
      }});
  </script>
</body>
</html>"#,
        max_zoom = max_zoom,
        world_w = world_w,
        height = height,
    );
    std::fs::write(dir.join("index.html"), html)?;
    Ok(())
}
