//! Standalone Three.js viewer for the 3D (`--3d`) bundle.
//!
//! The 3D analog of [`crate::tiled::html`]. Like the Leaflet viewer it is a
//! self-contained `index.html` built as a string and loads its rendering
//! library (here Three.js) from a CDN via an ES-module import map, so it
//! deploys to an HF Space with no build step. At runtime it fetches
//! `meta.json` and `volume.bin` (written alongside it by
//! [`crate::volume::render_volume`]).

use crate::registry::Branding;

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

const TEMPLATE: &str = r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>arbvis 3D</title>
<style>
  :root { color-scheme: dark; }
  html, body { margin: 0; height: 100%; background: #0a0b0e; color: #e8e8ea;
    font: 13px/1.4 system-ui, -apple-system, Segoe UI, Roboto, sans-serif; overflow: hidden; }
  #c { position: fixed; inset: 0; display: block; }
  #panel { position: fixed; top: 12px; left: 12px; width: 248px; padding: 12px 14px;
    background: rgba(18,20,26,.82); border: 1px solid #2a2d36; border-radius: 10px;
    backdrop-filter: blur(6px); box-shadow: 0 6px 24px rgba(0,0,0,.4); }
  #panel h1 { font-size: 14px; margin: 0 0 2px; font-weight: 600; }
  #panel a { color: #8ab4ff; text-decoration: none; }
  #panel .sub { color: #9aa0ab; font-size: 11px; margin-bottom: 10px; word-break: break-all; }
  .row { margin: 9px 0; }
  .row label { display: block; color: #c4c8d0; margin-bottom: 3px; font-size: 11px;
    display: flex; justify-content: space-between; }
  .row input[type=range] { width: 100%; accent-color: #8ab4ff; }
  .seg { display: flex; gap: 4px; }
  .seg button { flex: 1; background: #20232c; border: 1px solid #2f333d; color: #c4c8d0;
    padding: 4px 0; border-radius: 6px; cursor: pointer; font-size: 11px; }
  .seg button.on { background: #2f6bff; border-color: #2f6bff; color: #fff; }
  .hint { color: #80858f; font-size: 10px; margin-top: 8px; }
  #legend { margin-top: 10px; }
  #legend .bar { height: 10px; border-radius: 3px; margin: 3px 0; }
  #legend .lab { display: flex; justify-content: space-between; color: #80858f; font-size: 10px; }
  #legend .swrow { color: #c4c8d0; font-size: 11px; margin: 2px 0; }
  #legend .sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px;
    vertical-align: middle; margin-right: 5px; border: 1px solid #00000040; }
  #picked { color: #e8e8ea; font-size: 11px; margin-top: 10px; min-height: 2.4em; }
  #picked .pname { word-break: break-all; }
  #picked .pgroup { color: #9aa0ab; }
  #picked .phint { color: #80858f; }
  #status { position: fixed; inset: 0; display: flex; align-items: center; justify-content: center;
    color: #9aa0ab; pointer-events: none; }
  #status[hidden] { display: none; } /* id selector's display:flex would otherwise beat [hidden] */
  .load-box { display: flex; flex-direction: column; align-items: center; gap: 10px; }
  .load-track { width: 220px; height: 4px; background: #2a2d36; border-radius: 2px; overflow: hidden; }
  #load-bar { width: 0; height: 100%; background: #9aa0ab; border-radius: 2px; transition: width .15s ease-out; }
  .load-stats { color: #6b7078; font-size: 11px; font-variant-numeric: tabular-nums; min-height: 14px; }
  /* CSS2D layer labels float at each depth slab; the overlay must not eat pointer
     events or it would steal them from OrbitControls. */
  #labels { position: fixed; inset: 0; pointer-events: none; overflow: hidden; }
  .layer-label { pointer-events: none; white-space: nowrap; font-size: 11px; font-weight: 600;
    color: #e8e8ea; background: rgba(18,20,26,.78); border: 1px solid #2a2d36;
    border-radius: 6px; padding: 2px 7px; box-shadow: 0 2px 8px rgba(0,0,0,.4);
    transform: translate(-50%, -50%); }
  /* Per-voxel hover tooltip — follows the cursor, never intercepts it. */
  #tip { position: fixed; z-index: 10; pointer-events: none; max-width: 260px;
    padding: 6px 8px; background: rgba(18,20,26,.92); border: 1px solid #2a2d36;
    border-radius: 7px; box-shadow: 0 4px 16px rgba(0,0,0,.5); font-size: 11px; }
  #tip[hidden] { display: none; }
  #tip .tname { color: #e8e8ea; word-break: break-all; }
  #tip .tmeta { color: #9aa0ab; margin-top: 2px; }
  /* Background brick-streaming activity HUD — bottom-left, appears only while the
     view is still sharpening (demand-driven bricks streaming in), hidden at full res. */
  #bg { position: fixed; left: 12px; bottom: 12px; width: 200px; z-index: 5;
    padding: 8px 10px; pointer-events: none;
    background: rgba(18,20,26,.82); border: 1px solid #2a2d36; border-radius: 10px;
    backdrop-filter: blur(6px); box-shadow: 0 6px 24px rgba(0,0,0,.4);
    transition: opacity .3s ease-out; }
  #bg[hidden] { display: none; }
  #bg.fading { opacity: 0; }
  #bg .bg-row { display: flex; justify-content: space-between; align-items: baseline;
    color: #c4c8d0; font-size: 11px; margin-bottom: 6px; }
  #bg #bg-pct { color: #80858f; font-variant-numeric: tabular-nums; }
  #bg .load-track { width: 100%; } /* override the fixed 220px shared track to fit the card */
  #bg-bar { width: 0; height: 100%; background: #9aa0ab; border-radius: 2px;
    transition: width .15s ease-out; }
  #bg.warn #bg-bar { background: #d0a24a; }
  #bg.warn #bg-title { color: #d0a24a; }
  #bg .bg-sub { color: #6b7078; font-size: 10px; font-variant-numeric: tabular-nums;
    margin-top: 5px; min-height: 12px; }
</style>
</head>
<body>
<canvas id="c"></canvas>
<div id="labels"></div>
<div id="tip" hidden></div>
<div id="bg" hidden>
  <div class="bg-row"><span id="bg-title">sharpening…</span><span id="bg-pct"></span></div>
  <div class="load-track"><div id="bg-bar"></div></div>
  <div id="bg-eta" class="bg-sub"></div>
  <div id="bg-counts" class="bg-sub"></div>
</div>
<div id="status">
  <div class="load-box">
    <div class="load-label" id="load-label">loading…</div>
    <div class="load-track" id="load-track"><div id="load-bar"></div></div>
    <div class="load-stats" id="load-stats"></div>
  </div>
</div>
<div id="panel" hidden>
  <h1 id="title">arbvis</h1>
  <div class="sub"><a id="repo" href="#" target="_blank" rel="noopener"></a></div>
  <div id="ctl">
    <div class="row">
      <label>Opacity source</label>
      <div class="seg" id="src">
        <button data-v="0" class="on">Activity</button>
        <button data-v="1">Density</button>
      </div>
    </div>
    <div class="row"><label>Volume opacity <span id="opv"></span></label>
      <input id="opacity" type="range" min="0" max="1" step="0.01" value="0.45"></div>
    <div class="row"><label>Volume contrast <span id="gav"></span></label>
      <input id="gamma" type="range" min="0.2" max="3" step="0.05" value="1"></div>
    <div class="row"><label>Threshold <span id="thv"></span></label>
      <input id="threshold" type="range" min="0" max="0.95" step="0.01" value="0"></div>
  </div>
  <div id="inputs" class="hint"></div>
  <div id="legend"></div>
  <div id="picked"><span class="phint">click a region to identify</span></div>
  <div class="hint">drag rotate · right-drag pan · scroll zoom · hover or click to identify</div>
</div>

<script type="importmap">
{ "imports": {
  "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
  "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
}}
</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';

const CFG = __CONFIG_JSON__;
const $ = (id) => document.getElementById(id);

// ---- load progress ----------------------------------------------------------
// The overlay bar tracks the up-front blocking fetches in load() (volume.bin,
// the page table, and — non-streamed — the whole atlas). Its denominator is
// computed from meta.json sizes; demand-driven brick streaming runs after the
// overlay hides and is deliberately excluded (it never "completes").
let loadTotal = 0, loadDone = 0, loadStartMs = 0, loadTimer = null;
function setProgress() {
  const bar = $('load-bar');
  if (bar) bar.style.width = (loadTotal ? Math.min(100, loadDone / loadTotal * 100) : 0) + '%';
}
// Compact "12.3s" / "1m05s" duration formatting; sub-10s keeps one decimal so a
// fast load still shows motion.
function fmtDur(s) {
  if (!isFinite(s) || s < 0) return '—';
  if (s < 10) return s.toFixed(1) + 's';
  if (s < 60) return Math.round(s) + 's';
  return Math.floor(s / 60) + 'm' + String(Math.round(s % 60)).padStart(2, '0') + 's';
}
// Elapsed + a byte-rate ETA extrapolated from progress so far. The ETA appears
// as soon as any bytes land (loadDone > 0); it's inherently brief now that the
// up-front payload is a small coarse grid + octree (the fine detail streams
// after the overlay hides). At loadDone === loadTotal it reads "~0.0s left" for
// at most one 200ms tick before load() stops the clock — harmless.
function updateLoadStats() {
  const el = $('load-stats');
  if (!el || !loadStartMs) return;
  const elapsed = (Date.now() - loadStartMs) / 1000;
  let txt = fmtDur(elapsed) + ' elapsed';
  if (loadTotal && loadDone > 0) {
    txt += ' · ~' + fmtDur(elapsed * Math.max(0, loadTotal - loadDone) / loadDone) + ' left';
  }
  el.textContent = txt;
}
function startLoadClock() {
  loadStartMs = Date.now();
  if (!loadTimer) loadTimer = setInterval(updateLoadStats, 200);
  updateLoadStats();
}
function stopLoadClock() {
  if (loadTimer) { clearInterval(loadTimer); loadTimer = null; }
}
function setStatus(msg) {
  stopLoadClock();
  const label = $('load-label');
  if (label) label.textContent = msg;
  const track = $('load-track');
  if (track) track.hidden = true;
  const stats = $('load-stats');
  if (stats) stats.hidden = true;
}
// Stream a response body, feeding each chunk into the global progress. Falls
// back to a whole-buffer read when the body isn't a readable stream. Returns a
// Uint8Array (callers needing an ArrayBuffer use .buffer — byteOffset is 0).
async function fetchBytes(url, opts) {
  // Retry rate-limit / transient responses before failing. The up-front assets
  // are few, but a reload while the HF Spaces router is still throttling a prior
  // brick storm can 429 here — without a retry the whole load would error out
  // loud instead of just waiting out the window. Bounded attempts with
  // decorrelated-jitter backoff (mirrors the brick streamer's backoff).
  let res, wait = 0;
  for (let attempt = 0; ; attempt++) {
    if (wait) await new Promise((r) => setTimeout(r, wait));
    res = await fetch(url, opts);
    const throttled = res.status === 429 || res.status >= 500;
    if (!throttled || attempt >= 6) break;
    const base = wait || BACKOFF_MIN_MS;
    wait = Math.min(BACKOFF_MAX_MS, Math.floor(base * (1 + Math.random() * 2)));
  }
  // Fail loud: a persistent 5xx/OOM mid-load must surface as an error overlay
  // (via load()'s .catch → setStatus), not silently yield empty bytes and
  // render a blank cube.
  if (!res.ok) throw new Error('load failed: ' + res.status + ' ' + res.statusText + ' (' + url + ')');
  if (!res.body) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const chunks = []; let n = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); n += value.length; loadDone += value.length; setProgress();
  }
  const out = new Uint8Array(n); let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}

// ---- branding / info panel --------------------------------------------------
$('title').textContent = CFG.title || CFG.brandName || 'arbvis';
document.title = (CFG.title || 'arbvis') + ' 3D';
const repo = $('repo');
repo.textContent = CFG.brandName || 'arbvis';
repo.href = CFG.repoUrl || '#';
if (CFG.inputs && CFG.inputs.length) {
  $('inputs').textContent = CFG.inputs.length === 1
    ? CFG.inputs[0]
    : CFG.inputs.length + ' inputs';
}

// ---- scene ------------------------------------------------------------------
const canvas = $('c');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0b0e);
const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 100);
camera.position.set(1.4, 1.1, 1.6);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.target.set(0, 0, 0);

// DOM-overlay renderer for the per-layer text labels. Its element sits over the
// canvas; #labels is pointer-events:none so OrbitControls keeps the pointer.
const labelRenderer = new CSS2DRenderer({ element: $('labels') });

// faint bounding cube for orientation
const edges = new THREE.LineSegments(
  new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)),
  new THREE.LineBasicMaterial({ color: 0x3a3f4b }));
scene.add(edges);

// ---- hybrid compositing ----------------------------------------------------
// The volume view is drawn in three passes per frame (see renderHybrid): (1)
// the opaque scene (the orientation-cube edges) into sceneTarget, which carries
// a depth texture; (2) a verbatim blit of that color to the canvas; (3) the
// volume ray-march composited over it, with rays cut off at the opaque depth so
// the near cube edges correctly occlude the volume behind them. The volume
// lives on layer 1 so each pass can select edges-only / volume-only by camera
// layer rather than juggling per-object visibility.
const VOL_LAYER = 1;
let sceneTarget = null;
let volMesh = null; // the volume box mesh; declared here so resize() can reach it
// Brick-streaming re-evaluation flag. Declared here (not with the brick
// state below) so resize() can re-arm it: a resize that first gives the
// viewport real dimensions must re-run the feedback pass, which a prior
// zero-size frame would otherwise have consumed and left quiet.
let volumeDirty = true;
const _dbSize = new THREE.Vector2();
function makeSceneTarget(w, h) {
  const dt = new THREE.DepthTexture(w, h);
  dt.type = THREE.UnsignedIntType;
  // sRGB storage so edge/line colors match the old direct-to-canvas path; the
  // blit reads verbatim.
  const rt = new THREE.WebGLRenderTarget(w, h, {
    minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter,
    depthTexture: dt, depthBuffer: true, stencilBuffer: false,
    colorSpace: THREE.SRGBColorSpace,
  });
  return rt;
}
// Fullscreen blit of sceneTarget's color to the canvas. A custom GLSL3 shader
// (no colorspace/tonemap injection) copies the stored bytes verbatim, matching
// the original look exactly.
const copyMat = new THREE.ShaderMaterial({
  glslVersion: THREE.GLSL3, depthTest: false, depthWrite: false,
  uniforms: { tColor: { value: null } },
  vertexShader: 'out vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }',
  fragmentShader: 'precision highp float; uniform sampler2D tColor; in vec2 vUv; out vec4 o; void main(){ o = texture(tColor, vUv); }',
});
const copyScene = new THREE.Scene();
const copyCam = new THREE.Camera();
const copyQuad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), copyMat);
copyQuad.frustumCulled = false;
copyScene.add(copyQuad);

function resize() {
  const w = innerWidth, h = innerHeight;
  // Bail on a zero/degenerate viewport. On a cold (newly-restarted) Space the
  // deferred CDN module can run before the iframe's layout settles, so this
  // fires once with 0×0 — building sceneTarget at 0×0 and setting a NaN camera
  // aspect would blit an empty canvas (blank black) until the next resize. The
  // ResizeObserver below re-invokes us with real dimensions once they exist.
  if (!w || !h) return;
  // updateStyle must stay on: a bare <canvas> with `inset:0` is a replaced
  // element, so CSS won't stretch it — without an inline style size it renders
  // at its (devicePixelRatio-scaled) drawing-buffer size, overflowing the
  // viewport and throwing off OrbitControls' pointer mapping.
  renderer.setSize(w, h);
  labelRenderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  // (Re)size the offscreen target to the drawing buffer so gl_FragCoord lines up
  // with the depth lookup. Recreate it (a DepthTexture can't be resized in place).
  renderer.getDrawingBufferSize(_dbSize);
  if (sceneTarget) { sceneTarget.depthTexture.dispose(); sceneTarget.dispose(); }
  sceneTarget = makeSceneTarget(_dbSize.x, _dbSize.y);
  copyMat.uniforms.tColor.value = sceneTarget.texture;
  if (volMesh) volMesh.material.uniforms.uSceneDepth.value = sceneTarget.depthTexture;
  // Re-arm the feedback probe: if an earlier zero-size frame ran it with H=0
  // (streaming nothing) it left it quiet, so without this the view stays empty
  // until the camera first moves.
  volumeDirty = true;
}
addEventListener('resize', resize);
// A window 'resize' only fires on *changes*, never for the initial layout, so it
// can't rescue a first paint that raced ahead of the iframe getting its size. A
// ResizeObserver delivers an initial observation once the element is laid out
// (and every change after), which is what actually recovers the cold-start race.
new ResizeObserver(resize).observe(document.documentElement);
resize();

// ---- shaders ----------------------------------------------------------------
const volVert = `
  out vec3 vOrigin;
  out vec3 vDirection;
  void main() {
    vOrigin = (inverse(modelMatrix) * vec4(cameraPosition, 1.0)).xyz;
    vDirection = position - vOrigin;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }`;

const volFrag = `
  precision highp float;
  precision highp sampler3D;
  uniform sampler3D uBricks;      // brick-pool atlas / streamed cache (RGBA8)
  uniform sampler3D uPageTable;   // flat page table (non-streamed) — see pageCell()
  uniform sampler3D uNodePool;    // sparse octree node pool (streamed) — see descend()
  uniform sampler3D uCoarse;      // dense low-res fallback (streamed mode only)
  uniform sampler2D uLut;
  uniform sampler2D uSceneDepth; // depth of the opaque (edges) pass — ray cutoff
  uniform vec2 uResolution;      // drawing-buffer size, for the gl_FragCoord lookup
  uniform mat4 uInvProjView, uInvModel; // reconstruct the scene hit in box space
  uniform float uOpacity, uGamma, uThreshold, uNorm, uBrick, uBrickStride, uApron;
  // Density (B) rescale for a raw-count streamed atlas: 255/max_count, or 0 when
  // B is already baked (structured, non-streamed, coarse fallback, old bundles).
  uniform float uAtlasInvMax;
  uniform vec3 uNodePoolDim;     // octree node-pool dims in texels (2/node/axis)
  uniform float uTreeDepth;      // octree depth D (bricks per side = exp2(D))
  uniform int uSource, uSteps, uDirectColor, uStreamed;
  // Box world size per axis (longest axis = 1; isotropic cube = vec3(1)).
  uniform vec3 uSize;
  uniform vec3 uVolDim;       // volume voxel dims [ex,ey,ez]
  uniform vec3 uPageDim;      // page-table dims (bricks per axis)
  uniform vec3 uAtlasBricks;  // atlas/cache size in bricks per axis
  in vec3 vOrigin;
  in vec3 vDirection;
  out vec4 fragColor;

  vec2 hitBox(vec3 o, vec3 d) {
    vec3 h = uSize * 0.5;
    vec3 inv = 1.0 / d;
    vec3 a = (-h - o) * inv;
    vec3 b = ( h - o) * inv;
    vec3 tmin = min(a, b), tmax = max(a, b);
    return vec2(max(tmin.x, max(tmin.y, tmin.z)), min(tmax.x, min(tmax.y, tmax.z)));
  }

  // Decode a page-table cell. Returns the RGB payload (24-bit) and the A-channel
  // state byte. Non-streamed: A is unused (255), payload is the 1-based atlas
  // slot (0 = empty). Streamed: A tags the cell — 0 empty, 1 occupied but not
  // resident, 2 resident — and payload is the 0-based cache slot when resident.
  uint pageCell(vec3 cell, out uint state) {
    vec4 c = texture(uPageTable, (cell + 0.5) / uPageDim);
    state = uint(c.a * 255.0 + 0.5);
    return uint(c.r * 255.0 + 0.5) + (uint(c.g * 255.0 + 0.5) << 8u) + (uint(c.b * 255.0 + 0.5) << 16u);
  }

  // Read child octant o (each component 0/1) of octree node "node" from the node
  // pool. Each node is a 2x2x2 texel block; nodes are laid out near-cubically
  // (2 texels/node/axis). Returns the 24-bit RGB payload; outputs the A-state
  // byte and the texel it lives at (for the streaming feedback probe).
  uint readChild(float node, vec3 o, out uint a, out vec3 texel) {
    vec3 npa = uNodePoolDim * 0.5;                 // nodes per axis
    vec3 nb = vec3(mod(node, npa.x), mod(floor(node / npa.x), npa.y), floor(node / (npa.x * npa.y)));
    texel = nb * 2.0 + o;
    vec4 c = texture(uNodePool, (texel + 0.5) / uNodePoolDim);
    a = uint(c.a * 255.0 + 0.5);
    return uint(c.r * 255.0 + 0.5) + (uint(c.g * 255.0 + 0.5) << 8u) + (uint(c.b * 255.0 + 0.5) << 16u);
  }

  // Descend the sparse octree to the terminal node containing brick "cell".
  // Returns the payload (brick id when not resident, cache slot when resident;
  // 0 when empty) and, via out params: state (0 empty / 1 occupied-not-resident
  // / 2 resident), the terminal node's brick-space box [boxOrigin, +boxExtent)
  // for coarse empty-space leaps, and the leaf entry's pool texel (feedback).
  // Leaf vs internal is read straight off the entry: A>0 => leaf, else RGB>0 =>
  // internal pointer, else empty — so the walk is depth-agnostic (bounded by D).
  uint descend(vec3 cell, out uint state, out vec3 boxOrigin, out float boxExtent, out vec3 texel) {
    float node = 0.0;                 // root
    vec3 origin = vec3(0.0);
    float extent = exp2(uTreeDepth);  // bricks per side
    for (int d = 0; d < 24; d++) {
      float halfE = extent * 0.5;
      vec3 o = step(origin + halfE, cell + 0.5);   // octant per axis (0/1)
      vec3 childOrigin = origin + o * halfE;
      uint a; uint rgb = readChild(node, o, a, texel);
      if (a > 0u) { state = a; boxOrigin = childOrigin; boxExtent = 1.0; return rgb; }   // leaf brick
      if (rgb == 0u) { state = 0u; boxOrigin = childOrigin; boxExtent = halfE; return 0u; } // empty subtree
      node = float(rgb - 1u); origin = childOrigin; extent = halfE;                       // internal → descend
    }
    state = 0u; boxOrigin = cell; boxExtent = 1.0; return 0u;
  }

  // Sample the brick atlas/cache at a volume voxel position, for a 0-based slot.
  // Each stored brick is uBrickStride (= uBrick + 2*uApron) wide; the brick's
  // own voxels start at the apron offset, and linear filtering stays inside the
  // brick's apron border (so it never bleeds into a neighbouring atlas slot).
  vec4 sampleBrick(uint s, vec3 posVox) {
    uint axb = uint(uAtlasBricks.x), ayb = uint(uAtlasBricks.y);
    vec3 sb = vec3(float(s % axb), float((s / axb) % ayb), float(s / (axb * ayb)));
    vec3 local = posVox - floor(posVox / uBrick) * uBrick;   // in [0, brick)
    vec3 atlasVox = sb * uBrickStride + vec3(uApron) + local;
    return texture(uBricks, (atlasVox + 0.5) / (uAtlasBricks * uBrickStride));
  }

  void main() {
    vec3 dir = normalize(vDirection);
    vec2 bounds = hitBox(vOrigin, dir);
    if (bounds.x > bounds.y) discard;
    // Depth-correct compositing: stop the ray at the nearest opaque surface (the
    // orientation-cube edges). Reconstruct the opaque-pass hit position from its
    // depth and clamp the ray's far bound so volume haze never composites in
    // front of a cube edge that's physically closer.
    float sd = texture(uSceneDepth, gl_FragCoord.xy / uResolution).x;
    if (sd < 1.0) {
      vec3 ndc = vec3(gl_FragCoord.xy / uResolution * 2.0 - 1.0, sd * 2.0 - 1.0);
      vec4 wp = uInvProjView * vec4(ndc, 1.0); wp /= wp.w;
      vec3 mh = (uInvModel * wp).xyz;             // scene hit in box-local space
      bounds.y = min(bounds.y, dot(mh - vOrigin, dir)); // dir is normalized → t
      if (bounds.x > bounds.y) discard;
    }
    float t = max(bounds.x, 0.0);
    float stepLen = (bounds.y - t) / float(uSteps);
    float denom = max(1e-4, 1.0 - uThreshold);
    vec4 acc = vec4(0.0);
    // March in ray parameter t through the page table: occupied bricks step at
    // stepLen and sample the atlas; empty bricks are leapt to their far
    // boundary, so the step budget concentrates where there's actually data.
    // In streamed mode, occupied-but-not-resident bricks sample the coarse
    // fallback and step (they must NOT leap, or the region would be invisible
    // until its brick streams in).
    for (int i = 0; i < 2048; i++) {
      if (t >= bounds.y) break;
      vec3 uvw = (vOrigin + t * dir) / uSize + 0.5;
      vec3 posVox = uvw * uVolDim;
      vec3 cell = floor(posVox / uBrick);
      // Streamed: descend the octree (empty leaps span the whole empty node, not
      // just one brick). Non-streamed: one flat page-table lookup.
      uint state; uint payload;
      vec3 boxOrigin; float boxExtent; vec3 leafTexel;
      if (uStreamed == 1) {
        payload = descend(cell, state, boxOrigin, boxExtent, leafTexel);
      } else {
        payload = pageCell(cell, state);
        boxOrigin = cell; boxExtent = 1.0;
      }
      bool empty = (uStreamed == 1) ? (state == 0u) : (payload == 0u);
      if (empty) {
        // Empty region → jump to the exit boundary of its [boxOrigin, +boxExtent)
        // brick box (the octree's coarse leap; one brick in the flat path). Always
        // advances ≥ one step.
        vec3 nb = (boxOrigin + step(0.0, dir) * boxExtent) * uBrick / uVolDim;
        vec3 tb = ((nb - 0.5) * uSize - vOrigin) / dir;        // → ray t at each axis plane
        float tnext = min(tb.x, min(tb.y, tb.z));
        t = max(tnext, t + stepLen) + 1e-4;
        continue;
      }
      vec4 vox;
      if (uStreamed == 1) {
        vox = (state == 2u) ? sampleBrick(payload, posVox)  // resident → cache
                            : texture(uCoarse, uvw);         // not resident → coarse LOD
      } else {
        vox = sampleBrick(payload - 1u, posVox);             // 1-based slot → 0-based
      }
      if (vox.a > 0.0) {
        // "rgb" (structured) mode: RGB is final baked color, A the opacity
        // weight. "lut" (byte) mode: R indexes the LUT, G/B are the opacity
        // sources (activity / density) chosen by uSource.
        vec3 col;
        float d;
        if (uDirectColor == 1) {
          col = vox.rgb;
          d = vox.a;
        } else {
          col = texture(uLut, vec2(vox.r, 0.5)).rgb;
          d = (uSource == 0) ? vox.g : vox.b;
          // A resident streamed brick (state==2) stores the RAW per-voxel count
          // in B; normalize by the pool max here (deferred from build time). The
          // coarse fallback and all non-streamed atlases keep B pre-baked, so
          // uAtlasInvMax is 0 and this is skipped. vox.b is raw/255, so
          // ·(255/max) recovers the old baked min(count,255)/max.
          if (uSource == 1 && uStreamed == 1 && state == 2u && uAtlasInvMax > 0.0) {
            d *= uAtlasInvMax;
          }
        }
        d = max(0.0, d - uThreshold) / denom;
        // uNorm compensates for the data the threshold removed: as the field
        // thins, fewer voxels accumulate along each ray, so we scale per-voxel
        // alpha up to hold the cube's integrated opacity (its viewability)
        // roughly constant. Computed on the client from the channel histogram.
        float a = clamp(pow(d, uGamma) * uOpacity * uNorm, 0.0, 1.0);
        acc.rgb += (1.0 - acc.a) * col * a;
        acc.a   += (1.0 - acc.a) * a;
        // Early-ray termination: once the ray is near-opaque, voxels behind it
        // are occluded and never sampled (the raymarch analog of occlusion
        // culling). The denser default opacity above makes this fire much
        // sooner; 0.9 (vs 1.0) drops only the last <10% of an already-saturated
        // ray for a free step-budget win.
        if (acc.a >= 0.9) break;
      }
      t += stepLen;
    }
    if (acc.a <= 0.0) discard;
    fragColor = acc;
  }`;

// Ray-guided brick-request (feedback) shader — the GigaVoxels probe. Descends the
// same octree as volFrag (sharing volVert) but, instead of compositing color,
// reports the FIRST occupied-but-not-resident leaf along each ray: the leaf's
// node-pool texel index, linearized (+1, so 0 means "no miss") and packed
// little-endian across RGBA8. Rendered to a small off-screen target and read back
// so the CPU can fetch exactly the bricks the visible surface needs and locate
// the very texel to flip resident. Streamed path only.
const fbFrag = `
  precision highp float;
  precision highp sampler3D;
  uniform sampler3D uNodePool;
  uniform float uBrick, uTreeDepth;
  uniform int uSteps;
  uniform vec3 uSize, uVolDim, uNodePoolDim;
  in vec3 vOrigin;
  in vec3 vDirection;
  out vec4 fragColor;

  vec2 hitBox(vec3 o, vec3 d) {
    vec3 h = uSize * 0.5;
    vec3 inv = 1.0 / d;
    vec3 a = (-h - o) * inv;
    vec3 b = ( h - o) * inv;
    vec3 tmin = min(a, b), tmax = max(a, b);
    return vec2(max(tmin.x, max(tmin.y, tmin.z)), min(tmax.x, min(tmax.y, tmax.z)));
  }

  uint readChild(float node, vec3 o, out uint a, out vec3 texel) {
    vec3 npa = uNodePoolDim * 0.5;
    vec3 nb = vec3(mod(node, npa.x), mod(floor(node / npa.x), npa.y), floor(node / (npa.x * npa.y)));
    texel = nb * 2.0 + o;
    vec4 c = texture(uNodePool, (texel + 0.5) / uNodePoolDim);
    a = uint(c.a * 255.0 + 0.5);
    return uint(c.r * 255.0 + 0.5) + (uint(c.g * 255.0 + 0.5) << 8u) + (uint(c.b * 255.0 + 0.5) << 16u);
  }

  // Same walk as volFrag's descend, but returns only what the probe needs:
  // state, the terminal node box (for leaping) and the leaf's pool texel.
  void descend(vec3 cell, out uint state, out vec3 boxOrigin, out float boxExtent, out vec3 texel) {
    float node = 0.0;
    vec3 origin = vec3(0.0);
    float extent = exp2(uTreeDepth);
    for (int d = 0; d < 24; d++) {
      float halfE = extent * 0.5;
      vec3 o = step(origin + halfE, cell + 0.5);
      vec3 childOrigin = origin + o * halfE;
      uint a; uint rgb = readChild(node, o, a, texel);
      if (a > 0u) { state = a; boxOrigin = childOrigin; boxExtent = 1.0; return; }
      if (rgb == 0u) { state = 0u; boxOrigin = childOrigin; boxExtent = halfE; return; }
      node = float(rgb - 1u); origin = childOrigin; extent = halfE;
    }
    state = 0u; boxOrigin = cell; boxExtent = 1.0;
  }

  void main() {
    vec3 dir = normalize(vDirection);
    vec2 bounds = hitBox(vOrigin, dir);
    if (bounds.x > bounds.y) { fragColor = vec4(0.0); return; }
    float t = max(bounds.x, 0.0);
    float stepLen = (bounds.y - t) / float(uSteps);
    for (int i = 0; i < 1024; i++) {
      if (t >= bounds.y) break;
      vec3 uvw = (vOrigin + t * dir) / uSize + 0.5;
      vec3 cell = floor(uvw * uVolDim / uBrick);
      uint state; vec3 boxOrigin; float boxExtent; vec3 texel;
      descend(cell, state, boxOrigin, boxExtent, texel);
      if (state == 0u) {
        vec3 nb = (boxOrigin + step(0.0, dir) * boxExtent) * uBrick / uVolDim;
        vec3 tb = ((nb - 0.5) * uSize - vOrigin) / dir;
        float tnext = min(tb.x, min(tb.y, tb.z));
        t = max(tnext, t + stepLen) + 1e-4;
        continue;
      }
      if (state == 1u) {
        // Report the leaf's node-pool texel (linearized, +1) so the CPU can read
        // its brick id and flip exactly this texel resident.
        uint tl = uint(texel.x) + uint(texel.y) * uint(uNodePoolDim.x)
                + uint(texel.z) * uint(uNodePoolDim.x) * uint(uNodePoolDim.y);
        uint v = tl + 1u;
        fragColor = vec4(float(v & 255u) / 255.0, float((v >> 8u) & 255u) / 255.0,
                         float((v >> 16u) & 255u) / 255.0, float((v >> 24u) & 255u) / 255.0);
        return;
      }
      t += stepLen; // resident → keep walking to find the first miss behind it
    }
    fragColor = vec4(0.0);
  }`;

// ---- threshold re-normalization ---------------------------------------------
// As the threshold rises it removes signal, so fewer voxels accumulate along
// each ray and the cube dims. We counter that by scaling per-voxel alpha
// (uNorm) so the *total* surviving signal stays close to its threshold-0 value
// — the sparse high-threshold field then reads about as bright as the full one.
let histAct = null, histDen = null, histAlpha = null, s0Act = 0, s0Den = 0, s0Alpha = 0;
let directColor = false; // "rgb" color_mode: RGB baked, A is the lone opacity source
const NORM_MAX = 12; // cap the boost so a near-empty field doesn't white out

// Sum, over occupied voxels, of the post-threshold opacity signal the shader
// integrates per voxel: max(0, v - t)/(1 - t) with v = byte/255. At t=0 this is
// the full signal; it falls as the threshold thins the survivors out.
function surviveSignal(hist, t) {
  const denom = Math.max(1e-4, 1.0 - t);
  let s = 0;
  for (let k = 0; k < 256; k++) {
    const v = k / 255;
    if (v > t) s += (v - t) * hist[k];
  }
  return s / denom;
}

// Build per-channel histograms over occupied voxels once at load.
function buildHistograms(buf) {
  histAct = new Float64Array(256);
  histDen = new Float64Array(256);
  histAlpha = new Float64Array(256); // A = opacity weight (rgb mode)
  for (let i = 0; i < buf.length; i += 4) {
    if (buf[i + 3] === 0) continue; // empty voxel — never rendered
    histAct[buf[i + 1]]++; // G = activity
    histDen[buf[i + 2]]++; // B = density
    histAlpha[buf[i + 3]]++;
  }
  s0Act = surviveSignal(histAct, 0);
  s0Den = surviveSignal(histDen, 0);
  s0Alpha = surviveSignal(histAlpha, 0);
}

// Recompute uNorm from the current threshold + opacity source. In rgb mode the
// sole source is the alpha channel; in lut mode it's the chosen G/B channel.
function refreshNorm() {
  if (!volMesh || !histAct) return;
  const u = volMesh.material.uniforms;
  let hist, s0;
  if (directColor) {
    hist = histAlpha; s0 = s0Alpha;
  } else {
    const density = u.uSource.value === 1;
    hist = density ? histDen : histAct;
    s0 = density ? s0Den : s0Act;
  }
  const st = surviveSignal(hist, u.uThreshold.value);
  const n = st > 1e-9 ? s0 / st : 1.0;
  u.uNorm.value = Math.min(Math.max(n, 1.0), NORM_MAX);
}

// ---- load + build -----------------------------------------------------------
// Kept for click-to-pick: the raw volume buffer + grid extent + world box size
// + entity manifest. extG is the (possibly coarse) volume.bin extent the CPU
// ray-march samples; volDimG is the full detail resolution the manifest bboxes
// live in (== extG unless the structured path is streamed).
let volBufG = null, extG = [0, 0, 0], sizeG = [1, 1, 1], manifestG = [], volDimG = [1, 1, 1];
const raycaster = new THREE.Raycaster();

// Ray-guided brick streaming (meta.bricks.streamed): bounded GPU cache fed on
// demand by what the rays need. Null for the fully-resident (non-streamed) path.
// `volumeDirty` (declared near the scene setup) re-runs the feedback probe after
// the camera moves or a brick lands, and falls quiet once the set is resident.
let brickStream = null;

async function load() {
  startLoadClock();
  const meta = await (await fetch('meta.json')).json();
  // Grid box in voxels. `grid_extent` is [x,y,z]; older bundles carried a
  // single cube side as `grid_side`.
  const ext = meta.grid_extent || [meta.grid_side, meta.grid_side, meta.grid_side];
  const [ex, ey, ez] = ext;
  // World size per axis: scale the longest axis to the unit cube so voxels
  // stay cubic and the box keeps the data's proportions. A cube → [1,1,1].
  const mx = Math.max(ex, ey, ez);
  const size = [ex / mx, ey / mx, ez / mx];
  directColor = meta.color_mode === 'rgb';

  // Progress denominator: exact byte size (RGBA8) of the up-front blocking
  // fetches for this bundle's mode. Matches the fetchBytes calls below; brick
  // streaming is excluded.
  {
    const b = meta.bricks;
    loadTotal = ex * ey * ez * 4;                                   // volume.bin
    if (b.streamed) {
      loadTotal += b.tree_dim[0] * b.tree_dim[1] * b.tree_dim[2] * 4; // tree.bin (octree node pool)
    } else {
      loadTotal += b.page_dim[0] * b.page_dim[1] * b.page_dim[2] * 4; // pagetable.bin
      loadTotal += b.atlas_dim[0] * b.atlas_dim[1] * b.atlas_dim[2] * 4; // bricks.bin
    }
  }

  // The volume renders from a sparse brick pool indexed by a page table
  // (meta.bricks): only occupied bricks are stored, so the GPU binding
  // constraint is the atlas, not the full cube. The dense volume.bin is fetched
  // only for CPU-side histograms + pick (never uploaded to the GPU).
  const gl = renderer.getContext();
  const maxSide = gl.getParameter(gl.MAX_3D_TEXTURE_SIZE);

  // byte->color LUT as a 256x1 texture
  const lut = new Uint8Array(256 * 4);
  for (let i = 0; i < 256; i++) {
    const c = meta.lut[i] || [0, 0, 0];
    lut[i * 4] = c[0]; lut[i * 4 + 1] = c[1]; lut[i * 4 + 2] = c[2]; lut[i * 4 + 3] = 255;
  }
  const lutTex = new THREE.DataTexture(lut, 256, 1, THREE.RGBAFormat);
  lutTex.minFilter = lutTex.magFilter = THREE.NearestFilter;
  lutTex.needsUpdate = true;

  // Dense volume buffer: CPU only (threshold histograms + click/hover pick).
  const volBuf = await fetchBytes('volume.bin');
  buildHistograms(volBuf);

  // Page table + brick pool. Two modes (meta.bricks.streamed):
  //  • non-streamed (default / structured): the whole atlas + page table are
  //    fetched and uploaded once; the page table holds 1-based resident slots.
  //  • streamed (--volume-res): ray-guided GigaVoxels streaming — only a bounded
  //    GPU cache is resident, bricks fetch on demand into free cache slots, the
  //    page table is mutated per-cell as bricks arrive/evict, and not-resident
  //    regions sample the coarse dense grid. See setupBrickStream / streamBricks.
  const bm = meta.bricks;
  const streamed = !!bm.streamed;
  const bstride = bm.brick + 2 * bm.apron; // stored brick edge (incl. apron)
  const make3d = (buf, w, h, d, linear) => {
    const t = new THREE.Data3DTexture(buf, w, h, d);
    t.format = THREE.RGBAFormat; t.type = THREE.UnsignedByteType;
    const f = linear ? THREE.LinearFilter : THREE.NearestFilter;
    t.minFilter = t.magFilter = f;
    t.wrapS = t.wrapT = t.wrapR = THREE.ClampToEdgeWrapping;
    t.unpackAlignment = 1; t.needsUpdate = true;
    return t;
  };

  let pageTex, brickTex, coarseTex, atlasBricks, nodePoolTex;
  if (streamed) {
    // Coarse fallback LOD: the dense grid (already fetched for histograms/pick),
    // sampled linearly where a brick isn't yet resident → an instant blurry
    // image that sharpens in place. Clamp to ≤1024³ (the coarse cap) so VRAM
    // stays bounded; a bundle exceeding that shouldn't exist, but guard it.
    if (Math.max(ex, ey, ez) > 1024) {
      throw new Error('streamed coarse fallback expects coarse grid ≤ 1024, got ' + Math.max(ex, ey, ez));
    }
    coarseTex = make3d(volBuf, ex, ey, ez, true);

    // Fixed-size GPU cache atlas — bounded VRAM, independent of total data size.
    const cdim = BRICK_CACHE_DIM * bm.brick; // cache atlas edge in voxels (apron 0)
    if (cdim > maxSide) {
      throw new Error('brick cache ' + cdim + ' exceeds this GPU\'s max 3D texture (' + maxSide + ')');
    }
    brickTex = make3d(new Uint8Array(cdim * cdim * cdim * 4), cdim, cdim, cdim, false);
    atlasBricks = [BRICK_CACHE_DIM, BRICK_CACHE_DIM, BRICK_CACHE_DIM];

    // Sparse octree node pool (tree.bin). O(occupied), so this stays a few MB
    // however high the detail resolution — the whole point of the octree. The
    // fetched bytes are BOTH the uploaded texture and the CPU mirror the streamer
    // mutates in place: leaf entries ship with A=1 (occupied, not resident) + RGB
    // = 1-based brick id; the viewer flips A=2 + RGB = cache slot as bricks land.
    const nodePool = await fetchBytes(bm.tree_file);
    const [tw, th, tdp] = bm.tree_dim;
    if (Math.max(tw, th, tdp) > maxSide) {
      throw new Error('octree node pool ' + Math.max(tw, th, tdp) + ' exceeds this GPU\'s max 3D texture (' + maxSide + ')');
    }
    nodePoolTex = make3d(nodePool, tw, th, tdp, false);
    // Force GPU allocation now so per-texel/per-brick texSubImage3D works.
    renderer.initTexture(nodePoolTex);
    renderer.initTexture(brickTex);
    // uPageTable is unused in the octree path; bind a 1-voxel dummy.
    pageTex = make3d(new Uint8Array(4), 1, 1, 1, false);
    brickStream = setupBrickStream(bm, nodePoolTex, brickTex, nodePool, size);
  } else {
    const atlasMax = Math.max(bm.atlas_dim[0], bm.atlas_dim[1], bm.atlas_dim[2]);
    if (atlasMax > maxSide) {
      throw new Error(
        'brick atlas size ' + atlasMax + ' exceeds the max 3D texture size this ' +
        'GPU supports (' + maxSide + '). Re-run arbvis with a lower --grid.');
    }
    const pageBuf = await fetchBytes(bm.page_file);
    pageTex = make3d(pageBuf, bm.page_dim[0], bm.page_dim[1], bm.page_dim[2], false);
    const brickBuf = await fetchBytes(bm.atlas_file);
    // The apron border lets trilinear filtering cross brick edges smoothly.
    brickTex = make3d(brickBuf, bm.atlas_dim[0], bm.atlas_dim[1], bm.atlas_dim[2], bm.apron > 0);
    atlasBricks = [bm.atlas_dim[0] / bstride, bm.atlas_dim[1] / bstride, bm.atlas_dim[2] / bstride];
    // uCoarse and uNodePool are never sampled in this mode, but the samplers
    // still need a bound 3D texture — 1-voxel dummies.
    coarseTex = make3d(new Uint8Array(4), 1, 1, 1, false);
    nodePoolTex = make3d(new Uint8Array(4), 1, 1, 1, false);
  }

  const volMat = new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3,
    // Composited over the already-drawn opaque (edges) background. volFrag emits
    // premultiplied color (acc.rgb is pre-multiplied by acc.a), so blend with
    // (ONE, ONE_MINUS_SRC_ALPHA). Depth is handled manually via uSceneDepth, so
    // the hardware depth test is off.
    side: THREE.BackSide, transparent: true, depthTest: false, depthWrite: false,
    blending: THREE.CustomBlending, blendSrc: THREE.OneFactor, blendDst: THREE.OneMinusSrcAlphaFactor,
    uniforms: {
      uBricks: { value: brickTex }, uPageTable: { value: pageTex }, uLut: { value: lutTex },
      uNodePool: { value: nodePoolTex },
      uNodePoolDim: { value: new THREE.Vector3(bm.tree_dim[0] || 1, bm.tree_dim[1] || 1, bm.tree_dim[2] || 1) },
      uTreeDepth: { value: bm.tree_depth || 0 },
      uCoarse: { value: coarseTex }, uStreamed: { value: streamed ? 1 : 0 },
      uOpacity: { value: 0.45 }, uGamma: { value: 1.0 },
      // Quality is fixed high (no slider) — well above the old 384 max. The 2048
      // loop cap in volFrag keeps up with the finer step in dense regions.
      uThreshold: { value: 0.0 }, uSource: { value: 0 }, uSteps: { value: 512 },
      uNorm: { value: 1.0 }, uDirectColor: { value: directColor ? 1 : 0 },
      uSize: { value: new THREE.Vector3(size[0], size[1], size[2]) },
      uVolDim: { value: new THREE.Vector3(bm.vol_dim[0], bm.vol_dim[1], bm.vol_dim[2]) },
      uPageDim: { value: new THREE.Vector3(bm.page_dim[0], bm.page_dim[1], bm.page_dim[2]) },
      uAtlasBricks: { value: new THREE.Vector3(atlasBricks[0], atlasBricks[1], atlasBricks[2]) },
      uBrick: { value: bm.brick }, uBrickStride: { value: bstride }, uApron: { value: bm.apron },
      // Raw-count streamed atlas ⇒ normalize density in-shader by 255/max_count;
      // absent/0 (structured, non-streamed, pre-v6 bundles) ⇒ B is baked, no rescale.
      uAtlasInvMax: { value: (bm.max_count > 0) ? 255.0 / bm.max_count : 0.0 },
      // sceneTarget may not exist yet on a cold start (the ResizeObserver hasn't
      // fired — likely now that the small streamed up-front fetch lets load()
      // reach here before layout settles). renderHybrid re-points this every
      // volume-pass frame and bails while sceneTarget is null, so a null start
      // is safe; resize() also sets it once the target is built.
      uSceneDepth: { value: sceneTarget ? sceneTarget.depthTexture : null }, uResolution: { value: new THREE.Vector2(1, 1) },
      uInvProjView: { value: new THREE.Matrix4() }, uInvModel: { value: new THREE.Matrix4() },
    },
    vertexShader: volVert, fragmentShader: volFrag,
  });
  if (brickStream) brickStream.volMat = volMat;
  volMesh = new THREE.Mesh(new THREE.BoxGeometry(size[0], size[1], size[2]), volMat);
  volMesh.layers.set(VOL_LAYER); // volume-only pass selects this layer
  // Point the depth sampler at sceneTarget now that the material is built — but
  // only if it exists yet (see the cold-start note on uSceneDepth above;
  // renderHybrid/resize set it otherwise).
  if (sceneTarget) volMesh.material.uniforms.uSceneDepth.value = sceneTarget.depthTexture;
  scene.add(volMesh);
  // Match the orientation cube to the (possibly anisotropic) box.
  edges.scale.set(size[0], size[1], size[2]);

  // Frame the camera on the occupied region (data often fills only part of
  // the cube — a small file is a contiguous Hilbert prefix in one corner).
  const fc = meta.focus_center || [0, 0, 0];
  const fr = meta.focus_radius || 0.5;
  controls.target.set(fc[0], fc[1], fc[2]);
  const dist = Math.max(fr * 3.2, 0.18);
  camera.position.set(fc[0] + dist, fc[1] + dist * 0.85, fc[2] + dist * 1.1);
  controls.update();

  // rgb (structured) mode has a single opacity source (alpha), so the
  // Activity/Density toggle is meaningless — hide it.
  if (directColor) {
    const r = $('src').closest('.row');
    if (r) r.hidden = true;
  }

  // Click-to-pick + legend (structured cubes only — the byte path has an
  // empty manifest, so picking no-ops and the legend stays blank).
  volBufG = volBuf;
  extG = ext;
  sizeG = size;
  // Full detail resolution the manifest bboxes are expressed in. For a streamed
  // structured bundle volume.bin (extG) is a smaller coarse grid, so pick/labels
  // map through this; for non-streamed bundles it equals extG (identity).
  volDimG = bm.vol_dim || ext;
  manifestG = meta.manifest || [];
  buildLegend(meta);
  buildLayerLabels();

  stopLoadClock();
  $('status').hidden = true;
  $('panel').hidden = false;
  bindControls();
}

// ---- ray-guided brick streaming ---------------------------------------------
// For meta.bricks.streamed
// (--volume-res) bundles. A low-res feedback pass (fbFrag) reports the first
// occupied-but-not-resident brick along each ray; the CPU range-fetches those
// bricks into a bounded GPU cache atlas, mutating the page table per-cell as
// they arrive. A cheap CPU "touch walk" over the page-state mirror drives LRU
// eviction + light prefetch, so VRAM tracks the visible working set, not the
// data's total size. No compute shaders / SSBOs — feedback is a render-to-
// texture probe read back to the CPU.
const BRICK_CACHE_DIM = 32;                                  // cache atlas side in bricks → 32768 slots
const BRICK_CACHE_CAP = BRICK_CACHE_DIM ** 3;
const BRICK_CACHE_SOFT = Math.floor(BRICK_CACHE_CAP * 0.92); // evict above this (hysteresis vs CAP)
const BRICK_MAX_INFLIGHT = 8;                                // concurrent range fetches (each covers one aligned block)
// Bricks are fetched in FIXED, aligned blocks of this many ids: any pending
// brick in a block pulls the whole block in ONE Range request. Two payoffs vs
// per-brick fetches, both aimed at the HF Spaces per-window request limit that
// 429s the load: (1) request count is bounded by blocks-touched, not bricks
// (scattered misses across a block collapse to one fetch); (2) the Range is a
// function of the block alone, so it's byte-identical across probes, pans, and
// reloads — the immutable browser cache then serves repeats with no network at
// all (per-brick coalescing produced session-dependent ranges that never cache-
// hit, which is why reloads still stormed). 1024·(8³·4) = one 2 MiB request.
const BRICK_BLOCK = 1024;
const BACKOFF_MIN_MS = 800;                                  // first pause after a 429/5xx before retrying fetches
const BACKOFF_MAX_MS = 20000;                                // ceiling for the decorrelated-jitter backoff
const FB_DIV = 8;                                            // feedback target = drawing buffer / FB_DIV
const FB_EVERY = 2;                                          // probe at most every N frames
const brickDebug = location.search.includes('debug');
const nowMs = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());
const _tNdc = new THREE.Vector2();

// Build the brick-stream state. `nodeTex`/`brickTex` are the mutable octree node
// pool and the (initially empty) cache atlas; `nodePool` is the CPU mirror of the
// node pool (mutated in lock-step via texSub3D). Residency is keyed by a leaf's
// linear node-pool texel index `tl`: its RGB holds the 1-based file brick id
// while not resident (range offset = (id-1)·stride), the cache slot once resident.
function setupBrickStream(bm, nodeTex, brickTex, nodePool, size) {
  const freeSlots = [];
  for (let s = BRICK_CACHE_CAP - 1; s >= 0; s--) freeSlots.push(s); // pop() → 0,1,2,…
  // Feedback material: reuses volVert, shares the *mutable* node-pool texture (so
  // it never re-requests a brick that already arrived). Coarse step count — it
  // only needs to find the first miss, not integrate color.
  const fbMat = new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3, side: THREE.BackSide,
    uniforms: {
      uNodePool: { value: nodeTex },
      uNodePoolDim: { value: new THREE.Vector3(bm.tree_dim[0], bm.tree_dim[1], bm.tree_dim[2]) },
      uTreeDepth: { value: bm.tree_depth },
      uBrick: { value: bm.brick }, uSteps: { value: 128 },
      uSize: { value: new THREE.Vector3(size[0], size[1], size[2]) },
      uVolDim: { value: new THREE.Vector3(bm.vol_dim[0], bm.vol_dim[1], bm.vol_dim[2]) },
    },
    vertexShader: volVert, fragmentShader: fbFrag,
  });
  const fbScene = new THREE.Scene();
  fbScene.add(new THREE.Mesh(new THREE.BoxGeometry(size[0], size[1], size[2]), fbMat));
  const fbTarget = new THREE.WebGLRenderTarget(1, 1, {
    format: THREE.RGBAFormat, type: THREE.UnsignedByteType,
    minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter, depthBuffer: false,
  });
  return {
    bm, brick: bm.brick, stride: bm.brick ** 3 * 4,
    nodeTex, brickTex, nodePool,
    treeDim: bm.tree_dim, treeDepth: bm.tree_depth,
    vol: bm.vol_dim,
    fbScene, fbTarget, fbBuf: null,
    resident: new Map(),        // leaf texel tl → { slot, lastUsed, id }
    freeSlots, inflight: 0, pending: new Set(),
    // Activity-HUD accounting. inflightBricks tracks bricks (not block requests) in
    // flight so "outstanding" stays in brick units; sessionResolved counts bricks that
    // landed since the current sharpening session began (reset on each idle→active edge).
    inflightBricks: 0, sessionResolved: 0, sessionStartMs: 0,
    frame: 0, fbFrame: -100, probeIdle: true, glDirty: false,
    backoffUntil: 0, backoffMs: 0, // pause fetches after a 429/5xx (decorrelated jitter)
    stats: { requested: 0, evicted: 0, bytes: 0, loaded: 0 },
    size, volMat: null,
  };
}

// Descend the octree CPU-side (mirrors the shader) to the leaf texel for brick
// cell (cx,cy,cz), or -1 for an empty subtree. Used by the LRU touch walk.
function descendCell(bs, cx, cy, cz) {
  const np = bs.nodePool, tw = bs.treeDim[0], twh = bs.treeDim[0] * bs.treeDim[1];
  const npx = bs.treeDim[0] >> 1, npy = bs.treeDim[1] >> 1;
  let node = 0, ox = 0, oy = 0, oz = 0;
  let extent = 1 << bs.treeDepth, orx = 0, ory = 0, orz = 0;
  for (let d = 0; d < bs.treeDepth; d++) {
    const half = extent >> 1;
    ox = (cx - orx) >= half ? 1 : 0; oy = (cy - ory) >= half ? 1 : 0; oz = (cz - orz) >= half ? 1 : 0;
    const bx = node % npx, by = ((node / npx) | 0) % npy, bz = (node / (npx * npy)) | 0;
    const tl = (bx * 2 + ox) + (by * 2 + oy) * tw + (bz * 2 + oz) * twh;
    const o = tl * 4, a = np[o + 3], rgb = np[o] | (np[o + 1] << 8) | (np[o + 2] << 16);
    if (a > 0) return tl;      // leaf brick
    if (rgb === 0) return -1;  // empty subtree
    node = rgb - 1; orx += ox * half; ory += oy * half; orz += oz * half; extent = half;
  }
  return -1;
}

// Raw-GL subregion upload into a three-owned Data3DTexture (page cell or brick).
// three.js has no first-class partial 3D update, so go direct — but it caches GL
// binding state, so flag it stale (reset once per frame before the next draw).
function texSub3D(bs, tex, x, y, z, w, h, d, data) {
  const gl = renderer.getContext();
  const glTex = renderer.properties.get(tex).__webglTexture;
  if (!glTex) return;
  gl.bindTexture(gl.TEXTURE_3D, glTex);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texSubImage3D(gl.TEXTURE_3D, 0, x, y, z, w, h, d, gl.RGBA, gl.UNSIGNED_BYTE, data);
  gl.bindTexture(gl.TEXTURE_3D, null);
  bs.glDirty = true;
}

const _cellTexel = new Uint8Array(4);
// Leaf-entry state read off the CPU node-pool mirror at texel `tl` (A byte):
// 0 empty, 1 occupied-not-resident, 2 resident.
function leafState(bs, tl) { return bs.nodePool[tl * 4 + 3]; }
// The leaf's RGB payload (brick id while not resident, cache slot when resident).
function leafRgb(bs, tl) {
  const o = tl * 4; return bs.nodePool[o] | (bs.nodePool[o + 1] << 8) | (bs.nodePool[o + 2] << 16);
}

// Write one octree leaf entry (CPU mirror + GPU texel). state: 1 not resident,
// 2 resident; rgb is the 1-based brick id (state 1) or 0-based cache slot (2).
function writeLeaf(bs, tl, state, rgb) {
  const tw = bs.treeDim[0], twh = bs.treeDim[0] * bs.treeDim[1];
  const tx = tl % tw, ty = ((tl / tw) | 0) % bs.treeDim[1], tz = (tl / twh) | 0;
  _cellTexel[0] = rgb & 0xff; _cellTexel[1] = (rgb >> 8) & 0xff;
  _cellTexel[2] = (rgb >> 16) & 0xff; _cellTexel[3] = state;
  const o = tl * 4;
  bs.nodePool[o] = _cellTexel[0]; bs.nodePool[o + 1] = _cellTexel[1];
  bs.nodePool[o + 2] = _cellTexel[2]; bs.nodePool[o + 3] = _cellTexel[3];
  texSub3D(bs, bs.nodeTex, tx, ty, tz, 1, 1, 1, _cellTexel);
}

// Evict a resident brick: free its cache slot, flip its leaf back to
// not-resident and restore its brick id (RGB). Atlas texels are never cleared —
// the leaf state gates reads.
function evictBrick(bs, tl) {
  const e = bs.resident.get(tl);
  if (!e) return;
  bs.freeSlots.push(e.slot);
  bs.resident.delete(tl);
  writeLeaf(bs, tl, 1, e.id);
  bs.stats.evicted++;
}

// Fallback eviction when a brick arrives with no free slot: drop the global LRU
// brick, unless even the coldest was touched this frame (cache full of visible
// bricks → defer rather than evict something on screen).
function evictLru(bs) {
  let lru = null, used = Infinity;
  for (const [tl, e] of bs.resident) {
    if (e.lastUsed < used) { used = e.lastUsed; lru = tl; }
  }
  if (lru === null || used === bs.frame) return false;
  evictBrick(bs, lru);
  return true;
}

// Place an arrived brick (brick³ RGBA8, x-fastest) into a cache slot and mark its
// leaf resident. `id` is the 1-based brick id (kept so eviction can restore RGB).
// apron 0 → the stored brick edge is exactly bm.brick.
function uploadBrick(bs, tl, id, data) {
  if (leafState(bs, tl) !== 1) return; // evicted/raced before the fetch returned
  let slot = bs.freeSlots.pop();
  if (slot === undefined) { if (!evictLru(bs)) return; slot = bs.freeSlots.pop(); }
  const D = BRICK_CACHE_DIM, b = bs.brick;
  const sx = slot % D, sy = ((slot / D) | 0) % D, sz = (slot / (D * D)) | 0;
  texSub3D(bs, bs.brickTex, sx * b, sy * b, sz * b, b, b, b, data);
  bs.resident.set(tl, { slot, lastUsed: bs.frame, id });
  writeLeaf(bs, tl, 2, slot);
  bs.sessionResolved++; // for the activity-HUD progress fraction this sharpening session
  bs.stats.loaded++;    // cumulative "total loaded" counter (survives session resets/eviction)
  volumeDirty = true; // a brick landed — re-probe (more may be needed behind it)
}

// Re-queue a leaf for a later fetch (only if it's still occupied-not-resident).
function requeueBrick(bs, tl) { if (leafState(bs, tl) === 1) bs.pending.add(tl); }

// Enter a backoff window after a 429/5xx/network error: pause all fetches for a
// decorrelated-jitter interval (next ∈ [MIN, min(MAX, prev·3)]) so the streamer
// stops hammering a rate-limited host and converges instead of stalling. Mirrors
// the build-time throttle. `volumeDirty` keeps the tick loop revisiting so the
// pump resumes automatically once the window passes.
function brickBackoff(bs) {
  const prev = bs.backoffMs || BACKOFF_MIN_MS;
  const next = Math.min(BACKOFF_MAX_MS, BACKOFF_MIN_MS + Math.random() * (prev * 3 - BACKOFF_MIN_MS));
  bs.backoffMs = next;
  bs.backoffUntil = nowMs() + next;
  volumeDirty = true;
  if (brickDebug) console.warn('brick fetch backoff', Math.round(next), 'ms');
}
function brickFetchOk(bs) { bs.backoffMs = 0; bs.backoffUntil = 0; }

// Fetch one FIXED aligned block [blk·BRICK_BLOCK, (blk+1)·BRICK_BLOCK) of bricks
// in a single Range request and upload the needed ones. `group` is [[id, tl], …]
// (every pending brick that fell in this block). The request offset is derived
// from the block index alone — NOT from the group — so the Range is byte-
// identical every time the block is touched and the immutable browser cache can
// serve it with no network. 206 (partial) is the fast path; a host that ignores
// Range (200, whole file) is sliced client-side by absolute offset so it stays
// correct everywhere. On a throttle/5xx/error the whole group is re-queued and a
// backoff window opens (no partial/garbage upload). The block's end may run past
// EOF on the last block — the Space clamps it to the true size.
function loadBrickBlock(bs, blk, group) {
  const base = blk * BRICK_BLOCK * bs.stride;        // block-aligned file origin
  const end = base + BRICK_BLOCK * bs.stride - 1;    // inclusive last byte (clamped at EOF by the server)
  bs.inflight++;
  bs.inflightBricks += group.length;
  fetch(bs.bm.atlas_file, { headers: { Range: 'bytes=' + base + '-' + end } })
    .then((res) => {
      // Rate-limit / transient signals: the HF Spaces router in front of the app
      // 429s the browser→Space request storm; the app 500s on a Hub hiccup. Both
      // are retryable — requeue the whole group and back off so we stop hammering
      // (never drop the bricks, or the view would never sharpen).
      if (res.status === 429 || res.status >= 500) {
        for (const [, tl] of group) requeueBrick(bs, tl);
        brickBackoff(bs);
        return null;
      }
      if (!res.ok) { console.error('brick block failed', res.status); return null; }
      return res.arrayBuffer().then((ab) => {
        brickFetchOk(bs);
        bs.stats.bytes += ab.byteLength; // "total downloaded" (network + browser-cache hits)
        const u8 = new Uint8Array(ab);
        const whole = res.status !== 206; // 200 ⇒ whole file; index by absolute offset
        for (const [id, tl] of group) {
          const start = whole ? (id - 1) * bs.stride : (id - 1) * bs.stride - base;
          const data = u8.subarray(start, start + bs.stride);
          if (data.length === bs.stride) uploadBrick(bs, tl, id, data);
          else requeueBrick(bs, tl); // short/truncated slice — retry later
        }
      });
    })
    .catch((e) => {
      console.error('brick block load failed', e);
      for (const [, tl] of group) requeueBrick(bs, tl);
      brickBackoff(bs);
    })
    .finally(() => { bs.inflight--; bs.inflightBricks -= group.length; pumpBrickFetches(bs); });
}

// Drain queued misses into block fetches: bucket every pending brick by its
// fixed aligned block and issue at most one Range request per block. This is
// what keeps the browser under the HF Spaces per-window request limit — a whole
// scattered surface collapses to blocks-touched requests, not one-per-brick, and
// each block's Range is stable so already-fetched blocks re-serve from cache.
// Bounded by the inflight cap + free cache slots; paused during a backoff window.
// When the cache is full we wait for the touch walk to evict rather than evict
// blind here (it lacks the current visible cut).
function pumpBrickFetches(bs) {
  if (bs.pending.size === 0) return;
  if (bs.backoffUntil && nowMs() < bs.backoffUntil) return; // rate-limited: hold off
  // Bucket fetchable leaves (still occupied-not-resident) by aligned block.
  const blocks = new Map(); // block index → [[id, tl], …]
  for (const tl of bs.pending) {
    if (leafState(bs, tl) !== 1) { bs.pending.delete(tl); continue; }
    const id = leafRgb(bs, tl);
    if (!id) { bs.pending.delete(tl); continue; }
    const blk = Math.floor((id - 1) / BRICK_BLOCK);
    let g = blocks.get(blk);
    if (!g) { g = []; blocks.set(blk, g); }
    g.push([id, tl]);
  }
  if (blocks.size === 0) return;
  for (const [blk, group] of blocks) {
    if (bs.inflight >= BRICK_MAX_INFLIGHT || bs.freeSlots.length === 0) break;
    for (const [, tl] of group) bs.pending.delete(tl);
    loadBrickBlock(bs, blk, group);
  }
}

// Decode the feedback target: each pixel is (leaf texel + 1) of the first miss
// along its ray, little-endian RGBA8 (0 = no miss). Enqueue unique misses.
function processFeedback(bs, buf) {
  let added = 0;
  for (let i = 0; i < buf.length; i += 4) {
    const v = (buf[i] | (buf[i + 1] << 8) | (buf[i + 2] << 16) | (buf[i + 3] << 24)) >>> 0;
    if (v === 0) continue;
    const tl = v - 1;
    if (leafState(bs, tl) !== 1 || bs.pending.has(tl)) continue;
    bs.pending.add(tl);
    bs.stats.requested++;
    added++;
  }
  pumpBrickFetches(bs);
  // Converge: keep probing while there's outstanding work; otherwise fall quiet
  // until the camera moves (controls 'change' re-arms volumeDirty).
  if (added > 0 || bs.pending.size > 0 || bs.inflight > 0) volumeDirty = true;
}

// Render the feedback probe to a small target and read it back (async PBO when
// available; sync otherwise). One probe in flight at a time (probeIdle).
function runFeedback(bs) {
  bs.probeIdle = false;
  bs.fbFrame = bs.frame;
  volumeDirty = false; // consume; re-armed by misses / brick arrival / camera move
  const fw = Math.max(1, Math.ceil(renderer.domElement.width / FB_DIV));
  const fh = Math.max(1, Math.ceil(renderer.domElement.height / FB_DIV));
  if (bs.fbTarget.width !== fw || bs.fbTarget.height !== fh) bs.fbTarget.setSize(fw, fh);
  const prevTarget = renderer.getRenderTarget();
  renderer.setRenderTarget(bs.fbTarget);
  renderer.setClearColor(0x000000, 0); // outside-box / no-miss rays read back as 0
  renderer.render(bs.fbScene, camera);
  renderer.setRenderTarget(prevTarget);
  const n = fw * fh * 4;
  if (!bs.fbBuf || bs.fbBuf.length !== n) bs.fbBuf = new Uint8Array(n);
  const buf = bs.fbBuf;
  const finish = () => { bs.probeIdle = true; processFeedback(bs, buf); };
  if (renderer.readRenderTargetPixelsAsync) {
    renderer.readRenderTargetPixelsAsync(bs.fbTarget, 0, 0, fw, fh, buf)
      .then(finish)
      .catch((e) => { console.error('brick feedback readback failed', e); bs.probeIdle = true; });
  } else {
    renderer.readRenderTargetPixels(bs.fbTarget, 0, 0, fw, fh, buf);
    finish();
  }
}

// Coarse CPU march over the page-state mirror along a grid of view rays: stamp
// resident bricks seen as in-cut (lastUsed, for LRU), and lightly prefetch the
// nearest not-resident bricks just ahead. Then evict LRU non-cut bricks if the
// cache is over its soft cap. Cheap (≈ RAYS² · steps state lookups, no GPU).
function touchAndEvictBricks(bs) {
  const sz = bs.size, vd = bs.vol, brick = bs.brick;
  const cut = new Set();
  camera.updateMatrixWorld();
  const RAYS = 28, STEPS = 96;
  for (let iy = 0; iy < RAYS; iy++) {
    for (let ix = 0; ix < RAYS; ix++) {
      _tNdc.set((ix + 0.5) / RAYS * 2 - 1, (iy + 0.5) / RAYS * 2 - 1);
      raycaster.setFromCamera(_tNdc, camera);
      const ro = raycaster.ray.origin, rd = raycaster.ray.direction;
      let tmin = -Infinity, tmax = Infinity;
      for (const a of ['x', 'y', 'z']) {
        const half = sz[a === 'x' ? 0 : a === 'y' ? 1 : 2] / 2;
        const inv = 1 / rd[a];
        let t0 = (-half - ro[a]) * inv, t1 = (half - ro[a]) * inv;
        if (t0 > t1) { const tmp = t0; t0 = t1; t1 = tmp; }
        tmin = Math.max(tmin, t0); tmax = Math.min(tmax, t1);
      }
      let t = Math.max(tmin, 0);
      if (tmax < t) continue;
      const dt = (tmax - t) / STEPS;
      let pre = 0;
      for (let s = 0; s < STEPS; s++, t += dt) {
        const ux = (ro.x + rd.x * t) / sz[0] + 0.5;
        const uy = (ro.y + rd.y * t) / sz[1] + 0.5;
        const uz = (ro.z + rd.z * t) / sz[2] + 0.5;
        if (ux < 0 || uy < 0 || uz < 0 || ux >= 1 || uy >= 1 || uz >= 1) continue;
        const cx = (ux * vd[0] / brick) | 0, cy = (uy * vd[1] / brick) | 0, cz = (uz * vd[2] / brick) | 0;
        const tl = descendCell(bs, cx, cy, cz); // octree leaf texel, or -1 empty
        if (tl < 0) continue;
        const st = bs.nodePool[tl * 4 + 3];
        if (st === 2) {
          const e = bs.resident.get(tl);
          if (e) { e.lastUsed = bs.frame; cut.add(tl); }
        } else if (st === 1 && pre < 3 && !bs.pending.has(tl)) {
          bs.pending.add(tl); bs.stats.requested++; pre++;
        }
      }
    }
  }
  pumpBrickFetches(bs);
  if (bs.resident.size > BRICK_CACHE_SOFT) {
    const ev = [];
    for (const [tl, e] of bs.resident) if (!cut.has(tl)) ev.push([tl, e.lastUsed]);
    ev.sort((a, b) => a[1] - b[1]); // oldest first
    for (const [tl] of ev) {
      if (bs.resident.size <= BRICK_CACHE_SOFT) break;
      evictBrick(bs, tl);
    }
  }
}

// Per-frame brick streaming step (called from tick before the main render).
function streamBricks() {
  const bs = brickStream;
  if (!bs || !volMesh) return; // volume is always part of the unified view now
  bs.frame++;
  if ((volumeDirty || bs.pending.size > 0) && bs.probeIdle && (bs.frame - bs.fbFrame) >= FB_EVERY) {
    runFeedback(bs);
  }
  if (bs.pending.size > 0 || bs.resident.size > BRICK_CACHE_SOFT) touchAndEvictBricks(bs);
  // Raw-GL writes this frame left three's binding cache stale — clear it once
  // before the next draw (cheaper than resetting per texSubImage3D).
  if (bs.glDirty) { renderer.resetState(); bs.glDirty = false; }
  if (brickDebug && bs.frame % 30 === 0) {
    console.log('bricks: resident', bs.resident.size, '/', BRICK_CACHE_CAP,
      'free', bs.freeSlots.length, 'inflight', bs.inflight, 'pending', bs.pending.size,
      'requested', bs.stats.requested, 'evicted', bs.stats.evicted);
  }
}

// ---- background streaming activity HUD --------------------------------------
// A small bottom-left overlay that appears only while the view is still sharpening
// (demand-driven bricks streaming in) and hides once the current view is at full
// resolution. Driven by a throttled interval (below), reading only brickStream state.
function bytesFmt(n) {
  if (n < 1024) return n + ' B';
  if (n < 1048576) return (n / 1024).toFixed(0) + ' KB';
  if (n < 1073741824) return (n / 1048576).toFixed(1) + ' MB';
  return (n / 1073741824).toFixed(2) + ' GB';
}
// When the stream falls quiet we flash "full resolution" then fade out; these two
// timestamps gate that so the HUD doesn't flicker between back-to-back probes.
let bgFadeAt = 0, bgHideAt = 0;
function updateBgHud() {
  const bs = brickStream;
  const el = $('bg');
  if (!bs || !el) return; // non-streamed bundles never stream → HUD stays hidden
  const outstanding = bs.pending.size + bs.inflightBricks;
  const backoff = bs.backoffUntil && nowMs() < bs.backoffUntil;
  const active = outstanding > 0 || bs.inflight > 0 || backoff;

  // An idle→active edge starts a fresh sharpening "session" (progress denominator).
  if (active && bs.sessionStartMs === 0) { bs.sessionStartMs = nowMs(); bs.sessionResolved = 0; }

  if (active) {
    bgFadeAt = 0; bgHideAt = 0;
    el.hidden = false; el.classList.remove('fading');
    const denom = bs.sessionResolved + outstanding;
    let pct = Math.max(0, Math.min(100, Math.round((denom > 0 ? bs.sessionResolved / denom : 0) * 100)));
    if (outstanding > 0 && pct >= 100) pct = 99; // reserve 100% for true quiescence (below)
    $('bg-bar').style.width = pct + '%';
    $('bg-pct').textContent = pct + '%';
    if (backoff) {
      el.classList.add('warn');
      $('bg-title').textContent = 'waiting — rate-limited';
      $('bg-eta').textContent = 'retrying in ~' + fmtDur((bs.backoffUntil - nowMs()) / 1000);
    } else {
      el.classList.remove('warn');
      $('bg-title').textContent = 'sharpening…';
      const elapsed = (nowMs() - bs.sessionStartMs) / 1000;
      const rate = elapsed > 0 ? bs.sessionResolved / elapsed : 0; // bricks/sec
      $('bg-eta').textContent = (bs.sessionResolved > 2 && rate > 0)
        ? '~' + fmtDur(outstanding / rate) + ' left' : '—';
    }
    $('bg-counts').textContent =
      bs.inflight + ' active · ' + bs.pending.size + ' queued · ' +
      bs.stats.loaded.toLocaleString() + ' loaded · ' + bytesFmt(bs.stats.bytes);
  } else if (bs.sessionStartMs !== 0) {
    // Just reached quiescence: end the session, show a completed state, arm the fade.
    bs.sessionStartMs = 0;
    el.classList.remove('warn');
    $('bg-bar').style.width = '100%';
    $('bg-pct').textContent = '100%';
    $('bg-title').textContent = 'full resolution';
    $('bg-eta').textContent = '';
    bgFadeAt = nowMs() + 900; bgHideAt = bgFadeAt + 300; // fade window then remove
  } else if (bgFadeAt) {
    if (nowMs() >= bgFadeAt) el.classList.add('fading');
    if (nowMs() >= bgHideAt) { el.hidden = true; el.classList.remove('fading'); bgFadeAt = 0; bgHideAt = 0; }
  }
}
setInterval(updateBgHud, 150);

// HTML-escape model-derived tensor names before injecting into the panel.
const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

// March a ray through the grid (world box centered at the origin, each axis
// spanning [-size/2, size/2]) to the first opaque voxel — matching what the
// shader shows — then report that voxel: its grid coords, the alpha-encoded
// magnitude there, and the entity whose box contains it (null if unlabeled).
// Reuses the already-fetched volume buffer; no proxy meshes. Returns null when
// there's no manifest (byte mode), the ray misses the box, or the first opaque
// voxel it hits belongs to no entity.
function pickAt(clientX, clientY) {
  if (!manifestG.length || !volBufG) return null;
  const ndc = new THREE.Vector2((clientX / innerWidth) * 2 - 1, -(clientY / innerHeight) * 2 + 1);
  raycaster.setFromCamera(ndc, camera);
  const ro = raycaster.ray.origin, rd = raycaster.ray.direction;
  const half = [sizeG[0] / 2, sizeG[1] / 2, sizeG[2] / 2];
  // Ray vs box slab test.
  let tmin = -Infinity, tmax = Infinity;
  ['x', 'y', 'z'].forEach((a, ai) => {
    const inv = 1 / rd[a];
    let t0 = (-half[ai] - ro[a]) * inv, t1 = (half[ai] - ro[a]) * inv;
    if (t0 > t1) { const t = t0; t0 = t1; t1 = t; }
    tmin = Math.max(tmin, t0); tmax = Math.min(tmax, t1);
  });
  if (tmax < Math.max(tmin, 0)) return null;
  const [ex, ey, ez] = extG;
  const [dx, dy, dz] = volDimG;
  const steps = Math.max(ex, ey, ez) * 2; // ≥ Nyquist so we don't step over a voxel
  let t = Math.max(tmin, 0);
  const dt = (tmax - t) / steps;
  for (let i = 0; i < steps; i++, t += dt) {
    // world -> [0,1] per axis (p/size + 0.5)
    const u = (ro.x + rd.x * t) / sizeG[0] + 0.5;
    const v = (ro.y + rd.y * t) / sizeG[1] + 0.5;
    const w = (ro.z + rd.z * t) / sizeG[2] + 0.5;
    // Alpha comes off the (possibly coarse) volume.bin we actually hold.
    const x = Math.floor(u * ex), y = Math.floor(v * ey), z = Math.floor(w * ez);
    if (x < 0 || y < 0 || z < 0 || x >= ex || y >= ey || z >= ez) continue;
    const alpha = volBufG[(x + y * ex + z * ex * ey) * 4 + 3];
    if (alpha > 0) {
      // Entity test + reported coords are in full detail (vol_dim) space, where
      // the manifest bboxes live — so identification stays accurate and the
      // tooltip shows the tensor's native voxel coords even on a coarse grid.
      const fx = Math.min(dx - 1, Math.floor(u * dx));
      const fy = Math.min(dy - 1, Math.floor(v * dy));
      const fz = Math.min(dz - 1, Math.floor(w * dz));
      for (const e of manifestG) {
        const b = e.bbox;
        if (fx >= b.x0 && fx < b.x1 && fy >= b.y0 && fy < b.y1 && fz >= b.z0 && fz < b.z1) {
          return { entity: e, vx: fx, vy: fy, vz: fz, intensity: alpha / 255 };
        }
      }
      return null; // opaque voxel, but no entity owns it
    }
  }
  return null;
}

// Color legend keyed on the bundle's mode (structured/rgb only).
function buildLegend(meta) {
  const el = $('legend');
  if (meta.color_mode !== 'rgb') { el.innerHTML = ''; return; }
  if (meta.diff_mode) {
    el.innerHTML =
      '<div class="swrow"><span class="sw" style="background:#00ff00"></span>weight grew</div>' +
      '<div class="swrow"><span class="sw" style="background:#ff0000"></span>weight shrank</div>' +
      '<div class="swrow"><span class="sw" style="background:#fff"></span>non-finite</div>' +
      '<div class="hint">opacity = magnitude of change</div>';
  } else {
    el.innerHTML =
      '<div class="bar" style="background:linear-gradient(to right,rgb(0,34,78),rgb(124,123,120),rgb(254,232,56))"></div>' +
      '<div class="lab"><span>low</span><span>|weight|</span><span>high</span></div>';
  }
  const layers = new Set((meta.manifest || []).map((e) => e.group).filter((g) => /^layer /.test(g)));
  el.innerHTML += '<div class="hint">' +
    (layers.size ? layers.size + ' layers along depth (front → back)' : 'depth = layer order') +
    '</div>';
}

// Click (not drag) → pin the tensor under the cursor in the side panel.
let downX = 0, downY = 0, dragging = false;
canvas.addEventListener('pointerdown', (e) => { downX = e.clientX; downY = e.clientY; dragging = false; });
canvas.addEventListener('pointermove', (e) => {
  // Track orbit drags so the tooltip stays out of the way while rotating; reset
  // once the button is released so plain hover resumes.
  if (!e.buttons) dragging = false;
  else if (Math.hypot(e.clientX - downX, e.clientY - downY) > 5) dragging = true;
  showHover(e.clientX, e.clientY);
});
canvas.addEventListener('pointerleave', () => { $('tip').hidden = true; });
canvas.addEventListener('pointerup', (e) => {
  if (Math.hypot(e.clientX - downX, e.clientY - downY) > 5) return; // orbit drag, not a click
  const hit = pickAt(e.clientX, e.clientY);
  const el = $('picked');
  if (!el) return;
  el.innerHTML = hit
    ? '<div class="pname">' + esc(hit.entity.name) + '</div><div class="pgroup">' + esc(hit.entity.group) + '</div>'
    : '<span class="phint">no tensor there — click a colored region</span>';
});

// ---- per-voxel hover tooltip ------------------------------------------------
// Resolve the voxel under the cursor and show a tooltip beside it. pickAt is a
// CPU ray-march (~2·longest-axis steps + a manifest scan), but cheap enough to
// run per move; suppressed while the user is orbit-dragging.
function showHover(x, y) {
  const tip = $('tip');
  if (dragging) { tip.hidden = true; return; }
  const hit = pickAt(x, y);
  if (!hit) { tip.hidden = true; return; }
  tip.innerHTML = '<div class="tname">' + esc(hit.entity.name) + '</div>' +
    '<div class="tmeta">' + esc(hit.entity.group) +
    ' · (' + hit.vx + ', ' + hit.vy + ', ' + hit.vz + ')' +
    ' · ' + hit.intensity.toFixed(2) + '</div>';
  tip.hidden = false;
  // Offset from the cursor, clamped so the box stays on-screen.
  const r = tip.getBoundingClientRect();
  const px = Math.min(x + 14, innerWidth - r.width - 6);
  const py = Math.min(y + 14, innerHeight - r.height - 6);
  tip.style.left = Math.max(6, px) + 'px';
  tip.style.top = Math.max(6, py) + 'px';
}

// ---- per-layer depth labels -------------------------------------------------
// One floating label per group (each transformer layer + the top-level cap),
// positioned at the group's depth slab. Thinned to stay legible on deep models.
function buildLayerLabels() {
  if (!manifestG.length) return; // byte mode ships an empty manifest
  // Voxel v on axis a -> world position (matches the shader/box mapping). Group
  // bboxes are in full detail (vol_dim) space, so normalize by volDimG (== extG
  // for non-streamed bundles); sizeG carries the box aspect either way.
  const toWorld = (v, a) => ((v + 0.5) / volDimG[a] - 0.5) * sizeG[a];
  // Union bbox per group.
  const groups = new Map();
  for (const e of manifestG) {
    const b = e.bbox;
    let g = groups.get(e.group);
    if (!g) { g = { name: e.group, x0: b.x0, y0: b.y0, z0: b.z0, x1: b.x1, y1: b.y1, z1: b.z1 }; groups.set(e.group, g); }
    else { g.x0 = Math.min(g.x0, b.x0); g.y0 = Math.min(g.y0, b.y0); g.z0 = Math.min(g.z0, b.z0);
           g.x1 = Math.max(g.x1, b.x1); g.y1 = Math.max(g.y1, b.y1); g.z1 = Math.max(g.z1, b.z1); }
  }
  // Order by depth (slab Z center) so thinning keeps an even spread.
  const list = [...groups.values()].sort((a, b) => (a.z0 + a.z1) - (b.z0 + b.z1));
  // Thin to a sparse set: keep every k-th, plus the first, last, and any
  // non-"layer N" group (e.g. "top-level"). Kept low because the labels sit on
  // a single line along depth and foreshorten into each other at oblique camera
  // angles — a handful of evenly-spaced markers orients the viewer without mush.
  const CAP = 8;
  const k = Math.max(1, Math.ceil(list.length / CAP));
  list.forEach((g, i) => {
    const keep = i % k === 0 || i === 0 || i === list.length - 1 || !/^layer /.test(g.name);
    if (!keep) return;
    const el = document.createElement('div');
    el.className = 'layer-label';
    el.textContent = g.name;
    const obj = new CSS2DObject(el);
    // Pull X/Y to the box's top-left edge so labels sit beside the slab, not
    // buried inside it; Z at the slab center marches them back through depth.
    obj.position.set(toWorld(g.x0, 0) - 0.04, toWorld(g.y1 - 1, 1) + 0.04, toWorld((g.z0 + g.z1 - 1) / 2, 2));
    scene.add(obj);
  });
}

// ---- controls ---------------------------------------------------------------
function seg(id, cb) {
  const root = $(id);
  root.addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b) return;
    [...root.children].forEach((x) => x.classList.toggle('on', x === b));
    cb(b.dataset.v);
  });
}
function slider(id, label, fmt, cb) {
  const el = $(id), out = $(label);
  const apply = () => { out.textContent = fmt(parseFloat(el.value)); cb(parseFloat(el.value)); };
  el.addEventListener('input', apply); apply();
}

function bindControls() {
  const v = () => volMesh.material.uniforms;
  seg('src', (s) => { v().uSource.value = parseInt(s); refreshNorm(); });
  slider('opacity', 'opv', (x) => x.toFixed(2), (x) => v().uOpacity.value = x);
  slider('gamma', 'gav', (x) => x.toFixed(2), (x) => v().uGamma.value = x);
  // Threshold raises the volume floor (with the renorm compensation).
  slider('threshold', 'thv', (x) => x.toFixed(2), (x) => {
    v().uThreshold.value = x;
    refreshNorm();
  });
  refreshNorm();
}

// ---- loop -------------------------------------------------------------------
// Camera motion (incl. OrbitControls damping) re-arms the brick-streaming
// feedback probe.
controls.addEventListener('change', () => { volumeDirty = true; });

// Depth-correct compositing in three passes (see the "hybrid compositing" block):
//   1. opaque edges (layer 0) → sceneTarget (color + depth)
//   2. blit that color verbatim to the canvas
//   3. volume (layer 1) ray-marched over it, rays cut at the opaque depth
// labelRenderer draws the CSS2D layer text on top afterwards.
const _projView = new THREE.Matrix4();
function renderHybrid() {
  // No target yet (a zero-size viewport deferred its creation) — skip this frame
  // rather than dereference a null sceneTarget and kill the rAF loop. The
  // ResizeObserver will build it as soon as real dimensions arrive.
  if (!sceneTarget) return;
  // 1 — opaque pass.
  camera.layers.set(0);
  renderer.setRenderTarget(sceneTarget);
  renderer.render(scene, camera);
  renderer.setRenderTarget(null);
  // 2 — blit opaque color to the canvas (clears the canvas first).
  copyMat.uniforms.tColor.value = sceneTarget.texture;
  renderer.render(copyScene, copyCam);
  // 3 — volume composited over the blit.
  if (volMesh) {
    const u = volMesh.material.uniforms;
    renderer.getDrawingBufferSize(_dbSize);
    u.uResolution.value.copy(_dbSize);
    u.uSceneDepth.value = sceneTarget.depthTexture;
    camera.updateMatrixWorld();
    _projView.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    u.uInvProjView.value.copy(_projView).invert();
    volMesh.updateMatrixWorld();
    u.uInvModel.value.copy(volMesh.matrixWorld).invert();
    camera.layers.set(VOL_LAYER);
    renderer.autoClear = false;
    renderer.render(scene, camera);
    renderer.autoClear = true;
  }
  camera.layers.set(0); // restore for the label renderer + next-frame feedback
}

function tick() {
  controls.update();
  if (brickStream) streamBricks();
  renderHybrid();
  labelRenderer.render(scene, camera);
  requestAnimationFrame(tick);
}
tick();

load().catch((e) => { setStatus('error: ' + e.message); console.error(e); });
</script>
</body>
</html>
"##;
