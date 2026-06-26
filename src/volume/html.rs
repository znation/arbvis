//! Standalone Three.js viewer for the 3D (`--3d`) bundle.
//!
//! The 3D analog of [`crate::tiled::html`]. Like the Leaflet viewer it is a
//! self-contained `index.html` built as a string and loads its rendering
//! library (here Three.js) from a CDN via an ES-module import map, so it
//! deploys to an HF Space with no build step. At runtime it fetches
//! `meta.json`, `volume.bin`, and `points.bin` (written alongside it by
//! [`crate::volume::render_volume`]).

use crate::registry::Branding;

/// Build the 3D viewer HTML. Branding/title/inputs are injected as a JSON
/// config blob; everything else (grid extent, LUT, point count) is read from
/// `meta.json` at runtime.
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
</style>
</head>
<body>
<canvas id="c"></canvas>
<div id="labels"></div>
<div id="tip" hidden></div>
<div id="status">loading…</div>
<div id="panel" hidden>
  <h1 id="title">arbvis</h1>
  <div class="sub"><a id="repo" href="#" target="_blank" rel="noopener"></a></div>
  <div class="row">
    <label>View</label>
    <div class="seg" id="mode">
      <button data-v="volume" class="on">Volume</button>
      <button data-v="points">Points</button>
    </div>
  </div>
  <div id="volctl">
    <div class="row">
      <label>Opacity source</label>
      <div class="seg" id="src">
        <button data-v="0" class="on">Activity</button>
        <button data-v="1">Density</button>
      </div>
    </div>
    <div class="row"><label>Opacity <span id="opv"></span></label>
      <input id="opacity" type="range" min="0" max="1" step="0.01" value="0.2"></div>
    <div class="row"><label>Contrast <span id="gav"></span></label>
      <input id="gamma" type="range" min="0.2" max="3" step="0.05" value="1"></div>
    <div class="row"><label>Threshold <span id="thv"></span></label>
      <input id="threshold" type="range" min="0" max="0.95" step="0.01" value="0"></div>
    <div class="row"><label>Quality <span id="stv"></span></label>
      <input id="steps" type="range" min="48" max="384" step="16" value="192"></div>
  </div>
  <div id="ptctl" hidden>
    <div class="row"><label>Point size <span id="psv"></span></label>
      <input id="psize" type="range" min="0.5" max="6" step="0.1" value="1.6"></div>
    <div class="row"><label>Point opacity <span id="pov"></span></label>
      <input id="popacity" type="range" min="0.05" max="1" step="0.01" value="0.4"></div>
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

function resize() {
  const w = innerWidth, h = innerHeight;
  // updateStyle must stay on: a bare <canvas> with `inset:0` is a replaced
  // element, so CSS won't stretch it — without an inline style size it renders
  // at its (devicePixelRatio-scaled) drawing-buffer size, overflowing the
  // viewport and throwing off OrbitControls' pointer mapping.
  renderer.setSize(w, h);
  labelRenderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
addEventListener('resize', resize);
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
  uniform sampler3D uVolume;
  uniform sampler2D uLut;
  uniform float uOpacity, uGamma, uThreshold, uNorm;
  uniform int uSource, uSteps, uDirectColor;
  // Box world size per axis (longest axis = 1; isotropic cube = vec3(1)).
  uniform vec3 uSize;
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

  void main() {
    vec3 dir = normalize(vDirection);
    vec2 bounds = hitBox(vOrigin, dir);
    if (bounds.x > bounds.y) discard;
    bounds.x = max(bounds.x, 0.0);
    vec3 p = vOrigin + bounds.x * dir;
    float stepLen = (bounds.y - bounds.x) / float(uSteps);
    vec3 stepVec = dir * stepLen;
    float denom = max(1e-4, 1.0 - uThreshold);
    vec4 acc = vec4(0.0);
    for (int i = 0; i < 512; i++) {
      if (i >= uSteps) break;
      vec4 vox = texture(uVolume, p / uSize + 0.5);
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
        }
        d = max(0.0, d - uThreshold) / denom;
        // uNorm compensates for the data the threshold removed: as the field
        // thins, fewer voxels accumulate along each ray, so we scale per-voxel
        // alpha up to hold the cube's integrated opacity (its viewability)
        // roughly constant. Computed on the client from the channel histogram.
        float a = clamp(pow(d, uGamma) * uOpacity * uNorm, 0.0, 1.0);
        acc.rgb += (1.0 - acc.a) * col * a;
        acc.a   += (1.0 - acc.a) * a;
        if (acc.a >= 0.95) break;
      }
      p += stepVec;
    }
    if (acc.a <= 0.0) discard;
    fragColor = acc;
  }`;

const ptVert = `
  in vec4 aColor;
  out vec4 vColor;
  uniform float uPointSize;
  void main() {
    vColor = aColor;
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = uPointSize * (320.0 / -mv.z);
    gl_Position = projectionMatrix * mv;
  }`;

const ptFrag = `
  precision highp float;
  in vec4 vColor;
  out vec4 fragColor;
  uniform float uPointOpacity;
  void main() {
    vec2 d = gl_PointCoord - vec2(0.5);
    if (dot(d, d) > 0.25) discard;
    fragColor = vec4(vColor.rgb, vColor.a * uPointOpacity);
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
let volMesh = null, points = null;
// Kept for click-to-pick: the raw volume buffer + grid extent + world box size
// + entity manifest.
let volBufG = null, extG = [0, 0, 0], sizeG = [1, 1, 1], manifestG = [];
const raycaster = new THREE.Raycaster();

// Streamed point-LOD octree (format_version >= 2). Null when the bundle ships
// only the wholesale points.bin. `ptMaterial` is the shared point material so
// the size/opacity sliders drive both the flat cloud and every octree node.
let octree = null, ptMaterial = null, pointsActive = false, octreeDirty = true;

async function load() {
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

  // The volume uploads as a 3D texture sized [ex, ey, ez]; its largest axis is
  // the binding constraint. WebGL2 only guarantees MAX_3D_TEXTURE_SIZE ≥ 256,
  // and many GPUs cap at 1024 — so a large --grid can exceed what this device
  // can allocate. Check up front and fail with a clear message instead of a
  // cryptic GL allocation error.
  const gl = renderer.getContext();
  const maxSide = gl.getParameter(gl.MAX_3D_TEXTURE_SIZE);
  if (mx > maxSide) {
    throw new Error(
      'grid resolution ' + mx + ' exceeds the max 3D texture size this GPU ' +
      'supports (' + maxSide + '). Re-run arbvis with --grid ' + maxSide +
      ' or lower.');
  }

  // byte->color LUT as a 256x1 texture
  const lut = new Uint8Array(256 * 4);
  for (let i = 0; i < 256; i++) {
    const c = meta.lut[i] || [0, 0, 0];
    lut[i * 4] = c[0]; lut[i * 4 + 1] = c[1]; lut[i * 4 + 2] = c[2]; lut[i * 4 + 3] = 255;
  }
  const lutTex = new THREE.DataTexture(lut, 256, 1, THREE.RGBAFormat);
  lutTex.minFilter = lutTex.magFilter = THREE.NearestFilter;
  lutTex.needsUpdate = true;

  // volume RGBA8 -> Data3DTexture
  const volBuf = new Uint8Array(await (await fetch('volume.bin')).arrayBuffer());
  const volTex = new THREE.Data3DTexture(volBuf, ex, ey, ez);
  volTex.format = THREE.RGBAFormat;
  volTex.type = THREE.UnsignedByteType;
  volTex.minFilter = volTex.magFilter = THREE.LinearFilter;
  volTex.wrapS = volTex.wrapT = volTex.wrapR = THREE.ClampToEdgeWrapping;
  volTex.unpackAlignment = 1;
  volTex.needsUpdate = true;
  buildHistograms(volBuf);

  const volMat = new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3,
    side: THREE.BackSide, transparent: true, depthWrite: false,
    uniforms: {
      uVolume: { value: volTex }, uLut: { value: lutTex },
      uOpacity: { value: 0.2 }, uGamma: { value: 1.0 },
      uThreshold: { value: 0.0 }, uSource: { value: 0 }, uSteps: { value: 192 },
      uNorm: { value: 1.0 }, uDirectColor: { value: directColor ? 1 : 0 },
      uSize: { value: new THREE.Vector3(size[0], size[1], size[2]) },
    },
    vertexShader: volVert, fragmentShader: volFrag,
  });
  volMesh = new THREE.Mesh(new THREE.BoxGeometry(size[0], size[1], size[2]), volMat);
  scene.add(volMesh);
  // Match the orientation cube to the (possibly anisotropic) box.
  edges.scale.set(size[0], size[1], size[2]);

  // Point cloud. One shared material drives the size/opacity sliders for both
  // the wholesale cloud and the streamed octree nodes.
  const ptMat = new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3,
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    uniforms: { uPointSize: { value: 1.6 }, uPointOpacity: { value: 0.4 } },
    vertexShader: ptVert, fragmentShader: ptFrag,
  });
  ptMaterial = ptMat;

  if (meta.point_octree) {
    // Streamed LOD octree (the 3D analog of the 2D tile pyramid): nodes are
    // fetched on demand as the camera refines. See octreeUpdate().
    setupOctree(meta.point_octree, size, ptMat);
  } else {
    // Wholesale fallback: one buffer, one draw.
    const n = meta.points | 0;
    if (n > 0) {
      const pBuf = await (await fetch('points.bin')).arrayBuffer();
      const pos = new Float32Array(pBuf, 0, n * 3);
      const col = new Uint8Array(pBuf, n * 12, n * 4);
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      geo.setAttribute('aColor', new THREE.BufferAttribute(col, 4, true));
      points = new THREE.Points(geo, ptMat);
      // [0,1]-per-axis coords -> centered box: scale by the world size, then
      // shift so the box centers on the origin (a cube → scale 1, shift -0.5).
      points.scale.set(size[0], size[1], size[2]);
      points.position.set(-size[0] / 2, -size[1] / 2, -size[2] / 2);
      points.visible = false;
      scene.add(points);
    }
  }

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
  manifestG = meta.manifest || [];
  buildLegend(meta);
  buildLayerLabels();

  $('status').hidden = true;
  $('panel').hidden = false;
  bindControls();
}

// ---- streamed point-LOD octree ----------------------------------------------
// The 3D analog of the 2D Leaflet tile pyramid. The hierarchy index (one fixed
// record per node) is fetched once; each node's point block is range-fetched
// from points_octree.bin only when the camera refines into it. Points are laid
// in normalized [0,1] coords inside a Group whose transform centers the unit
// box (matching the volume mesh), so the two views register exactly.
const OCT_REFINE_PX = 220;     // refine a node when its on-screen diameter exceeds this
const OCT_BUDGET = 3_000_000;  // cap on points drawn per frame
const OCT_MAX_INFLIGHT = 8;    // concurrent range fetches

// Parse points_hierarchy.bin into nodes and rebuild the tree spatially: a
// node's parent is the depth-1 node whose cube contains it (origin floored to
// the parent's side grid) — no Hilbert math needed on the client.
function setupOctree(po, size, material) {
  const order = po.order, span = Math.pow(2, order);
  fetch(po.hierarchy_file).then((r) => r.arrayBuffer()).then((hb) => {
    const dv = new DataView(hb), REC = po.record_size;
    const nrec = (hb.byteLength / REC) | 0;
    const byKey = new Map(), nodes = [];
    let root = null;
    for (let i = 0; i < nrec; i++) {
      const o = i * REC;
      const node = {
        byteOffset: Number(dv.getBigUint64(o + 8, true)),
        byteLength: dv.getUint32(o + 16, true),
        pointCount: dv.getUint32(o + 20, true),
        origin: [dv.getUint32(o + 24, true), dv.getUint32(o + 28, true), dv.getUint32(o + 32, true)],
        depth: dv.getUint8(o + 36),
        coordBits: dv.getUint8(o + 38),
        children: [], obj: null, loading: false,
      };
      node.side = Math.pow(2, order - node.depth);
      nodes.push(node);
      byKey.set(node.depth + ':' + node.origin.join(','), node);
      if (node.depth === 0) root = node;
    }
    for (const node of nodes) {
      if (node.depth === 0) continue;
      const ps = node.side * 2;
      const par = node.origin.map((c) => Math.floor(c / ps) * ps);
      const parent = byKey.get((node.depth - 1) + ':' + par.join(','));
      if (parent) parent.children.push(node);
    }
    const group = new THREE.Group();
    group.scale.set(size[0], size[1], size[2]);
    group.position.set(-size[0] / 2, -size[1] / 2, -size[2] / 2);
    group.visible = false;
    scene.add(group);
    octree = { order, span, root, nodes, group, material, dataUrl: po.data_file, size, inflight: 0 };
    octreeDirty = true;
  }).catch((e) => console.error('octree hierarchy load failed', e));
}

// Decode a node's range-fetched block into a THREE.Points (normalized [0,1]
// coords reconstructed from per-node-local quantized coords + the node origin).
function makeNodePoints(node) {
  const n = node.pointCount, span = octree.span, o = node.origin, side = node.side;
  const cb = node.coordBits, stride = 3 * (cb / 8) + 4;
  const dv = new DataView(node.buf.buffer, node.buf.byteOffset, node.buf.byteLength);
  const pos = new Float32Array(n * 3), col = new Uint8Array(n * 4);
  for (let i = 0; i < n; i++) {
    const b = i * stride;
    let lx, ly, lz, coff;
    if (cb === 8) { lx = dv.getUint8(b); ly = dv.getUint8(b + 1); lz = dv.getUint8(b + 2); coff = b + 3; }
    else {
      lx = dv.getUint16(b, true); ly = dv.getUint16(b + 2, true); lz = dv.getUint16(b + 4, true); coff = b + 6;
      if (side > 65536) { const d = (q) => Math.round(q * (side - 1) / 65535); lx = d(lx); ly = d(ly); lz = d(lz); }
    }
    pos[i * 3] = (o[0] + lx) / span;
    pos[i * 3 + 1] = (o[1] + ly) / span;
    pos[i * 3 + 2] = (o[2] + lz) / span;
    col[i * 4] = dv.getUint8(coff); col[i * 4 + 1] = dv.getUint8(coff + 1);
    col[i * 4 + 2] = dv.getUint8(coff + 2); col[i * 4 + 3] = dv.getUint8(coff + 3);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('aColor', new THREE.BufferAttribute(col, 4, true));
  node.buf = null; // free the raw block
  return new THREE.Points(geo, octree.material);
}

// Range-fetch a node's block and add its points to the group. Falls back to a
// client-side slice if the host ignores the Range header (returns 200, whole
// file) — correct everywhere, fast where 206 is supported (HF Spaces serve
// many small tile files for 2D; a range-capable static host streams 3D nodes).
function loadNode(node) {
  node.loading = true; octree.inflight++;
  const end = node.byteOffset + node.byteLength - 1;
  fetch(octree.dataUrl, { headers: { Range: 'bytes=' + node.byteOffset + '-' + end } })
    .then((res) => res.arrayBuffer().then((buf) => {
      if (res.status !== 206 && buf.byteLength !== node.byteLength) {
        buf = buf.slice(node.byteOffset, node.byteOffset + node.byteLength);
      }
      node.buf = new Uint8Array(buf);
      node.obj = makeNodePoints(node);
      octree.group.add(node.obj);
      octreeDirty = true; // a new node arrived — re-evaluate the visible cut
    }))
    .catch((e) => console.error('octree node load failed', e))
    .finally(() => { node.loading = false; octree.inflight--; });
}

// Per-frame LOD pass: walk the tree, frustum-cull, render the "cut" (every node
// from the root down to where its on-screen size drops below the threshold or
// the budget runs out). Parent and child points are disjoint, so rendering the
// whole cut draws each point at most once. Out-of-view subtrees are skipped.
const _oBox = new THREE.Box3(), _oCenter = new THREE.Vector3();
function octreeUpdate() {
  if (!octree || !octree.root || !pointsActive) return;
  camera.updateMatrixWorld();
  const m = new THREE.Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
  const frustum = new THREE.Frustum().setFromProjectionMatrix(m);
  const H = innerHeight, tanHalf = Math.tan((camera.fov * Math.PI / 180) / 2);
  const sz = octree.size, span = octree.span;
  // Projected on-screen radius (px) of a node's cube, or -1 if frustum-culled.
  const projPx = (node) => {
    const o = node.origin, s = node.side;
    _oBox.min.set(o[0] / span * sz[0] - sz[0] / 2, o[1] / span * sz[1] - sz[1] / 2, o[2] / span * sz[2] - sz[2] / 2);
    _oBox.max.set((o[0] + s) / span * sz[0] - sz[0] / 2, (o[1] + s) / span * sz[1] - sz[1] / 2, (o[2] + s) / span * sz[2] - sz[2] / 2);
    if (!frustum.intersectsBox(_oBox)) return -1;
    _oBox.getCenter(_oCenter);
    const dist = Math.max(1e-4, _oCenter.distanceTo(camera.position));
    return (0.5 * _oBox.min.distanceTo(_oBox.max) / (dist * tanHalf)) * (H / 2);
  };
  const visible = new Set();
  let budget = OCT_BUDGET;
  const stack = [octree.root];
  while (stack.length) {
    const node = stack.pop();
    const p = projPx(node);
    if (p < 0 || budget - node.pointCount < 0) continue;
    visible.add(node);
    budget -= node.pointCount;
    if (node.children.length && p * 2 > OCT_REFINE_PX) {
      for (const c of node.children) stack.push(c);
    }
  }
  for (const node of octree.nodes) {
    if (visible.has(node)) {
      if (node.obj) node.obj.visible = true;
      else if (!node.loading && octree.inflight < OCT_MAX_INFLIGHT) loadNode(node);
    } else if (node.obj) {
      node.obj.visible = false;
    }
  }
}

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
  const steps = Math.max(ex, ey, ez) * 2; // ≥ Nyquist so we don't step over a voxel
  let t = Math.max(tmin, 0);
  const dt = (tmax - t) / steps;
  for (let i = 0; i < steps; i++, t += dt) {
    // world -> [0,1] per axis (p/size + 0.5) -> voxel index
    const x = Math.floor(((ro.x + rd.x * t) / sizeG[0] + 0.5) * ex);
    const y = Math.floor(((ro.y + rd.y * t) / sizeG[1] + 0.5) * ey);
    const z = Math.floor(((ro.z + rd.z * t) / sizeG[2] + 0.5) * ez);
    if (x < 0 || y < 0 || z < 0 || x >= ex || y >= ey || z >= ez) continue;
    const alpha = volBufG[(x + y * ex + z * ex * ey) * 4 + 3];
    if (alpha > 0) {
      for (const e of manifestG) {
        const b = e.bbox;
        if (x >= b.x0 && x < b.x1 && y >= b.y0 && y < b.y1 && z >= b.z0 && z < b.z1) {
          return { entity: e, vx: x, vy: y, vz: z, intensity: alpha / 255 };
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
  // Voxel v on axis a -> world position (matches the shader/box mapping).
  const toWorld = (v, a) => ((v + 0.5) / extG[a] - 0.5) * sizeG[a];
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
  seg('mode', (m) => {
    const vol = m === 'volume';
    if (volMesh) volMesh.visible = vol;
    pointsActive = !vol;
    if (points) points.visible = !vol;
    if (octree) { octree.group.visible = !vol; octreeDirty = true; }
    $('volctl').hidden = !vol;
    $('ptctl').hidden = vol;
  });
  seg('src', (s) => { v().uSource.value = parseInt(s); refreshNorm(); });
  slider('opacity', 'opv', (x) => x.toFixed(2), (x) => v().uOpacity.value = x);
  slider('gamma', 'gav', (x) => x.toFixed(2), (x) => v().uGamma.value = x);
  slider('threshold', 'thv', (x) => x.toFixed(2), (x) => { v().uThreshold.value = x; refreshNorm(); });
  slider('steps', 'stv', (x) => x.toFixed(0), (x) => v().uSteps.value = x);
  refreshNorm();
  // Point sliders drive the shared material (flat cloud or every octree node).
  if (ptMaterial) {
    const u = ptMaterial.uniforms;
    slider('psize', 'psv', (x) => x.toFixed(1), (x) => u.uPointSize.value = x);
    slider('popacity', 'pov', (x) => x.toFixed(2), (x) => u.uPointOpacity.value = x);
  }
}

// ---- loop -------------------------------------------------------------------
// Camera motion (incl. OrbitControls damping) re-evaluates the octree LOD cut.
controls.addEventListener('change', () => { octreeDirty = true; });
function tick() {
  controls.update();
  if (octreeDirty) { octreeUpdate(); octreeDirty = false; }
  renderer.render(scene, camera);
  labelRenderer.render(scene, camera);
  requestAnimationFrame(tick);
}
tick();

load().catch((e) => { $('status').textContent = 'error: ' + e.message; console.error(e); });
</script>
</body>
</html>
"##;
