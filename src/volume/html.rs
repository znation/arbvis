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
/// config blob; everything else (grid side, LUT, point count) is read from
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
  #status { position: fixed; inset: 0; display: flex; align-items: center; justify-content: center;
    color: #9aa0ab; pointer-events: none; }
  #status[hidden] { display: none; } /* id selector's display:flex would otherwise beat [hidden] */
</style>
</head>
<body>
<canvas id="c"></canvas>
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
  <div class="hint">drag rotate · right-drag pan · scroll zoom</div>
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

// faint bounding cube for orientation
const edges = new THREE.LineSegments(
  new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)),
  new THREE.LineBasicMaterial({ color: 0x3a3f4b }));
scene.add(edges);

function resize() {
  const w = innerWidth, h = innerHeight;
  renderer.setSize(w, h, false);
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
  uniform float uOpacity, uGamma, uThreshold;
  uniform int uSource, uSteps;
  in vec3 vOrigin;
  in vec3 vDirection;
  out vec4 fragColor;

  vec2 hitBox(vec3 o, vec3 d) {
    vec3 inv = 1.0 / d;
    vec3 a = (vec3(-0.5) - o) * inv;
    vec3 b = (vec3( 0.5) - o) * inv;
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
      vec4 vox = texture(uVolume, p + 0.5);
      if (vox.a > 0.0) {
        float d = (uSource == 0) ? vox.g : vox.b;
        d = max(0.0, d - uThreshold) / denom;
        float a = clamp(pow(d, uGamma) * uOpacity, 0.0, 1.0);
        vec3 col = texture(uLut, vec2(vox.r, 0.5)).rgb;
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

// ---- load + build -----------------------------------------------------------
let volMesh = null, points = null;

async function load() {
  const meta = await (await fetch('meta.json')).json();
  const side = meta.grid_side;

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
  const volTex = new THREE.Data3DTexture(volBuf, side, side, side);
  volTex.format = THREE.RGBAFormat;
  volTex.type = THREE.UnsignedByteType;
  volTex.minFilter = volTex.magFilter = THREE.LinearFilter;
  volTex.wrapS = volTex.wrapT = volTex.wrapR = THREE.ClampToEdgeWrapping;
  volTex.unpackAlignment = 1;
  volTex.needsUpdate = true;

  const volMat = new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3,
    side: THREE.BackSide, transparent: true, depthWrite: false,
    uniforms: {
      uVolume: { value: volTex }, uLut: { value: lutTex },
      uOpacity: { value: 0.2 }, uGamma: { value: 1.0 },
      uThreshold: { value: 0.0 }, uSource: { value: 0 }, uSteps: { value: 192 },
    },
    vertexShader: volVert, fragmentShader: volFrag,
  });
  volMesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), volMat);
  scene.add(volMesh);

  // point cloud
  const n = meta.points | 0;
  if (n > 0) {
    const pBuf = await (await fetch('points.bin')).arrayBuffer();
    const pos = new Float32Array(pBuf, 0, n * 3);
    const col = new Uint8Array(pBuf, n * 12, n * 4);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('aColor', new THREE.BufferAttribute(col, 4, true));
    const ptMat = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      uniforms: { uPointSize: { value: 1.6 }, uPointOpacity: { value: 0.4 } },
      vertexShader: ptVert, fragmentShader: ptFrag,
    });
    points = new THREE.Points(geo, ptMat);
    points.position.set(-0.5, -0.5, -0.5); // [0,1] coords -> centered cube
    points.visible = false;
    scene.add(points);
  }

  // Frame the camera on the occupied region (data often fills only part of
  // the cube — a small file is a contiguous Hilbert prefix in one corner).
  const fc = meta.focus_center || [0, 0, 0];
  const fr = meta.focus_radius || 0.5;
  controls.target.set(fc[0], fc[1], fc[2]);
  const dist = Math.max(fr * 3.2, 0.18);
  camera.position.set(fc[0] + dist, fc[1] + dist * 0.85, fc[2] + dist * 1.1);
  controls.update();

  $('status').hidden = true;
  $('panel').hidden = false;
  bindControls();
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
    if (points) points.visible = !vol;
    $('volctl').hidden = !vol;
    $('ptctl').hidden = vol;
  });
  seg('src', (s) => { v().uSource.value = parseInt(s); });
  slider('opacity', 'opv', (x) => x.toFixed(2), (x) => v().uOpacity.value = x);
  slider('gamma', 'gav', (x) => x.toFixed(2), (x) => v().uGamma.value = x);
  slider('threshold', 'thv', (x) => x.toFixed(2), (x) => v().uThreshold.value = x);
  slider('steps', 'stv', (x) => x.toFixed(0), (x) => v().uSteps.value = x);
  if (points) {
    const p = () => points.material.uniforms;
    slider('psize', 'psv', (x) => x.toFixed(1), (x) => p().uPointSize.value = x);
    slider('popacity', 'pov', (x) => x.toFixed(2), (x) => p().uPointOpacity.value = x);
  }
}

// ---- loop -------------------------------------------------------------------
function tick() {
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}
tick();

load().catch((e) => { $('status').textContent = 'error: ' + e.message; console.error(e); });
</script>
</body>
</html>
"##;
