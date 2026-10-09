// v2 car-paint viewer: what the multi-light network recovered from a flash
// photo plus side-lit photos, for every paint type the v5 data has.
//
// Each sample (exported by src/export_demo_multi.py into assets/samples_v2/)
// provides predicted and ground-truth textures plus all ten layer parameters
// in real units. They drive one MeshPhysicalMaterial:
//   base colour     -> map                    (sRGB texture)
//   roughness       -> roughnessMap           (green of the ORM texture)
//   metallic        -> metalnessMap           (blue of the ORM texture)
//   flake normals   -> normalMap
//   coat weight     -> clearcoat
//   coat roughness  -> clearcoatRoughness
//   thin film       -> iridescence 1, iridescenceIOR = film IOR,
//                      iridescenceThicknessRange = [d, d] (d in nm);
//                      iridescence 0 when the paint has no film.
//                      APPROXIMATE: three.js takes the film's colour from
//                      the viewing angle alone; Blender uses the light-viewer
//                      half-vector
//   coat tint       -> color = 1 - w + w * tint (w = coat weight). AN
//                      APPROXIMATION: three.js has no clear-coat tint, so the
//                      base colour is multiplied instead. Blender's Coat Tint
//                      is the colour left after the trip in and out of the
//                      coat at normal incidence (applied once, not squared),
//                      and Coat Weight blends it in. Blender also deepens it
//                      where the view grazes the coat, up to tint^1.34 at the
//                      sphere's rim (coat IOR 1.5); this doesn't.
// Blender and three.js both square roughness into the microfacet alpha, so
// roughness values carry across unchanged (as in the v1 viewer, main.js).
//
// One directional light, placed relative to the camera so the sliders mean the
// same thing however the view is turned, plus an optional dim RoomEnvironment.
// With the light alone the scene is like the dark room of the training photos.
// Elevation and azimuth follow the v5 generator's lights (params_*.json): the
// camera looks straight at the swatch, elevation is the angle above the
// swatch (90 = at the camera, the flash), azimuth 0 = from the right of the
// photo, 90 = from the top. The middle of the sphere faces the camera as the
// swatch did, so it sees a side light at the angles the photo had.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

const SAMPLES = 'assets/samples_v2/';
// "No film" is film thickness 0. The network can't output exactly 0, so a
// predicted film thinner than this counts as none: half the thinnest film the
// v5 generator draws (pearl, 100 nm).
const FILM_MIN_NM = 50;
const LIGHT_INTENSITY = 3.0;
const ENV_INTENSITY = 0.25;      // "dim": the light should dominate
const FLASH_LIKE = { elevation: 90, azimuth: 0 };

const LABELS = {                 // label, decimals, unit
  coat_weight: ['Clear-coat weight', 2, ''],
  coat_roughness: ['Clear-coat roughness', 3, ''],
  coat_tint_r: ['Coat tint, red', 2, ''],
  coat_tint_g: ['Coat tint, green', 2, ''],
  coat_tint_b: ['Coat tint, blue', 2, ''],
  flake_scale: ['Flake scale', 1, ''],
  flake_strength: ['Flake tilt strength', 3, ''],
  peel_strength: ['Orange peel', 3, ''],
  film_thickness: ['Film thickness', 0, ' nm'],
  film_ior: ['Film IOR', 2, ''],
};
const PIGMENT_NAMES = {
  solid: 'Solid', metallic: 'Metallic', pearl: 'Pearl', colorshift: 'Colour-shift', candy: 'Candy',
};
const TINT_KEYS = ['coat_tint_r', 'coat_tint_g', 'coat_tint_b'];

const $ = (id) => document.getElementById(id);
const setStatus = (msg = '') => { $('status').textContent = msg; };
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

const state = {
  manifest: null,
  index: 0,
  mode: 'pred',          // 'pred' | 'gt'
  repeat: 6,
  elevation: 35,         // degrees above the surface facing the camera; 90 = at the camera
  azimuth: 45,           // degrees around the view; 0 = from the right, 90 = from the top
  env: true,             // dim room on/off
  spin: !reducedMotion,
  phase: 'loading',      // 'loading' | 'ready' | 'empty' | 'error'
  message: '',
};

// ---------------------------------------------------------------- renderer
const canvas = $('view');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.toneMapping = THREE.AgXToneMapping;   // the tone curve the v5 photos used
renderer.toneMappingExposure = 1.0;

const scene = new THREE.Scene();
const pmrem = new THREE.PMREMGenerator(renderer);
const roomEnv = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
scene.environmentIntensity = ENV_INTENSITY;

const camera = new THREE.PerspectiveCamera(35, 1, 0.05, 100);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;

const light = new THREE.DirectionalLight(0xffffff, LIGHT_INTENSITY);
scene.add(light, light.target);

// ---------------------------------------------------------------- material
const paint = new THREE.MeshPhysicalMaterial({
  color: 0xffffff, roughness: 1, metalness: 1, clearcoat: 1, clearcoatRoughness: 0.12,
});

const texLoader = new THREE.TextureLoader();
const texCache = new Map();
function texture(url, isColour) {
  if (!texCache.has(url)) {
    const t = texLoader.load(url);
    t.colorSpace = isColour ? THREE.SRGBColorSpace : THREE.NoColorSpace;
    // Mirrored tiling hides the seams of a patch that was never made to tile.
    t.wrapS = t.wrapT = THREE.MirroredRepeatWrapping;
    t.anisotropy = renderer.capabilities.getMaxAnisotropy();
    t.userData.url = url;
    texCache.set(url, t);
  }
  return texCache.get(url);
}

// ---------------------------------------------------------------- sphere
function shadowDisc(radius, strength) {
  const c = document.createElement('canvas');
  c.width = c.height = 128;
  const g = c.getContext('2d');
  const grad = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  grad.addColorStop(0, `rgba(0,0,0,${strength})`);
  grad.addColorStop(1, 'rgba(0,0,0,0)');
  g.fillStyle = grad;
  g.fillRect(0, 0, 128, 128);
  const mesh = new THREE.Mesh(
    new THREE.PlaneGeometry(radius * 2, radius * 2),
    new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(c), transparent: true, depthWrite: false }),
  );
  mesh.rotation.x = -Math.PI / 2;
  mesh.position.y = 0.002;
  return mesh;
}

const ball = new THREE.Mesh(new THREE.SphereGeometry(0.75, 160, 120), paint);
ball.position.y = 0.95;
const plinth = new THREE.Mesh(
  new THREE.CylinderGeometry(0.5, 0.58, 0.2, 96),
  new THREE.MeshStandardMaterial({ color: 0x17191e, roughness: 0.85 }),
);
plinth.position.y = 0.1;
scene.add(ball, plinth, shadowDisc(1.3, 0.55));
light.target.position.copy(ball.position);

const target = new THREE.Vector3(0, 0.85, 0);
controls.target.copy(target);
camera.position.set(0, target.y + 0.35, 4.2);
controls.minDistance = 4.2 * 0.45;
controls.maxDistance = 4.2 * 2.5;
controls.update();

// ---------------------------------------------------------------- light
const lightDir = new THREE.Vector3();
const toCam = new THREE.Vector3();
const right = new THREE.Vector3();
const up = new THREE.Vector3();
function placeLight() {
  // Axes at the sphere's centre, as in generate_dataset_v5.py with the camera
  // on +z: z towards the camera, x to the right of the screen, y up it. From
  // the centre rather than along the camera's axis (which aims a little
  // below it), so elevation 90 is exactly at the camera, like the flash.
  toCam.subVectors(camera.position, ball.position).normalize();
  right.set(1, 0, 0).applyQuaternion(camera.quaternion);
  up.crossVectors(toCam, right).normalize();
  right.crossVectors(up, toCam);
  const el = THREE.MathUtils.degToRad(state.elevation);
  const az = THREE.MathUtils.degToRad(state.azimuth);
  lightDir.copy(toCam).multiplyScalar(Math.sin(el))
    .addScaledVector(right, Math.cos(el) * Math.cos(az))
    .addScaledVector(up, Math.cos(el) * Math.sin(az));
  light.position.copy(ball.position).addScaledVector(lightDir, 10);
}

function applyLightUi() {
  $('elev').value = state.elevation;
  $('azim').value = state.azimuth;
  $('elev-val').textContent = `${state.elevation}°`;
  $('azim-val').textContent = `${state.azimuth}°`;
  scene.environment = state.env ? roomEnv : null;
  document.querySelectorAll('[data-env]').forEach((b) => {
    b.classList.toggle('on', (b.dataset.env === 'room') === state.env);
  });
}

// ---------------------------------------------------------------- helpers
const num = (v, fallback) => (typeof v === 'number' && Number.isFinite(v) ? v : fallback);
const clamp01 = (v) => Math.min(Math.max(v, 0), 1);
const pigmentName = (p) => (p == null ? 'unknown' : PIGMENT_NAMES[p] || p);

function fmt(v, digits, unit = '') {
  return typeof v === 'number' && Number.isFinite(v) ? `${v.toFixed(digits)}${unit}` : '—';
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function swatch(linearRgb) {
  if (!Array.isArray(linearRgb) || linearRgb.length < 3) return '—';
  // Values are linear; getStyle() converts to sRGB for CSS.
  const c = new THREE.Color().setRGB(...linearRgb.slice(0, 3).map((v) => clamp01(num(v, 0))));
  return `<span class="v2-swatch" style="background:${c.getStyle()}"></span>`;
}

// ---------------------------------------------------------------- apply sample
const sample = () => state.manifest.samples[state.index];

// The film the material should show for these values: null for none.
function filmOf(values, mode) {
  const d = num(values.film_thickness, 0);
  const ior = num(values.film_ior, NaN);
  // The truth's "no film" is exactly 0; a prediction needs FILM_MIN_NM.
  const present = mode === 'gt' ? d > 0 : d >= FILM_MIN_NM;
  return present && Number.isFinite(ior) ? { thickness: d, ior } : null;
}

function setRepeat() {
  const r = state.repeat;
  for (const t of [paint.map, paint.roughnessMap, paint.normalMap]) {
    // Sphere UVs run twice as far around as top to bottom.
    if (t) t.repeat.set(2 * r, r);
  }
}

function applyMaterial() {
  const s = sample();
  const mode = state.mode;
  const v = s[mode] || {};
  const dir = `${SAMPLES}${s.folder}/`;

  paint.map = texture(`${dir}${mode}_basecolor.png`, true);
  paint.roughnessMap = paint.metalnessMap = texture(`${dir}${mode}_orm.png`, false);
  paint.normalMap = texture(`${dir}${mode}_normal.png`, false);
  paint.roughness = 1;
  paint.metalness = 1;

  paint.clearcoat = clamp01(num(v.coat_weight, 1));
  paint.clearcoatRoughness = clamp01(num(v.coat_roughness, 0.12));

  // Candy: the layers under the coat get 1 - w + w * tint, as in Blender
  // (see the top of this file). Linear values.
  const w = paint.clearcoat;
  paint.color.setRGB(...TINT_KEYS.map((k) => 1 - w + w * clamp01(num(v[k], 1))));

  const film = filmOf(v, mode);
  paint.iridescence = film ? 1 : 0;
  if (film) {
    paint.iridescenceIOR = film.ior;
    paint.iridescenceThicknessRange = [film.thickness, film.thickness];
  }
  paint.needsUpdate = true;
  setRepeat();

  $('map-basecolor').src = `${dir}${mode}_basecolor.png`;
  $('map-rough').src = `${dir}${mode}_rough.png`;
  $('map-normal').src = `${dir}${mode}_normal.png`;
}

function renderPhotos(s) {
  const seen = new Set(state.manifest.photos || []);
  const photos = s.photos || [];
  $('photos').innerHTML = photos.map((p) => {
    const used = seen.size === 0 || seen.has(p.name);
    const label = p.name === 'photo' ? 'Flash' : p.name.replace(/^side(\d+)$/, 'Side light $1');
    return `<figure class="${used ? '' : 'v2-unseen'}">
      <img src="${SAMPLES}${escapeHtml(s.folder)}/${escapeHtml(p.file)}" alt="${escapeHtml(label)} photo">
      <figcaption>${escapeHtml(label)}${used ? '' : ' · not shown to the network'}</figcaption>
    </figure>`;
  }).join('');
}

function renderType(s) {
  const pigments = state.manifest.pigments || [];
  const probs = s.pigment_probs || {};
  const pred = s.pigment_pred;
  const truth = s.pigment_gt;
  const finish = s.finish ? `, ${escapeHtml(s.finish)}` : '';
  const verdict = pred == null || truth == null ? ''
    : pred === truth ? ' · <span class="v2-right">right</span>' : ' · <span class="v2-wrong">wrong</span>';
  $('type-line').innerHTML =
    `Predicted <strong>${escapeHtml(pigmentName(pred))}</strong>` +
    `${pred != null && probs[pred] != null ? ` (${Math.round(100 * probs[pred])}%)` : ''}` +
    ` · true <strong>${escapeHtml(pigmentName(truth))}</strong>${finish}${verdict}`;
  $('probs').innerHTML = pigments.map((p) => {
    const pr = num(probs[p], 0);
    const cls = [p === pred ? 'v2-pred' : '', p === truth ? 'v2-true' : ''].join(' ');
    return `<div class="v2-prob ${cls}">
      <span class="v2-prob-name">${escapeHtml(pigmentName(p))}${p === truth ? ' <small>true</small>' : ''}</span>
      <span class="v2-bar"><span style="width:${(100 * pr).toFixed(1)}%"></span></span>
      <span class="v2-prob-val">${(100 * pr).toFixed(0)}%</span>
    </div>`;
  }).join('');
}

function renderTables(s) {
  const m = state.manifest;
  const keys = m.scalar_keys || Object.keys(s.pred || {});
  const ranges = m.scalar_ranges || {};
  const mask = s.mask || {};
  $('param-rows').innerHTML = keys.map((k) => {
    const [label, digits, unit] = LABELS[k] || [k, 3, ''];
    const r = ranges[k];
    const range = r ? `<span class="range">range ${r[0]}–${r[1]}${unit}</span>` : '';
    const meaningful = mask[k] !== false;
    const cell = (v) => (meaningful ? fmt(v, digits, unit) : 'n/a');
    return `<tr class="${meaningful ? '' : 'off'}" data-key="${escapeHtml(k)}">
      <td>${escapeHtml(label)}${range}</td>
      <td>${cell(s.pred?.[k])}</td>
      <td>${cell(s.gt?.[k])}</td>
    </tr>`;
  }).join('');

  const base = s.base || {};
  const tint = (v) => (v && TINT_KEYS.every((k) => typeof v[k] === 'number')
    ? swatch(TINT_KEYS.map((k) => v[k])) : '—');
  const filmText = (v, mode) => {
    const f = filmOf(v || {}, mode);
    return f ? `${f.thickness.toFixed(0)} nm, IOR ${f.ior.toFixed(2)}` : 'none';
  };
  const rows = [
    ['Base colour', swatch(base.pred_basecolor), swatch(base.gt_basecolor)],
    ['Coat tint', tint(s.pred), tint(s.gt)],
    ['Base roughness', fmt(base.pred_roughness, 3), fmt(base.gt_roughness, 3)],
    ['Base metallic', fmt(base.pred_metallic, 3), fmt(base.gt_metallic, 3)],
    ['Film as drawn', filmText(s.pred, 'pred'), filmText(s.gt, 'gt')],
  ];
  $('base-rows').innerHTML = rows.map(([label, p, t]) => (
    `<tr><td>${label}</td><td>${p}</td><td>${t}</td></tr>`)).join('');
}

function applySample() {
  const s = sample();
  applyMaterial();
  renderPhotos(s);
  renderType(s);
  renderTables(s);
  document.querySelectorAll('#samples button').forEach((b, i) => {
    b.setAttribute('aria-selected', String(i === state.index));
  });
}

function buildGallery() {
  const grid = $('samples');
  grid.innerHTML = '';
  state.manifest.samples.forEach((s, i) => {
    const flash = (s.photos || []).find((p) => p.name === 'photo') || (s.photos || [])[0];
    const type = pigmentName(s.pigment_gt);
    const b = document.createElement('button');
    b.type = 'button';
    b.setAttribute('role', 'option');
    b.setAttribute('aria-label', `Sample ${s.id}: ${type}${s.finish ? `, ${s.finish}` : ''}`);
    b.title = `${s.id}: ${type}${s.finish ? `, ${s.finish}` : ''}`;
    b.innerHTML = (flash ? `<img src="${SAMPLES}${escapeHtml(s.folder)}/${escapeHtml(flash.file)}" alt="" loading="lazy">` : '')
      + `<span class="v2-tag">${escapeHtml(type)}</span>`;
    b.addEventListener('click', () => { state.index = i; applySample(); });
    grid.appendChild(b);
  });
}

function buildFooter() {
  const m = state.manifest;
  const seen = m.photos || [];
  const sides = seen.filter((p) => p !== 'photo').length;
  const shown = seen.length === 1 && seen[0] === 'photo' ? 'the flash photo only'
    : `the flash photo and ${sides} side-lit photo${sides === 1 ? '' : 's'}`;
  $('photos-note').textContent = `The network was shown ${shown}, the photos this run trained on` +
    `${m.photo_policy ? ` (--photos ${m.photo_policy})` : ''}.`;
  let repo = '';
  // On GitHub Pages (user.github.io/repo/) link back to the repository.
  if (location.hostname.endsWith('github.io')) {
    const user = location.hostname.split('.')[0];
    const name = location.pathname.split('/').filter(Boolean)[0];
    if (name) repo = ` · <a href="https://github.com/${user}/${name}">Code and write-up</a>`;
  }
  // The exporter checks whether the network may have learned these samples;
  // claim "held out" only when it could tell (null: it couldn't).
  $('held-out').textContent = m.held_out === true
    ? ' None of these samples was used for training.'
    : m.held_out === false
      ? ' Careful: these samples are not all held out from training, so they may flatter the network (see the bottom of the panel).'
      : '';
  $('footer').innerHTML =
    `Checkpoint <code>${escapeHtml(m.checkpoint)}</code> (epoch ${escapeHtml(m.epoch ?? '?')}). ` +
    `Samples: ${escapeHtml(m.split || 'unknown split')}, <code>${escapeHtml(m.dataset || '')}</code>. ` +
    `Synthetic data rendered in Blender. <a href="./">Single-photo viewer (v1)</a>${repo}`;
}

function showEmpty(why) {
  state.phase = 'empty';
  if (why) $('empty-why').textContent = why;
  $('empty').hidden = false;
  // A plain grey paint, so the light controls still do something.
  paint.color.setRGB(0.18, 0.19, 0.21);
  paint.roughness = 0.4;
  paint.metalness = 0.6;
  paint.needsUpdate = true;
  setStatus('No v2 samples exported yet (see the panel).');
}

// ---------------------------------------------------------------- UI wiring
document.querySelectorAll('[data-mode]').forEach((b) => b.addEventListener('click', () => {
  state.mode = b.dataset.mode;
  document.querySelectorAll('[data-mode]').forEach((x) => x.classList.toggle('on', x === b));
  if (state.phase === 'ready') applySample();
}));
document.querySelectorAll('[data-env]').forEach((b) => b.addEventListener('click', () => {
  state.env = b.dataset.env === 'room';
  applyLightUi();
}));
$('elev').addEventListener('input', (e) => { state.elevation = Number(e.target.value); applyLightUi(); });
$('azim').addEventListener('input', (e) => { state.azimuth = Number(e.target.value); applyLightUi(); });
$('flash-btn').addEventListener('click', () => {
  state.elevation = FLASH_LIKE.elevation;
  state.azimuth = FLASH_LIKE.azimuth;
  applyLightUi();
});
$('spin').checked = state.spin;
$('spin').addEventListener('change', (e) => { state.spin = e.target.checked; });
$('repeat').addEventListener('input', (e) => { state.repeat = Number(e.target.value); setRepeat(); });

// ---------------------------------------------------------------- loop
function resize() {
  const { clientWidth: w, clientHeight: h } = canvas.parentElement;
  if (!w || !h) return;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(canvas.parentElement);
resize();
applyLightUi();

const clock = new THREE.Clock();
renderer.setAnimationLoop(() => {
  const dt = Math.min(clock.getDelta(), 0.1);
  // Spin the ball, not the camera: the light stays where the sliders put it
  // relative to the view, and the flakes move through the highlight.
  if (state.spin) ball.rotation.y += dt * 0.25;
  controls.update();
  placeLight();
  renderer.render(scene, camera);
});

// ---------------------------------------------------------------- debugging
// Read-only view of what is on screen, for anyone checking the viewer (and
// for automated tests): window.__v2.snapshot() returns a fresh plain object.
const v3 = (v) => [v.x, v.y, v.z];
Object.defineProperty(window, '__v2', {
  value: Object.freeze({
    snapshot: () => ({
      phase: state.phase,
      message: state.message,
      mode: state.mode,
      index: state.index,
      sampleId: state.manifest?.samples?.[state.index]?.id ?? null,
      sampleCount: state.manifest?.samples?.length ?? 0,
      elevation: state.elevation,
      azimuth: state.azimuth,
      environment: scene.environment !== null,
      environmentIntensity: scene.environmentIntensity,
      repeat: state.repeat,
      material: {
        clearcoat: paint.clearcoat,
        clearcoatRoughness: paint.clearcoatRoughness,
        iridescence: paint.iridescence,
        iridescenceIOR: paint.iridescenceIOR,
        iridescenceThicknessRange: [...paint.iridescenceThicknessRange],
        color: [paint.color.r, paint.color.g, paint.color.b],
        roughness: paint.roughness,
        metalness: paint.metalness,
        map: paint.map?.userData.url ?? null,
        roughnessMap: paint.roughnessMap?.userData.url ?? null,
        metalnessMap: paint.metalnessMap?.userData.url ?? null,
        normalMap: paint.normalMap?.userData.url ?? null,
        mapRepeat: paint.map ? [paint.map.repeat.x, paint.map.repeat.y] : null,
      },
      light: {
        intensity: light.intensity,
        // Unit vector from the sphere towards the light, world space...
        direction: v3(light.position.clone().sub(light.target.position).normalize()),
        // ...and towards the camera, to check the light follows the view.
        toCamera: v3(camera.position.clone().sub(ball.position).normalize()),
      },
    }),
  }),
  writable: false,
  configurable: false,
});

// ---------------------------------------------------------------- start
(async function start() {
  setStatus('Loading samples…');
  const manifestUrl = new URL(`${SAMPLES}manifest.json`, location.href).href;
  let res;
  try {
    res = await fetch(manifestUrl, { cache: 'no-cache' });
  } catch (err) {
    state.phase = 'error';
    state.message = `Could not load ${manifestUrl}: ${err.message}`;
    setStatus(location.protocol === 'file:'
      ? 'This page needs a web server. In a terminal inside the demo folder run "python -m http.server 8000", then open http://localhost:8000/v2.html'
      : `${state.message}.`);
    console.error(err);
    return;
  }
  if (res.status === 404) {
    showEmpty();
    return;
  }
  try {
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    state.manifest = await res.json();
  } catch (err) {
    // Say exactly what failed and where: that turns a vague bug into a fix.
    state.phase = 'error';
    state.message = `Could not load samples: ${err.message} (${manifestUrl})`;
    setStatus(`${state.message}.`);
    console.error(err);
    return;
  }
  if (!Array.isArray(state.manifest.samples) || !state.manifest.samples.length) {
    showEmpty(state.manifest.incomplete
      ? 'The last export into assets/samples_v2/ stopped before it finished.'
      : 'assets/samples_v2/manifest.json lists no samples.');
    return;
  }
  $('content').hidden = false;
  $('samples-block').hidden = false;
  $('repeat-block').hidden = false;
  // Runs on v3/v4 data have no paint types to show.
  $('type-block').hidden = !(state.manifest.pigments || []).length;
  buildGallery();
  buildFooter();
  state.phase = 'ready';
  applySample();
  setStatus('');
})();
