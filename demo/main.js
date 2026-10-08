// Car-paint viewer: shows what the network recovered from one flash photo.
//
// Each validation sample (exported by src/export_demo.py) provides predicted
// and ground-truth textures plus clear-coat values. They drive a single
// MeshPhysicalMaterial:
//   base colour   -> map                       (sRGB texture)
//   roughness     -> roughnessMap  (green of the ORM texture)
//   metallic      -> metalnessMap  (blue of the ORM texture)
//   flake normals -> normalMap
//   coat weight   -> clearcoat
//   coat rough.   -> clearcoatRoughness
// Blender and three.js both square roughness into the microfacet alpha, so the
// numbers carry across unchanged.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';
import { KTX2Loader } from 'three/addons/loaders/KTX2Loader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

const SAMPLES = 'assets/samples/';
const CONFIG_URL = 'assets/config.json';
// Decoders for compressed models are fetched from the same three.js release.
const THREE_CDN = 'https://cdn.jsdelivr.net/npm/three@0.170.0/';
const PAINT_WORDS = ['carpaint', 'car_paint', 'paint', 'body', 'exterior', 'shell', 'chassis'];

const PARAMS = [
  { key: 'coat_roughness', label: 'Clear-coat roughness', digits: 3, recoverable: true },
  { key: 'flake_strength', label: 'Flake tilt strength', digits: 3, recoverable: true },
  { key: 'flake_scale', label: 'Flake scale', digits: 1, recoverable: true },
  { key: 'coat_weight', label: 'Clear-coat weight', digits: 2, recoverable: false },
  { key: 'peel_strength', label: 'Orange peel', digits: 3, recoverable: false },
];

const $ = (id) => document.getElementById(id);
const setStatus = (msg = '') => { $('status').textContent = msg; };
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

const state = {
  config: {},
  manifest: null,
  index: 0,
  mode: 'pred',          // 'pred' | 'gt'
  object: 'sphere',      // 'sphere' | 'car'
  repeat: 6,
  car: null,             // { group, meshes, materials: Map(name -> info) }
  paintName: null,
};

// ---------------------------------------------------------------- renderer
const canvas = $('view');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.toneMapping = THREE.AgXToneMapping;   // the tone curve the training photos used
renderer.toneMappingExposure = 1.0;

const scene = new THREE.Scene();
const pmrem = new THREE.PMREMGenerator(renderer);
// A softly lit room to reflect. The training photos had nothing to reflect but
// the flash, which is why the coat looks so much wetter here than in them.
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;

const camera = new THREE.PerspectiveCamera(35, 1, 0.05, 100);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.autoRotateSpeed = 0.8;
controls.autoRotate = !reducedMotion;
$('spin').checked = controls.autoRotate;

// ---------------------------------------------------------------- materials
const paint = new THREE.MeshPhysicalMaterial({
  color: 0xffffff, roughness: 1, metalness: 1, clearcoat: 1, clearcoatRoughness: 0.12,
});
// For car parts that have no texture coordinates: the same paint, flattened
// to its average values (textures cannot be placed without UVs).
const paintFlat = new THREE.MeshPhysicalMaterial({ clearcoat: 1, clearcoatRoughness: 0.12 });

const texLoader = new THREE.TextureLoader();
const texCache = new Map();
function texture(url, isColour) {
  if (!texCache.has(url)) {
    const t = texLoader.load(url);
    t.colorSpace = isColour ? THREE.SRGBColorSpace : THREE.NoColorSpace;
    // Mirrored tiling hides the seams of a patch that was never made to tile.
    t.wrapS = t.wrapT = THREE.MirroredRepeatWrapping;
    t.anisotropy = renderer.capabilities.getMaxAnisotropy();
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

const sphereGroup = new THREE.Group();
const ball = new THREE.Mesh(new THREE.SphereGeometry(0.75, 160, 120), paint);
ball.position.y = 0.95;
const plinth = new THREE.Mesh(
  new THREE.CylinderGeometry(0.5, 0.58, 0.2, 96),
  new THREE.MeshStandardMaterial({ color: 0x17191e, roughness: 0.85 }),
);
plinth.position.y = 0.1;
sphereGroup.add(ball, plinth, shadowDisc(1.3, 0.55));
scene.add(sphereGroup);

// ---------------------------------------------------------------- framing
function frame(target, distance, height) {
  controls.target.copy(target);
  camera.position.set(target.x, target.y + height, target.z + distance);
  controls.minDistance = distance * 0.45;
  controls.maxDistance = distance * 2.5;
  controls.update();
}
const frameSphere = () => frame(new THREE.Vector3(0, 0.85, 0), 4.2, 0.35);

// ---------------------------------------------------------------- apply sample
function currentScalars() {
  const s = state.manifest.samples[state.index];
  const own = s[state.mode] || {};
  // A maps-only checkpoint has no predicted scalars; fall back to defaults.
  return {
    coat_weight: own.coat_weight ?? 1.0,
    coat_roughness: own.coat_roughness ?? 0.12,
  };
}

function setRepeat() {
  const r = state.repeat;
  for (const t of [paint.map, paint.roughnessMap, paint.normalMap]) {
    if (!t) continue;
    // Sphere UVs run twice as far around as top to bottom.
    if (state.object === 'sphere') t.repeat.set(2 * r, r);
    else t.repeat.set(r, r);
  }
}

function applySample() {
  const m = state.manifest;
  const s = m.samples[state.index];
  const dir = `${SAMPLES}${s.folder}/`;
  const mode = state.mode;

  paint.map = texture(`${dir}${mode}_basecolor.png`, true);
  paint.roughnessMap = paint.metalnessMap = texture(`${dir}${mode}_orm.png`, false);
  paint.normalMap = texture(`${dir}${mode}_normal.png`, false);
  const coat = currentScalars();
  paint.clearcoat = coat.coat_weight;
  paint.clearcoatRoughness = coat.coat_roughness;
  paint.needsUpdate = true;
  setRepeat();

  const base = s.base || {};
  const bc = base[`${mode}_basecolor`] || [0.5, 0.5, 0.5];
  paintFlat.color.setRGB(bc[0], bc[1], bc[2]);          // linear, as exported
  paintFlat.roughness = base[`${mode}_roughness`] ?? 0.35;
  paintFlat.metalness = base[`${mode}_metallic`] ?? 0.85;
  paintFlat.clearcoat = coat.coat_weight;
  paintFlat.clearcoatRoughness = coat.coat_roughness;

  $('input-photo').src = `${dir}photo.png`;
  $('map-basecolor').src = `${dir}${mode}_basecolor.png`;
  $('map-rough').src = `${dir}${mode}_rough.png`;
  $('map-normal').src = `${dir}${mode}_normal.png`;

  renderTable(s);
  document.querySelectorAll('#samples button').forEach((b, i) => {
    b.setAttribute('aria-selected', String(i === state.index));
  });
}

function fmt(v, digits) {
  return typeof v === 'number' && Number.isFinite(v) ? v.toFixed(digits) : '—';
}

function renderTable(s) {
  const ranges = state.manifest.scalar_ranges || {};
  $('param-rows').innerHTML = PARAMS.map((p) => {
    const r = ranges[p.key];
    const range = r ? `<span class="range">range ${r[0]}–${r[1]}</span>` : '';
    return `<tr class="${p.recoverable ? '' : 'off'}">
      <td>${p.label}${range}</td>
      <td>${fmt(s.pred?.[p.key], p.digits)}</td>
      <td>${fmt(s.gt?.[p.key], p.digits)}</td>
    </tr>`;
  }).join('');
}

function buildGallery() {
  const grid = $('samples');
  grid.innerHTML = '';
  state.manifest.samples.forEach((s, i) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.setAttribute('role', 'option');
    b.setAttribute('aria-label', `Sample ${s.id}`);
    b.innerHTML = `<img src="${SAMPLES}${s.folder}/photo.png" alt="" loading="lazy">`;
    b.addEventListener('click', () => { state.index = i; applySample(); });
    grid.appendChild(b);
  });
}

function buildFooter() {
  const m = state.manifest;
  let repo = '';
  // On GitHub Pages (user.github.io/repo/) link back to the repository.
  if (location.hostname.endsWith('github.io')) {
    const user = location.hostname.split('.')[0];
    const name = location.pathname.split('/').filter(Boolean)[0];
    if (name) repo = ` · <a href="https://github.com/${user}/${name}">Code and write-up</a>`;
  }
  const credit = state.config.car_credit
    ? `<br>${escapeHtml(state.config.car_credit)}` : '';
  $('footer').innerHTML =
    `Checkpoint <code>${escapeHtml(m.checkpoint)}</code> (epoch ${m.checkpoint_epoch ?? '?'}). ` +
    `Samples from the ${escapeHtml(m.split)}. Synthetic data rendered in Blender.${repo}${credit}`;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

// ---------------------------------------------------------------- car
function paintFor(mesh) {
  // A part exported without normals renders black under a lit material. The
  // model's own material hides this (the loader flat-shades it); ours can't.
  if (!mesh.geometry.attributes.normal) mesh.geometry.computeVertexNormals();
  return mesh.geometry.attributes.uv ? paint : paintFlat;
}

function assignPaint(name) {
  const car = state.car;
  state.paintName = name;
  for (const mesh of car.meshes) {
    const orig = car.originals.get(mesh);
    if (Array.isArray(orig)) {
      mesh.material = orig.map((mat) => (mat.name === name ? paintFor(mesh) : mat));
    } else {
      mesh.material = orig.name === name ? paintFor(mesh) : orig;
    }
  }
}

function guessPaint(materials) {
  // Never guess a hidden material, or one used only by flat parts (floors,
  // backdrops, number plates).
  const candidates = [...materials].filter(([, info]) => !info.hidden && info.area > 0);
  if (!candidates.length) return [...materials.keys()][0];
  const byWord = candidates.find(([n]) => PAINT_WORDS.some((w) => n.toLowerCase().includes(w)));
  if (byWord) return byWord[0];
  // Otherwise: the material covering the most surface (bounding-box proxy).
  return candidates.reduce((a, b) => (b[1].area > a[1].area ? b : a))[0];
}

function isFlat(box) {
  // A mesh whose thinnest side is under 2% of its longest is a sheet: very
  // likely a floor or backdrop that shipped with the model.
  const s = box.getSize(new THREE.Vector3());
  const dims = [s.x, s.y, s.z].sort((a, b) => a - b);
  return dims[2] > 0 && dims[0] < 0.02 * dims[2];
}

function boxOf(meshes) {
  const box = new THREE.Box3();
  for (const m of meshes) box.expandByObject(m);
  return box;
}

async function tryLoadCar() {
  const file = state.config.car_model;
  if (!file) return;   // no car configured: the sphere is the whole demo

  setStatus('Loading car model…');
  // Downloaded models are often compressed in one of three ways; support all.
  const loader = new GLTFLoader()
    .setDRACOLoader(new DRACOLoader().setDecoderPath(`${THREE_CDN}examples/jsm/libs/draco/gltf/`))
    .setKTX2Loader(new KTX2Loader().setTranscoderPath(`${THREE_CDN}examples/jsm/libs/basis/`)
      .detectSupport(renderer))
    .setMeshoptDecoder(MeshoptDecoder);
  const gltf = await loader.loadAsync(`assets/${file}`);
  const group = gltf.scene;
  group.updateMatrixWorld(true);

  // ---- collect parts and materials
  const hide = new Set(state.config.car_hide_materials || []);
  const meshes = [];
  const originals = new Map();
  const materials = new Map();
  let unnamed = 0;
  group.traverse((o) => {
    if (!o.isMesh) return;
    meshes.push(o);
    originals.set(o, o.material);
    const mats = Array.isArray(o.material) ? o.material : [o.material];
    const box = new THREE.Box3().setFromObject(o);
    const b = box.getSize(new THREE.Vector3());
    o.userData.flat = isFlat(box);
    const area = o.userData.flat ? 0 : 2 * (b.x * b.y + b.y * b.z + b.x * b.z);
    for (const mat of mats) {
      if (!mat.name) mat.name = `material ${++unnamed}`;
      const info = materials.get(mat.name) || { area: 0, parts: 0, hidden: hide.has(mat.name) };
      info.area += area / mats.length;
      info.parts += 1;
      materials.set(mat.name, info);
    }
    // Hide a part only if every material on it is on the hide list.
    if (mats.every((m) => hide.has(m.name))) o.visible = false;
  });
  if (!meshes.length) throw new Error('the file contains no meshes');

  // ---- normalise size and position using the car itself, not a floor plane
  const body = meshes.filter((m) => m.visible && !m.userData.flat);
  const sizing = body.length ? body : meshes.filter((m) => m.visible);
  let box = boxOf(sizing);
  const size = box.getSize(new THREE.Vector3());
  group.scale.multiplyScalar(3.2 / Math.max(size.x, size.y, size.z));
  group.updateMatrixWorld(true);
  box = boxOf(sizing);
  const centre = box.getCenter(new THREE.Vector3());
  group.position.x -= centre.x;
  group.position.z -= centre.z;
  group.position.y -= box.min.y;
  group.updateMatrixWorld(true);
  box = boxOf(sizing);

  const footprint = Math.max(box.max.x - box.min.x, box.max.z - box.min.z);
  group.add(shadowDisc(footprint * 0.75, 0.6));
  group.visible = false;
  scene.add(group);
  state.car = { group, meshes, originals, materials, height: box.max.y };

  // What was found, for anyone debugging a new model (open the console).
  const report = new Map();
  for (const m of meshes) {
    const mats = Array.isArray(m.material) ? m.material : [m.material];
    for (const mat of mats) {
      const r = report.get(mat.name) || { box: new THREE.Box3(), uv: 0, normals: 0, parts: 0 };
      r.box.expandByObject(m);
      r.parts += 1;
      r.uv += m.geometry.attributes.uv ? 1 : 0;
      r.normals += m.geometry.attributes.normal ? 1 : 0;
      report.set(mat.name, r);
    }
  }
  console.log(`car scaled to longest side 3.2, height ${box.max.y.toFixed(2)}`);
  console.table([...report].map(([name, r]) => {
    const s = r.box.getSize(new THREE.Vector3());
    return {
      material: name, parts: r.parts,
      'with UVs': `${r.uv}/${r.parts}`, 'with normals': `${r.normals}/${r.parts}`,
      size: `${s.x.toFixed(2)} x ${s.y.toFixed(2)} x ${s.z.toFixed(2)}`,
      hidden: materials.get(name).hidden, flat: materials.get(name).area === 0,
    };
  }));

  const select = $('paint-select');
  select.innerHTML = [...materials].map(([n, i]) => {
    const note = i.hidden ? ', hidden' : i.area === 0 ? ', flat' : '';
    return `<option value="${escapeHtml(n)}">${escapeHtml(n)} (${i.parts} part${i.parts === 1 ? '' : 's'}${note})</option>`;
  }).join('');
  // A material named in config.json wins over the automatic guess.
  const pinned = state.config.car_paint_material;
  const guess = pinned && materials.has(pinned) ? pinned : guessPaint(materials);
  if (pinned && !materials.has(pinned)) {
    console.warn(`car_paint_material "${pinned}" not found; available: ${[...materials.keys()].join(', ')}`);
  }
  select.value = guess;
  assignPaint(guess);
  select.addEventListener('change', () => assignPaint(select.value));

  $('car-btn').disabled = false;
  $('car-btn').title = '';
  setStatus('');
  setObject('car');
}

function setObject(which) {
  if (which === 'car' && !state.car) return;
  state.object = which;
  sphereGroup.visible = which === 'sphere';
  if (state.car) state.car.group.visible = which === 'car';
  $('paint-pick').hidden = which !== 'car';
  document.querySelectorAll('[data-object]').forEach((b) => {
    b.classList.toggle('on', b.dataset.object === which);
  });
  if (which === 'car') {
    const h = state.car.height;
    frame(new THREE.Vector3(0, h * 0.45, 0), 6.2, h * 0.9);
  } else {
    frameSphere();
  }
  setRepeat();
}

// ---------------------------------------------------------------- UI wiring
document.querySelectorAll('[data-mode]').forEach((b) => b.addEventListener('click', () => {
  state.mode = b.dataset.mode;
  document.querySelectorAll('[data-mode]').forEach((x) => x.classList.toggle('on', x === b));
  if (state.manifest) applySample();
}));
document.querySelectorAll('[data-object]').forEach((b) => b.addEventListener('click', () => {
  setObject(b.dataset.object);
}));
$('spin').addEventListener('change', (e) => { controls.autoRotate = e.target.checked; });
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
frameSphere();

renderer.setAnimationLoop(() => {
  controls.update();
  renderer.render(scene, camera);
});

// ---------------------------------------------------------------- start
(async function start() {
  setStatus('Loading samples…');
  try {
    const res = await fetch(CONFIG_URL);
    if (res.ok) state.config = await res.json();
  } catch { /* config is optional */ }
  const manifestUrl = new URL(`${SAMPLES}manifest.json`, location.href).href;
  try {
    const res = await fetch(manifestUrl, { cache: 'no-cache' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    state.manifest = await res.json();
  } catch (err) {
    // Say exactly what failed and where: that turns a vague bug into a fix.
    const local = location.protocol === 'file:'
      ? ' This page was opened as a file; run "python -m http.server 8000" inside demo/ and open http://localhost:8000 instead.'
      : ' If you have exported samples, check they were committed and pushed.';
    setStatus(`Could not load samples: ${err.message} (${manifestUrl}).${local}`);
    console.error(err);
    return;
  }
  buildGallery();
  buildFooter();
  applySample();
  setStatus('');
  try {
    await tryLoadCar();
  } catch (err) {
    console.error(err);
    setStatus(`Car model failed to load (${err.message || err}); showing the sphere.`);
    setTimeout(() => setStatus(''), 12000);
  }
})();
