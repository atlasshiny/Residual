import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { settings, save, reset, pullFrom } from "./settings.js";

/* ---------------- DOM refs ---------------- */
const viewport = document.getElementById("viewport");
const sceneCaption = document.getElementById("scene-caption");
const loadingOverlay = document.getElementById("loading-overlay");
const errorBanner = document.getElementById("error-banner");
const sceneStatus = document.getElementById("scene-status");
const trajStatus = document.getElementById("trajectory-status");
const trajDot = document.getElementById("traj-dot");
const trajKv = document.getElementById("traj-kv");
const pinnStatus = document.getElementById("pinn-status");
const pinnDot = document.getElementById("pinn-dot");
const eqTarget = document.getElementById("eq-target");
const eqSindy = document.getElementById("eq-sindy");
const eqSindyLabel = document.getElementById("eq-sindy-label");
const pinnMetrics = document.getElementById("pinn-metrics");
const pinnRaw = document.getElementById("pinn-raw");
const stripMetrics = document.getElementById("strip-metrics");

const autoRotateCb = document.getElementById("auto-rotate");
const showGridCb = document.getElementById("show-grid");
const showAxesCb = document.getElementById("show-axes");
const residualCanvas = document.getElementById("residual-canvas");
const guideCanvas = document.getElementById("guide-canvas");
const residualStatus = document.getElementById("residual-status");
const residualDot = document.getElementById("residual-dot");
const residualHover = document.getElementById("residual-hover");
const residualMetrics = document.getElementById("residual-metrics");
const provenanceEl = document.getElementById("provenance");
const orbitSlider = document.getElementById("orbit-slider");
const orbitPlayBtn = document.getElementById("orbit-play");
const orbitTimeEl = document.getElementById("orbit-time");
const filmstripEl = document.getElementById("filmstrip");
const jobBaseInput = document.getElementById("job-base");
const scanJobsBtn = document.getElementById("scan-jobs-btn");
const jobListEl = document.getElementById("job-list");
const cosmosMetricsEl = document.getElementById("cosmos-metrics");
const exportSnapshotBtn = document.getElementById("export-snapshot-btn");
const exportReportBtn = document.getElementById("export-report-btn");
const showRefCb = document.getElementById("show-ref");
const showPropCb = document.getElementById("show-prop");
const showCorrCb = document.getElementById("show-corr");
const showEnsembleCb = document.getElementById("show-ensemble");
const showSunCb = document.getElementById("show-sun");
const ensembleSlider = document.getElementById("ensemble-slider");
const ensembleInfo = document.getElementById("ensemble-info");
let ensembleData = null;
let ensemblePoints = null;
const pipelineLog = document.getElementById("pipeline-log");
const pipelineDot = document.getElementById("pipeline-dot");
const pipelineStatus = document.getElementById("pipeline-status");
const pickFolderBtn = document.getElementById("pick-folder-btn");

const tauri = window.__TAURI__ || null;
const hasBackend = !!tauri?.core?.invoke;
async function invokeCmd(name, args) {
  if (!hasBackend) throw new Error("Tauri backend unavailable (running in browser?)");
  return tauri.core.invoke(name, args || {});
}
async function openFolderDialog() {
  try {
    return await invokeCmd("pick_folder");
  } catch {
    return null;
  }
}
function isLocalPath(p) {
  return /^([a-zA-Z]:[\\/]|\/|\\\\)/.test(p);
}
async function loadJobFromPath(dir) {
  const names = await invokeCmd("list_dir", { path: dir });
  const files = [];
  for (const name of names) {
    if (!name.toLowerCase().endsWith(".json")) continue;
    const text = await invokeCmd("read_text_file", {
      path: `${dir.replace(/[\\/]+$/, "")}/${name}`,
    });
    files.push(new File([text], name, { type: "application/json" }));
  }
  if (!files.length) throw new Error(`no JSON files in ${dir}`);
  return loadJobFromFiles(files);
}

// data globals
let predictionsData = null;
let manifestData = null;
let uiManifestData = null;
let residualTargetCols = [];
let residualHoverIdx = -1;
let residualChart = {};
let orbitPlaying = false;
let orbitAcc = 0;
let orbitLastTick = null;
let traveledLine = null;
let traveledTubeRef = null;
let trajectoryData = null;
let trajRadius = 0;
let orbitPlane = null;
let ensembleEpoch = -1;
const resetViewBtn = document.getElementById("reset-view");
const topViewBtn = document.getElementById("top-view");

/* cosmos viewer */
const modeTabs = document.getElementById("mode-tabs");
const cosmosStatus = document.getElementById("cosmos-status");
const cosmosDot = document.getElementById("cosmos-dot");
const cosmosKv = document.getElementById("cosmos-kv");
const frameInfo = document.getElementById("cosmos-frame-info");
const frameEmptyNote = document.getElementById("frame-empty-note");
const frameMeta = document.getElementById("cosmos-frame-meta");
const poseMatrix = document.getElementById("cosmos-pose-matrix");
const manifestRaw = document.getElementById("cosmos-manifest-raw");
const loadJobBtn = document.getElementById("load-job-btn");
const loadDemoBtn = document.getElementById("load-demo-btn");
const jobDirInput = document.getElementById("job-dir-input");
const scrubber = document.getElementById("scrubber");
const playBtn = document.getElementById("play-btn");
const frameSlider = document.getElementById("frame-slider");
const frameLabel = document.getElementById("frame-label");
const timeLabel = document.getElementById("time-label");
const scrubProgress = document.getElementById("scrub-progress");
const scrubThumb = document.getElementById("scrub-thumb");
const frameRender = document.getElementById("frame-render");
const frameImg = document.getElementById("frame-img");
const frameRenderLabel = document.getElementById("frame-render-label");

/* ---------------- Scene ---------------- */
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x050816, 0.00002);

const camera = new THREE.PerspectiveCamera(55, 1, 1, 10000000);
camera.position.set(0, 6000, 22000);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(10, 10);
viewport.appendChild(renderer.domElement);

/* lighting */
const ambient = new THREE.AmbientLight(0x334466, 0.6);
scene.add(ambient);
const sunLight = new THREE.DirectionalLight(0xffffff, 1.1);
sunLight.position.set(5000, 4000, 6000);
scene.add(sunLight);
const rim = new THREE.PointLight(0x22d3ee, 0.8, 60000);
rim.position.set(-8000, -2000, -6000);
scene.add(rim);

/* Earth */
const EARTH_RADIUS = 1000;
const earth = new THREE.Mesh(
  new THREE.SphereGeometry(EARTH_RADIUS, 64, 64),
  new THREE.MeshStandardMaterial({
    color: 0x1f4e8c,
    roughness: 0.7,
    metalness: 0.1,
    emissive: 0x0a1a3a,
    emissiveIntensity: 0.35,
  })
);
scene.add(earth);

/* atmosphere glow */
const atmosphere = new THREE.Mesh(
  new THREE.SphereGeometry(EARTH_RADIUS * 1.12, 64, 64),
  new THREE.MeshBasicMaterial({
    color: 0x22d3ee,
    transparent: true,
    opacity: 0.12,
    side: THREE.BackSide,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  })
);
scene.add(atmosphere);

/* atmosphere outer halo */
const halo = new THREE.Mesh(
  new THREE.SphereGeometry(EARTH_RADIUS * 1.25, 64, 64),
  new THREE.MeshBasicMaterial({
    color: 0x38bdf8,
    transparent: true,
    opacity: 0.05,
    side: THREE.BackSide,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  })
);
scene.add(halo);

/* Sun.  The real Sun sits ~1 AU away (~390 lunar-orbit radii), which would put
   it far outside the lunar scene, so the marker is placed on a compressed
   radius along the true Earth->Sun direction; the label spells that out.  The
   directional light follows the same vector, so Earth's phase matches. */
const SUN_MARKER_DISTANCE = 1.45; // multiples of the trajectory radius
let sunDirections = null; // per-sample unit vectors from the bundle
let sunRangeKm = null; // per-sample Earth-Sun range
let sunAvailable = false;
let sunDirection = null;

function radialSpriteTexture() {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 128;
  const ctx = canvas.getContext("2d");
  const gradient = ctx.createRadialGradient(64, 64, 0, 64, 64, 64);
  gradient.addColorStop(0, "rgba(255,255,255,1)");
  gradient.addColorStop(0.22, "rgba(255,236,179,0.9)");
  gradient.addColorStop(0.55, "rgba(255,186,90,0.35)");
  gradient.addColorStop(1, "rgba(255,150,50,0)");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, 128, 128);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

const labelTextureCache = new Map();
function bodyLabelTexture(title, subtitle = "", color = "255,240,200") {
  const key = `${title}|${subtitle}|${color}`;
  if (labelTextureCache.has(key)) return labelTextureCache.get(key);
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 128;
  const ctx = canvas.getContext("2d");
  ctx.textAlign = "center";
  ctx.fillStyle = `rgba(${color},0.95)`;
  ctx.font = subtitle
    ? "600 44px system-ui, -apple-system, sans-serif"
    : "600 48px system-ui, -apple-system, sans-serif";
  ctx.fillText(title, 128, subtitle ? 54 : 68);
  if (subtitle) {
    ctx.fillStyle = "rgba(253,230,138,0.7)";
    ctx.font = "400 24px system-ui, -apple-system, sans-serif";
    ctx.fillText(subtitle, 128, 92);
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  labelTextureCache.set(key, texture);
  return texture;
}

function labelSprite(title, subtitle, color) {
  return new THREE.Sprite(
    new THREE.SpriteMaterial({
      map: bodyLabelTexture(title, subtitle, color),
      transparent: true,
      depthWrite: false,
      fog: false,
    })
  );
}

const sunGroup = new THREE.Group();
sunGroup.visible = false;
scene.add(sunGroup);

const sunBody = new THREE.Mesh(
  new THREE.SphereGeometry(1, 32, 32),
  new THREE.MeshBasicMaterial({ color: 0xfff3c4, fog: false })
);
sunGroup.add(sunBody);

const sunGlow = new THREE.Sprite(
  new THREE.SpriteMaterial({
    map: radialSpriteTexture(),
    color: 0xffcf6b,
    transparent: true,
    opacity: 0.9,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    fog: false,
  })
);
sunGroup.add(sunGlow);

const sunLabel = labelSprite("SUN", "not to scale");
sunGroup.add(sunLabel);

const sunBeam = new THREE.Line(
  new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
  new THREE.LineDashedMaterial({
    color: 0xffd27a,
    transparent: true,
    opacity: 0.3,
    dashSize: 1,
    gapSize: 1,
    fog: false,
  })
);
sunBeam.visible = false;
scene.add(sunBeam);

/* Earth and Moon markers.  At lunar-orbit scale the true bodies are only a
   few pixels wide (Earth is ~1.7% of the orbit radius), so the display radii
   are enlarged the same way scripts/render_guide.py does; the viewport
   caption says so. */
const EARTH_DISPLAY = 0.045; // fraction of the trajectory radius
const MOON_DISPLAY = 0.022;

const earthLabel = labelSprite("EARTH", "", "191,219,254");
earthLabel.visible = false;
scene.add(earthLabel);

const moonGroup = new THREE.Group();
moonGroup.visible = false;
scene.add(moonGroup);

const moonBody = new THREE.Mesh(
  new THREE.SphereGeometry(1, 48, 48),
  new THREE.MeshStandardMaterial({
    color: 0xb7bcc4,
    roughness: 0.95,
    metalness: 0.05,
    emissive: 0x2a2f3a,
    emissiveIntensity: 0.45,
  })
);
moonGroup.add(moonBody);

const moonGlow = new THREE.Sprite(
  new THREE.SpriteMaterial({
    map: radialSpriteTexture(),
    color: 0x9fb4d6,
    transparent: true,
    opacity: 0.28,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    fog: false,
  })
);
moonGroup.add(moonGlow);

const moonLabel = labelSprite("MOON", "", "226,232,240");
moonGroup.add(moonLabel);

/* Static starfield: the guide renderer paints stars, and they make the
   lunar-scale scene read as space rather than a CAD viewport. */
function makeStarField() {
  const count = 1600;
  const positions = new Float32Array(count * 3);
  let state = 0x2f6e2b1;
  const rand = () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
  for (let i = 0; i < count; i++) {
    const z = rand() * 2 - 1;
    const theta = rand() * Math.PI * 2;
    const r = Math.sqrt(Math.max(0, 1 - z * z));
    positions[i * 3] = r * Math.cos(theta);
    positions[i * 3 + 1] = r * Math.sin(theta);
    positions[i * 3 + 2] = z;
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  const material = new THREE.PointsMaterial({
    color: 0xcfe0ff,
    size: 1.6,
    sizeAttenuation: false,
    transparent: true,
    opacity: 0.75,
    depthWrite: false,
    fog: false,
  });
  return new THREE.Points(geometry, material);
}
const starGroup = makeStarField();
starGroup.visible = false;
scene.add(starGroup);

/* grid + axes */
const grid = new THREE.GridHelper(40000, 40, 0x1f2a4a, 0x121a33);
grid.material.opacity = 0.5;
grid.material.transparent = true;
scene.add(grid);

const axes = new THREE.AxesHelper(12000);
scene.add(axes);

/* trajectory container */
const trajGroup = new THREE.Group();
scene.add(trajGroup);

/* cosmos pose container */
const cosmosGroup = new THREE.Group();
cosmosGroup.visible = false;
scene.add(cosmosGroup);

/* controls */
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 2000;
controls.maxDistance = 80000;
controls.autoRotate = autoRotateCb.checked;
controls.autoRotateSpeed = 0.6;
controls.target.set(0, 0, 0);

/* The bundle is in kilometres and the arc spans ~384,400 km, so every
   fixed-size prop in the scene has to scale with the loaded trajectory. */
function applySceneScale(radius) {
  const R = Math.max(radius || 1, 1);
  const s = R / 20000;
  grid.scale.setScalar(s);
  axes.scale.setScalar(s);
  if (scene.fog) scene.fog.density = 0.25 / R;
  // GridHelper is built in the XZ plane; rotate it onto the orbital plane.
  grid.rotation.set(0, 0, 0);
  if (orbitPlane && orbitPlane.a === 0 && orbitPlane.b === 1) {
    grid.rotation.x = Math.PI / 2;
  } else if (orbitPlane && orbitPlane.a === 1 && orbitPlane.b === 2) {
    grid.rotation.z = Math.PI / 2;
  }
}
function fitCameraToTrajectory(radius) {
  const R = Math.max(radius || trajRadius || 20000, 1);
  trajRadius = R;
  camera.up.set(0, 1, 0);
  camera.near = Math.max(0.5, R * 0.00002);
  camera.far = Math.max(camera.far, R * 20);
  camera.updateProjectionMatrix();
  controls.minDistance = R * 0.02;
  controls.maxDistance = R * 20;
  camera.position.set(R * 0.32, R * 1.0, R * 2.15);
  controls.target.set(0, 0, 0);
  controls.update();
  applySceneScale(R);
  updateSunGeometry(R);
  updateBodyGeometry(R);
}
function fitCameraToCosmos() {
  // Poses were already rescaled into the scene by buildCosmosScene().
  let R = EARTH_RADIUS * 1.5;
  for (const p of cosmos.poses || []) {
    const v = p.position;
    if (v) R = Math.max(R, Math.hypot(v.x, v.y, v.z));
  }
  camera.near = Math.max(0.5, R * 0.00002);
  camera.far = Math.max(camera.far, R * 20);
  camera.updateProjectionMatrix();
  controls.minDistance = R * 0.05;
  controls.maxDistance = R * 20;
  camera.position.set(R * 0.35, R * 1.0, R * 2.0);
  controls.target.set(0, 0, 0);
  controls.update();
}

/* Frame the arc straight down the orbit normal — the 2D framing used by
   scripts/render_guide.py.  ``orbitPlane.c`` is the least-spanning axis, so
   looking along it shows the full ellipse instead of a foreshortened edge.
   ``camera.up`` has to avoid the view axis or lookAt() degenerates. */
function fitCameraTopDown() {
  const R = Math.max(trajRadius || 20000, 1);
  const c = orbitPlane?.c ?? 2;
  const normal = new THREE.Vector3(0, 0, 0);
  normal.setComponent(c, 1);
  const up = new THREE.Vector3(0, 0, 0);
  up.setComponent(c === 1 ? 2 : 1, 1);
  camera.up.copy(up);
  camera.near = Math.max(0.5, R * 0.00002);
  camera.far = Math.max(camera.far, R * 20);
  camera.updateProjectionMatrix();
  controls.minDistance = R * 0.02;
  controls.maxDistance = R * 20;
  // Half the arc's vertical extent is ~R; tan(25 deg) for the 50 deg fov.
  camera.position.copy(normal).multiplyScalar(R * 2.35);
  controls.target.set(0, 0, 0);
  controls.update();
}

/* Re-lay the Sun marker for the current scene scale.  The bundle stores unit
   Earth->Sun directions (one per trajectory sample), so the marker swings
   along the true ~28 deg solar motion over a 30-day arc as playback advances. */
function updateSunGeometry(radius) {
  const R = Math.max(radius || 1, 1);
  sunBody.scale.setScalar(R * 0.02);
  sunGlow.scale.setScalar(R * 0.16);
  sunLabel.scale.set(R * 0.22, R * 0.11, 1);
  sunLabel.position.set(0, R * 0.115, 0);
  if (!sunDirection) return;
  sunGroup.position.copy(sunDirection).multiplyScalar(R * SUN_MARKER_DISTANCE);
  sunLight.position.copy(sunDirection).multiplyScalar(R * 3);
  sunBeam.geometry.setFromPoints([new THREE.Vector3(), sunGroup.position.clone()]);
  sunBeam.material.dashSize = R * 0.008;
  sunBeam.material.gapSize = R * 0.008;
  sunBeam.computeLineDistances();
}

/* Re-lay the body markers for the current scene scale.  The radii are display
   sizes (fractions of the trajectory radius), not physical radii, so Earth and
   the Moon stay visible on a lunar-orbit view. */
function updateBodyGeometry(radius) {
  const R = Math.max(radius || 1, 1);
  const earthScale = (R * EARTH_DISPLAY) / EARTH_RADIUS;
  earth.scale.setScalar(earthScale);
  atmosphere.scale.setScalar(earthScale);
  halo.scale.setScalar(earthScale);
  earthLabel.scale.set(R * 0.16, R * 0.08, 1);
  earthLabel.position.set(0, R * 0.085, 0);

  moonBody.scale.setScalar(R * MOON_DISPLAY);
  moonGlow.scale.setScalar(R * MOON_DISPLAY * 3.2);
  moonLabel.scale.set(R * 0.13, R * 0.065, 1);
  moonLabel.position.set(0, R * (MOON_DISPLAY * 1.6 + 0.03), 0);

  starGroup.scale.setScalar(R * 8);
}

function applySunSample(index) {
  if (!sunAvailable || !Array.isArray(sunDirections) || !sunDirections.length) return;
  const i = Math.min(Math.max(index | 0, 0), sunDirections.length - 1);
  const d = sunDirections[i];
  if (!Array.isArray(d) || d.length < 3) return;
  const v = new THREE.Vector3(d[0], d[1], d[2]);
  if (!Number.isFinite(v.x) || !Number.isFinite(v.y) || !Number.isFinite(v.z)) return;
  if (v.lengthSq() < 1e-12) return;
  sunDirection = v.normalize();
  updateSunGeometry(trajRadius);
}

function setOrbitPropsVisible() {
  const orbit = mode === "orbit";
  sunGroup.visible = orbit && !!showSunCb?.checked && sunAvailable;
  sunBeam.visible = sunGroup.visible;
  starGroup.visible = orbit;
  earthLabel.visible = orbit;
  moonGroup.visible = orbit && !!trajectoryData?.points?.length;
}

/* ---------------- helpers ---------------- */
function showError(msg) {
  errorBanner.textContent = msg;
  errorBanner.hidden = false;
  console.error(msg);
}
function clearError() { errorBanner.hidden = true; errorBanner.textContent = ""; }

function setLoading(on) {
  loadingOverlay.classList.toggle("hidden", !on);
}

function setDot(el, state) {
  el.classList.remove("ok", "err", "loading");
  if (state) el.classList.add(state);
}

function fmt(n, digits = 4) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  if (Math.abs(n) < 0.001 || Math.abs(n) >= 10000) return n.toExponential(3);
  return Number(n).toFixed(digits);
}

function buildMetricCard(label, value, opts = {}) {
  const div = document.createElement("div");
  div.className = "metric-card" + (opts.accent ? " accent" : "");
  div.innerHTML = `<div class="label">${label}</div><div class="value">${value}</div>`;
  return div;
}

function buildStripCard(label, value, cls = "") {
  const div = document.createElement("div");
  div.className = "strip-card" + (cls ? " " + cls : "");
  div.innerHTML = `<div class="label">${label}</div><div class="value">${value}</div>`;
  return div;
}
function r2Class(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return "";
  if (n >= 0.9) return "good";
  if (n >= 0.5) return "mid";
  if (n >= 0) return "low";
  return "bad";
}

function buildFallbackNote(msg) {
  const div = document.createElement("div");
  div.className = "fallback-note";
  div.dataset.role = "data-fallback";
  /* allow placement inside the 2-column metrics grid */
  div.style.gridColumn = "1 / -1";
  div.textContent = msg;
  return div;
}

/* Show a symbolic string for the equation. Prefer the SINDy expression from
   the PINN result; nothing else is needed for the panel. */

/* ---------------- resize ---------------- */
function resize() {
  const w = viewport.clientWidth || 800;
  const h = viewport.clientHeight || 600;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
}
window.addEventListener("resize", resize);

/* ---------------- data loading ---------------- */
let trajectoryLoaded = false;
let pinnLoaded = false;
let trajectoryFailed = false;
let pinnFailed = false;
let settledCount = 0;
function settle() {
  settledCount++;
  if (settledCount >= 2) setLoading(false);
}

function fileError(prefix, path, error) {
  const detail = error && error.message ? error.message : String(error);
  if (/JSON/i.test(detail)) {
    return `${prefix}: ${path} is not valid JSON`;
  }
  if (/HTTP 404/.test(detail)) {
    return `${prefix}: ${path} not found`;
  }
  return `${prefix}: ${path} (${detail})`;
}

async function loadTrajectory() {
  const src = settings.trajectoryPath || "./data/trajectory.json";
  trajectoryLoaded = false;
  const card = document.getElementById("trajectory-card");
  card?.querySelectorAll(".fallback-note").forEach((n) => n.remove());
  stripMetrics.querySelector('[data-strip="traj-samples"]')?.remove();
  // clear previous geometry (fixes duplicate meshes on reload)
  while (trajGroup.children.length) {
    const c = trajGroup.children[0];
    trajGroup.remove(c);
    if (c.geometry) c.geometry.dispose();
    if (c.material) (Array.isArray(c.material) ? c.material : [c.material]).forEach((m) => m?.dispose?.());
  }
  ensemblePoints = null;
  traveledLine = null;
  traveledTubeRef = null;
  ensembleEpoch = -1;
  moonGroup.visible = false;
  sunAvailable = false;
  setOrbitPropsVisible();
  setDot(trajDot, "loading");
  try {
    const response = await fetch(src);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    const rawPoints = data.points || [];
    const points = rawPoints.map((p) => new THREE.Vector3(p[0], p[1], p[2]));
    trajectoryData = data;
    if (points.length < 2) throw new Error("Not enough trajectory samples");

    // Sun direction track (optional; bundles built before it existed omit it).
    sunDirections =
      Array.isArray(data.sun_directions) && data.sun_directions.length
        ? data.sun_directions
        : null;
    sunRangeKm = Array.isArray(data.sun_range_km) ? data.sun_range_km : null;
    sunAvailable = !!sunDirections;
    if (showSunCb) showSunCb.disabled = !sunAvailable;
    if (sunAvailable) applySunSample(0);
    else sunDirection = null;
    setOrbitPropsVisible();

    // glowing tube along curve
    let maxR = 0, minR = Infinity;
    const lo = [Infinity, Infinity, Infinity];
    const hi = [-Infinity, -Infinity, -Infinity];
    for (const p of points) {
      const v = [p.x, p.y, p.z];
      const r = Math.hypot(p.x, p.y, p.z);
      maxR = Math.max(maxR, r);
      minR = Math.min(minR, r);
      for (let k = 0; k < 3; k++) {
        lo[k] = Math.min(lo[k], v[k]);
        hi[k] = Math.max(hi[k], v[k]);
      }
    }
    const spans = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
    const order = [0, 1, 2].sort((a, b) => spans[b] - spans[a]);
    const planar = spans[order[2]] <= Math.max(maxR, 1) * 0.02;
    orbitPlane = {
      a: order[0],
      b: order[1],
      c: order[2],
      name: `${"XYZ"[order[0]]}–${"XYZ"[order[1]]}`,
      planar,
    };
    const closedLoop =
      points.length > 2 && points[0].distanceTo(points[points.length - 1]) <= maxR * 0.01;
    const curvePoints = closedLoop ? points.slice(0, -1) : points;
    const curve = new THREE.CatmullRomCurve3(curvePoints, closedLoop, "centripetal");
    const tubeRadius = Math.max(20, maxR * 0.0035);
    const tubularSegments = Math.max(points.length * 2, 64);
    // Planned path: thin and faint, like the ghosted full path in
    // scripts/render_guide.py.
    const tube = new THREE.TubeGeometry(curve, tubularSegments, tubeRadius * 0.7, 12, false);
    const tubeMat = new THREE.MeshStandardMaterial({
      color: 0xf59e0b,
      emissive: 0xf59e0b,
      emissiveIntensity: 0.6,
      roughness: 0.3,
      metalness: 0.1,
      transparent: true,
      opacity: 0.4,
    });
    const tubeMesh = new THREE.Mesh(tube, tubeMat);
    trajGroup.add(tubeMesh);

    // Traveled portion: the same curve with a fatter, brighter tube that grows
    // as playback advances.  TubeGeometry emits its indices in path order
    // (radialSegments*6 per tubular segment), so the trail is just a capped
    // draw range over a second tube.
    const traveledTube = new THREE.Mesh(
      new THREE.TubeGeometry(curve, tubularSegments, tubeRadius * 1.5, 12, false),
      new THREE.MeshStandardMaterial({
        color: 0xffd27a,
        emissive: 0xffb733,
        emissiveIntensity: 1.4,
        roughness: 0.35,
        metalness: 0.1,
        transparent: true,
        opacity: 0.95,
      })
    );
    traveledTube.userData.indexPerSegment = 12 * 6;
    traveledTube.userData.tubularSegments = tubularSegments;
    traveledTube.geometry.setDrawRange(0, 12 * 6);
    trajGroup.add(traveledTube);

    const fullLine = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(points),
      new THREE.LineBasicMaterial({ color: 0xfde68a, transparent: true, opacity: 0.22 })
    );
    trajGroup.add(fullLine);

    traveledLine = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(points),
      new THREE.LineBasicMaterial({ color: 0xfff7d6, transparent: true, opacity: 1 })
    );
    traveledLine.geometry.setDrawRange(0, 1);
    trajGroup.add(traveledLine);
    traveledTubeRef = traveledTube;

    // overlays (optional)
    const overlaySpecs = [
      { key: "points_propagated", color: 0x22d3ee, name: "prop" },
      { key: "points_corrected", color: 0xf472b6, name: "corr" },
    ];
    for (const spec of overlaySpecs) {
      const arr = data[spec.key];
      if (!Array.isArray(arr) || arr.length < 2) continue;
      const pts = arr.map((p) => new THREE.Vector3(p[0], p[1], p[2]));
      const geo = new THREE.BufferGeometry().setFromPoints(pts);
      const mat = new THREE.LineBasicMaterial({ color: spec.color, transparent: true, opacity: 0.85 });
      const line = new THREE.Line(geo, mat);
      line.name = `overlay-${spec.name}`;
      line.visible = spec.name === "prop" ? (showPropCb?.checked ?? false) : (showCorrCb?.checked ?? false);
      trajGroup.add(line);
    }

    // Moon marker at the current epoch.
    moonGroup.position.copy(points[0]);

    // slider range
    if (orbitSlider) { orbitSlider.max = String(Math.max(points.length - 1, 1)); orbitSlider.value = "0"; }
    if (orbitTimeEl) {
      const t0 = Array.isArray(data.times) ? data.times[0] : 0;
      orbitTimeEl.textContent = `t = ${Number(t0).toFixed(2)} s`;
    }

    // metrics
    trajKv.innerHTML = "";
    const kvPairs = [
      ["Samples", points.length],
      ["Max radius", fmt(maxR, 1)],
      ["Min radius", fmt(minR, 1)],
      [
        "Plane",
        planar ? `${orbitPlane.name} (${"XYZ"[order[2]]} ≈ 0)` : "3D",
      ],
      ["Loop", closedLoop ? "Closed" : "Open"],
    ];
    const sunRanges = (sunRangeKm || []).filter((v) => Number.isFinite(v) && v > 0);
    if (sunRanges.length) {
      const AU = 149597870.7;
      const lo = Math.min(...sunRanges) / AU;
      const hi = Math.max(...sunRanges) / AU;
      kvPairs.push([
        "Sun range",
        lo === hi ? `${lo.toFixed(3)} AU` : `${lo.toFixed(3)}–${hi.toFixed(3)} AU`,
      ]);
    }
    for (const [k, v] of kvPairs) {
      const dt = document.createElement("dt"); dt.textContent = k;
      const dd = document.createElement("dd"); dd.textContent = v;
      trajKv.appendChild(dt); trajKv.appendChild(dd);
    }

    trajStatus.textContent = `${points.length} samples loaded`;
    setDot(trajDot, "ok");

    // strip metric
    const stripCard = buildStripCard("Trajectory samples", points.length, "amber");
    stripCard.dataset.strip = "traj-samples";
    stripMetrics.prepend(stripCard);

    trajectoryLoaded = true;
    fitCameraToTrajectory(maxR);
    sceneStatus.textContent = (trajectoryLoaded && pinnLoaded) ? "Scene ready" : sceneStatus.textContent;
    setOrbitIndex(0);
    settle();
  } catch (error) {
    const message = fileError("Trajectory data unavailable", src, error);
    trajStatus.textContent = `Trajectory data unavailable — see note below`;
    setDot(trajDot, "err");
    trajKv.innerHTML = "";
    trajStatus.after(buildFallbackNote(`${message}. The 3D path will not render.`));
    trajectoryFailed = true;
    showError(message);
    settle();
  }
}

async function loadPinnResult() {
  const src = settings.pinnPath || "./data/pinn_result.json";
  pinnLoaded = false;
  const card = document.getElementById("pinn-card");
  card?.querySelectorAll(".fallback-note").forEach((n) => n.remove());
  stripMetrics.querySelectorAll('[data-strip="pinn"]').forEach((n) => n.remove());
  setDot(pinnDot, "loading");
  try {
    const response = await fetch(src);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();

    const target = (data.target || "residual").toString();
    const pinn = data.pinn || {};
    const sindy = data.sindy || {};
    const mlp = data.mlp || {};
    const linreg = data.linreg || {};
    const sindyExpr = sindy.expr || sindy.expression || null;
    eqSindyLabel.textContent = "SINDy discovered";
    eqTarget.textContent = target;
    eqSindy.textContent = sindyExpr || "—";

    pinnRaw.textContent = JSON.stringify(data, null, 2);

    // metrics cards
    pinnMetrics.innerHTML = "";
    const cards = [
      ["PINN train R²", fmt(pinn.train_r2, 4), true],
      ["PINN train MSE", fmt(pinn.train_mse, 5), false],
      ["PINN test R²", fmt(pinn.test_r2, 4), false],
      ["SINDy train R²", fmt(sindy.train_r2, 4), true],
      ["SINDy train MSE", fmt(sindy.train_mse, 5), false],
      ["MLP train R²", fmt(mlp.train_r2, 4), false],
      ["MLP train MSE", fmt(mlp.train_mse, 5), false],
      ["LinearReg train R²", fmt(linreg.train_r2, 4), false],
      ["LinearReg train MSE", fmt(linreg.train_mse, 5), false],
      ["Samples", data.n_samples ?? "—", false],
    ];
    for (const [label, value, accent] of cards) {
      pinnMetrics.appendChild(buildMetricCard(label, value, { accent }));
    }

    pinnStatus.textContent = "Loaded";
    setDot(pinnDot, "ok");

    // strip metrics
    [
      buildStripCard("PINN R²", fmt(pinn.train_r2, 4), r2Class(pinn.train_r2)),
      buildStripCard("SINDy R²", fmt(sindy.train_r2, 4), r2Class(sindy.train_r2)),
      buildStripCard("MLP R²", fmt(mlp.train_r2, 4), r2Class(mlp.train_r2)),
      buildStripCard("LinearReg R²", fmt(linreg.train_r2, 4), r2Class(linreg.train_r2)),
      buildStripCard("Samples", data.n_samples ?? "—", ""),
    ].forEach((c) => {
      c.dataset.strip = "pinn";
      stripMetrics.appendChild(c);
    });

    pinnLoaded = true;
    sceneStatus.textContent = (trajectoryLoaded && pinnLoaded) ? "Scene ready" : sceneStatus.textContent;
    // residual metrics table
    if (residualMetrics) {
      residualMetrics.innerHTML = "";
      const models = [
        ["PINN", data.pinn], ["MLP", data.mlp], ["SINDy", data.sindy], ["LinReg", data.linreg],
      ];
      for (const [name, m] of models) {
        if (!m) continue;
        const card = buildMetricCard(`${name} test R²`, fmt(m.test_r2 ?? m.train_r2, 4), { accent: name === "PINN" });
        card.classList.add(r2Class(m.test_r2 ?? m.train_r2));
        residualMetrics.appendChild(card);
      }
      const sc = buildMetricCard("Samples", data.n_samples ?? "—", {});
      residualMetrics.appendChild(sc);
    }
    settle();
  } catch (error) {
    const message = fileError("PINN result unavailable", src, error);
    pinnStatus.textContent = "PINN result unavailable — see note below";
    setDot(pinnDot, "err");
    pinnMetrics.innerHTML = "";
    eqSindyLabel.textContent = "SINDy discovered";
    eqTarget.textContent = "—";
    eqSindy.textContent = "—";
    pinnMetrics.appendChild(buildFallbackNote(`${message}. Equations and metrics will stay empty.`));
    pinnFailed = true;
    showError(message);
    settle();
  }
}

async function loadEnsemble() {
  try {
    const res = await fetch("./data/ensemble.json");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    ensembleData = await res.json();
    if (ensembleSlider && ensembleData.n_epochs) {
      ensembleSlider.max = String(ensembleData.n_epochs - 1);
    }
    const idx = parseInt(orbitSlider?.value || "0", 10) || 0;
    const epoch = ensembleEpochForOrbitIndex(idx);
    if (ensembleSlider) ensembleSlider.value = String(epoch);
    renderEnsembleEpoch(epoch, { force: true });
  } catch (e) {
    ensembleData = null;
  }
}

/* The ensemble artifact covers 61 epochs over 10 hours, while the reference
   arc covers 30 days.  They cannot share wall-clock time, so the cloud is
   driven by trajectory *progress*: as the main arc runs from sample 0 to the
   last sample, the cloud runs from epoch 0 to the final epoch.  That keeps the
   two animations locked together for the whole playback loop. */
function ensembleEpochCount() {
  const n = ensembleData?.clouds?.length ?? ensembleData?.n_epochs ?? 0;
  return Math.max(0, n - 1);
}

function ensembleEpochForOrbitIndex(idx) {
  const last = ensembleEpochCount();
  const n = trajectoryData?.points?.length ?? 0;
  if (!last || n < 2) return 0;
  const progress = Math.max(0, Math.min(1, idx / (n - 1)));
  return Math.round(progress * last);
}

function orbitIndexForEnsembleEpoch(epoch) {
  const last = ensembleEpochCount();
  const n = trajectoryData?.points?.length ?? 0;
  if (!last || n < 2) return 0;
  const progress = Math.max(0, Math.min(1, epoch / last));
  return Math.round(progress * (n - 1));
}

function syncEnsembleToOrbit(idx) {
  if (!ensembleData) return;
  const epoch = ensembleEpochForOrbitIndex(idx);
  if (ensembleSlider) ensembleSlider.value = String(epoch);
  renderEnsembleEpoch(epoch);
}

function ensembleRmsKm(epoch) {
  const cloud = ensembleData?.clouds?.[epoch];
  const nominal = ensembleData?.nominal?.[epoch];
  if (Array.isArray(cloud) && Array.isArray(nominal) && cloud.length) {
    let sum = 0;
    for (let i = 0; i < cloud.length; i++) {
      const dx = cloud[i][0] - nominal[0];
      const dy = cloud[i][1] - nominal[1];
      const dz = cloud[i][2] - nominal[2];
      sum += dx * dx + dy * dy + dz * dz;
    }
    return Math.sqrt(sum / (cloud.length * 3)) / 1000;
  }
  // stats[i] describes the cloud after step i, i.e. epoch i+1.
  const st = epoch > 0 ? ensembleData?.stats?.[epoch - 1] : null;
  return st ? st.rms_m / 1000 : 0;
}

function renderEnsembleEpoch(idx, { force = false } = {}) {
  if (!ensembleData) return;
  const last = ensembleEpochCount();
  idx = Math.max(0, Math.min(last, idx | 0));
  const t = ensembleData.epochs_s?.[idx] ?? idx;
  if (ensembleInfo) {
    ensembleInfo.textContent = `epoch ${idx}/${last} · t=${Number(t).toFixed(0)}s · rms=${ensembleRmsKm(idx).toFixed(1)} km`;
  }
  if (!force && idx === ensembleEpoch) return;
  ensembleEpoch = idx;
  if (ensemblePoints) {
    trajGroup.remove(ensemblePoints);
    ensemblePoints.geometry?.dispose();
    ensemblePoints.material?.dispose();
    ensemblePoints = null;
  }
  if (!showEnsembleCb?.checked) return;
  const cloud = ensembleData.clouds?.[idx];
  if (!Array.isArray(cloud)) return;
  const positions = new Float32Array(cloud.length * 3);
  for (let i = 0; i < cloud.length; i++) {
    positions[i*3] = cloud[i][0] / 1000;
    positions[i*3+1] = cloud[i][1] / 1000;
    positions[i*3+2] = cloud[i][2] / 1000;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  const mat = new THREE.PointsMaterial({
    color: 0xa78bfa,
    size: 2.5,
    sizeAttenuation: false,
    transparent: true,
    opacity: 0.7,
  });
  ensemblePoints = new THREE.Points(geo, mat);
  trajGroup.add(ensemblePoints);
}

async function loadPredictions() {
  const src = settings.predictionsPath || "./data/pinn_predictions.json";
  setDot(residualDot, "loading");
  try {
    const res = await fetch(src);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    predictionsData = await res.json();
    residualTargetCols = Array.isArray(predictionsData.target_columns) ? predictionsData.target_columns : [];
    residualStatus.textContent = `Loaded ${predictionsData.rows?.length ?? 0} predictions`;
    setDot(residualDot, "ok");
    drawResidual();
  } catch (e) {
    predictionsData = null;
    residualStatus.textContent = `Predictions unavailable — ${e.message}`;
    setDot(residualDot, "err");
  }
}

async function loadManifest() {
  const src = settings.manifestPath || "./data/manifest.json";
  try {
    const res = await fetch(src);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    manifestData = await res.json();
  } catch (e) {
    manifestData = null;
  }
  updateProvenance();
}

async function loadUiManifest() {
  try {
    const res = await fetch("./data/ui_manifest.json");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    uiManifestData = await res.json();
  } catch (e) {
    uiManifestData = null;
  }
  updateProvenance();
}

function updateProvenance() {
  if (!provenanceEl) return;
  const parts = [];
  if (manifestData) {
    parts.push(manifestData.dataset || "dataset");
    if (manifestData.source) parts.push(manifestData.source.provider || "");
    if (manifestData.source?.target) parts.push(`target ${manifestData.source.target}`);
    if (manifestData.generated_utc) parts.push(`generated ${manifestData.generated_utc}`);
    if (manifestData.source?.retrieved_utc) parts.push(`retrieved ${manifestData.source.retrieved_utc}`);
  }
  if (uiManifestData?.generated_utc) parts.push(`UI bundle ${uiManifestData.generated_utc}`);
  if (manifestData?.generated_utc && uiManifestData?.generated_utc && uiManifestData.generated_utc < manifestData.generated_utc) {
    parts.push("⚠ bundle may be stale");
  }
  provenanceEl.textContent = parts.filter(Boolean).join(" · ") || "No provenance available";
}

function drawResidual() {
  if (!residualCanvas || !predictionsData?.rows?.length) return;
  const ctx = residualCanvas.getContext("2d");
  const W = residualCanvas.width, H = residualCanvas.height;
  ctx.clearRect(0, 0, W, H);
  const rows = predictionsData.rows;
  const ts = rows.map((r) => r.t);
  const tMin = Math.min(...ts), tMax = Math.max(...ts);
  const cols = residualTargetCols.length ? residualTargetCols : Object.keys(rows[0].actual || {});
  let yMin = Infinity, yMax = -Infinity;
  for (const r of rows) {
    for (const c of cols) {
      const a = r.actual?.[c], p = r.pinn?.[c], m = r.mlp?.[c];
      for (const v of [a, p, m]) if (Number.isFinite(v)) { yMin = Math.min(yMin, v); yMax = Math.max(yMax, v); }
    }
  }
  if (!Number.isFinite(yMin)) return;
  const pad = (yMax - yMin) * 0.08 || 1;
  yMin -= pad; yMax += pad;
  const xOf = (t) => 30 + (t - tMin) / (tMax - tMin || 1) * (W - 40);
  const yOf = (v) => H - 20 - (v - yMin) / (yMax - yMin || 1) * (H - 30);
  ctx.strokeStyle = "rgba(255,255,255,0.15)";
  ctx.beginPath(); ctx.moveTo(30, 10); ctx.lineTo(30, H-20); ctx.lineTo(W-10, H-20); ctx.stroke();
  const showActual = document.getElementById("show-actual")?.checked ?? true;
  const showPinn = document.getElementById("show-pinn")?.checked ?? true;
  const showMlp = document.getElementById("show-mlp")?.checked ?? false;
  const drawSeries = (key, color) => {
    ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.beginPath();
    let first = true;
    for (const r of rows) {
      const v = r[key]?.[cols[0]];
      if (!Number.isFinite(v)) continue;
      const x = xOf(r.t), y = yOf(v);
      if (first) { ctx.moveTo(x, y); first = false; } else ctx.lineTo(x, y);
    }
    ctx.stroke();
  };
  if (showActual) drawSeries("actual", "#f59e0b");
  if (showPinn) drawSeries("pinn", "#22d3ee");
  if (showMlp) drawSeries("mlp", "#f472b6");
  ctx.font = "10px sans-serif";
  let lx = 34;
  const legend = [];
  if (showActual) legend.push(["Actual", "#f59e0b"]);
  if (showPinn) legend.push(["PINN", "#22d3ee"]);
  if (showMlp) legend.push(["MLP", "#f472b6"]);
  for (const [lab, col] of legend) {
    ctx.fillStyle = col; ctx.fillRect(lx, 4, 10, 10);
    ctx.fillStyle = "#cbd5e1"; ctx.fillText(lab, lx + 14, 13);
    lx += 70;
  }
  residualChart = { rows, xOf, yOf, tMin, tMax };
}

function drawGuide(idx) {
  if (!guideCanvas) return;
  const ctx = guideCanvas.getContext("2d");
  const W = guideCanvas.width, H = guideCanvas.height;
  ctx.fillStyle = "#050816"; ctx.fillRect(0, 0, W, H);
  if (!trajectoryData?.points?.length) return;
  const pts = trajectoryData.points;
  const i = Math.max(0, Math.min(pts.length - 1, idx ?? 0));
  // analytic moon position at this sample's time
  const t = Array.isArray(trajectoryData.times) ? Number(trajectoryData.times[i]) : i;
  const MU_E = 3.986004418e14, MU_M = 4.9048695e12, D = 3.844e8;
  const total = MU_E + MU_M;
  const theta = Math.sqrt(total / (D*D*D)) * t;
  const mf = MU_M / total;
  const c = Math.cos(theta), s = Math.sin(theta);
  // metre model -> kilometres, matching the bundle
  const moon3 = [D*(1-mf)*c/1000, D*(1-mf)*s/1000, 0];
  const earth3 = [-D*mf*c/1000, -D*mf*s/1000, 0];
  // pick the two axes that actually span the orbit (this dataset is XY, not XZ)
  const ax = orbitPlane?.a ?? 0;
  const bx = orbitPlane?.b ?? 2;
  const proj = (p) => [p[ax], p[bx]];
  // collect all points for scale
  const all = pts.map(proj);
  all.push(proj(moon3), proj(earth3));
  if (Array.isArray(trajectoryData.points_corrected)) {
    for (const p of trajectoryData.points_corrected) all.push(proj(p));
  }
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const p of all) { minX = Math.min(minX, p[0]); maxX = Math.max(maxX, p[0]); minY = Math.min(minY, p[1]); maxY = Math.max(maxY, p[1]); }
  const cx = (minX+maxX)/2, cy = (minY+maxY)/2;
  const scale = 0.85 * Math.min(W/(maxX-minX || 1), H/(maxY-minY || 1));
  const px = (p) => [W/2 + (p[0]-cx)*scale, H/2 - (p[1]-cy)*scale];
  // stars
  ctx.fillStyle = "rgba(200,200,200,0.6)";
  for (let k = 0; k < 80; k++) {
    const sx = (k*37) % W, sy = (k*73) % H;
    ctx.fillRect(sx, sy, 1, 1);
  }
  // moon trail (none stored) skip; reference trail
  ctx.strokeStyle = "rgba(255,255,255,0.9)"; ctx.lineWidth = 1.5; ctx.beginPath();
  for (let k = 0; k <= i; k++) {
    const p = px(proj(pts[k]));
    if (k === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]);
  }
  ctx.stroke();
  // corrected trail
  const corr = trajectoryData.points_corrected;
  if (Array.isArray(corr) && corr.length) {
    ctx.strokeStyle = "rgba(244,114,182,0.9)"; ctx.lineWidth = 1.5; ctx.beginPath();
    for (let k = 0; k <= Math.min(i, corr.length-1); k++) {
      const p = px(proj(corr[k]));
      if (k === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]);
    }
    ctx.stroke();
  }
  // earth
  const ep = px(proj(earth3)); ctx.fillStyle = "#2563eb"; ctx.beginPath(); ctx.arc(ep[0], ep[1], 6, 0, Math.PI*2); ctx.fill();
  // moon
  const mp = px(proj(moon3)); ctx.fillStyle = "#9ca3af"; ctx.beginPath(); ctx.arc(mp[0], mp[1], 4, 0, Math.PI*2); ctx.fill();
  // sun bearing (the real Sun is ~1 AU out, so only the direction fits)
  if (sunAvailable && Array.isArray(sunDirections) && sunDirections.length) {
    const d3 = sunDirections[Math.min(i, sunDirections.length - 1)];
    const d2 = d3 ? proj(d3) : null;
    const dlen = d2 ? Math.hypot(d2[0], d2[1]) : 0;
    if (dlen > 0) {
      const ux = d2[0] / dlen;
      const uy = -d2[1] / dlen; // canvas y grows downwards
      const x0 = ep[0] + ux * 14;
      const y0 = ep[1] + uy * 14;
      const x1 = ep[0] + ux * 36;
      const y1 = ep[1] + uy * 36;
      ctx.strokeStyle = "rgba(253,230,138,0.75)";
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
      ctx.fillStyle = "#fde68a";
      ctx.beginPath(); ctx.arc(x1, y1, 3, 0, Math.PI*2); ctx.fill();
      ctx.fillStyle = "rgba(253,230,138,0.85)";
      ctx.font = "10px sans-serif";
      ctx.fillText("Sun", x1 + 5, y1 + 3);
    }
  }
  // spacecraft
  const sp = px(proj(pts[i])); ctx.fillStyle = "#f59e0b"; ctx.beginPath(); ctx.arc(sp[0], sp[1], 5, 0, Math.PI*2); ctx.fill();
  // labels
  ctx.fillStyle = "#94a3b8"; ctx.font = "10px sans-serif";
  ctx.fillText("Earth", ep[0]+8, ep[1]);
  ctx.fillText("Moon", mp[0]+6, mp[1]);
}

function maybeFinishLoading() {
  if (trajectoryLoaded && pinnLoaded) {
    setLoading(false);
    sceneStatus.textContent = "Scene ready";
  }
}

/* ================================================================
   Cosmos job viewer
   ================================================================ */
const COSMOS_ACCENT = 0x22d3ee;
const COSMOS_ACCENT_SOFT = 0x0e7490;
const cosmos = {
  job: null,
  outputs: null,
  manifest: null,
  loaded: false,
  frames: 0,
  fps: 24,
  times: [],
  poses: [],          // { position: Vector3(scene units), quat: Quaternion, posKm: Vector3 }
  current: 0,
  playing: false,
  playAcc: 0,
  lastTick: null,
  images: [],         // { frame, name, url } for the rendered-frame preview
  blobURLs: new Set(),
};

function m3FromColumns(c0, c1, c2) {
  // columns are camera axes (OpenCV: +Z forward, +X right, +Y down)
  return new THREE.Matrix4().set(
    c0.x, c1.x, c2.x, 0,
    c0.y, c1.y, c2.y, 0,
    c0.z, c1.z, c2.z, 0,
    0, 0, 0, 1
  );
}
function orthonormalize(c0, c1) {
  const x = c0.clone().normalize();
  let y = c1.clone().sub(x.clone().multiplyScalar(c1.dot(x))).normalize();
  let z = new THREE.Vector3().crossVectors(x, y).normalize();
  // re-orthogonalize against z to guard numerical drift
  y = new THREE.Vector3().crossVectors(z, x).normalize();
  return [x, y, z];
}

/* Integrate relative 9D actions from first_pose into absolute per-frame poses. */
function integratePoses(job, outputs) {
  const units = outputs.translation_unit || "cosmos_metre";
  const k = Number(outputs.scale_k_km_per_m || job.scale_k_km_per_m || 1000) || 1000;
  const kmPerUnit = units === "cosmos_metre" ? k : 1; // translation rows in cosmos metres -> km

  const fp = outputs.first_pose || job.first_pose || null;
  if (!fp || !fp.rotation || !fp.translation_km) {
    throw new Error("No first_pose in job/sample_outputs — cannot reconstruct poses");
  }
  const rows = outputs.poses || [];
  if (!Array.isArray(rows) || rows.length === 0) {
    throw new Error("sample_outputs.json has an empty poses array");
  }

  const c0 = new THREE.Vector3(fp.rotation[0][0], fp.rotation[1][0], fp.rotation[2][0]);
  const c1 = new THREE.Vector3(fp.rotation[0][1], fp.rotation[1][1], fp.rotation[2][1]);
  const [ax, ay, az] = orthonormalize(c0, c1);
  let R = m3FromColumns(ax, ay, az);
  let p = new THREE.Vector3(...fp.translation_km);

  const fps = Number(outputs.fps || job.fps || 24);
  const times = Array.isArray(job.frame_times_tdb_s) && job.frame_times_tdb_s.length
    ? job.frame_times_tdb_s.slice(0, rows.length + 1)
    : Array.from({ length: rows.length + 1 }, (_, i) => (job.window?.t0_tdb ?? 0) + i / fps);

  const out = [{ position: p.clone(), quat: new THREE.Quaternion().setFromRotationMatrix(R), posKm: p.clone() }];
  for (let i = 0; i < rows.length; i++) {
    const row = rows[i];
    if (!Array.isArray(row) || row.length !== 9) throw new Error(`Pose row ${i} is not 9 values`);
    const dT = new THREE.Vector3(row[0] * kmPerUnit, row[1] * kmPerUnit, row[2] * kmPerUnit);
    const relC0 = new THREE.Vector3(row[3], row[4], row[5]);
    const relC1 = new THREE.Vector3(row[6], row[7], row[8]);
    const [rx, ry, rz] = orthonormalize(relC0, relC1);
    const Rrel = m3FromColumns(rx, ry, rz);

    p = p.clone().add(dT.applyMatrix3(new THREE.Matrix3().setFromMatrix4(R)));
    R = new THREE.Matrix4().multiplyMatrices(R, Rrel);

    out.push({
      position: p.clone(),
      quat: new THREE.Quaternion().setFromRotationMatrix(R),
      posKm: p.clone(),
    });
  }
  return { poses: out, fps, times };
}

function buildFrustumGeometry(poses, scale) {
  // every frame gets a faint glyph; drawn as one LineSegments
  const pts = [];
  const f = scale * 0.35, s = scale * 0.18;
  for (const pose of poses) {
    const q = pose.quat;
    const x = new THREE.Vector3(1, 0, 0).applyQuaternion(q);
    const y = new THREE.Vector3(0, 1, 0).applyQuaternion(q);
    const z = new THREE.Vector3(0, 0, 1).applyQuaternion(q);
    const p = pose.position;
    const corner = (dx, dy) => new THREE.Vector3().copy(p)
      .add(z.clone().multiplyScalar(f)).add(x.clone().multiplyScalar(dx * s)).add(y.clone().multiplyScalar(dy * s));
    const c = [corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1)];
    // frustum edges
    c.forEach((cc) => pts.push(p.clone(), cc));
    // frame border
    for (let i = 0; i < 4; i++) pts.push(c[i], c[(i + 1) % 4]);
  }
  const geo = new THREE.BufferGeometry().setFromPoints(pts);
  const mat = new THREE.LineBasicMaterial({ color: COSMOS_ACCENT_SOFT, transparent: true, opacity: 0.28 });
  return new THREE.LineSegments(geo, mat);
}

function pathGeometryFor(poses) {
  return new THREE.BufferGeometry().setFromPoints(poses.map((pp) => pp.position));
}

function setCosmosDot(state) { setDot(cosmosDot, state); }

function fmtTime(t) {
  return t == null || !Number.isFinite(floatOf(t)) ? "—" : `${floatOf(t).toFixed(4)} s`;
}
function floatOf(v) { return typeof v === "number" ? v : parseFloat(v); }

function updateFrameUI(idx) {
  const pose = cosmos.poses[idx];
  const n = cosmos.frames;
  frameSlider.value = String(idx);
  const sliderMax = Number(frameSlider.max) || Math.max(n - 1, 1);
  const frac = sliderMax > 0 ? idx / sliderMax : 0;
  scrubProgress.style.width = `${(frac * 100).toFixed(2)}%`;
  scrubThumb.style.left = `${(frac * 100).toFixed(2)}%`;
  frameLabel.innerHTML = `frame <b>${String(idx).padStart(3, "0")}</b> / ${String(n - 1).padStart(3, "0")}`;
  timeLabel.textContent = `t = ${fmtTime(cosmos.times[idx])} · fps ${cosmos.fps}`;

  if (pose) {
    frameEmptyNote?.classList.add("is-hidden");
    const r = pose.posKm.length();
    const px = pose.posKm.x, py = pose.posKm.y, pz = pose.posKm.z;
    frameMeta.innerHTML = "";
    const pairs = [
      ["Position (km)", `${px.toFixed(1)}, ${py.toFixed(1)}, ${pz.toFixed(1)}`],
      ["Radius", `${r.toFixed(1)} km`],
    ];
    for (const [kKey, v] of pairs) {
      const dt = document.createElement("dt"); dt.textContent = kKey;
      const dd = document.createElement("dd"); dd.textContent = v;
      frameMeta.appendChild(dt); frameMeta.appendChild(dd);
    }

    // camera basis display: columns = camera axes (+Z forward, +X right, +Y down)
    const bx = new THREE.Vector3(1, 0, 0).applyQuaternion(pose.quat);
    const by = new THREE.Vector3(0, 1, 0).applyQuaternion(pose.quat);
    const bz = new THREE.Vector3(0, 0, 1).applyQuaternion(pose.quat);
    const fmtCell = (v) => (Math.abs(v) < 0.001 ? " 0 " : v.toFixed(2));
    const mGrid = [
      [bx.x, by.x, bz.x],
      [bx.y, by.y, bz.y],
      [bx.z, by.z, bz.z],
    ];
    poseMatrix.innerHTML = mGrid
      .map((rr) => `[ ${rr.map(fmtCell).join("  ")} ]`)
      .join("\n");

    if (currentPoseHighlight) {
      currentPoseHighlight.position.copy(pose.position);
      currentPoseHighlight.quaternion.copy(pose.quat);
      currentPoseHighlight.visible = true;
    }
    if (currentFrustum) {
      const f = poseScale * 0.35, s = poseScale * 0.18;
      const x = new THREE.Vector3(1, 0, 0).applyQuaternion(pose.quat);
      const y = new THREE.Vector3(0, 1, 0).applyQuaternion(pose.quat);
      const z = new THREE.Vector3(0, 0, 1).applyQuaternion(pose.quat);
      const corner = (dx, dy) => new THREE.Vector3().copy(pose.position)
        .add(z.clone().multiplyScalar(f)).add(x.clone().multiplyScalar(dx * s)).add(y.clone().multiplyScalar(dy * s));
      const cc = [corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1)];
      const pts = [];
      cc.forEach((c2) => pts.push(pose.position.clone(), c2));
      for (let i = 0; i < 4; i++) pts.push(cc[i], cc[(i + 1) % 4]);
      currentFrustum.geometry.setFromPoints(pts);
    }
  }

  updateFrameImage(idx);
}

function updateFrameImage(idx) {
  const img = cosmos.images.find((im) => im.frame === idx);
  if (img && img.url) {
    if (frameImg.src !== img.url) frameImg.src = img.url;
    frameRenderLabel.textContent = img.name;
    frameRender.classList.remove("is-hidden");
  } else {
    frameRender.classList.add("is-hidden");
  }
}

function destroyCosmosScene() {
  cosmosGroup.traverse((obj) => {
    if (obj.geometry) obj.geometry.dispose();
    if (obj.material) obj.material.dispose?.();
  });
  cosmosGroup.clear();
  currentPoseHighlight = null;
  currentFrustum = null;
  for (const url of cosmos.blobURLs) URL.revokeObjectURL(url);
  cosmos.blobURLs.clear();
  cosmos.images = [];
  frameRender.classList.add("is-hidden");
}

function fillCosmosInfo(job, outputs, manifest) {
  cosmosKv.innerHTML = "";
  const rows = [
    ["Job", job.job_id || job.id || "—"],
    ["Mode", job.mode || outputs?.mode || "—"],
    ["Model", job.model || outputs?.model || "—"],
    ["Frames", job.frames ?? "—"],
    ["FPS", job.fps ?? "—"],
    ["Pointing", job.pointing_law || outputs?.pointing_law || "—"],
    ["Scale k", fmt(job.scale_k_km_per_m, 1)],
    ["Resolution", Array.isArray(job.resolution) ? job.resolution.join(" × ") : "—"],
    ["Seed", job.seed ?? "—"],
    ["Window", job.window ? `${fmt(job.window.t0_tdb, 2)} – ${fmt(job.window.t1_tdb, 2)} s` : "—"],
    ["Mock", outputs?.mock ? "yes" : "no"],
  ];
  for (const [kV, v] of rows) {
    const dt = document.createElement("dt"); dt.textContent = kV;
    const dd = document.createElement("dd"); dd.textContent = String(v);
    cosmosKv.appendChild(dt); cosmosKv.appendChild(dd);
  }
  manifestRaw.textContent = manifest ? JSON.stringify(manifest, null, 2) : "// manifest.json not present in this directory";
  const prevCard = stripMetrics.querySelector('[data-strip="cosmos-frames"]');
  if (prevCard) prevCard.remove();
  const jobCard = buildStripCard("Job frames", cosmos.frames, "accent");
  jobCard.dataset.strip = "cosmos-frames";
  stripMetrics.appendChild(jobCard);
}

/* Render the pose reconstruction into the shared 3D scene. */
let poseScale = 1;
let currentPoseHighlight = null;
let currentFrustum = null;

function buildCosmosScene(poses) {
  const maxR = Math.max(...poses.map((pp) => pp.position.length()), 1e-3);
  // fit the pose ring into the scene just outside the earth, like the orbit view
  poseScale = (EARTH_RADIUS * 10) / maxR;
  cosmosGroup.clear();
  for (const pp of poses) pp.position = pp.posKm.clone().multiplyScalar(poseScale);

  // faint path through every frame
  cosmosGroup.add(new THREE.Line(pathGeometryFor(poses), new THREE.LineBasicMaterial({
    color: COSMOS_ACCENT, transparent: true, opacity: 0.35,
  })));

  // dim frustum glyphs for every frame
  cosmosGroup.add(buildFrustumGeometry(poses, poseScale));

  // current-frame highlight frustum (its geometry is rewritten on scrub)
  currentFrustum = new THREE.LineSegments(
    new THREE.BufferGeometry(),
    new THREE.LineBasicMaterial({ color: 0x67e8f9, transparent: false })
  );
  cosmosGroup.add(currentFrustum);

  // camera position marker
  currentPoseHighlight = new THREE.Mesh(
    new THREE.SphereGeometry(poseScale * 0.05, 16, 16),
    new THREE.MeshBasicMaterial({ color: 0xffffff })
  );
  cosmosGroup.add(currentPoseHighlight);
}

async function readJSON(fileLike, url) {
  const res = url ? await fetch(url) : await fetch(fileLike);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const text = await res.text();
  return JSON.parse(text);
}

async function applyCosmosJob({ job, outputs, manifest, dirURLBase, dirFiles }) {
  destroyCosmosScene();
  try {
    let integrated;
    try {
      integrated = integratePoses(job, outputs);
    } catch (e) {
      setCosmosDot("err");
      cosmosStatus.textContent = `Pose reconstruction failed — ${e.message}`;
      showError(`Cosmos job: ${e.message}`);
      return false;
    }

    cosmos.job = job;
    cosmos.outputs = outputs;
    cosmos.manifest = manifest;
    cosmos.poses = integrated.poses;
    cosmos.frames = integrated.poses.length;
    cosmos.fps = integrated.fps;
    cosmos.times = integrated.times;
    cosmos.current = 0;
    cosmos.playing = false;
    playBtn.textContent = "▶";
    playBtn.classList.remove("is-playing");

    // collect rendered-frame artifacts (image kind, or png/jpg names beside the job)
    const isImageName = (n) => /\.(png|jpe?g|webp)$/i.test(n || "");
    const frameOfName = (n) => {
      const mm = /(?:^|[^0-9])0*(\d{1,4})[^0-9]*$/.exec(n || "");
      return mm ? parseInt(mm[1], 10) : NaN;
    };
    if (dirFiles) {
      for (const f of dirFiles) {
        if (!isImageName(f.name)) continue;
        const fr = frameOfName(f.name);
        if (!Number.isNaN(fr) && fr < cosmos.frames) {
          const url = URL.createObjectURL(f);
          cosmos.blobURLs.add(url);
          cosmos.images.push({ frame: fr, name: f.name, url });
        }
      }
    } else if (manifest && Array.isArray(manifest.artifacts) && dirURLBase) {
      for (const a of manifest.artifacts) {
        if (!isImageName(a.path || a.name) || a.kind !== "image") continue;
        const fr = frameOfName(a.path || a.name);
        if (!Number.isNaN(fr) && fr < cosmos.frames) {
          cosmos.images.push({ frame: fr, name: a.path || a.name, url: dirURLBase + (a.path || a.name).split("/").map(encodeURIComponent).join("/") });
        }
      }
    }
    cosmos.images.sort((a, b) => a.frame - b.frame);

    buildCosmosScene(cosmos.poses);
    fillCosmosInfo(job, outputs, manifest);

    // load metrics.json if available
    (async () => {
      let metrics = null;
      if (dirFiles) {
        const m = dirFiles.find((f) => f.name === "metrics.json");
        if (m) { try { metrics = JSON.parse(await m.text()); } catch {} }
      } else if (dirURLBase) {
        try { const r = await fetch(dirURLBase + "metrics.json"); if (r.ok) metrics = await r.json(); } catch {}
      }
      if (cosmosMetricsEl) {
        cosmosMetricsEl.innerHTML = "";
        if (metrics) {
          const pairs = [
            ["RRE", `${Number(metrics.rre_degrees).toFixed(4)} deg`],
            ["RTE", `${Number(metrics.rte_metres).toFixed(2)} m`],
            ["ATE", `${Number(metrics.ate_metres).toFixed(2)} m`],
            ["Frames", metrics.frames],
            ["Aligned", metrics.aligned ? "yes" : "no"],
            ["Scale k", Number(metrics.scale_k_km_per_m).toFixed(1)],
          ];
          for (const [k, v] of pairs) {
            const dt = document.createElement("dt"); dt.textContent = k;
            const dd = document.createElement("dd"); dd.textContent = String(v);
            cosmosMetricsEl.appendChild(dt); cosmosMetricsEl.appendChild(dd);
          }
        } else {
          cosmosMetricsEl.innerHTML = "<dt>Metrics</dt><dd>not available (run evaluator)</dd>";
        }
      }
    })();

    // filmstrip
    if (filmstripEl) {
      filmstripEl.innerHTML = "";
      for (const img of cosmos.images) {
        const thumb = document.createElement("img");
        thumb.src = img.url; thumb.alt = img.name; thumb.className = "film-thumb";
        thumb.addEventListener("click", () => { cosmos.current = img.frame; updateFrameUI(img.frame); });
        filmstripEl.appendChild(thumb);
      }
    }

    frameSlider.max = String(cosmos.frames - 1);
    frameSlider.value = "0";
    cosmos.loaded = true;
    cosmosStatus.textContent = `${job.job_id || job.id || "job"} · ${cosmos.frames} frames · pointing ${job.pointing_law || "—"}`;
    setCosmosDot("ok");

    scrubber.classList.toggle("is-hidden", mode !== "cosmos");
    updateFrameUI(0);
    if (mode === "cosmos") {
      updateViewForMode();
      fitCameraToCosmos();
    }
    return true;
  } catch (error) {
    setCosmosDot("err");
    cosmosStatus.textContent = `Job loaded but rendering failed — ${error.message}`;
    return false;
  }
}

async function loadJobFromFiles(files) {
  setCosmosDot("loading");
  try {
    const byName = (n) => files.find((f) => f.name === n);
    if (!byName("job.json")) throw new Error("Selected folder has no job.json");
    const job = await readJSON(byName("job.json"));
    const outEntry = byName("sample_outputs.json");
    const outputs = outEntry ? await readJSON(outEntry) : null;
    const mnfEntry = byName("manifest.json");
    const manifest = mnfEntry ? await readJSON(mnfEntry) : null;
    if (!outputs) {
      // job looks queued but unfinished — still show metadata, no poses
      destroyCosmosScene();
      cosmos.job = job; cosmos.outputs = null; cosmos.manifest = manifest;
      cosmos.frames = job.frames ?? 0; cosmos.fps = job.fps ?? 24;
      cosmos.times = job.frame_times_tdb_s?.slice() ?? [];
      cosmos.poses = [];
      cosmos.loaded = false;
      fillCosmosInfo(job, outputs, manifest);
      frameSlider.max = "0";
      cosmosStatus.textContent = "No sample_outputs.json — job has not produced frames yet.";
      setCosmosDot("loading");
      scrubber.classList.add("is-hidden");
      frameEmptyNote?.classList.remove("is-hidden");
      return false;
    }
    await applyCosmosJob({ job, outputs, manifest, dirFiles: files });
    return true;
  } catch (error) {
    setCosmosDot("err");
    cosmosStatus.textContent = `Could not load job — ${error.message}`;
    showError(`Cosmos job: ${error.message}`);
    return false;
  }
}

async function loadDemoJob() {
  setCosmosDot("loading");
  const base = settings.jobDir || "./data/demo_job/";
  try {
    const job = await readJSON(null, base + (base.endsWith("/") ? "" : "/") + "job.json");
    const outputs = await readJSON(null, base + (base.endsWith("/") ? "" : "/") + "sample_outputs.json");
    let manifest = null;
    try { manifest = await readJSON(null, base + (base.endsWith("/") ? "" : "/") + "manifest.json"); } catch { /* demo manifest optional */ }
    await applyCosmosJob({ job, outputs, manifest, dirURLBase: base.endsWith("/") ? base : base + "/" });
  } catch (error) {
    setCosmosDot("err");
    cosmosStatus.textContent = `Job unavailable — ${base} (${error.message})`;
    showError(`Cosmos job: "${base}" missing or invalid (${error.message})`);
  }
}

/* ---------------- mode switching ---------------- */
let mode = "orbit";
function updateViewForMode() {
  for (const tab of modeTabs.querySelectorAll(".mode-tab")) {
    tab.classList.toggle("is-active", tab.dataset.mode === mode);
  }
  document.querySelectorAll("[data-mode-panel]").forEach((el) => {
    const show = el.dataset.modePanel === mode;
    el.classList.toggle("is-hidden", !show);
  });
  // The two modes live at very different scales (lunar arc vs. a pose ring
  // around Earth), so grid size and zoom limits have to follow the mode.
  if (mode === "cosmos") {
    applySceneScale(EARTH_RADIUS * 10);
    // Cosmos poses use the physical Earth, not the enlarged orbit-view marker.
    earth.scale.setScalar(1);
    atmosphere.scale.setScalar(1);
    halo.scale.setScalar(1);
    if (cosmos.loaded) fitCameraToCosmos();
  } else if (trajRadius) {
    applySceneScale(trajRadius);
    updateSunGeometry(trajRadius);
    updateBodyGeometry(trajRadius);
    controls.minDistance = trajRadius * 0.02;
    controls.maxDistance = trajRadius * 20;
  }
  trajGroup.visible = mode === "orbit" && settings.showTrajectory;
  cosmosGroup.visible = mode === "cosmos" && cosmos.poses.length > 0 && settings.showCosmosFrames;
  scrubber.classList.toggle("is-hidden", !(mode === "cosmos" && cosmos.loaded));
  if (mode !== "cosmos") frameRender.classList.add("is-hidden");
  else if (cosmos.loaded) updateFrameImage(cosmos.current);
  grid.visible = showGridCb.checked;
  // The orbit view enlarges bodies and compresses the Sun distance; Cosmos
  // frames are already to-scale renders, so the caption only applies here.
  sceneCaption?.classList.toggle("is-hidden", mode !== "orbit");
  setOrbitPropsVisible();
}
modeTabs.addEventListener("click", (ev) => {
  const tab = ev.target.closest(".mode-tab");
  if (!tab || tab.dataset.mode === mode) return;
  mode = tab.dataset.mode;
  if (mode !== "cosmos") stopPlayback();
  updateViewForMode();
});

/* ---------------- scrubber wiring ---------------- */
function stopPlayback() {
  cosmos.playing = false;
  playBtn.textContent = "▶";
  playBtn.classList.remove("is-playing");
}
playBtn.addEventListener("click", () => {
  cosmos.playing = !cosmos.playing;
  playBtn.textContent = cosmos.playing ? "❚❚" : "▶";
  playBtn.classList.toggle("is-playing", cosmos.playing);
});
frameSlider.addEventListener("input", () => {
  stopPlayback();
  cosmos.current = Number(frameSlider.value) || 0;
  updateFrameUI(cosmos.current);
});


/* ---------------- controls wiring ---------------- */
autoRotateCb.addEventListener("change", () => {
  controls.autoRotate = autoRotateCb.checked;
});
showGridCb.addEventListener("change", () => {
  grid.visible = showGridCb.checked;
});
showAxesCb.addEventListener("change", () => {
  axes.visible = showAxesCb.checked;
});
showSunCb?.addEventListener("change", setOrbitPropsVisible);
orbitPlayBtn?.addEventListener("click", orbitPlayToggle);
orbitSlider?.addEventListener("input", () => stepOrbit(0));

ensembleSlider?.addEventListener("input", () => {
  seekEnsembleEpoch(parseInt(ensembleSlider.value, 10) || 0);
});
showEnsembleCb?.addEventListener("change", () => {
  const idx = parseInt(orbitSlider?.value || "0", 10) || 0;
  const epoch = ensembleEpochForOrbitIndex(idx);
  if (ensembleSlider) ensembleSlider.value = String(epoch);
  renderEnsembleEpoch(epoch, { force: true });
});
resetViewBtn.addEventListener("click", () => {
  if (mode === "cosmos" && cosmos.loaded) {
    fitCameraToCosmos();
  } else {
    fitCameraToTrajectory(trajRadius);
  }
});
topViewBtn?.addEventListener("click", () => {
  if (mode !== "orbit" || !trajectoryData?.points?.length) return;
  // A spinning top-down view is useless; the guide framing is static.
  controls.autoRotate = false;
  if (autoRotateCb) autoRotateCb.checked = false;
  fitCameraTopDown();
});

showPropCb?.addEventListener("change", () => {
  trajGroup.children.forEach((c) => { if (c.name === "overlay-prop") c.visible = !!showPropCb.checked; });
});
showCorrCb?.addEventListener("change", () => {
  trajGroup.children.forEach((c) => { if (c.name === "overlay-corr") c.visible = !!showCorrCb.checked; });
});

/* ---------------- settings modal ---------------- */
const settingsOverlay = document.getElementById("settings-overlay");
const openSettingsBtn = document.getElementById("open-settings");
const settingsClose = document.getElementById("settings-close");
const settingsResetBtn = document.getElementById("settings-reset");
const reloadDataBtn = document.getElementById("reload-data-btn");
const settingsSavedNote = document.getElementById("settings-saved");
const playbackSpeedReadout = document.getElementById("playback-speed-readout");
const settingFields = Array.from(document.querySelectorAll("[data-setting]"));

function populateSettingsFields() {
  for (const el of settingFields) {
    const name = el.dataset.setting;
    if (!(name in settings)) continue;
    if (el.type === "checkbox") el.checked = Boolean(settings[name]);
    else el.value = String(settings[name] ?? "");
  }
  if (playbackSpeedReadout) playbackSpeedReadout.textContent = `${settings.playbackSpeed}×`;
}

function applyDisplaySettings() {
  document.body.classList.toggle("light", settings.theme === "light");
  // re-evaluate trajectory / cosmos-frame visibility with the new flags
  updateViewForMode();
}

let savedFlashTimer = null;
function flashSaved() {
  settingsSavedNote.classList.add("is-visible");
  clearTimeout(savedFlashTimer);
  savedFlashTimer = setTimeout(() => settingsSavedNote.classList.remove("is-visible"), 1600);
}

function openSettings() {
  populateSettingsFields();
  settingsOverlay.classList.remove("is-hidden");
  settingsOverlay.querySelector(".settings-input")?.focus();
}
function closeSettings() {
  settingsOverlay.classList.add("is-hidden");
}

openSettingsBtn.addEventListener("click", openSettings);
settingsClose.addEventListener("click", closeSettings);
settingsOverlay.addEventListener("click", (ev) => {
  if (ev.target === settingsOverlay) closeSettings();
});
window.addEventListener("keydown", (ev) => {
  if (ev.key === "Escape" && !settingsOverlay.classList.contains("is-hidden")) { closeSettings(); return; }
  if (settingsOverlay && !settingsOverlay.classList.contains("is-hidden")) return;
  if (ev.key === " ") { ev.preventDefault(); orbitPlayToggle(); }
  else if (ev.key === "ArrowRight") { stepOrbit(1); }
  else if (ev.key === "ArrowLeft") { stepOrbit(-1); }
  else if (ev.key === "1") { mode = "orbit"; updateViewForMode(); }
  else if (ev.key === "f" || ev.key === "F") {
    fitCameraToTrajectory(trajRadius);
  }
});

/* live-apply + persist on any change (persist on <input> and <change> so
   number fields with arrows and selects both behave) */
settingsOverlay.addEventListener("input", (ev) => {
  if (!pullFrom(ev.target)) return;
  applyDisplaySettings();
  save();
  flashSaved();
});
settingsOverlay.addEventListener("change", (ev) => {
  if (!pullFrom(ev.target)) return;
  applyDisplaySettings();
  save();
  flashSaved();
});

settingsResetBtn.addEventListener("click", () => {
  reset();
  populateSettingsFields();
  applyDisplaySettings();
  flashSaved();
});

/* re-fetch trajectory / PINN result from the configured paths */
reloadDataBtn.addEventListener("click", () => {
  loadTrajectory();
  loadPinnResult();
});

/* orbit playback */
function orbitPlayToggle() {
  orbitPlaying = !orbitPlaying;
  if (orbitPlayBtn) orbitPlayBtn.textContent = orbitPlaying ? "⏸" : "▶";
  orbitLastTick = performance.now();
  orbitAcc = 0;
}
function setOrbitIndex(idx, { syncEnsemble = true } = {}) {
  const pts = trajectoryData?.points || [];
  if (!pts.length) return 0;
  idx = Math.max(0, Math.min(pts.length - 1, idx | 0));
  if (orbitSlider) orbitSlider.value = String(idx);
  moonGroup.position.set(pts[idx][0], pts[idx][1], pts[idx][2]);
  if (traveledLine) traveledLine.geometry.setDrawRange(0, idx + 1);
  if (traveledTubeRef) {
    const { indexPerSegment, tubularSegments } = traveledTubeRef.userData;
    // Map the sample index onto the resampled tube's path parameter.
    const t = pts.length > 1 ? idx / (pts.length - 1) : 0;
    const seg = Math.max(1, Math.round(t * tubularSegments));
    traveledTubeRef.geometry.setDrawRange(0, seg * indexPerSegment);
  }
  if (orbitTimeEl) {
    const t = Array.isArray(trajectoryData.times) ? trajectoryData.times[idx] : idx;
    orbitTimeEl.textContent = `t = ${Number(t).toFixed(2)} s`;
  }
  applySunSample(idx);
  drawGuide(idx);
  if (syncEnsemble) syncEnsembleToOrbit(idx);
  return idx;
}
function stepOrbit(delta) {
  const cur = parseInt(orbitSlider?.value || "0", 10) || 0;
  setOrbitIndex(cur + delta);
}
function seekEnsembleEpoch(epoch) {
  if (!trajectoryData?.points?.length) return;
  const idx = orbitIndexForEnsembleEpoch(epoch);
  setOrbitIndex(idx, { syncEnsemble: false });
  if (ensembleSlider) ensembleSlider.value = String(epoch);
  renderEnsembleEpoch(epoch, { force: true });
}
function updateOrbitPlayback(dt) {
  if (!orbitPlaying || !trajectoryData?.points?.length) return;
  const max = trajectoryData.points.length - 1;
  orbitAcc += dt * settings.playbackSpeed * 2;
  let idx = parseInt(orbitSlider?.value || "0", 10) || 0;
  let stepped = false;
  while (orbitAcc >= 1) {
    orbitAcc -= 1;
    idx = (idx + 1) % (max + 1);
    stepped = true;
  }
  if (stepped) setOrbitIndex(idx);
}

/* ---------------- animate ---------------- */
function animate() {
  requestAnimationFrame(animate);
  earth.rotation.y += 0.0008;
  atmosphere.rotation.y += 0.0004;
  halo.rotation.y += 0.0002;
  controls.update();

  if (mode === "cosmos" && cosmos.playing && cosmos.loaded && cosmos.frames > 1) {
    const now = performance.now();
    const dt = Math.min((now - (cosmos.lastTick ?? now)) / 1000, 0.25);
    cosmos.lastTick = now;
    cosmos.playAcc += dt * cosmos.fps * settings.playbackSpeed;
    while (cosmos.playAcc >= 1) {
      cosmos.playAcc -= 1;
      cosmos.current = (cosmos.current + 1) % cosmos.frames;
    }
    updateFrameUI(cosmos.current);
  } else if (mode === "orbit" && orbitPlaying) {
    const now = performance.now();
    const dt = Math.min((now - (orbitLastTick ?? now)) / 1000, 0.25);
    orbitLastTick = now;
    updateOrbitPlayback(dt);
  } else {
    cosmos.lastTick = performance.now();
    orbitLastTick = performance.now();
  }

  renderer.render(scene, camera);
}

document.querySelectorAll("#pipeline-card [data-target]").forEach((btn) => {
  btn.addEventListener("click", async () => {
    const target = btn.dataset.target;
    if (pipelineLog) pipelineLog.textContent = `Running make ${target}…`;
    setDot(pipelineDot, "loading");
    try {
      let out = await invokeCmd("run_make", { target });
      // residual/train rewrite sources; rebuild the bundle so the viewer sees them.
      if (target === "residual" || target === "train") {
        out += "\n$ make ui-data\n";
        out += await invokeCmd("run_make", { target: "ui-data" });
      }
      if (pipelineLog) pipelineLog.textContent = out;
      if (pipelineStatus) pipelineStatus.textContent = `make ${target} completed`;
      setDot(pipelineDot, "ok");
      loadTrajectory();
      loadPinnResult();
      loadPredictions();
      loadManifest();
      loadUiManifest();
      loadEnsemble();
    } catch (e) {
      if (pipelineLog) pipelineLog.textContent = String(e);
      if (pipelineStatus) pipelineStatus.textContent = `make ${target} failed`;
      setDot(pipelineDot, "err");
    }
  });
});
pickFolderBtn?.addEventListener("click", async () => {
  const p = await openFolderDialog();
  if (!p) {
    // Browser fallback: the <input webkitdirectory> picker still works.
    jobDirInput?.click();
    return;
  }
  if (jobBaseInput) jobBaseInput.value = p;
  settings.jobDir = p;
  save();
  try {
    await loadJobFromPath(p);
  } catch (e) {
    setCosmosDot("err");
    cosmosStatus.textContent = `Could not load job — ${e.message}`;
    showError(`Cosmos job: ${e.message}`);
  }
});

/* ---------------- boot ---------------- */
loadDemoBtn?.addEventListener("click", () => loadDemoJob());

exportSnapshotBtn?.addEventListener("click", () => {
  try {
    const url = renderer.domElement.toDataURL("image/png");
    const a = document.createElement("a");
    a.href = url; a.download = "residual-snapshot.png"; a.click();
  } catch (e) { showError(`Export failed: ${e.message}`); }
});
exportReportBtn?.addEventListener("click", () => {
  const report = {
    generated: new Date().toISOString(),
    provenance: provenanceEl?.textContent || "",
    predictions_rows: predictionsData?.rows?.length ?? null,
    cosmos: cosmos.job ? { job_id: cosmos.job.job_id || cosmos.job.id, frames: cosmos.frames, fps: cosmos.fps } : null,
  };
  const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = "residual-report.json"; a.click();
  URL.revokeObjectURL(url);
});
loadJobBtn.addEventListener("click", () => jobDirInput.click());
jobDirInput.addEventListener("change", () => {
  const files = Array.from(jobDirInput.files || []);
  if (files.length) loadJobFromFiles(files);
  jobDirInput.value = "";
});
scanJobsBtn?.addEventListener("click", async () => {
  const base = (jobBaseInput?.value || "").trim();
  if (!base) return;
  settings.jobDir = base.endsWith("/") ? base : base + "/";
  save();
  if (isLocalPath(base) && hasBackend) {
    try {
      await loadJobFromPath(base);
    } catch (e) {
      setCosmosDot("err");
      cosmosStatus.textContent = `Could not load job — ${e.message}`;
      showError(`Cosmos job: ${e.message}`);
      return;
    }
  } else {
    await loadDemoJob();
  }
  if (jobListEl && cosmos.job) {
    jobListEl.innerHTML = "";
    const li = document.createElement("li");
    li.textContent = `${cosmos.job.job_id || cosmos.job.id || "job"} · ${cosmos.job.mode || "—"} · ${cosmos.frames} frames · ${cosmos.job.pointing_law || "—"}`;
    jobListEl.appendChild(li);
  }
});

setLoading(true);
sceneStatus.textContent = "Loading data…";
populateSettingsFields();
applyDisplaySettings();
resize();
updateViewForMode();
animate();
loadTrajectory();
loadPinnResult();
loadPredictions();
loadManifest();
loadUiManifest();
loadEnsemble();

// residual canvas hover
if (residualCanvas) {
  residualCanvas.addEventListener("mousemove", (ev) => {
    if (!residualChart.rows?.length) return;
    const rect = residualCanvas.getBoundingClientRect();
    const mx = ev.clientX - rect.left;
    const { rows, xOf } = residualChart;
    let best = -1, bestD = Infinity;
    for (let i = 0; i < rows.length; i++) {
      const x = xOf(rows[i].t);
      const d = Math.abs(x - mx);
      if (d < bestD) { bestD = d; best = i; }
    }
    if (best >= 0 && bestD < 12) {
      const r = rows[best];
      const col = residualTargetCols[0];
      residualHover.textContent = `t=${Number(r.t).toFixed(1)} actual=${Number(r.actual?.[col]).toFixed(4)} PINN=${Number(r.pinn?.[col]).toFixed(4)}`;
      residualHoverIdx = best;
    } else {
      residualHover.textContent = "";
      residualHoverIdx = -1;
    }
  });
  // redraw on toggle
  for (const id of ["show-actual","show-pinn","show-mlp"]) {
    document.getElementById(id)?.addEventListener("change", drawResidual);
  }
}

// failsafe: hide loader after 6s even if something stalls
setTimeout(() => setLoading(false), 6000);
