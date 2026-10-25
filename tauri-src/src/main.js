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
const missionDot = document.getElementById("mission-dot");
const missionStatus = document.getElementById("mission-status");
const missionSummary = document.getElementById("mission-summary");
const missionCenterSel = document.getElementById("mission-center");
const missionDurationEl = document.getElementById("mission-duration");
const missionDurationUnit = document.getElementById("mission-duration-unit");
const missionRunBtn = document.getElementById("mission-run");
const missionRestoreBtn = document.getElementById("mission-restore");
const missionAddBurnBtn = document.getElementById("mission-add-burn");
const missionBurnsEl = document.getElementById("mission-burns");
const missionPositionInputs = ["rx", "ry", "rz"].map((id) =>
  document.getElementById(`mission-${id}`)
);
const missionVelocityInputs = ["vx", "vy", "vz"].map((id) =>
  document.getElementById(`mission-${id}`)
);
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

const tauri = window.__TAURI__ || null;
const hasBackend = !!tauri?.core?.invoke;
async function invokeCmd(name, args) {
  if (!hasBackend) throw new Error("Tauri backend unavailable (running in browser?)");
  return tauri.core.invoke(name, args || {});
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
let traveledLine = null;
let traveledTubeRef = null;
let trajectoryData = null;
let trajectorySource = "bundle"; // "bundle" | "mission"
let trajRadius = 0;
let orbitPlane = null;
let ensembleEpoch = -1;
const resetViewBtn = document.getElementById("reset-view");
const topViewBtn = document.getElementById("top-view");

/* ---------------- Scene ---------------- */
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x050816, 0.00002);

const camera = new THREE.PerspectiveCamera(55, 1, 1, 10000000);
camera.position.set(0, 6000, 22000);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(10, 10);
viewport.appendChild(renderer.domElement);

/* ---------------- Render scheduling ----------------
   The viewer used to render every animation frame forever, even with a static
   camera and auto-rotate off.  Now a frame is drawn only when something
   actually changed (camera movement, playback, data load, resize, toggles).
   This keeps idle CPU/GPU near zero, which matters most on low-end machines
   and battery-powered laptops. */
let needsRender = true;
let windowFocused = !document.hidden;
function requestRender() {
  needsRender = true;
}

function setAutoRotate(on) {
  controls.autoRotate = !!on;
  if (autoRotateCb.checked !== !!on) autoRotateCb.checked = !!on;
  settings.autoRotate = !!on;
  requestRender();
}

/* Explain whether WebGL is running on real hardware.  Software rendering
   (llvmpipe / SwiftShader) is the usual reason a WebGL app feels slow. */
const glContext = renderer.getContext();
const gpuRenderer = (() => {
  try {
    const dbg = glContext.getExtension("WEBGL_debug_renderer_info");
    return String(
      dbg ? glContext.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : glContext.getParameter(glContext.RENDERER)
    );
  } catch {
    return "unknown";
  }
})();
const softwareGl = /swiftshader|llvmpipe|softpipe|software|basic render/i.test(gpuRenderer);
console.info(`[residual] WebGL renderer: ${gpuRenderer}${softwareGl ? " (software rendering — perf will be limited)" : ""}`);

/* ---------------- Perf HUD (dev) ----------------
   Toggle with Ctrl+Shift+P, persist with localStorage["residual-perf"]="1",
   or open the page with ?perf=1.  Shows the numbers needed to verify the
   render-on-demand / device-tier work without a profiler attached. */
const PERF_HUD_KEY = "residual-perf";
const bootStart = performance.now();
let bootReadyMs = null;
const perfHud = document.getElementById("perf-hud");
let perfVisible = (() => {
  try {
    return localStorage.getItem(PERF_HUD_KEY) === "1" || /(?:^|[?&])perf(?:=1)?(?:&|$)/.test(location.search);
  } catch {
    return false;
  }
})();
let perfFrames = 0;
let perfWindowStart = performance.now();
let perfFps = 0;
let perfDrawMs = 0;

function setPerfVisible(on) {
  perfVisible = !!on;
  if (perfHud) perfHud.hidden = !perfVisible;
  try {
    localStorage.setItem(PERF_HUD_KEY, perfVisible ? "1" : "0");
  } catch {
    /* storage unavailable */
  }
  requestRender();
}

function updatePerfHud(now, drawMs) {
  if (!perfHud || !perfVisible) return;
  perfDrawMs = perfDrawMs * 0.85 + drawMs * 0.15;
  perfFrames++;
  if (now - perfWindowStart < 500) return;
  perfFps = (perfFrames * 1000) / (now - perfWindowStart);
  perfFrames = 0;
  perfWindowStart = now;
  const info = renderer.info.render;
  const boot = bootReadyMs == null ? "…" : `${Math.round(bootReadyMs)}ms`;
  perfHud.textContent =
    `${perfFps.toFixed(0)} fps · ${perfDrawMs.toFixed(1)} ms · ` +
    `${info.calls} calls · ${(info.triangles / 1000).toFixed(1)}k tris · ` +
    `dpr ${renderer.getPixelRatio().toFixed(2)} · boot ${boot}\n${gpuRenderer}`;
}

window.addEventListener("keydown", (ev) => {
  if (ev.ctrlKey && ev.shiftKey && ev.key.toLowerCase() === "p") {
    ev.preventDefault();
    setPerfVisible(!perfVisible);
  }
});
if (perfVisible && perfHud) perfHud.hidden = false;

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
moonGroup.name = "moon-marker";
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

/* Mission marker.  A live mission's moving body is the spacecraft, not the
   Moon, so it gets its own group and the lunar reference view stays unchanged. */
const craftGroup = new THREE.Group();
craftGroup.name = "craft-marker";
craftGroup.visible = false;
scene.add(craftGroup);

const craftBody = new THREE.Mesh(
  new THREE.SphereGeometry(1, 24, 24),
  new THREE.MeshStandardMaterial({
    color: 0x22d3ee,
    emissive: 0x0e7490,
    emissiveIntensity: 1.4,
    roughness: 0.35,
    metalness: 0.2,
  })
);
craftGroup.add(craftBody);

const craftGlow = new THREE.Sprite(
  new THREE.SpriteMaterial({
    map: radialSpriteTexture(),
    color: 0x22d3ee,
    transparent: true,
    opacity: 0.7,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    fog: false,
  })
);
craftGroup.add(craftGlow);

const craftLabel = labelSprite("CRAFT", "", "165,243,252");
craftGroup.add(craftLabel);

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
  requestRender();
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
  requestRender();
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

  craftBody.scale.setScalar(R * 0.006);
  craftGlow.scale.setScalar(R * 0.03);
  craftLabel.scale.set(R * 0.11, R * 0.055, 1);
  craftLabel.position.set(0, R * 0.05, 0);

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
  const hasPath = !!trajectoryData?.points?.length;
  const missionMode = trajectorySource === "mission";
  sunGroup.visible = !!showSunCb?.checked && sunAvailable;
  sunBeam.visible = sunGroup.visible;
  starGroup.visible = true;
  earthLabel.visible = true;
  moonGroup.visible = !missionMode && hasPath;
  craftGroup.visible = missionMode && hasPath;
  requestRender();
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
  requestRender();
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
  if (trajectoryLoaded && pinnLoaded && bootReadyMs == null) {
    bootReadyMs = performance.now() - bootStart;
  }
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

/* ---------------- mission runner ----------------
   The form is the front end of `propagate_mission`: the Tauri backend runs the
   Rust Dormand-Prince core and returns the same `points`/`times` shape as the
   static bundle, so `applyTrajectoryBundle` can draw both. */
const MISSION_STORAGE_KEY = "residual-mission";
const MISSION_PRESETS = {
  leo: {
    center: "earth",
    duration: 2,
    unit: 3600,
    r: [6778.137, 0, 0],
    v: [0, 7.6686, 0],
    burns: [],
  },
  tli: {
    center: "earth",
    duration: 3,
    unit: 86400,
    r: [6778.137, 0, 0],
    v: [0, 10.75, 0],
    burns: [],
  },
};

function fmtKm(km) {
  if (!Number.isFinite(km)) return "—";
  if (Math.abs(km) >= 100_000) return `${Math.round(km / 1000)}k km`;
  return `${Math.round(km).toLocaleString()} km`;
}

function syncBurnEmptyNote() {
  if (!missionBurnsEl) return;
  const rows = missionBurnsEl.querySelectorAll(".mission-burn-row");
  let note = missionBurnsEl.querySelector(".mission-burn-empty");
  if (rows.length === 0 && !note) {
    note = document.createElement("p");
    note.className = "mission-burn-empty";
    note.textContent = "No burns — pure coast.";
    missionBurnsEl.appendChild(note);
  } else if (rows.length > 0 && note) {
    note.remove();
  }
}

function addMissionBurnRow(burn = null) {
  if (!missionBurnsEl) return;
  const row = document.createElement("div");
  row.className = "mission-burn-row";
  const fields = [
    ["t", "t (s)", burn?.t_s],
    ["dvx", "Δvx", burn?.dv_m_s?.[0]],
    ["dvy", "Δvy", burn?.dv_m_s?.[1]],
    ["dvz", "Δvz", burn?.dv_m_s?.[2]],
  ];
  for (const [key, label, value] of fields) {
    const input = document.createElement("input");
    input.type = "number";
    input.step = "any";
    input.className = "settings-input";
    input.dataset.burn = key;
    input.placeholder = label;
    input.setAttribute("aria-label", label);
    if (value != null && Number.isFinite(Number(value))) input.value = String(value);
    row.appendChild(input);
  }
  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "mission-burn-remove";
  remove.textContent = "✕";
  remove.setAttribute("aria-label", "Remove burn");
  remove.addEventListener("click", () => {
    row.remove();
    syncBurnEmptyNote();
  });
  row.appendChild(remove);
  missionBurnsEl.appendChild(row);
  syncBurnEmptyNote();
}

function fillMissionForm(preset) {
  if (missionCenterSel) missionCenterSel.value = preset.center || "earth";
  if (missionDurationEl) missionDurationEl.value = String(preset.duration);
  if (missionDurationUnit) missionDurationUnit.value = String(preset.unit);
  missionPositionInputs.forEach((el, i) => {
    if (el) el.value = String(preset.r[i]);
  });
  missionVelocityInputs.forEach((el, i) => {
    if (el) el.value = String(preset.v[i]);
  });
  if (missionBurnsEl) missionBurnsEl.innerHTML = "";
  for (const burn of preset.burns || []) addMissionBurnRow(burn);
  syncBurnEmptyNote();
}

function readMissionForm() {
  const number = (el, label) => {
    const value = Number(el?.value);
    if (!Number.isFinite(value)) throw new Error(`${label} must be a number`);
    return value;
  };
  const duration =
    number(missionDurationEl, "duration") * (Number(missionDurationUnit?.value) || 1);
  if (!(duration > 0)) throw new Error("duration must be positive");

  const position_km = missionPositionInputs.map((el, i) => number(el, `r${"xyz"[i]}`));
  const velocity_km_s = missionVelocityInputs.map((el, i) => number(el, `v${"xyz"[i]}`));

  const burns = [];
  const rows = missionBurnsEl
    ? Array.from(missionBurnsEl.querySelectorAll(".mission-burn-row"))
    : [];
  for (const row of rows) {
    const values = ["t", "dvx", "dvy", "dvz"].map(
      (key) => row.querySelector(`[data-burn="${key}"]`)?.value ?? ""
    );
    if (values.every((value) => String(value).trim() === "")) continue;
    const [t, dvx, dvy, dvz] = values.map(Number);
    if (![t, dvx, dvy, dvz].every(Number.isFinite)) {
      throw new Error("burn fields must be numbers");
    }
    burns.push({ t_s: t, dv_m_s: [dvx, dvy, dvz] });
  }
  if (burns.length > 8) throw new Error("at most 8 impulsive burns");

  return {
    name: "UI mission",
    center: missionCenterSel?.value === "barycentric" ? "barycentric" : "earth",
    position_km,
    velocity_km_s,
    duration_s: duration,
    samples: 1200,
    start_epoch_s: 0,
    burns,
  };
}

function saveMissionForm() {
  try {
    const burns = missionBurnsEl
      ? Array.from(missionBurnsEl.querySelectorAll(".mission-burn-row"))
          .map((row) => {
            const read = (key) => row.querySelector(`[data-burn="${key}"]`)?.value ?? "";
            return { t: read("t"), dvx: read("dvx"), dvy: read("dvy"), dvz: read("dvz") };
          })
          .filter((row) => [row.t, row.dvx, row.dvy, row.dvz].some((v) => String(v).trim() !== ""))
      : [];
    localStorage.setItem(
      MISSION_STORAGE_KEY,
      JSON.stringify({
        center: missionCenterSel?.value || "earth",
        duration: missionDurationEl?.value ?? "2",
        unit: missionDurationUnit?.value ?? "3600",
        r: missionPositionInputs.map((el) => el?.value ?? ""),
        v: missionVelocityInputs.map((el) => el?.value ?? ""),
        burns,
      })
    );
  } catch {
    /* storage unavailable */
  }
}

function restoreMissionForm() {
  let saved = null;
  try {
    saved = JSON.parse(localStorage.getItem(MISSION_STORAGE_KEY) || "null");
  } catch {
    saved = null;
  }
  if (!saved || !Array.isArray(saved.r) || !Array.isArray(saved.v)) {
    fillMissionForm(MISSION_PRESETS.leo);
    return;
  }
  if (missionCenterSel) missionCenterSel.value = saved.center || "earth";
  if (missionDurationEl) missionDurationEl.value = String(saved.duration ?? "2");
  if (missionDurationUnit) missionDurationUnit.value = String(saved.unit ?? "3600");
  missionPositionInputs.forEach((el, i) => {
    if (el) el.value = String(saved.r[i] ?? "");
  });
  missionVelocityInputs.forEach((el, i) => {
    if (el) el.value = String(saved.v[i] ?? "");
  });
  if (missionBurnsEl) missionBurnsEl.innerHTML = "";
  for (const burn of saved.burns || []) {
    addMissionBurnRow({ t_s: burn.t, dv_m_s: [burn.dvx, burn.dvy, burn.dvz] });
  }
  syncBurnEmptyNote();
}

async function runMission() {
  if (!hasBackend) {
    setDot(missionDot, "err");
    if (missionStatus) missionStatus.textContent = "Mission runner needs the Tauri desktop backend.";
    if (missionSummary) {
      missionSummary.textContent = "Launch the desktop build (npm run dev in tauri-src) to propagate in-app.";
    }
    return;
  }

  let request;
  try {
    request = readMissionForm();
  } catch (error) {
    const detail = error?.message || String(error);
    setDot(missionDot, "err");
    if (missionStatus) missionStatus.textContent = detail;
    return;
  }

  if (missionRunBtn) missionRunBtn.disabled = true;
  setDot(missionDot, "loading");
  if (missionStatus) missionStatus.textContent = "Propagating…";
  const started = performance.now();
  try {
    const data = await invokeCmd("propagate_mission", { request });
    applyTrajectoryBundle(data, { source: "mission" });
    const elapsed = performance.now() - started;
    const stats = data.stats || {};
    if (missionStatus) {
      missionStatus.textContent = `Live mission · ${data.frame || "earth-centered-inertial"}`;
    }
    if (missionSummary) {
      missionSummary.textContent = [
        `${data.count ?? 0} samples`,
        `${fmtKm(stats.min_radius_km)} – ${fmtKm(stats.max_radius_km)} radius`,
        `${Math.round(elapsed)} ms`,
      ].join(" · ");
    }
    setDot(missionDot, "ok");
    saveMissionForm();
  } catch (error) {
    const detail = typeof error === "string" ? error : error?.message || String(error);
    if (missionStatus) missionStatus.textContent = `Mission failed — ${detail}`;
    if (missionSummary) {
      missionSummary.textContent = "Check the state vector, duration, and burn times.";
    }
    setDot(missionDot, "err");
    showError(`Mission: ${detail}`);
  } finally {
    if (missionRunBtn) missionRunBtn.disabled = false;
  }
}

function setupMissionUI() {
  missionRunBtn?.addEventListener("click", runMission);
  missionRestoreBtn?.addEventListener("click", async () => {
    setDot(missionDot, "loading");
    if (missionStatus) missionStatus.textContent = "Loading bundled reference…";
    await loadTrajectory();
    if (trajectoryLoaded) {
      setDot(missionDot, "");
      if (missionStatus) missionStatus.textContent = "Bundled reference loaded.";
      if (missionSummary) {
        missionSummary.textContent = "Runs the Rust Dormand–Prince core. Earth-centred by default.";
      }
    } else {
      setDot(missionDot, "err");
    }
  });
  missionAddBurnBtn?.addEventListener("click", () => addMissionBurnRow());
  document.querySelectorAll("[data-preset]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const preset = MISSION_PRESETS[btn.dataset.preset];
      if (!preset) return;
      fillMissionForm(preset);
      saveMissionForm();
      setDot(missionDot, "");
      if (missionStatus) missionStatus.textContent = "Preset loaded — press Run mission.";
    });
  });
  document.getElementById("mission-card")?.addEventListener("change", saveMissionForm);
}

/* Tear down the previous path and reset the per-bundle view state.  Shared by
   the static bundle loader and the live mission runner. */
function resetTrajectoryScene() {
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
  craftGroup.visible = false;
  sunAvailable = false;
  sunDirections = null;
  sunRangeKm = null;
  sunDirection = null;
  setOrbitPropsVisible();
}

/* Geometry is built from a bounded number of samples so a long trajectory does
   not turn into an enormous tube mesh.  The full point list stays in
   `trajectoryData` for markers, timing, charts, and the playback slider. */
const MAX_TRAJECTORY_RENDER_POINTS = 4000;
const MAX_TUBULAR_SEGMENTS = 4096;

function downsampleArray(items, maxItems) {
  if (items.length <= maxItems) return items;
  if (maxItems < 2) return items.slice(0, 1);
  // Keep the first and last sample and spread the rest evenly across the arc.
  const out = new Array(maxItems);
  const step = (items.length - 1) / (maxItems - 1);
  for (let i = 0; i < maxItems; i++) out[i] = items[Math.round(i * step)];
  return out;
}

/* Build the scene from a trajectory document.  Both the bundled JSON file and
   the mission runner's live response use this shape (`points`, `times`,
   optional overlays), so the renderer never needs to know where it came from. */
function applyTrajectoryBundle(data, { source = "bundle" } = {}) {
  const rawPoints = data.points || [];
  const points = rawPoints.map((p) => new THREE.Vector3(p[0], p[1], p[2]));
  if (points.length < 2) throw new Error("Not enough trajectory samples");
  trajectoryData = data;
  trajectorySource = source;

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
  // Tube and line geometry use the downsampled set; the duplicate closing
  // sample is added back to the line so a closed orbit still meets itself.
  const renderPoints = downsampleArray(curvePoints, MAX_TRAJECTORY_RENDER_POINTS);
  const renderLinePoints = closedLoop
    ? [...renderPoints, renderPoints[0]]
    : renderPoints;
  const curve = new THREE.CatmullRomCurve3(renderPoints, closedLoop, "centripetal");
  const tubeRadius = Math.max(20, maxR * 0.0035);
  const tubularSegments = Math.max(
    Math.min(renderPoints.length * 2, MAX_TUBULAR_SEGMENTS),
    64
  );
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
    new THREE.BufferGeometry().setFromPoints(renderLinePoints),
    new THREE.LineBasicMaterial({ color: 0xfde68a, transparent: true, opacity: 0.22 })
  );
  trajGroup.add(fullLine);

  traveledLine = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(renderLinePoints),
    new THREE.LineBasicMaterial({ color: 0xfff7d6, transparent: true, opacity: 1 })
  );
  traveledLine.geometry.setDrawRange(0, 1);
  // The draw range is expressed in render samples, not full trajectory
  // samples, so playback maps its progress before applying it.
  traveledLine.userData.renderSamples = renderLinePoints.length;
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
    const pts = downsampleArray(arr, MAX_TRAJECTORY_RENDER_POINTS).map((p) =>
      new THREE.Vector3(p[0], p[1], p[2])
    );
    const geo = new THREE.BufferGeometry().setFromPoints(pts);
    const mat = new THREE.LineBasicMaterial({ color: spec.color, transparent: true, opacity: 0.85 });
    const line = new THREE.Line(geo, mat);
    line.name = `overlay-${spec.name}`;
    line.visible = spec.name === "prop" ? (showPropCb?.checked ?? false) : (showCorrCb?.checked ?? false);
    trajGroup.add(line);
  }

  // A live mission has no residual overlays or ensemble cloud.  Disable the
  // controls instead of leaving toggles that silently do nothing.
  const hasPropOverlay =
    Array.isArray(data.points_propagated) && data.points_propagated.length >= 2;
  const hasCorrOverlay =
    Array.isArray(data.points_corrected) && data.points_corrected.length >= 2;
  if (showPropCb) {
    showPropCb.disabled = !hasPropOverlay;
    if (!hasPropOverlay) showPropCb.checked = false;
  }
  if (showCorrCb) {
    showCorrCb.disabled = !hasCorrOverlay;
    if (!hasCorrOverlay) showCorrCb.checked = false;
  }
  const missionMode = source === "mission";
  if (showEnsembleCb) {
    showEnsembleCb.disabled = missionMode;
    if (missionMode) showEnsembleCb.checked = false;
  }
  if (ensembleSlider) ensembleSlider.disabled = missionMode;
  if (missionMode && ensembleInfo) ensembleInfo.textContent = "not available for live missions";

  // Moon marker at the current epoch (the craft marker in mission mode).
  moonGroup.position.copy(points[0]);
  craftGroup.position.copy(points[0]);

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
    ["Source", missionMode ? `live ${data.name || "mission"}` : "bundle"],
    ["Frame", data.frame || (missionMode ? "earth-centered-inertial" : "barycentric")],
    ["Max radius", fmt(maxR, 1)],
    ["Min radius", fmt(minR, 1)],
    [
      "Plane",
      planar ? `${orbitPlane.name} (${"XYZ"[order[2]]} ≈ 0)` : "3D",
    ],
    ["Loop", closedLoop ? "Closed" : "Open"],
  ];
  if (renderLinePoints.length < points.length) {
    kvPairs.push(["Render samples", renderLinePoints.length]);
  }
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

  trajStatus.textContent = missionMode
    ? `Live mission · ${points.length} samples`
    : `${points.length} samples loaded`;
  setDot(trajDot, "ok");

  // strip metric
  const stripCard = buildStripCard("Trajectory samples", points.length, "amber");
  stripCard.dataset.strip = "traj-samples";
  stripMetrics.prepend(stripCard);

  trajectoryLoaded = true;
  trajectoryFailed = false;
  if (source === "mission") {
    provenanceEl.textContent = `Live mission · ${data.schema || "mission-trajectory"} · Rust Dormand–Prince (Earth–Moon)`;
    if (sceneCaption) {
      sceneCaption.textContent = `${data.frame || "earth-centered-inertial"} · live propagation · bodies enlarged, not to scale`;
    }
  } else {
    updateProvenance();
    if (sceneCaption) {
      sceneCaption.textContent = "Bodies and Sun enlarged · distances compressed · not to scale";
    }
  }
  setOrbitPropsVisible();
  fitCameraToTrajectory(maxR);
  sceneStatus.textContent = (trajectoryLoaded && pinnLoaded) ? "Scene ready" : sceneStatus.textContent;
  setOrbitIndex(0);
}

async function loadTrajectory() {
  const src = settings.trajectoryPath || "./data/trajectory.json";
  trajectoryLoaded = false;
  trajectorySource = "bundle";
  resetTrajectoryScene();
  setDot(trajDot, "loading");
  try {
    const response = await fetch(src);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    applyTrajectoryBundle(data, { source: "bundle" });
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
  // The ensemble is sampled on the bundled arc's epoch grid, so it has no
  // meaning while a live mission owns the scene.
  if (trajectorySource === "mission") return;
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
  // The cloud is sampled on the static arc; never overlay it on a mission.
  if (trajectorySource === "mission") return;
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
  if (!showEnsembleCb?.checked) { requestRender(); return; }
  const cloud = ensembleData.clouds?.[idx];
  if (!Array.isArray(cloud)) { requestRender(); return; }
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
  requestRender();
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
  ctx.fillStyle = "#030611"; ctx.fillRect(0, 0, W, H);
  if (!trajectoryData?.points?.length) return;
  const pts = trajectoryData.points;
  const i = Math.max(0, Math.min(pts.length - 1, idx ?? 0));
  const missionMode = trajectorySource === "mission";
  const earthCentered =
    missionMode && (trajectoryData.frame || "earth-centered-inertial") === "earth-centered-inertial";

  // Analytic Earth/Moon positions at this sample's absolute time, kilometres.
  // The bundled arc is barycentric, so it carries an Earth offset; a live
  // Earth-centred mission puts Earth at the origin and only shifts the Moon.
  const MU_E = 3.986004418e14, MU_M = 4.9048695e12, D = 3.844e8;
  const total = MU_E + MU_M;
  const relativeT = Array.isArray(trajectoryData.times) ? Number(trajectoryData.times[i]) : i;
  const absoluteT =
    Array.isArray(trajectoryData.times_absolute) && trajectoryData.times_absolute.length
      ? Number(trajectoryData.times_absolute[i])
      : (Number(trajectoryData.start_epoch_s) || 0) + relativeT;
  const theta = Math.sqrt(total / (D * D * D)) * absoluteT;
  const mf = MU_M / total;
  const c = Math.cos(theta), s = Math.sin(theta);
  const earth3 = earthCentered
    ? [0, 0, 0]
    : [-D * mf * c / 1000, -D * mf * s / 1000, 0];
  const moon3 = earthCentered
    ? [D * c / 1000, D * s / 1000, 0]
    : [D * (1 - mf) * c / 1000, D * (1 - mf) * s / 1000, 0];

  // pick the two axes that actually span the orbit (this dataset is XY, not XZ)
  const ax = orbitPlane?.a ?? 0;
  const bx = orbitPlane?.b ?? 2;
  const proj = (p) => [p[ax], p[bx]];

  // Frame the full path, as the Python guide renderer does. A remote Moon is
  // omitted for Earth-centred live missions to preserve the useful arc scale.
  const boundsOf = (list) => {
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const p of list) {
      minX = Math.min(minX, p[0]); maxX = Math.max(maxX, p[0]);
      minY = Math.min(minY, p[1]); maxY = Math.max(maxY, p[1]);
    }
    return { minX, maxX, minY, maxY };
  };
  const reference2 = pts.map(proj);
  const path = [...reference2];
  const corr = trajectoryData.points_corrected;
  const showCorrected = Array.isArray(corr) && corr.length && !missionMode && (showCorrCb?.checked ?? false);
  if (showCorrected) for (const p of corr) path.push(proj(p));
  let box = boundsOf([...path, proj(earth3)]);
  const cx0 = (box.minX + box.maxX) / 2;
  const cy0 = (box.minY + box.maxY) / 2;
  const pathRadius = Math.hypot(box.maxX - box.minX, box.maxY - box.minY) / 2 || 1;
  const moon2 = proj(moon3);
  const showMoon =
    !missionMode || Math.hypot(moon2[0] - cx0, moon2[1] - cy0) <= pathRadius * 1.6;
  if (showMoon) box = boundsOf([...path, proj(earth3), moon2]);
  const cx = (box.minX + box.maxX) / 2;
  const cy = (box.minY + box.maxY) / 2;
  const pad = 20;
  const scale = 0.88 * Math.min((W - 2 * pad) / (box.maxX - box.minX || 1), (H - 2 * pad) / (box.maxY - box.minY || 1));
  const px = (p) => [W / 2 + (p[0] - cx) * scale, H / 2 - (p[1] - cy) * scale];

  // Deterministic starfield, so stars stay still while the epoch advances.
  let seed = 3;
  const random = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
  for (let k = 0; k < 140; k++) {
    const x = random() * W, y = random() * H;
    ctx.fillStyle = `rgba(210,220,240,${0.2 + random() * 0.55})`;
    const size = random() > 0.92 ? 2 : 1;
    ctx.fillRect(x, y, size, size);
  }

  const drawPath = (arr, color, width, alpha = 1) => {
    if (!arr?.length) return;
    ctx.strokeStyle = color; ctx.globalAlpha = alpha; ctx.lineWidth = width;
    ctx.lineJoin = "round"; ctx.lineCap = "round"; ctx.beginPath();
    arr.forEach((p, k) => {
      const q = px(p);
      if (k === 0) ctx.moveTo(q[0], q[1]); else ctx.lineTo(q[0], q[1]);
    });
    ctx.stroke(); ctx.globalAlpha = 1;
  };
  // Dim complete planned arc with the traveled segment highlighted.
  drawPath(reference2, "#e2e8f0", 1.25, 0.24);
  drawPath(reference2.slice(0, i + 1), "#f59e0b", 4, 0.28);
  drawPath(reference2.slice(0, i + 1), "#fff7d6", 1.7);

  if (showCorrected) {
    const corrected2 = corr.map(proj);
    drawPath(corrected2, "#22d3ee", 1.1, 0.28);
    drawPath(corrected2.slice(0, Math.min(i, corrected2.length - 1) + 1), "#22d3ee", 1.6, 0.95);
  }

  // earth
  const ep = px(proj(earth3));
  ctx.fillStyle = "rgba(37,99,235,0.24)";
  ctx.beginPath(); ctx.arc(ep[0], ep[1], 10, 0, Math.PI*2); ctx.fill();
  ctx.fillStyle = "#285fca";
  ctx.beginPath(); ctx.arc(ep[0], ep[1], 5.5, 0, Math.PI*2); ctx.fill();

  // moon (may sit outside the framed path and be omitted)
  if (showMoon) {
    const mp = px(moon2);
    ctx.fillStyle = "rgba(203,213,225,0.22)";
    ctx.beginPath(); ctx.arc(mp[0], mp[1], 7, 0, Math.PI*2); ctx.fill();
    ctx.fillStyle = "#b7bcc4";
    ctx.beginPath(); ctx.arc(mp[0], mp[1], 3.5, 0, Math.PI*2); ctx.fill();
  }

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

  // current body: the spacecraft in both modes, cyan for live missions
  const sp = px(proj(pts[i]));
  ctx.fillStyle = missionMode ? "rgba(34,211,238,0.22)" : "rgba(245,158,11,0.24)";
  ctx.beginPath(); ctx.arc(sp[0], sp[1], 9, 0, Math.PI*2); ctx.fill();
  ctx.fillStyle = missionMode ? "#22d3ee" : "#ff9b35";
  ctx.beginPath(); ctx.arc(sp[0], sp[1], 4, 0, Math.PI*2); ctx.fill();

  // labels
  ctx.fillStyle = "#cbd5e1"; ctx.font = "600 10px sans-serif";
  ctx.fillText("EARTH", ep[0]+9, ep[1]-7);
  if (showMoon) {
    const mp = px(moon2);
    ctx.fillText("MOON", mp[0]+7, mp[1]-5);
  }
  if (missionMode) {
    ctx.fillStyle = "#a5f3fc";
    ctx.fillText("CRAFT", sp[0] + 7, sp[1] - 7);
  }

  // Small legend echoes the reference renderer's color roles.
  const legendY = H - 10;
  ctx.font = "9px sans-serif";
  ctx.fillStyle = "#cbd5e1";
  ctx.fillText("Reference", 10, legendY);
  ctx.fillStyle = "#fff7d6";
  ctx.fillText("Traveled", 82, legendY);
  if (showCorrected) {
    ctx.fillStyle = "#22d3ee";
    ctx.fillText("Corrected", 147, legendY);
  }
}

/* ---------------- view visibility ---------------- */
/* The viewer only has the orbit mode now; this applies display settings
   without re-running any camera framing or scene-scale work. */
function applyViewSettings() {
  trajGroup.visible = !!settings.showTrajectory;
  grid.visible = showGridCb.checked;
  axes.visible = showAxesCb.checked;
  sceneCaption?.classList.remove("is-hidden");
  setOrbitPropsVisible();
  requestRender();
}

/* ---------------- pause when the window is not visible ---------------- */
document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    windowFocused = false;
    return;
  }
  windowFocused = document.hasFocus();
  lastFrameTime = performance.now();
  requestRender();
});
window.addEventListener("focus", () => {
  windowFocused = true;
  lastFrameTime = performance.now();
  requestRender();
});
window.addEventListener("blur", () => {
  windowFocused = false;
});

/* ---------------- controls wiring ---------------- */
autoRotateCb.addEventListener("change", () => {
  setAutoRotate(autoRotateCb.checked);
  save();
});
showGridCb.addEventListener("change", () => {
  grid.visible = showGridCb.checked;
  requestRender();
});
showAxesCb.addEventListener("change", () => {
  axes.visible = showAxesCb.checked;
  requestRender();
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
  fitCameraToTrajectory(trajRadius);
});
topViewBtn?.addEventListener("click", () => {
  if (!trajectoryData?.points?.length) return;
  // A spinning top-down view is useless; the guide framing is static.
  setAutoRotate(false);
  save();
  fitCameraTopDown();
});

showPropCb?.addEventListener("change", () => {
  trajGroup.children.forEach((c) => { if (c.name === "overlay-prop") c.visible = !!showPropCb.checked; });
  requestRender();
});
showCorrCb?.addEventListener("change", () => {
  trajGroup.children.forEach((c) => { if (c.name === "overlay-corr") c.visible = !!showCorrCb.checked; });
  requestRender();
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
  setAutoRotate(settings.autoRotate);
  applyViewSettings();
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
  orbitAcc = 0;
  lastFrameTime = performance.now();
  requestRender();
}
function setOrbitIndex(idx, { syncEnsemble = true } = {}) {
  const pts = trajectoryData?.points || [];
  if (!pts.length) return 0;
  idx = Math.max(0, Math.min(pts.length - 1, idx | 0));
  if (orbitSlider) orbitSlider.value = String(idx);
  moonGroup.position.set(pts[idx][0], pts[idx][1], pts[idx][2]);
  craftGroup.position.set(pts[idx][0], pts[idx][1], pts[idx][2]);
  const t = pts.length > 1 ? idx / (pts.length - 1) : 0;
  if (traveledLine) {
    // The line geometry is downsampled, so map trajectory progress onto the
    // render sample count before capping the draw range.
    const samples = traveledLine.userData.renderSamples || pts.length;
    const renderIdx = samples > 1 ? Math.round(t * (samples - 1)) : 0;
    traveledLine.geometry.setDrawRange(0, renderIdx + 1);
  }
  if (traveledTubeRef) {
    const { indexPerSegment, tubularSegments } = traveledTubeRef.userData;
    // Map the sample index onto the resampled tube's path parameter.
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
  requestRender();
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
let lastFrameTime = performance.now();
let lastIdleHud = 0;

function animate(now) {
  requestAnimationFrame(animate);
  // rAF normally stops when the page is hidden; this also covers platforms
  // (and minimized windows) where it keeps ticking.
  if (document.hidden) return;

  const t = typeof now === "number" ? now : performance.now();
  const dt = Math.min((t - lastFrameTime) / 1000, 0.25);
  lastFrameTime = t;

  // OrbitControls.update() returns true while the camera is still moving
  // (damping / auto-rotate) and false once the view has settled.  Skip it
  // entirely when the window is unfocused so auto-rotate doesn't silently
  // keep moving the camera in the background.
  const cameraMoved = windowFocused ? controls.update() : false;
  const playing = orbitPlaying && (trajectoryData?.points?.length ?? 0) > 1;
  if (playing && windowFocused) updateOrbitPlayback(dt);

  const animating = windowFocused && controls.autoRotate;
  const shouldDraw = needsRender || animating || (cameraMoved && windowFocused);

  if (shouldDraw) {
    // Decorative spin only advances while frames are actually drawn, so an
    // idle view is truly static and costs no GPU time.
    earth.rotation.y += 0.0008;
    atmosphere.rotation.y += 0.0004;
    halo.rotation.y += 0.0002;

    const drawStart = performance.now();
    renderer.render(scene, camera);
    needsRender = false;
    if (perfVisible) updatePerfHud(t, performance.now() - drawStart);
  } else if (perfVisible && perfHud && t - lastIdleHud > 1000) {
    lastIdleHud = t;
    perfHud.textContent = `idle (0 fps) · dpr ${renderer.getPixelRatio().toFixed(2)} · boot ${bootReadyMs == null ? "…" : Math.round(bootReadyMs) + "ms"}\n${gpuRenderer}`;
  }
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
setLoading(true);
sceneStatus.textContent = "Loading data…";
// Auto-rotate is opt-in now: an idle viewer must not render at 60 fps.
autoRotateCb.checked = !!settings.autoRotate;
controls.autoRotate = autoRotateCb.checked;
populateSettingsFields();
applyDisplaySettings();
resize();
animate();
restoreMissionForm();
setupMissionUI();
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
