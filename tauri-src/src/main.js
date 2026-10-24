import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js";
import { OrbitControls } from "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js";

/* ---------------- DOM refs ---------------- */
const viewport = document.getElementById("viewport");
const loadingOverlay = document.getElementById("loading-overlay");
const errorBanner = document.getElementById("error-banner");
const sceneStatus = document.getElementById("scene-status");
const trajStatus = document.getElementById("trajectory-status");
const trajDot = document.getElementById("traj-dot");
const trajKv = document.getElementById("traj-kv");
const kanStatus = document.getElementById("kan-status");
const kanDot = document.getElementById("kan-dot");
const eqTarget = document.getElementById("eq-target");
const eqSindy = document.getElementById("eq-sindy");
const kanMetrics = document.getElementById("kan-metrics");
const kanRaw = document.getElementById("kan-raw");
const stripMetrics = document.getElementById("strip-metrics");

const autoRotateCb = document.getElementById("auto-rotate");
const showGridCb = document.getElementById("show-grid");
const showAxesCb = document.getElementById("show-axes");
const resetViewBtn = document.getElementById("reset-view");

/* ---------------- Scene ---------------- */
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x050816, 0.00002);

const camera = new THREE.PerspectiveCamera(55, 1, 0.1, 100000);
camera.position.set(0, 6000, 22000);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(10, 10);
viewport.appendChild(renderer.domElement);

/* lighting */
const ambient = new THREE.AmbientLight(0x334466, 0.6);
scene.add(ambient);
const sun = new THREE.DirectionalLight(0xffffff, 1.1);
sun.position.set(5000, 4000, 6000);
scene.add(sun);
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
let kanLoaded = false;
let trajectoryFailed = false;
let kanFailed = false;
let settledCount = 0;
function settle() {
  settledCount++;
  if (settledCount >= 2) setLoading(false);
}

async function loadTrajectory() {
  setDot(trajDot, "loading");
  try {
    const response = await fetch("./data/trajectory.json");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    const rawPoints = data.points || [];
    const points = rawPoints.map((p) => new THREE.Vector3(p[0], p[1], p[2]));
    if (points.length < 2) throw new Error("Not enough trajectory samples");

    // glowing tube along curve
    const curve = new THREE.CatmullRomCurve3(points, false, "centripetal");
    const tube = new THREE.TubeGeometry(curve, Math.max(points.length * 2, 64), 120, 12, false);
    const tubeMat = new THREE.MeshStandardMaterial({
      color: 0xf59e0b,
      emissive: 0xf59e0b,
      emissiveIntensity: 1.2,
      roughness: 0.3,
      metalness: 0.1,
      transparent: true,
      opacity: 0.95,
    });
    const tubeMesh = new THREE.Mesh(tube, tubeMat);
    trajGroup.add(tubeMesh);

    // core line for crispness
    const lineGeo = new THREE.BufferGeometry().setFromPoints(points);
    const line = new THREE.Line(
      lineGeo,
      new THREE.LineBasicMaterial({ color: 0xfde68a, transparent: true, opacity: 0.9 })
    );
    trajGroup.add(line);

    // head marker
    const head = new THREE.Mesh(
      new THREE.SphereGeometry(220, 24, 24),
      new THREE.MeshBasicMaterial({ color: 0xffffff })
    );
    head.position.copy(points[points.length - 1]);
    trajGroup.add(head);

    // metrics
    let maxR = 0, minR = Infinity;
    for (const p of points) {
      const r = Math.hypot(p.x, p.z);
      maxR = Math.max(maxR, r);
      minR = Math.min(minR, r);
    }
    trajKv.innerHTML = "";
    const kvPairs = [
      ["Samples", points.length],
      ["Max radius", fmt(maxR, 1)],
      ["Min radius", fmt(minR, 1)],
      ["Plane", "XZ (y ≈ 0)"],
    ];
    for (const [k, v] of kvPairs) {
      const dt = document.createElement("dt"); dt.textContent = k;
      const dd = document.createElement("dd"); dd.textContent = v;
      trajKv.appendChild(dt); trajKv.appendChild(dd);
    }

    trajStatus.textContent = `${points.length} samples loaded`;
    setDot(trajDot, "ok");

    // strip metric
    stripMetrics.appendChild(buildStripCard("Trajectory samples", points.length, "amber"));

    trajectoryLoaded = true;
    sceneStatus.textContent = (trajectoryLoaded && kanLoaded) ? "Scene ready" : sceneStatus.textContent;
    settle();
  } catch (error) {
    trajStatus.textContent = `Trajectory load failed: ${error.message}`;
    setDot(trajDot, "err");
    trajectoryFailed = true;
    showError(`Trajectory load failed: ${error.message}`);
    settle();
  }
}

async function loadKanResult() {
  setDot(kanDot, "loading");
  try {
    const response = await fetch("./data/kan_result.json");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();

    const target = (data.target || "—").toString();
    const sindyExpr = (data.sindy && data.sindy.expression) ? data.sindy.expression : "—";
    eqTarget.textContent = target;
    eqSindy.textContent = sindyExpr;

    kanRaw.textContent = JSON.stringify(data, null, 2);

    // metrics cards
    kanMetrics.innerHTML = "";
    const kan = data.kan || {};
    const mlp = data.mlp || {};
    const sindy = data.sindy || {};
    const cards = [
      ["KAN train R²", fmt(kan.train_r2, 4), true],
      ["KAN train MSE", fmt(kan.train_mse, 5), false],
      ["KAN params", kan.num_parameters ?? "—", false],
      ["KAN degree", kan.degree ?? "—", false],
      ["SINDy train R²", fmt(sindy.train_r2, 4), true],
      ["SINDy train MSE", fmt(sindy.train_mse, 5), false],
      ["MLP train R²", fmt(mlp.train_r2, 4), false],
      ["MLP train MSE", fmt(mlp.train_mse, 5), false],
      ["Samples", data.n_samples ?? "—", false],
    ];
    for (const [label, value, accent] of cards) {
      kanMetrics.appendChild(buildMetricCard(label, value, { accent }));
    }

    kanStatus.textContent = "Loaded";
    setDot(kanDot, "ok");

    // strip metrics
    stripMetrics.appendChild(buildStripCard("KAN R²", fmt(kan.train_r2, 4), "accent"));
    stripMetrics.appendChild(buildStripCard("SINDy R²", fmt(sindy.train_r2, 4), "accent"));
    stripMetrics.appendChild(buildStripCard("MLP R²", fmt(mlp.train_r2, 4), ""));
    stripMetrics.appendChild(buildStripCard("Samples", data.n_samples ?? "—", ""));

    kanLoaded = true;
    sceneStatus.textContent = (trajectoryLoaded && kanLoaded) ? "Scene ready" : sceneStatus.textContent;
    settle();
  } catch (error) {
    kanStatus.textContent = `KAN result load failed: ${error.message}`;
    setDot(kanDot, "err");
    kanFailed = true;
    showError(`KAN result load failed: ${error.message}`);
    settle();
  }
}

function maybeFinishLoading() {
  if (trajectoryLoaded && kanLoaded) {
    setLoading(false);
    sceneStatus.textContent = "Scene ready";
  }
}

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
resetViewBtn.addEventListener("click", () => {
  camera.position.set(0, 6000, 22000);
  controls.target.set(0, 0, 0);
  controls.update();
});

/* ---------------- animate ---------------- */
function animate() {
  requestAnimationFrame(animate);
  earth.rotation.y += 0.0008;
  atmosphere.rotation.y += 0.0004;
  halo.rotation.y += 0.0002;
  controls.update();
  renderer.render(scene, camera);
}

/* ---------------- boot ---------------- */
setLoading(true);
sceneStatus.textContent = "Loading data…";
resize();
animate();
loadTrajectory();
loadKanResult();

// failsafe: hide loader after 6s even if something stalls
setTimeout(() => setLoading(false), 6000);
