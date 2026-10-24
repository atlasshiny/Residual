import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { settings, save, reset, pullFrom } from "./settings.js";

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
const eqSindyLabel = document.getElementById("eq-sindy-label");
const kanMetrics = document.getElementById("kan-metrics");
const kanRaw = document.getElementById("kan-raw");
const stripMetrics = document.getElementById("strip-metrics");

const autoRotateCb = document.getElementById("auto-rotate");
const showGridCb = document.getElementById("show-grid");
const showAxesCb = document.getElementById("show-axes");
const resetViewBtn = document.getElementById("reset-view");

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

function buildFallbackNote(msg) {
  const div = document.createElement("div");
  div.className = "fallback-note";
  div.dataset.role = "data-fallback";
  /* allow placement inside the 2-column metrics grid */
  div.style.gridColumn = "1 / -1";
  div.textContent = msg;
  return div;
}

/* Show a symbolic string for the equation. Prefer the SINDy expression; if
   absent, summarize the KAN edge splines so the panel is never empty. */
function describeKanEdges(edges) {
  if (!Array.isArray(edges) || edges.length === 0) return null;
  return edges
    .map((e) => {
      const coeffs = (e.coeffs || []).map((c) => c.toFixed(4)).join(", ");
      return `spline x${e.i}→z${e.j} · coeffs (${coeffs}) · r² ${fmt(e.r2, 3)}`;
    })
    .join("  |  ");
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
  setDot(trajDot, "loading");
  try {
    const response = await fetch(src);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    const rawPoints = data.points || [];
    const points = rawPoints.map((p) => new THREE.Vector3(p[0], p[1], p[2]));
    if (points.length < 2) throw new Error("Not enough trajectory samples");

    // glowing tube along curve
    let maxR = 0, minR = Infinity, maxY = 0;
    for (const p of points) {
      const r = Math.hypot(p.x, p.z);
      maxR = Math.max(maxR, r);
      minR = Math.min(minR, r);
      maxY = Math.max(maxY, Math.abs(p.y));
    }
    const closedLoop =
      points.length > 2 && points[0].distanceTo(points[points.length - 1]) <= maxR * 0.01;
    const curvePoints = closedLoop ? points.slice(0, -1) : points;
    const curve = new THREE.CatmullRomCurve3(curvePoints, closedLoop, "centripetal");
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
    trajKv.innerHTML = "";
    const flat = maxY <= maxR * 0.005;
    const kvPairs = [
      ["Samples", points.length],
      ["Max radius", fmt(maxR, 1)],
      ["Min radius", fmt(minR, 1)],
      ["Plane", flat ? "XZ (y ≈ 0)" : "3D"],
      ["Loop", closedLoop ? "Closed" : "Open"],
    ];
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
    sceneStatus.textContent = (trajectoryLoaded && kanLoaded) ? "Scene ready" : sceneStatus.textContent;
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

async function loadKanResult() {
  const src = settings.kanPath || "./data/kan_result.json";
  kanLoaded = false;
  const card = document.getElementById("kan-card");
  card?.querySelectorAll(".fallback-note").forEach((n) => n.remove());
  stripMetrics.querySelectorAll('[data-strip="kan"]').forEach((n) => n.remove());
  setDot(kanDot, "loading");
  try {
    const response = await fetch(src);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();

    const target = (data.target || "—").toString();
    const kan = data.kan || {};
    const sindy = data.sindy || {};
    let sindyExpr = (data.sindy && data.sindy.expression) ? data.sindy.expression : null;
    let eqKind = "SINDy discovered";
    if (!sindyExpr) {
      sindyExpr = describeKanEdges(kan.edges);
      eqKind = "Learned (KAN edges)";
    }
    eqSindyLabel.textContent = eqKind;
    eqTarget.textContent = target;
    eqSindy.textContent = sindyExpr || "—";

    kanRaw.textContent = JSON.stringify(data, null, 2);

    // metrics cards
    kanMetrics.innerHTML = "";
    const mlp = data.mlp || {};
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
    [
      buildStripCard("KAN R²", fmt(kan.train_r2, 4), "accent"),
      buildStripCard("SINDy R²", fmt(sindy.train_r2, 4), "accent"),
      buildStripCard("MLP R²", fmt(mlp.train_r2, 4), ""),
      buildStripCard("Samples", data.n_samples ?? "—", ""),
    ].forEach((c) => {
      c.dataset.strip = "kan";
      stripMetrics.appendChild(c);
    });

    kanLoaded = true;
    sceneStatus.textContent = (trajectoryLoaded && kanLoaded) ? "Scene ready" : sceneStatus.textContent;
    settle();
  } catch (error) {
    const message = fileError("KAN result unavailable", src, error);
    kanStatus.textContent = "KAN result unavailable — see note below";
    setDot(kanDot, "err");
    kanMetrics.innerHTML = "";
    eqSindyLabel.textContent = "SINDy discovered";
    eqTarget.textContent = "—";
    eqSindy.textContent = "—";
    kanMetrics.appendChild(buildFallbackNote(`${message}. Equations and metrics will stay empty.`));
    kanFailed = true;
    showError(message);
    settle();
  }
}

function maybeFinishLoading() {
  if (trajectoryLoaded && kanLoaded) {
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

    frameSlider.max = String(cosmos.frames - 1);
    frameSlider.value = "0";
    cosmos.loaded = true;
    cosmosStatus.textContent = `${job.job_id || job.id || "job"} · ${cosmos.frames} frames · pointing ${job.pointing_law || "—"}`;
    setCosmosDot("ok");

    scrubber.classList.toggle("is-hidden", mode !== "cosmos");
    updateFrameUI(0);
    if (mode === "cosmos") {
      updateViewForMode();
      // gentle refit of the camera on the pose ring
      controls.target.set(0, 0, 0);
      const r = EARTH_RADIUS * 10;
      camera.position.set(r * 0.12, r * 1.15, r * 2.6);
      controls.update();
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
  trajGroup.visible = mode === "orbit" && settings.showTrajectory;
  cosmosGroup.visible = mode === "cosmos" && cosmos.poses.length > 0 && settings.showCosmosFrames;
  scrubber.classList.toggle("is-hidden", !(mode === "cosmos" && cosmos.loaded));
  if (mode !== "cosmos") frameRender.classList.add("is-hidden");
  else if (cosmos.loaded) updateFrameImage(cosmos.current);
  grid.visible = showGridCb.checked;
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
resetViewBtn.addEventListener("click", () => {
  if (mode === "cosmos" && cosmos.loaded) {
    // refit the camera on the reconstructed pose ring
    controls.target.set(0, 0, 0);
    const r = EARTH_RADIUS * 10;
    camera.position.set(r * 0.12, r * 1.15, r * 2.6);
  } else {
    camera.position.set(0, 6000, 22000);
    controls.target.set(0, 0, 0);
  }
  controls.update();
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
  if (ev.key === "Escape" && !settingsOverlay.classList.contains("is-hidden")) closeSettings();
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

/* re-fetch trajectory / kan result from the configured paths */
reloadDataBtn.addEventListener("click", () => {
  loadTrajectory();
  loadKanResult();
});

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
  } else {
    cosmos.lastTick = performance.now();
  }

  renderer.render(scene, camera);
}

/* ---------------- boot ---------------- */
loadJobBtn.addEventListener("click", () => jobDirInput.click());
jobDirInput.addEventListener("change", () => {
  const files = Array.from(jobDirInput.files || []);
  if (files.length) loadJobFromFiles(files);
  jobDirInput.value = "";
});
loadDemoBtn.addEventListener("click", () => loadDemoJob());

setLoading(true);
sceneStatus.textContent = "Loading data…";
populateSettingsFields();
applyDisplaySettings();
resize();
updateViewForMode();
animate();
loadTrajectory();
loadKanResult();

// failsafe: hide loader after 6s even if something stalls
setTimeout(() => setLoading(false), 6000);
