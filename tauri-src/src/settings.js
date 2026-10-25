/* Settings store — persisted to localStorage, applied on load.
   No framework, no deps; plain object with defaults + save()/reset(). */

const KEY = "residual-settings-v1";

export const DEFAULTS = Object.freeze({
  /* data source paths */
  trajectoryPath: "./data/trajectory.json",
  pinnPath: "./data/pinn_result.json",
  predictionsPath: "./data/pinn_predictions.json",
  manifestPath: "./data/manifest.json",

  /* display */
  theme: "dark", // "dark" | "light"
  autoRotate: false,
  showTrajectory: true,
  playbackSpeed: 1,

  /* propagation params (re-run hints; not used by the viewer itself) */
  propDt: 1,
  propSteps: 10000,
  propBodies: 2,
});

function readStored() {
  let stored = {};
  try {
    stored = JSON.parse(localStorage.getItem(KEY) || "{}") || {};
  } catch {
    stored = {};
  }
  return { ...structuredClone(DEFAULTS), ...stored };
}

export const settings = readStored();

export function save() {
  try {
    localStorage.setItem(KEY, JSON.stringify(settings));
  } catch {
    /* storage full/blocked — settings stay session-only */
  }
}

export function reset() {
  Object.assign(settings, structuredClone(DEFAULTS));
  save();
}

/* read one value back into the store from a typed field */
export function pullFrom(el) {
  const name = el?.dataset?.setting;
  if (!name || !(name in settings)) return false;
  if (el.type === "checkbox") settings[name] = el.checked;
  else if (el.type === "range" || el.type === "number") {
    const n = Number(el.value);
    settings[name] = Number.isFinite(n) ? n : DEFAULTS[name];
  } else settings[name] = el.value;
  return true;
}
