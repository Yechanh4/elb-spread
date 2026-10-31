/* Static build only: runs the Python app (core/, app/, data/) inside Pyodide, off the main thread.
   Messages in: {id, path, body}. Out: {id, ok, data | error} and {progress} while loading. */
"use strict";
const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.27.2/full/";
importScripts(PYODIDE + "pyodide.js");
const loaded = new Set();          // tickers whose snapshot file is already in the in-memory cache

const ready = (async () => {
  postMessage({ progress: "파이썬 엔진 받는 중 (처음 1회, 약 20MB)" });
  const py = await loadPyodide({ indexURL: PYODIDE });
  postMessage({ progress: "numpy·scipy 불러오는 중" });
  await py.loadPackage(["numpy", "scipy", "sqlite3", "tzdata"]);
  postMessage({ progress: "시뮬레이터 코드·종목 목록 불러오는 중" });
  const get = (url, kind) => fetch(url, { cache: "no-cache" }).then(r => {
    if (!r.ok) throw new Error(url + " " + r.status);
    return kind === "text" ? r.text() : r.arrayBuffer();
  });
  const [code, core, index] = await Promise.all([get("py.zip"), get("data/core.sqlite"), get("data/index.json", "text")]);
  py.unpackArchive(code, "zip", { extractDir: "/home/pyodide/elb" });
  py.FS.mkdirTree("/data");
  py.FS.writeFile("/data/core.sqlite", new Uint8Array(core));
  py.runPython(`
import sys, warnings
sys.path.insert(0, "/home/pyodide/elb")
warnings.filterwarnings("ignore")
from app import static_api
`);
  const api = py.pyimport("app.static_api");
  api.set_index(index);
  api.service();
  postMessage({ progress: null });
  return api;
})();

// Each ticker's data is a separate file, fetched the first time a request needs it.
async function ensureTickers(api, path, body) {
  const need = JSON.parse(api.needed_tickers(path, body === undefined ? "" : JSON.stringify(body))).filter(t => !loaded.has(t));
  if (!need.length) return;
  postMessage({ progress: `종목 데이터 받는 중: ${need.join(", ")}` });
  const texts = await Promise.all(need.map(t =>
    fetch(`data/t/${encodeURIComponent(t)}.json`, { cache: "no-cache" }).then(r => (r.ok ? r.text() : null)).catch(() => null)));
  texts.forEach((txt, k) => { if (txt) { api.load_ticker(txt); loaded.add(need[k]); } });
  postMessage({ progress: null });
}

onmessage = async e => {
  const { id, path, body } = e.data;
  try {
    const api = await ready;
    await ensureTickers(api, path, body);
    const out = JSON.parse(api.dispatch(path, body === undefined ? "" : JSON.stringify(body)));
    if (out && out.error && Object.keys(out).length === 1) postMessage({ id, ok: false, error: out.error });
    else postMessage({ id, ok: true, data: out });
  } catch (err) {
    postMessage({ id, ok: false, error: String(err && err.message || err) });
  }
};
