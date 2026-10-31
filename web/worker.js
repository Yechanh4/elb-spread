/* Static build only: runs the Python app (core/, app/, data/) inside Pyodide, off the main thread.
   Messages in: {id, path, body}. Out: {id, ok, data | error} and {progress} while loading. */
"use strict";
const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.27.2/full/";
importScripts(PYODIDE + "pyodide.js");

const ready = (async () => {
  postMessage({ progress: "파이썬 엔진 받는 중 (처음 1회, 약 20MB)" });
  const py = await loadPyodide({ indexURL: PYODIDE });
  postMessage({ progress: "numpy·scipy 불러오는 중" });
  await py.loadPackage(["numpy", "scipy", "sqlite3", "tzdata"]);
  postMessage({ progress: "시뮬레이터 코드·데이터 스냅샷 불러오는 중" });
  const [code, db] = await Promise.all([
    fetch("py.zip", { cache: "no-cache" }).then(r => { if (!r.ok) throw new Error("py.zip " + r.status); return r.arrayBuffer(); }),
    fetch("data/market.sqlite", { cache: "no-cache" }).then(r => { if (!r.ok) throw new Error("market.sqlite " + r.status); return r.arrayBuffer(); }),
  ]);
  py.unpackArchive(code, "zip", { extractDir: "/home/pyodide/elb" });
  py.FS.mkdirTree("/data");
  py.FS.writeFile("/data/market.sqlite", new Uint8Array(db));
  py.runPython(`
import sys, warnings
sys.path.insert(0, "/home/pyodide/elb")
warnings.filterwarnings("ignore")
from app.static_api import dispatch
`);
  postMessage({ progress: null });
  return py.globals.get("dispatch");
})();

onmessage = async e => {
  const { id, path, body } = e.data;
  try {
    const dispatch = await ready;
    const out = JSON.parse(dispatch(path, body === undefined ? "" : JSON.stringify(body)));
    if (out && out.error && Object.keys(out).length === 1) postMessage({ id, ok: false, error: out.error });
    else postMessage({ id, ok: true, data: out });
  } catch (err) {
    postMessage({ id, ok: false, error: String(err && err.message || err) });
  }
};
