// Runs Pyodide and scopecard off the page's main thread, so the page stays responsive during
// long audits. Messages: {id, kind: 'ready' | 'wordnet' | 'run', argv, files: [{path, data: ArrayBuffer}]}.
// During a run, the audit's '[stage] ...' lines on stderr are forwarded as {type: 'stage', text}. A file that
// does not have the expected format comes back as {ok: false, inputError}; any other failure as
// {ok: false, error, trace}.
"use strict";
const PYODIDE_VERSION = "0.27.7";
importScripts(`https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/pyodide.js`);

let py = null;
const ready = (async () => {
  py = await loadPyodide();
  py.setStderr({ batched: (line) => { if (line.startsWith("[stage] ")) postMessage({ type: "stage", text: line.slice(8) }); } });
  postMessage({ type: "status", text: "Loading NumPy…" });
  await py.loadPackage("numpy");
  postMessage({ type: "status", text: "Loading scopecard…" });
  const zip = await (await fetch("scopecard.zip", { cache: "no-cache" })).arrayBuffer();
  py.unpackArchive(zip, "zip", { extractDir: "/bsa" });
  py.runPython("import sys; sys.path.insert(0, '/bsa'); import scopecard");
  return py.runPython("scopecard.__version__");
})();

let wordnetPath = null;
async function ensureWordnet() {
  if (wordnetPath) return wordnetPath;
  const buf = await (await fetch("wordnet30_wn18.tsv.gz", { cache: "no-cache" })).arrayBuffer();
  py.FS.mkdirTree("/data");
  py.FS.writeFile("/data/wordnet30_wn18.tsv.gz", new Uint8Array(buf));
  wordnetPath = "/data/wordnet30_wn18.tsv.gz";
  return wordnetPath;
}

const dirname = (p) => p.slice(0, p.lastIndexOf("/"));

onmessage = async (ev) => {
  const { id, kind, argv, files } = ev.data;
  try {
    const version = await ready;
    if (kind === "ready") { postMessage({ id, ok: true, version, pyodide: PYODIDE_VERSION }); return; }
    if (kind === "wordnet") { postMessage({ id, ok: true, path: await ensureWordnet() }); return; }
    for (const f of files) {
      py.FS.mkdirTree(dirname(f.path));
      py.FS.writeFile(f.path, new Uint8Array(f.data));
    }
    const out = argv[argv.indexOf("--out") + 1], md = argv[argv.indexOf("--md") + 1];
    for (const p of [out, md]) py.FS.mkdirTree(dirname(p));
    py.globals.set("ARGV", py.toPy(argv));
    // the command line's own main(): the browser card equals the CLI card. Input errors are re-raised so the
    // page can show them as such.
    const result = py.runPython(`
import io, contextlib
from scopecard.cli import main as _main
from scopecard.io import InputError as _InputError
_buf = io.StringIO()
try:
    with contextlib.redirect_stdout(_buf):
        _main(list(ARGV), catch_errors=False)
    _res = ('ok', _buf.getvalue())
except _InputError as _e:
    _res = ('input', str(_e))
_res`).toJs();
    if (result[0] === "input") { postMessage({ id, ok: false, inputError: result[1] }); return; }
    postMessage({ id, ok: true, stdout: result[1], json: py.FS.readFile(out, { encoding: "utf8" }), md: py.FS.readFile(md, { encoding: "utf8" }) });
  } catch (e) {
    const msg = String((e && e.message) || e);
    const last = msg.trim().split("\n").slice(-1)[0];
    postMessage({ id, ok: false, error: last, trace: msg });
  }
};
