"""A local browser demo: type or speak English, see the ASL gloss and the signing animation.

    s2a demo            # opens http://localhost:8000

Uses only the Python standard library for the server. The microphone button uses the
browser's built-in speech recognition (Chrome / Edge / Safari), so no Whisper install is needed.
"""
from __future__ import annotations

import base64
import json
import tempfile
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>S2A Demo</title>
<style>
  :root { --bg:#f6f5f2; --card:#fff; --ink:#1d1d1f; --muted:#6b6b70; --accent:#2f6fde; --line:#e3e1dc;
          --sign:#1f8a4c; --spell:#b7791f; --miss:#c53030; }
  @media (prefers-color-scheme: dark) { :root { --bg:#141416; --card:#1e1e21; --ink:#f2f2f3; --muted:#a0a0a8;
          --line:#33333a; --accent:#6c9cff; } }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink); font:16px/1.5 -apple-system, system-ui, sans-serif; }
  main { max-width:960px; margin:0 auto; padding:32px 16px; }
  h1 { margin:0 0 4px; font-size:28px; }
  .sub { color:var(--muted); margin:0 0 24px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:20px; margin-bottom:16px; }
  .row { display:flex; gap:8px; flex-wrap:wrap; }
  input[type=text] { flex:1 1 320px; font-size:18px; padding:12px 14px; border-radius:10px;
          border:1px solid var(--line); background:var(--bg); color:var(--ink); }
  button { font-size:16px; padding:12px 18px; border-radius:10px; border:0; cursor:pointer; }
  .primary { background:var(--accent); color:#fff; }
  .ghost { background:transparent; border:1px solid var(--line); color:var(--ink); }
  .recording { background:var(--miss); color:#fff; border-color:var(--miss); }
  select { font-size:14px; padding:6px; border-radius:8px; background:var(--bg); color:var(--ink); border:1px solid var(--line); }
  .examples { margin-top:12px; color:var(--muted); font-size:14px; }
  .examples a { color:var(--accent); cursor:pointer; margin-right:12px; text-decoration:none; }
  .grid { display:grid; grid-template-columns: 1fr 1fr; gap:16px; }
  @media (max-width: 760px) { .grid { grid-template-columns: 1fr; } }
  .label { font-size:12px; letter-spacing:.06em; text-transform:uppercase; color:var(--muted); margin-bottom:6px; }
  .gloss { font-size:24px; font-weight:600; word-break:break-word; }
  .chip { display:inline-block; padding:2px 10px; margin:4px 4px 0 0; border-radius:999px; font-size:14px; color:#fff; }
  .sign { background:var(--sign); } .fingerspell { background:var(--spell); } .missing { background:var(--miss); }
  img { width:100%; max-width:420px; border-radius:12px; display:block; margin:0 auto; }
  .status { color:var(--muted); min-height:1.5em; }
  .legend { font-size:13px; color:var(--muted); margin-top:10px; }
</style></head>
<body><main>
  <h1>Speech / Text &rarr; ASL</h1>
  <p class="sub">Low-fidelity prototype &middot; English &rarr; ASL gloss (our Transformer) &rarr; sign poses &rarr; animation</p>

  <div class="card">
    <div class="row">
      <input id="text" type="text" placeholder="Type an English sentence..." value="My name is Cheryl.">
      <button id="mic" class="ghost" title="Speak instead of typing">&#127908; Speak</button>
      <button id="go" class="primary">Translate</button>
    </div>
    <div class="examples">Try:
      <a>Where is the library?</a><a>The children went to school yesterday.</a><a>I like pizza.</a>
    </div>
    <div class="row" style="margin-top:12px; align-items:center; gap:12px">
      <label class="examples" style="margin:0">Glosser
        <select id="glosser"><option value="model">Our model</option><option value="rules">Rule baseline</option></select>
      </label>
      <span id="status" class="status"></span>
    </div>
  </div>

  <div class="grid">
    <div class="card">
      <div class="label">Heard / typed</div><div id="heard">&mdash;</div>
      <div class="label" style="margin-top:16px">ASL gloss</div><div id="gloss" class="gloss">&mdash;</div>
      <div class="label" style="margin-top:16px">Lexicon coverage</div><div id="coverage">&mdash;</div>
      <div class="legend"><span class="chip sign">sign</span><span class="chip fingerspell">fingerspelled</span><span class="chip missing">missing</span></div>
    </div>
    <div class="card"><div class="label">Signing</div><div id="video" class="status">The animation appears here.</div></div>
  </div>
</main>
<script>
const $ = id => document.getElementById(id);
async function translate() {
  const text = $("text").value.trim();
  if (!text) return;
  $("status").textContent = "Translating...";
  $("go").disabled = true;
  try {
    const res = await fetch("/translate", { method: "POST", headers: { "Content-Type": "application/json" },
                                            body: JSON.stringify({ text, glosser: $("glosser").value }) });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
    $("heard").textContent = data.text;
    $("gloss").textContent = data.gloss || "(empty)";
    $("coverage").innerHTML = "";
    for (const [g, kind] of data.coverage) {
      const chip = document.createElement("span");
      chip.className = "chip " + kind; chip.textContent = g; $("coverage").appendChild(chip);
    }
    $("video").innerHTML = data.gif ? `<img alt="Signing animation" src="data:image/gif;base64,${data.gif}">`
                                    : "Nothing to sign: no gloss matched the lexicon.";
    $("status").textContent = "";
  } catch (e) {
    $("status").textContent = "Error: " + e.message;
  } finally {
    $("go").disabled = false;
  }
}
$("go").onclick = translate;
$("text").addEventListener("keydown", e => { if (e.key === "Enter") translate(); });
document.querySelectorAll(".examples a").forEach(a => a.onclick = () => { $("text").value = a.textContent; translate(); });

const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
if (!Recognition) { $("mic").disabled = true; $("mic").title = "Speech input needs Chrome, Edge or Safari"; }
else {
  const rec = new Recognition(); rec.lang = "en-US"; rec.interimResults = true;
  rec.onresult = e => { $("text").value = Array.from(e.results).map(r => r[0].transcript).join(""); };
  rec.onend = () => { $("mic").classList.remove("recording"); $("mic").innerHTML = "&#127908; Speak"; translate(); };
  rec.onerror = e => { $("status").textContent = "Microphone: " + e.error; };
  $("mic").onclick = () => { $("mic").classList.add("recording"); $("mic").textContent = "Listening..."; rec.start(); };
}
</script></body></html>
"""


class DemoApp:
    def __init__(self, model_dir=None, lexicons=None):
        from s2a.gloss_to_pose.lexicon import DEFAULT_LEXICON, Lexicon

        self.model_dir = model_dir
        self.lexicon = Lexicon([*(lexicons or []), DEFAULT_LEXICON])
        self.glossers = {}
        self.lock = threading.Lock()

    def glosser(self, name: str):
        from s2a.text_to_gloss import load_glosser

        if name not in self.glossers:
            self.glossers[name] = load_glosser(name, self.model_dir, beam=5)
        return self.glossers[name]

    def translate(self, text: str, glosser: str = "model") -> dict:
        from s2a.gloss_to_pose import gloss_to_pose
        from s2a.pose_to_video import write_video

        with self.lock:  # torch + OpenCV are not worth parallelizing for a local demo
            gloss = self.glosser(glosser).translate(text)
            pose, matches = gloss_to_pose(gloss, self.lexicon)
            gif = None
            if len(pose):
                with tempfile.TemporaryDirectory() as tmp:
                    path = write_video(pose, Path(tmp) / "out.gif", size=420)
                    gif = base64.b64encode(path.read_bytes()).decode()
        return {"text": text, "gloss": gloss, "coverage": [[m.gloss, m.kind] for m in matches], "gif": gif}


def make_handler(app: DemoApp):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes, content_type: str):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/translate":
                return self._send(404, b"not found", "text/plain")
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                text = str(req.get("text", "")).strip()[:500]
                if not text:
                    raise ValueError("empty text")
                result = app.translate(text, "rules" if req.get("glosser") == "rules" else "model")
                self._send(200, json.dumps(result).encode(), "application/json")
            except Exception as e:  # report errors to the page instead of dropping the connection
                self._send(400, json.dumps({"error": str(e)}).encode(), "application/json")

        def log_message(self, *args):
            pass

    return Handler


def serve(port: int = 8000, model_dir=None, lexicons=None, open_browser: bool = True) -> None:
    app = DemoApp(model_dir, lexicons)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app))
    url = f"http://localhost:{port}"
    print(f"S2A demo running at {url}  (press Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.5, webbrowser.open, [url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
