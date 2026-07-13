"""ui.py — The single-page HTML/CSS/JS shell for the fleet web demo.

Kept out of :mod:`code.apps.fleet_web.app` so the Flask module stays small and
readable. The page is a self-contained dark-theme dashboard: a live MJPEG BEV on
the left; a mission banner, four robot status chips, a colour-coded auto-scroll
comms transcript and a command box on the right. It talks to three endpoints —
``GET /stream`` (image), ``GET /state?after=<id>`` (poll every ~500 ms) and
``POST /command`` — with no third-party JS.
"""

from __future__ import annotations

PAGE = r"""
<!-- fleet_web single-page UI -->
<style>
  :root {
    --bg:#0d1117; --panel:#161b22; --panel2:#1c2230; --line:#2a3140;
    --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; --good:#3fb950;
    --bad:#f85149; --warn:#e3b341;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }
  header { display:flex; align-items:center; gap:14px; padding:12px 20px;
    border-bottom:1px solid var(--line); background:var(--panel); }
  header h1 { font-size:17px; margin:0; letter-spacing:.3px; font-weight:600; }
  header .sub { color:var(--muted); font-size:12.5px; }
  #status { margin-left:auto; font-size:12.5px; color:var(--muted);
    background:var(--panel2); border:1px solid var(--line); padding:6px 12px;
    border-radius:20px; max-width:46%; overflow:hidden; text-overflow:ellipsis;
    white-space:nowrap; }
  main { display:grid; grid-template-columns:minmax(0,1.35fr) minmax(340px,1fr);
    gap:16px; padding:16px; align-items:start; }
  @media (max-width:980px){ main{ grid-template-columns:1fr; } }
  .card { background:var(--panel); border:1px solid var(--line);
    border-radius:12px; overflow:hidden; }
  .card h2 { font-size:12px; text-transform:uppercase; letter-spacing:1px;
    color:var(--muted); margin:0; padding:11px 14px; border-bottom:1px solid var(--line); }
  #stage { position:relative; background:#05070a; }
  #bev { display:block; width:100%; height:auto; }
  #banner { position:absolute; left:12px; top:12px; right:12px; display:flex;
    gap:10px; align-items:center; font-size:13px; }
  #phase { background:rgba(13,17,23,.82); border:1px solid var(--line);
    padding:6px 12px; border-radius:8px; font-weight:600; }
  #task { background:rgba(13,17,23,.72); border:1px solid var(--line);
    padding:6px 12px; border-radius:8px; color:var(--muted); }
  .chips { display:grid; grid-template-columns:1fr 1fr; gap:10px; padding:12px; }
  .chip { background:var(--panel2); border:1px solid var(--line);
    border-radius:10px; padding:10px 12px; }
  .chip .top { display:flex; align-items:center; gap:8px; }
  .dot { width:11px; height:11px; border-radius:50%; flex:0 0 auto;
    box-shadow:0 0 0 3px rgba(255,255,255,.05); }
  .chip .nm { font-weight:600; font-size:14px; }
  .chip .st { margin-left:auto; font-size:11px; color:var(--muted); }
  .chip .st.busy { color:var(--warn); }
  .chip .tk { color:var(--muted); font-size:12px; margin-top:6px;
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  #log { height:300px; overflow-y:auto; padding:10px 12px; font-size:13px;
    line-height:1.5; }
  .msg { margin:3px 0; }
  .msg .who { font-weight:600; }
  .msg.user { color:var(--accent); }
  .msg.system { color:var(--muted); font-style:italic; }
  .msg .body { color:var(--text); }
  .cmd { display:flex; gap:8px; padding:12px; border-top:1px solid var(--line); }
  #text { flex:1; background:var(--panel2); border:1px solid var(--line);
    color:var(--text); border-radius:8px; padding:10px 12px; font-size:14px; }
  #text:focus { outline:none; border-color:var(--accent); }
  button { cursor:pointer; border:1px solid var(--line); border-radius:8px;
    background:var(--panel2); color:var(--text); font-size:13px; padding:10px 14px; }
  #send { background:var(--accent); color:#04121f; border-color:var(--accent);
    font-weight:600; }
  button:hover { filter:brightness(1.12); }
  .examples { display:flex; flex-wrap:wrap; gap:8px; padding:0 12px 12px; }
  .examples button { font-size:12px; color:var(--muted); padding:7px 11px; }
</style>

<header>
  <h1>Warehouse Fleet — Live Ops</h1>
  <span class="sub">four G1 robots · type an order and watch</span>
  <span id="status">connecting…</span>
</header>

<main>
  <section class="card">
    <h2>Bird's-eye view</h2>
    <div id="stage">
      <img id="bev" src="/stream" alt="fleet bird's-eye stream"/>
      <div id="banner">
        <span id="phase">STANDING BY</span>
        <span id="task"></span>
      </div>
    </div>
  </section>

  <section>
    <div class="card" style="margin-bottom:16px;">
      <h2>Fleet status</h2>
      <div class="chips" id="chips"></div>
    </div>
    <div class="card">
      <h2>Comms transcript</h2>
      <div id="log"></div>
      <div class="cmd">
        <input id="text" placeholder="e.g. Alpha, fetch the red cube to the delivery pad"
               autocomplete="off"/>
        <button id="send">Send</button>
      </div>
      <div class="examples" id="examples"></div>
    </div>
  </section>
</main>

<script>
const EXAMPLES = {{ examples|tojson }};
const ACCENTS  = {{ accents|tojson }};
let lastId = 0;

function el(t, cls, txt){ const e=document.createElement(t); if(cls)e.className=cls;
  if(txt!=null)e.textContent=txt; return e; }

function renderChips(robots){
  const box = document.getElementById('chips'); box.innerHTML='';
  robots.forEach(r=>{
    const c = el('div','chip');
    const top = el('div','top');
    const dot = el('span','dot'); dot.style.background = r.color;
    top.appendChild(dot);
    top.appendChild(el('span','nm', r.name));
    const st = el('span', 'st'+(r.busy?' busy':''), r.state + (r.dist?(' · '+r.dist):''));
    top.appendChild(st);
    c.appendChild(top);
    c.appendChild(el('div','tk', r.task));
    box.appendChild(c);
  });
}

function appendLines(lines){
  const log = document.getElementById('log');
  const atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 40;
  lines.forEach(m=>{
    const row = el('div','msg '+m.kind);
    const who = el('span','who', (m.sender==='you'?'You':m.sender)+': ');
    if(ACCENTS[m.sender]) who.style.color = ACCENTS[m.sender];
    row.appendChild(who);
    row.appendChild(el('span','body', m.text));
    log.appendChild(row);
    lastId = Math.max(lastId, m.id);
  });
  if(atBottom) log.scrollTop = log.scrollHeight;
}

async function poll(){
  try{
    const r = await fetch('/state?after='+lastId);
    const s = await r.json();
    document.getElementById('status').textContent = s.status || '';
    renderChips(s.robots||[]);
    document.getElementById('phase').textContent = (s.mission&&s.mission.phase)||'STANDING BY';
    const t = document.getElementById('task');
    t.textContent = (s.mission&&s.mission.target)? ('fetch: '+s.mission.target) : '';
    if(s.transcript && s.transcript.length) appendLines(s.transcript);
  }catch(e){ document.getElementById('status').textContent='reconnecting…'; }
}

async function send(text){
  if(!text.trim()) return;
  document.getElementById('text').value='';
  try{
    const r = await fetch('/command',{method:'POST',
      headers:{'Content-Type':'application/json'}, body:JSON.stringify({text})});
    const j = await r.json();
    if(!j.ok){ appendLines([{id:++lastId?lastId:1, sender:'system', kind:'system',
      text:j.error, recipient:''}]); }
  }catch(e){}
  poll();
}

document.getElementById('send').onclick = ()=> send(document.getElementById('text').value);
document.getElementById('text').addEventListener('keydown', e=>{
  if(e.key==='Enter') send(e.target.value); });

const exBox = document.getElementById('examples');
EXAMPLES.forEach(x=>{ const b=el('button',null,x.label);
  b.onclick=()=>send(x.text); exBox.appendChild(b); });

poll();
setInterval(poll, 500);
</script>
"""


def render_examples_accents():
    """Return ``(examples, accents)`` for injection into the page template."""
    from code.apps.fleet_web.commands import example_commands
    from code.apps.fleet_web.status import ACCENT_HEX

    accents = dict(ACCENT_HEX)
    accents["you"] = "#58a6ff"
    accents["allocator"] = "#3fb950"
    return example_commands(), accents
