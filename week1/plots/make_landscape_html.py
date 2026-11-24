"""Build figures/landscape_3d.html: an interactive, self-contained 3D explorer.

Reads results/landscape.json (experiments/04_landscape.py) and, if present,
results/variance_sweep.csv for the gradient-variance read-outs. Plotly.js is
loaded from cdnjs; everything else is inline, so the file can be opened
locally, embedded in slides, or hosted on GitHub Pages.
"""
import csv
import json

from bp.data import results_dir, figures_dir
from bp.style import COLORS

data = json.loads((results_dir(1) / "landscape.json").read_text())

var = {}
sweep = results_dir(1) / "variance_sweep.csv"
if sweep.exists():
    for r in csv.DictReader(sweep.open()):
        if r["depth"] == "log":
            var.setdefault(r["n"], {})[r["cost"]] = float(r["var_k"])
data["var"] = var

TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Barren Plateau Explorer</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.34.0/plotly.min.js"></script>
<style>
:root{
  --bg:#f7f6f3; --panel:#ffffff; --ink:#1d1d1f; --muted:#5f6368; --line:#e3e1dc;
  --global:__GLOBAL__; --local:__LOCAL__; --accent-ink:#ffffff;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#121417; --panel:#1b1e23; --ink:#ecebe8; --muted:#a3a7ad; --line:#2c3037;
  }
}
:root[data-theme="dark"]{ --bg:#121417; --panel:#1b1e23; --ink:#ecebe8; --muted:#a3a7ad; --line:#2c3037; }
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1240px;margin:0 auto;padding:28px 16px 40px}
h1{font-size:28px;line-height:1.15;margin:0 0 6px;letter-spacing:-.01em}
.lede{color:var(--muted);max-width:760px;margin:0 0 22px}
.lede b.g{color:var(--global)} .lede b.l{color:var(--local)}
.controls{display:flex;flex-wrap:wrap;gap:14px 22px;align-items:center;
  background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin-bottom:16px}
.ctl{display:flex;align-items:center;gap:10px}
.ctl label{font-size:13px;color:var(--muted)}
input[type=range]{width:220px;accent-color:var(--ink)}
.nval{font-variant-numeric:tabular-nums;font-weight:600;min-width:7.5em}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:8px;overflow:hidden}
.seg button,.btn{font:inherit;font-size:13px;border:0;background:transparent;color:var(--ink);
  padding:7px 12px;cursor:pointer}
.seg button[aria-pressed=true]{background:var(--ink);color:var(--bg)}
.btn{border:1px solid var(--line);border-radius:8px}
.btn[aria-pressed=true]{background:var(--ink);color:var(--bg)}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media (max-width:820px){.grid{grid-template-columns:1fr} input[type=range]{width:160px}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 14px 10px;min-width:0}
.card h2{font-size:16px;margin:0;display:flex;align-items:center;gap:8px}
.dot{width:10px;height:10px;border-radius:50%}
.card .sub{color:var(--muted);font-size:13px;margin:2px 0 6px}
.plot{height:440px;width:100%}
.stats{display:flex;gap:10px;flex-wrap:wrap;margin-top:4px}
.stat{flex:1 1 140px;border-top:1px solid var(--line);padding-top:8px}
.stat .k{font-size:12px;color:var(--muted)}
.stat .v{font-size:20px;font-weight:600;font-variant-numeric:tabular-nums}
.note{color:var(--muted);font-size:13px;margin-top:18px;max-width:900px}
</style>
</head>
<body>
<main>
  <h1>Barren Plateau Explorer</h1>
  <p class="lede">One circuit, one set of parameters, two ways to score it. As qubits are added,
  the <b class="g">global cost</b> flattens into a plateau with a single narrow hole,
  while the <b class="l">local cost</b> keeps slopes that a gradient can follow.</p>

  <div class="controls" role="group" aria-label="Controls">
    <div class="ctl">
      <label for="n">Qubits</label>
      <input id="n" type="range" min="0" max="0" step="1" value="0">
      <span class="nval" id="nval"></span>
    </div>
    <div class="ctl">
      <label>Slice through</label>
      <div class="seg" id="mode">
        <button data-mode="solution" aria-pressed="true">the solution</button>
        <button data-mode="random" aria-pressed="false">a random start</button>
      </div>
    </div>
    <div class="ctl">
      <button class="btn" id="play" aria-pressed="false">&#9654; Play n</button>
      <button class="btn" id="spin" aria-pressed="false">&#8635; Rotate</button>
    </div>
  </div>

  <div class="grid">
    <section class="card">
      <h2><span class="dot" style="background:var(--global)"></span><span>Global cost C<sub>G</sub> = 1 &minus; P(0&hellip;0)</span></h2>
      <div class="sub">Rewards only the exact all-zeros outcome</div>
      <div class="plot" id="pg"></div>
      <div class="stats">
        <div class="stat"><div class="k">Spread of C on this slice (std)</div><div class="v" id="sg"></div></div>
        <div class="stat"><div class="k">Var[&part;C/&part;&theta;] at random start</div><div class="v" id="vg"></div></div>
      </div>
    </section>
    <section class="card">
      <h2><span class="dot" style="background:var(--local)"></span><span>Local cost C<sub>L</sub> = 1 &minus; (1/n) &Sigma;<sub>j</sub> P(q<sub>j</sub>=0)</span></h2>
      <div class="sub">Gives partial credit qubit by qubit</div>
      <div class="plot" id="pl"></div>
      <div class="stats">
        <div class="stat"><div class="k">Spread of C on this slice (std)</div><div class="v" id="sl"></div></div>
        <div class="stat"><div class="k">Var[&part;C/&part;&theta;] at random start</div><div class="v" id="vl"></div></div>
      </div>
    </section>
  </div>

  <p class="note">Each surface is a 2D slice of a landscape with 2nL parameters, taken along two random
  orthogonal directions (one unit = about 1 rad per parameter) through either the exact solution
  (&theta; = 0, cost 0) or a uniformly random parameter vector. Both panels use the same slice.
  Hardware-efficient ansatz, L = &lceil;log<sub>2</sub> n&rceil; layers, exact state-vector simulation.
  Slices illustrate; the variance values (200 random draws, from the sweep) are the evidence.</p>
</main>

<script>
const D = __DATA__;
const NS = D.ns, AX = D.axis;
let iN = NS.indexOf(8) >= 0 ? NS.indexOf(8) : 0, mode = "solution", spinning = false, playing = null;
// deep links for slides: landscape_3d.html?n=12&mode=random
const Q = new URLSearchParams(location.search);
if (Q.has("n") && NS.indexOf(+Q.get("n")) >= 0) iN = NS.indexOf(+Q.get("n"));
if (Q.get("mode") === "random") mode = "random";
let camera = {eye:{x:1.55,y:-1.55,z:0.95}, up:{x:0,y:0,z:1}, center:{x:0,y:0,z:-0.08}};

const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const scales = {
  global: [[0,"#5c1402"],[0.35,"__GLOBAL__"],[0.75,"#f4a98f"],[1,"#fdeee8"]],
  local:  [[0,"#082b4c"],[0.35,"__LOCAL__"],[0.75,"#8fbfe3"],[1,"#e6f1fa"]],
};
function std(Z){ let s=0,s2=0,c=0; for(const r of Z) for(const v of r){s+=v;s2+=v*v;c++;} const m=s/c; return Math.sqrt(Math.max(0,s2/c-m*m)); }
function fmtVar(v){ if(v==null) return "–"; const e=Math.floor(Math.log10(v)); return (v/10**e).toFixed(1)+"×10"+sup(e); }
function sup(e){ const m={"-":"⁻","0":"⁰","1":"¹","2":"²","3":"³","4":"⁴","5":"⁵","6":"⁶","7":"⁷","8":"⁸","9":"⁹"}; return String(e).split("").map(c=>m[c]).join(""); }

function layout(){
  const ink = css("--ink"), muted = css("--muted"), line = css("--line");
  const ax = t => ({title:{text:t,font:{size:11,color:muted}}, tickfont:{size:10,color:muted},
                    gridcolor:line, zerolinecolor:line, showbackground:false});
  return {margin:{l:0,r:0,t:0,b:0}, paper_bgcolor:"rgba(0,0,0,0)", font:{color:ink},
    scene:{xaxis:ax("direction 1"), yaxis:ax("direction 2"),
           zaxis:Object.assign(ax("cost"),{range:[0,1.02]}),
           aspectmode:"manual", aspectratio:{x:1,y:1,z:0.62}, camera:camera}, showlegend:false};
}
function traces(cost){
  const Z = D[mode][NS[iN]][cost], mid = (AX.length-1)/2;
  return [{type:"surface", x:AX, y:AX, z:Z, cmin:0, cmax:1, colorscale:scales[cost], showscale:false,
           lighting:{ambient:0.65,diffuse:0.6,specular:0.15,roughness:0.6},
           contours:{z:{show:true,usecolormap:true,project:{z:true},width:1,start:0.05,end:0.95,size:0.1}},
           hovertemplate:"a=%{x:.2f}<br>b=%{y:.2f}<br>C=%{z:.3f}<extra></extra>"},
          {type:"scatter3d", mode:"markers", x:[0], y:[0], z:[Z[mid][mid]+0.015],
           marker:{size:4,color:css("--ink")},
           hovertemplate:(mode==="solution"?"solution":"random start")+"<br>C=%{z:.3f}<extra></extra>"}];
}
function draw(){
  const n = NS[iN];
  document.getElementById("n").value = iN;
  document.getElementById("nval").textContent = `n = ${n}  ·  L = ${D.Ls[iN]}  ·  ${2*n*D.Ls[iN]} params`;
  for (const [id,cost,sid,vid] of [["pg","global","sg","vg"],["pl","local","sl","vl"]]){
    Plotly.react(id, traces(cost), layout(), {displaylogo:false, responsive:true,
      modeBarButtonsToRemove:["toImage","resetCameraLastSave3d"]});
    document.getElementById(sid).textContent = std(D[mode][n][cost]).toFixed(3);
    document.getElementById(vid).textContent = fmtVar((D.var[n]||{})[cost]);
  }
}
// keep both cameras in sync when the user drags one
let syncing = false;
function hook(src, dst){
  document.getElementById(src).on("plotly_relayout", e => {
    if (syncing || !e["scene.camera"]) return;
    camera = e["scene.camera"]; syncing = true;
    Plotly.relayout(dst, {"scene.camera": camera}).then(()=>{syncing=false;});
  });
}
function spinLoop(){
  if (!spinning) return;
  const r = Math.hypot(camera.eye.x, camera.eye.y), a = Math.atan2(camera.eye.y, camera.eye.x) + 0.006;
  camera = {...camera, eye:{x:r*Math.cos(a), y:r*Math.sin(a), z:camera.eye.z}};
  syncing = true;
  Promise.all(["pg","pl"].map(id=>Plotly.relayout(id,{"scene.camera":camera})))
    .then(()=>{syncing=false; requestAnimationFrame(spinLoop);});
}
const nIn = document.getElementById("n");
nIn.max = NS.length-1;
nIn.addEventListener("input", e => { iN = +e.target.value; draw(); });
document.querySelectorAll("#mode button").forEach(b => b.addEventListener("click", () => {
  mode = b.dataset.mode;
  document.querySelectorAll("#mode button").forEach(x => x.setAttribute("aria-pressed", x===b));
  draw();
}));
document.getElementById("play").addEventListener("click", e => {
  const btn = e.currentTarget;
  if (playing){ clearInterval(playing); playing=null; btn.setAttribute("aria-pressed","false"); btn.innerHTML="&#9654; Play n"; return; }
  btn.setAttribute("aria-pressed","true"); btn.innerHTML="&#10073;&#10073; Pause";
  iN = 0; draw();
  playing = setInterval(()=>{ iN = (iN+1) % NS.length; draw(); }, 1700);
});
document.getElementById("spin").addEventListener("click", e => {
  spinning = !spinning; e.currentTarget.setAttribute("aria-pressed", spinning);
  if (spinning) requestAnimationFrame(spinLoop);
});
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);
document.querySelectorAll("#mode button").forEach(x => x.setAttribute("aria-pressed", x.dataset.mode === mode));
draw(); hook("pg","pl"); hook("pl","pg");
</script>
</body>
</html>
"""

html = (TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":")))
        .replace("__GLOBAL__", COLORS["global"]).replace("__LOCAL__", COLORS["local"]))
out = figures_dir(1) / "landscape_3d.html"
out.parent.mkdir(exist_ok=True)
out.write_text(html, encoding="utf-8")
print(f"wrote {out} ({len(html) / 1e3:.0f} kB)")
