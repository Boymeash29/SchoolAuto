#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════╗
║         PDF Annotator — Powered by Ollama            ║
║  Fully offline · No API keys · No internet needed    ║
╚══════════════════════════════════════════════════════╝

SETUP (one time):
    pip install flask pdfplumber requests pymupdf

RUN:
    python annotate.py

Open http://localhost:5000  —  make sure `ollama serve` is running.
"""

import io
import json
import os
import re
import sys
import textwrap
import threading
import uuid
from pathlib import Path

# ── Dependency check ──────────────────────────────────────────────────────────
missing = []
try:
    from flask import (Flask, Response, jsonify, render_template_string,
                       request, send_file, stream_with_context)
except ImportError:
    missing.append("flask")
try:
    import pdfplumber
except ImportError:
    missing.append("pdfplumber")
try:
    import requests as req_lib
except ImportError:
    missing.append("requests")
try:
    import fitz  # pymupdf
except ImportError:
    missing.append("pymupdf")

if missing:
    print(f"\n❌  Missing packages. Run:\n\n    pip install {' '.join(missing)}\n")
    sys.exit(1)

import requests as req_lib

# ─────────────────────────────────────────────────────────────────────────────
OLLAMA_HOST   = os.environ.get("OLLAMA_HOST",  "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB

# Temporary store: job_id -> {"pdf_bytes": bytes, "annotations": dict}
_jobs: dict = {}

# ── Annotation type styling ───────────────────────────────────────────────────
_ACCENT = {
    "definition": (0.12, 0.75, 0.68),
    "question":   (0.90, 0.65, 0.05),
    "reaction":   (0.58, 0.45, 0.92),
    "device":     (0.18, 0.55, 0.92),
    "theme":      (0.92, 0.45, 0.12),
    "notation":   (0.45, 0.50, 0.58),
    "summary":    (0.18, 0.75, 0.38),
}
_FILL = {
    "definition": (0.88, 1.00, 0.98),
    "question":   (1.00, 0.97, 0.82),
    "reaction":   (0.96, 0.92, 1.00),
    "device":     (0.88, 0.95, 1.00),
    "theme":      (1.00, 0.93, 0.86),
    "notation":   (0.94, 0.94, 0.96),
    "summary":    (0.88, 1.00, 0.93),
}
_LABEL = {
    "definition": "DEF",
    "question":   "Q?",
    "reaction":   "RXN",
    "device":     "LIT",
    "theme":      "THM",
    "notation":   "NB",
    "summary":    "SUM",
}

# ─────────────────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>PDF Annotator · Ollama</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:ital,wght@0,400;0,600;1,400&display=swap');
:root{
  --bg:#0f1117;--surface:#181c25;--s2:#1e2330;--border:#2a3040;--b2:#3a4560;
  --text:#e2e8f0;--t2:#8899aa;--accent:#4ade80;--a2:#22c55e;
  --blue:#60a5fa;--yellow:#fbbf24;--red:#f87171;--purple:#a78bfa;--orange:#fb923c;--teal:#2dd4bf;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:'IBM Plex Sans',system-ui,sans-serif;font-size:15px;line-height:1.6;min-height:100vh}
/* topbar */
.topbar{background:var(--surface);border-bottom:1px solid var(--border);padding:.85rem 1.5rem;display:flex;align-items:center;position:sticky;top:0;z-index:100}
.logo{font-family:'IBM Plex Mono',monospace;font-weight:600;font-size:1rem;color:var(--accent)}
.logo span{color:var(--t2);font-weight:400}
.badge{margin-left:auto;display:flex;align-items:center;gap:.4rem;font-size:.8rem;color:var(--t2);font-family:'IBM Plex Mono',monospace}
.dot{width:7px;height:7px;border-radius:50%;background:var(--b2)}
.dot.ok{background:var(--accent);box-shadow:0 0 6px var(--a2)}
.dot.err{background:var(--red)}
.dot.spin{background:var(--yellow);animation:blink .8s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.2}}
/* layout */
.layout{display:grid;grid-template-columns:370px 1fr;min-height:calc(100vh - 50px)}
@media(max-width:780px){.layout{grid-template-columns:1fr}}
/* left panel */
.left{background:var(--surface);border-right:1px solid var(--border);padding:1.5rem;display:flex;flex-direction:column;gap:1.4rem;overflow-y:auto;max-height:calc(100vh - 50px);position:sticky;top:50px}
.slabel{font-size:.68rem;font-weight:600;letter-spacing:.14em;text-transform:uppercase;color:var(--t2);margin-bottom:.5rem}
/* upload */
.dropzone{border:2px dashed var(--b2);border-radius:8px;padding:1.8rem 1rem;text-align:center;cursor:pointer;transition:border-color .2s,background .2s;position:relative;background:var(--s2)}
.dropzone:hover,.dropzone.drag{border-color:var(--accent);background:rgba(74,222,128,.05)}
.dropzone input{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%}
.dropzone h3{font-size:.92rem;font-weight:600;margin:.4rem 0 .2rem}
.dropzone p{font-size:.8rem;color:var(--t2)}
.filebadge{display:none;align-items:center;gap:.5rem;background:rgba(74,222,128,.1);border:1px solid rgba(74,222,128,.3);border-radius:6px;padding:.45rem .75rem;font-size:.82rem;color:var(--accent);font-family:'IBM Plex Mono',monospace;word-break:break-all;margin-top:.7rem}
.filebadge.show{display:flex}
/* inputs */
.field{display:flex;flex-direction:column;gap:.35rem}
.field label{font-size:.75rem;color:var(--t2)}
.field select,.field input[type=text],.field input[type=number]{background:var(--s2);border:1px solid var(--border);color:var(--text);padding:.5rem .75rem;border-radius:6px;font-family:'IBM Plex Mono',monospace;font-size:.86rem;outline:none;transition:border-color .2s;width:100%}
.field select:focus,.field input:focus{border-color:var(--accent)}
textarea{background:var(--s2);border:1px solid var(--border);color:var(--text);padding:.7rem .85rem;border-radius:6px;font-family:'IBM Plex Sans',sans-serif;font-size:.88rem;line-height:1.55;resize:vertical;min-height:150px;width:100%;outline:none;transition:border-color .2s}
textarea:focus{border-color:var(--accent)}
textarea::placeholder{color:var(--t2)}
.row2{display:flex;gap:.7rem}
.row2 .field{flex:1}
/* button */
.btnrun{background:var(--accent);color:#071407;border:none;padding:.8rem 1.4rem;border-radius:8px;font-family:'IBM Plex Sans',sans-serif;font-size:.98rem;font-weight:600;cursor:pointer;width:100%;transition:background .15s,transform .1s;display:flex;align-items:center;justify-content:center;gap:.5rem}
.btnrun:hover:not(:disabled){background:var(--a2);transform:translateY(-1px)}
.btnrun:disabled{opacity:.38;cursor:not-allowed}
.spin16{display:none;width:16px;height:16px;border:2px solid rgba(0,0,0,.2);border-top-color:#071407;border-radius:50%;animation:spin .7s linear infinite;flex-shrink:0}
@keyframes spin{to{transform:rotate(360deg)}}
.spin16.show{display:block}
/* right panel */
.right{padding:1.5rem 2rem;overflow-y:auto;max-height:calc(100vh - 50px)}
.rhead{display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:1.4rem;flex-wrap:wrap;gap:.8rem}
.rtitle{font-size:1.08rem;font-weight:600}
.rmeta{font-size:.78rem;color:var(--t2);font-family:'IBM Plex Mono',monospace;margin-top:.15rem}
/* download button — big and obvious */
.btn-dl{
  display:none;align-items:center;gap:.6rem;
  background:var(--blue);color:#05111f;
  border:none;padding:.65rem 1.3rem;border-radius:8px;
  font-family:'IBM Plex Sans',sans-serif;font-size:.95rem;font-weight:600;
  cursor:pointer;text-decoration:none;
  transition:background .15s,transform .1s;
  box-shadow:0 2px 10px rgba(96,165,250,.35);
}
.btn-dl:hover{background:#93c5fd;transform:translateY(-1px)}
.btn-dl.show{display:inline-flex}
.btn-dl-wrap{margin-bottom:1.2rem}
/* progress */
.progwrap{background:var(--border);border-radius:4px;height:4px;overflow:hidden;margin-bottom:.45rem;display:none}
.progwrap.show{display:block}
.progfill{height:100%;background:linear-gradient(to right,var(--accent),var(--blue));border-radius:4px;transition:width .35s ease;width:0%}
.proglabel{font-size:.78rem;color:var(--t2);font-family:'IBM Plex Mono',monospace;display:none;margin-bottom:.9rem}
.proglabel.show{display:block}
/* error */
.errbanner{background:rgba(248,113,113,.1);border:1px solid rgba(248,113,113,.3);border-radius:8px;padding:.85rem 1rem;color:var(--red);font-size:.88rem;margin-bottom:1rem;display:none}
.errbanner.show{display:block}
/* page sections */
.pgsec{margin-bottom:2rem}
.pghead{display:flex;align-items:center;gap:.7rem;margin-bottom:.9rem}
.pgpill{background:var(--s2);border:1px solid var(--border);color:var(--t2);font-family:'IBM Plex Mono',monospace;font-size:.7rem;padding:.18rem .6rem;border-radius:20px;white-space:nowrap}
.pgline{flex:1;height:1px;background:var(--border)}
/* annotation cards */
.ann{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:.9rem 1rem;margin-bottom:.65rem;border-left:3px solid var(--b2);transition:transform .15s;animation:ci .28s ease forwards;opacity:0}
@keyframes ci{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:translateY(0)}}
.ann:hover{transform:translateX(2px)}
.ann[data-type=definition]{border-left-color:var(--teal)}
.ann[data-type=question]{border-left-color:var(--yellow)}
.ann[data-type=reaction]{border-left-color:var(--purple)}
.ann[data-type=device]{border-left-color:var(--blue)}
.ann[data-type=theme]{border-left-color:var(--orange)}
.ann[data-type=notation]{border-left-color:var(--t2)}
.ann[data-type=summary]{border-left-color:var(--accent)}
.ann-hdr{display:flex;align-items:center;gap:.45rem;flex-wrap:wrap;margin-bottom:.45rem}
.chip{font-size:.66rem;font-weight:600;letter-spacing:.1em;text-transform:uppercase;padding:.15rem .5rem;border-radius:4px;font-family:'IBM Plex Mono',monospace}
.chip-definition{background:rgba(45,212,191,.14);color:var(--teal)}
.chip-question{background:rgba(251,191,36,.14);color:var(--yellow)}
.chip-reaction{background:rgba(167,139,250,.14);color:var(--purple)}
.chip-device{background:rgba(96,165,250,.14);color:var(--blue)}
.chip-theme{background:rgba(251,146,60,.14);color:var(--orange)}
.chip-notation{background:rgba(136,153,170,.14);color:var(--t2)}
.chip-summary{background:rgba(74,222,128,.14);color:var(--accent)}
.tchips{display:flex;gap:.28rem;flex-wrap:wrap}
.tchip{font-size:.62rem;font-weight:600;letter-spacing:.07em;text-transform:uppercase;padding:.1rem .4rem;border-radius:20px;background:rgba(251,146,60,.11);color:var(--orange);border:1px solid rgba(251,146,60,.22)}
.ann-quote{font-style:italic;color:var(--t2);font-size:.86rem;border-left:2px solid var(--border);padding-left:.65rem;margin-bottom:.5rem;line-height:1.45}
.ann-body{font-size:.9rem;line-height:1.62}
/* empty */
.empty{display:flex;flex-direction:column;align-items:center;justify-content:center;height:55vh;text-align:center;gap:.7rem;color:var(--t2)}
.empty-icon{font-size:2.8rem;opacity:.25}
.empty-title{font-size:1rem;font-weight:600;color:var(--text)}
.empty-sub{font-size:.85rem;max-width:280px}
::-webkit-scrollbar{width:5px}::-webkit-scrollbar-track{background:var(--bg)}::-webkit-scrollbar-thumb{background:var(--b2);border-radius:3px}
</style>
</head>
<body>

<div class="topbar">
  <div class="logo">pdf<span>/</span>annotate</div>
  <div class="badge">
    <span class="dot spin" id="sDot"></span>
    <span id="sTxt">Checking Ollama…</span>
  </div>
</div>

<div class="layout">

  <!-- LEFT -->
  <div class="left">

    <div>
      <div class="slabel">1 · Upload PDF</div>
      <div class="dropzone" id="dz">
        <input type="file" id="fileIn" accept=".pdf"/>
        <div style="font-size:2rem;margin-bottom:.4rem">📄</div>
        <h3>Drop PDF here</h3>
        <p>or click to browse</p>
      </div>
      <div class="filebadge" id="fb">📎 <span id="fn"></span></div>
    </div>

    <div>
      <div class="slabel">2 · Model</div>
      <div class="field">
        <select id="modelSel">
          <option value="llama3.2">llama3.2 (recommended)</option>
          <option value="llama3.1">llama3.1</option>
          <option value="llama3">llama3</option>
          <option value="mistral">mistral</option>
          <option value="mistral-nemo">mistral-nemo</option>
          <option value="gemma2">gemma2</option>
          <option value="phi4">phi4</option>
          <option value="deepseek-r1">deepseek-r1</option>
          <option value="custom">custom…</option>
        </select>
      </div>
      <div class="field" id="customWrap" style="display:none;margin-top:.45rem">
        <input type="text" id="customMdl" placeholder="e.g. llama3.2:3b"/>
      </div>
    </div>

    <div>
      <div class="slabel">3 · Page Range <span style="font-weight:400;text-transform:none;letter-spacing:0">(optional)</span></div>
      <div class="row2">
        <div class="field"><label>From</label><input type="number" id="pgFrom" min="1" placeholder="1"/></div>
        <div class="field"><label>To</label><input type="number" id="pgTo" min="1" placeholder="all"/></div>
      </div>
      <div style="font-size:.75rem;color:var(--t2);margin-top:.35rem" id="pgNote"></div>
    </div>

    <div>
      <div class="slabel">4 · Annotation Instructions</div>
      <textarea id="instr" placeholder="Tell the AI what to annotate. Examples:

• Identify literary devices like metaphor and irony
• Define archaic or unusual words
• Note themes: love, death, identity, gender, madness
• Write reactions to key speeches or dialogue
• Ask questions about confusing passages
• Flag important character moments"></textarea>
    </div>

    <button class="btnrun" id="runBtn" disabled>
      <span class="spin16" id="spin"></span>
      <span id="runLbl">✦  Annotate PDF</span>
    </button>

  </div>

  <!-- RIGHT -->
  <div class="right">

    <div class="rhead">
      <div>
        <div class="rtitle">Annotations</div>
        <div class="rmeta" id="rmeta">No file loaded</div>
      </div>
    </div>

    <!-- Download button — prominent, always visible when ready -->
    <div class="btn-dl-wrap">
      <a class="btn-dl" id="dlBtn" href="#" download>
        ⬇&nbsp; Download Annotated PDF
      </a>
    </div>

    <div class="errbanner" id="err"></div>
    <div class="progwrap" id="pw"><div class="progfill" id="pf"></div></div>
    <div class="proglabel" id="pl"></div>

    <div id="results">
      <div class="empty">
        <div class="empty-icon">✍</div>
        <div class="empty-title">Ready to annotate</div>
        <div class="empty-sub">Upload a PDF, enter your instructions, and click Annotate.</div>
      </div>
    </div>

  </div>
</div>

<script>
let pdfFile       = null;
let totalPages    = 0;
let allAnns       = [];   // collected annotations from stream
let currentPage   = null;

// ── Ollama check ──────────────────────────────────────────────────────────────
async function checkOllama() {
  const dot = document.getElementById('sDot');
  const txt = document.getElementById('sTxt');
  dot.className = 'dot spin'; txt.textContent = 'Checking…';
  try {
    const d = await fetch('/api/check_ollama').then(r => r.json());
    if (d.ok) { dot.className = 'dot ok'; txt.textContent = 'Ollama ready'; }
    else throw 0;
  } catch {
    dot.className = 'dot err';
    txt.textContent = 'Ollama offline — run: ollama serve';
  }
}

// ── File drop/select ──────────────────────────────────────────────────────────
const dz = document.getElementById('dz');
dz.addEventListener('dragover',  e => { e.preventDefault(); dz.classList.add('drag'); });
dz.addEventListener('dragleave', () => dz.classList.remove('drag'));
dz.addEventListener('drop', e => {
  e.preventDefault(); dz.classList.remove('drag');
  if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
});
document.getElementById('fileIn').addEventListener('change', e => {
  if (e.target.files[0]) handleFile(e.target.files[0]);
});

async function handleFile(f) {
  if (!f.name.toLowerCase().endsWith('.pdf')) { showErr('Please upload a PDF.'); return; }
  pdfFile = f;
  document.getElementById('fn').textContent = f.name;
  document.getElementById('fb').classList.add('show');
  document.getElementById('rmeta').textContent = f.name;
  const fd = new FormData(); fd.append('pdf', f);
  try {
    const d = await fetch('/api/page_count', { method: 'POST', body: fd }).then(r => r.json());
    totalPages = d.pages || 0;
    document.getElementById('pgNote').textContent = totalPages + ' pages detected';
    document.getElementById('pgTo').placeholder = String(totalPages);
  } catch { document.getElementById('pgNote').textContent = 'Could not read page count'; }
  document.getElementById('runBtn').disabled = false;
}

// ── Model picker ──────────────────────────────────────────────────────────────
document.getElementById('modelSel').addEventListener('change', function () {
  document.getElementById('customWrap').style.display = this.value === 'custom' ? 'block' : 'none';
});
function getModel() {
  const v = document.getElementById('modelSel').value;
  return v === 'custom' ? (document.getElementById('customMdl').value.trim() || 'llama3.2') : v;
}

// ── Run ───────────────────────────────────────────────────────────────────────
document.getElementById('runBtn').addEventListener('click', run);

async function run() {
  if (!pdfFile) { showErr('Upload a PDF first.'); return; }
  const instr = document.getElementById('instr').value.trim();
  if (!instr)  { showErr('Enter annotation instructions.'); return; }

  hideErr();
  allAnns = []; currentPage = null;
  document.getElementById('results').innerHTML = '';
  document.getElementById('dlBtn').classList.remove('show');
  setProgress(0, '');
  document.getElementById('runBtn').disabled = true;
  document.getElementById('spin').classList.add('show');
  document.getElementById('runLbl').textContent = 'Annotating…';

  const pgFrom = parseInt(document.getElementById('pgFrom').value) || 1;
  const pgTo   = parseInt(document.getElementById('pgTo').value)   || totalPages || 9999;

  try {
    // ── Phase 1: stream annotations ──────────────────────────────────────────
    const fd = new FormData();
    fd.append('pdf',          pdfFile);
    fd.append('instructions', instr);
    fd.append('model',        getModel());
    fd.append('page_from',    pgFrom);
    fd.append('page_to',      pgTo);

    const resp = await fetch('/api/annotate', { method: 'POST', body: fd });
    if (!resp.ok) { const e = await resp.json(); throw new Error(e.error || 'Server error'); }

    const reader  = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split('\n'); buf = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const raw = line.slice(6).trim();
        if (!raw || raw === '[DONE]') continue;
        try { handleMsg(JSON.parse(raw)); } catch {}
      }
    }

    // ── Phase 2: build the annotated PDF ─────────────────────────────────────
    if (allAnns.length > 0) {
      setProgress(96, 'Building annotated PDF…');
      await buildAndDownloadPdf(instr, pgFrom, pgTo);
    }

  } catch(e) {
    showErr('Error: ' + e.message);
  } finally {
    document.getElementById('runBtn').disabled = false;
    document.getElementById('spin').classList.remove('show');
    document.getElementById('runLbl').textContent = '✦  Annotate PDF';
    setProgress(100, '');
  }
}

function handleMsg(msg) {
  if      (msg.type === 'progress')   { setProgress(msg.pct, msg.label); }
  else if (msg.type === 'page_start') { startSection(msg.page); }
  else if (msg.type === 'annotation') {
    renderCard(msg.page, msg.data);
    allAnns.push({ page: msg.page, ...msg.data });
  }
  else if (msg.type === 'error')      { showErr(msg.message); }
}

// ── Phase 2: POST annotations + PDF → get annotated PDF back ─────────────────
async function buildAndDownloadPdf(instructions, pgFrom, pgTo) {
  const fd = new FormData();
  fd.append('pdf',          pdfFile);
  fd.append('annotations',  JSON.stringify(allAnns));
  fd.append('page_from',    pgFrom);
  fd.append('page_to',      pgTo);

  const resp = await fetch('/api/build_pdf', { method: 'POST', body: fd });

  if (!resp.ok) {
    const txt = await resp.text();
    throw new Error('PDF build failed: ' + txt);
  }

  // Get the PDF bytes and create a blob URL for direct download
  const blob     = await resp.blob();
  const blobUrl  = URL.createObjectURL(blob);
  const origName = pdfFile.name.replace(/\.pdf$/i, '');
  const dlName   = origName + '_annotated.pdf';

  const btn = document.getElementById('dlBtn');
  btn.href     = blobUrl;
  btn.download = dlName;
  btn.classList.add('show');
  btn.textContent = '⬇  Download Annotated PDF — ' + dlName;
}

// ── UI helpers ────────────────────────────────────────────────────────────────
function startSection(page) {
  if (currentPage === page) return;
  currentPage = page;
  const area = document.getElementById('results');
  const sec  = document.createElement('div');
  sec.className = 'pgsec'; sec.id = 'ps-' + page;
  sec.innerHTML = `<div class="pghead"><span class="pgpill">Page ${page}</span><div class="pgline"></div></div>`;
  area.appendChild(sec);
}

function renderCard(page, d) {
  const sec = document.getElementById('ps-' + page);
  if (!sec) return;
  const type   = (d.type || 'notation').toLowerCase();
  const themes = Array.isArray(d.themes) ? d.themes : [];
  const tHTML  = themes.length
    ? `<div class="tchips">${themes.map(t => `<span class="tchip">${esc(t)}</span>`).join('')}</div>` : '';
  const qHTML  = d.quote ? `<div class="ann-quote">"${esc(d.quote)}"</div>` : '';
  const el     = document.createElement('div');
  el.className = 'ann'; el.dataset.type = type;
  el.innerHTML = `<div class="ann-hdr"><span class="chip chip-${type}">${esc(type)}</span>${tHTML}</div>${qHTML}<div class="ann-body">${esc(d.annotation || '')}</div>`;
  sec.appendChild(el);
  el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function esc(s) {
  return String(s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function showErr(m) {
  const b = document.getElementById('err'); b.textContent = '⚠ ' + m; b.classList.add('show');
}
function hideErr() { document.getElementById('err').classList.remove('show'); }
function setProgress(pct, label) {
  document.getElementById('pw').classList.toggle('show', pct > 0 && pct < 100);
  document.getElementById('pl').classList.toggle('show', !!label && pct < 100);
  document.getElementById('pf').style.width = pct + '%';
  document.getElementById('pl').textContent = label;
}

checkOllama();
setInterval(checkOllama, 15000);
</script>
</body>
</html>
"""


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/check_ollama")
def check_ollama():
    try:
        req_lib.get(OLLAMA_HOST.rstrip("/") + "/api/tags", timeout=4).raise_for_status()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/page_count", methods=["POST"])
def page_count():
    if "pdf" not in request.files:
        return jsonify({"error": "No file"}), 400
    try:
        with pdfplumber.open(io.BytesIO(request.files["pdf"].read())) as pdf:
            return jsonify({"pages": len(pdf.pages)})
    except Exception as e:
        return jsonify({"error": str(e), "pages": 0})


@app.route("/api/annotate", methods=["POST"])
def annotate():
    """Phase 1: stream annotations as SSE."""
    if "pdf" not in request.files:
        return jsonify({"error": "No PDF uploaded"}), 400

    f            = request.files["pdf"]
    instructions = request.form.get("instructions", "").strip()
    model        = request.form.get("model", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    page_from    = max(1, int(request.form.get("page_from", 1) or 1))
    page_to      = int(request.form.get("page_to", 9999) or 9999)
    pdf_bytes    = f.read()

    if not instructions:
        return jsonify({"error": "No instructions provided"}), 400

    def generate():
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                total   = len(pdf.pages)
                p_start = page_from - 1
                p_end   = min(page_to, total)
                pages   = pdf.pages[p_start:p_end]
                n       = len(pages)

                if n == 0:
                    yield _sse({"type": "error", "message": "No pages found in that range."})
                    return

                for i, page in enumerate(pages):
                    pg_num = p_start + i + 1
                    yield _sse({"type": "progress",
                                "pct":   int((i / n) * 92),
                                "label": f"Annotating page {pg_num} of {p_end}…"})
                    yield _sse({"type": "page_start", "page": pg_num})

                    text = (page.extract_text() or "").strip()
                    if not text:
                        yield _sse({"type": "annotation", "page": pg_num, "data": {
                            "type": "notation", "quote": "",
                            "annotation": "No extractable text on this page (may be an image).",
                            "themes": []}})
                        continue

                    if len(text) > 4000:
                        text = text[:4000] + "\n[truncated]"

                    anns = _call_ollama(model, _build_prompt(text, instructions, pg_num))
                    for ann in anns:
                        yield _sse({"type": "annotation", "page": pg_num, "data": ann})

                yield _sse({"type": "progress", "pct": 93, "label": "Annotations complete…"})
                yield "data: [DONE]\n\n"

        except Exception as e:
            yield _sse({"type": "error", "message": str(e)})

    return Response(stream_with_context(generate()),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/build_pdf", methods=["POST"])
def build_pdf():
    """
    Phase 2: receive the original PDF + all annotations as JSON,
    render highlights + margin callout boxes, return the PDF file directly.
    """
    if "pdf" not in request.files:
        return "No PDF", 400

    pdf_bytes   = request.files["pdf"].read()
    orig_name   = request.files["pdf"].filename or "document.pdf"
    ann_json    = request.form.get("annotations", "[]")
    page_from   = max(1, int(request.form.get("page_from", 1) or 1))
    page_to     = int(request.form.get("page_to", 9999) or 9999)

    try:
        raw_anns = json.loads(ann_json)
    except json.JSONDecodeError:
        return "Invalid annotations JSON", 400

    # Group by page
    by_page: dict = {}
    for a in raw_anns:
        pg = a.get("page", 0)
        by_page.setdefault(pg, []).append(a)

    try:
        out_bytes = _render_pdf(pdf_bytes, by_page)
    except Exception as e:
        return f"PDF render error: {e}", 500

    stem     = Path(orig_name).stem
    out_name = f"{stem}_annotated.pdf"

    return send_file(
        io.BytesIO(out_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=out_name,
    )


# ─── PDF Renderer ─────────────────────────────────────────────────────────────

def _render_pdf(pdf_bytes: bytes, by_page: dict) -> bytes:
    """
    Draw highlights + colour-coded margin annotation boxes onto each page.
    Returns the modified PDF as bytes.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    for page_num, anns in by_page.items():
        pg_idx = page_num - 1
        if pg_idx < 0 or pg_idx >= len(doc):
            continue

        page = doc[pg_idx]
        pw   = page.rect.width
        ph   = page.rect.height

        # Margin strip — 23% of width, capped at 180pt
        margin_w = min(pw * 0.23, 180)
        sep_x    = pw - margin_w - 2

        # Separator line
        sh = page.new_shape()
        sh.draw_line(fitz.Point(sep_x, 14), fitz.Point(sep_x, ph - 14))
        sh.finish(color=(0.68, 0.72, 0.80), width=0.5, stroke_opacity=0.45)
        sh.commit()

        # Layout constants
        BOX_X    = sep_x + 3
        BOX_W    = margin_w - 5
        BOX_PAD  = 4.0
        FS_LBL   = 5.8
        FS_QT    = 6.4
        FS_TX    = 6.9
        LH_QT    = FS_QT * 1.32
        LH_TX    = FS_TX * 1.36
        CHARS    = max(int(BOX_W / (FS_TX * 0.52)), 14)

        margin_y = 14.0

        for ann in anns:
            atype  = ann.get("type",       "notation")
            quote  = ann.get("quote",      "").strip()
            text   = ann.get("annotation", "").strip()
            themes = ann.get("themes",     [])
            if not isinstance(themes, list):
                themes = []

            accent = _ACCENT.get(atype, (0.45, 0.50, 0.58))
            fill   = _FILL.get(atype,   (0.95, 0.95, 0.97))
            label  = _LABEL.get(atype,  "ANN")

            # ── Highlight quote on page ───────────────────────────────────────
            if quote and len(quote) >= 5:
                hits = page.search_for(quote)
                for rect in hits[:5]:
                    hl = page.new_shape()
                    hl.draw_rect(rect)
                    hl.finish(fill=fill, color=accent,
                              fill_opacity=0.32, stroke_opacity=0.55, width=0.5)
                    hl.commit()

            # ── Wrap text ─────────────────────────────────────────────────────
            ann_lines   = textwrap.wrap(text, width=CHARS)
            q_short     = (quote[:50] + "…") if len(quote) > 50 else quote
            quote_lines = textwrap.wrap(f'"{q_short}"', width=CHARS) if quote else []
            theme_str   = "  ".join(f"#{t}" for t in themes[:4]) if themes else ""

            box_h = (BOX_PAD * 2
                     + FS_LBL + 2.5
                     + len(quote_lines) * LH_QT + (2 if quote_lines else 0)
                     + len(ann_lines)   * LH_TX
                     + (8 if theme_str else 0))

            # Wrap to top if overflow
            if margin_y + box_h > ph - 12:
                margin_y = 14.0

            box = fitz.Rect(BOX_X, margin_y, BOX_X + BOX_W, margin_y + box_h)

            # Box background
            sh = page.new_shape()
            sh.draw_rect(box)
            sh.finish(fill=fill, color=accent,
                      fill_opacity=0.90, stroke_opacity=0.82, width=0.65)
            sh.commit()

            # Left accent bar
            bar = fitz.Rect(BOX_X, margin_y, BOX_X + 2.8, margin_y + box_h)
            sh  = page.new_shape()
            sh.draw_rect(bar)
            sh.finish(fill=accent, color=accent, fill_opacity=1.0, stroke_opacity=0.0)
            sh.commit()

            # Text rendering
            tx = BOX_X + 5.0
            ty = margin_y + BOX_PAD

            # Label
            page.insert_text(fitz.Point(tx, ty + FS_LBL),
                             label, fontname="helv", fontsize=FS_LBL, color=accent)
            ty += FS_LBL + 2.5

            # Quote
            for ql in quote_lines:
                page.insert_text(fitz.Point(tx, ty + FS_QT),
                                 ql, fontname="helv", fontsize=FS_QT,
                                 color=(0.28, 0.30, 0.40))
                ty += LH_QT
            if quote_lines:
                ty += 2.0

            # Annotation text
            for al in ann_lines:
                if ty + LH_TX > box.y1 - 2:
                    break
                page.insert_text(fitz.Point(tx, ty + FS_TX),
                                 al, fontname="helv", fontsize=FS_TX,
                                 color=(0.08, 0.10, 0.16))
                ty += LH_TX

            # Themes
            if theme_str:
                page.insert_text(fitz.Point(tx, box.y1 - 3.5),
                                 theme_str, fontname="helv",
                                 fontsize=5.0, color=accent)

            margin_y = box.y1 + 3.5

    buf = io.BytesIO()
    doc.save(buf, garbage=4, deflate=True)
    doc.close()
    return buf.getvalue()


# ─── Ollama + prompt helpers ──────────────────────────────────────────────────

def _build_prompt(page_text: str, instructions: str, page_num: int) -> str:
    return (
        "You are a literary annotation assistant. Read the page text and produce annotations as JSON.\n\n"
        f"ANNOTATION INSTRUCTIONS:\n{instructions}\n\n"
        f"PAGE {page_num} TEXT:\n{page_text}\n\n"
        "Return a JSON array. Each item must have EXACTLY these keys:\n"
        '  "type"       : one of: notation, definition, question, reaction, device, theme, summary\n'
        '  "quote"      : a SHORT verbatim phrase from the text (4-8 words). Must appear exactly in text.\n'
        '  "annotation" : your annotation in 1-3 sentences. No newlines inside.\n'
        '  "themes"     : array like ["love","identity"] or []\n\n'
        "Rules:\n"
        "- Produce 3 to 6 annotations per page\n"
        "- The quote must be exact words from the text so they can be highlighted on the page\n"
        "- No apostrophes inside string values (write out: do not instead of don't)\n"
        "- No newlines inside any value\n"
        "- Output ONLY the raw JSON array. No markdown. No code fences. No explanation.\n\n"
        "Example output:\n"
        '[{"type":"definition","quote":"thou art","annotation":"thou art means you are. An archaic second-person form used throughout Shakespeare.","themes":[]},\n'
        '{"type":"theme","quote":"music be the food of love","annotation":"Orsino frames love as a hunger fed by music. This opens the central theme of love as an irresistible appetite.","themes":["love","music"]}]\n'
    )


def _repair_json(raw: str) -> list:
    """Try multiple strategies to parse potentially malformed JSON from the model."""
    # Strategy 1 — direct
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Extract array bounds
    s = raw.find("[")
    e = raw.rfind("]")
    chunk = raw[s:e+1] if s != -1 and e > s else raw

    # Strategies 2-5 on the extracted chunk
    candidates = [
        chunk,
        re.sub(r"```(?:json)?", "", chunk).strip(),
        re.sub(r'(?<=["\w])\n(?=["\w ])', ' ', chunk),
        chunk[:chunk.rfind("}")+1] + "]" if chunk.rfind("}") != -1 else chunk,
    ]
    for c in candidates:
        try:
            return json.loads(c)
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 6 — extract individual objects via regex
    parsed = []
    for obj in re.findall(r'\{[^{}]+\}', chunk, re.DOTALL):
        obj = re.sub(r'\n', ' ', obj)
        for variant in [obj, obj.replace("'", '"')]:
            try:
                parsed.append(json.loads(variant))
                break
            except json.JSONDecodeError:
                pass
    if parsed:
        return parsed

    raise ValueError(f"JSON repair failed. Raw output: {raw[:300]!r}")


def _clean(raw_list: list) -> list:
    valid = {"notation","definition","question","reaction","device","theme","summary"}
    out   = []
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        t = str(item.get("type", "notation")).lower().strip()
        out.append({
            "type":       t if t in valid else "notation",
            "quote":      str(item.get("quote",      "")).replace("\n", " ").strip(),
            "annotation": str(item.get("annotation", "")).replace("\n", " ").strip(),
            "themes":     [str(x) for x in item["themes"]]
                          if isinstance(item.get("themes"), list) else [],
        })
    return out


def _call_ollama(model: str, prompt: str) -> list:
    url  = OLLAMA_HOST.rstrip("/") + "/api/generate"
    data = {"model": model, "prompt": prompt, "stream": False,
            "options": {"temperature": 0.2, "num_predict": 2048}}
    raw  = ""
    try:
        r   = req_lib.post(url, json=data, timeout=180)
        r.raise_for_status()
        raw = r.json().get("response", "").strip()
        if not raw:
            return [{"type":"notation","quote":"","annotation":"Model returned no output.","themes":[]}]
        cleaned = _clean(_repair_json(raw))
        return cleaned or [{"type":"notation","quote":"","annotation":"No annotations produced.","themes":[]}]
    except ValueError:
        snippet = raw[:500] if raw else "(empty)"
        return [{"type":"notation","quote":"","annotation":f"JSON parse failed. Raw: {snippet}","themes":[]}]
    except req_lib.exceptions.ConnectionError:
        raise RuntimeError("Cannot connect to Ollama. Run: ollama serve")
    except req_lib.exceptions.Timeout:
        raise RuntimeError("Ollama timed out. Try a faster/smaller model.")
    except Exception as ex:
        raise RuntimeError(str(ex))


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════╗
║           PDF Annotator — Powered by Ollama              ║
╠══════════════════════════════════════════════════════════╣
║  1. ollama serve                                         ║
║  2. ollama pull llama3.2                                 ║
║  3. pip install flask pdfplumber requests pymupdf        ║
╚══════════════════════════════════════════════════════════╝

  → http://localhost:5000
""")
    def _open_browser():
        import time, webbrowser
        time.sleep(1.2)
        webbrowser.open("http://localhost:5000")

    threading.Thread(target=_open_browser, daemon=True).start()
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
