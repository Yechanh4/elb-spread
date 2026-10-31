/* ELB call-spread simulator — page 1 (ticker-driven simulation). Vanilla JS + D3 7.9.0, no build step.
   All math runs in Python (/api/page1); this file only collects inputs and draws. */
"use strict";

const DEFAULT_STATE = {
  tickers: [{ ticker: "AMD" }, { ticker: "AMZN" }, { ticker: "GOOGL" }],
  globals: {
    V_deposit: 1.5e9, V_own_rental: 3.8e8, r: null, T: 2, rate_shift: 0, G: 756e6, f: 1.0,
    k2_rule: "cost-aware", k2_criterion: "expected", cost_mode: "chain", rt_cost: 0.04, leg_cross: 1.0,
    tax: true, broker_confirmed: null, n_paths: 10000, copula: "gaussian", seed: 0, start: "2026-12-01",
    mortgage_rate: 0.04, remaining_months: 264,
  },
};
const GOAL_PRESETS = [[756e6, "756M · 잔여 대출 전액"], [500e6, "500M · 대출의 2/3"], [300e6, "300M · 40%"],
  [70e6, "70M · 채권 기회비용만"]];
const KEY = "elb.page1.state.v2";   // v2: new default tickers
const S = { state: loadState(), result: null, req: 0, timer: null, densityTab: 0, path: 0, playing: false };

function loadState() {
  try {
    const s = JSON.parse(localStorage.getItem(KEY));
    if (s && s.tickers && s.globals) return { tickers: s.tickers, globals: { ...DEFAULT_STATE.globals, ...s.globals } };
  } catch (e) { /* storage blocked or corrupt: defaults */ }
  return JSON.parse(JSON.stringify(DEFAULT_STATE));
}
function saveState() { try { localStorage.setItem(KEY, JSON.stringify(S.state)); } catch (e) { /* ignore */ } }

// ------------------------------------------------------------------ formatting
const $ = (sel, el = document) => el.querySelector(sel);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const isNum = v => typeof v === "number" && isFinite(v);
const pct = (v, d = 1) => isNum(v) ? (v * 100).toFixed(d) + "%" : "–";
const num = (v, d = 2) => isNum(v) ? v.toLocaleString("en-US", { maximumFractionDigits: d, minimumFractionDigits: d }) : "–";
const int = v => isNum(v) ? Math.round(v).toLocaleString("en-US") : "–";
function won(v) {
  if (!isNum(v)) return "–";
  const a = Math.abs(v), s = v < 0 ? "−" : "";
  if (a >= 1e12) return s + (a / 1e12).toFixed(2) + "조";
  if (a >= 1e8) return s + (a / 1e8).toFixed(a >= 1e10 ? 0 : 1) + "억";
  if (a >= 1e4) return s + Math.round(a / 1e4).toLocaleString() + "만";
  return s + Math.round(a).toLocaleString();
}
const mdl = (txt, tip) => `<span class="mdl"${tip ? ` data-tip="${esc(tip)}"` : ""}>${txt}</span>`;
const mkt = (txt, tip) => `<span${tip ? ` data-tip="${esc(tip)}"` : ""}>${txt}</span>`;
const color = i => `var(--s${(i % 5) + 1})`;
const statusBadge = (st, label) => {
  const cls = { pass: "good", warn: "warn", fail: "bad", "n/a": "" }[st] ?? "";
  const icon = { pass: "✓", warn: "!", fail: "✕", "n/a": "–" }[st] ?? "";
  return `<span class="badge ${cls}">${icon} ${esc(label)}</span>`;
};

// ------------------------------------------------------------------ tooltip
const tipEl = () => $("#tip");
function showTip(ev, html) {
  const t = tipEl(); t.innerHTML = html; t.style.display = "block";
  const x = Math.min(ev.clientX + 14, window.innerWidth - t.offsetWidth - 8);
  const y = Math.min(ev.clientY + 14, window.innerHeight - t.offsetHeight - 8);
  t.style.left = x + "px"; t.style.top = y + "px";
}
function hideTip() { tipEl().style.display = "none"; }
document.addEventListener("mouseover", ev => {
  const el = ev.target.closest("[data-tip]");
  if (el) showTip(ev, esc(el.dataset.tip));
});
document.addEventListener("mouseout", ev => { if (ev.target.closest("[data-tip]")) hideTip(); });

// ------------------------------------------------------------------ api
// Static build (GitHub Pages): the same Python runs in a Pyodide Web Worker instead of a server.
const STATIC = document.documentElement.dataset.static === "1";
let worker = null, wseq = 0;
const wpending = new Map();
function pyWorker() {
  if (worker) return worker;
  worker = new Worker("worker.js");
  worker.onmessage = e => {
    const m = e.data;
    if ("progress" in m) {
      const b = $("#busy");
      if (m.progress) { b.textContent = m.progress + "…"; b.style.display = "block"; }
      else { b.textContent = "계산 중…"; }
      return;
    }
    const p = wpending.get(m.id); if (!p) return;
    wpending.delete(m.id);
    m.ok ? p.resolve(m.data) : p.reject(new Error(m.error));
  };
  worker.onerror = e => { for (const p of wpending.values()) p.reject(new Error("파이썬 엔진 오류: " + (e.message || "로딩 실패"))); wpending.clear(); };
  return worker;
}
async function api(path, body) {
  if (STATIC) {
    return new Promise((resolve, reject) => {
      const id = ++wseq;
      wpending.set(id, { resolve, reject });
      pyWorker().postMessage({ id, path, body });
    });
  }
  const r = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {});
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || j.detail?.error || j.detail || r.statusText);
  return j;
}
// Share links on the static site carry the whole input state in the URL: deflate + base64url.
async function packState(obj) {
  const bytes = new TextEncoder().encode(JSON.stringify(obj));
  let out = bytes, tag = "j";
  if (window.CompressionStream) {
    out = new Uint8Array(await new Response(new Blob([bytes]).stream().pipeThrough(new CompressionStream("deflate-raw"))).arrayBuffer());
    tag = "d";
  }
  let bin = ""; out.forEach(b => (bin += String.fromCharCode(b)));
  return tag + btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}
async function unpackState(z) {
  const tag = z[0], b64 = z.slice(1).replace(/-/g, "+").replace(/_/g, "/");
  const bin = atob(b64 + "===".slice((b64.length + 3) % 4));
  let bytes = Uint8Array.from(bin, c => c.charCodeAt(0));
  if (tag === "d") bytes = new Uint8Array(await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream("deflate-raw"))).arrayBuffer());
  return JSON.parse(new TextDecoder().decode(bytes));
}
function schedule(delay = 500) {
  saveState();
  clearTimeout(S.timer);
  S.timer = setTimeout(compute, delay);
}
async function compute() {
  const id = ++S.req;
  $("#busy").style.display = "block";
  try {
    const res = await api("/api/page1", S.state);
    if (id !== S.req) return;                     // a newer request is in flight: drop this answer
    S.result = res;
    if ($("#results")) renderResults();          // the user may have switched pages meanwhile
  } catch (e) {
    if (id === S.req && $("#results")) $("#results").innerHTML = `<div class="panel banner">계산 실패: ${esc(e.message)}</div>`;
  } finally {
    if (id === S.req) $("#busy").style.display = "none";
  }
}

// ------------------------------------------------------------------ router
const ROUTES = { sim: renderSim, optimizer: renderLater("2 · 역최적화", "4단계"), timeline: renderLater("3 · 8년 타임라인", "5단계") };  // page2.js / page3.js replace these
window.ROUTES = ROUTES;
async function route() {
  const [r0, qs] = (location.hash.replace("#/", "") || "sim").split("?");
  const r = ROUTES[r0] ? r0 : "sim";
  const params = new URLSearchParams(qs || "");
  const sid = params.get("s"), z = params.get("z");
  if (sid || z) {                              // shared link: load that state, then drop it from the URL
    try {
      const got = z ? await unpackState(z) : await api("/api/share/" + encodeURIComponent(sid));
      if (got.page === "sim") { S.state = got.state; S.result = null; saveState(); }
      else if (got.page === "optimizer") { P2.state = got.state; P2.result = null; p2save(); }
      else if (got.page === "timeline") { P3.state = got.state; P3.seed = JSON.parse(JSON.stringify(got.state)); P3.result = null; p3save(); }
      history.replaceState(null, "", "#/" + got.page);
      return route();
    } catch (e) { alert("공유 링크를 열지 못함: " + e.message); }
  }
  document.querySelectorAll("nav.routes a").forEach(a => a.classList.toggle("active", a.dataset.route === r));
  ROUTES[r]();
}
$("#share").onclick = async () => {
  const page = (location.hash.replace("#/", "") || "sim").split("?")[0];
  const state = { sim: () => S.state, optimizer: () => P2.state, timeline: () => P3.state }[page]?.() || S.state;
  try {
    const pg = ROUTES[page] ? page : "sim";
    const url = STATIC ? `${location.origin}${location.pathname}#/${pg}?z=${await packState({ page: pg, state })}`
      : `${location.origin}${location.pathname}#/${pg}?s=${(await api("/api/share", { page: pg, state })).id}`;
    try { await navigator.clipboard.writeText(url); alert("링크를 복사했어요:\n" + url); } catch (e) { prompt("이 링크를 복사하세요", url); }
  } catch (e) { alert("공유 실패: " + e.message); }
};
function renderLater(title, step) {
  return () => { $("#view").innerHTML = `<div class="layout" style="grid-template-columns:1fr"><div class="panel empty">${title}는 ${step}에서 구현됩니다.</div></div>`; };
}
window.addEventListener("hashchange", route);

// ------------------------------------------------------------------ page 1 skeleton
function renderSim() {
  $("#view").innerHTML = `
  <div class="layout">
    <aside class="rail">
      <div class="card"><h2>종목 (1–5개, 권장 3–5)</h2>
        <div class="search"><input id="q" placeholder="티커 검색 · 예: NVDA" autocomplete="off"><div class="suggest" id="sug" hidden></div></div>
        <div class="sub" style="margin-top:6px">섹터당 최대 2개. 종목을 늘리면 분포가 평균 쪽으로 모이는데 평균이 목표보다 낮아서 대체로 불리함.</div>
      </div>
      <div id="tcards"></div>
      <div class="card" id="gcard"></div>
    </aside>
    <main class="canvas" id="results"><div class="panel empty">${STATIC ? "브라우저 안에서 파이썬 엔진을 준비 중… (처음 1회 10–30초, 이후 빠름)" : "데이터 받는 중… (처음 받는 종목은 수 초 걸림)"}</div></main>
  </div>`;
  bindSearch();
  renderTickerCards();
  renderGlobals();
  compute();
}

function bindSearch() {
  const q = $("#q"), sug = $("#sug");
  let t = null, items = [], on = -1;
  const add = sym => {
    sym = sym.trim().toUpperCase();
    if (!sym) return;
    if (S.state.tickers.some(x => x.ticker === sym)) return;
    if (S.state.tickers.length >= 5) { alert("최대 5종목"); return; }
    S.state.tickers.push({ ticker: sym });
    q.value = ""; sug.hidden = true;
    renderTickerCards(); schedule(0);
  };
  const draw = () => {
    sug.innerHTML = items.map((it, i) => `<div class="${i === on ? "on" : ""}" data-s="${esc(it.symbol)}"><b>${esc(it.symbol)}</b> <span class="sub">${esc(it.name)}</span></div>`).join("");
    sug.hidden = !items.length;
  };
  q.addEventListener("input", () => {
    clearTimeout(t);
    t = setTimeout(async () => { try { items = await api("/api/search?q=" + encodeURIComponent(q.value)); } catch (e) { items = []; } on = -1; draw(); }, 250);
  });
  q.addEventListener("keydown", e => {
    if (e.key === "ArrowDown") { on = Math.min(on + 1, items.length - 1); draw(); e.preventDefault(); }
    else if (e.key === "ArrowUp") { on = Math.max(on - 1, 0); draw(); e.preventDefault(); }
    else if (e.key === "Enter") { add(on >= 0 && items[on] ? items[on].symbol : q.value); }
    else if (e.key === "Escape") { sug.hidden = true; }
  });
  sug.addEventListener("mousedown", e => { const d = e.target.closest("[data-s]"); if (d) add(d.dataset.s); });
  q.addEventListener("blur", () => setTimeout(() => (sug.hidden = true), 150));
}

// ------------------------------------------------------------------ ticker cards (inputs)
function tField(i, key, label, attrs, tip) {
  const v = S.state.tickers[i][key];
  return `<div class="field"><label data-tip="${esc(tip)}">${label}</label><input data-t="${i}" data-k="${key}" ${attrs} value="${v ?? ""}"></div>`;
}
function renderTickerCards() {
  const el = $("#tcards");
  el.innerHTML = S.state.tickers.map((t, i) => `
    <div class="card tcard" style="--c:${color(i)}" id="tc${i}">
      <div class="head"><b>${esc(t.ticker)}</b><span class="sub" id="tname${i}"></span><button class="x" data-del="${i}" title="삭제">×</button></div>
      <div class="facts" id="tfacts${i}"><div><span>현물</span> –</div></div>
      <div class="field"><label data-tip="상장 월물 중 60일 넘게 남은 3개. 기본은 q 검증을 통과한 것 중 3개월에 가장 가까운 것.">만기</label><select data-t="${i}" data-k="expiry" id="texp${i}"><option value="">자동</option></select></div>
      ${tField(i, "sigma_hat", "σ̂ (연, 소수)", 'type="number" step="0.01" min="0.05" max="3" placeholder="HV 1y"', "리서치 변동성 견해. 비우면 HV 1y 자리표시(합성 아님, 근거 없음 표시). Edge = σ̂ − ATM IV > 0 이어야 시뮬레이션에 들어감.")}
      ${tField(i, "mu_hat", "μ̂ 드리프트 (연)", 'type="number" step="0.01" placeholder="견해 없음"', "방향성 견해. 옵션 가격엔 안 들어가지만(BS에서 상쇄) 손익엔 들어감 — 기여분을 따로 표시.")}
      <div class="field"><label data-tip="σ_used = conf·σ̂ + (1−conf)·IV. 확신이 낮을수록 시장으로 수축.">확신도 <span id="tconfv${i}">${t.conf ?? 0.7}</span></label><input type="range" min="0" max="1" step="0.05" data-t="${i}" data-k="conf" value="${t.conf ?? 0.7}"></div>
      ${tField(i, "weight", "비중 (비우면 균등)", 'type="number" step="0.05" min="0" max="1" placeholder="균등"', "옵션 예산 중 이 종목 몫. 합이 1이 아니면 자동 정규화.")}
      <div id="tedge${i}"></div>
      <details ${t.catalysts?.length ? "open" : ""}><summary>카탈리스트 계산기</summary>
        <div class="sub">총분산 = σ_base²·T + Σ move² → σ̂ = √(총분산/T), T=0.25</div>
        ${tField(i, "catalyst_base", "σ_base", 'type="number" step="0.01" placeholder="HV 1y"', "이벤트 없는 평상시 변동성. 비우면 HV 1y.")}
        <div id="tcats${i}">${(t.catalysts || []).map((c, j) => catRow(i, j, c)).join("")}</div>
        <button class="btn" data-addcat="${i}" style="margin-top:4px">+ 이벤트</button>
        <div class="field"><label>계산기 σ̂ 사용</label><input type="checkbox" data-t="${i}" data-k="use_catalyst" ${t.use_catalyst ? "checked" : ""}></div>
        <div id="tcatout${i}" class="sub"></div>
      </details>
      <div id="tqfail${i}"></div>
    </div>`).join("");
  el.querySelectorAll("[data-del]").forEach(b => b.onclick = () => { S.state.tickers.splice(+b.dataset.del, 1); renderTickerCards(); schedule(0); });
  el.querySelectorAll("[data-addcat]").forEach(b => b.onclick = () => {
    const t = S.state.tickers[+b.dataset.addcat]; (t.catalysts ||= []).push({ name: "이벤트", date: "", move: 0.08 }); renderTickerCards();
  });
  el.querySelectorAll("input[data-t], select[data-t]").forEach(inp => inp.addEventListener(inp.type === "range" ? "input" : "change", onTickerInput));
  el.querySelectorAll("[data-cat]").forEach(inp => inp.addEventListener("change", onCatInput));
  el.querySelectorAll("[data-delcat]").forEach(b => b.onclick = () => {
    const [i, j] = b.dataset.delcat.split(",").map(Number); S.state.tickers[i].catalysts.splice(j, 1); renderTickerCards(); schedule();
  });
  if (S.result) updateCardFacts();
}
function catRow(i, j, c) {
  return `<div class="catrow"><input data-cat="${i},${j},name" value="${esc(c.name)}" placeholder="이름">
    <input data-cat="${i},${j},date" type="date" value="${esc(c.date)}"><input data-cat="${i},${j},move" type="number" step="0.01" value="${c.move}" title="예상 |변동| (소수, 0.08 = 8%)">
    <button class="x btn" data-delcat="${i},${j}" style="padding:0 4px">×</button></div>`;
}
function onTickerInput(e) {
  const i = +e.target.dataset.t, k = e.target.dataset.k, t = S.state.tickers[i];
  let v = e.target.type === "checkbox" ? e.target.checked : e.target.value;
  if (["sigma_hat", "mu_hat", "conf", "weight", "catalyst_base"].includes(k)) v = v === "" ? null : +v;
  if (k === "expiry" && v === "") v = null;
  t[k] = v;
  if (k === "conf") $("#tconfv" + i).textContent = v;
  schedule(k === "conf" ? 700 : 400);
}
function onCatInput(e) {
  const [i, j, k] = e.target.dataset.cat.split(",");
  S.state.tickers[+i].catalysts[+j][k] = k === "move" ? +e.target.value : e.target.value;
  schedule();
}

function updateCardFacts() {
  const R = S.result;
  S.state.tickers.forEach((t, i) => {
    const b = R.tickers.find(x => x.ticker === t.ticker.toUpperCase());
    if (!b) return;
    if (b.fatal) { $("#tfacts" + i).innerHTML = `<div class="banner" style="grid-column:1/-1">${esc(b.fatal)}</div>`; return; }
    const m = b.market, md = b.model;
    $("#tname" + i).textContent = m.name;
    $("#tfacts" + i).innerHTML = `
      <div><span>현물</span> ${mkt("$" + num(m.spot), "Yahoo 현재가 (15~20분 지연)")}</div>
      <div><span>섹터</span> ${mkt(esc(m.sector), m.sector_source)}</div>
      <div><span>HV 1y</span> ${mdl(pct(md.hv.hv1y), "2년 일봉 종가 로그수익률 표준편차 (최근 252일) — 과거 데이터로 계산한 값")}</div>
      <div><span>ATM IV</span> ${mdl(pct(md.iv_atm), "선택 만기 체인을 SVI로 맞춘 뒤 선도가격에서 읽은 IV (시장 호가에서 도출)")}</div>
      <div><span>배당</span> ${mkt(pct(m.div_yield, 2), "최근 12개월 현금배당 / 현물")}</div>
      <div><span>Parkinson</span> ${mdl(pct(md.hv.parkinson_1y), "고가·저가 범위 추정 변동성 (1y)")}</div>`;
    const sel = $("#texp" + i);
    sel.innerHTML = `<option value="">자동 (${esc(b.chosen_expiry)})</option>` + b.expiries.map(x =>
      `<option value="${x.expiry}" ${t.expiry === x.expiry ? "selected" : ""}>${x.expiry} · ${(x.T * 365).toFixed(0)}일 · q ${x.q_pass ? "✓" : "✕"}</option>`).join("");
    const e = md.edge;
    $("#tedge" + i).innerHTML = `<span class="badge ${e > 0 ? "good" : "bad"}" data-tip="Edge = σ̂ − ATM IV. BS 가격에서 드리프트가 상쇄되므로 '싸다'는 변동성으로만 주장 가능. Edge>0만 시뮬레이션에 포함.">
      Edge ${mdl((e >= 0 ? "+" : "") + pct(e))}</span> <span class="sub">σ̂ ${mdl(pct(md.sigma_hat))} · ${esc(md.sigma_hat_source)}</span>`;
    const c = md.catalyst;
    let co = c.sigma_hat ? `계산기 σ̂ = ${mdl(pct(c.sigma_hat))} (base ${pct(c.base)})` : "이벤트 없음";
    if (c.priced_in) {
      co += c.priced_in.error ? `<div class="note">${esc(c.priced_in.error)}</div>` :
        `<div><span class="badge ${c.priced_in.badge === "underpriced" ? "good" : c.priced_in.badge === "priced in" ? "bad" : "warn"}"
          data-tip="이벤트 전 만기(${c.priced_in.excluding_expiry}) IV ${pct(c.priced_in.iv_excluding)} → 포함 만기 IV ${pct(c.priced_in.iv_including)}. 시장의 IV 상승폭 / 예상 상승폭 = ${num(c.priced_in.ratio)}. 0.5 미만이면 저평가.">
          ${esc(c.priced_in.badge)}</span> 시장 내재 이벤트 ${mdl(pct(c.priced_in.implied_move))} vs 예상 ${mdl(pct(c.priced_in.expected_move))}</div>`;
    }
    co += c.timing.filter(x => !x.ok).map(x => `<div class="note alert">⚠ ${esc(x.name)} ${x.date}: 만기 ${x.days_before_expiry}일 전 — 1개월 규칙 위반, 일정이 밀리면 포지션 무효</div>`).join("");
    $("#tcatout" + i).innerHTML = co;
    $("#tqfail" + i).innerHTML = (md.q_pass ? "" : `<div class="note alert">q 추출 검증 실패 — 기본은 시뮬레이션 제외.
      <label><input type="checkbox" data-t="${i}" data-k="use_failed_q" ${t.use_failed_q ? "checked" : ""}> 그래도 사용</label></div>`)
      + (md.liquidity_offhours ? `<div class="note">장 마감 후 호가라 호가폭 기준은 보류하고 OI만 확인함 — 비용은 넓은 장외 호가로 계산돼 보수적. 주문 전 장중 호가로 재확인.</div>` : "")
      + (md.spread.liquidity_ok || md.liquidity_offhours ? "" : `<div class="note alert">유동성 필터 미달 (OI ≥ 100, 다리 호가 ≤ mid의 10%) — 기본은 제외.
      <label><input type="checkbox" data-t="${i}" data-k="use_illiquid" ${t.use_illiquid ? "checked" : ""}> 그래도 사용</label></div>`)
      + (b.in_simulation ? "" : `<div class="note">시뮬레이션 제외: ${esc(b.excluded_reason)}</div>`);
    $("#tqfail" + i).querySelectorAll("input").forEach(inp => inp.addEventListener("change", onTickerInput));
  });
}

// ------------------------------------------------------------------ globals card
function gField(key, label, attrs, tip, fmt) {
  const v = S.state.globals[key];
  return `<div class="field"><label data-tip="${esc(tip)}">${label}${fmt ? ` <span id="gv_${key}">${fmt(v)}</span>` : ""}</label><input data-g="${key}" ${attrs} value="${v ?? ""}"></div>`;
}
function gSelect(key, label, opts, tip) {
  const v = S.state.globals[key];
  return `<div class="field"><label data-tip="${esc(tip)}">${label}</label><select data-g="${key}">${opts.map(([val, txt]) =>
    `<option value="${val}" ${String(v) === String(val) ? "selected" : ""}>${txt}</option>`).join("")}</select></div>`;
}
function renderGlobals() {
  const g = S.state.globals;
  $("#gcard").innerHTML = `<h2>전역 설정</h2>
    <h3>예산 B = (V − V_own)(1 − (1+r)^−T)</h3>
    ${gField("V_deposit", "전세보증금 V", 'type="number" step="10000000"', "2년마다 돌려줘야 하는 받은 보증금")}
    ${gField("V_own_rental", "자가 전세 V_own", 'type="number" step="10000000"', "본인이 살 집의 전세보증금 (경기 평균 3.8억)")}
    ${gField("r", "국고채 2년 r", 'type="number" step="0.0005" placeholder="자동"', "비우면 데이터 레이어 값 (ECOS 키 없으면 3.76% 가정값). 금리가 내리면 B가 줄어듦.")}
    ${gField("rate_shift", "사이클당 금리 변화", 'type="number" step="0.0025"', "다음 사이클마다 r에 더하는 값. −0.0075 = 사이클마다 75bp 하락 → 이후 B가 계속 줄어듦.")}
    ${gField("T", "채권 만기 T (년)", 'type="number" step="0.5" min="0.5"', "전세 롤오버 주기")}
    <h3>목표·운용</h3>
    ${gSelect("G", "목표 G", GOAL_PRESETS.map(([v, t]) => [v, t]), "8년 누적 회수액(세후) 목표. 목적함수 = P(누적 ≥ G)")}
    ${gField("f", "스테이킹 f", 'type="range" min="0.1" max="1" step="0.05"', "롤마다 슬리브의 f만 투입, 나머지는 무위험. 손실 후 증액 없음(마틴게일 아님).", v => (+v).toFixed(2))}
    ${gSelect("k2_rule", "K2 규칙", [["cost-aware", "비용 반영 폭 (체인 호가)"], ["narrowest", "최좁음 (스펙)"]], "비용 반영: 후보 K2마다 실제 호가 왕복비용으로 점수를 매겨 고름. 최좁음은 비교용으로 항상 같이 표시.")}
    ${gSelect("k2_criterion", "K2 점수", [["expected", "롤당 기대배수 (빠름)"], ["goal", "목표확률 (느림)"]], "비용 반영 규칙에서 K2 후보를 무엇으로 비교할지")}
    ${gSelect("cost_mode", "최좁음 비용 기준", [["chain", "체인 호가"], ["flat", "고정 %"]], "최좁음 규칙일 때만 사용")}
    ${gField("rt_cost", "고정 왕복비용", 'type="number" step="0.01"', "고정 % 모드의 프리미엄 대비 왕복비용 (스펙 기본 4%)")}
    ${gField("leg_cross", "다리 호가 지불 비율", 'type="range" min="0.2" max="1" step="0.05"', "1 = 진입·청산 때 두 다리 호가를 전부 넘김(최악). 콤보 주문은 보통 그 안쪽에서 체결.", v => (+v).toFixed(2))}
    <div class="field"><label data-tip="해외파생 양도세 22%, 연 250만 공제, 연도별 손익통산, 이월공제 없음">세금 반영</label><input type="checkbox" data-g="tax" ${g.tax ? "checked" : ""}></div>
    ${gSelect("broker_confirmed", "버티컬 증거금 서면 확인 (R-3)", [["null", "미확인"], ["true", "확인함"], ["false", "인식 안 함"]], "미확인/불인식이면 매도 다리를 무담보로 봐서 기초자산 20% 증거금")}
    <h3>시뮬레이션</h3>
    ${gField("n_paths", "경로 수", 'type="number" step="1000" min="1000" max="50000"', "몬테카를로 경로 수")}
    ${gSelect("copula", "의존구조", [["gaussian", "가우시안"], ["t", "t-copula (df=5)"]], "t-copula는 동시 전손(꼬리 의존)을 키움 — 롱콜 바스켓의 핵심 위험")}
    ${gField("start", "시작일", 'type="date"', "첫 롤 진입일. 청산일의 연도로 세금 연도가 정해짐")}
    ${gField("mortgage_rate", "대출 금리", 'type="number" step="0.0025"', "부분 달성 시 월 상환액 감소 계산용")}
    ${gField("remaining_months", "8년 후 잔여 기간(월)", 'type="number" step="12"', "부분 달성 시 월 상환액 감소 계산용")}`;
  $("#gcard").querySelectorAll("[data-g]").forEach(inp => inp.addEventListener(inp.type === "range" ? "input" : "change", e => {
    const k = e.target.dataset.g;
    let v = e.target.type === "checkbox" ? e.target.checked : e.target.value;
    if (k === "broker_confirmed") v = v === "null" ? null : v === "true";
    else if (!["k2_rule", "k2_criterion", "cost_mode", "copula", "start"].includes(k) && typeof v === "string") v = v === "" ? null : +v;
    S.state.globals[k] = v;
    const lab = $("#gv_" + k); if (lab) lab.textContent = (+v).toFixed(2);
    schedule(e.target.type === "range" ? 700 : 400);
  }));
}

// ------------------------------------------------------------------ results
function renderResults() {
  const R = S.result;
  updateHeader(R);
  updateCardFacts();
  const t = R.tickers.filter(b => !b.fatal);
  const P = R.portfolio;
  $("#results").innerHTML = `
    ${t.filter(b => b.model.pin_banner).map(b => `<div class="banner">⛔ ${esc(b.model.pin_banner.message)}</div>`).join("")}
    ${R.warnings.map(w => `<div class="note alert">⚠ ${esc(w)}</div>`).join("")}
    ${panelBudget(R)}
    <section class="panel"><h2>① 구조표 — 실제 상장 체인 <span class="badge info">${R.globals.k2_rule === "cost-aware" ? "K2: 비용 반영" : "K2: 최좁음 (스펙)"}</span></h2>
      <div class="sub">굵은 행이 사용하는 구조, 흐린 행은 스펙의 최좁음 규칙(비교용). 호가·OI는 시장값, 나머지는 모델값.</div>
      <div class="scroll">${structureTable(t)}</div>${k2Tables(t)}</section>
    <section class="panel"><h2>② 만기 손익 (KRW, 첫 롤 예산 기준)</h2>
      <div class="sub">빨간 띠 = 핀 구간 [K1, K2): 매수 다리만 자동행사되어 행사대금이 청구되는 구간. 표시 확률은 주관 분포 p 기준.</div>
      <div class="grid2" id="payoffs"></div></section>
    <section class="panel"><h2>③ 밀도 — 시장 q vs 견해 p, 우도비 ℓ = p/q</h2>
      <div class="tabs" id="dtabs"></div><div id="dbody"></div></section>
    ${P && !P.error ? `
    <section class="panel"><h2>④ 8년 타임라인 — 32롤</h2>
      <div class="sub">한 몬테카를로 경로의 롤별 투입·회수와 누적 회수액(세후). 회색 띠 = 2년 사이클, 점선 = 목표 G, ▲ = 흡수상태(목표 도달 → 그 사이클 나머지는 무위험 대기).</div>
      <div style="display:flex;gap:8px;align-items:center;margin-bottom:6px;flex-wrap:wrap">
        <button class="btn" id="pprev">◀</button><span id="plabel" class="sub"></span><button class="btn" id="pnext">▶</button>
        <button class="btn primary" id="pplay">▶ 재생: 경로를 쌓아 분포 채우기</button></div>
      <div class="chart" id="timeline"></div></section>
    ${panelOutcome(P, R)}
    ${panelRisk(t, P, R)}` : `<section class="panel banner">${esc(P?.error || "포트폴리오 없음")}</section>`}`;
  drawPayoffs(t);
  drawDensityTabs(t);
  if (P && !P.error) {
    drawTimeline(P, R);
    $("#pprev").onclick = () => { S.path = (S.path + P.paths.cumulative.length - 1) % P.paths.cumulative.length; drawTimeline(P, R); };
    $("#pnext").onclick = () => { S.path = (S.path + 1) % P.paths.cumulative.length; drawTimeline(P, R); };
    $("#pplay").onclick = () => playPaths(P, R);
    drawFrontier(P, R);
    drawCountCurve(P, R);
  }
}

function updateHeader(R) {
  const ts = R.tickers.map(b => b.data_asof).filter(Boolean).sort();
  const stale = R.tickers.some(b => b.stale) || Object.values(R.macro).some(m => m && m.stale);
  $("#asof").innerHTML = `데이터 기준: <b>${ts.length ? new Date(ts[0]).toLocaleString("ko-KR") : "-"}</b> · 오늘 ${R.status.today} (뉴욕)`;
  $("#stale").innerHTML = stale ? `<span class="badge bad" data-tip="갱신에 실패해 캐시를 보여주는 항목이 있음. 각 카드의 상세를 확인.">STALE</span>` : `<span class="badge good">최신</span>`;
  $("#delay").textContent = " · " + R.status.delay_note + " 자동 갱신: " + R.status.schedule;
}

function panelBudget(R) {
  const b = R.budget, g = R.globals, m = R.macro;
  const rOrigin = { user: "사용자 입력", market: "시장(ECOS)", fallback: "가정값 3.76% — 확인 필요" }[b.r_origin] || b.r_origin;
  const max = Math.max(...b.budgets);
  return `<section class="panel"><h2>예산 · 금리 연동</h2>
    <div class="sub">${esc(b.formula)} — r: ${pct(g.r, 2)} <span class="badge ${b.r_origin === "fallback" ? "warn" : "info"}">${esc(rOrigin)}</span>
      · USD/KRW ${m.usdkrw ? mkt(num(m.usdkrw.value), m.usdkrw.source) : "–"} · 미국 3개월 ${m.ust_3m ? mkt(pct(m.ust_3m.value, 2), m.ust_3m.source + " — 옵션 할인율") : "–"}
      · 미국 2년 ${m.ust_2y ? mkt(pct(m.ust_2y.value, 2), "참고용") : "–"}</div>
    <div class="tiles">${b.budgets.map((B, c) => `<div class="tile"><div class="k">사이클 ${c + 1} · r ${pct(b.rates[c], 2)}</div>
      <div class="v mdl">${won(B)}</div><div style="height:6px;border-radius:3px;background:var(--s1);width:${(B / max * 100).toFixed(0)}%;transition:width .5s"></div>
      <div class="s">채권 매입가 ${won((g.V_deposit - g.V_own_rental) * Math.pow(1 + b.rates[c], -g.T))}</div></div>`).join("")}</div></section>`;
}

function structureTable(t) {
  const head = `<tr><th>종목</th><th>만기</th><th data-tip="K1 = S0(1 + 1.56 σ̂ √0.25) 이상인 최저 상장 행사가">K1</th><th>K2</th>
    <th>c1 bid/ask/mid<span class="mkt-tag">mkt</span></th><th>c2 bid/ask/mid<span class="mkt-tag">mkt</span></th>
    <th data-tip="(c1−c2)·(1+왕복비용), 주당">순비용</th><th data-tip="체인 호가 기준: 2·[(ask1−mid1)+(mid2−bid2)]/(mid1−mid2)">왕복비용 적용/체인</th>
    <th>계약 n</th><th>최대지급</th><th data-tip="(K2−K1)/순비용">지급/예산</th><th>손익분기</th><th>필요 상승률</th>
    <th>OI K1/K2<span class="mkt-tag">mkt</span></th><th>호가% K1/K2<span class="mkt-tag">mkt</span></th><th>유동성</th>
    <th data-tip="n(K2−K1)·FX / (G/4 × 비중). 1 이상이어야 절단 조건 충족">n(K2−K1)/목표</th></tr>`;
  const row = (b, i, sp, alt) => {
    const L = sp.liquidity;
    return `<tr class="${alt ? "alt" : ""}" style="${alt ? "" : "font-weight:600"}">
      <td><span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:${color(i)};margin-right:6px"></span>${esc(b.ticker)}${alt ? " · 최좁음" : ""}</td>
      <td>${sp.expiry}</td><td>${num(sp.K1, 1)}</td><td>${num(sp.K2, 1)}</td>
      <td>${num(sp.c1.bid)}/${num(sp.c1.ask)}/${num(sp.c1.mid)}</td><td>${num(sp.c2.bid)}/${num(sp.c2.ask)}/${num(sp.c2.mid)}</td>
      <td>${mdl("$" + num(sp.net_cost, 3))}</td><td>${mdl(pct(sp.rt_cost, 0))} / ${mdl(pct(sp.rt_cost_chain, 0))}</td>
      <td>${mdl(int(sp.n_int))}</td><td>${mdl(won(sp.max_payoff_krw))}</td><td>${mdl(num(sp.payoff_multiple, 1) + "×")}</td>
      <td>${mdl("$" + num(sp.breakeven))}</td><td>${mdl("+" + pct(sp.breakeven_pct))}</td>
      <td>${int(sp.c1.oi)}/${int(sp.c2.oi)}</td><td>${pct(sp.c1.spread_pct)}/${pct(sp.c2.spread_pct)}</td>
      <td>${(!alt && b.model.liquidity_offhours) ? `<span class="badge warn" data-tip="장 마감 후(주말 포함) Yahoo 호가는 장중보다 매수·매도 차이가 훨씬 넓어서 호가폭 10% 기준은 판정하지 않고 OI(미결제약정)만 확인함. 비용은 그 넓은 호가로 계산돼 결과가 보수적. 실제 주문 전 장중 호가로 다시 확인.">! 장외: OI만</span>` : statusBadge(sp.liquidity_ok ? "pass" : "fail", sp.liquidity_ok ? "통과" : "미달")}</td>
      <td>${mdl(num(sp.target_ratio, 1))}</td></tr>`;
  };
  return `<table class="t">${head}${t.map((b, i) => {
    const sp = b.model.spread, nr = b.model.narrowest;
    return row(b, i, sp, false) + (nr && nr.K2 !== sp.K2 ? row(b, i, nr, true) : "");
  }).join("")}</table>`;
}
function k2Tables(t) {
  return t.filter(b => b.model.k2_table).map(b => `<details><summary>${esc(b.ticker)} K2 후보 ${b.model.k2_table.length}개 — 점수: ${b.model.k2_criterion === "goal" ? "목표확률" : "롤당 기대배수 E^p[M]"}</summary>
    <div class="scroll"><table class="t"><tr><th>K2</th><th>폭</th><th>순 mid</th><th>왕복비용</th><th>지급/예산</th><th>E^p[M]</th>${b.model.k2_criterion === "goal" ? "<th>P(목표)</th>" : ""}<th>유동성</th><th></th></tr>
    ${b.model.k2_table.map(r => `<tr style="${r.chosen ? "font-weight:650" : ""}"><td>${num(r.K2, 1)}</td><td>${num(r.width, 1)}</td><td>${num(r.net_mid, 3)}</td>
      <td>${mdl(pct(r.rt_cost, 0))}</td><td>${mdl(num(r.payoff_multiple, 1) + "×")}</td><td>${mdl(num(r.e_multiple, 3))}</td>
      ${b.model.k2_criterion === "goal" ? `<td>${mdl(pct(r.p_goal))}</td>` : ""}<td>${r.liquidity_ok && r.target_ok ? "✓" : "✕"}</td>
      <td>${r.chosen ? "← 선택" : r.narrowest ? "최좁음" : ""}</td></tr>`).join("")}</table></div></details>`).join("");
}

// ------------------------------------------------------------------ charts: helpers
function svgFor(el, h, m = { t: 12, r: 16, b: 30, l: 56 }) {
  const w = Math.max(280, el.clientWidth || 600);
  const svg = d3.select(el).append("svg").attr("viewBox", `0 0 ${w} ${h}`);
  return { svg, w, h, m, iw: w - m.l - m.r, ih: h - m.t - m.b, g: svg.append("g").attr("transform", `translate(${m.l},${m.t})`) };
}
function axes(c, x, y, xfmt, yfmt, nx = 5, ny = 4) {
  c.g.append("g").attr("class", "gridl").call(d3.axisLeft(y).ticks(ny).tickSize(-c.iw).tickFormat(""));
  c.g.append("g").attr("class", "axis").attr("transform", `translate(0,${c.ih})`).call(d3.axisBottom(x).ticks(nx).tickFormat(xfmt).tickSizeOuter(0));
  c.g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(ny).tickFormat(yfmt).tickSizeOuter(0));
}
function vline(c, x, val, label, dash, col = "var(--ink-2)") {
  c.g.append("line").attr("x1", x(val)).attr("x2", x(val)).attr("y1", 0).attr("y2", c.ih).attr("stroke", col).attr("stroke-width", 1).attr("stroke-dasharray", dash || null);
  if (label) c.g.append("text").attr("class", "lbl").attr("x", x(val) + 3).attr("y", 10).text(label);
}
function crosshair(c, x, xs, rowsAt) {
  const line = c.g.append("line").attr("y1", 0).attr("y2", c.ih).attr("stroke", "var(--axis)").style("display", "none");
  c.g.append("rect").attr("width", c.iw).attr("height", c.ih).attr("fill", "transparent")
    .on("mousemove", ev => {
      const [mx] = d3.pointer(ev); const v = x.invert(mx);
      const i = Math.max(0, Math.min(xs.length - 1, d3.bisectLeft(xs, v)));
      line.attr("x1", x(xs[i])).attr("x2", x(xs[i])).style("display", null);
      showTip(ev, rowsAt(i));
    }).on("mouseleave", () => { line.style("display", "none"); hideTip(); });
}

// ------------------------------------------------------------------ ② payoff
function drawPayoffs(t) {
  const host = $("#payoffs"); if (!host) return;
  const fx = S.result.macro.usdkrw?.value || 1400;
  t.forEach((b, i) => {
    const sp = b.model.spread, S0 = b.market.spot, md = b.model;
    const div = document.createElement("div"); div.className = "chart"; host.appendChild(div);
    div.insertAdjacentHTML("beforeend", `<div class="sub"><b style="color:var(--ink)">${esc(b.ticker)}</b> ${sp.K1}/${sp.K2} · 핀 구간 확률 ${mdl(pct(md.pin_per_roll))}/롤 · K2 위 ${mdl(pct(md.p_finish_above_K2))} · K1 아래(전손) ${mdl(pct(md.p_finish_below_K1))}</div>`);
    const c = svgFor(div, 200);
    const lo = Math.min(S0 * 0.85, sp.K1 * 0.8), hi = sp.K2 + (sp.K2 - sp.K1) * 0.6 + S0 * 0.05;
    const xs = d3.range(0, 241).map(k => lo + (hi - lo) * k / 240);
    const pnl = s => sp.n * 100 * fx * Math.min(Math.max(s - sp.K1, 0), sp.K2 - sp.K1) - sp.budget_krw;
    const x = d3.scaleLinear([lo, hi], [0, c.iw]);
    const ys = xs.map(pnl);
    const y = d3.scaleLinear([d3.min(ys) * 1.1, d3.max(ys) * 1.08], [c.ih, 0]).nice();
    c.g.append("rect").attr("x", x(sp.K1)).attr("width", x(sp.K2) - x(sp.K1)).attr("y", 0).attr("height", c.ih).attr("fill", "var(--pin)");
    axes(c, x, y, d => "$" + d3.format(",.0f")(d), d => won(d), 5, 4);
    c.g.append("line").attr("x1", 0).attr("x2", c.iw).attr("y1", y(0)).attr("y2", y(0)).attr("stroke", "var(--axis)");
    c.g.append("path").datum(xs).attr("fill", "none").attr("stroke", color(i)).attr("stroke-width", 2)
      .attr("d", d3.line().x(s => x(s)).y(s => y(pnl(s))));
    vline(c, x, S0, "현물", "3,3");
    vline(c, x, sp.breakeven, "BE", "1,3");
    c.g.append("text").attr("class", "lbl").attr("x", x(sp.K1) + 2).attr("y", c.ih - 6).text(`핀 ${pct(md.pin_per_roll)}`);
    crosshair(c, x, xs, k => `S_T $${num(xs[k])}<br>손익 <i>${won(ys[k])}</i><br>${xs[k] >= sp.K1 && xs[k] < sp.K2 ? "⚠ 핀 구간: 행사대금 " + won(sp.exercise_cash_krw) : ""}`);
  });
}

// ------------------------------------------------------------------ ③ density
function drawDensityTabs(t) {
  const tabs = $("#dtabs"); if (!tabs || !t.length) return;
  S.densityTab = Math.min(S.densityTab, t.length - 1);
  tabs.innerHTML = t.map((b, i) => `<button class="${i === S.densityTab ? "on" : ""}" data-i="${i}">${esc(b.ticker)}</button>`).join("");
  tabs.querySelectorAll("button").forEach(bt => bt.onclick = () => { S.densityTab = +bt.dataset.i; drawDensityTabs(t); });
  const b = t[S.densityTab], md = b.model, D = md.density, sp = md.spread;
  const chk = Object.entries(md.q_checks).map(([k, v]) => statusBadge(v.ok ? "pass" : "fail", `${k} ${num(v.value, 4)}`)).join(" ");
  const klc = { humble: "good", ok: "good", amber: "warn", alarm: "bad" }[md.kl_level];
  const dc = md.decomposition;
  $("#dbody").innerHTML = `<div class="rules" style="margin-bottom:6px">
      <span data-tip="필수 체크: ∫S·q(S)dS = S0·e^((r−δ)T) 1% 이내. 나머지는 꼬리 절단·배당 오입력·음수 밀도·재가격 오차를 잡는 보조 체크">q 검증</span> ${chk}
      <span class="badge ${klc}" data-tip="KL(p‖q): 견해가 시장에서 얼마나 벗어났나. <0.1 겸손, >0.5 경보(과적합 의심)">KL ${mdl(num(md.kl, 3))} · ${md.kl_level}</span></div>
    <div class="sub">E^q[S] ${mdl("$" + num(md.mean_q))} vs 선도 ${mdl("$" + num(md.forward))} · σ_used ${mdl(pct(md.sigma_used))} · 매수영역 하한 ${mdl("$" + num(D.K_lo))} (상장 K1 ${sp.K1})
      · 롤당 기대배수: 시장 q ${mdl(num(dc.e_q, 3))} → 변동성 견해 ${mdl((dc.vol_contribution >= 0 ? "+" : "") + num(dc.vol_contribution, 3))} → 드리프트 견해 ${mdl((dc.drift_contribution >= 0 ? "+" : "") + num(dc.drift_contribution, 3))} = ${mdl(num(dc.e_total, 3))}</div>
    <div class="chart" id="dchart"></div><div class="chart" id="lchart"></div>
    ${md.tilt_edge_mass > 0.01 ? `<div class="note alert">p의 질량 ${pct(md.tilt_edge_mass)}가 그리드 끝에 쌓임 — σ̂가 IV보다 너무 커서 틸팅이 꼬리로 밀어냄</div>` : ""}`;
  const S_ = D.S, xs = S_;
  // densities
  const c = svgFor($("#dchart"), 220);
  const x = d3.scaleLinear(d3.extent(xs), [0, c.iw]);
  const y = d3.scaleLinear([0, d3.max([...D.q, ...D.p]) * 1.1], [c.ih, 0]);
  let start = null;
  D.buy.forEach((on, k) => {
    if (on && start === null) start = k;
    if ((!on || k === D.buy.length - 1) && start !== null) {
      const end = on ? k : k - 1;
      c.g.append("rect").attr("x", x(xs[start])).attr("width", Math.max(1, x(xs[end]) - x(xs[start]))).attr("y", 0).attr("height", c.ih).attr("fill", "var(--buy)");
      start = null;
    }
  });
  c.g.append("rect").attr("x", x(sp.K1)).attr("width", Math.max(1, x(sp.K2) - x(sp.K1))).attr("y", 0).attr("height", c.ih).attr("fill", "var(--pin)");
  axes(c, x, y, d => "$" + d3.format(",.0f")(d), d3.format(".2~e"), 6, 3);
  [[D.q, "var(--s1)", "q 시장"], [D.p, "var(--s2)", "p 견해"]].forEach(([arr, col, lab], j) => {
    c.g.append("path").datum(arr).attr("fill", "none").attr("stroke", col).attr("stroke-width", 2)
      .attr("d", d3.line().x((_, k) => x(xs[k])).y(v => y(v)));
    c.g.append("text").attr("class", "lbl").attr("x", c.iw - 4).attr("y", 12 + j * 14).attr("text-anchor", "end").attr("fill", col).text("— " + lab);
  });
  vline(c, x, b.market.spot, "현물", "3,3");
  crosshair(c, x, xs, k => `S $${num(xs[k])}<br><span style="color:var(--s1)">q</span> ${D.q[k].toExponential(2)}<br><span style="color:var(--s2)">p</span> ${D.p[k].toExponential(2)}`);
  // likelihood ratio
  const c2 = svgFor($("#lchart"), 150);
  const y2 = d3.scaleLinear([0, Math.max(d3.max(D.ell) * 1.1, D.c * 1.2, 1.2)], [c2.ih, 0]);
  axes(c2, x, y2, d => "$" + d3.format(",.0f")(d), d3.format(".2f"), 6, 3);
  c2.g.append("line").attr("x1", 0).attr("x2", c2.iw).attr("y1", y2(1)).attr("y2", y2(1)).attr("stroke", "var(--axis)");
  c2.g.append("line").attr("x1", 0).attr("x2", c2.iw).attr("y1", y2(D.c)).attr("y2", y2(D.c)).attr("stroke", "var(--critical)").attr("stroke-dasharray", "4,3");
  c2.g.append("text").attr("class", "lbl").attr("x", 4).attr("y", y2(D.c) - 4).text(`임계 c = ${num(D.c, 3)} (예산이 binding되도록 이분법)`);
  c2.g.append("path").datum(D.ell).attr("fill", "none").attr("stroke", "var(--ink-2)").attr("stroke-width", 2)
    .attr("d", d3.line().x((_, k) => x(xs[k])).y(v => y2(v)));
  c2.g.append("text").attr("class", "lbl").attr("x", c2.iw - 4).attr("y", 12).attr("text-anchor", "end").text("ℓ = p/q · 파란 띠 = 매수영역 {ℓ > c, S ≥ F}");
  crosshair(c2, x, xs, k => `S $${num(xs[k])}<br>ℓ ${num(D.ell[k], 3)} ${D.ell[k] > D.c ? "(> c)" : ""}`);
}

// ------------------------------------------------------------------ ④ timeline
function drawTimeline(P, R, overlay) {
  const host = $("#timeline"); if (!host) return;
  host.innerHTML = "";
  const n = P.paths.cumulative.length, p = S.path % n;
  $("#plabel").textContent = `경로 ${p + 1} / ${n} (표본) · 최종 누적 ${won(P.paths.cumulative[p][31])}`;
  const c = svgFor(host, 300, { t: 14, r: 150, b: 32, l: 60 });
  const G = R.globals.G;
  const cum = P.paths.cumulative.map(a => a);
  const ymax = Math.max(G * 1.35, d3.quantile(cum.map(a => a[31]).sort(d3.ascending), 0.9) || 0);
  const x = d3.scaleBand(d3.range(32), [0, c.iw]).padding(0.15);
  const y = d3.scaleLinear([0, ymax], [c.ih, 0]).nice();
  for (let cy = 0; cy < 4; cy++) {
    c.g.append("rect").attr("x", x(cy * 8)).attr("width", x(cy * 8 + 7) + x.bandwidth() - x(cy * 8)).attr("y", 0).attr("height", c.ih)
      .attr("fill", cy % 2 ? "var(--surface-2)" : "transparent");
    const bw = x(cy * 8 + 7) + x.bandwidth() - x(cy * 8);
    c.g.append("text").attr("class", "lbl").attr("x", x(cy * 8) + 4).attr("y", 11)
      .text(bw > 130 ? `사이클 ${cy + 1} · B ${won(P.cycle_budgets[cy])}` : `C${cy + 1} · ${won(P.cycle_budgets[cy])}`);
  }
  c.g.append("g").attr("class", "gridl").call(d3.axisLeft(y).ticks(4).tickSize(-c.iw).tickFormat(""));
  c.g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(4).tickFormat(won));
  c.g.append("g").attr("class", "axis").attr("transform", `translate(0,${c.ih})`)
    .call(d3.axisBottom(x).tickValues(d3.range(0, 32, 4)).tickFormat(k => P.calendar[k].close.slice(0, 7)));
  // background: faint paths (filled in by play)
  const bg = c.g.append("g").attr("class", "bgpaths");
  const line = d3.line().x((v, k) => x(k) + x.bandwidth() / 2).y(v => y(Math.min(v, ymax)));
  (overlay || []).forEach(k => bg.append("path").datum(cum[k]).attr("fill", "none").attr("stroke", "var(--muted)").attr("stroke-opacity", 0.25).attr("d", line));
  const dep = P.paths.deployed[p], pro = P.paths.proceeds[p], ab = P.paths.absorbed[p];
  c.g.selectAll(".dep").data(dep).join("rect").attr("x", (v, k) => x(k)).attr("width", x.bandwidth() / 2 - 1)
    .attr("y", v => y(Math.min(v, ymax))).attr("height", v => c.ih - y(Math.min(v, ymax))).attr("fill", "var(--muted)").attr("rx", 2);
  c.g.selectAll(".pro").data(pro).join("rect").attr("x", (v, k) => x(k) + x.bandwidth() / 2).attr("width", x.bandwidth() / 2 - 1)
    .attr("y", v => y(Math.min(v, ymax))).attr("height", v => c.ih - y(Math.min(v, ymax))).attr("fill", "var(--s1)").attr("rx", 2);
  c.g.append("path").datum(cum[p]).attr("fill", "none").attr("stroke", "var(--s2)").attr("stroke-width", 2).attr("d", line);
  c.g.append("line").attr("x1", 0).attr("x2", c.iw).attr("y1", y(G)).attr("y2", y(G)).attr("stroke", "var(--critical)").attr("stroke-dasharray", "5,4");
  c.g.append("text").attr("class", "lbl").attr("x", c.iw - 4).attr("y", y(G) - 4).attr("text-anchor", "end").text(`목표 G ${won(G)}`);
  ab.forEach((v, k) => { if (v && (k % 8 === 0 || !ab[k - 1])) c.g.append("text").attr("x", x(k) + x.bandwidth() / 2).attr("y", y(Math.min(cum[p][k], ymax)) - 6).attr("text-anchor", "middle").attr("fill", "var(--ink)").text("▲"); });
  c.g.append("text").attr("class", "lbl").attr("x", 0).attr("y", c.ih + 28).text("회색 막대 = 투입, 파란 막대 = 회수, 주황 선 = 누적 회수액(세후)");
  c.g.selectAll(".hit").data(d3.range(32)).join("rect").attr("x", k => x(k)).attr("width", x.bandwidth()).attr("y", 0).attr("height", c.ih).attr("fill", "transparent")
    .on("mousemove", (ev, k) => showTip(ev, `롤 ${k + 1} · ${P.calendar[k].entry} → 청산 ${P.calendar[k].close} (만기 ${P.calendar[k].expiry})<br>투입 <i>${won(dep[k])}</i> · 회수 <i>${won(pro[k])}</i><br>슬리브 <i>${won(P.paths.sleeve[p][k])}</i> · 누적 <i>${won(cum[p][k])}</i>${ab[k] ? "<br>▲ 흡수상태: 투입 0" : ""}`))
    .on("mouseleave", hideTip);
  // right edge histogram of the full terminal distribution
  const hx = c.iw + 16, hw = 120;
  const H = P.hist, maxc = d3.max(H.counts);
  const hy = d3.scaleLinear([H.edges[0], H.edges[H.edges.length - 1]], [c.ih, 0]);
  const hs = c.g.append("g").attr("class", "hist");
  hs.selectAll("rect").data(H.counts).join("rect").attr("x", hx).attr("y", (v, k) => y(Math.min(H.edges[k + 1], ymax)))
    .attr("height", (v, k) => Math.max(0, y(Math.min(H.edges[k], ymax)) - y(Math.min(H.edges[k + 1], ymax)) - 1))
    .attr("width", v => overlay ? 0 : Math.sqrt(v / maxc) * hw).attr("fill", "var(--s3)").attr("rx", 1)
    .on("mousemove", (ev, v) => showTip(ev, `${pct(v / d3.sum(H.counts))} 경로`)).on("mouseleave", hideTip);
  c.g.append("text").attr("class", "lbl").attr("x", hx).attr("y", -2).text(`최종 분포 (${int(P.n_paths)} 경로, 폭 √)`);
  return { c, x, y, line, bg, hs, hx, hw, H, maxc, ymax };
}
function playPaths(P, R) {
  if (S.playing) return;
  S.playing = true;
  const ctx = drawTimeline(P, R, []);
  const n = P.paths.cumulative.length;
  const counts = new Array(ctx.H.counts.length).fill(0);
  const smp = P.sample_terminal, E = ctx.H.edges;
  let k = 0, s = 0;
  const step = () => {
    for (let j = 0; j < 2 && k < n; j++, k++)
      ctx.bg.append("path").datum(P.paths.cumulative[k]).attr("fill", "none").attr("stroke", "var(--muted)").attr("stroke-opacity", 0.22).attr("d", ctx.line);
    for (let j = 0; j < 40 && s < smp.length; j++, s++) {
      const v = Math.min(Math.max(smp[s], E[0]), E[E.length - 1]);
      let b = d3.bisectRight(E, v) - 1; b = Math.max(0, Math.min(counts.length - 1, b)); counts[b]++;
    }
    const mx = d3.max(counts) || 1;
    ctx.hs.selectAll("rect").data(counts).attr("width", v => Math.sqrt(v / mx) * ctx.hw);
    if (k < n || s < smp.length) setTimeout(step, 40); else S.playing = false;   // ~1.5 s; runs even in a hidden tab
  };
  step();
}

// ------------------------------------------------------------------ ⑤ outcome
function panelOutcome(P, R) {
  const m = P.metrics_net, gm = P.metrics_gross, G = R.globals.G;
  const regimeTxt = P.regime === "timid"
    ? "유리한 게임(롤당 기대배수 > 무위험 성장): Dubins–Savage상 작은 베팅을 여러 번 하는 쪽(작은 f)이 유리"
    : "불리한 게임(롤당 기대배수 ≤ 무위험 성장): Dubins–Savage상 크게 적게 거는 bold play(f=1)가 목표확률 최대 — 대신 파산과 맞바꿈";
  const qrow = (lab, a, b) => `<tr><td>${lab}</td><td>${mdl(won(a))}</td><td>${mdl(won(b))}</td></tr>`;
  return `<section class="panel"><h2>⑤ 결과 — 8년 누적 회수액 (원금과 별개, ${R.globals.tax ? "세후" : "세전"})</h2>
    <div class="sub">${int(P.n_paths)} 경로 · ${P.copula === "t" ? "t-copula" : "가우시안"} · 종목 ${P.included.join(", ")} · 비중 ${P.weights.map(w => pct(w, 0)).join("/")}</div>
    <div class="tiles">
      <div class="tile big"><div class="k" data-tip="P(8년 누적 회수액 ≥ G). 최적화 기본 목적함수">목표 달성확률 P(≥ ${won(G)})</div><div class="v mdl">${pct(m.p_goal)}</div><div class="s">세전 ${pct(gm.p_goal)}</div></div>
      <div class="tile big"><div class="k" data-tip="사이클 말 슬리브가 그 사이클 예산의 5% 미만 (목표 도달로 대기 중인 경로 제외). 목표확률과 같은 비대칭의 다른 얼굴">사이클 파산확률</div><div class="v mdl">${pct(m.ruin_per_cycle)}</div><div class="s">한 번이라도 ${pct(m.ruin_any_cycle)} · 4번 모두 ${pct(m.ruin_all_cycles)}</div></div>
      <div class="tile big"><div class="k" data-tip="분포가 오른쪽으로 크게 치우쳐서 평균보다 중앙값이 대표값">중앙값</div><div class="v mdl">${won(m.median)}</div><div class="s">세전 ${won(gm.median)}</div></div>
      <div class="tile"><div class="k">평균</div><div class="v mdl">${won(m.mean)}</div><div class="s">${m.mean_driven_by_tail ? "⚠ 평균 > 2×중앙값: 소수의 큰 결과가 평균을 끌어올림" : "평균 ≈ 중앙값"}</div></div>
      <div class="tile"><div class="k" data-tip="Σ B_c (1+r_c)^T — 옵션 대신 전액 국고채였다면">전액 국고채 대안</div><div class="v mdl">${won(m.bond_only_alternative)}</div><div class="s">평균 − 대안 ${won(m.excess_mean_vs_bond)}</div></div>
      <div class="tile"><div class="k">롤당 기대배수 E[M]</div><div class="v mdl">${num(m.roll_multiple_mean, 3)}</div><div class="s">무위험 ${num(P.rf_growth_per_roll, 4)} → ${P.regime === "timid" ? "timid" : "bold"}</div></div>
    </div>
    <div class="note">${regimeTxt}</div>
    ${m.p_tax_exceeds_proceeds > 0 ? `<div class="note alert">⚠ 세금 함정: ${pct(m.p_tax_exceeds_proceeds)} 경로에서 세금 > 회수액 (한 해 실현이익 과세 후 다음 해 손실, 이월공제 없음) → 원금에서 납부. 최악 ${won(m.worst_net)}</div>` : ""}
    <div class="grid2">
      <div><h3 class="sub">분위수</h3><table class="t"><tr><th>분위</th><th>세후</th><th>세전</th></tr>
        ${Object.keys(m.quantiles).map(q => qrow((q * 100).toFixed(0) + "%", m.quantiles[q], gm.quantiles[q])).join("")}</table></div>
      <div><h3 class="sub">부분 달성 — 8년 후 일시 상환 시 월 상환액 감소 (대출 ${pct(R.globals.mortgage_rate, 2)}, ${R.globals.remaining_months}개월)</h3>
        <table class="t"><tr><th>누적 ≥</th><th>확률 (세후)</th><th>월 상환 감소</th></tr>
        ${m.ladder.map(l => `<tr><td>${won(l.level)}</td><td>${mdl(pct(l.prob))}</td><td>${mdl(won(l.monthly_reduction))}/월</td></tr>`).join("")}</table></div>
    </div>
    <div class="grid2" style="margin-top:12px">
      <div><h3 class="sub">f 프런티어 — 목표확률 vs 파산확률 (같은 난수)</h3><div class="chart" id="frontier"></div></div>
      <div><h3 class="sub">종목 수 효과 — ${esc(P.ticker_count_template)}을 상관 ${num(P.ticker_count_rho, 2)}로 복제, 균등비중</h3><div class="chart" id="countcurve"></div>
        <div class="note">계단처럼 오르내리는 건 노이즈가 아님: k종목 균등이면 한 종목만 이겼을 때 배수가 (최대지급/예산)/k라서, 그게 G/B를 넘는지 여부가 k에 따라 바뀜. 단조 감소는 롤당 기대배수가 1보다 확실히 낮을 때만 성립(README 발견 A).</div></div>
    </div></section>`;
}
function drawFrontier(P) {
  const host = $("#frontier"); if (!host) return;
  const c = svgFor(host, 200);
  const F = P.f_frontier;
  const x = d3.scaleLinear([0, Math.max(0.05, d3.max(F, d => d.ruin)) * 1.1], [0, c.iw]);
  const y = d3.scaleLinear([0, Math.max(0.05, d3.max(F, d => d.p_goal)) * 1.15], [c.ih, 0]);
  axes(c, x, y, d3.format(".0%"), d3.format(".0%"), 5, 4);
  c.g.append("path").datum(F).attr("fill", "none").attr("stroke", "var(--s1)").attr("stroke-width", 2).attr("d", d3.line().x(d => x(d.ruin)).y(d => y(d.p_goal)));
  const cur = S.result.globals.f;
  c.g.selectAll("circle").data(F).join("circle").attr("cx", d => x(d.ruin)).attr("cy", d => y(d.p_goal)).attr("r", 5)
    .attr("fill", d => Math.abs(d.f - cur) < 1e-9 ? "var(--s2)" : "var(--s1)").attr("stroke", "var(--surface)").attr("stroke-width", 2)
    .on("mousemove", (ev, d) => showTip(ev, `f = ${d.f}<br>P(목표) <i>${pct(d.p_goal)}</i><br>사이클 파산 <i>${pct(d.ruin)}</i><br>중앙값 <i>${won(d.median)}</i>`)).on("mouseleave", hideTip);
  c.g.selectAll(".fl").data(F).join("text").attr("class", "lbl").attr("x", d => x(d.ruin) + 7).attr("y", d => y(d.p_goal) - 6).text(d => "f " + d.f);
  c.g.append("text").attr("class", "lbl").attr("x", c.iw).attr("y", c.ih - 4).attr("text-anchor", "end").text("x = 사이클 파산확률");
}
function drawCountCurve(P) {
  const host = $("#countcurve"); if (!host) return;
  const c = svgFor(host, 200);
  const ks = Object.keys(P.ticker_count_curve).map(Number).sort((a, b) => a - b);
  const pts = ks.map(k => ({ k, ...P.ticker_count_curve[k] }));
  const x = d3.scalePoint(ks, [0, c.iw]).padding(0.3);
  const y = d3.scaleLinear([0, Math.max(0.05, d3.max(pts, d => d.p_goal)) * 1.15], [c.ih, 0]);
  c.g.append("g").attr("class", "gridl").call(d3.axisLeft(y).ticks(4).tickSize(-c.iw).tickFormat(""));
  c.g.append("g").attr("class", "axis").attr("transform", `translate(0,${c.ih})`).call(d3.axisBottom(x));
  c.g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(4).tickFormat(d3.format(".0%")));
  c.g.selectAll("rect").data(pts).join("rect").attr("x", d => x(d.k) - 9).attr("width", 18).attr("y", d => y(d.p_goal)).attr("height", d => c.ih - y(d.p_goal))
    .attr("fill", d => [3, 4, 5].includes(d.k) ? "var(--s1)" : "var(--muted)").attr("rx", 3)
    .on("mousemove", (ev, d) => showTip(ev, `${d.k}종목<br>P(목표) <i>${pct(d.p_goal)}</i><br>중앙값 <i>${won(d.median)}</i> · 평균 <i>${won(d.mean)}</i><br>사이클 파산 <i>${pct(d.ruin)}</i>`)).on("mouseleave", hideTip);
  c.g.selectAll(".lv").data(pts).join("text").attr("class", "lbl").attr("x", d => x(d.k)).attr("y", d => y(d.p_goal) - 4).attr("text-anchor", "middle").text(d => pct(d.p_goal, 0));
  c.g.append("text").attr("class", "lbl").attr("x", c.iw).attr("y", c.ih + 26).attr("text-anchor", "end").text("종목 수 (파랑 = 설계 범위 3–5)");
}

// ------------------------------------------------------------------ ⑥ risk
function panelRisk(t, P, R) {
  const fx = R.macro.usdkrw?.value || 1400;
  const rows = t.map((b, i) => {
    const md = b.model, sp = md.spread;
    const xd = b.market.ex_dividends.filter(e => e.date <= b.chosen_expiry);
    const bad = md.catalyst.timing.filter(x => !x.ok);
    return `<tr><td><span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:${color(i)};margin-right:6px"></span>${esc(b.ticker)}${b.in_simulation ? "" : ` <span class="badge bad">제외: ${esc(b.excluded_reason)}</span>`}</td>
      <td>${mdl(pct(md.pin_per_roll, 2))}</td><td>${mdl(pct(md.pin_cumulative))}</td>
      <td>${mdl(won(sp.exercise_cash_krw))} <span class="sub">(${mdl(int(sp.exercise_to_premium) + "×")} 프리미엄)</span></td>
      <td>${md.close_by} <span class="sub" data-tip="R-1: 만기 2영업일 전 시장 청산. 만기까지 ${md.bdays_to_expiry}영업일">(${md.bdays_to_close}영업일 남음)</span></td>
      <td>${xd.length ? xd.map(e => e.date + (e.projected ? " (추정)" : "")).join(", ") : "없음"}</td>
      <td>${bad.length ? bad.map(x => `<span class="badge bad">${esc(x.name)} ${x.date}</span>`).join(" ") : statusBadge("pass", "OK")}</td>
      <td>${md.liquidity_offhours ? `<span class="badge warn" data-tip="장 마감 후(주말 포함) Yahoo 호가는 장중보다 매수·매도 차이가 훨씬 넓어서 호가폭 10% 기준은 판정하지 않고 OI(미결제약정)만 확인함. 비용은 그 넓은 호가로 계산돼 결과가 보수적. 실제 주문 전 장중 호가로 다시 확인.">! 장외: OI만</span>` : statusBadge(sp.liquidity_ok ? "pass" : "fail", sp.liquidity_ok ? "통과" : "미달")}</td>
      <td><div class="rules">${md.rules.map(r => `<span data-tip="${esc(r.message)}">${statusBadge(r.status, r.code)}</span>`).join("")}</div></td></tr>`;
  }).join("");
  const notes = Object.values(R.notes).map(n => `<div class="note">${esc(n)}</div>`).join("");
  return `<section class="panel"><h2>⑥ 위험 — 핀·배당·유동성·운용규칙</h2>
    <div class="banner" style="margin-bottom:8px">핀 리스크: 만기 현물이 [K1, K2)에서 끝나면 매수 다리만 자동행사되고 상계할 매도 다리가 없음 → 프리미엄의 수백~수천 배 행사대금 청구.
      유일한 완전한 해결책은 <b>만기 2영업일 전 청산 (R-1)</b>. 프로그램 전체 1회 이상 핀 구간 진입: ${mdl(pct(P.metrics_net.pin_any))} (MC) · 99% 최악 행사대금 ${mdl(won(P.max_exercise_cash_p99))} · 롤당 전 종목 동시 전손 ${mdl(pct(P.joint_total_loss))}</div>
    <div class="scroll"><table class="t"><tr><th>종목</th><th data-tip="P(K1 ≤ S_T < K2), 주관분포 p">핀/롤</th><th data-tip="1 − (1 − p)^32">32롤 누적</th>
      <th data-tip="n × K1 × 100 × USD/KRW">핀 시 행사대금 (첫 롤)</th><th>청산 기한</th><th>만기 전 배당락 (R-4)</th><th>카탈리스트 1개월 규칙</th><th>유동성</th><th>R-1…R-6</th></tr>${rows}</table></div>
    <div class="sub" style="margin-top:6px">R-1 만기 1–2영업일 전 청산 · R-2 콤보 단일 주문 · R-3 버티컬 증거금 서면 확인 · R-4 ITM 매도 다리로 배당락 넘기지 않기 · R-5 부득이한 만기 보유 시 행사 포기 통지 · R-6 만기 직전 두 행사가 사이면 무조건 청산</div>
    ${notes}</section>`;
}

// ------------------------------------------------------------------ boot
if (STATIC) {
  $("#refresh").title = "정적 사이트: 데이터는 평일 미국 장중에 GitHub Actions가 자동 갱신";
}
$("#refresh").onclick = async () => {
  if (STATIC) {
    try {
      const st = await api("/api/status");
      alert(`${st.refresh_note}\n\n스냅샷 기준(UTC): ${st.snapshot_asof || "-"}\n종목 ${st.tickers.length}개: ${st.tickers.join(", ")}`);
    } catch (e) { alert(e.message); }
    return;
  }
  const b = $("#refresh"); b.disabled = true; b.textContent = "받는 중…";
  const onP2 = location.hash.includes("optimizer");
  try {
    await api("/api/refresh", { tickers: (onP2 ? P2.state.candidates : S.state.tickers).map(t => t.ticker) });
    if (onP2) { if (P2.result) await p2run(); } else await compute();
  }
  catch (e) { alert("새로고침 실패: " + e.message); }
  finally { b.disabled = false; b.textContent = "새로고침"; }
};
let rz = null;
window.addEventListener("resize", () => { clearTimeout(rz); rz = setTimeout(() => {
  const h = location.hash;
  if (h.includes("optimizer")) window.P2redraw?.();
  else if (h.includes("timeline")) { const el = $("#p3tl"); if (el) el._c = null; window.P3redraw?.(); }
  else if (S.result) renderResults();
}, 250); });
window.addEventListener("DOMContentLoaded", route);
