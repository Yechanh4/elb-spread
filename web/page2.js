/* Page 2 — inverse optimizer. Uses helpers from app.js ($, esc, pct, num, int, won, mdl, mkt, color,
   statusBadge, svgFor, axes, showTip, hideTip, api). The solve runs on the server; the slider moves
   between the 11 precomputed slider positions on the client, so dragging is instant. */
"use strict";

const P2KEY = "elb.page2.state.v2";
const P2DEF = {
  candidates: [{ ticker: "AMD" }, { ticker: "AMZN" }, { ticker: "GOOGL" }, { ticker: "BA" }, { ticker: "META" }, { ticker: "TSLA" }],
  scenarios: [{ name: "up", shift_bp: 75, prob: 0.25 }, { name: "neutral", shift_bp: 0, prob: 0.5 }, { name: "down", shift_bp: -75, prob: 0.25 }],
  thresholds_bp: [-50, 50],
  constraints: { k_min: 3, k_max: 5, max_per_sector: 2, min_oi: 100, max_spread_pct: 0.10 },
  globals: { r: null, G: 756e6, leg_cross: 1.0, tax: true, copula: "gaussian", n_paths: 3000, n_check: 10000, seed: 0,
    V_deposit: 1.5e9, V_own_rental: 3.8e8, T: 2 },
};
const SCN_KO = { up: "금리↑", neutral: "중립", down: "금리↓" };
const P2 = { state: p2load(), result: null, lam: 0, sel: null, screener: null, busy: false };

function p2load() {
  try {
    const s = JSON.parse(localStorage.getItem(P2KEY));
    if (s && s.candidates) return { ...JSON.parse(JSON.stringify(P2DEF)), ...s, globals: { ...P2DEF.globals, ...s.globals },
      constraints: { ...P2DEF.constraints, ...s.constraints } };
  } catch (e) { /* blocked storage: defaults */ }
  return JSON.parse(JSON.stringify(P2DEF));
}
function p2save() { try { localStorage.setItem(P2KEY, JSON.stringify(P2.state)); } catch (e) { /* ignore */ } }

// ------------------------------------------------------------------ skeleton
function renderOptimizer() {
  $("#view").innerHTML = `
  <div class="layout">
    <aside class="rail">
      <div class="card" id="p2scen"></div>
      <div class="card" id="p2cands"></div>
      <div class="card" id="p2screen"></div>
      <div class="card" id="p2cons"></div>
      <button class="btn primary" id="p2run" style="padding:8px">역최적화 실행</button>
      <div class="sub">후보 수·경로 수에 따라 수 초~수십 초. 슬라이더는 계산 후 즉시 반응.</div>
    </aside>
    <main class="canvas" id="p2out"><div class="panel empty">왼쪽에서 시나리오와 후보를 정하고 <b>역최적화 실행</b>.</div></main>
  </div>`;
  p2Scenarios(); p2Candidates(); p2Screener(); p2Constraints();
  $("#p2run").onclick = p2run;
  if (P2.result) p2render();
}
window.ROUTES.optimizer = renderOptimizer;
window.P2redraw = () => P2.result && p2render();

function p2budgets(r, shiftBp) {
  const g = P2.state.globals;
  return [0, 1, 2, 3].map(c => {
    const rc = Math.max(r + c * shiftBp / 1e4, 0.0025);
    return { r: rc, B: (g.V_deposit - g.V_own_rental) * (1 - Math.pow(1 + rc, -g.T)) };
  });
}
function p2Scenarios() {
  const st = P2.state, r = st.globals.r ?? P2.result?.globals?.r ?? 0.0376;
  const sum = st.scenarios.reduce((a, s) => a + (+s.prob || 0), 0);
  const maxB = Math.max(...st.scenarios.flatMap(s => p2budgets(r, s.shift_bp).map(x => x.B)));
  $("#p2scen").innerHTML = `<h2>금리 시나리오</h2>
    <div class="sub">분류: ↓ &lt; ${st.thresholds_bp[0]}bp ≤ 중립 ≤ ${st.thresholds_bp[1]}bp &lt; ↑ · 각 시나리오는 사이클마다 r을 옮기고 → B가 따라 바뀜</div>
    <div class="field"><label>하한 / 상한 (bp)</label><span style="display:flex;gap:4px">
      <input data-thr="0" type="number" step="5" value="${st.thresholds_bp[0]}"><input data-thr="1" type="number" step="5" value="${st.thresholds_bp[1]}"></span></div>
    <div class="field"><label data-tip="비우면 데이터 레이어 값(ECOS 없으면 3.76% 가정값)">현재 r (국고채 2년)</label><input data-p2g="r" type="number" step="0.0005" placeholder="자동" value="${st.globals.r ?? ""}"></div>
    ${st.scenarios.map((s, i) => `<h3>${SCN_KO[s.name] || s.name}</h3>
      <div class="field"><label>사이클당 이동 (bp)</label><input data-sc="${i}" data-k="shift_bp" type="number" step="5" value="${s.shift_bp}"></div>
      <div class="field"><label>확률</label><input data-sc="${i}" data-k="prob" type="number" step="0.05" min="0" max="1" value="${s.prob}"></div>
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:4px">${p2budgets(r, s.shift_bp).map((x, c) => `
        <div data-tip="사이클 ${c + 1}: r ${pct(x.r, 2)} → B ${won(x.B)}"><div style="height:${(6 + 30 * x.B / maxB).toFixed(0)}px;background:var(--s1);border-radius:3px 3px 0 0;transition:height .4s;margin-top:${(36 - 30 * x.B / maxB).toFixed(0)}px"></div>
        <div class="sub mdl" style="font-size:10.5px;text-align:center">${won(x.B)}</div></div>`).join("")}</div>`).join("")}
    ${Math.abs(sum - 1) > 1e-9 ? `<div class="note alert">확률 합 ${num(sum, 2)} ≠ 1</div>` : ""}`;
  $("#p2scen").querySelectorAll("[data-sc]").forEach(inp => inp.addEventListener("input", e => {
    st.scenarios[+e.target.dataset.sc][e.target.dataset.k] = +e.target.value; p2save(); p2Scenarios();
    const f = $(`#p2scen [data-sc="${e.target.dataset.sc}"][data-k="${e.target.dataset.k}"]`); f.focus(); f.setSelectionRange?.(99, 99);
  }));
  $("#p2scen").querySelectorAll("[data-thr]").forEach(inp => inp.addEventListener("change", e => { st.thresholds_bp[+e.target.dataset.thr] = +e.target.value; p2save(); p2Scenarios(); }));
  $("#p2scen").querySelector("[data-p2g=r]").addEventListener("change", e => { st.globals.r = e.target.value === "" ? null : +e.target.value; p2save(); p2Scenarios(); });
}

function p2Candidates() {
  const st = P2.state;
  const sv = (c, k, s) => (c[k] && typeof c[k] === "object" ? c[k][s] : c[k]) ?? "";
  $("#p2cands").innerHTML = `<h2>후보 (최대 10)</h2>
    <div class="search"><input id="p2q" placeholder="티커 추가 · Enter" autocomplete="off"></div>
    <div class="sub" style="margin:6px 0">σ̂·μ̂는 시나리오별(↑/중립/↓). 비우면 σ̂ = HV 1y 자리표시, μ̂ = 견해 없음.</div>
    ${st.candidates.map((c, i) => `<div style="border-left:3px solid ${color(i)};padding-left:8px;margin:8px 0">
      <div style="display:flex;align-items:center;gap:6px"><b>${esc(c.ticker)}</b><span class="sub" id="p2cinfo${i}"></span>
        <button class="x btn" data-cdel="${i}" style="margin-left:auto;padding:0 6px">×</button></div>
      ${["sigma", "mu"].map(k => `<div style="display:grid;grid-template-columns:28px repeat(3,1fr);gap:4px;margin:3px 0;font-size:12px">
        <span class="sub">${k === "sigma" ? "σ̂" : "μ̂"}</span>${P2.state.scenarios.map(s => `<input data-c="${i}" data-k="${k}" data-s="${s.name}" type="number" step="0.01"
          placeholder="${SCN_KO[s.name]}" value="${sv(c, k, s.name)}" style="width:100%;padding:2px 4px;border:1px solid var(--border);border-radius:4px;background:var(--surface-2)">`).join("")}</div>`).join("")}
      <div class="field"><label>확신도</label><input data-c="${i}" data-k="conf" type="number" step="0.05" min="0" max="1" value="${c.conf ?? 0.7}"></div></div>`).join("")}`;
  const q = $("#p2q");
  q.addEventListener("keydown", e => {
    if (e.key !== "Enter") return;
    const t = q.value.trim().toUpperCase();
    if (t && !st.candidates.some(c => c.ticker === t) && st.candidates.length < 10) { st.candidates.push({ ticker: t }); p2save(); p2Candidates(); }
  });
  $("#p2cands").querySelectorAll("[data-cdel]").forEach(b => b.onclick = () => { st.candidates.splice(+b.dataset.cdel, 1); p2save(); p2Candidates(); });
  $("#p2cands").querySelectorAll("[data-c]").forEach(inp => inp.addEventListener("change", e => {
    const c = st.candidates[+e.target.dataset.c], k = e.target.dataset.k, v = e.target.value === "" ? null : +e.target.value;
    if (k === "conf") c.conf = v ?? 0.7;
    else {
      const o = (c[k] && typeof c[k] === "object") ? c[k] : {};
      o[e.target.dataset.s] = v;
      c[k] = Object.values(o).some(x => x !== null && x !== undefined) ? o : null;
    }
    p2save();
  }));
  p2CandInfo();
}
function p2CandInfo() {
  if (!P2.result) return;
  P2.state.candidates.forEach((c, i) => {
    const r = P2.result.candidates.find(x => x.ticker === c.ticker.toUpperCase()), el = $("#p2cinfo" + i);
    if (!r || !el) return;
    el.innerHTML = r.eligible ? `<span class="badge good">Edge ${mdl("+" + pct(r.edge_mix))}</span>` : `<span class="badge bad" data-tip="${esc(r.reason)}">제외</span>`;
  });
}

function p2Screener() {
  const S2 = P2.screener;
  $("#p2screen").innerHTML = `<h2>스크리너 <button class="btn" id="p2scr" style="margin-left:auto">${STATIC ? "전체 종목 스크리닝" : "워치리스트 스크리닝"}</button></h2>
    <div class="sub">옵션이 싼 순: (ATM IV − HV 1y) 오름차순, 그다음 IV 순위(proxy) 오름차순.${S2 && S2.total ? ` 전체 ${S2.total}개 중 상위 ${S2.rows.length}개만 표시 (데이터 갱신 때 미리 계산). 목록에 없어도 위 후보 칸에 티커를 치면 S&P 500·나스닥 100 등 사이트의 모든 종목을 넣을 수 있음.` : ""}
      섹터당 최대 개수 제약이 있으니 섹터를 섞어서 추가해줘.</div>
    ${S2 ? `<div class="scroll"><table class="t"><tr><th>종목</th><th>IV−HV</th><th data-tip="${esc(S2.note)}">IV 순위*</th><th>섹터</th><th></th></tr>
      ${S2.rows.map(r => `<tr><td>${esc(r.ticker)}${r.stale ? ' <span class="badge bad">STALE</span>' : ""}</td><td>${mdl((r.iv_minus_hv >= 0 ? "+" : "") + pct(r.iv_minus_hv))}</td>
        <td>${mdl(pct(r.iv_rank_proxy, 0))}</td><td class="sub">${esc(r.sector)}</td>
        <td>${P2.state.candidates.some(c => c.ticker === r.ticker) ? "✓" : `<button class="btn" data-add="${esc(r.ticker)}" style="padding:0 6px">+</button>`}</td></tr>`).join("")}</table></div>
      ${Object.entries(S2.errors).map(([t, e]) => `<div class="note">${esc(t)}: ${esc(e)}</div>`).join("")}
      <div class="sub">* IV 순위 = 오늘 ATM IV가 지난 1년 21일 실현변동성 분포에서 어디쯤인지. 무료 데이터엔 IV 이력이 없어 대리지표.</div>` : ""}`;
  $("#p2scr").onclick = async () => {
    $("#p2scr").textContent = "받는 중…";
    try { P2.screener = await api("/api/screener", {}); } catch (e) { alert("스크리너 실패: " + e.message); }
    p2Screener();
  };
  $("#p2screen").querySelectorAll("[data-add]").forEach(b => b.onclick = () => {
    if (P2.state.candidates.length >= 10) return;
    P2.state.candidates.push({ ticker: b.dataset.add }); p2save(); p2Candidates(); p2Screener();
  });
}

function p2Constraints() {
  const c = P2.state.constraints, g = P2.state.globals;
  const f = (obj, k, label, attrs, tip) => `<div class="field"><label data-tip="${esc(tip || "")}">${label}</label><input data-obj="${obj}" data-k="${k}" ${attrs} value="${(obj === "c" ? c : g)[k]}"></div>`;
  $("#p2cons").innerHTML = `<h2>제약 · 설정</h2>
    ${f("c", "k_min", "종목 수 최소", 'type="number" min="1" max="5"', "설계 3–5. 많을수록 분포가 평균으로 모임")}
    ${f("c", "k_max", "종목 수 최대", 'type="number" min="1" max="5"')}
    ${f("c", "max_per_sector", "섹터당 최대", 'type="number" min="1" max="5"', "GICS 섹터당 최대 종목 수")}
    ${f("c", "min_oi", "최소 OI", 'type="number" step="10"', "K1·K2 미결제약정 하한")}
    ${f("c", "max_spread_pct", "최대 호가폭 (mid 대비)", 'type="number" step="0.01"', "다리별 (ask−bid)/mid 상한")}
    <div class="field"><label>목표 G</label><select data-obj="g" data-k="G">${[[756e6, "756M"], [500e6, "500M"], [300e6, "300M"], [70e6, "70M"]].map(([v, t]) =>
      `<option value="${v}" ${+g.G === v ? "selected" : ""}>${t}</option>`).join("")}</select></div>
    ${f("g", "leg_cross", "다리 호가 지불 비율", 'type="number" step="0.05" min="0.2" max="1"', "1 = 진입·청산 때 두 다리 호가를 전부 넘김")}
    ${f("g", "n_paths", "탐색 경로 수", 'type="number" step="500" min="500" max="20000"', "후보 조합 비교용 (공통 난수)")}
    ${f("g", "n_check", "확인 경로 수", 'type="number" step="1000" min="1000" max="50000"', "선택된 조합만 다른 난수로 재평가 — 탐색 잡음 확인")}
    <div class="field"><label>의존구조</label><select data-obj="g" data-k="copula"><option value="gaussian" ${g.copula === "gaussian" ? "selected" : ""}>가우시안</option><option value="t" ${g.copula === "t" ? "selected" : ""}>t-copula (df=5)</option></select></div>
    <div class="field"><label>세금 반영</label><input type="checkbox" data-obj="g" data-k="tax" ${g.tax ? "checked" : ""}></div>`;
  $("#p2cons").querySelectorAll("[data-obj]").forEach(inp => inp.addEventListener("change", e => {
    const o = e.target.dataset.obj === "c" ? c : g, k = e.target.dataset.k;
    o[k] = e.target.type === "checkbox" ? e.target.checked : (k === "copula" ? e.target.value : +e.target.value);
    p2save();
  }));
}

async function p2run() {
  if (P2.busy) return;
  P2.busy = true;
  const b = $("#p2run"); b.disabled = true; b.textContent = "계산 중…"; $("#busy").style.display = "block";
  try {
    P2.result = await api("/api/page2", P2.state);
    P2.sel = null;
    p2render(); p2Scenarios(); p2CandInfo();
  } catch (e) {
    $("#p2out").innerHTML = `<div class="panel banner">역최적화 실패: ${esc(e.message)}</div>`;
  } finally {
    P2.busy = false; b.disabled = false; b.textContent = "역최적화 실행"; $("#busy").style.display = "none";
  }
}

// ------------------------------------------------------------------ results
function p2pick() {
  const R = P2.result;
  return R.picks.reduce((a, p) => Math.abs(p.lam - P2.lam) < Math.abs(a.lam - P2.lam) ? p : a, R.picks[0]);
}
function p2render() {
  const R = P2.result;
  const cand = Object.fromEntries(R.candidates.map(c => [c.ticker, c]));
  $("#p2out").innerHTML = `
    ${R.warnings.map(w => `<div class="note alert">⚠ ${esc(w)}</div>`).join("")}
    <section class="panel"><h2>후보 — 시나리오별 견해와 구조</h2>
      <div class="sub">K1은 σ̂ 규칙(시나리오 확률가중 σ̂), K2는 혼합 분포 p_mix 기준 비용 반영 폭. E[M] = 롤당 기대배수(비용 포함).</div>
      <div class="scroll"><table class="t"><tr><th>종목</th><th>섹터</th><th>ATM IV<span class="mkt-tag">mkt</span></th><th>HV 1y</th>
        ${R.scenarios.map(s => `<th>σ̂ ${SCN_KO[s.name]}</th>`).join("")}<th>Edge (가중)</th><th>K1/K2</th><th>왕복비용</th>
        ${R.scenarios.map(s => `<th>E[M] ${SCN_KO[s.name]}</th>`).join("")}<th>KL</th><th>상태</th></tr>
      ${R.candidates.map(c => c.K1 ? `<tr><td>${esc(c.ticker)}</td><td class="sub">${esc(c.sector)}</td><td>${pct(c.iv_atm)}</td><td>${mdl(pct(c.hv1y))}</td>
        ${R.scenarios.map(s => `<td>${mdl(pct(c.sigma[s.name]))}</td>`).join("")}
        <td>${statusBadge(c.edge_mix > 0 ? "pass" : "fail", (c.edge_mix >= 0 ? "+" : "") + pct(c.edge_mix))}</td>
        <td>${num(c.K1, 1)}/${num(c.K2, 1)}</td><td>${mdl(pct(c.rt_cost, 0))}</td>
        ${R.scenarios.map(s => `<td>${mdl(num(c.e_multiple[s.name], 3))}</td>`).join("")}<td>${mdl(num(c.kl_mix, 3))}</td>
        <td>${c.eligible ? statusBadge("pass", "포함") : `<span data-tip="${esc(c.reason)}">${statusBadge("fail", "제외")}</span>`}${c.sigma_placeholder ? ' <span class="badge warn" data-tip="리서치 σ̂ 없음: HV 1y 자리표시">σ̂ 자리표시</span>' : ""}${c.liquidity_offhours ? ' <span class="badge warn" data-tip="장 마감 후 호가: 호가폭 기준 보류, OI만 확인. 비용은 넓은 호가로 계산(보수적)">장외: OI만</span>' : ""}</td></tr>`
        : `<tr><td>${esc(c.ticker)}</td><td colspan="20">${statusBadge("fail", "제외")} ${esc(c.reason)}</td></tr>`).join("")}</table></div></section>
    ${R.error ? `<section class="panel banner">${esc(R.error)}</section>` : `
    <section class="panel"><h2>① 트레이드오프 — 목표확률 ↔ 기대수익</h2>
      <div class="sub">점 하나 = 평가한 조합(종목·비중·f). 파란 선 = 파레토 프런티어. 주황 = 슬라이더 위치의 최적 조합. 점을 클릭하면 그 조합을 아래에 불러옴.
        탐색 ${int(R.n_paths)}경로 · 평가 ${int(R.n_evals)}회 · 공통 난수.</div>
      <div style="display:grid;grid-template-columns:auto 1fr auto;gap:10px;align-items:center;margin:8px 0">
        <span class="sub">목표확률 최대</span><input type="range" id="p2lam" min="0" max="1" step="0.1" value="${P2.lam}">
        <span class="sub">기대수익 최대</span></div>
      <div class="sub" style="text-align:center" id="p2lamtxt"></div>
      <div class="chart" id="p2front"></div></section>
    <div class="grid2">
      <section class="panel" id="p2card"></section>
      <section class="panel"><h2>② 섹터 배분 <span class="sub">크기 = 투입 자본, 색 = Edge</span></h2><div class="chart" id="p2tree"></div></section>
    </div>
    <section class="panel"><h2>③ 시나리오 비교</h2><div class="sub">모두 목표확률 최대화(λ=0) 기준. 확인 ${int(R.n_check)}경로(다른 난수) 값.</div>
      <div class="scroll">${p2table(R)}</div>
      <div class="note"><b>권장: 혼합 최적.</b> 내가 매긴 시나리오 확률 아래에서 목표확률을 최대화함. min-max는 최악 시나리오에 대한 보험이지만, 더 그럴듯하다고 본 시나리오의 확률을 그만큼 포기함.</div></section>
    <section class="panel"><h2>④ 민감도 — 현재 조합</h2><div class="sub" id="p2senssub"></div><div class="grid2" id="p2sens"></div></section>`}`;
  if (R.error) return;
  $("#p2lam").addEventListener("input", e => { P2.lam = +e.target.value; P2.sel = null; p2update(true); });
  p2update(false);
}
function p2current() {
  const R = P2.result;
  if (P2.sel !== null) return { ...R.points[P2.sel], fromClick: true };
  return p2pick();
}
function p2update(animate) {
  const R = P2.result, p = p2pick();
  $("#p2lamtxt").innerHTML = `λ = ${num(P2.lam, 1)} · objective = (1−λ)·정규화 P(목표) + λ·정규화 E[누적]`;
  p2frontier(R, p, animate);
  p2card(R);
  p2tree(R);
  p2sens(R, p);
}

function p2frontier(R, pick, animate) {
  const host = $("#p2front");
  let c = host._c;
  const pts = R.points;
  if (!c) {
    host.innerHTML = "";
    c = svgFor(host, 320, { t: 14, r: 20, b: 36, l: 56 });
    host._c = c;
    const x = d3.scaleLinear(d3.extent(pts, d => d.mean), [0, c.iw]).nice();
    const [plo, phi] = d3.extent(pts, d => d.p_goal);
    const pad = Math.max((phi - plo) * 0.15, 0.005);
    const y = d3.scaleLinear([Math.max(0, plo - pad), phi + pad], [c.ih, 0]).nice();   // scatter: zoom to the data
    c.x = x; c.y = y;
    axes(c, x, y, won, d3.format(".0%"), 6, 5);
    c.g.append("text").attr("class", "lbl").attr("x", c.iw).attr("y", c.ih + 30).attr("text-anchor", "end").text("x = 기대 누적 회수액 E (세후)");
    c.g.append("text").attr("class", "lbl").attr("x", 4).attr("y", -2).text("y = P(누적 ≥ G)");
    const fr = pts.map((d, i) => ({ ...d, i })).filter(d => d.frontier).sort((a, b) => a.mean - b.mean);
    c.g.selectAll(".pt").data(pts.map((d, i) => ({ ...d, i }))).join("circle").attr("class", "pt")
      .attr("cx", d => x(d.mean)).attr("cy", d => y(d.p_goal)).attr("r", d => d.frontier ? 5 : 3.5)
      .attr("fill", d => d.frontier ? "var(--s1)" : "var(--muted)").attr("fill-opacity", d => d.frontier ? 1 : 0.45)
      .attr("stroke", "var(--surface)").attr("stroke-width", 1.5).style("cursor", "pointer")
      .on("mousemove", (ev, d) => showTip(ev, `${d.names.map((n, k) => `${esc(n)} ${pct(d.weights[k], 0)}`).join(" · ")}<br>f ${num(d.f, 2)}
        <br>P(목표) <i>${pct(d.p_goal)}</i> · E <i>${won(d.mean)}</i><br>중앙값 <i>${won(d.median)}</i> · 파산 <i>${pct(d.ruin)}</i>${d.frontier ? "<br>파레토 프런티어" : ""}`))
      .on("mouseleave", hideTip)
      .on("click", (ev, d) => { P2.sel = d.i; p2card(R); p2tree(R); p2sens(R, null); c.sel.attr("cx", x(d.mean)).attr("cy", y(d.p_goal)).style("display", null); });
    c.g.append("path").datum(fr).attr("fill", "none").attr("stroke", "var(--s1)").attr("stroke-width", 2)
      .attr("d", d3.line().x(d => x(d.mean)).y(d => y(d.p_goal)));
    c.sel = c.g.append("circle").attr("r", 9).attr("fill", "none").attr("stroke", "var(--ink)").attr("stroke-width", 1.5).style("display", "none");
    c.cur = c.g.append("circle").attr("r", 8).attr("fill", "var(--s2)").attr("stroke", "var(--surface)").attr("stroke-width", 2).style("pointer-events", "none");
    c.curLbl = c.g.append("text").attr("class", "lbl").attr("text-anchor", "middle");
  }
  if (P2.sel === null) c.sel.style("display", "none");
  const pt = R.points[pick.point];
  const t = animate ? c.cur.transition().duration(450) : c.cur;
  t.attr("cx", c.x(pt.mean)).attr("cy", c.y(pt.p_goal));
  (animate ? c.curLbl.transition().duration(450) : c.curLbl).attr("x", c.x(pt.mean)).attr("y", c.y(pt.p_goal) - 13).text(`λ=${num(P2.lam, 1)}`);
}

function p2combo(R, e) {
  const cand = Object.fromEntries(R.candidates.map(c => [c.ticker, c]));
  const B0 = R.budgets.neutral ? R.budgets.neutral.budgets[0] : Object.values(R.budgets)[0].budgets[0];
  return e.names.map((n, k) => {
    const c = cand[n], w = e.weights[k];
    return { ...c, w, n: B0 * w / (c.net_cost * 100 * R.macro.usdkrw), capital: B0 * w };
  });
}
function p2card(R) {
  const e = p2current(), rows = p2combo(R, e), ck = e.check;
  const chk = (a, b, f) => ck ? `${mdl(f(a))} <span class="sub">/ 확인 ${mdl(f(b))}</span>` : mdl(f(a));
  $("#p2card").innerHTML = `<h2>조합 ${e.fromClick ? '<span class="badge info">클릭한 점</span>' : `<span class="badge info">λ = ${num(P2.lam, 1)}</span>`}</h2>
    <div class="tiles">
      <div class="tile"><div class="k">P(목표, 혼합)</div><div class="v">${chk(e.p_goal, ck?.p_goal, pct)}</div><div class="s">최악 시나리오 ${mdl(pct(e.p_worst))}</div></div>
      <div class="tile"><div class="k">사이클 파산확률</div><div class="v">${chk(e.ruin, ck?.ruin, pct)}</div></div>
      <div class="tile"><div class="k">중앙값</div><div class="v">${chk(e.median, ck?.median, won)}</div></div>
      <div class="tile"><div class="k">기대 누적 E</div><div class="v">${chk(e.mean, ck?.mean, won)}</div><div class="s">${e.mean > 2 * Math.max(e.median, 0) ? "⚠ 평균 > 2×중앙값: 소수 결과가 끌어올림" : ""}</div></div>
    </div>
    <div class="sub">스테이킹 f = ${mdl(num(e.f, 2))} · 시나리오별 P(목표): ${Object.entries(e.p_by_scenario).map(([s, v]) => `${SCN_KO[s] || s} ${mdl(pct(v))}`).join(" · ")}</div>
    <div class="scroll"><table class="t"><tr><th>종목</th><th>섹터</th><th>비중</th><th>K1/K2</th><th>계약 n (사이클1)</th><th>투입 자본</th><th>Edge</th></tr>
      ${rows.map((r, k) => `<tr><td><span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:${color(k)};margin-right:6px"></span>${esc(r.ticker)}</td>
        <td class="sub">${esc(r.sector)}</td><td>${mdl(pct(r.w, 0))}</td><td>${num(r.K1, 1)}/${num(r.K2, 1)}</td><td>${mdl(int(r.n))}</td><td>${mdl(won(r.capital))}</td>
        <td>${mdl("+" + pct(r.edge_mix))}</td></tr>`).join("")}</table></div>
    ${ck ? `<div class="sub">앞 = 탐색 ${int(R.n_paths)}경로(선택에 쓴 난수), 뒤 = 다른 난수 ${int(R.n_check)}경로로 재평가. 차이가 크면 탐색 잡음.</div>` : `<div class="sub">클릭한 점은 탐색 경로 값만 있음 (재평가는 슬라이더 위치의 조합만).</div>`}`;
}

function p2tree(R) {
  const host = $("#p2tree"); host.innerHTML = "";
  const e = p2current(), rows = p2combo(R, e);
  const bySec = d3.group(rows, r => r.sector);
  const root = d3.hierarchy({ children: [...bySec].map(([s, rs]) => ({ name: s, children: rs.map(r => ({ name: r.ticker, value: r.capital, r })) })) })
    .sum(d => d.value || 0);
  const c = svgFor(host, 240, { t: 4, r: 4, b: 4, l: 4 });
  d3.treemap().size([c.iw, c.ih]).paddingInner(2).paddingTop(16).round(true)(root);
  const emax = d3.max(rows, r => r.edge_mix) || 0.1;
  const col = d3.scaleSequential([0, emax], t => d3.interpolateRgb("#cde2fb", "#184f95")(t));
  c.g.selectAll(".sec").data(root.children || []).join("text").attr("class", "lbl").attr("x", d => d.x0 + 3).attr("y", d => d.y0 + 12).text(d => d.data.name);
  const leaf = c.g.selectAll(".lf").data(root.leaves()).join("g").attr("transform", d => `translate(${d.x0},${d.y0})`);
  leaf.append("rect").attr("width", d => Math.max(0, d.x1 - d.x0)).attr("height", d => Math.max(0, d.y1 - d.y0)).attr("rx", 4)
    .attr("fill", d => col(d.data.r.edge_mix))
    .on("mousemove", (ev, d) => showTip(ev, `${esc(d.data.name)} · ${esc(d.data.r.sector)}<br>자본 <i>${won(d.data.value)}</i> (${pct(d.data.r.w, 0)})<br>Edge <i>+${pct(d.data.r.edge_mix)}</i>`)).on("mouseleave", hideTip);
  leaf.append("text").attr("x", 6).attr("y", 18).style("fill", d => d.data.r.edge_mix > emax * 0.55 ? "#ffffff" : "#0b0b0b").style("font-size", "12px").style("font-weight", 600)
    .text(d => (d.x1 - d.x0 > 46 ? d.data.name : ""));
  leaf.append("text").attr("x", 6).attr("y", 33).style("fill", d => d.data.r.edge_mix > emax * 0.55 ? "#ffffff" : "#0b0b0b").style("font-size", "11px")
    .text(d => (d.x1 - d.x0 > 60 ? pct(d.data.r.w, 0) : ""));
  host.insertAdjacentHTML("beforeend", `<div class="sub">색: Edge 0 → ${pct(emax)} (진할수록 큼)</div>`);
}

function p2table(R) {
  const order = [...R.scenarios.map(s => s.name), "mixture", "minmax"];
  const lab = { mixture: "혼합 최적 (권장)", minmax: "min-max (최악 최대화)" };
  return `<table class="t"><tr><th>목적</th><th>조합 (비중)</th><th>f</th>${R.scenarios.map(s => `<th>P ${SCN_KO[s.name]}</th>`).join("")}
    <th>P 혼합</th><th>P 최악</th><th>E 혼합</th><th>중앙값</th></tr>
    ${order.filter(k => R.table[k]).map(k => { const e = R.table[k], ck = e.check || e; return `<tr style="${k === "mixture" ? "font-weight:650" : ""}">
      <td>${lab[k] || "시나리오 " + (SCN_KO[k] || k) + " 최적"}</td><td>${e.names.map((n, i) => `${esc(n)} ${pct(e.weights[i], 0)}`).join(" · ")}</td><td>${num(e.f, 2)}</td>
      ${R.scenarios.map(s => `<td>${mdl(pct(ck.p_by_scenario[s.name]))}</td>`).join("")}
      <td>${mdl(pct(ck.p_goal))}</td><td>${mdl(pct(ck.p_worst))}</td><td>${mdl(won(ck.mean))}</td><td>${mdl(won(ck.median))}</td></tr>`; }).join("")}</table>`;
}

function p2sens(R, pick) {
  const host = $("#p2sens"); host.innerHTML = "";
  if (!pick) { $("#p2senssub").textContent = "민감도는 슬라이더 위치의 조합에 대해서만 계산됨 (클릭한 점은 슬라이더를 움직이면 돌아옴)."; return; }
  const s = pick.sensitivity;
  $("#p2senssub").textContent = `P(목표) 변화. σ̂는 모든 시나리오에서 ±5%p (K1/K2 고정), 왕복비용은 mid 대비 고정 %로 대체, 종목 수는 ${s.count_template}을 상관 ${num(s.count_rho, 2)}로 복제.`;
  const groups = [["σ̂ 이동", s.sigma, k => (+k >= 0 ? "+" : "") + (+k * 100).toFixed(0) + "%p", "0.0"],
    ["왕복비용 (주황 = 현재 체인 호가 비용)", s.rt_cost, k => k === "chain" ? "체인" : (+k * 100).toFixed(0) + "%", "chain"], ["종목 수", s.count, k => k, null], ["스테이킹 f", s.f, k => num(+k, 2), String(pick.f)]];
  const ymax = d3.max(groups.flatMap(g => Object.values(g[1]).map(v => v.p_goal))) * 1.15 || 0.1;
  groups.forEach(([title, data, fmt, base]) => {
    const div = document.createElement("div"); div.className = "chart"; host.appendChild(div);
    div.insertAdjacentHTML("beforeend", `<div class="sub"><b style="color:var(--ink)">${title}</b></div>`);
    const keys = Object.keys(data).sort((a, b) => (isNaN(+a) ? 1e9 : +a) - (isNaN(+b) ? 1e9 : +b));
    const c = svgFor(div, 150, { t: 14, r: 8, b: 24, l: 40 });
    const x = d3.scaleBand(keys, [0, c.iw]).padding(0.3), y = d3.scaleLinear([0, ymax], [c.ih, 0]);
    c.g.append("g").attr("class", "gridl").call(d3.axisLeft(y).ticks(3).tickSize(-c.iw).tickFormat(""));
    c.g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(3).tickFormat(d3.format(".0%")));
    c.g.append("g").attr("class", "axis").attr("transform", `translate(0,${c.ih})`).call(d3.axisBottom(x).tickFormat(fmt));
    c.g.selectAll("rect").data(keys).join("rect").attr("x", k => x(k)).attr("width", x.bandwidth()).attr("y", k => y(data[k].p_goal))
      .attr("height", k => c.ih - y(data[k].p_goal)).attr("rx", 3)
      .attr("fill", k => (base !== null && (k === base || Math.abs(+k - +base) < 1e-9)) ? "var(--s2)" : "var(--s1)")
      .on("mousemove", (ev, k) => showTip(ev, `${title} ${fmt(k)}<br>P(목표) <i>${pct(data[k].p_goal)}</i><br>중앙값 <i>${won(data[k].median)}</i> · E <i>${won(data[k].mean)}</i><br>파산 <i>${pct(data[k].ruin)}</i>`))
      .on("mouseleave", hideTip);
    c.g.selectAll(".v").data(keys).join("text").attr("class", "lbl").attr("x", k => x(k) + x.bandwidth() / 2).attr("y", k => y(data[k].p_goal) - 3)
      .attr("text-anchor", "middle").text(k => pct(data[k].p_goal, 0));
  });
}
