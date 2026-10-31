/* Page 3 — 8-year execution timeline, exports. Uses helpers from app.js and page2.js. */
"use strict";

const P3KEY = "elb.page3.state.v2";
const P3 = { state: p3load(), seed: null, result: null, scen: "neutral", rep: "median", roll: null, busy: false };
const REP_KO = { median: "중앙값 경로", goal: "목표 달성 경로", low: "하위 10% 경로" };

function p3default() {
  return {
    source: "manual", tickers: [{ ticker: "AMD" }, { ticker: "AMZN" }, { ticker: "GOOGL" }], weights: null, f: [1, 1, 1, 1],
    scenarios: JSON.parse(JSON.stringify(P2DEF.scenarios)),
    globals: { G: 756e6, r: null, n_paths: 20000, start: "2026-12-01", tax: true, copula: "gaussian", seed: 0,
      broker_confirmed: null, V_deposit: 1.5e9, V_own_rental: 3.8e8, T: 2, mortgage_rate: 0.04, remaining_months: 264 },
  };
}
function p3load() {
  try { const s = JSON.parse(localStorage.getItem(P3KEY)); if (s && s.state) { return s.state; } } catch (e) { /* defaults */ }
  return p3default();
}
function p3save() { try { localStorage.setItem(P3KEY, JSON.stringify({ state: P3.state, seed: P3.seed })); } catch (e) { /* ignore */ } }
try { P3.seed = JSON.parse(localStorage.getItem(P3KEY))?.seed || null; } catch (e) { P3.seed = null; }

// ------------------------------------------------------------------ seeding from page 1 / page 2
function seedFromPage1() {
  const st = S.state, R = S.result;
  const inc = R?.portfolio?.included || st.tickers.map(t => t.ticker.toUpperCase());
  const w = R?.portfolio?.weights || inc.map(() => 1 / inc.length);
  const tickers = st.tickers.map(t => ({ ticker: t.ticker.toUpperCase(), sigma: t.sigma_hat ?? null, mu: t.mu_hat ?? null, conf: t.conf ?? 0.7 }));
  const row = tickers.map(t => { const k = inc.indexOf(t.ticker); return k >= 0 ? w[k] : 0; });
  const g = st.globals;
  return { source: "page1", tickers, weights: [row, row, row, row].map(r => [...r]), f: [g.f, g.f, g.f, g.f],
    scenarios: JSON.parse(JSON.stringify(P2.state.scenarios)),
    globals: { ...p3default().globals, G: g.G, r: g.r, start: g.start, tax: g.tax, copula: g.copula, broker_confirmed: g.broker_confirmed,
      V_deposit: g.V_deposit, V_own_rental: g.V_own_rental, T: g.T, mortgage_rate: g.mortgage_rate, remaining_months: g.remaining_months } };
}
function seedFromPage2() {
  const R = P2.result; if (!R || R.error) return null;
  const p = R.picks.reduce((a, x) => Math.abs(x.lam - P2.lam) < Math.abs(a.lam - P2.lam) ? x : a, R.picks[0]);
  const cand = Object.fromEntries(P2.state.candidates.map(c => [c.ticker.toUpperCase(), c]));
  const tickers = p.names.map(n => ({ ticker: n, sigma: cand[n]?.sigma ?? null, mu: cand[n]?.mu ?? null, conf: cand[n]?.conf ?? 0.7 }));
  return { source: "page2", tickers, weights: [0, 1, 2, 3].map(() => [...p.weights]), f: [p.f, p.f, p.f, p.f],
    scenarios: JSON.parse(JSON.stringify(R.scenarios)),
    globals: { ...p3default().globals, G: R.globals.G, r: P2.state.globals.r, tax: R.globals.tax, copula: R.globals.copula } };
}

// ------------------------------------------------------------------ skeleton
function renderTimeline() {
  $("#view").innerHTML = `
  <div class="layout">
    <aside class="rail">
      <div class="card"><h2>불러오기</h2>
        <div style="display:flex;gap:6px;flex-wrap:wrap">
          <button class="btn" id="p3l1" data-tip="Page 1의 종목·σ̂·μ̂·비중·f·전역 설정">Page 1에서</button>
          <button class="btn" id="p3l2" data-tip="Page 2 슬라이더 위치의 조합">Page 2에서</button>
          <button class="btn" id="p3reset" data-tip="불러온 값으로 되돌리기 (편집 취소)">시뮬레이션 값으로 리셋</button></div>
        <div class="sub" id="p3src" style="margin-top:6px"></div></div>
      <div class="card" id="p3edit"></div>
      <button class="btn primary" id="p3run" style="padding:8px">타임라인 계산</button>
      <div class="card"><h2>내보내기 · 공유</h2>
        <div style="display:flex;gap:6px;flex-wrap:wrap">
          <button class="btn" id="x_sched">롤 스케줄 CSV</button><button class="btn" id="x_order">주문서 CSV</button>
          <button class="btn" id="x_svg">타임라인 SVG</button><button class="btn" id="x_png">타임라인 PNG</button>
          <button class="btn" id="x_json">입력 JSON</button><label class="btn">JSON 불러오기<input type="file" id="x_load" accept=".json" hidden></label></div>
        <div class="sub" style="margin-top:6px">JSON은 모든 입력의 스냅샷 — 같은 데이터·시드로 같은 결과가 재현됨. 링크 공유는 상단 <b>공유</b>.</div></div>
    </aside>
    <main class="canvas" id="p3out"><div class="panel empty">Page 1 또는 Page 2에서 불러온 뒤 <b>타임라인 계산</b>.</div></main>
  </div>`;
  $("#p3l1").onclick = () => { P3.state = seedFromPage1(); P3.seed = JSON.parse(JSON.stringify(P3.state)); p3save(); p3edit(); p3run(); };
  $("#p3l2").onclick = () => {
    const s = seedFromPage2();
    if (!s) { alert("Page 2에서 먼저 역최적화를 실행하세요."); return; }
    P3.state = s; P3.seed = JSON.parse(JSON.stringify(s)); p3save(); p3edit(); p3run();
  };
  $("#p3reset").onclick = () => { if (!P3.seed) return; P3.state = JSON.parse(JSON.stringify(P3.seed)); p3save(); p3edit(); p3run(); };
  $("#p3run").onclick = p3run;
  $("#x_sched").onclick = exportSchedule; $("#x_order").onclick = exportOrders;
  $("#x_svg").onclick = () => exportImage("svg"); $("#x_png").onclick = () => exportImage("png");
  $("#x_json").onclick = () => download("elb_inputs.json", JSON.stringify({ page: "timeline", state: P3.state }, null, 2), "application/json");
  $("#x_load").onchange = async e => {
    const f = e.target.files[0]; if (!f) return;
    try { const j = JSON.parse(await f.text()); P3.state = j.state || j; P3.seed = JSON.parse(JSON.stringify(P3.state)); p3save(); p3edit(); p3run(); }
    catch (err) { alert("JSON을 읽지 못함: " + err.message); }
  };
  p3edit();
  if (P3.result) p3render(); else if (P3.state.tickers.length) p3run();
}
window.ROUTES.timeline = renderTimeline;
window.P3redraw = () => P3.result && p3render();

function p3edit() {
  const st = P3.state, n = st.tickers.length;
  if (!st.weights || st.weights.length !== 4 || st.weights[0].length !== n) st.weights = [0, 1, 2, 3].map(() => st.tickers.map(() => 1 / n));
  if (!Array.isArray(st.f)) st.f = [st.f, st.f, st.f, st.f];
  const edited = P3.seed && JSON.stringify(P3.seed) !== JSON.stringify(st);
  $("#p3src").innerHTML = `출처: <b>${{ page1: "Page 1", page2: "Page 2", manual: "직접 입력" }[st.source] || st.source}</b>${edited ? ' <span class="badge warn">편집됨</span>' : ""}`;
  const sv = (t, k, s) => (t[k] && typeof t[k] === "object" ? t[k][s] : t[k]) ?? "";
  const inp = (attrs, v) => `<input ${attrs} value="${v}" style="width:100%;padding:2px 4px;border:1px solid var(--border);border-radius:4px;background:var(--surface-2);font-size:12px">`;
  $("#p3edit").innerHTML = `<h2>편집 (모든 값)</h2>
    <h3>사이클별 비중 (%) · f</h3>
    <div style="display:grid;grid-template-columns:64px repeat(4,1fr);gap:4px;font-size:12px;align-items:center">
      <span></span>${[1, 2, 3, 4].map(c => `<span class="sub" style="text-align:center">C${c}</span>`).join("")}
      ${st.tickers.map((t, i) => `<b style="color:${color(i)}">${esc(t.ticker)}</b>${[0, 1, 2, 3].map(c => inp(`data-w="${c},${i}" type="number" step="5" min="0" max="100"`, Math.round(st.weights[c][i] * 100))).join("")}`).join("")}
      <span class="sub">f</span>${[0, 1, 2, 3].map(c => inp(`data-f="${c}" type="number" step="0.05" min="0" max="1"`, st.f[c])).join("")}
    </div>
    <h3>시나리오별 σ̂ / μ̂ (비우면 HV 1y / 견해 없음)</h3>
    ${st.tickers.map((t, i) => ["sigma", "mu"].map(k => `<div style="display:grid;grid-template-columns:64px repeat(3,1fr);gap:4px;margin:2px 0;font-size:12px">
      <span class="sub">${k === "sigma" ? esc(t.ticker) + " σ̂" : "μ̂"}</span>${st.scenarios.map(s => inp(`data-v="${i},${k},${s.name}" type="number" step="0.01" placeholder="${SCN_KO[s.name]}"`, sv(t, k, s.name))).join("")}</div>`).join("")).join("")}
    <h3>금리 시나리오 (사이클당 bp / 확률)</h3>
    ${st.scenarios.map((s, i) => `<div style="display:grid;grid-template-columns:64px 1fr 1fr;gap:4px;margin:2px 0;font-size:12px"><span class="sub">${SCN_KO[s.name]}</span>
      ${inp(`data-s="${i},shift_bp" type="number" step="5"`, s.shift_bp)}${inp(`data-s="${i},prob" type="number" step="0.05"`, s.prob)}</div>`).join("")}
    <h3>전역</h3>
    <div class="field"><label>목표 G</label><select data-g3="G">${[[756e6, "756M"], [500e6, "500M"], [300e6, "300M"], [70e6, "70M"]].map(([v, t]) => `<option value="${v}" ${+st.globals.G === v ? "selected" : ""}>${t}</option>`).join("")}</select></div>
    <div class="field"><label>국고채 2년 r</label><input data-g3="r" type="number" step="0.0005" placeholder="자동" value="${st.globals.r ?? ""}"></div>
    <div class="field"><label>시작일</label><input data-g3="start" type="date" value="${st.globals.start}"></div>
    <div class="field"><label>경로 수</label><input data-g3="n_paths" type="number" step="5000" min="2000" max="50000" value="${st.globals.n_paths}"></div>
    <div class="field"><label>세금 반영</label><input data-g3="tax" type="checkbox" ${st.globals.tax ? "checked" : ""}></div>
    <div class="field"><label>의존구조</label><select data-g3="copula"><option value="gaussian" ${st.globals.copula === "gaussian" ? "selected" : ""}>가우시안</option><option value="t" ${st.globals.copula === "t" ? "selected" : ""}>t-copula</option></select></div>
    <div class="field"><label>버티컬 증거금 확인 (R-3)</label><select data-g3="broker_confirmed"><option value="null">미확인</option><option value="true" ${st.globals.broker_confirmed === true ? "selected" : ""}>확인함</option></select></div>`;
  const E = $("#p3edit");
  E.querySelectorAll("[data-w]").forEach(x => x.addEventListener("change", e => { const [c, i] = e.target.dataset.w.split(",").map(Number); st.weights[c][i] = (+e.target.value || 0) / 100; p3save(); p3edit(); }));
  E.querySelectorAll("[data-f]").forEach(x => x.addEventListener("change", e => { st.f[+e.target.dataset.f] = Math.min(1, Math.max(0, +e.target.value)); p3save(); p3edit(); }));
  E.querySelectorAll("[data-v]").forEach(x => x.addEventListener("change", e => {
    const [i, k, s] = e.target.dataset.v.split(","), t = st.tickers[+i], v = e.target.value === "" ? null : +e.target.value;
    const o = (t[k] && typeof t[k] === "object") ? t[k] : (t[k] != null ? Object.fromEntries(st.scenarios.map(sc => [sc.name, t[k]])) : {});
    o[s] = v; t[k] = Object.values(o).some(z => z != null) ? o : null; p3save(); p3edit();
  }));
  E.querySelectorAll("[data-s]").forEach(x => x.addEventListener("change", e => { const [i, k] = e.target.dataset.s.split(","); st.scenarios[+i][k] = +e.target.value; p3save(); p3edit(); }));
  E.querySelectorAll("[data-g3]").forEach(x => x.addEventListener("change", e => {
    const k = e.target.dataset.g3; let v = e.target.type === "checkbox" ? e.target.checked : e.target.value;
    if (k === "broker_confirmed") v = v === "true" ? true : null;
    else if (k === "r") v = v === "" ? null : +v;
    else if (!["start", "copula"].includes(k) && typeof v === "string") v = +v;
    st.globals[k] = v; p3save(); p3edit();
  }));
}

async function p3run() {
  if (P3.busy) return;
  P3.busy = true; const b = $("#p3run"); if (b) { b.disabled = true; b.textContent = "계산 중…"; } $("#busy").style.display = "block";
  try { P3.result = await api("/api/page3", P3.state); P3.roll = null; p3render(); }
  catch (e) { $("#p3out").innerHTML = `<div class="panel banner">계산 실패: ${esc(e.message)}</div>`; }
  finally { P3.busy = false; if (b) { b.disabled = false; b.textContent = "타임라인 계산"; } $("#busy").style.display = "none"; }
}

// ------------------------------------------------------------------ render
function p3render() {
  const R = P3.result;
  if (R.error) { $("#p3out").innerHTML = `<div class="panel banner">${esc(R.error)}</div>${p3tickers(R)}`; return; }
  if (!R.scenarios_out[P3.scen]) P3.scen = Object.keys(R.scenarios_out)[0];
  $("#p3out").innerHTML = `
    ${R.warnings.map(w => `<div class="note alert">⚠ ${esc(w)}</div>`).join("")}
    ${p3tickers(R)}
    <section class="panel"><h2>8년 실행 타임라인 — 4사이클 × 8롤</h2>
      <div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:6px">
        <div class="tabs" id="p3scn" style="margin:0">${Object.keys(R.scenarios_out).map(s => `<button data-s="${s}" class="${s === P3.scen ? "on" : ""}">${SCN_KO[s] || s} · ${pct(R.scenarios_out[s].prob, 0)}</button>`).join("")}</div>
        <div class="tabs" id="p3rep" style="margin:0">${Object.keys(R.scenarios_out[P3.scen].reps).map(r => `<button data-r="${r}" class="${r === P3.rep ? "on" : ""}">${REP_KO[r]}</button>`).join("")}</div></div>
      <div class="sub">막대 = 선택 경로의 롤별 투입(회색)·회수(파랑), 주황 선 = 그 경로의 누적 회수액(세후), 초록 띠 = ${int(R.n_band)} 경로 표본의 10–90% 구간, 점선 = 중앙값, 빨간 점선 = 목표 G, ▲ = 흡수상태.
        롤을 클릭하면 그 롤의 구조표가 열림. 아래 레인: 국고채(원금 보장) · 섹터 배분.</div>
      <div class="chart" id="p3tl"></div><div id="p3roll"></div></section>
    <div class="grid2">
      <section class="panel" id="p3dist"></section>
      <section class="panel" id="p3bond"></section>
    </div>
    <section class="panel"><h2>운용 — 롤별 체크리스트 (R-1…R-6)</h2><div class="sub">롤 1은 지금 체인 기준 실제 행사가·계약 수, 이후 롤은 진입 시점 현물 대비 비율(같은 머니니스로 재진입). 배당락은 최근 지급 주기로 추정.</div>
      <div class="scroll" id="p3ops"></div></section>`;
  $("#p3scn").querySelectorAll("button").forEach(b => b.onclick = () => { P3.scen = b.dataset.s; if (!R.scenarios_out[P3.scen].reps[P3.rep]) P3.rep = "median"; p3update(true); });
  $("#p3rep").querySelectorAll("button").forEach(b => b.onclick = () => { P3.rep = b.dataset.r; p3update(true); });
  p3update(false);
}
function p3update(animate) {
  const R = P3.result;
  $("#p3scn").querySelectorAll("button").forEach(b => b.classList.toggle("on", b.dataset.s === P3.scen));
  $("#p3rep").innerHTML = Object.keys(R.scenarios_out[P3.scen].reps).map(r => `<button data-r="${r}" class="${r === P3.rep ? "on" : ""}">${REP_KO[r]}</button>`).join("");
  $("#p3rep").querySelectorAll("button").forEach(b => b.onclick = () => { P3.rep = b.dataset.r; p3update(true); });
  p3timeline(R, animate);
  p3dist(R); p3bond(R); p3ops(R);
  if (P3.roll !== null) p3rollDetail(R, P3.roll);
}

function p3tickers(R) {
  return `<section class="panel"><h2>종목 <span class="sub">${R.source === "page2" ? "Page 2" : R.source === "page1" ? "Page 1" : "직접 입력"}에서</span></h2><div class="scroll"><table class="t">
    <tr><th>종목</th><th>섹터</th><th>현물<span class="mkt-tag">mkt</span></th><th>상장 만기</th><th>K1/K2 (지금)</th><th>K1/K2 (현물 대비)</th><th>왕복비용</th><th>Edge (가중)</th><th>상태</th></tr>
    ${R.tickers.map((c, i) => c.K1 ? `<tr><td><span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:${color(i)};margin-right:6px"></span>${esc(c.ticker)}</td><td class="sub">${esc(c.sector)}</td>
      <td>$${num(c.spot)}</td><td>${c.expiry}</td><td>${num(c.K1, 1)}/${num(c.K2, 1)}</td><td>${mdl("+" + pct(c.K1 / c.spot - 1))} / ${mdl("+" + pct(c.K2 / c.spot - 1))}</td>
      <td>${mdl(pct(c.rt_cost, 0))}</td><td>${mdl((c.edge_mix >= 0 ? "+" : "") + pct(c.edge_mix))}</td>
      <td>${c.eligible ? statusBadge("pass", "포함") : `<span data-tip="${esc(c.reason)}">${statusBadge("fail", "제외")}</span>`}</td></tr>`
      : `<tr><td>${esc(c.ticker)}</td><td colspan="8">${statusBadge("fail", "제외")} ${esc(c.reason)}</td></tr>`).join("")}</table></div></section>`;
}

const SECTOR_COLORS = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)", "var(--s5)"];
function p3timeline(R, animate) {
  const host = $("#p3tl");
  const S_ = R.scenarios_out[P3.scen], rep = S_.reps[P3.rep] || S_.reps.median, G = R.globals.G;
  const dur = animate ? 600 : 0;
  let c = host._c;
  const laneH = { main: 300, bond: 64, sector: 70 };
  if (!c) {
    host.innerHTML = "";
    c = svgFor(host, laneH.main + laneH.bond + laneH.sector + 70, { t: 18, r: 16, b: 34, l: 64 });
    host._c = c;
    c.x = d3.scaleBand(d3.range(32), [0, c.iw]).padding(0.14);
    c.y = d3.scaleLinear([0, 1], [laneH.main, 0]);
    c.bands = c.g.append("g"); c.grid = c.g.append("g").attr("class", "gridl"); c.yax = c.g.append("g").attr("class", "axis");
    c.area = c.g.append("path").attr("fill", "var(--s3)").attr("fill-opacity", 0.16);
    c.med = c.g.append("path").attr("fill", "none").attr("stroke", "var(--s3)").attr("stroke-width", 1.5).attr("stroke-dasharray", "4,3");
    c.dep = c.g.append("g"); c.pro = c.g.append("g");
    c.cum = c.g.append("path").attr("fill", "none").attr("stroke", "var(--s2)").attr("stroke-width", 2);
    c.goal = c.g.append("line").attr("stroke", "var(--critical)").attr("stroke-dasharray", "5,4");
    c.goalT = c.g.append("text").attr("class", "lbl").attr("text-anchor", "end");
    c.abs = c.g.append("g"); c.hit = c.g.append("g");
    c.bond = c.g.append("g").attr("transform", `translate(0,${laneH.main + 26})`);
    c.sec = c.g.append("g").attr("transform", `translate(0,${laneH.main + 26 + laneH.bond + 14})`);
    c.g.append("g").attr("class", "axis").attr("transform", `translate(0,${laneH.main})`)
      .call(d3.axisBottom(c.x).tickValues(d3.range(0, 32, 4)).tickFormat(k => R.calendar[k].close.slice(0, 7)));
  }
  const x = c.x, xb = x.bandwidth();
  const ymax = Math.max(G * 1.3, d3.max(S_.band.p90) * 1.05, d3.max(rep.cumulative) * 1.05);
  c.y.domain([0, ymax]).nice();
  const y = c.y, clip = v => y(Math.min(Math.max(v, 0), ymax));
  c.grid.transition().duration(dur).call(d3.axisLeft(y).ticks(5).tickSize(-c.iw).tickFormat(""));
  c.yax.transition().duration(dur).call(d3.axisLeft(y).ticks(5).tickFormat(won));
  c.bands.selectAll("rect").data([0, 1, 2, 3]).join("rect").attr("x", k => x(k * 8)).attr("width", k => x(k * 8 + 7) + xb - x(k * 8))
    .attr("y", 0).attr("height", laneH.main).attr("fill", k => k % 2 ? "var(--surface-2)" : "transparent");
  c.bands.selectAll("text").data([0, 1, 2, 3]).join("text").attr("class", "lbl").attr("x", k => x(k * 8) + 4).attr("y", -6)
    .text(k => (x(k * 8 + 7) + xb - x(k * 8) > 210 ? `사이클 ${k + 1} · B ${won(S_.budgets[k])} · f ${num(R.f_cycles[k], 2)}`
      : `C${k + 1} · ${won(S_.budgets[k])}`));
  const cx = k => x(k) + xb / 2;
  c.area.transition().duration(dur).attr("d", d3.area().x((v, k) => cx(k)).y0((v, k) => clip(S_.band.p10[k])).y1((v, k) => clip(S_.band.p90[k]))(S_.band.p50));
  c.med.transition().duration(dur).attr("d", d3.line().x((v, k) => cx(k)).y(v => clip(v))(S_.band.p50));
  c.dep.selectAll("rect").data(rep.deployed).join("rect").attr("x", (v, k) => x(k)).attr("width", xb / 2 - 1).attr("rx", 2).attr("fill", "var(--muted)")
    .transition().duration(dur).attr("y", v => clip(v)).attr("height", v => laneH.main - clip(v));
  c.pro.selectAll("rect").data(rep.proceeds).join("rect").attr("x", (v, k) => x(k) + xb / 2).attr("width", xb / 2 - 1).attr("rx", 2).attr("fill", "var(--s1)")
    .transition().duration(dur).attr("y", v => clip(v)).attr("height", v => laneH.main - clip(v));
  c.cum.transition().duration(dur).attr("d", d3.line().x((v, k) => cx(k)).y(v => clip(v))(rep.cumulative));
  c.goal.transition().duration(dur).attr("x1", 0).attr("x2", c.iw).attr("y1", y(G)).attr("y2", y(G));
  c.goalT.transition().duration(dur).attr("x", c.iw - 4).attr("y", y(G) - 4).text(`목표 G ${won(G)}`);
  const firstAbs = rep.absorbed.map((v, k) => v && (k % 8 === 0 || !rep.absorbed[k - 1]) ? k : -1).filter(k => k >= 0);
  c.abs.selectAll("text").data(firstAbs).join("text").attr("text-anchor", "middle").style("fill", "var(--ink)").text("▲")
    .attr("x", cx).attr("y", k => clip(rep.cumulative[k]) - 8);
  c.hit.selectAll("rect").data(d3.range(32)).join("rect").attr("x", k => x(k)).attr("width", xb).attr("y", 0).attr("height", laneH.main)
    .attr("fill", "transparent").style("cursor", "pointer")
    .on("mousemove", (ev, k) => showTip(ev, p3rollTip(R, k)))
    .on("mouseleave", hideTip)
    .on("click", (ev, k) => { P3.roll = k; p3rollDetail(R, k); });
  // bond lane
  c.bond.selectAll("*").remove();
  c.bond.append("text").attr("class", "lbl").attr("x", -60).attr("y", 14).text("국고채");
  S_.bond.forEach((b, k) => {
    const x0 = x(k * 8), w = x(k * 8 + 7) + xb - x0;
    c.bond.append("rect").attr("x", x0 + 1).attr("y", 0).attr("width", w - 2).attr("height", 40).attr("rx", 4).attr("fill", "var(--surface-2)").attr("stroke", "var(--axis)")
      .on("mousemove", ev => showTip(ev, `사이클 ${k + 1} 국고채 2년물<br>매입 ${b.start} @ r ${pct(b.r, 2)}: <i>${won(b.bond_cost)}</i><br>만기 ${b.end}: <b>${won(b.matures_to)}</b> (= V − V_own)<br>할인분 = 옵션 예산 B <i>${won(b.B)}</i>`)).on("mouseleave", hideTip);
    c.bond.append("text").attr("class", "lbl").attr("x", x0 + 6).attr("y", 16).text(`매입 ${won(b.bond_cost)} @ ${pct(b.r, 2)}`);
    c.bond.append("text").attr("class", "lbl").attr("x", x0 + 6).attr("y", 32).text(`→ 만기 ${won(b.matures_to)} · B ${won(b.B)}`);
    if (k < 3) c.bond.append("text").attr("x", x0 + w).attr("y", 54).attr("text-anchor", "middle").style("fill", "var(--ink-2)").style("font-size", "10.5px").text("⇄ 전세 반환·재계약 → B 재계산");
  });
  // sector lane
  c.sec.selectAll("*").remove();
  c.sec.append("text").attr("class", "lbl").attr("x", -60).attr("y", 14).text("섹터");
  const sectors = [...new Set(R.sectors)];
  for (let k = 0; k < 32; k++) {
    const cyc = Math.floor(k / 8), w = R.weights[cyc];
    let acc = 0;
    sectors.forEach((s, si) => {
      const v = R.sectors.reduce((a, ss, i) => a + (ss === s ? w[i] : 0), 0);
      if (v <= 0) return;
      c.sec.append("rect").attr("x", x(k)).attr("width", xb).attr("y", acc * 44).attr("height", Math.max(0, v * 44 - 1)).attr("fill", SECTOR_COLORS[si % 5]).attr("rx", 1)
        .on("mousemove", ev => showTip(ev, `롤 ${k + 1}: ${esc(s)} ${pct(v, 0)}`)).on("mouseleave", hideTip);
      acc += v;
    });
  }
  c.sec.append("text").attr("class", "lbl").attr("x", 0).attr("y", 60).text(sectors.map((s, i) => `■ ${s}`).join("   "));
  c.sec.selectAll(".sl").data(sectors).join("text");
}
function p3rollTip(R, k) {
  const S_ = R.scenarios_out[P3.scen], rep = S_.reps[P3.rep] || S_.reps.median, sch = R.schedule[k], cal = R.calendar[k];
  return `<b>롤 ${k + 1}</b> · 사이클 ${sch.cycle + 1}<br>진입 ${cal.entry} · 청산 ${cal.close} · 만기 ${cal.expiry}<br>
    ${sch.legs.map(l => `${esc(l.ticker)} ${pct(l.weight, 0)}: K1 <i>$${num(rep.strikes[l.ticker].K1[k], 0)}</i> / K2 <i>$${num(rep.strikes[l.ticker].K2[k], 0)}</i>${rep.pins[l.ticker][k] ? " ⚠핀" : ""}`).join("<br>")}
    <br>투입 <i>${won(rep.deployed[k])}</i> · 손익 <i>${won(rep.pnl[k])}</i><br>누적 <i>${won(rep.cumulative[k])}</i> · 표본 중앙값 <i>${won(S_.band.p50[k])}</i>${rep.absorbed[k] ? "<br>▲ 흡수상태: 투입 0" : ""}`;
}
function p3rollDetail(R, k) {
  const S_ = R.scenarios_out[P3.scen], rep = S_.reps[P3.rep] || S_.reps.median, sch = R.schedule[k], fx = R.macro.usdkrw;
  const rows = sch.legs.map(l => {
    const st = rep.strikes[l.ticker], spot = rep.spots[l.ticker][k];
    const tk = R.tickers.find(t => t.ticker === l.ticker);
    const costRel = tk.net_cost / tk.spot, dep = rep.deployed[k] * l.weight;
    const n = dep / (costRel * spot * 100 * fx);
    return `<tr><td>${esc(l.ticker)}</td><td>${mdl(pct(l.weight, 0))}</td><td>${mdl("$" + num(spot, 0))}</td><td>${k === 0 ? "$" + num(l.K1, 1) : mdl("$" + num(st.K1[k], 0))}</td>
      <td>${k === 0 ? "$" + num(l.K2, 1) : mdl("$" + num(st.K2[k], 0))}</td><td>${mdl(int(k === 0 ? l.n_contracts : n))}</td><td>${mdl(won(dep))}</td>
      <td>${mdl("$" + num(st.S_exp[k], 0))}</td><td>${rep.pins[l.ticker][k] ? statusBadge("fail", "핀 구간") : st.S_exp[k] >= st.K2[k] ? statusBadge("pass", "최대지급") : statusBadge("n/a", "전손")}</td>
      <td>${mdl(pct(l.pin_by_scenario[P3.scen], 2))}</td><td>${l.ex_div_in_window.join(", ") || "없음"}</td></tr>`;
  }).join("");
  $("#p3roll").innerHTML = `<div class="card" style="margin-top:10px"><h2>롤 ${k + 1} 구조 — ${REP_KO[P3.rep]}, ${SCN_KO[P3.scen]} <button class="btn" id="p3rclose" style="margin-left:auto">닫기</button></h2>
    <div class="sub">진입 ${sch.entry} · 청산 기한 ${sch.close} · 만기 ${sch.expiry} · f ${num(sch.f, 2)}${k === 0 ? " · 롤 1 행사가는 지금 상장 체인(만기 " + esc(sch.legs[0].listed_expiry) + ")" : " · 이후 롤의 현물·행사가는 이 경로에서 복원한 값"}</div>
    <div class="scroll"><table class="t"><tr><th>종목</th><th>비중</th><th>진입 현물</th><th>K1</th><th>K2</th><th>계약 n</th><th>투입</th><th>만기 현물</th><th>결과</th><th>핀 확률</th><th>배당락</th></tr>${rows}</table></div></div>`;
  $("#p3rclose").onclick = () => { P3.roll = null; $("#p3roll").innerHTML = ""; };
}

function p3dist(R) {
  const S_ = R.scenarios_out[P3.scen], m = S_.metrics_net, gm = S_.metrics_gross, mix = R.mixture.metrics_net, G = R.globals.G;
  $("#p3dist").innerHTML = `<h2>최종 분포 — ${SCN_KO[P3.scen]} (${int(R.n_paths)} 경로)</h2>
    <div class="tiles">
      <div class="tile"><div class="k">P(≥ ${won(G)})</div><div class="v mdl">${pct(m.p_goal)}</div><div class="s">혼합 ${pct(mix.p_goal)} · 세전 ${pct(gm.p_goal)}</div></div>
      <div class="tile"><div class="k">사이클 파산</div><div class="v mdl">${pct(m.ruin_per_cycle)}</div><div class="s">혼합 ${pct(mix.ruin_per_cycle)}</div></div>
      <div class="tile"><div class="k">중앙값</div><div class="v mdl">${won(m.median)}</div><div class="s">평균 ${won(m.mean)}${m.mean_driven_by_tail ? " ⚠ 꼬리가 끌어올림" : ""}</div></div>
    </div><div class="chart" id="p3hist"></div>
    <div class="grid2"><table class="t"><tr><th>분위</th><th>세후</th><th>세전</th></tr>${Object.keys(m.quantiles).map(q => `<tr><td>${(q * 100).toFixed(0)}%</td><td>${mdl(won(m.quantiles[q]))}</td><td>${mdl(won(gm.quantiles[q]))}</td></tr>`).join("")}</table>
    <table class="t"><tr><th>누적 ≥</th><th>확률</th><th>월 상환 감소</th></tr>${m.ladder.map(l => `<tr><td>${won(l.level)}</td><td>${mdl(pct(l.prob))}</td><td>${mdl(won(l.monthly_reduction))}</td></tr>`).join("")}</table></div>`;
  const H = S_.hist, c = svgFor($("#p3hist"), 170);
  const x = d3.scaleLinear([H.edges[0], H.edges[H.edges.length - 1]], [0, c.iw]);
  const y = d3.scaleSqrt([0, d3.max(H.counts)], [c.ih, 0]);
  axes(c, x, y, won, d => int(d), 5, 3);
  c.g.selectAll("rect").data(H.counts).join("rect").attr("x", (v, k) => x(H.edges[k]) + 1).attr("width", (v, k) => Math.max(1, x(H.edges[k + 1]) - x(H.edges[k]) - 2))
    .attr("y", v => y(v)).attr("height", v => c.ih - y(v)).attr("fill", "var(--s3)").attr("rx", 1)
    .on("mousemove", (ev, v) => showTip(ev, `${pct(v / d3.sum(H.counts))} 경로`)).on("mouseleave", hideTip);
  vline(c, x, G, "목표 G", "5,4", "var(--critical)");
  vline(c, x, m.median, "중앙값", "2,3");
  c.g.append("text").attr("class", "lbl").attr("x", c.iw).attr("y", -2).attr("text-anchor", "end").text("y = 경로 수 (√ 스케일)");
}
function p3bond(R) {
  const S_ = R.scenarios_out[P3.scen];
  $("#p3bond").innerHTML = `<h2>원금 보장 레그 — 금리 → 예산</h2><div class="sub">사이클마다 (V − V_own)(1+r_c)^−T에 국고채 2년물을 사면 만기에 V − V_own이 되고, 자가 전세보증금 V_own과 합쳐 세입자에게 V를 돌려줌. 새 세입자의 V로 다음 국고채를 r_(c+1)에 사면서 그 사이클의 B가 다시 정해짐.</div>
    <table class="t"><tr><th>사이클</th><th>r</th><th>국고채 매입</th><th>만기 상환</th><th>옵션 예산 B</th></tr>
    ${S_.bond.map(b => `<tr><td>${b.cycle + 1} · ${b.start.slice(0, 7)}</td><td>${mdl(pct(b.r, 2))}</td><td>${mdl(won(b.bond_cost))}</td><td>${won(b.matures_to)}</td><td><b class="mdl">${won(b.B)}</b></td></tr>`).join("")}</table>
    <div class="note">시나리오를 바꾸면 사이클 2–4의 r이 바뀌고 B가 그만큼 늘거나 줄어듦 (금리↓ → B↓).</div>`;
}
function p3ops(R) {
  $("#p3ops").innerHTML = `<table class="t"><tr><th>롤</th><th>진입</th><th>청산 기한</th><th>만기</th><th>f</th><th>종목 · 비중</th><th>핀 확률 (${SCN_KO[P3.scen]})</th><th>배당락 충돌</th><th>R-1…R-6</th><th>할 일</th></tr>
    ${R.schedule.map(s => `<tr><td>${s.roll + 1}</td><td>${s.entry}</td><td><b>${s.close}</b></td><td>${s.expiry}</td><td>${num(s.f, 2)}</td>
      <td>${s.legs.map(l => `${esc(l.ticker)} ${pct(l.weight, 0)}`).join(" · ")}</td>
      <td>${s.legs.map(l => mdl(pct(l.pin_by_scenario[P3.scen], 1))).join(" · ")}</td>
      <td>${s.legs.filter(l => l.ex_div_in_window.length).map(l => `<span class="badge warn">${esc(l.ticker)} ${l.ex_div_in_window[0]}</span>`).join(" ") || "–"}</td>
      <td><div class="rules">${["R-1", "R-2", "R-3", "R-4", "R-5", "R-6"].map(code => {
        const sts = s.legs.map(l => l.rules.find(r => r.code === code));
        const worst = ["fail", "warn", "pass", "n/a"].find(x => sts.some(r => r && r.status === x)) || "n/a";
        return `<span data-tip="${esc(sts.map((r, i) => s.legs[i].ticker + ": " + (r?.message || "")).join(" | "))}">${statusBadge(worst, code)}</span>`;
      }).join("")}</div></td>
      <td><details><summary>${s.legs.reduce((a, l) => a + l.actions.length, 0)}개</summary>${s.legs.map(l => `<div class="sub"><b>${esc(l.ticker)}</b><br>${l.actions.map(esc).join("<br>")}</div>`).join("")}</details></td></tr>`).join("")}</table>`;
}

// ------------------------------------------------------------------ exports
function download(name, text, type) {
  const blob = text instanceof Blob ? text : new Blob([text], { type });
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = name;
  document.body.appendChild(a); a.click(); setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 500);
}
const csvCell = v => { const s = v == null ? "" : String(v); return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; };
const toCSV = rows => "﻿" + rows.map(r => r.map(csvCell).join(",")).join("\n");
function exportSchedule() {
  const R = P3.result; if (!R || R.error) return alert("먼저 계산하세요.");
  const S_ = R.scenarios_out[P3.scen], rep = S_.reps[P3.rep] || S_.reps.median;
  const rows = [["roll", "cycle", "entry", "expiry", "close_by", "f", "ticker", "weight", "K1_pct_of_spot", "K2_pct_of_spot",
    `K1_on_${P3.rep}_path`, `K2_on_${P3.rep}_path`, `pin_prob_${P3.scen}`, "ex_div_in_window", "cum_p10", "cum_median", "cum_p90", "share_absorbed",
    "scenario", "budget_cycle_krw"]];
  R.schedule.forEach(s => s.legs.forEach(l => rows.push([s.roll + 1, s.cycle + 1, s.entry, s.expiry, s.close, s.f, l.ticker, l.weight.toFixed(4),
    (l.k1 - 1).toFixed(4), (l.k2 - 1).toFixed(4), s.roll === 0 ? l.K1 : rep.strikes[l.ticker].K1[s.roll].toFixed(2), s.roll === 0 ? l.K2 : rep.strikes[l.ticker].K2[s.roll].toFixed(2),
    l.pin_by_scenario[P3.scen].toFixed(4), l.ex_div_in_window.join(" "), Math.round(S_.band.p10[s.roll]), Math.round(S_.band.p50[s.roll]),
    Math.round(S_.band.p90[s.roll]), S_.band.absorbed_share[s.roll].toFixed(4), P3.scen, Math.round(S_.budgets[s.cycle])])));
  download(`elb_roll_schedule_${P3.scen}.csv`, toCSV(rows), "text/csv");
}
function exportOrders() {
  const R = P3.result; if (!R || R.error) return alert("먼저 계산하세요.");
  const fx = R.macro.usdkrw, s0 = R.schedule[0];
  const rows = [["roll", "ticker", "order", "leg", "strike", "expiry", "contracts", "order_type", "limit", "est_net_premium_per_share_usd",
    "est_total_premium_krw", "close_by", "ex_div_in_window", "exercise_cash_if_pin_krw", "rules"]];
  s0.legs.forEach(l => {
    const prem = l.net_cost * l.n_contracts * 100 * fx;
    const rules = l.rules.map(r => `${r.code}:${r.status}`).join(" ");
    rows.push([1, l.ticker, "BUY", "long call", l.K1, l.listed_expiry, l.n_contracts, "COMBO vertical (both legs one order)", "net debit <= mid + allowance",
      l.net_cost.toFixed(3), Math.round(prem), s0.close, l.ex_div_in_window.join(" "), Math.round(l.exercise_cash_krw), rules]);
    rows.push([1, l.ticker, "SELL", "short call", l.K2, l.listed_expiry, l.n_contracts, "same combo", "", "", "", s0.close, "", "", ""]);
    rows.push([1, l.ticker, "CLOSE", "sell the vertical", `${l.K1}/${l.K2}`, l.listed_expiry, l.n_contracts, "COMBO", "market or limit; fill first if between strikes",
      "", "", s0.close, "", "", "R-1 / R-6"]);
  });
  R.schedule.slice(1).forEach(s => s.legs.forEach(l => rows.push([s.roll + 1, l.ticker, "TEMPLATE", "call vertical",
    `K1>=spot*${l.k1.toFixed(3)} / K2~spot*${l.k2.toFixed(3)}`, s.expiry, "sleeve*f*weight / net premium", "COMBO", "", "", "", s.close, l.ex_div_in_window.join(" "), "", ""])));
  download("elb_order_sheet.csv", toCSV(rows), "text/csv");
}
function inlineStyles(src, dst) {
  const cs = getComputedStyle(src);
  ["fill", "stroke", "stroke-width", "stroke-dasharray", "fill-opacity", "stroke-opacity", "font-size", "font-family", "font-weight", "opacity", "text-anchor"]
    .forEach(p => { const v = cs.getPropertyValue(p); if (v) dst.style.setProperty(p, v); });
  for (let i = 0; i < src.children.length; i++) inlineStyles(src.children[i], dst.children[i]);
}
function exportImage(kind) {
  const svg = $("#p3tl svg"); if (!svg) return alert("먼저 계산하세요.");
  const clone = svg.cloneNode(true);
  inlineStyles(svg, clone);
  const [, , w, h] = svg.getAttribute("viewBox").split(" ").map(Number);
  clone.setAttribute("xmlns", "http://www.w3.org/2000/svg"); clone.setAttribute("width", w); clone.setAttribute("height", h);
  const bg = document.createElementNS("http://www.w3.org/2000/svg", "rect");
  bg.setAttribute("width", w); bg.setAttribute("height", h); bg.setAttribute("fill", getComputedStyle(document.body).backgroundColor);
  clone.insertBefore(bg, clone.firstChild);
  const text = new XMLSerializer().serializeToString(clone);
  if (kind === "svg") return download(`elb_timeline_${P3.scen}.svg`, text, "image/svg+xml");
  const k = Math.max(2, 1600 / w);              // vector source: render large regardless of the pane width
  clone.setAttribute("width", w * k); clone.setAttribute("height", h * k);
  const big = new XMLSerializer().serializeToString(clone);
  const img = new Image();
  img.onload = () => {
    const cv = document.createElement("canvas"); cv.width = Math.round(w * k); cv.height = Math.round(h * k);
    const ctx = cv.getContext("2d"); ctx.drawImage(img, 0, 0, cv.width, cv.height);
    cv.toBlob(b => download(`elb_timeline_${P3.scen}.png`, b), "image/png");
  };
  img.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(big);
}
