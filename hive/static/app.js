// HIVEMIND dashboard: live websocket client + rendering (no build step, no dependencies)

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

const state = {
  snap: null,
  events: [],
  feedFilter: "key",
  boardMode: "all",
  sort: { key: "rank", dir: 1 },
  seenLeaders: null,
  drawerUid: null,
  openAgents: new Set(),
  live: false,
  hoverX: null,
};

// ---------------------------------------------------------------- formatting
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (x, d = 0) => (x == null || Number.isNaN(x) ? "—" : `${(x * 100).toFixed(d)}%`);
const num = (x, d = 2) => (x == null || !Number.isFinite(x) ? "—" : x.toFixed(d));
const money = (x, d = 0) => {
  if (x == null || !Number.isFinite(x)) return "—";
  const s = Math.abs(x).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
  return `${x < 0 ? "−" : ""}$${s}`;
};
const compact = (n) => {
  if (n == null) return "—";
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e4) return `${(n / 1e3).toFixed(1)}k`;
  return n.toLocaleString();
};
const clock = (t) => new Date(t * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
const hm = (t) => new Date(t * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
const dur = (s) => {
  if (!s && s !== 0) return "—";
  if (s < 90) return `${Math.round(s)}s`;
  if (s < 5400) return `${Math.round(s / 60)}m`;
  if (s < 172800) return `${(s / 3600).toFixed(1)}h`;
  return `${(s / 86400).toFixed(1)}d`;
};
const tod = (m) => `${String(Math.floor(m / 60)).padStart(2, "0")}:${String(m % 60).padStart(2, "0")}`;
const signCls = (x) => (x > 0 ? "pos" : x < 0 ? "neg" : "");

function toast(msg, ms = 2600) {
  const el = $("#toast");
  el.textContent = msg;
  el.classList.add("show");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => el.classList.remove("show"), ms);
}

async function api(path, body) {
  const res = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    headers: body === undefined ? {} : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || res.statusText);
  return data;
}

// ---------------------------------------------------------------- websocket
let backoff = 800;
function connect() {
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.onopen = () => { backoff = 800; setLive(true); };
  ws.onclose = () => { setLive(false); setTimeout(connect, backoff); backoff = Math.min(backoff * 1.7, 8000); };
  ws.onerror = () => ws.close();
  ws.onmessage = (m) => {
    const msg = JSON.parse(m.data);
    if (msg.type === "hello") {
      state.events = msg.events || [];
      renderFeed();
      onSnapshot(msg.data);
    } else if (msg.type === "events") {
      onEvents(msg.events);
    } else if (msg.type === "snapshot") {
      onSnapshot(msg.data);
    }
  };
}
function setLive(on) {
  state.live = on;
  $("#conn").classList.toggle("live", on);
}

function onEvents(evs) {
  state.events.push(...evs);
  if (state.events.length > 600) state.events.splice(0, state.events.length - 600);
  renderFeed();
  for (const ev of evs) {
    if (ev.kind === "champion") toast(`🏆 ${ev.msg}`, 4200);
    else if (ev.kind === "death") toast(`☠ ${ev.msg}`, 3200);
  }
}

function onSnapshot(d) {
  if (!d) return;
  if (d.booting) { showBoot(d.boot); return; }
  hideBoot();
  state.snap = d;
  renderHeader(d);
  renderKpis(d);
  renderChampion(d);
  renderAgents(d);
  renderBoard(d);
  renderCombos(d);
  renderPaper(d);
  renderMemory(d);
  renderGrave(d);
  drawEvo();
}

// ---------------------------------------------------------------- boot
function showBoot(boot) {
  const el = $("#boot");
  el.classList.remove("hidden");
  $("#boot-stage").innerHTML = boot.error ? `<span class="err">Boot failed: ${esc(boot.error)}</span>` : esc(boot.stage);
  $("#boot-log").textContent = (boot.log || []).map(([t, m]) => `${clock(t)}  ${m}`).join("\n");
}
function hideBoot() { $("#boot").classList.add("hidden"); }

// ---------------------------------------------------------------- header + KPIs
function renderHeader(d) {
  const st = $("#status-pill");
  st.textContent = d.status === "running" ? "● running" : "❚❚ paused";
  st.className = `pill ${d.status}`;
  $("#pause-btn").textContent = d.status === "running" ? "Pause" : "Resume";
  const obj = $("#objective-chip");
  obj.textContent = `⬢ ${d.mode.rules_name}`;
  obj.title = d.mode.rules_text;
  const data = $("#data-chip");
  const src = d.mode.data;
  data.textContent = src === "synthetic" ? "SYNTHETIC DEMO DATA" : `${src.toUpperCase()} DATA`;
  data.className = `chip ${src === "synthetic" ? "demo" : ""}`;
  data.title = (d.datasets || []).map((x) => `${x.instrument} ${x.timeframe}m via ${x.proxy}: ${x.days} days ${x.start}→${x.end}`).join("\n");
  const model = $("#model-chip");
  model.textContent = d.mode.llm === "claude" ? `Claude · ${d.mode.overmind_model}` : "OFFLINE · procedural agents";
  model.className = `chip ${d.mode.llm === "claude" ? "" : "demo"}`;
  model.title = d.mode.llm === "claude" ? `Overmind: ${d.mode.overmind_model}\nDrones: ${d.mode.drone_model}` : "Set ANTHROPIC_API_KEY in .env to give the hive Claude agents";
}

function renderKpis(d) {
  const s = d.stats;
  $("#k-tested").textContent = compact(s.tested);
  $("#k-rate").textContent = `${compact(Math.round(s.rate_per_min))}/min · ${s.workers} workers · ${dur(s.elapsed)}`;
  const fastMode = !!s.target_days;
  $("#k-val-label").textContent = fastMode ? `Pass ≤${s.fast_days}d` : "Champion pass rate";
  $("#k-test-label").textContent = fastMode ? `Holdout pass ≤${s.fast_days}d` : "Holdout pass rate";
  $("#k-val").textContent = pct(fastMode ? s.best_val_fast : s.best_val_pass);
  $("#k-fit").textContent = `score ${num(s.best_fitness, 3)} · ${s.best_robust ? "✓ robust" : "robustness check pending"}${s.robust_queue ? ` · ${s.robust_queue} queued` : ""}`;
  const headline = fastMode ? s.best_test_fast : s.best_test_pass;
  $("#k-test").textContent = pct(headline);
  const fee = d.mode.rules.fee_usd || 0;
  const luckV = fastMode ? s.best_test_luck_fast : s.best_test_luck;
  const luck = luckV != null ? `luck ${pct(luckV)} · ` : "";
  const other = fastMode ? `30d ${pct(s.best_test_pass)} · ` : "";
  if (headline > 0.001) {
    const attempts = 1 / headline;
    $("#k-attempts").textContent = `${luck}${other}≈ ${attempts.toFixed(1)} tries${fee ? ` (~${money(attempts * fee)})` : ""}`;
  } else {
    $("#k-attempts").textContent = `${luck}${other}no fast holdout passes yet`;
  }
  $("#k-agents").textContent = `${s.drones_alive}/${s.max_drones}`;
  $("#k-deaths").textContent = `drones alive · ${s.deaths} deleted`;
  $("#k-om").textContent = `Gen ${s.overmind_gen}`;
  $("#k-stag").textContent = `cycle ${s.cycle} · stagnation ${s.stagnation}/${s.patience} · ${s.evolver_focus}`;
  $("#k-spend").textContent = money(s.spend, 2);
  $("#k-calls").textContent = `${s.llm_calls} calls${s.budget ? ` · budget ${money(s.budget, 0)}` : ""}`;
}

function renderChampion(d) {
  const el = $("#champion-strip");
  const top = d.leaders[0];
  if (!top) { el.innerHTML = `<span class="muted">Waiting for the first strategies…</span>`; return; }
  el.dataset.uid = top.uid;
  el.innerHTML = `
    <span class="crown">♛</span>
    <span class="cname">${esc(top.name)}</span>
    <span class="cmeta">${esc(top.instrument)} ${top.timeframe}m · ${esc(top.family)} · by ${esc(top.author)}</span>
    <span class="cmeta">score ${num(top.score, 3)} ${top.robust ? "✓" : "(checking)"} · val pass <b class="warn">${pct(top.val_pass)}</b> · holdout <b class="pos">${pct(top.test_pass)}</b> vs luck ${pct(top.test_luck)} · ${top.trades} trades · median ${top.median_days ? Math.round(top.median_days) : "—"} days to pass</span>`;
}

// ---------------------------------------------------------------- agents
function renderAgents(d) {
  const box = $("#agents");
  const s = d.stats;
  $("#hive-sub").textContent = `${s.drones_alive} drones · ${s.max_strikes} strikes = deletion`;
  if (!d.agents.length) { box.innerHTML = `<div class="empty">No agents alive.</div>`; return; }
  box.innerHTML = d.agents.map((a) => agentCard(a, d)).join("");
}

function agentCard(a, d) {
  const isOM = a.kind === "overmind";
  const st = a.status;
  const label = st === "thinking" ? "thinking…" : st === "evaluating" ? "backtesting…" : "idle";
  const maxStrikes = isOM ? d.stats.patience : d.stats.max_strikes;
  const lives = isOM
    ? `<div class="stagbar" title="stagnant cycles before this Overmind is deleted">${Array.from({ length: maxStrikes }, (_, i) => `<i class="${i < a.strikes ? "on" : ""}"></i>`).join("")}</div>`
    : "";
  const pips = !isOM
    ? `<span class="lives" title="lives left before permanent deletion">${Array.from({ length: maxStrikes }, (_, i) => `<i class="${i >= maxStrikes - a.strikes ? "lost" : ""}"></i>`).join("")}</span>`
    : "";
  return `
  <div class="agent ${isOM ? "overmind" : ""} ${state.openAgents.has(a.id) ? "open" : ""}" data-id="${a.id}">
    <div class="agent-top">
      ${isOM ? `<span class="badge">GEN ${a.generation}</span>` : ""}
      <span class="agent-name">${esc(a.name)}</span>
      ${!isOM && a.instrument_focus ? `<span class="focus">${esc(a.instrument_focus)}</span>` : ""}
      <span class="state ${st}"><span class="dot"></span>${label}</span>
    </div>
    <div class="agent-spec">${esc(isOM ? "Hive operator · spawns, directs and deletes drones" : a.specialty || a.mission)}</div>
    <div class="agent-stats">
      ${isOM ? `<span>cycles <b>${a.rounds}</b></span>` : `<span>rounds <b>${a.rounds}</b></span><span>valid <b>${a.valid}/${a.submitted}</b></span>`}
      <span>best <b>${num(a.best_fitness, 3)}</b></span>
      <span>age <b>${dur(Date.now() / 1000 - a.born)}</b></span>
      ${a.cost_usd ? `<span>cost <b>${money(a.cost_usd, 2)}</b></span>` : ""}
      ${pips}
    </div>
    ${lives}
    ${a.last_thought ? `<div class="thought" title="click to expand">${esc(a.last_thought)}</div>` : ""}
    <button class="kill" title="delete this agent permanently">✕</button>
  </div>`;
}

// ---------------------------------------------------------------- leaderboard
function renderBoard(d) {
  let rows = d.leaders.map((r, i) => ({ ...r, rank: i + 1 }));
  if (state.boardMode === "niche") {
    const seen = new Set();
    rows = rows.filter((r) => {
      const k = `${r.family}/${r.instrument}`;
      if (seen.has(k)) return false;
      seen.add(k);
      return true;
    });
  }
  const { key, dir } = state.sort;
  rows.sort((a, b) => {
    const va = a[key] ?? -Infinity;
    const vb = b[key] ?? -Infinity;
    if (typeof va === "string") return dir * va.localeCompare(vb);
    return dir * (va - vb);
  });
  $$(".board th").forEach((th) => th.classList.toggle("sorted", th.dataset.sort === key));
  const prev = state.seenLeaders;
  const now = new Set(rows.map((r) => r.uid));
  $("#board").innerHTML = rows.length ? rows.map((r) => `
    <tr data-uid="${r.uid}" class="${r.rank === 1 ? "top1" : ""} ${prev && !prev.has(r.uid) ? "flash" : ""}">
      <td>${r.rank}</td>
      <td class="left"><span class="sname" title="${esc(r.name)}">${esc(r.name)}</span><span class="ssub">${esc(r.author)} · ${esc(r.origin)} · ${esc(r.sides)}</span></td>
      <td>${esc(r.instrument)} ${r.timeframe}m</td>
      <td><span class="fam">${esc(r.family.replace("_", " "))}</span></td>
      <td><span class="fitbar" title="${r.robust ? `robustness-checked; neighbours' median fitness ${num(r.robust_median, 3)}` : "robustness check pending"}">${r.robust ? '<b class="pos">✓</b>' : '<span class="muted">·</span>'} ${num(r.score, 3)}<span class="bar"><i style="width:${Math.max(2, r.score * 100)}%"></i></span></span></td>
      <td class="warn">${pct(r.val_fast)}</td>
      <td class="${r.test_fast >= 0.4 ? "pos" : r.test_fast < 0.1 ? "neg" : ""}">${pct(r.test_fast)}</td>
      <td class="${r.test_luck_fast != null && r.test_fast > r.test_luck_fast + 0.1 ? "pos" : "muted"}">${pct(r.test_luck_fast)}</td>
      <td>${pct(r.test_pass)}</td>
      <td>${r.median_days ? Math.round(r.median_days) : "—"}</td>
      <td>${r.trades}</td>
      <td>${pct(r.win_rate)}</td>
      <td>${num(r.pf, 2)}</td>
      <td class="${signCls(r.net)}">${money(r.net)}</td>
    </tr>`).join("") : `<tr><td colspan="14" class="left muted">No strategies yet.</td></tr>`;
  if (d.stats.fast_days) $$(".fast-label").forEach((th) => (th.textContent = th.textContent.replace(/≤\d+d/, `≤${d.stats.fast_days}d`)));
  state.seenLeaders = now;
}

// ---------------------------------------------------------------- feed
const FEED_FILTERS = {
  key: new Set(["champion", "spawn", "death", "overmind", "directive", "operator", "paper", "memory", "system", "error"]),
  agents: new Set(["thought", "round", "spawn", "death", "overmind", "directive", "memory"]),
  errors: new Set(["error"]),
};
function renderFeed() {
  const f = FEED_FILTERS[state.feedFilter];
  const evs = state.events.filter((e) => !f || f.has(e.kind)).slice(-220).reverse();
  $("#feed").innerHTML = evs.length
    ? evs.map((e) => `<div class="ev ${e.kind}"><span class="t">${clock(e.t)}</span>${esc(e.msg)}</div>`).join("")
    : `<div class="empty">Nothing here yet.</div>`;
}

// ---------------------------------------------------------------- paper trading
function renderCombos(d) {
  const rows = d.combos || [];
  $("#combos").innerHTML = rows.length ? rows.map((c, i) => `
    <tr data-combo="${esc(c.key)}">
      <td>${i + 1}</td>
      <td class="left"><div class="members">${c.members.map((m) => `<span><span class="mk">${esc(m.instrument)} ${m.timeframe}m</span>${esc(m.name)}</span>`).join("")}</div></td>
      <td>×${c.scale}</td>
      <td>${num(c.fitness, 3)}</td>
      <td class="warn">${pct(c.val_fast)}</td>
      <td class="${c.test_fast >= 0.4 ? "pos" : c.test_fast < 0.1 ? "neg" : ""}">${pct(c.test_fast)}</td>
      <td class="${c.test_luck_fast != null && c.test_fast > c.test_luck_fast + 0.1 ? "pos" : "muted"}">${pct(c.test_luck_fast)}</td>
      <td>${pct(c.test_pass)}</td>
      <td>${c.median_days ? Math.round(c.median_days) : "—"}</td>
      <td>${num(c.corr, 2)}</td>
      <td class="${signCls(c.test_net)}">${money(c.test_net)}</td>
      <td><button class="btn small" data-deploy-combo="${esc(c.key)}" ${d.paper ? "" : "disabled title='Configure Alpaca keys first'"}>Paper trade</button></td>
    </tr>`).join("") : `<tr><td colspan="12" class="left muted">Combos appear once there are good strategies on at least two different markets (checked every ~45 s).</td></tr>`;
}

function renderPaper(d) {
  const el = $("#paper");
  const badge = $("#paper-badge");
  const p = d.paper;
  if (!p) {
    badge.textContent = "not configured";
    badge.className = "pill small";
    el.innerHTML = `<div class="paper-empty">
      Connect Alpaca to forward-test the champion on a paper account while the hive keeps evolving.<br>
      Add <code>ALPACA_API_KEY</code> and <code>ALPACA_SECRET_KEY</code> (paper keys) to <code>.env</code> and restart.
      Alpaca has no futures, so strategies trade the ETF proxy (QQQ ≈ NQ, SPY ≈ ES, GLD ≈ gold) sized to the same dollars per point,
      and the dashboard runs a virtual ${esc(d.mode.rules_name)} on that P&L.</div>`;
    return;
  }
  if (p.error) { badge.textContent = "error"; badge.className = "pill small bad"; }
  else if (p.deployed) { badge.textContent = p.paper ? "paper · live" : "LIVE MONEY"; badge.className = `pill small ${p.paper ? "good" : "bad"}`; }
  else { badge.textContent = "connected"; badge.className = "pill small"; }
  const a = p.account || {};
  const ch = p.challenge || {};
  const rules = d.mode.rules;
  let html = `<div class="paper-grid">
    <div class="mini"><div class="l">Equity</div><div class="v">${money(a.equity)}</div></div>
    <div class="mini"><div class="l">Buying power</div><div class="v">${money(a.buying_power)}</div></div>
    <div class="mini"><div class="l">Account</div><div class="v">${esc(a.number || "—")} ${esc(a.status || "")}</div></div>
    <div class="mini"><div class="l">Size scale</div><div class="v">${p.scale}×</div></div>
  </div>`;
  if (p.error) html += `<div class="ev error">${esc(p.error)}</div>`;
  if (p.deployed) {
    const lo = ch.floor ?? rules.start_balance - rules.max_loss;
    const hi = rules.start_balance + rules.profit_target;
    const bal = ch.balance ?? rules.start_balance;
    const span = hi - lo || 1;
    const posOf = (v) => Math.min(100, Math.max(0, ((v - lo) / span) * 100));
    const chCls = ch.status === "passed" ? "good" : ch.status === "failed" ? "bad" : "warn";
    html += `
      <div class="row"><span class="muted">Running</span> <b>${esc(p.deployed.name)}</b> <span class="muted">on ${esc(p.proxy)}</span>
        <span class="pill small ${chCls}">challenge ${esc(ch.status || "—")}</span></div>
      <div class="progress">
        <div class="track"></div>
        <div class="mark" style="left:calc(${posOf(bal)}% - 1px)"></div>
        <span class="lab" style="left:${posOf(rules.start_balance)}%">start</span>
      </div>
      <div class="progress-legend"><span>floor ${money(lo)}</span><span>balance ${money(bal)}</span><span>target ${money(hi)}</span></div>
      <div class="paper-grid" style="margin-top:12px">
        <div class="mini"><div class="l">Challenge P&L</div><div class="v ${signCls(ch.pnl)}">${money(ch.pnl)}</div></div>
        <div class="mini"><div class="l">Today</div><div class="v ${signCls(ch.today_pnl)}">${money(ch.today_pnl)}</div></div>
        <div class="mini"><div class="l">Days traded</div><div class="v">${ch.days_traded ?? 0}</div></div>
        <div class="mini"><div class="l">Position</div><div class="v">${p.position ? `${esc(p.position.side)} ${p.position.qty}` : "flat"}</div></div>
      </div>
      <div class="muted small">${esc(p.message || "")}${p.combo_scale && p.combo_scale !== 1 ? ` · combo size ×${p.combo_scale}` : ""}</div>
      <div class="legs">${(p.legs || []).map((l) => `<div class="leg"><span><b>${esc(l.proxy)}</b> ${esc(l.name)}</span><span class="lm">${l.position ? `${esc(l.position.side)} ${l.position.qty} · ${money(l.position.unrealized)}` : "flat"} · ${esc(l.message)}</span></div>`).join("")}</div>
      <div class="trades-mini">${(p.trades || []).slice().reverse().map((t) => `<div><span>${hm(t.t)} ${esc(t.symbol || "")} ${esc(t.side)} ${t.shares} sh (~${t.contracts_equiv} ct)</span><span class="${signCls(t.pnl)}">${t.pnl == null ? "open" : money(t.pnl)}</span></div>`).join("")}</div>
      <div class="row" style="margin-top:10px"><button class="btn small danger" id="paper-stop">Stop & flatten</button></div>`;
  } else {
    const top = d.leaders[0];
    const combo = (d.combos || [])[0];
    html += `<div class="row"><span class="muted">Nothing deployed.</span>
      ${combo ? `<button class="btn small primary" data-deploy-combo="${esc(combo.key)}">Deploy best combo (${combo.members.length} strategies)</button>` : ""}
      ${top ? `<button class="btn small ${combo ? "" : "primary"}" id="paper-deploy-champ" data-uid="${top.uid}">Deploy champion: ${esc(top.name)}</button>` : ""}</div>`;
  }
  el.innerHTML = html;
}

// ---------------------------------------------------------------- memory / graveyard
function renderMemory(d) {
  const dir = $("#directive");
  const orders = (d.operator_notes || []).slice(-2);
  dir.innerHTML = `<b>Overmind directive</b>${esc(d.directive || "—")}${orders.length ? `<div style="margin-top:6px;color:var(--amber2)"><b style="color:var(--amber)">Your orders</b>${orders.map(esc).join("<br>")}</div>` : ""}`;
  const mem = (d.memory || []).slice().reverse();
  $("#memory").innerHTML = mem.length ? mem.map((m) => `<li>${esc(m)}</li>`).join("") : `<li class="muted">The hive has not learned anything yet.</li>`;
}

function renderGrave(d) {
  const g = d.graveyard || [];
  $("#grave-sub").textContent = `${d.stats.deaths} deleted`;
  $("#graveyard").innerHTML = g.length ? g.map((a) => `
    <li><div class="gname">${esc(a.name)}</div>
      <div class="gwhy">${esc(a.death_reason)}</div>
      <div class="gmeta">${a.kind} · lived ${dur((a.died || 0) - a.born)} · ${a.rounds} rounds · best ${num(a.best_fitness, 3)}</div></li>`).join("")
    : `<li class="muted" style="border:none;background:none;padding:0">No agent has been deleted yet.</li>`;
}

// ---------------------------------------------------------------- canvas helpers
function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

function drawEvo() {
  const d = state.snap;
  if (!d) return;
  const canvas = $("#evo-chart");
  const { ctx, w, h } = setupCanvas(canvas);
  const hist = d.history || [];
  const sc = d.scatter || [];
  const pad = { l: 44, r: 14, t: 12, b: 26 };
  const now = d.t;
  const firsts = [hist[0]?.[0], sc[0]?.[0]].filter((x) => x);
  let t0 = firsts.length ? Math.min(...firsts) : now - 60;
  if (now - t0 < 60) t0 = now - 60;
  const X = (t) => pad.l + ((t - t0) / (now - t0)) * (w - pad.l - pad.r);
  const Y = (v) => pad.t + (1 - Math.max(0, Math.min(1, v))) * (h - pad.t - pad.b);
  ctx.font = `11px ${css("--mono") || "monospace"}`;
  ctx.lineWidth = 1;
  // grid
  for (const v of [0, 0.25, 0.5, 0.75, 1]) {
    ctx.strokeStyle = v === 0 ? "#2a3441" : "rgba(42,52,65,0.55)";
    ctx.beginPath(); ctx.moveTo(pad.l, Y(v)); ctx.lineTo(w - pad.r, Y(v)); ctx.stroke();
    ctx.fillStyle = "#5c6677"; ctx.textAlign = "right"; ctx.textBaseline = "middle";
    ctx.fillText(`${v * 100}%`, pad.l - 8, Y(v));
  }
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  const ticks = Math.max(2, Math.min(6, Math.floor((w - pad.l) / 110)));
  const fmtT = now - t0 < 900 ? clock : hm;
  for (let i = 0; i <= ticks; i++) {
    const t = t0 + ((now - t0) * i) / ticks;
    ctx.textAlign = i === 0 ? "left" : i === ticks ? "right" : "center";
    ctx.fillText(fmtT(t), X(t), h - pad.b + 8);
  }
  // scatter of tested strategies
  ctx.fillStyle = "rgba(245,181,68,0.30)";
  for (const [t, f] of sc) {
    if (t < t0) continue;
    ctx.beginPath(); ctx.arc(X(t), Y(f), 1.6, 0, Math.PI * 2); ctx.fill();
  }
  if (!hist.length) {
    ctx.fillStyle = "#5c6677"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText("waiting for the first results…", w / 2, h / 2);
    return;
  }
  const step = (idx, color, width, dash = []) => {
    ctx.save();
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash);
    ctx.beginPath();
    hist.forEach((p, i) => {
      const x = X(p[0]); const y = Y(p[idx]);
      if (i === 0) ctx.moveTo(x, y);
      else { ctx.lineTo(x, Y(hist[i - 1][idx])); ctx.lineTo(x, y); }
    });
    ctx.lineTo(X(now), Y(hist[hist.length - 1][idx]));
    ctx.stroke();
    ctx.restore();
  };
  // glow under the best line
  const grad = ctx.createLinearGradient(0, pad.t, 0, h - pad.b);
  grad.addColorStop(0, "rgba(245,181,68,0.18)"); grad.addColorStop(1, "rgba(245,181,68,0)");
  ctx.fillStyle = grad;
  ctx.beginPath(); ctx.moveTo(X(hist[0][0]), Y(0));
  hist.forEach((p, i) => { if (i) ctx.lineTo(X(p[0]), Y(hist[i - 1][1])); ctx.lineTo(X(p[0]), Y(p[1])); });
  ctx.lineTo(X(now), Y(hist[hist.length - 1][1])); ctx.lineTo(X(now), Y(0)); ctx.closePath(); ctx.fill();
  step(3, "#3ddc97", 1.5, [5, 4]);
  step(2, "#4cc9f0", 1.5);
  step(1, "#f5b544", 2.2);
  const last = hist[hist.length - 1];
  ctx.fillStyle = "#f5b544";
  ctx.beginPath(); ctx.arc(X(now), Y(last[1]), 4, 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = "rgba(245,181,68,0.35)"; ctx.lineWidth = 6;
  ctx.beginPath(); ctx.arc(X(now), Y(last[1]), 7, 0, Math.PI * 2); ctx.stroke();
  // hover crosshair
  const tip = $("#evo-tip");
  if (state.hoverX != null && state.hoverX > pad.l) {
    const t = t0 + ((state.hoverX - pad.l) / (w - pad.l - pad.r)) * (now - t0);
    let p = hist[0];
    for (const q of hist) { if (q[0] <= t) p = q; else break; }
    ctx.strokeStyle = "rgba(231,234,240,0.25)"; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(state.hoverX, pad.t); ctx.lineTo(state.hoverX, h - pad.b); ctx.stroke(); ctx.setLineDash([]);
    tip.innerHTML = `<b>${clock(t)}</b><br>best score <b style="color:#f5b544">${num(p[1], 3)}</b><br>validation pass <b style="color:#4cc9f0">${pct(p[2])}</b><br>holdout pass <b style="color:#3ddc97">${pct(p[3])}</b><br>tested ${compact(p[4])}`;
    tip.style.opacity = 1;
    tip.style.left = `${Math.min(state.hoverX + 12, w - 170)}px`;
    tip.style.top = "14px";
  } else {
    tip.style.opacity = 0;
  }
}

function drawEquity(canvas, rec) {
  const { ctx, w, h } = setupCanvas(canvas);
  const eq = rec.result.equity || [];
  if (eq.length < 2) { ctx.fillStyle = "#5c6677"; ctx.font = "12px sans-serif"; ctx.fillText("no trades", 12, 20); return; }
  const pad = { l: 58, r: 12, t: 14, b: 24 };
  const ys = eq.map((p) => p[1]);
  let lo = Math.min(0, ...ys); let hi = Math.max(0, ...ys);
  if (hi - lo < 1) hi = lo + 1;
  const m = (hi - lo) * 0.08; lo -= m; hi += m;
  const t0 = eq[0][0]; const t1 = eq[eq.length - 1][0];
  const X = (t) => pad.l + ((t - t0) / (t1 - t0 || 1)) * (w - pad.l - pad.r);
  const Y = (v) => pad.t + (1 - (v - lo) / (hi - lo)) * (h - pad.t - pad.b);
  const [vs, ts] = rec.result.split_ts || [];
  const regions = [[t0, vs || t1, "TRAIN", "rgba(139,149,167,0.05)"], [vs || t1, ts || t1, "VALIDATION", "rgba(76,201,240,0.06)"], [ts || t1, t1, "HOLDOUT", "rgba(61,220,151,0.07)"]];
  ctx.font = "10px Inter, sans-serif";
  for (const [a, b, label, fill] of regions) {
    if (b <= a) continue;
    ctx.fillStyle = fill; ctx.fillRect(X(a), pad.t, X(b) - X(a), h - pad.t - pad.b);
    ctx.fillStyle = "#5c6677"; ctx.textAlign = "left"; ctx.textBaseline = "top"; ctx.fillText(label, X(a) + 6, pad.t + 4);
  }
  ctx.strokeStyle = "#2a3441"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad.l, Y(0)); ctx.lineTo(w - pad.r, Y(0)); ctx.stroke();
  ctx.fillStyle = "#5c6677"; ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.font = "10.5px JetBrains Mono, monospace";
  for (let i = 0; i <= 4; i++) { const v = lo + ((hi - lo) * i) / 4; ctx.fillText(money(v), pad.l - 6, Y(v)); }
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (let i = 0; i <= 4; i++) {
    const t = t0 + ((t1 - t0) * i) / 4;
    ctx.textAlign = i === 0 ? "left" : i === 4 ? "right" : "center";
    ctx.fillText(new Date(t * 1000).toISOString().slice(0, 10), X(t), h - pad.b + 7);
  }
  const last = ys[ys.length - 1];
  ctx.strokeStyle = last >= 0 ? "#3ddc97" : "#ff5d6c"; ctx.lineWidth = 1.8;
  ctx.beginPath(); eq.forEach((p, i) => (i ? ctx.lineTo(X(p[0]), Y(p[1])) : ctx.moveTo(X(p[0]), Y(p[1])))); ctx.stroke();
}

// ---------------------------------------------------------------- drawer
async function openDrawer(uid) {
  state.drawerUid = uid;
  $("#drawer").classList.add("open");
  $("#scrim").classList.add("open");
  $("#drawer").setAttribute("aria-hidden", "false");
  $("#d-title").textContent = "Loading…";
  $("#d-kicker").textContent = "";
  $("#d-body").innerHTML = `<div class="empty">Loading strategy…</div>`;
  try {
    const rec = await api(`/api/strategy/${uid}`);
    if (state.drawerUid !== uid) return;
    renderDrawer(rec);
  } catch (e) {
    $("#d-body").innerHTML = `<div class="ev error">${esc(e.message)}</div>`;
  }
}
async function openComboDrawer(key) {
  state.drawerUid = `combo:${key}`;
  $("#drawer").classList.add("open");
  $("#scrim").classList.add("open");
  $("#d-title").textContent = "Loading…";
  $("#d-kicker").textContent = "";
  $("#d-body").innerHTML = `<div class="empty">Loading combo…</div>`;
  try {
    const c = await api(`/api/combo/${encodeURIComponent(key)}`);
    if (state.drawerUid !== `combo:${key}`) return;
    const r = c.result;
    $("#d-kicker").textContent = `${c.members.length}-strategy combo · size ×${r.scale} · daily correlation ${num(r.corr, 2)}`;
    $("#d-title").textContent = c.members.map((m) => m.instrument).join(" + ");
    const segs = [["Train", r.train, ""], ["Validation", r.val, ""], ["Holdout", r.test, "hold"]];
    $("#d-body").innerHTML = `
      <div class="drawer-actions"><button class="btn primary small" data-deploy-combo="${esc(key)}" ${state.snap?.paper ? "" : "disabled"}>Paper trade this combo</button></div>
      <div><div class="section-title">Members (each risk_usd is multiplied by ${r.scale})</div>
        <div class="rules">${c.members.map((m) => `<div class="rule" data-open-uid="${m.uid}" style="cursor:pointer"><div class="k">${esc(m.instrument)} ${m.timeframe}m · ${esc(m.family)} · risk $${Math.round(m.risk_usd * r.scale)}</div><code>${esc(m.name)}</code></div>`).join("")}</div></div>
      <div><div class="section-title">Challenge simulation (${esc(state.snap?.mode.rules_name || "")})</div>
      <table class="seg-table">
        <tr><th>Segment</th><th>≤${r.fast_days}d</th><th>Luck</th><th>30d</th><th>Trades</th><th>Win days</th><th>Net</th><th>Max DD</th><th>Days to pass</th></tr>
        ${segs.map(([n, m, cls]) => `<tr class="${cls}"><td>${n}${cls ? " ✓ untouched" : ""}${n === "Train" && r.folds ? `<div class="muted small">folds ${r.folds.map((f) => pct(f)).join(" / ")}</div>` : ""}</td><td>${pct(m.pass_fast)}</td><td>${pct(m.luck_fast)}</td><td>${pct(m.pass_mc)}</td><td>${m.trades}</td><td>${pct(m.win_days)}</td><td class="${signCls(m.net)}">${money(m.net)}</td><td>${money(m.max_dd)}</td><td>${m.median_days ? Math.round(m.median_days) : "—"}</td></tr>`).join("")}
      </table></div>
      <div><div class="section-title">Combined equity curve</div><div class="eq-box"><canvas id="eq-chart"></canvas></div></div>`;
    state._drawerRec = c;
    drawEquity($("#eq-chart"), c);
  } catch (e) {
    $("#d-body").innerHTML = `<div class="ev error">${esc(e.message)}</div>`;
  }
}

async function deployCombo(key) {
  if (!confirm("Paper trade this combo on Alpaca? Anything currently deployed is stopped and flattened first.")) return;
  try { await api("/api/paper/deploy", { combo: key }); toast("Combo deployed to Alpaca paper"); } catch (err) { toast(err.message, 4000); }
}

function closeDrawer() {
  state.drawerUid = null;
  $("#drawer").classList.remove("open");
  $("#scrim").classList.remove("open");
  $("#drawer").setAttribute("aria-hidden", "true");
}

function renderDrawer(rec) {
  const s = rec.spec;
  const r = rec.result;
  $("#d-kicker").textContent = `${s.instrument} · ${s.timeframe}m · ${s.family.replace("_", " ")} · by ${s.author || "?"} (${s.origin || "?"})`;
  $("#d-title").textContent = s.name;
  if (!r.ok) {
    $("#d-body").innerHTML = `<div class="ev error">${esc(r.error)}</div>`;
    return;
  }
  const segs = [["Train", r.train, ""], ["Validation", r.val, ""], ["Holdout", r.test, "hold"]];
  const hasPaper = !!(state.snap && state.snap.paper);
  const outcome = (m) => `<div class="ocbar"><i class="p" style="width:${m.pass_mc * 100}%"></i><i class="f" style="width:${m.fail_maxloss * 100}%"></i><i class="d" style="width:${m.fail_dll * 100}%"></i><i class="o" style="width:${m.timeout * 100}%"></i></div>`;
  const rule = (k, v) => (v ? `<div class="rule"><div class="k">${k}</div><code>${esc(v)}</code></div>` : "");
  const mini = (l, v) => `<div class="mini"><div class="l">${l}</div><div class="v">${v}</div></div>`;
  const reasons = Object.entries(r.exit_reasons || {}).map(([k, v]) => `${k} ${v}`).join(" · ");
  $("#d-body").innerHTML = `
    <div class="drawer-actions">
      <button class="btn primary small" id="d-deploy" ${hasPaper ? "" : "disabled title='Configure Alpaca keys first'"}>Deploy to Alpaca paper</button>
      <a class="btn small" href="/api/strategy/${s.uid}/export" download>Export JSON</a>
      <button class="btn small" id="d-copy">Copy rules</button>
    </div>
    <div>
      <div class="section-title">Challenge simulation (${esc(state.snap?.mode.rules_name || "")})</div>
      <table class="seg-table">
        <tr><th>Segment</th><th>Pass (MC)</th><th>≤${r.fast_days || 10}d</th><th title="same risk, zero edge">Luck</th><th>Pass (replay)</th><th>Trades</th><th>Win</th><th>PF</th><th>Net</th><th>Max DD</th><th>Days to pass</th></tr>
        ${segs.map(([n, m, cls]) => `<tr class="${cls}"><td>${n}${cls ? " ✓ untouched" : ""}${n === "Train" && r.folds ? `<div class="muted small">folds ${r.folds.map((f) => pct(f)).join(" / ")}</div>` : ""}</td><td>${pct(m.pass_mc)}</td><td>${pct(m.pass_fast)}</td><td>${pct(m.luck)}</td><td>${pct(m.pass_hist)}</td><td>${m.trades}</td><td>${pct(m.win_rate)}</td><td>${num(Math.min(m.profit_factor, 99), 2)}</td><td class="${signCls(m.net)}">${money(m.net)}</td><td>${money(m.max_dd)}</td><td>${m.median_days ? Math.round(m.median_days) : "—"}</td></tr>`).join("")}
      </table>
    </div>
    <div>
      <div class="section-title">Equity curve (contract P&L, all segments)</div>
      <div class="eq-box"><canvas id="eq-chart"></canvas></div>
    </div>
    <div>
      <div class="section-title">Attempt outcomes</div>
      <div class="outcomes">${segs.map(([n, m]) => `<div class="oc"><span>${n}</span>${outcome(m)}</div>`).join("")}</div>
      <div class="oc-legend" style="margin-top:8px"><span><i style="background:var(--green)"></i>passed</span><span><i style="background:var(--red)"></i>max loss</span><span><i style="background:#ff9f43"></i>daily limit</span><span><i style="background:#3a4656"></i>ran out of days</span></div>
    </div>
    <div>
      <div class="section-title">Rules</div>
      <div class="rules">
        ${rule("Long entry", s.long_entry)}${rule("Short entry", s.short_entry)}${rule("Exit when", s.exit_when)}
      </div>
    </div>
    <div class="kv">
      ${mini("Stop", `${s.stop_type} ${s.stop_value}`)}
      ${mini("Target", s.target_type === "none" ? "none" : `${s.target_type} ${s.target_value}`)}
      ${mini("Breakeven", s.breakeven_at_r ? `${s.breakeven_at_r}R` : "off")}
      ${mini("Session", `${tod(s.session_start)}–${tod(s.session_end)}`)}
      ${mini("Flat at", tod(s.flat_at))}
      ${mini("Max hold", `${s.max_hold_bars} bars`)}
      ${mini("Risk / trade", money(s.risk_usd))}
      ${mini("Trades / day", s.max_trades_per_day)}
      ${mini("Daily stops", `${s.daily_loss_stop_usd ? "−" + money(s.daily_loss_stop_usd) : "off"} / ${s.daily_profit_stop_usd ? "+" + money(s.daily_profit_stop_usd) : "off"}`)}
      ${mini("Avg contracts", num(r.avg_qty, 1))}
      ${mini("Avg hold", `${num(r.avg_hold_bars, 1)} bars`)}
      ${mini("Long share", pct(r.long_share))}
    </div>
    <div class="muted small">${r.robust ? `Robustness: ${r.robust.n} nudged variants, median fitness ${num(r.robust.median, 3)} (own ${num(r.fitness, 3)}, worst ${num(r.robust.min, 3)}) → score ${num(r.robust_score, 3)}` : `Robustness check pending (unchecked strategies rank at 85% of fitness ${num(r.fitness, 3)})`}${r.conditions != null ? ` · ${r.conditions} conditions per side` : ""}</div>
    <div class="muted small">Exits: ${esc(reasons)} · uid ${s.uid}${s.parents ? ` · parents ${esc(s.parents)}` : ""}</div>
    ${s.rationale ? `<div><div class="section-title">Rationale</div><div class="rationale">${esc(s.rationale)}</div></div>` : ""}
  `;
  drawEquity($("#eq-chart"), rec);
  $("#d-copy").onclick = () => {
    const text = [`${s.name} (${s.instrument} ${s.timeframe}m)`, s.long_entry && `LONG: ${s.long_entry}`, s.short_entry && `SHORT: ${s.short_entry}`, s.exit_when && `EXIT: ${s.exit_when}`,
      `stop ${s.stop_type} ${s.stop_value} | target ${s.target_type} ${s.target_value} | session ${tod(s.session_start)}-${tod(s.session_end)} flat ${tod(s.flat_at)} | risk $${s.risk_usd}`].filter(Boolean).join("\n");
    navigator.clipboard?.writeText(text).then(() => toast("Rules copied"), () => toast("Copy failed"));
  };
  $("#d-deploy").onclick = async () => {
    try { await api("/api/paper/deploy", { uid: s.uid }); toast("Deployed to Alpaca paper"); } catch (e) { toast(e.message, 4000); }
  };
  state._drawerRec = rec;
}

// ---------------------------------------------------------------- events
function bind() {
  $("#pause-btn").onclick = async () => {
    const running = state.snap?.status === "running";
    try { await api(running ? "/api/pause" : "/api/resume", {}); } catch (e) { toast(e.message); }
  };
  const send = async () => {
    const input = $("#order-input");
    const text = input.value.trim();
    if (!text) return;
    try { await api("/api/order", { text }); input.value = ""; toast("Order delivered to the Overmind"); } catch (e) { toast(e.message); }
  };
  $("#order-send").onclick = send;
  $("#order-input").addEventListener("keydown", (e) => { if (e.key === "Enter") send(); });
  $("#board").addEventListener("click", (e) => { const tr = e.target.closest("tr[data-uid]"); if (tr) openDrawer(tr.dataset.uid); });
  $("#champion-strip").addEventListener("click", (e) => { const uid = e.currentTarget.dataset.uid; if (uid) openDrawer(uid); });
  $("#d-close").onclick = closeDrawer;
  $("#scrim").onclick = closeDrawer;
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDrawer(); });
  $$("#board-mode button").forEach((b) => (b.onclick = () => {
    state.boardMode = b.dataset.mode;
    $$("#board-mode button").forEach((x) => x.classList.toggle("on", x === b));
    state.seenLeaders = null;
    if (state.snap) renderBoard(state.snap);
  }));
  $$("#feed-filter button").forEach((b) => (b.onclick = () => {
    state.feedFilter = b.dataset.f;
    $$("#feed-filter button").forEach((x) => x.classList.toggle("on", x === b));
    renderFeed();
  }));
  $$(".board th").forEach((th) => (th.onclick = () => {
    const key = th.dataset.sort;
    if (state.sort.key === key) state.sort.dir *= -1;
    else state.sort = { key, dir: ["rank", "name", "instrument", "family", "median_days"].includes(key) ? 1 : -1 };
    state.seenLeaders = null;
    if (state.snap) renderBoard(state.snap);
  }));
  $("#agents").addEventListener("click", async (e) => {
    const card = e.target.closest(".agent");
    if (!card) return;
    if (e.target.classList.contains("kill")) {
      const name = card.querySelector(".agent-name").textContent;
      if (!confirm(`Permanently delete ${name}?`)) return;
      try { await api(`/api/agent/${card.dataset.id}/delete`, {}); } catch (err) { toast(err.message); }
      return;
    }
    if (e.target.classList.contains("thought")) {
      const id = card.dataset.id;
      if (state.openAgents.has(id)) state.openAgents.delete(id); else state.openAgents.add(id);
      card.classList.toggle("open");
    }
  });
  document.addEventListener("click", (e) => {
    const dep = e.target.closest("[data-deploy-combo]");
    if (dep) { e.stopPropagation(); deployCombo(dep.dataset.deployCombo); return; }
    const member = e.target.closest("[data-open-uid]");
    if (member) { openDrawer(member.dataset.openUid); return; }
    const row = e.target.closest("tr[data-combo]");
    if (row) openComboDrawer(row.dataset.combo);
  });
  $("#paper").addEventListener("click", async (e) => {
    if (e.target.id === "paper-stop") {
      if (!confirm("Stop the strategy and flatten the paper account?")) return;
      try { await api("/api/paper/stop", {}); toast("Paper trading stopped"); } catch (err) { toast(err.message); }
    }
    if (e.target.id === "paper-deploy-champ") {
      try { await api("/api/paper/deploy", { uid: e.target.dataset.uid }); toast("Champion deployed to Alpaca paper"); } catch (err) { toast(err.message, 4000); }
    }
  });
  const evo = $("#evo-chart");
  evo.addEventListener("mousemove", (e) => { state.hoverX = e.offsetX; drawEvo(); });
  evo.addEventListener("mouseleave", () => { state.hoverX = null; drawEvo(); });
  window.addEventListener("resize", () => { drawEvo(); if (state._drawerRec && state.drawerUid) drawEquity($("#eq-chart"), state._drawerRec); });
}

bind();
connect();
