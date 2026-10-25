// Farm app over the cattle3d REST API: weigh an animal from three PLY clouds,
// see what the cameras saw and the trunk volume in 3-D, browse the farm's
// scans and its calibration. No libraries, so it runs offline.
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const VIEWS = ["left", "right", "top"];
  const VIEW_UK = { left: "Ліва камера", right: "Права камера", top: "Верхня камера" };
  const STATUS_UK = { trusted: "Довірено", flagged: "Позначено", refused: "Відмова" };
  const REASONS_UK = [
    [/^herd not calibrated.*/, "стадо не відкалібровано: зважте 20 тварин на вагах і відкалібруйте"],
    [/^body volume outside the training range$/, "об’єм тулуба поза діапазоном, на якому навчено модель"],
    [/^too few points in (left|right|top) view$/, (m, v) => `замало точок у кадрі: ${VIEW_UK[v].toLowerCase()}`],
    [/^rescan: (.*)$/, "пересканувати: $1"],
  ];
  const reasonUk = (r) => {
    for (const [re, uk] of REASONS_UK) if (re.test(r)) return r.replace(re, uk);
    return r;
  };
  const fmt = (x, d = 0) => (x == null ? "—" : Number(x).toLocaleString("uk-UA", { maximumFractionDigits: d, minimumFractionDigits: d }));
  const mb = (bytes) => `${fmt(bytes / 1048576, 1)} МБ`;
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch { /* storage blocked */ } },
  };

  // ---- API key: from the link the server printed (#key=…), else remembered ----
  const fromHash = new URLSearchParams(location.hash.slice(1)).get("key");
  if (fromHash) {
    store.set("cattle3d-key", fromHash);
    history.replaceState(null, "", location.pathname + location.search);
  }
  let apiKey = store.get("cattle3d-key") || "";

  async function api(path, opts = {}) {
    const res = await fetch(path, { ...opts, headers: { ...(opts.headers || {}), "X-API-Key": apiKey } });
    if (res.status === 401) { await askKey(); return api(path, opts); }
    if (!res.ok) {
      let detail = res.statusText;
      try { detail = (await res.json()).detail; } catch { /* not JSON */ }
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    return res;
  }
  const json = async (path, opts) => (await api(path, opts)).json();

  function askKey() {
    return new Promise((resolve) => {
      const dlg = $("key-dlg");
      dlg.addEventListener("close", () => { apiKey = $("key-in").value.trim(); store.set("cattle3d-key", apiKey); resolve(); }, { once: true });
      dlg.showModal();
    });
  }

  // ---- tabs --------------------------------------------------------------------
  for (const b of document.querySelectorAll(".tab")) {
    b.addEventListener("click", () => {
      for (const o of document.querySelectorAll(".tab")) o.setAttribute("aria-selected", String(o === b));
      for (const p of document.querySelectorAll(".pane")) p.hidden = p.id !== `tab-${b.dataset.tab}`;
      if (b.dataset.tab === "history") loadHistory();
      if (b.dataset.tab === "farm") loadFarm();
      window.scrollTo({ top: 0 });
    });
  }

  // ---- state -------------------------------------------------------------------
  let demo = null;          // /v1/demo, when the server runs `cattle3d demo`
  let herdId = null;
  const files = { left: null, right: null, top: null }; // Blob with a .name
  let sample = null;        // chosen demo animal

  // ---- connection and farms ----------------------------------------------------
  async function init() {
    try {
      const health = await (await fetch("/health")).json();
      $("cloud").className = "cloud on";
      $("cloud-text").textContent = `Хмарний сервіс · ${location.host} · v${health.version}`;
    } catch {
      $("cloud").className = "cloud off";
      $("cloud-text").textContent = "Сервер недоступний";
      return;
    }
    try { demo = await json("/v1/demo"); } catch { demo = null; }
    const herds = await json("/v1/herds");
    const sel = $("farm");
    sel.innerHTML = "";
    for (const h of herds) sel.add(new Option(h.name, h.id));
    if (!herds.length) {
      const h = await json("/v1/herds", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name: "Моя ферма" }) });
      sel.add(new Option(h.name, h.id));
    }
    const saved = store.get("cattle3d-herd");
    if (saved && [...sel.options].some((o) => o.value === saved)) sel.value = saved;
    herdId = Number(sel.value);
    sel.addEventListener("change", () => { herdId = Number(sel.value); store.set("cattle3d-herd", sel.value); renderSamples(); });
    renderSamples();
  }

  // ---- choosing clouds ---------------------------------------------------------
  let pickingView = null;
  for (const slot of document.querySelectorAll(".slot")) {
    slot.addEventListener("click", () => { pickingView = slot.dataset.view; $("file").value = ""; $("file").click(); });
    slot.addEventListener("dragover", (e) => { e.preventDefault(); slot.classList.add("dragover"); });
    slot.addEventListener("dragleave", () => slot.classList.remove("dragover"));
    slot.addEventListener("drop", (e) => {
      e.preventDefault(); slot.classList.remove("dragover");
      if (e.dataTransfer.files[0]) setFile(slot.dataset.view, e.dataTransfer.files[0]);
    });
  }
  $("file").addEventListener("change", (e) => { if (e.target.files[0]) setFile(pickingView, e.target.files[0]); });
  $("pick3").addEventListener("click", () => { $("files3").value = ""; $("files3").click(); });
  $("files3").addEventListener("change", (e) => {
    // Files named after their view (…left…, …right…, …top…) go to that slot; others fill free slots in order.
    const list = [...e.target.files];
    for (const f of list) {
      const v = VIEWS.find((x) => new RegExp(x, "i").test(f.webkitRelativePath || f.name));
      if (v) setFile(v, f);
    }
    for (const f of list) if (!Object.values(files).includes(f)) { const v = VIEWS.find((x) => !files[x]); if (v) setFile(v, f); }
  });

  function setFile(view, blob, label) {
    files[view] = blob;
    if (!label) { sample = null; markSample(); }
    const slot = document.querySelector(`.slot[data-view="${view}"]`);
    slot.classList.toggle("ready", !!blob);
    slot.querySelector(".slot-s").textContent = blob ? `${label || blob.name} · ${mb(blob.size)}` : "Обрати .ply";
    const cam = document.querySelector(`.rig .cam[data-view="${view}"]`);
    if (cam) cam.classList.toggle("ready", !!blob);
    updateGo();
  }
  const updateGo = () => { $("go").disabled = !VIEWS.every((v) => files[v]) || !$("tag").value.trim(); };
  $("tag").addEventListener("input", updateGo);

  function renderSamples() {
    const box = $("samples-box");
    const list = demo ? demo.samples.filter((s) => s.herd_id === herdId) : [];
    box.hidden = !list.length;
    const chips = $("samples");
    chips.innerHTML = "";
    for (const s of list) {
      const b = document.createElement("button");
      b.type = "button"; b.className = "chip"; b.dataset.id = s.id; b.setAttribute("aria-pressed", "false");
      b.innerHTML = `Корова №${s.animal_id} <small>· ${s.breed}</small>`;
      b.addEventListener("click", () => chooseSample(s));
      chips.append(b);
    }
  }
  const markSample = () => { for (const c of document.querySelectorAll(".chip")) c.setAttribute("aria-pressed", String(!!sample && c.dataset.id === sample.id)); };

  async function chooseSample(s) {
    hideError();
    sample = s; markSample();
    $("tag").value = `№${s.animal_id}`;
    $("scale").value = s.scale_weight_kg;
    for (const v of VIEWS) {
      const slot = document.querySelector(`.slot[data-view="${v}"]`);
      slot.classList.remove("ready");
      slot.querySelector(".slot-s").textContent = "завантаження…";
    }
    try {
      await Promise.all(VIEWS.map(async (v) => {
        const blob = await (await api(`/v1/demo/samples/${encodeURIComponent(s.id)}/${v}.ply`)).blob();
        if (sample === s) setFile(v, blob, `${s.id}_${v}.ply`);
      }));
    } catch (e) { showError(`Не вдалося отримати хмари: ${e.message}`); }
    updateGo();
  }

  // ---- weighing ----------------------------------------------------------------
  const STEP_SHARE = [0.22, 0.2, 0.08, 0.3, 0.12, 0.08]; // rough share of server time per stage, for the animation
  let stepTimer = null;
  function runSteps() {
    const items = [...$("steps").children];
    $("steps").hidden = false;
    items.forEach((li) => { li.className = ""; li.querySelector("em")?.remove(); });
    let k = 0;
    items[0].className = "run";
    const expect = 1400; // ms, typical server time per animal
    const next = () => {
      if (k >= items.length - 1) return;
      stepTimer = setTimeout(() => { items[k].className = "done"; k += 1; items[k].className = "run"; next(); }, STEP_SHARE[k] * expect);
    };
    next();
  }
  function finishSteps(timing, ok) {
    clearTimeout(stepTimer);
    const items = [...$("steps").children];
    const total = timing ? timing.segmentation_and_features_ms : null;
    items.forEach((li, i) => {
      li.className = ok ? "done" : (li.className === "run" ? "" : li.className);
      if (ok && total != null && i < 4) {
        const em = document.createElement("em"); em.textContent = `${fmt(total * STEP_SHARE[i] / 0.8)} мс`; li.append(em);
      }
    });
    if (!ok) $("steps").hidden = true;
  }

  $("go").addEventListener("click", () => {
    hideError();
    const form = new FormData();
    form.append("animal_tag", $("tag").value.trim());
    const scale = parseFloat($("scale").value);
    if (scale > 0) form.append("scale_weight_kg", String(scale));
    form.append("preview", "true");
    if (sample && sample.rig) form.append("rig", JSON.stringify(sample.rig)); // calibration of the session it was captured in
    for (const v of VIEWS) form.append(v, files[v], `${v}.ply`);
    const total = VIEWS.reduce((a, v) => a + files[v].size, 0);

    $("go").disabled = true;
    $("up").hidden = false; $("up-bar").style.width = "0";
    $("up-text").textContent = `Надсилання в хмару: 0 з ${mb(total)}`;
    $("steps").hidden = true;

    const xhr = new XMLHttpRequest();
    xhr.open("POST", `/v1/herds/${herdId}/scans`);
    xhr.setRequestHeader("X-API-Key", apiKey);
    xhr.upload.onprogress = (e) => {
      if (!e.lengthComputable) return;
      $("up-bar").style.width = `${(100 * e.loaded) / e.total}%`;
      $("up-text").textContent = `Надсилання в хмару: ${mb(e.loaded)} з ${mb(e.total)}`;
    };
    xhr.upload.onload = () => { $("up-text").textContent = `Надіслано ${mb(total)}. Обробка на сервері…`; runSteps(); };
    xhr.onload = () => {
      updateGo();
      let body = null;
      try { body = JSON.parse(xhr.responseText); } catch { /* not JSON */ }
      if (xhr.status === 401) { finishSteps(null, false); $("up").hidden = true; askKey(); return; }
      if (xhr.status >= 300 || !body) {
        finishSteps(null, false);
        showError(body && body.detail ? (typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail)) : `Помилка сервера ${xhr.status}`);
        return;
      }
      finishSteps(body.preview && body.preview.timing_ms, body.status !== "refused");
      $("up-text").textContent = `Готово: ${mb(total)} надіслано, обробка ${fmt(body.preview?.timing_ms?.total_ms)} мс`;
      showResult(body);
    };
    xhr.onerror = () => { finishSteps(null, false); updateGo(); showError("Немає зв’язку з сервером."); };
    xhr.send(form);
  });

  function showError(text) { $("err").textContent = text; $("err").hidden = false; }
  function hideError() { $("err").hidden = true; }

  function showResult(s) {
    $("placeholder").hidden = true;
    $("result").hidden = false;
    $("r-status").textContent = STATUS_UK[s.status];
    $("r-status").className = `pill ${s.status}`;
    $("r-tag").textContent = `${s.animal_tag} · #${s.id}`;
    $("r-weight").textContent = fmt(s.weight_kg);
    $("r-interval").textContent = s.weight_kg == null ? "Маса не видана: пересканувати тварину"
      : `90 % інтервал: ${fmt(s.lower_kg)}–${fmt(s.upper_kg)} кг (±${fmt((s.upper_kg - s.lower_kg) / 2)})`;
    const cmp = $("r-scale");
    if (s.scale_weight_kg != null && s.weight_kg != null) {
      const d = s.weight_kg - s.scale_weight_kg;
      cmp.hidden = false;
      cmp.innerHTML = `На вагах: <b>${fmt(s.scale_weight_kg)} кг</b> · різниця ${d >= 0 ? "+" : "−"}${fmt(Math.abs(d))} кг (${fmt(Math.abs(100 * d / s.scale_weight_kg), 1)} %)`;
    } else cmp.hidden = true;
    const ul = $("r-reasons");
    ul.innerHTML = "";
    for (const r of s.reasons) { const li = document.createElement("li"); li.textContent = reasonUk(r); ul.append(li); }

    const p = s.preview;
    const dl = $("r-measures");
    dl.innerHTML = "";
    if (p && p.measures) {
      const m = p.measures;
      for (const [label, value] of [["Об’єм тулуба", `${fmt(m.trunk_volume_m3, 3)} м³`], ["Довжина тулуба", `${fmt(m.trunk_length_m, 2)} м`],
        ["Ширина тулуба", `${fmt(m.trunk_width_m, 2)} м`], ["Висота", `${fmt(m.height_m, 2)} м`]]) {
        const d = document.createElement("div");
        d.innerHTML = `<dt>${label}</dt><dd>${value}</dd>`;
        dl.append(d);
      }
    }
    $("r-timing").textContent = p && p.timing_ms ? `Обробка на сервері: ${fmt(p.timing_ms.total_ms)} мс · модель ${s.model_version}` : "";

    $("seen").hidden = !p;
    $("model3d").hidden = !p;
    if (p) {
      const thumbs = $("thumbs");
      thumbs.innerHTML = "";
      for (const v of VIEWS) {
        const f = document.createElement("figure");
        f.innerHTML = `<img alt="${VIEW_UK[v]}: тварину виділено зеленим" src="${p.views[v].image}"><figcaption>${VIEW_UK[v]} · ${fmt(p.views[v].animal_points)} точок</figcaption>`;
        thumbs.append(f);
      }
      viewer.load(p);
    }
    if (window.matchMedia("(max-width: 899px)").matches) $("result").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // ---- 3-D viewer (canvas, painter's algorithm) ----------------------------------
  const viewer = (() => {
    const canvas = $("view3d");
    const ctx = canvas.getContext("2d");
    let pts = null; let order = null; let slices = []; let zmax = 1; let viewOf = null; let viewNames = [];
    let yaw = -0.6, pitch = 0.42, dist = 4.2, auto = true;
    let showPoints = true, showSlices = false, byCamera = false, raf = 0, last = 0;
    const bytes = (b64) => Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    const CAMERA_RGB = { left: [90, 150, 255], right: [70, 200, 130], top: [235, 150, 60] };

    function load(p) {
      const i16 = new Int16Array(bytes(p.points_mm).buffer);
      pts = new Float32Array(i16.length);
      for (let i = 0; i < i16.length; i++) pts[i] = i16[i] / 1000;
      viewOf = bytes(p.point_view); viewNames = p.view_order;
      order = new Uint32Array(pts.length / 3).map((_, i) => i);
      zmax = 0;
      for (let i = 2; i < pts.length; i += 3) zmax = Math.max(zmax, pts[i]);
      slices = p.slices;
      $("view3d-note").textContent = p.views_merged
        ? "Три хмари зведено в одну систему координат за калібруванням станції."
        : "Для цієї ферми станцію не відкалібровано, тому показано лише верхній ракурс.";
      $("t-cams").hidden = !p.views_merged;
      auto = true; dist = 4.2; yaw = -0.6; pitch = 0.42;
      start();
    }

    function colour(i, shade) {
      let r, g, b;
      if (byCamera) [r, g, b] = CAMERA_RGB[viewNames[viewOf[i]]] || [200, 200, 200];
      else { // hide tones: dark at the belly, light along the back
        const t = Math.min(1, Math.max(0, pts[3 * i + 2] / zmax));
        r = 70 + 175 * t; g = 42 + 135 * t; b = 30 + 95 * t * t;
      }
      return `rgb(${(r * shade) | 0},${(g * shade) | 0},${(b * shade) | 0})`;
    }

    function draw(ts) {
      raf = 0;
      if (!pts) return;
      if (auto) yaw += Math.min(50, ts - (last || ts)) * 0.00035;
      last = ts;
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      const w = canvas.clientWidth, h = canvas.clientHeight;
      if (canvas.width !== Math.round(w * dpr)) { canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr); }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
      const f = Math.min(w, h) * 1.45;
      const zc = zmax * 0.5;
      // world (x along body, y lateral, z up) -> camera (u right, v up, d depth)
      const project = (x, y, z) => {
        const xr = x * cy - y * sy, yr = x * sy + y * cy, zr = z - zc;
        const depth = dist + yr * cp - zr * sp; // camera in front of the animal, raised by `pitch`
        const v = zr * cp + yr * sp;
        const s = f / depth;
        return [w / 2 + xr * s, h / 2 - v * s, depth, s];
      };
      // floor grid
      ctx.strokeStyle = "rgba(160,190,195,0.12)"; ctx.lineWidth = 1;
      for (let g = -1.4; g <= 1.41; g += 0.35) {
        let a = project(g, -1, 0), b = project(g, 1, 0);
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
        a = project(-1.4, g * 0.72, 0); b = project(1.4, g * 0.72, 0);
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
      }
      const n = pts.length / 3;
      const sx = new Float32Array(n), syy = new Float32Array(n), sd = new Float32Array(n), ss = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        const q = project(pts[3 * i], pts[3 * i + 1], pts[3 * i + 2]);
        sx[i] = q[0]; syy[i] = q[1]; sd[i] = q[2]; ss[i] = q[3];
      }
      const ring = (sl) => {
        ctx.beginPath();
        sl.outline.forEach(([y, z], j) => { const q = project(sl.x, y, z); if (j) ctx.lineTo(q[0], q[1]); else ctx.moveTo(q[0], q[1]); });
        ctx.closePath();
      };
      if (showSlices) { // the "loaf" whose slices add up to the trunk volume, behind the points
        ctx.fillStyle = "rgba(124,198,224,0.045)"; ctx.strokeStyle = "rgba(124,198,224,0.2)"; ctx.lineWidth = 1;
        for (const sl of slices) { ring(sl); ctx.fill(); ctx.stroke(); }
      }
      if (showPoints) {
        order.sort((a, b) => sd[b] - sd[a]);
        const dmin = dist - 1.2, dspan = 2.4;
        for (let k = 0; k < n; k++) {
          const i = order[k];
          const shade = 1.05 - 0.55 * Math.min(1, Math.max(0, (sd[i] - dmin) / dspan));
          ctx.fillStyle = colour(i, shade);
          const r = Math.max(1.5, ss[i] * 0.011);
          ctx.fillRect(sx[i] - r / 2, syy[i] - r / 2, r, r);
        }
      }
      if (showSlices) { // every 8th slice highlighted in front, so the cut is readable
        ctx.lineWidth = 1.6; ctx.strokeStyle = "rgba(255,190,90,0.9)";
        slices.forEach((sl, k) => { if (k % 8 === 4) { ring(sl); ctx.stroke(); } });
      }
      if (auto) start();
    }
    const start = () => { if (!raf) raf = requestAnimationFrame(draw); };

    // pointer: drag rotates, wheel or two-finger pinch zooms
    const touches = new Map();
    let pinch0 = 0, dist0 = 0;
    canvas.addEventListener("pointerdown", (e) => { canvas.setPointerCapture(e.pointerId); touches.set(e.pointerId, [e.clientX, e.clientY]); auto = false; canvas.style.cursor = "grabbing"; });
    canvas.addEventListener("pointermove", (e) => {
      if (!touches.has(e.pointerId)) return;
      const [x0, y0] = touches.get(e.pointerId);
      touches.set(e.pointerId, [e.clientX, e.clientY]);
      if (touches.size === 2) {
        const [a, b] = [...touches.values()];
        const d = Math.hypot(a[0] - b[0], a[1] - b[1]);
        if (!pinch0) { pinch0 = d; dist0 = dist; }
        dist = Math.min(9, Math.max(2.2, dist0 * pinch0 / d));
      } else {
        yaw += (e.clientX - x0) * 0.01;
        pitch = Math.min(1.45, Math.max(-0.2, pitch + (e.clientY - y0) * 0.008));
      }
      start();
    });
    const up = (e) => { touches.delete(e.pointerId); if (touches.size < 2) pinch0 = 0; canvas.style.cursor = "grab"; };
    canvas.addEventListener("pointerup", up);
    canvas.addEventListener("pointercancel", up);
    canvas.addEventListener("wheel", (e) => { e.preventDefault(); dist = Math.min(9, Math.max(2.2, dist * (1 + e.deltaY * 0.001))); start(); }, { passive: false });
    canvas.addEventListener("dblclick", () => { auto = true; start(); });
    window.addEventListener("resize", start);
    for (const [id, set] of [["t-points", (v) => { showPoints = v; }], ["t-cams", (v) => { byCamera = v; }],
      ["t-slices", (v) => { showSlices = v; }]]) {
      $(id).addEventListener("click", (e) => {
        const on = e.currentTarget.getAttribute("aria-pressed") !== "true";
        e.currentTarget.setAttribute("aria-pressed", String(on)); set(on); start();
      });
    }
    return { load };
  })();

  // ---- history -----------------------------------------------------------------
  async function loadHistory() {
    const ul = $("history");
    ul.innerHTML = "<li class='h-meta'>Завантаження…</li>";
    try {
      const scans = await json(`/v1/herds/${herdId}/scans?limit=60`);
      ul.innerHTML = scans.length ? "" : "<li class='h-meta'>Ще немає зважувань.</li>";
      for (const s of scans) {
        const li = document.createElement("li");
        const when = new Date(s.created_utc).toLocaleString("uk-UA", { dateStyle: "short", timeStyle: "short" });
        const diff = s.scale_weight_kg != null && s.weight_kg != null ? ` · ваги ${fmt(s.scale_weight_kg)} кг` : "";
        li.innerHTML = `<span class="h-tag"></span><span class="h-w">${s.weight_kg == null ? "—" : `${fmt(s.weight_kg)} кг`}</span>
          <span class="h-meta">${when}${diff}</span><span class="h-side"><span class="pill ${s.status}">${STATUS_UK[s.status]}</span></span>`;
        li.querySelector(".h-tag").textContent = s.animal_tag;
        ul.append(li);
      }
    } catch (e) { ul.innerHTML = ""; const li = document.createElement("li"); li.className = "h-meta"; li.textContent = e.message; ul.append(li); }
  }
  $("refresh").addEventListener("click", loadHistory);

  // ---- farm --------------------------------------------------------------------
  const kv = (dl, rows) => {
    dl.innerHTML = "";
    for (const [k, v] of rows) { const dt = document.createElement("dt"); dt.textContent = k; const dd = document.createElement("dd"); dd.textContent = v; dl.append(dt, dd); }
  };
  async function loadFarm() {
    try {
      const st = await json(`/v1/herds/${herdId}`);
      const c = st.calibration;
      const counts = st.scans_by_status;
      kv($("farm-kv"), [
        ["Ферма", st.herd.name],
        ["Калібрування", c ? `за ${c.n_reference} контрольними зважуваннями` : "немає"],
        ...(c ? [["Поправка", `${fmt(c.slope, 3)} × прогноз ${c.intercept_kg >= 0 ? "+" : "−"} ${fmt(Math.abs(c.intercept_kg), 1)} кг`],
          ["Інтервал після калібрування", `±${fmt(c.interval_half_width_kg, 1)} кг`]] : []),
        ["Контрольних тварин", `${st.n_reference_animals} з рекомендованих ${st.recommended_reference_animals}`],
        ["Сканувань", `${fmt((counts.trusted || 0) + (counts.flagged || 0) + (counts.refused || 0))} (довірено ${counts.trusted || 0}, позначено ${counts.flagged || 0}, відмов ${counts.refused || 0})`],
      ]);
      const d = st.drift;
      $("farm-drift").textContent = !d ? "" : d.n_weighings < 5
        ? `Контроль дрейфу: потрібно щонайменше 5 вибіркових зважувань після калібрування, зараз ${d.n_weighings}.`
        : d.drifted ? `Виявлено дрейф: прогноз відхиляється в середньому на ${fmt(d.mean_residual_kg, 1)} кг. Перекалібруйте стадо.`
          : `Дрейфу немає: середнє відхилення ${fmt(d.mean_residual_kg, 1)} кг за ${d.n_weighings} зважуваннями.`;
      const m = await json("/v1/model");
      const cv = m.cross_validated || {};
      kv($("model-kv"), [
        ["Навчено на", `${m.n_training_animals} тваринах`],
        ["Діапазон мас", `${m.weight_range_kg.map((x) => fmt(x)).join("–")} кг`],
        ["MAE (перехресна перевірка)", `${fmt(cv.mae, 1)} кг`],
        ["MAPE", `${fmt(cv.mape, 2)} %`],
        ["Інтервал прогнозу", `±${fmt(m.interval_half_width_kg ?? parseFloat(String(m.interval).split("±")[1]), 1)} кг`],
      ]);
    } catch (e) { kv($("farm-kv"), [["Помилка", e.message]]); }
    if (demo && demo.phone_url) {
      $("phone").hidden = false;
      $("phone-url").textContent = demo.phone_url.replace(/#key=.*/, "");
      try {
        const svg = await (await api("/v1/demo/qr.svg")).text();
        $("qr").src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
        $("qr").hidden = false;
      } catch { $("qr").hidden = true; }
    }
  }

  init();
})();
