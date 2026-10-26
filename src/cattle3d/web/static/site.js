// Language switch and the live demo. The demo talks to the REST API that
// serves this page (`cattle3d serve`); opened anywhere else, the page shows
// the static example ticket only.
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch { /* storage blocked */ } },
    sget(k) { try { return sessionStorage.getItem(k); } catch { return null; } },
    sset(k, v) { try { sessionStorage.setItem(k, v); } catch { /* storage blocked */ } },
  };

  // ---- language ---------------------------------------------------------------
  let lang = "uk";
  const T = {
    uk: { trusted: "довірено", flagged: "позначено", refused: "відмова", kg: "кг", animals: "тварин",
          notCal: "не відкалібровано", weighing: "Зважую…", calibrated: "відкалібровано", ref: "контрольних тварин",
          drift: "дрейф", none: "—", herdMissing: "Спершу створіть стадо.", live: "Результат зважування через API." },
    en: { trusted: "trusted", flagged: "flagged", refused: "refused", kg: "kg", animals: "animals",
          notCal: "not calibrated", weighing: "Weighing…", calibrated: "calibrated", ref: "reference animals",
          drift: "drift", none: "—", herdMissing: "Create a herd first.", live: "Result of a weighing through the API." },
  };
  const t = (k) => T[lang][k];
  // The API explains decisions in English; show its known reasons in Ukrainian too.
  const REASONS_UK = [
    [/^herd not calibrated.*/, "стадо не відкалібровано: очікуйте систематичний зсув; зважте 20 тварин і відкалібруйте"],
    [/^body volume outside the training range$/, "об’єм тулуба поза діапазоном навчання"],
    [/^too few points in left view$/, "замало точок у лівому ракурсі"],
    [/^too few points in right view$/, "замало точок у правому ракурсі"],
    [/^too few points in top view$/, "замало точок у верхньому ракурсі"],
    [/^depth noise in (left|right|top) view ([\d.]+) % of range.*$/, "шум глибини в кадрі ($1) $2 % відстані, більше за допустимі 0,75 %"],
    [/^many scattered missing pixels in (left|right|top) view.*$/, "багато розсіяних пікселів без глибини в кадрі ($1)"],
    [/^rescan: (.*)$/, "пересканувати: $1"],
  ];
  const reasonText = (r) => {
    if (lang !== "uk") return r;
    for (const [re, uk] of REASONS_UK) if (re.test(r)) return r.replace(re, uk);
    return r;
  };
  const num = (x, d = 0) => (x == null ? t("none") : x.toLocaleString(lang === "uk" ? "uk-UA" : "en-GB",
    { maximumFractionDigits: d, minimumFractionDigits: d }));

  function setLang(next) {
    lang = next === "en" ? "en" : "uk";
    document.documentElement.lang = lang;
    document.documentElement.dataset.lang = lang;
    for (const b of document.querySelectorAll("[data-set-lang]")) {
      b.setAttribute("aria-pressed", String(b.dataset.setLang === lang));
    }
    store.set("cattle3d-lang", lang);
    if (lastScan) renderTicket(lastScan);
    if (lastHerd) renderHerd(lastHerd);
  }
  for (const b of document.querySelectorAll("[data-set-lang]")) {
    b.addEventListener("click", () => setLang(b.dataset.setLang));
  }

  // ---- ticket ------------------------------------------------------------------
  let lastScan = null;
  let lastHerd = null;
  const word = (uk, en) => `<span lang="uk">${uk}</span><span lang="en">${en}</span>`;

  function renderTicket(s) {
    lastScan = s;
    $("t-no").textContent = `№ ${s.id}`;
    $("t-weight").textContent = s.weight_kg == null ? "—" : num(s.weight_kg);
    const stamp = $("t-status");
    const cls = { trusted: "ok", flagged: "warn", refused: "bad" }[s.status];
    stamp.className = `stamp ${cls}`;
    stamp.innerHTML = word(T.uk[s.status], T.en[s.status]);
    $("t-interval").textContent = s.lower_kg == null ? "—" : `${num(s.lower_kg)} – ${num(s.upper_kg)}`;
    $("t-scale").textContent = s.scale_weight_kg == null ? "—" : num(s.scale_weight_kg, 1);
    $("t-tag").textContent = s.animal_tag;
    $("t-cal").innerHTML = s.calibration_id == null ? word(T.uk.notCal, T.en.notCal) : `#${s.calibration_id}`;
    const note = $("t-note");
    note.textContent = "";
    const lines = s.reasons.length ? s.reasons : [t("live")];
    for (const r of lines) {
      const p = document.createElement("span");
      p.textContent = reasonText(r);
      p.style.display = "block";
      note.append(p);
    }
    document.querySelector(".views").hidden = true;  // the example cow's views, not this animal's
  }

  // ---- API ---------------------------------------------------------------------
  async function api(path, options = {}) {
    const headers = new Headers(options.headers || {});
    const key = $("f-key").value.trim();
    if (key) headers.set("X-API-Key", key);
    const res = await fetch(path, { ...options, headers });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) {
      const detail = Array.isArray(body.detail) ? body.detail.map((d) => d.msg).join("; ") : body.detail;
      const known = {
        401: ["Потрібен API-ключ: його друкує команда cattle3d demo у терміналі.", "An API key is needed: cattle3d demo prints it in the terminal."],
        413: ["Файл завеликий для хмари точок Kinect.", "The file is too large for a Kinect point cloud."],
      }[res.status];
      throw new Error(known ? known[lang === "uk" ? 0 : 1] : detail || `HTTP ${res.status}`);
    }
    return body;
  }
  const showError = (e) => { $("demo-error").textContent = e ? String(e.message || e) : ""; };

  function renderHerd(h) {
    lastHerd = h;
    const c = h.calibration;
    const parts = [
      h.herd.name,
      c ? `${t("calibrated")}: ${num(c.slope, 3)} × m ${c.intercept_kg >= 0 ? "+" : "−"} ${num(Math.abs(c.intercept_kg), 1)} ${t("kg")}, ±${num(c.interval_half_width_kg)} ${t("kg")}`
        : t("notCal"),
      `${h.n_reference_animals}/${h.recommended_reference_animals} ${t("ref")}`,
    ];
    if (h.drift) parts.push(`${t("drift")}: ${h.drift.message}`);
    $("herd-status").textContent = parts.join(" · ");
  }

  async function loadHerds(selectId) {
    const herds = await api("/v1/herds");
    const sel = $("f-herd");
    sel.textContent = "";
    for (const h of herds) sel.append(new Option(h.name, String(h.id)));
    if (selectId) sel.value = String(selectId);
    if (sel.value) renderHerd(await api(`/v1/herds/${sel.value}`));
    else $("herd-status").textContent = t("herdMissing");
  }

  async function startDemo() {
    try {
      const res = await fetch("/health", { cache: "no-store" });
      if (!res.ok) return;
      const health = await res.json();
      if (health.status !== "ok") return;
    } catch { return; }  // not served by the API: keep the static example
    $("offline").hidden = true;
    $("demo").hidden = false;
    $("f-key").value = store.sget("cattle3d-key") || "";
    $("f-key").addEventListener("change", () => { store.sset("cattle3d-key", $("f-key").value.trim()); loadHerds().catch(showError); });
    $("f-herd").addEventListener("change", async () => {
      try { renderHerd(await api(`/v1/herds/${$("f-herd").value}`)); showError(); } catch (e) { showError(e); }
    });
    $("b-new-herd").addEventListener("click", async () => {
      const name = $("f-new-herd").value.trim();
      if (!name) return;
      try { const h = await api("/v1/herds", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name }) });
            $("f-new-herd").value = ""; await loadHerds(h.id); showError(); } catch (e) { showError(e); }
    });
    $("b-cal").addEventListener("click", async () => {
      const id = $("f-herd").value;
      if (!id) return showError(t("herdMissing"));
      try { await api(`/v1/herds/${id}/calibration`, { method: "POST" }); renderHerd(await api(`/v1/herds/${id}`)); showError(); }
      catch (e) { showError(e); }
    });
    $("demo").addEventListener("submit", async (ev) => {
      ev.preventDefault();
      const id = $("f-herd").value;
      if (!id) return showError(t("herdMissing"));
      const form = new FormData();
      form.append("animal_tag", $("f-tag").value.trim());
      for (const v of ["left", "right", "top"]) form.append(v, $(`f-${v}`).files[0]);
      if ($("f-scale").value) form.append("scale_weight_kg", $("f-scale").value);
      const button = $("b-scan");
      button.disabled = true;
      $("herd-status").textContent = t("weighing");
      try {
        renderTicket(await api(`/v1/herds/${id}/scans`, { method: "POST", body: form }));
        renderHerd(await api(`/v1/herds/${id}`));
        showError();
        $("ticket").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
      } catch (e) { showError(e); renderHerd(await api(`/v1/herds/${id}`).catch(() => lastHerd)); }
      finally { button.disabled = false; }
    });
    await loadHerds().catch(showError);
  }

  // ---- motion: scroll reveal, the light on glass panels, the hero counter --------
  const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const reveal = document.querySelectorAll("[data-reveal]");
  if ("IntersectionObserver" in window && !still) {
    const io = new IntersectionObserver((entries) => {
      for (const e of entries) if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
    }, { rootMargin: "0px 0px -8% 0px" });
    reveal.forEach((el) => io.observe(el));
  } else {
    reveal.forEach((el) => el.classList.add("in"));
  }
  document.addEventListener("pointermove", (e) => {
    const card = e.target.closest?.(".glass");
    if (!card) return;
    const r = card.getBoundingClientRect();
    card.style.setProperty("--mx", `${e.clientX - r.left}px`);
    card.style.setProperty("--my", `${e.clientY - r.top}px`);
  }, { passive: true });
  const counter = document.querySelector(".hud [data-count]");
  if (counter && !still) {
    const target = Number(counter.dataset.count);
    const begin = performance.now() + 2300;
    const tick = (now) => {
      const k = Math.min(1, Math.max(0, (now - begin) / 1400));
      counter.textContent = String(Math.round(target * (1 - Math.pow(1 - k, 4))));
      if (k < 1) requestAnimationFrame(tick);
    };
    counter.textContent = "0";
    requestAnimationFrame(tick);
  }

  const saved = store.get("cattle3d-lang");
  setLang(saved || "uk");
  startDemo();
})();
