// Weigh one animal: three clouds (files or a demo animal), upload with
// progress, the processing stages, then the weight with its interval, the
// trust status and what the cameras saw.
import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import { VIEWS, type DemoSample, type ScanResult, type View } from "../api/types";
import { CloudViewer, cloudFromPreview } from "../components/CloudViewer";
import { Card, ErrorBox, KeyValues, Pill } from "../components/ui";
import { useApp } from "../lib/state";
import { errorUk, fmt, mb, reasonUk, VIEW_UK } from "../lib/text";

type Picked = Partial<Record<View, { blob: Blob; label: string }>>;

const STEPS = ["Пошук підлоги (RANSAC)", "Виділення тварини", "Система координат тіла", "Об’єм і ознаки тулуба", "Ансамбль моделей", "Інтервал і рішення"];
const STEP_SHARE = [0.22, 0.2, 0.08, 0.3, 0.12, 0.08];

/** Files named after their view go to that slot; the rest fill free slots in order. */
export function assignFiles(files: File[]): { picked: Picked; guessed: boolean } {
  const picked: Picked = {};
  const rest: File[] = [];
  for (const f of files) {
    const v = VIEWS.find((x) => new RegExp(x, "i").test(f.webkitRelativePath || f.name));
    if (v && !picked[v]) picked[v] = { blob: f, label: f.name };
    else rest.push(f);
  }
  for (const f of rest) {
    const v = VIEWS.find((x) => !picked[x]);
    if (v) picked[v] = { blob: f, label: f.name };
  }
  return { picked, guessed: rest.length > 0 };
}

function Slot({ view, file, onPick }: { view: View; file?: { blob: Blob; label: string }; onPick: (f: File) => void }) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  return (
    <button
      type="button"
      className={`slot ${file ? "ready" : ""} ${over ? "over" : ""}`}
      onClick={() => input.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => { e.preventDefault(); setOver(false); if (e.dataTransfer.files[0]) onPick(e.dataTransfer.files[0]); }}
    >
      <span className={`cam-dot ${view}`} aria-hidden="true" />
      <span className="slot-t">{VIEW_UK[view]}</span>
      <span className="slot-s">{file ? `${file.label} · ${mb(file.blob.size)}` : "Обрати або перетягнути .ply"}</span>
      <input ref={input} type="file" accept=".ply" hidden onChange={(e) => { const f = e.target.files?.[0]; if (f) onPick(f); e.target.value = ""; }} />
    </button>
  );
}

export function Weigh() {
  const { herdId, demo, changed } = useApp();
  const [files, setFiles] = useState<Picked>({});
  const [guessed, setGuessed] = useState(false);
  const [sample, setSample] = useState<DemoSample | null>(null);
  const [tag, setTag] = useState("");
  const [scale, setScale] = useState("");
  const [upload, setUpload] = useState<{ loaded: number; total: number; done: boolean } | null>(null);
  const [step, setStep] = useState(-1);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ScanResult | null>(null);
  const many = useRef<HTMLInputElement>(null);
  const timer = useRef<number>(0);
  const resultRef = useRef<HTMLDivElement>(null);

  const samples = useMemo(() => (demo ? demo.samples.filter((s) => s.herd_id === herdId) : []), [demo, herdId]);
  useEffect(() => { setSample(null); setFiles({}); setTag(""); setScale(""); }, [herdId]);
  useEffect(() => () => clearTimeout(timer.current), []);

  const ready = VIEWS.every((v) => files[v]) && tag.trim() !== "" && !busy && herdId != null;

  const pickOne = (v: View, f: File) => { setSample(null); setGuessed(false); setFiles((p) => ({ ...p, [v]: { blob: f, label: f.name } })); };
  const pickMany = (list: FileList | null) => {
    if (!list) return;
    const { picked, guessed: g } = assignFiles([...list]);
    setSample(null); setFiles(picked); setGuessed(g);
  };

  async function chooseSample(s: DemoSample) {
    setError(null); setSample(s); setTag(`№${s.animal_id}`); setScale(String(s.scale_weight_kg)); setFiles({}); setGuessed(false);
    try {
      const blobs = await Promise.all(VIEWS.map((v) => api.demoCloud(s.id, v)));
      setFiles(Object.fromEntries(VIEWS.map((v, i) => [v, { blob: blobs[i], label: `${s.id}_${v}.ply` }])) as Picked);
    } catch (e) { setError(`Не вдалося отримати хмари: ${errorUk(e)}`); }
  }

  function animateSteps() {
    let k = 0;
    setStep(0);
    const next = () => {
      if (k >= STEPS.length - 1) return;
      timer.current = window.setTimeout(() => { k += 1; setStep(k); next(); }, STEP_SHARE[k] * 1400);
    };
    next();
  }

  async function weigh() {
    if (!ready || herdId == null) return;
    setError(null); setBusy(true); setStep(-1);
    const form = new FormData();
    form.append("animal_tag", tag.trim());
    const kg = parseFloat(scale.replace(",", "."));
    if (kg > 0) form.append("scale_weight_kg", String(kg));
    form.append("preview", "true");
    if (sample?.rig) form.append("rig", JSON.stringify(sample.rig));
    for (const v of VIEWS) form.append(v, files[v]!.blob, `${v}.ply`);
    const total = VIEWS.reduce((a, v) => a + files[v]!.blob.size, 0);
    setUpload({ loaded: 0, total, done: false });
    try {
      const r = await api.scan(herdId, form, (loaded, t) => setUpload({ loaded, total: t, done: false }),
        () => { setUpload({ loaded: total, total, done: true }); animateSteps(); });
      clearTimeout(timer.current);
      setStep(r.status === "refused" ? -1 : STEPS.length);
      setResult(r);
      changed();
      if (matchMedia("(max-width: 899px)").matches) setTimeout(() => resultRef.current?.scrollIntoView({ behavior: "smooth" }), 50);
    } catch (e) {
      clearTimeout(timer.current); setStep(-1); setError(errorUk(e));
    } finally { setBusy(false); }
  }

  const cloud = useMemo(() => (result?.preview ? cloudFromPreview(result.preview) : null), [result]);
  const p = result?.preview;
  const diff = result && result.weight_kg != null && result.scale_weight_kg != null ? result.weight_kg - result.scale_weight_kg : null;

  return (
    <div className="page two">
      <div className="col">
        <Card title="Зважування тварини">
          <p className="muted">Три кадри глибини однієї тварини: ліва, права й верхня камери станції. Сервер виділяє тварину, будує об’єм тулуба й повертає масу з інтервалом.</p>
          {samples.length > 0 && (
            <div className="samples">
              <h3>Демонстраційні тварини</h3>
              <p className="muted small">Ці тварини не брали участі в навчанні демонстраційної моделі.</p>
              <div className="chips">
                {samples.map((s) => (
                  <button key={s.id} type="button" className="chip" aria-pressed={sample?.id === s.id} onClick={() => chooseSample(s)}>
                    Корова №{s.animal_id} <small>· {s.breed}</small>
                  </button>
                ))}
              </div>
            </div>
          )}
          <div className="slots">
            {VIEWS.map((v) => <Slot key={v} view={v} file={files[v]} onPick={(f) => pickOne(v, f)} />)}
          </div>
          <button type="button" className="btn ghost" onClick={() => many.current?.click()}>Обрати три файли разом</button>
          <input ref={many} type="file" accept=".ply" multiple hidden onChange={(e) => { pickMany(e.target.files); e.target.value = ""; }} />
          {guessed && <p className="warn-note">Назви файлів не вказують камеру, тому порядок вгадано. Перевірте, що кожна хмара у своєму слоті.</p>}
          <div className="fields">
            <label>Бирка тварини<input value={tag} onChange={(e) => setTag(e.target.value)} placeholder="напр. UA-0427" maxLength={100} /></label>
            <label>Маса на вагах, кг <small>(якщо зважено)</small><input value={scale} onChange={(e) => setScale(e.target.value)} inputMode="decimal" placeholder="—" /></label>
          </div>
          <button type="button" className="btn primary wide" disabled={!ready} onClick={weigh}>{busy ? "Обробка…" : "Визначити масу"}</button>
          {upload && (
            <div className="upload" aria-live="polite">
              <div className="bar"><span style={{ width: `${(100 * upload.loaded) / Math.max(1, upload.total)}%` }} /></div>
              <span className="small muted">{upload.done ? (busy ? `Надіслано ${mb(upload.total)}. Обробка на сервері…` : `Надіслано ${mb(upload.total)}`) : `Надсилання: ${mb(upload.loaded)} з ${mb(upload.total)}`}</span>
            </div>
          )}
          {step >= 0 && (
            <ol className="steps">
              {STEPS.map((s, i) => <li key={s} className={i < step ? "done" : i === step ? "run" : ""}>{s}</li>)}
            </ol>
          )}
          <ErrorBox text={error} />
        </Card>
      </div>
      <div className="col" ref={resultRef}>
        {!result && (
          <Card className="placeholder">
            <p className="muted">Результат з’явиться тут: маса, 90 % інтервал, рішення щодо довіри, кадри камер і тривимірна модель тулуба.</p>
          </Card>
        )}
        {result && (
          <>
            <Card title={<>Результат <Pill status={result.status} /></>} aside={<Link className="small" to={`/animals/${encodeURIComponent(result.animal_tag)}`}>{result.animal_tag} · історія</Link>}>
              <div className="weight">
                <span className="weight-v">{fmt(result.weight_kg)}</span><span className="weight-u">кг</span>
              </div>
              <p className="muted">
                {result.weight_kg == null ? "Масу не видано: проскануйте тварину ще раз."
                  : `90 % інтервал: ${fmt(result.lower_kg)}–${fmt(result.upper_kg)} кг (±${fmt((result.upper_kg! - result.lower_kg!) / 2)} кг)`}
              </p>
              {diff != null && (
                <p className="scale-cmp">На вагах <b>{fmt(result.scale_weight_kg)} кг</b>, різниця {diff >= 0 ? "+" : "−"}{fmt(Math.abs(diff))} кг ({fmt(Math.abs((100 * diff) / result.scale_weight_kg!), 1)} %)</p>
              )}
              {result.reasons.length > 0 && <ul className="reasons">{result.reasons.map((r) => <li key={r}>{reasonUk(r)}</li>)}</ul>}
              {p?.measures && (
                <KeyValues rows={[
                  ["Об’єм тулуба", `${fmt(p.measures.trunk_volume_m3, 3)} м³`],
                  ["Довжина тулуба", `${fmt(p.measures.trunk_length_m, 2)} м`],
                  ["Ширина тулуба", `${fmt(p.measures.trunk_width_m, 2)} м`],
                  ["Висота", `${fmt(p.measures.height_m, 2)} м`],
                ]} />
              )}
              <p className="small muted">
                {p?.timing_ms ? `Обробка на сервері ${fmt(p.timing_ms.total_ms)} мс · ` : ""}модель {result.model_version} · сканування #{result.id}
              </p>
            </Card>
            {p && (
              <Card title="Що побачили камери">
                <div className="thumbs">
                  {VIEWS.map((v) => (
                    <figure key={v}>
                      <img src={p.views[v].image} alt={`${VIEW_UK[v]}: тварину виділено кольором`} />
                      <figcaption>{VIEW_UK[v]} · {fmt(p.views[v].animal_points)} точок</figcaption>
                    </figure>
                  ))}
                </div>
              </Card>
            )}
            {cloud && (
              <Card title="Тривимірна модель">
                <CloudViewer cloud={cloud} />
                <p className="small muted">{p!.views_merged ? "Три хмари зведено в одну систему координат за калібруванням станції." : "Для цієї ферми станцію не відкалібровано, тому показано лише верхній ракурс."}</p>
              </Card>
            )}
          </>
        )}
      </div>
    </div>
  );
}
