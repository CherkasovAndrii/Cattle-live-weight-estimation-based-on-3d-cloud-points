// The farm: herd calibration from reference weighings, drift control from
// spot weighings, all scans with scale readings added after the fact, new
// farms, and the phone link.
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import type { Scan, ScanStatus } from "../api/types";
import { Card, ErrorBox, KeyValues, Loading, Pill } from "../components/ui";
import { useApp, useLoad } from "../lib/state";
import { errorUk, fmt, when } from "../lib/text";
import { driftText } from "./Overview";

function WeighingCell({ scan }: { scan: Scan }) {
  const { changed } = useApp();
  const [value, setValue] = useState("");
  const [err, setErr] = useState<string | null>(null);
  if (scan.scale_weight_kg != null) return <>{fmt(scan.scale_weight_kg)}</>;
  if (scan.status === "refused") return <>—</>;
  const save = async () => {
    const kg = parseFloat(value.replace(",", "."));
    if (!(kg > 0)) return;
    try { await api.recordWeighing(scan.id, kg); changed(); } catch (e) { setErr(errorUk(e)); }
  };
  return (
    <span className="inline-form">
      <input aria-label={`Маса на вагах для ${scan.animal_tag}`} value={value} onChange={(e) => setValue(e.target.value)} inputMode="decimal" placeholder="кг" onKeyDown={(e) => e.key === "Enter" && save()} />
      <button type="button" className="btn tiny" onClick={save} disabled={!value}>Внести</button>
      {err && <span className="err small">{err}</span>}
    </span>
  );
}

export function Herd() {
  const { herdId, herds, demo, reloadHerds, setHerdId, changed } = useApp();
  const [filter, setFilter] = useState<ScanStatus | "all">("all");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [qr, setQr] = useState<string | null>(null);
  const { data, error, loading } = useLoad(async () => {
    if (herdId == null) return null;
    const [status, scans] = await Promise.all([api.herdStatus(herdId), api.scans(herdId, 500)]);
    return { status, scans };
  }, [herdId]);
  useEffect(() => { if (demo?.phone_url) api.demoQr().then(setQr); }, [demo]);

  if (herdId == null || (loading && !data)) return <Loading />;
  if (!data) return <ErrorBox text={error} />;
  const { status, scans } = data;
  const c = status.calibration;
  const drift = driftText(status);
  const shown = filter === "all" ? scans : scans.filter((s) => s.status === filter);
  const herd = herds.find((h) => h.id === herdId);
  const share = Math.min(1, status.n_reference_animals / status.recommended_reference_animals);

  const calibrate = async () => {
    setBusy(true); setErr(null); setMsg(null);
    try {
      const r = (await api.calibrate(herdId)) as { n_reference: number; interval_half_width_kg: number };
      setMsg(`Стадо відкалібровано за ${r.n_reference} тваринами. Нові зважування використовують поправку, інтервал ±${fmt(r.interval_half_width_kg, 1)} кг.`);
      changed();
    } catch (e) { setErr(errorUk(e)); } finally { setBusy(false); }
  };
  const create = async () => {
    if (!name.trim()) return;
    try { const h = await api.createHerd(name.trim()); await reloadHerds(); setHerdId(h.id); setName(""); } catch (e) { setErr(errorUk(e)); }
  };

  return (
    <div className="page">
      <div className="page-h"><h1>{herd?.name ?? "Ферма"}</h1></div>
      <div className="grid2">
        <Card title="Калібрування стада">
          <p className="muted">Модель навчено на інших стадах. На новій фермі 20–30 тварин зважують на вагах і сканують, після чого система підбирає нахил і зсув для цього стада.</p>
          <div className="progress" aria-label="Контрольні тварини">
            <div className="bar"><span style={{ width: `${100 * share}%` }} /></div>
            <span className="small">{status.n_reference_animals} з {status.recommended_reference_animals} контрольних тварин</span>
          </div>
          {c ? (
            <KeyValues rows={[
              ["Поправка", `${fmt(c.slope, 3)} × прогноз ${c.intercept_kg >= 0 ? "+" : "−"} ${fmt(Math.abs(c.intercept_kg), 1)} кг`],
              ["Контрольних тварин", fmt(c.n_reference)],
              ["Інтервал після калібрування", `±${fmt(c.interval_half_width_kg, 1)} кг`],
              ["Дата", when(c.created_utc)],
            ]} />
          ) : <p className="warn-note">Стадо не відкалібровано: оцінки позначаються як недовірені.</p>}
          <button type="button" className="btn primary" onClick={calibrate} disabled={busy || status.n_reference_animals < 3}>
            {c ? "Перекалібрувати" : "Відкалібрувати"}
          </button>
          {status.n_reference_animals < 10 && <p className="small muted">До 10 контрольних тварин система уточнює лише зсув; нахил підбирається з 10 тварин, щонайменше потрібно 3.</p>}
          {msg && <p className="ok-note">{msg}</p>}
          <ErrorBox text={err} />
        </Card>
        <Card title="Контроль дрейфу">
          <p className="muted">Після калібрування частину тварин періодично зважують на вагах. Якщо середня різниця між вагами й оцінкою стабільно відходить від нуля, система просить перекалібрувати стадо.</p>
          <p className={`${drift.tone}-note`}>{drift.text}</p>
          {status.drift && status.drift.mean_residual_kg != null && (
            <KeyValues rows={[
              ["Вибіркових зважувань", fmt(status.drift.n_weighings)],
              ["Середня різниця", `${fmt(status.drift.mean_residual_kg, 1)} кг`],
              ["95 % довірчий інтервал", `${fmt(status.drift.ci_low_kg, 1)}…${fmt(status.drift.ci_high_kg, 1)} кг`],
            ]} />
          )}
        </Card>
      </div>
      <Card
        title="Сканування"
        aside={
          <div className="seg" role="group" aria-label="Фільтр за статусом">
            {(["all", "trusted", "flagged", "refused"] as const).map((k) => (
              <button key={k} type="button" aria-pressed={filter === k} onClick={() => setFilter(k)}>
                {k === "all" ? `Усі ${scans.length}` : `${{ trusted: "Довірено", flagged: "Позначено", refused: "Відмова" }[k]} ${status.scans_by_status[k] ?? 0}`}
              </button>
            ))}
          </div>
        }
      >
        {shown.length === 0 ? <p className="muted">Немає сканувань.</p> : (
          <div className="table-wrap">
            <table className="table">
              <thead><tr><th>#</th><th>Тварина</th><th>Час</th><th className="num">Маса, кг</th><th className="num">Без поправки</th><th>Ваги, кг</th><th>Статус</th></tr></thead>
              <tbody>
                {shown.map((s) => (
                  <tr key={s.id}>
                    <td className="muted">{s.id}</td>
                    <td><Link to={`/animals/${encodeURIComponent(s.animal_tag)}`}>{s.animal_tag}</Link></td>
                    <td>{when(s.created_utc)}</td>
                    <td className="num">{fmt(s.weight_kg)}</td>
                    <td className="num muted">{fmt(s.raw_weight_kg)}</td>
                    <td><WeighingCell scan={s} /></td>
                    <td><Pill status={s.status} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
      <div className="grid2">
        <Card title="Нова ферма">
          <p className="muted">Кожна ферма має власне калібрування, бо породи й умови зйомки різняться.</p>
          <div className="inline-form">
            <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Назва ферми" aria-label="Назва ферми" maxLength={200} onKeyDown={(e) => e.key === "Enter" && create()} />
            <button type="button" className="btn" onClick={create} disabled={!name.trim()}>Створити</button>
          </div>
        </Card>
        {demo?.phone_url && (
          <Card title="Відкрити на телефоні">
            <div className="phone">
              {qr && <img className="qr" alt="QR-код адреси для телефона" src={`data:image/svg+xml;charset=utf-8,${encodeURIComponent(qr)}`} />}
              <p className="small">Наведіть камеру телефона на код. Адреса: <code>{demo.phone_url.replace(/#key=.*/, "")}</code></p>
            </div>
          </Card>
        )}
      </div>
    </div>
  );
}
