// Animals of the farm built from its scans, and one animal's weight history.
import { useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../api/client";
import type { Scan } from "../api/types";
import { WeightHistory } from "../components/charts";
import { Card, ErrorBox, Loading, Pill, Stat } from "../components/ui";
import { useApp, useLoad } from "../lib/state";
import { fmt, reasonUk, when } from "../lib/text";

export interface AnimalRow {
  tag: string;
  scans: number;
  last: Scan;
  lastWeight: number | null;
  gainKgPerDay: number | null;
}

/** One row per animal tag; the gain is a straight line through the trusted and flagged weights. */
export function animalsFrom(scans: Scan[]): AnimalRow[] {
  const by = new Map<string, Scan[]>();
  for (const s of scans) by.set(s.animal_tag, [...(by.get(s.animal_tag) ?? []), s]);
  return [...by.entries()].map(([tag, list]) => {
    const sorted = [...list].sort((a, b) => b.created_utc.localeCompare(a.created_utc));
    const weighed = sorted.filter((s) => s.weight_kg != null);
    let gain: number | null = null;
    if (weighed.length >= 2) {
      const t = weighed.map((s) => Date.parse(s.created_utc) / 864e5), w = weighed.map((s) => s.weight_kg!);
      const mt = t.reduce((a, b) => a + b) / t.length, mw = w.reduce((a, b) => a + b) / w.length;
      const sxx = t.reduce((a, x) => a + (x - mt) ** 2, 0);
      if (sxx > 1) gain = t.reduce((a, x, i) => a + (x - mt) * (w[i] - mw), 0) / sxx;
    }
    return { tag, scans: list.length, last: sorted[0], lastWeight: weighed[0]?.weight_kg ?? null, gainKgPerDay: gain };
  }).sort((a, b) => b.last.created_utc.localeCompare(a.last.created_utc));
}

export function Animals() {
  const { herdId } = useApp();
  const [q, setQ] = useState("");
  const { data, error, loading } = useLoad(() => (herdId == null ? Promise.resolve([]) : api.scans(herdId, 500)), [herdId]);
  const rows = useMemo(() => animalsFrom(data ?? []), [data]);
  if (loading && !data) return <Loading />;
  const shown = rows.filter((r) => r.tag.toLowerCase().includes(q.trim().toLowerCase()));
  return (
    <div className="page">
      <div className="page-h">
        <h1>Тварини</h1>
        <input className="search" value={q} onChange={(e) => setQ(e.target.value)} placeholder="Пошук за біркою" aria-label="Пошук за біркою" />
      </div>
      <ErrorBox text={error} />
      <Card>
        {shown.length === 0 ? <p className="muted">{rows.length ? "Нічого не знайдено." : "Ще немає тварин: вони з’являються після першого зважування."}</p> : (
          <div className="table-wrap">
            <table className="table">
              <thead><tr><th>Бирка</th><th className="num">Остання маса, кг</th><th className="num">Приріст, кг/добу</th><th className="num">Сканувань</th><th>Останнє</th><th>Статус</th></tr></thead>
              <tbody>
                {shown.map((r) => (
                  <tr key={r.tag}>
                    <td><Link to={`/animals/${encodeURIComponent(r.tag)}`}>{r.tag}</Link></td>
                    <td className="num">{fmt(r.lastWeight)}</td>
                    <td className="num">{r.gainKgPerDay == null ? "—" : `${r.gainKgPerDay >= 0 ? "+" : "−"}${fmt(Math.abs(r.gainKgPerDay), 2)}`}</td>
                    <td className="num">{r.scans}</td>
                    <td>{when(r.last.created_utc)}</td>
                    <td><Pill status={r.last.status} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}

export function Animal() {
  const { tag = "" } = useParams();
  const { herdId } = useApp();
  const { data, error, loading } = useLoad(() => (herdId == null ? Promise.resolve([]) : api.animal(herdId, tag)), [herdId, tag]);
  if (loading && !data) return <Loading />;
  const scans = data ?? [];
  const row = animalsFrom(scans)[0];
  const weighed = scans.filter((s) => s.weight_kg != null && s.scale_weight_kg != null);
  const mae = weighed.length ? weighed.reduce((a, s) => a + Math.abs(s.weight_kg! - s.scale_weight_kg!), 0) / weighed.length : null;
  return (
    <div className="page">
      <div className="page-h">
        <h1>Тварина {tag}</h1>
        <Link to="/weigh" className="btn">Зважити ще раз</Link>
      </div>
      <ErrorBox text={error} />
      {!row ? <Card><p className="muted">Ця тварина ще не зважувалась на вибраній фермі.</p></Card> : (
        <>
          <div className="stats">
            <Stat label="Остання маса" value={fmt(row.lastWeight)} unit="кг" note={when(row.last.created_utc)} />
            <Stat label="Приріст" value={row.gainKgPerDay == null ? "—" : fmt(row.gainKgPerDay, 2)} unit={row.gainKgPerDay == null ? undefined : "кг/добу"} note="лінійний тренд усіх оцінок" />
            <Stat label="Сканувань" value={fmt(row.scans)} />
            <Stat label="Похибка проти ваг" value={mae == null ? "—" : fmt(mae, 1)} unit={mae == null ? undefined : "кг"} note={`${weighed.length} зважувань на вагах`} />
          </div>
          <Card title="Маса за датами"><WeightHistory scans={scans} /></Card>
          <Card title="Усі зважування">
            <div className="table-wrap">
              <table className="table">
                <thead><tr><th>Час</th><th className="num">Маса, кг</th><th className="num">Інтервал</th><th className="num">Ваги, кг</th><th>Статус</th><th>Примітки</th></tr></thead>
                <tbody>
                  {scans.map((s) => (
                    <tr key={s.id}>
                      <td>{when(s.created_utc)}</td>
                      <td className="num">{fmt(s.weight_kg)}</td>
                      <td className="num">{s.lower_kg == null ? "—" : `${fmt(s.lower_kg)}–${fmt(s.upper_kg)}`}</td>
                      <td className="num">{fmt(s.scale_weight_kg)}</td>
                      <td><Pill status={s.status} /></td>
                      <td className="small muted">{s.reasons.map(reasonUk).join("; ")}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}
    </div>
  );
}
