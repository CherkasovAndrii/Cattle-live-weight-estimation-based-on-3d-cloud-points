// The farm at a glance: what happened in the last two weeks, how many
// estimates can be trusted, whether the herd is calibrated and drifting.
import { Link } from "react-router-dom";
import { api } from "../api/client";
import type { HerdStatus, Scan } from "../api/types";
import { AgreementPlot, ScansPerDay } from "../components/charts";
import { Card, ErrorBox, Loading, Pill, Stat } from "../components/ui";
import { useApp, useLoad } from "../lib/state";
import { fmt, plural, when } from "../lib/text";

export function driftText(st: HerdStatus): { text: string; tone: "ok" | "warn" | "bad" } {
  const d = st.drift;
  if (!st.calibrated) return { text: "стадо ще не відкалібровано", tone: "warn" };
  if (!d || d.n_weighings < 5) return { text: `потрібно 5 вибіркових зважувань, є ${d?.n_weighings ?? 0}`, tone: "warn" };
  if (d.drifted) return { text: `зсув ${fmt(d.mean_residual_kg, 1)} кг, перекалібруйте`, tone: "bad" };
  return { text: `зсуву немає (${fmt(d.mean_residual_kg, 1)} кг)`, tone: "ok" };
}

export function Overview() {
  const { herdId, herds } = useApp();
  const herd = herds.find((h) => h.id === herdId);
  const { data, error, loading } = useLoad(async () => {
    if (herdId == null) return null;
    const [status, scans] = await Promise.all([api.herdStatus(herdId), api.scans(herdId, 500)]);
    return { status, scans };
  }, [herdId]);

  if (herdId == null || (loading && !data)) return <Loading />;
  if (!data) return <ErrorBox text={error} />;
  const { status, scans } = data;
  const since = Date.now() - 14 * 864e5;
  const recent = scans.filter((s) => Date.parse(s.created_utc) >= since);
  const trusted = recent.filter((s) => s.status === "trusted").length;
  const animals = new Set(scans.map((s) => s.animal_tag)).size;
  const drift = driftText(status);
  const c = status.calibration;

  return (
    <div className="page">
      <div className="page-h">
        <h1>{herd?.name ?? "Ферма"}</h1>
        <Link to="/weigh" className="btn primary">Зважити тварину</Link>
      </div>
      <ErrorBox text={error} />
      <div className="stats">
        <Stat label="Тварин у системі" value={fmt(animals)} note={`${fmt(scans.length)} ${plural(scans.length, "сканування", "сканування", "сканувань")} загалом`} />
        <Stat label="Сканувань за 14 днів" value={fmt(recent.length)} note={recent.length ? `довірено ${fmt((100 * trusted) / recent.length)} %` : "ще немає"} />
        <Stat label="Калібрування" value={c ? `${c.n_reference}` : "немає"} unit={c ? "тварин" : undefined}
          note={c ? `інтервал ±${fmt(c.interval_half_width_kg, 1)} кг` : `контрольних тварин ${status.n_reference_animals} з ${status.recommended_reference_animals}`}
          tone={c ? "ok" : "warn"} />
        <Stat label="Контроль дрейфу" value={drift.tone === "ok" ? "норма" : drift.tone === "bad" ? "дрейф" : "чекає"} note={drift.text} tone={drift.tone} />
      </div>
      <div className="grid2">
        <Card title="Сканування за 14 днів"><ScansPerDay scans={scans} /></Card>
        <Card title="Оцінка проти ваг"><AgreementPlot scans={scans} /></Card>
      </div>
      <Card title="Останні зважування" aside={<Link to="/herd" className="small">усі сканування</Link>}>
        <RecentTable scans={scans.slice(0, 8)} />
      </Card>
    </div>
  );
}

export function RecentTable({ scans }: { scans: Scan[] }) {
  if (!scans.length) return <p className="muted">Ще немає зважувань. Почніть з розділу «Зважити».</p>;
  return (
    <div className="table-wrap">
      <table className="table">
        <thead><tr><th>Тварина</th><th>Час</th><th className="num">Маса, кг</th><th className="num">Інтервал</th><th className="num">Ваги, кг</th><th>Статус</th></tr></thead>
        <tbody>
          {scans.map((s) => (
            <tr key={s.id}>
              <td><Link to={`/animals/${encodeURIComponent(s.animal_tag)}`}>{s.animal_tag}</Link></td>
              <td>{when(s.created_utc)}</td>
              <td className="num">{fmt(s.weight_kg)}</td>
              <td className="num">{s.lower_kg == null ? "—" : `${fmt(s.lower_kg)}–${fmt(s.upper_kg)}`}</td>
              <td className="num">{fmt(s.scale_weight_kg)}</td>
              <td><Pill status={s.status} /></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
