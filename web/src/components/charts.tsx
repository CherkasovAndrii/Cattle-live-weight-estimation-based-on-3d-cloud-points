// Small SVG charts drawn from the API data: no chart library, so the console
// stays light and works offline. All of them scale with their container.
import type { Scan } from "../api/types";
import { day, fmt } from "../lib/text";

const W = 640, H = 260, PAD = { l: 52, r: 16, t: 14, b: 30 };

function ticks(lo: number, hi: number, n = 5): number[] {
  const span = hi - lo || 1;
  const step0 = span / n, mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((k) => k * mag).find((s) => s >= step0) ?? mag * 10;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(6));
  return out;
}

function Axes({ y, yLabel, xLabels }: { y: { lo: number; hi: number; map: (v: number) => number }; yLabel: string; xLabels: [number, string][] }) {
  return (
    <g className="axes">
      {ticks(y.lo, y.hi).map((v) => (
        <g key={v}>
          <line x1={PAD.l} x2={W - PAD.r} y1={y.map(v)} y2={y.map(v)} className="grid" />
          <text x={PAD.l - 6} y={y.map(v) + 4} textAnchor="end">{fmt(v)}</text>
        </g>
      ))}
      {xLabels.map(([x, t]) => <text key={`${x}-${t}`} x={x} y={H - 8} textAnchor="middle">{t}</text>)}
      <text x={12} y={PAD.t + 4} className="unit">{yLabel}</text>
    </g>
  );
}

/** One animal's weighings over time: estimate with its 90 % interval, scale weights as diamonds. */
export function WeightHistory({ scans }: { scans: Scan[] }) {
  const pts = scans.filter((s) => s.weight_kg != null || s.scale_weight_kg != null)
    .sort((a, b) => a.created_utc.localeCompare(b.created_utc));
  if (pts.length === 0) return <p className="muted">Немає зважувань з масою.</p>;
  const vals = pts.flatMap((s) => [s.lower_kg, s.upper_kg, s.scale_weight_kg, s.weight_kg].filter((v): v is number => v != null));
  const lo = Math.min(...vals) - 10, hi = Math.max(...vals) + 10;
  const x = (i: number) => PAD.l + (pts.length === 1 ? (W - PAD.l - PAD.r) / 2 : (i * (W - PAD.l - PAD.r)) / (pts.length - 1));
  const y = (v: number) => PAD.t + ((hi - v) * (H - PAD.t - PAD.b)) / (hi - lo);
  const est = pts.map((s, i) => [i, s] as const).filter(([, s]) => s.weight_kg != null);
  const band = est.length > 1
    ? `M${est.map(([i, s]) => `${x(i)},${y(s.upper_kg!)}`).join("L")}L${[...est].reverse().map(([i, s]) => `${x(i)},${y(s.lower_kg!)}`).join("L")}Z`
    : "";
  const every = Math.max(1, Math.ceil(pts.length / 6));
  return (
    <figure className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Маса тварини за датами: оцінка системи з 90-відсотковим інтервалом і маса на вагах">
        <Axes y={{ lo, hi, map: y }} yLabel="кг" xLabels={pts.map((s, i) => [x(i), day(s.created_utc)] as [number, string]).filter((_, i) => i % every === 0)} />
        {band && <path d={band} className="band" />}
        {est.length > 1 && <path d={`M${est.map(([i, s]) => `${x(i)},${y(s.weight_kg!)}`).join("L")}`} className="line" />}
        {est.map(([i, s]) => (
          <g key={s.id}>
            {est.length === 1 && <line x1={x(i)} x2={x(i)} y1={y(s.lower_kg!)} y2={y(s.upper_kg!)} className="whisker" />}
            <circle cx={x(i)} cy={y(s.weight_kg!)} r={4.5} className={`dot ${s.status}`}><title>{`${day(s.created_utc)}: ${fmt(s.weight_kg)} кг`}</title></circle>
          </g>
        ))}
        {pts.map((s, i) => s.scale_weight_kg != null && (
          <rect key={`w${s.id}`} x={x(i) - 4.5} y={y(s.scale_weight_kg) - 4.5} width={9} height={9} className="scale" transform={`rotate(45 ${x(i)} ${y(s.scale_weight_kg)})`}>
            <title>{`ваги: ${fmt(s.scale_weight_kg)} кг`}</title>
          </rect>
        ))}
      </svg>
      <figcaption><span className="key est" /> оцінка системи <span className="key bandk" /> 90 % інтервал <span className="key sc" /> маса на вагах</figcaption>
    </figure>
  );
}

/** Estimate against scale weight for every weighed scan of a farm: how well the herd correction fits. */
export function AgreementPlot({ scans }: { scans: Scan[] }) {
  const pts = scans.filter((s) => s.weight_kg != null && s.scale_weight_kg != null);
  if (pts.length < 2) return <p className="muted">Потрібно щонайменше два зважування на вагах, щоб показати узгодженість.</p>;
  const vals = pts.flatMap((s) => [s.weight_kg!, s.scale_weight_kg!]);
  const lo = Math.min(...vals) - 15, hi = Math.max(...vals) + 15;
  const S = 300, P = 44;
  const m = (v: number) => P + ((v - lo) * (S - P - 10)) / (hi - lo);
  const mY = (v: number) => S - m(v);
  const mae = pts.reduce((a, s) => a + Math.abs(s.weight_kg! - s.scale_weight_kg!), 0) / pts.length;
  return (
    <figure className="chart square">
      <svg viewBox={`0 0 ${S} ${S}`} role="img" aria-label={`Оцінка системи проти маси на вагах, ${pts.length} тварин, середня абсолютна похибка ${fmt(mae, 1)} кг`}>
        {ticks(lo, hi, 4).map((v) => (
          <g key={v} className="axes">
            <line x1={P} x2={S - 10} y1={mY(v)} y2={mY(v)} className="grid" />
            <text x={P - 5} y={mY(v) + 4} textAnchor="end">{fmt(v)}</text>
            <text x={m(v)} y={S - 22} textAnchor="middle">{fmt(v)}</text>
          </g>
        ))}
        <line x1={m(lo)} y1={mY(lo)} x2={m(hi)} y2={mY(hi)} className="diag" />
        {pts.map((s) => (
          <circle key={s.id} cx={m(s.scale_weight_kg!)} cy={mY(s.weight_kg!)} r={3.6} className={`dot ${s.status}`}>
            <title>{`${s.animal_tag}: ваги ${fmt(s.scale_weight_kg)} кг, оцінка ${fmt(s.weight_kg)} кг`}</title>
          </circle>
        ))}
        <text x={(S + P) / 2} y={S - 4} textAnchor="middle" className="unit">маса на вагах, кг</text>
      </svg>
      <figcaption>Середня абсолютна похибка {fmt(mae, 1)} кг за {pts.length} зважуваннями. Точки на діагоналі означають збіг з вагами.</figcaption>
    </figure>
  );
}

/** Daily scan counts split by status, last 14 days. */
export function ScansPerDay({ scans }: { scans: Scan[] }) {
  const days: string[] = [];
  const today = new Date();
  for (let k = 13; k >= 0; k--) {
    const d = new Date(today); d.setDate(today.getDate() - k);
    days.push(d.toISOString().slice(0, 10));
  }
  const counts = days.map((d) => {
    const c = { trusted: 0, flagged: 0, refused: 0 };
    for (const s of scans) if (s.created_utc.slice(0, 10) === d) c[s.status] += 1;
    return c;
  });
  const max = Math.max(1, ...counts.map((c) => c.trusted + c.flagged + c.refused));
  const bw = (W - PAD.l - PAD.r) / days.length;
  const y = (v: number) => PAD.t + ((max - v) * (H - PAD.t - PAD.b)) / max;
  return (
    <figure className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Кількість сканувань за останні 14 днів за статусом">
        <Axes y={{ lo: 0, hi: max, map: y }} yLabel="шт." xLabels={days.map((d, i) => [PAD.l + bw * (i + 0.5), i % 2 ? "" : day(d)] as [number, string]).filter(([, t]) => t)} />
        {counts.map((c, i) => {
          let base = 0;
          return (["trusted", "flagged", "refused"] as const).map((k) => {
            const v = c[k];
            if (!v) return null;
            const r = <rect key={`${i}${k}`} x={PAD.l + bw * i + 3} width={bw - 6} y={y(base + v)} height={y(base) - y(base + v)} className={`stack ${k}`}><title>{`${day(days[i])}: ${v}`}</title></rect>;
            base += v;
            return r;
          });
        })}
      </svg>
      <figcaption><span className="key trusted" /> довірено <span className="key flagged" /> позначено <span className="key refused" /> відмова</figcaption>
    </figure>
  );
}
