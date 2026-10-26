import type { ReactNode } from "react";
import type { ScanStatus } from "../api/types";
import { STATUS_UK } from "../lib/text";

export const Pill = ({ status }: { status: ScanStatus }) => <span className={`pill ${status}`}>{STATUS_UK[status]}</span>;

export function Card({ title, aside, children, className = "" }: { title?: ReactNode; aside?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={`card ${className}`}>
      {(title || aside) && (
        <header className="card-h">
          {title && <h2>{title}</h2>}
          {aside}
        </header>
      )}
      {children}
    </section>
  );
}

export function Stat({ label, value, unit, note, tone }: { label: string; value: ReactNode; unit?: string; note?: ReactNode; tone?: "ok" | "warn" | "bad" }) {
  return (
    <div className={`stat ${tone ?? ""}`}>
      <div className="stat-l">{label}</div>
      <div className="stat-v">{value}{unit && <small> {unit}</small>}</div>
      {note && <div className="stat-n">{note}</div>}
    </div>
  );
}

export const ErrorBox = ({ text }: { text: string | null }) => (text ? <p className="err" role="alert">{text}</p> : null);
export const Loading = () => <p className="muted" aria-live="polite">Завантаження…</p>;

export function KeyValues({ rows }: { rows: [string, ReactNode][] }) {
  return (
    <dl className="kv">
      {rows.map(([k, v]) => (
        <div key={k}><dt>{k}</dt><dd>{v}</dd></div>
      ))}
    </dl>
  );
}
