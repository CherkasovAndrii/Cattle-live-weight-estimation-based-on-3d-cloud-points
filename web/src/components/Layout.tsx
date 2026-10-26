// Frame of every page: header with the connection and the farm in use, side
// navigation on a computer, a bottom tab bar on a phone.
import { useEffect, useRef, useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { api } from "../api/client";
import { useApp } from "../lib/state";
import { StarField } from "./StarField";

const NAV = [
  { to: "/", label: "Огляд", icon: "M3 13h8V3H3zm0 8h8v-6H3zm10 0h8V11h-8zm0-18v6h8V3z" },
  { to: "/weigh", label: "Зважити", icon: "M12 3a2 2 0 0 1 2 2h5l2 7a4 4 0 0 1-8 0l2-6h-2.2A2 2 0 0 1 13 7.7V19h4v2H7v-2h4V7.7A2 2 0 0 1 9.2 6H7l2 6a4 4 0 0 1-8 0l2-7h5a2 2 0 0 1 2-2" },
  { to: "/herd", label: "Ферма", icon: "M12 3 2 9v12h7v-6h6v6h7V9z" },
  { to: "/animals", label: "Тварини", icon: "M4 6h16v2H4zm0 5h16v2H4zm0 5h10v2H4z" },
  { to: "/model", label: "Модель", icon: "M12 2 3 7v10l9 5 9-5V7zm0 2.3 6.7 3.7L12 11.7 5.3 8zM5 9.7l6 3.3v6.6l-6-3.3zm8 9.9V13l6-3.3v6.6z" },
  { to: "/about", label: "Про систему", icon: "M11 7h2v2h-2zm0 4h2v6h-2zm1-9a10 10 0 1 0 0 20 10 10 0 0 0 0-20" },
];

function Icon({ d }: { d: string }) {
  return <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true"><path d={d} fill="currentColor" /></svg>;
}

function KeyDialog() {
  const ref = useRef<HTMLDialogElement>(null);
  const resolve = useRef<((k: string) => void) | null>(null);
  const [value, setValue] = useState("");
  useEffect(() => {
    api.onKeyNeeded(() => new Promise((r) => { resolve.current = r; ref.current?.showModal(); }));
  }, []);
  return (
    <dialog ref={ref} className="dlg" onClose={() => resolve.current?.(value.trim())}>
      <form method="dialog">
        <h2>Ключ доступу</h2>
        <p className="muted">Сервер захищено ключем. Його надруковано в терміналі поруч з адресою і закладено в QR-код.</p>
        <input autoFocus value={value} onChange={(e) => setValue(e.target.value)} aria-label="Ключ доступу" autoComplete="off" />
        <button className="btn primary" type="submit">Продовжити</button>
      </form>
    </dialog>
  );
}

export function Layout() {
  const { connection, herds, herdId, setHerdId, error } = useApp();
  return (
    <>
      <StarField />
      <header className="top">
        <div className="top-in">
          <NavLink to="/" className="brand">Cattle<b>3D</b></NavLink>
          <span className={`conn ${connection.state}`} title="Стан з’єднання з сервером">
            <span className="dot" />
            <span className="conn-t">
              {connection.state === "online" ? `Сервіс · ${location.host} · v${connection.version}` : connection.state === "offline" ? "Сервер недоступний" : "З’єднання…"}
            </span>
          </span>
          {herds.length > 0 && (
            <label className="farm">
              <span className="sr">Ферма</span>
              <select value={herdId ?? ""} onChange={(e) => setHerdId(Number(e.target.value))}>
                {herds.map((h) => <option key={h.id} value={h.id}>{h.name}</option>)}
              </select>
            </label>
          )}
        </div>
      </header>
      <div className="shell">
        <nav className="nav" aria-label="Розділи">
          {NAV.map((n) => (
            <NavLink key={n.to} to={n.to} end={n.to === "/"} className="nav-i">
              <Icon d={n.icon} /><span>{n.label}</span>
            </NavLink>
          ))}
        </nav>
        <main className="main">
          {connection.state === "offline" && <p className="err" role="alert">Сервер не відповідає. Запустіть <code>cattle3d serve</code> або <code>cattle3d demo</code> і оновіть сторінку.</p>}
          {error && <p className="err" role="alert">{error}</p>}
          <Outlet />
        </main>
      </div>
      <KeyDialog />
    </>
  );
}
