// QR codes of the addresses a phone can open, one per route: through the
// internet (a tunnel, works on mobile data) and on the same Wi-Fi.
import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import type { DemoInfo } from "../api/types";

type Link = NonNullable<DemoInfo["phone_urls"]>[number];

export function phoneLinks(demo: DemoInfo | null): Link[] {
  if (!demo) return [];
  if (demo.phone_urls?.length) return demo.phone_urls;
  return demo.phone_url ? [{ route: "lan", label: "та сама Wi-Fi мережа", url: demo.phone_url }] : [];
}

export function PhoneLinks({ links }: { links: Link[] }) {
  const [i, setI] = useState(0);
  const [qr, setQr] = useState<Record<number, string | null>>({});
  useEffect(() => {
    if (!(i in qr)) api.demoQr(i).then((svg) => setQr((q) => ({ ...q, [i]: svg })));
  }, [i, qr]);
  if (!links.length) return null;
  const link = links[Math.min(i, links.length - 1)];
  const svg = qr[i];
  return (
    <div className="phone-links">
      {links.length > 1 && (
        <div className="seg" role="tablist" aria-label="Адреса для телефона">
          {links.map((l, k) => (
            <button key={l.url} type="button" role="tab" aria-selected={k === i} onClick={() => setI(k)}>
              {l.route === "internet" ? "Будь-яка мережа" : links.filter((x) => x.route === "lan").length > 1 ? `Wi-Fi ${l.url.split("/")[2].split(":")[0]}` : "Та сама Wi-Fi"}
            </button>
          ))}
        </div>
      )}
      <div className="phone">
        {svg ? <img className="qr" alt={`QR-код: ${link.label}`} src={`data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`} /> : <div className="qr" aria-hidden="true" />}
        <div className="small">
          <p>Наведіть камеру телефона на код. {link.route === "internet" ? "Працює в будь-якій мережі, зокрема з мобільним інтернетом." : "Телефон має бути в тій самій Wi-Fi мережі, що й комп’ютер."}</p>
          <p>Адреса: <code>{link.url.replace(/#key=.*/, "")}</code></p>
          {link.route === "lan" && (
            <p className="muted">Не відкривається? Телефон має бути в тій самій Wi-Fi мережі, а брандмауер комп’ютера має пропускати Python. Гостьові й університетські мережі часто ізолюють пристрої, тоді оберіть «Будь-яка мережа» або роздайте інтернет з телефона.</p>
          )}
        </div>
      </div>
    </div>
  );
}

/** Header button and dialog; opens by itself when the page was started with ?phone. */
export function PhoneButton({ demo }: { demo: DemoInfo | null }) {
  const ref = useRef<HTMLDialogElement>(null);
  const links = phoneLinks(demo);
  useEffect(() => {
    const params = new URLSearchParams(location.search);
    if (!links.length || !params.has("phone")) return;
    params.delete("phone");
    const rest = params.toString();
    history.replaceState(null, "", location.pathname + (rest ? `?${rest}` : "") + location.hash);
    // A phone that opened the link does not need the code for itself.
    if (window.matchMedia("(min-width: 700px)").matches) ref.current?.showModal?.();
  }, [links.length]);
  if (!links.length) return null;
  return (
    <>
      <button type="button" className="phone-btn" onClick={() => ref.current?.showModal?.()} title="Відкрити на телефоні">
        <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path fill="currentColor" d="M7 1h10a2 2 0 0 1 2 2v18a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V3a2 2 0 0 1 2-2m0 3v15h10V4zm5 16.2a1 1 0 1 0 0 2 1 1 0 0 0 0-2" /></svg>
        <span>Телефон</span>
      </button>
      <dialog ref={ref} className="dlg wide" onClick={(e) => e.target === ref.current && ref.current?.close()}>
        <h2>Відкрити на телефоні</h2>
        <PhoneLinks links={links} />
        <form method="dialog"><button className="btn" type="submit">Закрити</button></form>
      </dialog>
    </>
  );
}
