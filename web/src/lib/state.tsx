// Shared state of the console: connection, farms (herds), the farm in use,
// and the demo information when the server runs `cattle3d demo`.
import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { api } from "../api/client";
import type { DemoInfo, Herd, HerdStatus, Scan } from "../api/types";
import { store, stored } from "./storage";
import { errorUk } from "./text";

const HERD_STORE = "cattle3d-herd";

export type Connection = { state: "checking" } | { state: "online"; version: string } | { state: "offline" };

interface AppState {
  connection: Connection;
  herds: Herd[];
  herdId: number | null;
  setHerdId: (id: number) => void;
  reloadHerds: () => Promise<void>;
  demo: DemoInfo | null;
  error: string | null;
  /** Bumped after every change on the server, so open pages refetch. */
  revision: number;
  changed: () => void;
}

const Ctx = createContext<AppState | null>(null);

export function useApp(): AppState {
  const v = useContext(Ctx);
  if (!v) throw new Error("useApp outside <AppProvider>");
  return v;
}

function remembered(): number | null {
  const v = stored(HERD_STORE);
  return v ? Number(v) : null;
}

export function AppProvider({ children }: { children: ReactNode }) {
  const [connection, setConnection] = useState<Connection>({ state: "checking" });
  const [herds, setHerds] = useState<Herd[]>([]);
  const [herdId, setHerd] = useState<number | null>(remembered());
  const [demo, setDemo] = useState<DemoInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [revision, setRevision] = useState(0);

  const setHerdId = useCallback((id: number) => {
    setHerd(id);
    store(HERD_STORE, String(id));
  }, []);

  const reloadHerds = useCallback(async () => {
    let list = await api.herds();
    if (!list.length) list = [await api.createHerd("Моя ферма")];
    setHerds(list);
    setHerd((cur) => (cur != null && list.some((h) => h.id === cur) ? cur : list[0].id));
  }, []);

  useEffect(() => {
    (async () => {
      try {
        const h = await api.health();
        setConnection({ state: "online", version: h.version });
      } catch {
        setConnection({ state: "offline" });
        return;
      }
      try {
        setDemo(await api.demo());
        await reloadHerds();
      } catch (e) {
        setError(errorUk(e));
      }
    })();
  }, [reloadHerds]);

  const changed = useCallback(() => setRevision((r) => r + 1), []);
  const value = useMemo(
    () => ({ connection, herds, herdId, setHerdId, reloadHerds, demo, error, revision, changed }),
    [connection, herds, herdId, setHerdId, reloadHerds, demo, error, revision, changed],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

/** Fetch on mount and whenever deps or the server revision change. */
export function useLoad<T>(load: () => Promise<T>, deps: unknown[]): { data: T | null; error: string | null; loading: boolean; reload: () => void } {
  const { revision } = useApp();
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [tick, setTick] = useState(0);
  useEffect(() => {
    let live = true;
    setLoading(true);
    load().then(
      (d) => { if (live) { setData(d); setError(null); setLoading(false); } },
      (e) => { if (live) { setError(errorUk(e)); setLoading(false); } },
    );
    return () => { live = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, revision, tick]);
  return { data, error, loading, reload: () => setTick((t) => t + 1) };
}

/** The farm in use: its status and its latest scans, for the overview and farm pages. */
export function useHerdSummary() {
  const { herdId } = useApp();
  return useLoad(async (): Promise<{ status: HerdStatus; scans: Scan[] } | null> => {
    if (herdId == null) return null;
    const [status, scans] = await Promise.all([api.herdStatus(herdId), api.scans(herdId, 500)]);
    return { status, scans };
  }, [herdId]);
}
