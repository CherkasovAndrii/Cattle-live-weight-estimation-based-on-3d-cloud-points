// Browser storage that may be blocked (private windows, strict settings):
// reads fall back to null and writes are skipped, so the console still works.
export function stored(key: string): string | null {
  try { return localStorage.getItem(key); } catch { return null; }
}

export function store(key: string, value: string): void {
  try { localStorage.setItem(key, value); } catch { /* storage blocked: the value lives for this tab only */ }
}
