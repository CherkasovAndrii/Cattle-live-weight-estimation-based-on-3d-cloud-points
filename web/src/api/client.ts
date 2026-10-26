// One place that talks HTTP. Every /v1 call carries the API key; a 401 asks the
// operator for the key once and retries. Errors surface as ApiError with the
// server's own message, which lib/text.ts turns into Ukrainian.
import type { DemoInfo, Herd, HerdStatus, ModelCard, Scan, ScanResult, View } from "./types";

const KEY_STORE = "cattle3d-key";

export class ApiError extends Error {
  constructor(public status: number, public detail: unknown) {
    super(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
}

function load(key: string): string | null {
  try { return localStorage.getItem(key); } catch { return null; }
}
function save(key: string, value: string): void {
  try { localStorage.setItem(key, value); } catch { /* storage blocked: key lives for this tab only */ }
}

type KeyPrompt = () => Promise<string>;

export class Api {
  private key: string;
  private prompt: KeyPrompt | null = null;

  constructor(private base = "") {
    // The server prints links with #key=…: keep the key and drop it from the address bar.
    const fromHash = new URLSearchParams(location.hash.slice(1)).get("key");
    if (fromHash) {
      save(KEY_STORE, fromHash);
      history.replaceState(null, "", location.pathname + location.search);
    }
    this.key = fromHash ?? load(KEY_STORE) ?? "";
  }

  onKeyNeeded(prompt: KeyPrompt): void { this.prompt = prompt; }

  setKey(key: string): void { this.key = key; save(KEY_STORE, key); }

  private async request(path: string, init: RequestInit = {}, retried = false): Promise<Response> {
    const res = await fetch(this.base + path, { ...init, headers: { ...(init.headers ?? {}), "X-API-Key": this.key } });
    if (res.status === 401 && this.prompt && !retried) {
      this.setKey(await this.prompt());
      return this.request(path, init, true);
    }
    if (!res.ok) {
      let detail: unknown = res.statusText;
      try { detail = (await res.json()).detail; } catch { /* not JSON */ }
      throw new ApiError(res.status, detail);
    }
    return res;
  }

  private async json<T>(path: string, init?: RequestInit): Promise<T> {
    return (await this.request(path, init)).json() as Promise<T>;
  }

  private send<T>(method: string, path: string, body?: unknown): Promise<T> {
    return this.json<T>(path, {
      method,
      headers: { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  }

  async health(): Promise<{ status: string; version: string }> {
    const res = await fetch(this.base + "/health");
    if (!res.ok) throw new ApiError(res.status, res.statusText);
    return res.json();
  }

  model(): Promise<ModelCard> { return this.json("/v1/model"); }
  herds(): Promise<Herd[]> { return this.json("/v1/herds"); }
  createHerd(name: string): Promise<Herd> { return this.send("POST", "/v1/herds", { name }); }
  herdStatus(id: number): Promise<HerdStatus> { return this.json(`/v1/herds/${id}`); }
  calibrate(id: number) { return this.send("POST", `/v1/herds/${id}/calibration`); }
  scans(herdId: number, limit = 200): Promise<Scan[]> { return this.json(`/v1/herds/${herdId}/scans?limit=${limit}`); }
  animal(herdId: number, tag: string): Promise<Scan[]> {
    return this.json(`/v1/herds/${herdId}/animals/${encodeURIComponent(tag)}/scans`);
  }
  recordWeighing(scanId: number, weightKg: number): Promise<Scan> {
    return this.send("PUT", `/v1/scans/${scanId}/weighing`, { weight_kg: weightKg });
  }

  async demo(): Promise<DemoInfo | null> {
    try { return await this.json<DemoInfo>("/v1/demo"); } catch (e) {
      if (e instanceof ApiError && e.status === 404) return null;
      throw e;
    }
  }
  async demoCloud(sampleId: string, view: View): Promise<Blob> {
    return (await this.request(`/v1/demo/samples/${encodeURIComponent(sampleId)}/${view}.ply`)).blob();
  }
  async demoQr(): Promise<string | null> {
    try { return await (await this.request("/v1/demo/qr.svg")).text(); } catch { return null; }
  }

  /** Upload three clouds with progress (fetch has no upload progress, XHR does). */
  scan(
    herdId: number,
    form: FormData,
    progress: (loaded: number, total: number) => void,
    uploaded: () => void,
  ): Promise<ScanResult> {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${this.base}/v1/herds/${herdId}/scans`);
      xhr.setRequestHeader("X-API-Key", this.key);
      xhr.upload.onprogress = (e) => { if (e.lengthComputable) progress(e.loaded, e.total); };
      xhr.upload.onload = uploaded;
      xhr.onerror = () => reject(new ApiError(0, "network"));
      xhr.onload = () => {
        let body: unknown = null;
        try { body = JSON.parse(xhr.responseText); } catch { /* not JSON */ }
        if (xhr.status >= 300 || body === null) {
          const detail = body && typeof body === "object" && "detail" in body ? (body as { detail: unknown }).detail : xhr.statusText;
          reject(new ApiError(xhr.status, detail));
        } else resolve(body as ScanResult);
      };
      xhr.send(form);
    });
  }
}

export const api = new Api();
