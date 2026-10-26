// Rotatable 3-D view of the animal: the merged body points and the trunk slices
// whose areas add up to the volume. Canvas 2-D with the painter's algorithm, so
// it needs no WebGL and runs on any phone. Drag rotates, wheel or pinch zooms,
// double click resumes the slow spin.
import { useEffect, useRef, useState } from "react";
import type { Preview, Slice, View } from "../api/types";

export interface Cloud {
  /** x along the body, y lateral, z height above the floor; metres */
  points: Float32Array;
  /** index into viewNames for every point, or null when only one view is known */
  viewOf: Uint8Array | null;
  viewNames: View[];
  slices: Slice[];
}

const bytes = (b64: string) => Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));

export function cloudFromPreview(p: Preview): Cloud {
  const i16 = new Int16Array(bytes(p.points_mm).buffer);
  const points = new Float32Array(i16.length);
  for (let i = 0; i < i16.length; i++) points[i] = i16[i] / 1000;
  return { points, viewOf: p.views_merged ? bytes(p.point_view) : null, viewNames: p.view_order, slices: p.slices };
}

/** img/cow70_points.bin: int16 x, y, z in mm, uint8 view, uint8 pad per point. */
export function cloudFromBin(buffer: ArrayBuffer, keepEvery = 1): Cloud {
  const d = new DataView(buffer);
  const n = Math.floor(buffer.byteLength / 8 / keepEvery);
  const points = new Float32Array(n * 3), viewOf = new Uint8Array(n);
  for (let k = 0; k < n; k++) {
    const o = k * keepEvery * 8;
    points[3 * k] = d.getInt16(o, true) / 1000;
    points[3 * k + 1] = d.getInt16(o + 2, true) / 1000;
    points[3 * k + 2] = d.getInt16(o + 4, true) / 1000;
    viewOf[k] = d.getUint8(o + 6);
  }
  return { points, viewOf, viewNames: ["left", "right", "top"], slices: [] };
}

const CAMERA_RGB: Record<View, [number, number, number]> = { left: [90, 150, 255], right: [70, 200, 130], top: [235, 150, 60] };

interface Props {
  cloud: Cloud;
  height?: number;
  controls?: boolean;
  label?: string;
}

export function CloudViewer({ cloud, height = 340, controls = true, label = "Тривимірна модель тварини" }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  const [showPoints, setPoints] = useState(true);
  const [byCamera, setByCamera] = useState(false);
  const [showSlices, setSlices] = useState(false);
  const opts = useRef({ showPoints, byCamera, showSlices });
  opts.current = { showPoints, byCamera, showSlices };
  const kick = useRef<() => void>(() => {});

  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    const pts = cloud.points;
    const n = pts.length / 3;
    const order = new Uint32Array(n).map((_, i) => i);
    let zmax = 0;
    for (let i = 2; i < pts.length; i += 3) zmax = Math.max(zmax, pts[i]);
    let yaw = -0.6, pitch = 0.42, dist = 4.2, auto = !matchMedia("(prefers-reduced-motion: reduce)").matches;
    let raf = 0, last = 0;
    const sx = new Float32Array(n), sy = new Float32Array(n), sd = new Float32Array(n), ss = new Float32Array(n);

    const colour = (i: number, shade: number) => {
      let r: number, g: number, b: number;
      if (opts.current.byCamera && cloud.viewOf) [r, g, b] = CAMERA_RGB[cloud.viewNames[cloud.viewOf[i]]] ?? [200, 200, 200];
      else { // violet at the belly, cyan, white along the back
        const t = Math.min(1, Math.max(0, pts[3 * i + 2] / zmax));
        if (t < 0.55) { const k = t / 0.55; r = 82 - 18 * k; g = 56 + 161 * k; b = 242; }
        else { const k = (t - 0.55) / 0.45; r = 64 + 191 * k; g = 217 + 30 * k; b = 242 - 7 * k; }
      }
      return `rgb(${(r * shade) | 0},${(g * shade) | 0},${(b * shade) | 0})`;
    };

    const draw = (ts: number) => {
      raf = 0;
      if (auto) yaw += Math.min(50, ts - (last || ts)) * 0.00035;
      last = ts;
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      const w = canvas.clientWidth, h = canvas.clientHeight;
      if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
        canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      const cy = Math.cos(yaw), syw = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
      const f = Math.min(w, h) * 1.45, zc = zmax * 0.5;
      const project = (x: number, y: number, z: number): [number, number, number, number] => {
        const xr = x * cy - y * syw, yr = x * syw + y * cy, zr = z - zc;
        const depth = dist + yr * cp - zr * sp;
        const s = f / depth;
        return [w / 2 + xr * s, h / 2 - (zr * cp + yr * sp) * s, depth, s];
      };
      ctx.strokeStyle = "rgba(160,190,195,0.12)"; ctx.lineWidth = 1;
      for (let g = -1.4; g <= 1.41; g += 0.35) {
        let a = project(g, -1, 0), b = project(g, 1, 0);
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
        a = project(-1.4, g * 0.72, 0); b = project(1.4, g * 0.72, 0);
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
      }
      for (let i = 0; i < n; i++) {
        const q = project(pts[3 * i], pts[3 * i + 1], pts[3 * i + 2]);
        sx[i] = q[0]; sy[i] = q[1]; sd[i] = q[2]; ss[i] = q[3];
      }
      const ring = (sl: Slice) => {
        ctx.beginPath();
        sl.outline.forEach(([y, z], j) => { const q = project(sl.x, y, z); if (j) ctx.lineTo(q[0], q[1]); else ctx.moveTo(q[0], q[1]); });
        ctx.closePath();
      };
      const { showPoints: pOn, showSlices: sOn } = opts.current;
      if (sOn) {
        ctx.fillStyle = "rgba(124,198,224,0.045)"; ctx.strokeStyle = "rgba(124,198,224,0.2)";
        for (const sl of cloud.slices) { ring(sl); ctx.fill(); ctx.stroke(); }
      }
      if (pOn) {
        order.sort((a, b) => sd[b] - sd[a]);
        const dmin = dist - 1.2;
        for (let k = 0; k < n; k++) {
          const i = order[k];
          ctx.fillStyle = colour(i, 1.05 - 0.55 * Math.min(1, Math.max(0, (sd[i] - dmin) / 2.4)));
          const r = Math.max(1.5, ss[i] * 0.011);
          ctx.fillRect(sx[i] - r / 2, sy[i] - r / 2, r, r);
        }
      }
      if (sOn) {
        ctx.lineWidth = 1.6; ctx.strokeStyle = "rgba(255,190,90,0.9)";
        cloud.slices.forEach((sl, k) => { if (k % 8 === 4) { ring(sl); ctx.stroke(); } });
      }
      if (auto) start();
    };
    const start = () => { if (!raf) raf = requestAnimationFrame(draw); };
    kick.current = start;

    const touches = new Map<number, [number, number]>();
    let pinch0 = 0, dist0 = 0;
    const down = (e: PointerEvent) => { canvas.setPointerCapture(e.pointerId); touches.set(e.pointerId, [e.clientX, e.clientY]); auto = false; };
    const move = (e: PointerEvent) => {
      const prev = touches.get(e.pointerId);
      if (!prev) return;
      touches.set(e.pointerId, [e.clientX, e.clientY]);
      if (touches.size === 2) {
        const [a, b] = [...touches.values()];
        const d = Math.hypot(a[0] - b[0], a[1] - b[1]);
        if (!pinch0) { pinch0 = d; dist0 = dist; }
        dist = Math.min(9, Math.max(2.2, (dist0 * pinch0) / d));
      } else {
        yaw += (e.clientX - prev[0]) * 0.01;
        pitch = Math.min(1.45, Math.max(-0.2, pitch + (e.clientY - prev[1]) * 0.008));
      }
      start();
    };
    const up = (e: PointerEvent) => { touches.delete(e.pointerId); if (touches.size < 2) pinch0 = 0; };
    const wheel = (e: WheelEvent) => { e.preventDefault(); dist = Math.min(9, Math.max(2.2, dist * (1 + e.deltaY * 0.001))); start(); };
    const dbl = () => { auto = true; start(); };
    canvas.addEventListener("pointerdown", down);
    canvas.addEventListener("pointermove", move);
    canvas.addEventListener("pointerup", up);
    canvas.addEventListener("pointercancel", up);
    canvas.addEventListener("wheel", wheel, { passive: false });
    canvas.addEventListener("dblclick", dbl);
    window.addEventListener("resize", start);
    start();
    return () => {
      cancelAnimationFrame(raf);
      canvas.removeEventListener("pointerdown", down);
      canvas.removeEventListener("pointermove", move);
      canvas.removeEventListener("pointerup", up);
      canvas.removeEventListener("pointercancel", up);
      canvas.removeEventListener("wheel", wheel);
      canvas.removeEventListener("dblclick", dbl);
      window.removeEventListener("resize", start);
    };
  }, [cloud]);

  useEffect(() => kick.current(), [showPoints, byCamera, showSlices]);

  return (
    <div className="viewer">
      <canvas ref={ref} className="viewer-canvas" style={{ height }} role="img" aria-label={label} />
      {controls && (
        <div className="toggles" role="group" aria-label="Що показати">
          <button type="button" aria-pressed={showPoints} onClick={() => setPoints((v) => !v)}>Точки</button>
          {cloud.viewOf && <button type="button" aria-pressed={byCamera} onClick={() => setByCamera((v) => !v)}>За камерами</button>}
          {cloud.slices.length > 0 && <button type="button" aria-pressed={showSlices} onClick={() => setSlices((v) => !v)}>Зрізи об’єму</button>}
        </div>
      )}
    </div>
  );
}
