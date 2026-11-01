// Night sky behind the console: three layers of drifting stars with twinkle and
// an occasional meteor. Drawn only while the tab is visible; with reduced motion
// it is drawn once and stays still.
import { useEffect, useRef } from "react";

const LAYERS = [
  { density: 1 / 5200, size: [0.35, 0.9], speed: 2.5, parallax: 0.04, alpha: 0.55 },
  { density: 1 / 14000, size: [0.7, 1.4], speed: 6, parallax: 0.1, alpha: 0.8 },
  { density: 1 / 52000, size: [1.2, 2.1], speed: 13, parallax: 0.22, alpha: 1 },
];
const TINTS = ["255,255,255", "255,255,255", "205,225,255", "180,205,255", "255,236,210", "255,214,170"];

interface Star { x: number; y: number; li: number; r: number; tint: string; phase: number; rate: number }
interface Meteor { x: number; y: number; vx: number; vy: number; t: number; life: number }

export function StarField() {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    const still = matchMedia("(prefers-reduced-motion: reduce)");
    let w = 0, h = 0, stars: Star[] = [], meteors: Meteor[] = [], last = 0, raf = 0;
    let px = 0, py = 0, tx = 0, ty = 0;

    const draw = (time: number, dt: number) => {
      ctx.clearRect(0, 0, w, h);
      px += (tx - px) * Math.min(1, dt * 2.5);
      py += (ty - py) * Math.min(1, dt * 2.5);
      for (const s of stars) {
        const layer = LAYERS[s.li];
        s.x -= layer.speed * dt; s.y -= layer.speed * 0.35 * dt;
        if (s.x < -4) s.x += w + 8;
        if (s.y < -4) s.y += h + 8;
        const x = (((s.x - px * layer.parallax * 40) % (w + 8)) + w + 8) % (w + 8) - 4;
        const y = (((s.y - py * layer.parallax * 40) % (h + 8)) + h + 8) % (h + 8) - 4;
        ctx.globalAlpha = layer.alpha * (0.65 + 0.35 * Math.sin(s.phase + time * s.rate));
        if (s.r > 1.3) {
          const g = ctx.createRadialGradient(x, y, 0, x, y, s.r * 4);
          g.addColorStop(0, `rgba(${s.tint},0.9)`);
          g.addColorStop(0.25, `rgba(${s.tint},0.25)`);
          g.addColorStop(1, `rgba(${s.tint},0)`);
          ctx.fillStyle = g;
          ctx.fillRect(x - s.r * 4, y - s.r * 4, s.r * 8, s.r * 8);
        } else {
          ctx.fillStyle = `rgb(${s.tint})`;
          ctx.beginPath(); ctx.arc(x, y, s.r, 0, Math.PI * 2); ctx.fill();
        }
      }
      for (const m of meteors) {
        m.t += dt;
        const x = m.x + m.vx * m.t, y = m.y + m.vy * m.t;
        const g = ctx.createLinearGradient(x, y, x - m.vx * 0.18, y - m.vy * 0.18);
        g.addColorStop(0, "rgba(255,255,255,0.95)");
        g.addColorStop(1, "rgba(160,190,255,0)");
        ctx.globalAlpha = Math.sin(Math.PI * Math.min(1, m.t / m.life));
        ctx.strokeStyle = g; ctx.lineWidth = 1.4;
        ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x - m.vx * 0.18, y - m.vy * 0.18); ctx.stroke();
      }
      meteors = meteors.filter((m) => m.t < m.life);
      ctx.globalAlpha = 1;
    };
    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      w = window.innerWidth; h = window.innerHeight;
      canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      stars = LAYERS.flatMap((layer, li) => Array.from({ length: Math.round(w * h * layer.density) }, () => ({
        x: Math.random() * w, y: Math.random() * h, li,
        r: layer.size[0] + Math.random() * (layer.size[1] - layer.size[0]),
        tint: TINTS[(Math.random() * TINTS.length) | 0],
        phase: Math.random() * Math.PI * 2, rate: 0.4 + Math.random() * 1.6,
      })));
      if (still.matches) draw(0, 0);
    };
    const frame = (now: number) => {
      const time = now / 1000;
      const dt = last ? Math.min(0.05, time - last) : 0;
      last = time;
      if (Math.random() < dt * 0.12 && meteors.length < 2) {
        const angle = Math.PI * (0.78 + Math.random() * 0.12), speed = 520 + Math.random() * 380;
        meteors.push({ x: w * (0.3 + Math.random() * 0.8), y: h * Math.random() * 0.45,
          vx: Math.cos(angle) * speed, vy: Math.sin(angle) * speed, t: 0, life: 0.9 + Math.random() * 0.5 });
      }
      draw(time, dt);
      raf = requestAnimationFrame(frame);
    };
    const start = () => {
      cancelAnimationFrame(raf);
      last = 0;
      if (still.matches || document.hidden) draw(0, 0);
      else raf = requestAnimationFrame(frame);
    };
    const move = (e: PointerEvent) => { tx = (e.clientX / Math.max(1, w) - 0.5) * 2; ty = (e.clientY / Math.max(1, h) - 0.5) * 2; };
    const vis = () => (document.hidden ? cancelAnimationFrame(raf) : start());
    window.addEventListener("resize", resize);
    window.addEventListener("pointermove", move, { passive: true });
    document.addEventListener("visibilitychange", vis);
    still.addEventListener("change", start);
    resize();
    start();
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
      window.removeEventListener("pointermove", move);
      document.removeEventListener("visibilitychange", vis);
      still.removeEventListener("change", start);
    };
  }, []);
  return <canvas ref={ref} className="space-bg" aria-hidden="true" />;
}
