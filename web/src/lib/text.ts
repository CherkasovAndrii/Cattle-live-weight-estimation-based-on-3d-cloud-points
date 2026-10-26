// Ukrainian wording for the operator: numbers, statuses, and the server's
// English reasons and errors turned into what to do next.
import { ApiError } from "../api/client";
import type { ScanStatus, View } from "../api/types";

export const VIEW_UK: Record<View, string> = { left: "Ліва камера", right: "Права камера", top: "Верхня камера" };
export const STATUS_UK: Record<ScanStatus, string> = { trusted: "Довірено", flagged: "Позначено", refused: "Відмова" };

export function fmt(x: number | null | undefined, digits = 0): string {
  if (x == null || Number.isNaN(x)) return "—";
  return Number(x).toLocaleString("uk-UA", { maximumFractionDigits: digits, minimumFractionDigits: digits });
}
export const mb = (bytes: number) => `${fmt(bytes / 1048576, 1)} МБ`;
export const when = (iso: string) =>
  new Date(iso).toLocaleString("uk-UA", { dateStyle: "short", timeStyle: "short" });
export const day = (iso: string) => new Date(iso).toLocaleDateString("uk-UA", { day: "2-digit", month: "2-digit" });

type Rule = [RegExp, string | ((...m: string[]) => string)];
const lower = (v: string) => VIEW_UK[v as View].toLowerCase();
const REASONS: Rule[] = [
  [/^herd not calibrated.*/, "стадо не відкалібровано: зважте 20 тварин на вагах і відкалібруйте"],
  [/^the three views do not seem to show the same animal.*/, "три хмари, схоже, від різних тварин: вони не сходяться за калібруванням станції; проскануйте тварину ще раз"],
  [/^depth noise in (left|right|top) view ([\d.]+) % of range.*$/, (_m, v, x) => `${VIEW_UK[v as View]}: шум глибини ${x.replace(".", ",")} % відстані, більше за допустимі 0,75 %; перевірте камеру`],
  [/^many scattered missing pixels in (left|right|top) view.*$/, (_m, v) => `${VIEW_UK[v as View]}: багато розсіяних пікселів без глибини; протріть скло камери`],
  [/^body volume outside the training range$/, "об’єм тулуба поза діапазоном, на якому навчено модель"],
  [/^too few points in (left|right|top) view$/, (_m, v) => `замало точок у кадрі: ${lower(v)}`],
  [/^rescan: the three views contradict each other.*$/, "хмари суперечать одна одній: файл не в тому слоті або хмари від різних тварин"],
  [/^rescan: (left|right|top): too few depth returns in range$/, (_m, v) => `пересканувати: ${lower(v)} майже нічого не бачить у робочій зоні`],
  [/^rescan: no surface large enough to be an animal$/, "пересканувати: в одному з кадрів не знайдено тварину"],
  [/^rescan: (not enough points to fit a floor|no floor plane within the allowed tilt)$/, "пересканувати: в одному з кадрів не знайдено підлогу; можливо, файл не в тому слоті"],
  [/^rescan: (.*)$/, "пересканувати: $1"],
];

export function reasonUk(reason: string): string {
  for (const [re, uk] of REASONS) {
    const m = reason.match(re);
    if (m) return typeof uk === "string" ? reason.replace(re, uk) : uk(...m);
  }
  return reason;
}

const FIELD_UK: Record<string, string> = {
  animal_tag: "номер бирки", scale_weight_kg: "маса на вагах", left: "ліва хмара", right: "права хмара",
  top: "верхня хмара", rig: "калібрування", name: "назва", weight_kg: "маса",
};

export function errorUk(e: unknown): string {
  if (!(e instanceof ApiError)) return e instanceof Error ? e.message : String(e);
  if (e.status === 0) return "Немає зв’язку з сервером.";
  if (e.status === 413) return "Файл завеликий: хмара Kinect v2 важить близько 3,5 МБ, сервер приймає до 16 МБ.";
  const d = e.detail;
  if (Array.isArray(d)) {
    return d.map((x: { loc: string[]; msg: string }) => `Перевірте поле «${FIELD_UK[x.loc[x.loc.length - 1]] ?? x.loc.join(".")}»: ${x.msg}`).join("; ");
  }
  if (typeof d === "string") {
    const cloud = d.match(/^(left|right|top) cloud: (.*)$/);
    if (cloud) return `${VIEW_UK[cloud[1] as View]}: файл не схожий на хмару точок Kinect (${cloud[2]}). Потрібен PLY-кадр 512×424.`;
    const w = d.match(/^scale weight (.*) kg is outside (.*) kg$/);
    if (w) return `Маса на вагах ${w[1]} кг поза межами ${w[2]} кг.`;
    const few = d.match(/^at least (\d+) reference weighings are needed to calibrate a herd, got (\d+)$/);
    if (few) return `Для калібрування потрібно щонайменше ${few[1]} контрольних зважувань, зараз ${few[2]}.`;
    return d;
  }
  return `Помилка сервера ${e.status}`;
}

/** Ukrainian plural: 1 тварина, 2 тварини, 5 тварин. */
export function plural(n: number, one: string, few: string, many: string): string {
  const a = Math.abs(n) % 100, b = a % 10;
  if (a > 10 && a < 20) return many;
  if (b === 1) return one;
  if (b >= 2 && b <= 4) return few;
  return many;
}
