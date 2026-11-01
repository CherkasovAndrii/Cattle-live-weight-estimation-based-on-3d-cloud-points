import { describe, expect, it } from "vitest";
import { ApiError } from "../api/client";
import type { Scan } from "../api/types";
import { cloudFromBin } from "../components/CloudViewer";
import { animalsFrom } from "../pages/Animals";
import { assignFiles } from "../pages/Weigh";
import { errorUk, plural, reasonUk } from "../lib/text";

const scan = (id: number, tag: string, day: number, w: number | null, extra: Partial<Scan> = {}): Scan => ({
  id, herd_id: 1, animal_tag: tag, created_utc: new Date(Date.UTC(2026, 9, day, 8)).toISOString(), status: "trusted",
  reasons: [], weight_kg: w, lower_kg: w == null ? null : w - 40, upper_kg: w == null ? null : w + 40, raw_weight_kg: w,
  calibration_id: null, model_version: "0.1.0", scale_weight_kg: null, weighed_utc: null, ...extra,
});

describe("Ukrainian wording", () => {
  it("translates server reasons into actions", () => {
    expect(reasonUk("depth noise in top view 1.2 % of range")).toContain("Верхня камера: шум глибини 1,2 %");
    expect(reasonUk("rescan: no surface large enough to be an animal")).toBe("пересканувати: в одному з кадрів не знайдено тварину");
    expect(reasonUk("something new")).toBe("something new");
  });
  it("explains errors", () => {
    expect(errorUk(new ApiError(422, "scale weight 5 kg is outside 30-1500 kg"))).toBe("Маса на вагах 5 кг поза межами 30-1500 кг.");
    expect(errorUk(new ApiError(422, "at least 3 reference weighings are needed to calibrate a herd, got 1"))).toContain("щонайменше 3");
    expect(errorUk(new ApiError(0, "network"))).toBe("Немає зв’язку з сервером.");
  });
  it("chooses the plural form", () => {
    expect([1, 2, 5, 11, 21, 24, 112].map((n) => plural(n, "a", "b", "c"))).toEqual(["a", "b", "c", "c", "a", "b", "c"]);
  });
});

describe("choosing clouds", () => {
  it("puts files named after a view into its slot and guesses the rest", () => {
    const f = (name: string) => new File(["x"], name);
    const named = assignFiles([f("cow_top.ply"), f("cow_LEFT.ply"), f("cow_right.ply")]);
    expect(named.guessed).toBe(false);
    expect(named.picked.top?.label).toBe("cow_top.ply");
    const anon = assignFiles([f("cloud.ply"), f("cloud (1).ply"), f("cloud (2).ply")]);
    expect(anon.guessed).toBe(true);
    expect(anon.picked.left?.label).toBe("cloud.ply");
  });
});

describe("animals from scans", () => {
  it("keeps the newest scan and fits the daily gain", () => {
    const rows = animalsFrom([scan(1, "A", 1, 500), scan(2, "A", 11, 510), scan(3, "B", 5, null, { status: "refused" })]);
    expect(rows.map((r) => r.tag)).toEqual(["A", "B"]);
    expect(rows[0].lastWeight).toBe(510);
    expect(rows[0].gainKgPerDay).toBeCloseTo(1, 6);
    expect(rows[1].gainKgPerDay).toBeNull();
  });
});

describe("point cloud file", () => {
  it("decodes int16 millimetres and the camera of each point", () => {
    const buf = new ArrayBuffer(16), d = new DataView(buf);
    d.setInt16(0, 1500, true); d.setInt16(2, -200, true); d.setInt16(4, 1300, true); d.setUint8(6, 2);
    d.setInt16(8, -10, true); d.setUint8(14, 1);
    const c = cloudFromBin(buf);
    expect(Array.from(c.points.slice(0, 3))).toEqual([1.5, -0.2, 1.3].map((v) => Math.fround(v)));
    expect(Array.from(c.viewOf!)).toEqual([2, 1]);
  });
});
