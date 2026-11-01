import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { Scan } from "../api/types";
import { WeightHistory } from "../components/charts";
import { Pill } from "../components/ui";

describe("components", () => {
  it("labels the status in Ukrainian", () => {
    render(<Pill status="flagged" />);
    expect(screen.getByText("Позначено").className).toContain("flagged");
  });
  it("draws one point per estimate and one diamond per scale weight", () => {
    const base = { herd_id: 1, animal_tag: "A", status: "trusted" as const, reasons: [] as string[], raw_weight_kg: null, calibration_id: null, model_version: "x", weighed_utc: null };
    const scans: Scan[] = [
      { ...base, id: 1, created_utc: "2026-10-01T08:00:00Z", weight_kg: 600, lower_kg: 560, upper_kg: 640, scale_weight_kg: 610 },
      { ...base, id: 2, created_utc: "2026-10-08T08:00:00Z", weight_kg: 612, lower_kg: 572, upper_kg: 652, scale_weight_kg: null },
    ];
    const { container } = render(<WeightHistory scans={scans} />);
    expect(container.querySelectorAll("circle.dot")).toHaveLength(2);
    expect(container.querySelectorAll("rect.scale")).toHaveLength(1);
    expect(container.querySelector("path.band")).not.toBeNull();
  });
});
