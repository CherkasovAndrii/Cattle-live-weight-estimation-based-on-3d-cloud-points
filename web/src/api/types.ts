// Shapes returned by the cattle3d REST API (docs/API.md, src/cattle3d/api/app.py).

export type View = "left" | "right" | "top";
export const VIEWS: View[] = ["left", "right", "top"];
export type ScanStatus = "trusted" | "flagged" | "refused";

export interface Herd {
  id: number;
  name: string;
  created_utc: string;
}

export interface Calibration {
  id: number;
  created_utc: string;
  slope: number;
  intercept_kg: number;
  n_reference: number;
  interval_half_width_kg: number;
}

export interface Drift {
  n_weighings: number;
  mean_residual_kg: number | null;
  ci_low_kg: number | null;
  ci_high_kg: number | null;
  drifted: boolean;
  message: string;
}

export interface HerdStatus {
  herd: Herd;
  calibrated: boolean;
  calibration: Calibration | null;
  n_reference_animals: number;
  recommended_reference_animals: number;
  scans_by_status: Partial<Record<ScanStatus, number>>;
  drift: Drift | null;
}

export interface Scan {
  id: number;
  herd_id: number;
  animal_tag: string;
  created_utc: string;
  status: ScanStatus;
  reasons: string[];
  weight_kg: number | null;
  lower_kg: number | null;
  upper_kg: number | null;
  raw_weight_kg: number | null;
  calibration_id: number | null;
  model_version: string;
  scale_weight_kg: number | null;
  weighed_utc: string | null;
}

export interface Slice {
  x: number;
  outline: [number, number][];
}

export interface Preview {
  views: Record<View, { image: string; animal_points: number }>;
  points_mm: string;
  point_view: string;
  view_order: View[];
  n_points: number;
  views_merged: boolean;
  view_agreement: { same_animal: boolean; footprint_share: number; back_gap_m: number; shift_m: number } | null;
  trunk_m: [number, number];
  slices: Slice[];
  measures?: { trunk_volume_m3: number; trunk_length_m: number; trunk_width_m: number; height_m: number };
  timing_ms?: { total_ms: number; segmentation_and_features_ms: number };
}

export interface ScanResult extends Scan {
  preview: Preview | null;
}

export interface ModelCard {
  package_version: string;
  created_utc: string;
  training_herds: string[];
  n_training_animals: number;
  weight_range_kg: [number, number];
  trunk_volume_range_m3: [number, number];
  interval: string;
  cross_validated: Record<string, number>;
}

export interface DemoSample {
  id: string;
  herd_id: number;
  herd: string;
  animal_id: string;
  breed: string;
  scale_weight_kg: number;
  rig: unknown;
}

export interface DemoInfo {
  herds: { id: number; key: string; farm: string; breed: string; dataset: string }[];
  samples: DemoSample[];
  phone_url: string | null;
}
