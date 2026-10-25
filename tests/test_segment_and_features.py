import numpy as np
import pytest

from cattle3d.features.geometry_features import (
    animal_features,
    cross_section_volume,
    trunk_interval,
    trunk_profile,
    view_features,
)
from cattle3d.geometry.segment import SegmentationError, segment_view
from cattle3d.types import OrganizedCloud, View


def test_side_view_keeps_the_box_and_drops_floor_and_far_wall(side_scene):
    seg = segment_view(OrganizedCloud(side_scene), View.LEFT)
    height = seg.body_points[:, 2]
    # The visible face of the box spans 0.5-1.3 m above the floor.
    assert height.min() == pytest.approx(0.5, abs=0.03)
    assert height.max() == pytest.approx(1.3, abs=0.03)
    length = np.ptp(seg.body_points[:, 0])
    assert length == pytest.approx(1.6, abs=0.05)
    assert seg.floor_inlier_ratio > 0.2


def test_floor_normal_points_to_the_camera(side_scene):
    seg = segment_view(OrganizedCloud(side_scene), View.RIGHT)
    assert seg.floor.signed_distance(np.zeros((1, 3)))[0] == pytest.approx(1.2, abs=0.02)


def test_top_view_volume_matches_the_box(top_scene):
    seg = segment_view(OrganizedCloud(top_scene), View.TOP)
    assert seg.body_points[:, 2].max() == pytest.approx(1.2, abs=0.02)
    volume = cross_section_volume(seg.body_points)
    # 1.5 x 0.5 x 1.2 = 0.9 m³; percentile trimming of the ends loses ~4 %.
    assert volume == pytest.approx(0.9, rel=0.08)


def test_volume_is_metric_not_normalised(top_scene):
    seg = segment_view(OrganizedCloud(top_scene), View.TOP)
    doubled = seg.body_points * np.array([2.0, 2.0, 2.0])
    assert cross_section_volume(doubled) == pytest.approx(8 * cross_section_volume(seg.body_points), rel=0.02)


def test_volume_tolerates_holes(top_scene):
    seg = segment_view(OrganizedCloud(top_scene), View.TOP)
    rng = np.random.default_rng(0)
    holed = seg.body_points[rng.random(len(seg.body_points)) > 0.3]
    assert cross_section_volume(holed) == pytest.approx(cross_section_volume(seg.body_points), rel=0.03)


def test_empty_scene_fails_loudly():
    with pytest.raises(SegmentationError):
        segment_view(OrganizedCloud(np.zeros((424, 512, 3))), View.TOP)


def test_trunk_interval_excludes_a_narrow_neck():
    rng = np.random.default_rng(0)
    trunk = np.column_stack([rng.uniform(0, 1.4, 20_000), rng.uniform(-0.3, 0.3, 20_000), rng.uniform(1, 1.3, 20_000)])
    neck = np.column_stack([rng.uniform(-0.6, 0, 3_000), rng.uniform(-0.08, 0.08, 3_000), rng.uniform(0.9, 1.1, 3_000)])
    lo, hi = trunk_interval(np.vstack([trunk, neck]))
    assert lo == pytest.approx(0.0, abs=0.06)
    assert hi == pytest.approx(1.4, abs=0.06)


@pytest.mark.parametrize("flip", [1.0, -1.0])
def test_back_profile_starts_at_the_head_end(flip):
    """A wide chest at the neck end must land in slice 0 whichever way PCA points the axis."""
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1.4, 30_000)
    half_width = np.where(x < 0.3, 0.35, 0.25)  # chest wider than the rest of the trunk
    trunk = np.column_stack([x, rng.uniform(-1, 1, x.size) * half_width, rng.uniform(1, 1.3, x.size)])
    neck = np.column_stack([rng.uniform(-0.6, 0, 3_000), rng.uniform(-0.08, 0.08, 3_000), rng.uniform(0.9, 1.1, 3_000)])
    tail = np.column_stack([rng.uniform(1.4, 1.5, 300), rng.uniform(-0.05, 0.05, 300), rng.uniform(1.0, 1.2, 300)])
    body = np.vstack([trunk, neck, tail]) * np.array([flip, 1.0, 1.0])
    lo, hi = trunk_interval(body)
    profile = trunk_profile(body, lo, hi)
    assert profile["profile_width_0"] > profile["profile_width_9"] + 0.15


def test_animal_features_combine_all_views(side_scene, top_scene):
    per_view = {
        View.LEFT: view_features(segment_view(OrganizedCloud(side_scene), View.LEFT)),
        View.RIGHT: view_features(segment_view(OrganizedCloud(side_scene), View.RIGHT)),
        View.TOP: view_features(segment_view(OrganizedCloud(top_scene), View.TOP)),
    }
    feats = animal_features(per_view)
    assert feats["lr_side_area_disagreement"] == pytest.approx(0.0)
    assert feats["top_trunk_length"] == pytest.approx(1.5, abs=0.08)
    assert all(np.isfinite(v) for v in feats.values())
