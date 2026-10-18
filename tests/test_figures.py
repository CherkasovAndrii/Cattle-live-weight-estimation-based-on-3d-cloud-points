import numpy as np
import pytest

from cattle3d.figures import schemes, stages
from cattle3d.figures.draw import Lang
from cattle3d.geometry.segment import segment_view, trace_view
from cattle3d.types import AnimalRecord, OrganizedCloud, View
from tests.conftest import write_pcl_ply


def test_trace_is_the_same_computation_as_segment_view(side_scene):
    cloud = OrganizedCloud(side_scene)
    trace = trace_view(cloud, View.LEFT)
    plain = segment_view(cloud, View.LEFT)
    np.testing.assert_allclose(trace.result.body_points, plain.body_points)
    assert (trace.picked <= trace.animal).all()  # merging only adds pixels
    assert (trace.animal <= trace.above).all()
    assert (trace.above <= trace.valid).all()


@pytest.mark.parametrize("code", ["uk", "en"])
def test_schemes_render_in_both_languages(tmp_path, code):
    t = Lang(code)
    paths = [
        schemes.pipeline_scheme(tmp_path / "pipeline.png", t),
        schemes.evaluation_scheme(tmp_path / "evaluation.png", t),
        schemes.decision_scheme(tmp_path / "decision.png", t, 46.7),
        schemes.integration_scheme(tmp_path / "integration.png", t),
        schemes.calibration_sequence(tmp_path / "sequence.png", t),
    ]
    assert all(p.stat().st_size > 10_000 for p in paths)


def test_stage_figures_render_from_a_synthetic_animal(tmp_path, side_scene, top_scene):
    paths = {}
    for view, scene in ((View.LEFT, side_scene), (View.RIGHT, side_scene), (View.TOP, top_scene)):
        paths[view] = tmp_path / f"{view.value}.ply"
        write_pcl_ply(paths[view], scene.reshape(-1, 3))
    record = AnimalRecord("test", "1", 500.0, paths, {})
    t = Lang("uk")
    out = [
        stages.raw_views(record, tmp_path / "raw.png", t),
        stages.segmentation_steps(record, tmp_path / "seg.png", t),
        stages.body_frame(record, tmp_path / "body.png", t),
        stages.cross_sections(record, tmp_path / "cs.png", t),
    ]
    assert all(p.stat().st_size > 10_000 for p in out)


def test_unknown_language_is_refused():
    with pytest.raises(ValueError):
        Lang("de")
