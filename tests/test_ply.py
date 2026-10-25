import numpy as np
import pytest

from cattle3d.io.ply import PlyFormatError, read_organized_cloud, read_points
from cattle3d.types import DEPTH_HEIGHT, DEPTH_WIDTH
from tests.conftest import write_pcl_ply


def test_round_trip_keeps_points_and_layout(tmp_path):
    xyz = np.random.default_rng(0).normal(size=(DEPTH_HEIGHT * DEPTH_WIDTH, 3)).astype(np.float32)
    path = tmp_path / "cloud.ply"
    write_pcl_ply(path, xyz)
    np.testing.assert_allclose(read_points(path), xyz, rtol=0, atol=0)
    cloud = read_organized_cloud(path)
    assert cloud.xyz.shape == (DEPTH_HEIGHT, DEPTH_WIDTH, 3)
    np.testing.assert_allclose(cloud.xyz[1, 0], xyz[DEPTH_WIDTH])


def test_rejects_a_cloud_that_is_not_a_full_depth_frame(tmp_path):
    path = tmp_path / "cloud.ply"
    write_pcl_ply(path, np.zeros((10, 3)))
    with pytest.raises(PlyFormatError, match="dense"):
        read_organized_cloud(path)


def test_rejects_ascii_ply(tmp_path):
    path = tmp_path / "a.ply"
    path.write_text("ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\nend_header\n")
    with pytest.raises(PlyFormatError, match="binary_little_endian"):
        read_points(path)


def test_rejects_truncated_file(tmp_path):
    path = tmp_path / "cloud.ply"
    write_pcl_ply(path, np.zeros((100, 3)))
    path.write_bytes(path.read_bytes()[:-1000])
    with pytest.raises(PlyFormatError, match="shorter"):
        read_points(path)
