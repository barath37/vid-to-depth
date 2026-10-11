import numpy as np
import pytest

from src.common.types import (
    TelemetrySample,
    KeyframeData,
    CameraPose,
    PointCloudModel,
    ConfidenceTag
)


def test_telemetry_sample_creation():
    sample = TelemetrySample(
        timestamp=10.5,
        latitude=11.0168,
        longitude=76.9558,
        altitude_msl=412.5,
        altitude_rel=100.0,
        pitch=0.0,
        roll=0.0,
        yaw=90.0,
        speed=6.0
    )
    assert sample.timestamp == 10.5
    assert sample.latitude == 11.0168
    assert sample.speed == 6.0


def test_keyframe_data_structure():
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    keyframe = KeyframeData(
        frame_idx=10,
        timestamp=2.5,
        image=dummy_img,
        laplacian_var=120.5,
        is_i_frame=True
    )
    assert keyframe.frame_idx == 10
    assert keyframe.laplacian_var > 80.0
    assert keyframe.is_i_frame is True


def test_camera_pose_properties():
    R = np.eye(3)
    t = np.array([10.0, 20.0, 30.0])
    pose = CameraPose(frame_idx=1, R=R, t=t, timestamp=1.0)
    # Camera center c = -R.T @ t
    assert np.allclose(pose.center, -t)


def test_point_cloud_model():
    points = np.zeros((100, 3), dtype=np.float32)
    colors = np.ones((100, 3), dtype=np.uint8) * 255
    confidence = np.ones(100, dtype=np.float32) * 0.85
    tags = [ConfidenceTag.GREEN] * 100

    cloud = PointCloudModel(
        points=points,
        colors=colors,
        confidence=confidence,
        tags=tags
    )
    assert len(cloud.points) == 100
    assert cloud.tags[0] == ConfidenceTag.GREEN
