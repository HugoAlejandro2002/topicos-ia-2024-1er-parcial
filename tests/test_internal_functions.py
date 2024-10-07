import pytest
import numpy as np
from src.predictor import GunDetector, match_gun_bbox, annotate_segmentation
from src.models import Detection, Segmentation

@pytest.fixture
def gun_detector():
    return GunDetector()

def test_match_gun_bbox():
    segment = [100, 150, 200, 250]
    bboxes = [[120, 140, 180, 240], [300, 350, 400, 450]]
    result = match_gun_bbox(segment, bboxes, max_distance=20)
    assert result == [120, 140, 180, 240]

def test_annotate_segmentation():
    img_array = np.zeros((500, 500, 3), dtype=np.uint8)
    segmentation = Segmentation(
        pred_type="SEG",
        n_detections=1,
        polygons=[[[100, 100], [150, 100], [150, 150], [100, 150]]],
        boxes=[[100, 100, 150, 150]],
        labels=["safe"]
    )
    result_img = annotate_segmentation(img_array, segmentation)
    assert isinstance(result_img, np.ndarray)
    assert result_img.shape == img_array.shape

def test_detect_guns(gun_detector):
    img_array = np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8)
    detection = gun_detector.detect_guns(img_array, threshold=0.5)
    assert isinstance(detection, Detection)
    assert hasattr(detection, 'n_detections')
    assert hasattr(detection, 'boxes')
    assert hasattr(detection, 'labels')

def test_segment_people(gun_detector):
    img_array = np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8)
    segmentation = gun_detector.segment_people(img_array, threshold=0.5)
    assert isinstance(segmentation, Segmentation)
    assert hasattr(segmentation, 'n_detections')
    assert hasattr(segmentation, 'polygons')
    assert hasattr(segmentation, 'labels')
