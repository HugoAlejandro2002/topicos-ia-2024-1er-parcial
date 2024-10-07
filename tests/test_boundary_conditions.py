import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

@pytest.mark.parametrize("threshold, expected_detections, expected_danger, expected_safe", [
    (0.3, 4, 3, 1),  # Threshold 0.3: 4 detections, 3 danger, 1 safe
    (0.5, 2, 1, 1),  # Threshold 0.5: 2 detections, 1 danger, 1 safe
    (0.7, 1, 0, 1)   # Threshold 0.7: 1 detection, 0 danger, 1 safe
])
def test_boundary_detection(image_file_gun6, threshold, expected_detections, expected_danger, expected_safe):
    response = client.post("/detect_people", params={"threshold": threshold}, files={"file": ("gun6.jpg", image_file_gun6, "image/jpeg")})
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["n_detections"] == expected_detections
    assert json_response["labels"].count("danger") == expected_danger
    assert json_response["labels"].count("safe") == expected_safe
