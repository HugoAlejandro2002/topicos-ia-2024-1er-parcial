import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_model_info():
    """Test the /model_info endpoint."""
    response = client.get("/model_info")
    assert response.status_code == 200
    assert "model_name" in response.json()
    assert "gun_detector_model" in response.json()
    assert "semantic_segmentation_model" in response.json()
    assert "input_type" in response.json()

def test_detect_guns(image_file):
    """Test the /detect_guns endpoint."""
    response = client.post("/detect_guns", files={"file": ("test.jpeg", image_file, "image/jpeg")})
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["pred_type"] == "OD"
    assert "n_detections" in json_response
    assert isinstance(json_response["n_detections"], int)
    assert "boxes" in json_response
    assert isinstance(json_response["boxes"], list)
    assert "labels" in json_response
    assert isinstance(json_response["labels"], list)
    assert "confidences" in json_response
    assert isinstance(json_response["confidences"], list)

def test_detect_guns_validation_error():
    """Test /detect_guns endpoint for validation error."""
    response = client.post("/detect_guns", files={"file": ("not_an_image.txt", b"Not an image", "text/plain")})
    assert response.status_code == 415
    assert response.json() == {"detail": "Not an image"}

def test_annotate_guns(image_file):
    """Test the /annotate_guns endpoint."""
    response = client.post("/annotate_guns", files={"file": ("test.jpeg", image_file, "image/jpeg")})
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"

def test_detect_people(image_file):
    """Test the /detect_people endpoint."""
    response = client.post("/detect_people", files={"file": ("test.jpeg", image_file, "image/jpeg")})
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["pred_type"] == "SEG"
    assert "n_detections" in json_response
    assert isinstance(json_response["n_detections"], int)
    assert "polygons" in json_response
    assert isinstance(json_response["polygons"], list)
    assert "boxes" in json_response
    assert isinstance(json_response["boxes"], list)
    assert "labels" in json_response
    assert isinstance(json_response["labels"], list)

def test_detect_people_validation_error():
    """Test /detect_people endpoint for validation error."""
    response = client.post("/detect_people", files={"file": ("not_an_image.txt", b"Not an image", "text/plain")})
    assert response.status_code == 415
    assert response.json() == {"detail": "Not an image"}

def test_annotate_people(image_file):
    """Test the /annotate_people endpoint."""
    response = client.post("/annotate_people", files={"file": ("test.jpeg", image_file, "image/jpeg")})
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"

def test_detect_endpoint(image_file):
    """Test the /detect endpoint that combines gun detection and people segmentation."""
    response = client.post("/detect", files={"file": ("test.jpeg", image_file, "image/jpeg")})
    assert response.status_code == 200
    json_response = response.json()
    assert "detection" in json_response, "La respuesta debería contener la detección de armas"
    assert "segmentation" in json_response, "La respuesta debería contener la segmentación de personas"
    assert isinstance(json_response["detection"], dict), "La detección debe ser un objeto de tipo dict"
    assert isinstance(json_response["segmentation"], dict), "La segmentación debe ser un objeto de tipo dict"

def test_detect_validation_error():
    """Test /detect endpoint for validation error."""
    response = client.post("/detect", files={"file": ("not_an_image.txt", b"Not an image", "text/plain")})
    assert response.status_code == 415
    assert response.json() == {"detail": "Not an image"}

def test_annotate_combined(image_file):
    """Test the /annotate endpoint that combines gun and people annotations."""
    response = client.post("/annotate", files={"file": ("test.jpeg", image_file, "image/jpeg")})
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg", "La respuesta debería ser una imagen JPEG anotada"

def test_annotate_combined_validation_error():
    """Test /annotate endpoint for validation error."""
    response = client.post("/annotate", files={"file": ("not_an_image.txt", b"Not an image", "text/plain")})
    assert response.status_code == 415
    assert response.json() == {"detail": "Not an image"}

def test_guns_info(image_file_gun6):
    """Test the /guns endpoint that returns detected gun information."""
    response = client.post("/guns", files={"file": ("gun6.jpg", image_file_gun6, "image/jpeg")})
    assert response.status_code == 200
    json_response = response.json()
    assert isinstance(json_response, list)
    for gun in json_response:
        assert "gun_type" in gun
        assert "location" in gun
        assert isinstance(gun["location"], dict)
        assert "x" in gun["location"]
        assert isinstance(gun["location"]["x"], int)
        assert "y" in gun["location"]
        assert isinstance(gun["location"]["y"], int)

def test_people_info(image_file_gun6):
    """Test the /people endpoint that returns detected people information."""
    response = client.post("/people", files={"file": ("gun6.jpg", image_file_gun6, "image/jpeg")})
    assert response.status_code == 200
    json_response = response.json()
    assert isinstance(json_response, list)
    for person in json_response:
        assert "person_type" in person
        assert "location" in person
        assert isinstance(person["location"], dict)
        assert "x" in person["location"]
        assert isinstance(person["location"]["x"], int)
        assert "y" in person["location"]
        assert isinstance(person["location"]["y"], int)
        assert "area" in person
        assert isinstance(person["area"], int)

def test_people_info_validation_error():
    """Test /people endpoint for validation error."""
    response = client.post("/people", files={"file": ("not_an_image.txt", b"Not an image", "text/plain")})
    assert response.status_code == 415
    assert response.json() == {"detail": "Not an image"}