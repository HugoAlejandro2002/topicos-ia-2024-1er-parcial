import io
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Depends
from fastapi.responses import Response
import numpy as np
from functools import cache
from PIL import Image, UnidentifiedImageError
from src.models import Gun, GunType, Person, PersonType, PixelLocation
from src.predictor import GunDetector, Detection, Segmentation, annotate_detection, annotate_segmentation
from src.config import get_settings

SETTINGS = get_settings()

app = FastAPI(title=SETTINGS.api_name, version=SETTINGS.revision)


@cache
def get_gun_detector() -> GunDetector:
    print("Creating model...")
    return GunDetector()


def detect_uploadfile(detector: GunDetector, file, threshold) -> tuple[Detection, np.ndarray]:
    img_stream = io.BytesIO(file.file.read())
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Not an image"
        )
    # convertir a una imagen de Pillow
    try:
        img_obj = Image.open(img_stream)
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Image format not suported"
        )
    # crear array de numpy
    img_array = np.array(img_obj)
    return detector.detect_guns(img_array, threshold), img_array


@app.get("/model_info")
def get_model_info(detector: GunDetector = Depends(get_gun_detector)):
    return {
        "model_name": "Gun detector",
        "gun_detector_model": detector.od_model.model.__class__.__name__,
        "semantic_segmentation_model": detector.seg_model.model.__class__.__name__,
        "input_type": "image",
    }


@app.post("/detect_guns")
def detect_guns(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> Detection:
    results, _ = detect_uploadfile(detector, file, threshold)

    return results


@app.post("/annotate_guns")
def annotate_guns(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> Response:
    detection, img = detect_uploadfile(detector, file, threshold)
    annotated_img = annotate_detection(img, detection)

    img_pil = Image.fromarray(annotated_img)
    image_stream = io.BytesIO()
    img_pil.save(image_stream, format="JPEG")
    image_stream.seek(0)
    return Response(content=image_stream.read(), media_type="image/jpeg")

@app.post("/detect_people")
def detect_people(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> Segmentation:
    _, img_array = detect_uploadfile(detector, file, threshold)
    segmentation = detector.segment_people(img_array, threshold)
    return segmentation

@app.post("/annotate_people")
def annotate_people(
    threshold: float = 0.5,
    draw_boxes: bool = True,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> Response:
    _, img_array = detect_uploadfile(detector, file, threshold)
    segmentation = detector.segment_people(img_array, threshold)
    annotated_img = annotate_segmentation(img_array, segmentation, draw_boxes)
    
    img_pil = Image.fromarray(annotated_img)
    image_stream = io.BytesIO()
    img_pil.save(image_stream, format="JPEG")
    image_stream.seek(0)
    return Response(content=image_stream.read(), media_type="image/jpeg")

@app.post("/detect")
def detect(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> dict:
    # Detectar armas y personas
    detection, img_array = detect_uploadfile(detector, file, threshold)
    segmentation = detector.segment_people(img_array, threshold)
    
    return {
        "detection": detection,
        "segmentation": segmentation
    }

@app.post("/annotate")
def annotate(
    threshold: float = 0.5,
    draw_boxes: bool = True,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> Response:
    detection, img_array = detect_uploadfile(detector, file, threshold)
    segmentation = detector.segment_people(img_array, threshold)
    
    annotated_img = annotate_segmentation(img_array, segmentation, draw_boxes)
    annotated_img = annotate_detection(annotated_img, detection)
    
    img_pil = Image.fromarray(annotated_img)
    image_stream = io.BytesIO()
    img_pil.save(image_stream, format="JPEG")
    image_stream.seek(0)
    return Response(content=image_stream.read(), media_type="image/jpeg")

@app.post("/guns")
def detect_guns_info(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> list[Gun]:
    detection, _ = detect_uploadfile(detector, file, threshold)
    guns = []
    
    for label, box in zip(detection.labels, detection.boxes):
        gun_type = GunType.pistol if 'pistol' in label.lower() else GunType.rifle
        x_center = (box[0] + box[2]) // 2
        y_center = (box[1] + box[3]) // 2
        guns.append(Gun(gun_type=gun_type, location=PixelLocation(x=x_center, y=y_center)))
    
    return guns

@app.post("/people")
def detect_people_info(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> list[Person]:
    _, img_array = detect_uploadfile(detector, file, threshold)
    segmentation = detector.segment_people(img_array, threshold)
    people = []
    
    for label, box in zip(segmentation.labels, segmentation.boxes):
        x_center = (box[0] + box[2]) // 2
        y_center = (box[1] + box[3]) // 2
        area = (box[2] - box[0]) * (box[3] - box[1])
        people.append(Person(person_type=PersonType(label), location=PixelLocation(x=x_center, y=y_center), area=area))
    
    return people


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.app:app", port=8080, host="0.0.0.0")
