from ultralytics import YOLO
import numpy as np
import cv2
from shapely.geometry import Polygon, box
from src.models import Detection, PredictionType, Segmentation
from src.config import get_settings

SETTINGS = get_settings()


def match_gun_bbox(segment: list[list[int]], bboxes: list[list[int]], max_distance: int = 10) -> list[int] | None:
    segment_box = box(*segment)
    matched_box = None
    min_distance = float('inf')
    
    for bbox in bboxes:
        gun_box = box(*bbox)
        distance = segment_box.distance(gun_box)
        
        if distance < min_distance and distance <= max_distance:
            min_distance = distance
            matched_box = bbox 
    
    return matched_box


def annotate_detection(image_array: np.ndarray, detection: Detection) -> np.ndarray:
    ann_color = (255, 0, 0)
    annotated_img = image_array.copy()
    for label, conf, box in zip(detection.labels, detection.confidences, detection.boxes):
        x1, y1, x2, y2 = box
        cv2.rectangle(annotated_img, (x1, y1), (x2,y2), ann_color, 3)
        cv2.putText(
            annotated_img,
            f"{label}: {conf:.1f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            ann_color,
            2,
        )
    return annotated_img


def annotate_segmentation(image_array: np.ndarray, segmentation: Segmentation, draw_boxes: bool = True) -> np.ndarray:
    annotated_img = image_array.copy()
    overlay = annotated_img.copy()

    for label, polygon in zip(segmentation.labels, segmentation.polygons):
        polygon_points = np.array(polygon, np.int32)
        polygon_points = polygon_points.reshape((-1, 1, 2))

        if label == "danger":
            color = (255, 0, 0)
        else:
            color = (0, 255, 0)

        cv2.fillPoly(overlay, [polygon_points], color)

    alpha = 0.6
    cv2.addWeighted(overlay, alpha, annotated_img, 1 - alpha, 0, annotated_img)

    if draw_boxes:
        for label, box in zip(segmentation.labels, segmentation.boxes):
            x1, y1, x2, y2 = box
            color = (255, 0, 0) if label == "danger" else (0, 255, 0)
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated_img,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2,
            )

    return annotated_img


class GunDetector:
    def __init__(self) -> None:
        print(f"loading od model: {SETTINGS.od_model_path}")
        self.od_model = YOLO(SETTINGS.od_model_path)
        print(f"loading seg model: {SETTINGS.seg_model_path}")
        self.seg_model = YOLO(SETTINGS.seg_model_path)

    def detect_guns(self, image_array: np.ndarray, threshold: float = 0.5):
        results = self.od_model(image_array, conf=threshold)[0]
        labels = results.boxes.cls.tolist()
        indexes = [
            i for i in range(len(labels)) if labels[i] in [3, 4]
        ]  # 0 = "person"
        boxes = [
            [int(v) for v in box]
            for i, box in enumerate(results.boxes.xyxy.tolist())
            if i in indexes
        ]
        confidences = [
            c for i, c in enumerate(results.boxes.conf.tolist()) if i in indexes
        ]
        labels_txt = [
            results.names[labels[i]] for i in indexes
        ]
        return Detection(
            pred_type=PredictionType.object_detection,
            n_detections=len(boxes),
            boxes=boxes,
            labels=labels_txt,
            confidences=confidences,
        )
    
    def segment_people(self, image_array: np.ndarray, threshold: float = 0.5, max_distance: int = 10):
        seg_results = self.seg_model(image_array, conf=threshold)[0]
        people_indexes = [
            i for i, label in enumerate(seg_results.boxes.cls.tolist()) if label == 0
        ]

        person_polygons = [
            [[int(coord[0]), int(coord[1])] for coord in seg_results.masks.xy[i]]
            for i in people_indexes
        ]

        person_boxes = [
            [int(v) for v in box]
            for i, box in enumerate(seg_results.boxes.xyxy.tolist())
            if i in people_indexes
        ]
        
        gun_detection = self.detect_guns(image_array, threshold)
        
        labels = []
        for person_box in person_boxes:
            gun_bbox = match_gun_bbox(person_box, gun_detection.boxes, max_distance)
            if gun_bbox is not None:
                labels.append('danger')
            else:
                labels.append('safe')
        
        return Segmentation(
            pred_type=PredictionType.segmentation,
            n_detections=len(person_polygons),
            polygons=person_polygons,
            boxes=person_boxes,
            labels=labels
        )