# app/services/pipeline_service.py
from typing import List

from app.services.accident_service import predict_accident
from app.services.vehicle_service import predict_vehicle_type
from app.services.yolo_service import YoloService, select_primary_damage
from app.services.accident_rule_service import estimate_accident_type
from app.services.drive_cnn_service import CNNService

# Services (싱글톤)
yolo_service = YoloService()
cnn_service = CNNService()


def analyze_single_pipeline(image_path: str):
    accident = predict_accident(image_path)
    vehicle = predict_vehicle_type(image_path)

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    damage_result = yolo_service.detect(image_bytes)
    detections = damage_result["detections"]

    car_result = yolo_service.detect_with_car_crop(image_bytes)
    car_count = car_result["car_count"]

    if not detections:
        return {
            "mode": "single",
            "status": "NO_DAMAGE",
            "accident": accident,
            "vehicle": vehicle,
            "accident_type": None,
            "primary_damage": None,
            "detections": [],
            "drivable": {
                "drivable": True,
                "reason": "no_damage_detected"
            }
        }

    accident_type = estimate_accident_type(
        detections,
        car_count=car_count,
        mode="single"
    )

    primary_damage = select_primary_damage(detections)

    drivable = cnn_service.judge(
        detections=detections,
        image_bytes=image_bytes
    )

    return {
        "mode": "single",
        "status": "ANALYZED",
        "accident": accident,
        "vehicle": vehicle,
        "accident_type": accident_type,
        "primary_damage": primary_damage,
        "detections": detections,
        "car_count": car_count,
        "drivable": drivable
    }


def analyze_multi_pipeline(image_paths: List[str]):
    all_detections = []
    image_results = []
    first_image_bytes = None
    total_car_count = 0

    accident = None
    vehicle = None

    for idx, path in enumerate(image_paths):
        if idx == 0:
            accident = predict_accident(path)
            vehicle = predict_vehicle_type(path)
            with open(path, "rb") as f:
                first_image_bytes = f.read()

        with open(path, "rb") as f:
            image_bytes = f.read()

        damage_result = yolo_service.detect(image_bytes)
        detections = damage_result["detections"]

        car_result = yolo_service.detect_with_car_crop(image_bytes)
        total_car_count = max(total_car_count, car_result["car_count"])

        image_results.append({
            "image_path": path,
            "detection_count": len(detections),
            "detections": detections
        })

        all_detections.extend(detections)

    if not all_detections:
        return {
            "mode": "multi",
            "status": "NO_DAMAGE",
            "accident": accident,
            "vehicle": vehicle,
            "accident_type": None,
            "primary_damage": None,
            "total_detection_count": 0,
            "images": image_results,
            "drivable": {
                "drivable": True,
                "reason": "no_damage_detected"
            }
        }

    accident_type = estimate_accident_type(
        all_detections,
        car_count=total_car_count,
        mode="multi"
    )

    primary_damage = select_primary_damage(all_detections)

    # multi는 CNN을 1회만. (대표 이미지 1장 기준)
    drivable = cnn_service.judge(
        detections=all_detections,
        image_bytes=first_image_bytes
    )

    return {
        "mode": "multi",
        "status": "ANALYZED",
        "accident": accident,
        "vehicle": vehicle,
        "accident_type": accident_type,
        "primary_damage": primary_damage,
        "total_detection_count": len(all_detections),
        "images": image_results,
        "car_count": total_car_count,
        "drivable": drivable
    }
