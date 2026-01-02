from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse

from app.services.yolo_service import YoloService, select_primary_damage
from app.services.accident_service import estimate_accident_type
from app.services.cnn_service import CNNService

import io
import cv2
import numpy as np


# =========================
# Router / Services
# =========================
router = APIRouter(prefix="/damage", tags=["Damage Detection"])

yolo_service = YoloService()
cnn_service = CNNService()


# =========================
# Health Check
# =========================
@router.get("/health")
def health_check():
    return {
        "status": "ok",
        "yolo_loaded": yolo_service.is_ready(),
        "cnn_loaded": True
    }


# =========================
# CNN 기반 주행 가능 판단
# =========================
def judge_drivable_by_cnn(detections, image_bytes):
    """
    YOLO detections → crop → CNN
    하나라도 severe면 주행 불가
    """
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        return {
            "drivable": False,
            "reason": "invalid_image"
        }

    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        crop = img[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        cnn_result = cnn_service.predict(crop)

        if cnn_result["label"] == "severe":
            return {
                "drivable": False,
                "reason": "cnn_severe_detected",
                "cnn": cnn_result
            }

    return {
        "drivable": True,
        "reason": "cnn_normal_only"
    }


# =========================
# JSON 기반 사고 판단 + CNN
# =========================
@router.post("/detect")
async def detect_damage(file: UploadFile = File(...)):
    if not yolo_service.is_ready():
        raise HTTPException(status_code=500, detail="YOLO model not loaded")

    image_bytes = await file.read()

    # 1. 파손 탐지
    damage_result = yolo_service.detect(image_bytes)

    # 2. 차량 기준 탐지 (차량 수)
    car_result = yolo_service.detect_with_car_crop(image_bytes)

    detections = damage_result["detections"]
    car_count = car_result["car_count"]

    # 파손 없음
    if not detections:
        return {
            "status": "UNSURE",
            "message": "파손 여부를 판단하기 어렵습니다",
            "vehicle": False,
            "drivable": True,
            "detections": []
        }

    # 3. 사고 판단
    accident = estimate_accident_type(
        detections,
        car_count=car_count
    )

    primary = select_primary_damage(detections)

    # 4. CNN 주행 가능 판단
    cnn_judge = judge_drivable_by_cnn(detections, image_bytes)

    return {
        "status": "DETECTED",
        "vehicle": True,

        # 🔥 CNN 결과
        "drivable": cnn_judge["drivable"],
        "drivable_reason": cnn_judge["reason"],
        "cnn": cnn_judge.get("cnn"),

        # 사고 판단
        "accident": accident,
        "primary_damage": primary,
        "detections": detections,
        "car_count": car_count
    }


# =========================
# 이미지 + 시각화 (YOLO 전용)
# =========================
@router.post("/detect/image")
async def detect_damage_image(
    file: UploadFile = File(...),
    conf: float = Query(0.4, ge=0.0, le=1.0)
):
    if not yolo_service.is_ready():
        raise HTTPException(status_code=500, detail="YOLO model not loaded")

    image_bytes = await file.read()

    img_bytes, detections = yolo_service.detect_and_draw(
        image_bytes,
        conf_threshold=conf
    )

    if not detections:
        return JSONResponse(
            status_code=200,
            content={
                "status": "UNSURE",
                "message": "파손 여부를 판단하기 어렵습니다",
                "detections": []
            }
        )

    accident_result = estimate_accident_type(detections)

    return StreamingResponse(
        io.BytesIO(img_bytes),
        media_type="image/jpeg",
        headers={
            "X-Detection-Status": "DETECTED",
            "X-Detection-Count": str(len(detections)),
            "X-Primary-Damage": detections[0]["class_name"],
            "X-Accident-Type": accident_result["accident_type"],
            "X-Accident-Detected": str(accident_result["accident_detected"]),
            "X-Confidence-Threshold": str(conf)
        }
    )


# =========================
# 멀티 이미지 (YOLO 기준)
# =========================
@router.post("/detect/multi")
async def detect_damage_multi(files: list[UploadFile] = File(...)):
    if not yolo_service.is_ready():
        raise HTTPException(status_code=500, detail="YOLO model not loaded")

    all_detections = []
    image_results = []
    total_car_count = 0

    for idx, file in enumerate(files):
        image_bytes = await file.read()

        result = yolo_service.detect(image_bytes)
        if not result["detections"]:
            result = yolo_service.detect_with_car_crop(image_bytes)

        total_car_count = max(total_car_count, result["car_count"])

        image_results.append({
            "image_index": idx,
            "filename": file.filename,
            "detection_count": len(result["detections"]),
            "detections": result["detections"]
        })

        all_detections.extend(result["detections"])

    if not all_detections:
        return {
            "status": "UNSURE",
            "message": "모든 이미지에서 파손을 탐지하지 못했습니다",
            "image_count": len(files),
            "vehicle": False,
            "drivable": True,
            "images": image_results
        }

    accident = estimate_accident_type(
        all_detections,
        car_count=total_car_count
    )

    return {
        "status": "DETECTED",
        "vehicle": True,
        "accident": accident,
        "total_detection_count": len(all_detections),
        "images": image_results
    }
