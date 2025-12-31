from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import Response, StreamingResponse, JSONResponse

from app.services.yolo_service import YoloService, select_primary_damage
from app.services.accident_service import estimate_accident_type

import json
import uuid
import io

router = APIRouter(prefix="/damage", tags=["Damage Detection"])
yolo_service = YoloService()


@router.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": yolo_service.is_ready()
    }


# =========================
# JSON 기반 사고 판단
# =========================
@router.post("/detect")
async def detect_damage(file: UploadFile = File(...)):
    if not yolo_service.is_ready():
        raise HTTPException(status_code=500, detail="YOLO model not loaded")

    image_bytes = await file.read()

    #  파손 탐지 (항상 실행)
    damage_result = yolo_service.detect(image_bytes)

    #  차량 수 탐지 (항상 실행)
    car_result = yolo_service.detect_with_car_crop(image_bytes)

    detections = damage_result["detections"]
    car_count = car_result["car_count"]

    #  파손이 하나도 없으면 UNSURE
    if not detections:
        return {
            "status": "UNSURE",
            "message": "파손 여부를 판단하기 어렵습니다",
            "accident": {
                "accident_detected": False,
                "accident_state": "NO_ACCIDENT",
                "accident_type": "UNKNOWN",
                "confidence_level": "LOW",
                "scores": {}
            },
            "detections": []
        }

    #  사고 판단 (★ car_count가 이제 의미 있음)
    accident = estimate_accident_type(
        detections,
        car_count=car_count
    )

    
    primary = select_primary_damage(detections)

    return {
        "status": "DETECTED",
        "accident": accident,
        "primary_damage": primary,     # 사고 기여도 기반
        "detections": detections,
        "car_count": car_count
    }



# =========================
# 이미지 + 사고 판단 (헤더)
# =========================
@router.post("/detect/image")
async def detect_damage_image(
    file: UploadFile = File(...),
    conf: float = Query(0.4, ge=0.0, le=1.0)
):
    if not yolo_service.is_ready():
        raise HTTPException(status_code=500, detail="YOLO model not loaded")

    image_bytes = await file.read()

    # detect_and_draw는 시각화 목적 → car-crop 미적용
    img_bytes, detections = yolo_service.detect_and_draw(
        image_bytes,
        conf_threshold=conf
    )

    if not detections:
        return JSONResponse(
            status_code=200,
            content={
                "status": "UNSURE",
                "confidence_threshold": conf,
                "message": "파손 여부를 판단하기 어렵습니다",
                "detections": []
            }
        )

    # 사고 판단
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
# 이미지 + JSON multipart
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
            "accident": {
                "accident_detected": False,
                "accident_state": "NO_ACCIDENT",
                "accident_type": "UNKNOWN",
                "confidence_level": "LOW",
                "scores": {}
            },
            "images": image_results
        }

    accident = estimate_accident_type(
        all_detections,
        car_count=total_car_count
    )

    return {
        "status": "DETECTED",
        "image_count": len(files),
        "accident": accident,
        "total_detection_count": len(all_detections),
        "images": image_results
    }