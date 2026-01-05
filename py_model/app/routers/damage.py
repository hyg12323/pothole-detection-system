from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse

from app.services.yolo_service import YoloService
from app.services.accident_rule_service import estimate_accident_type
from app.services import drive_cnn_service

import io

router = APIRouter(prefix="/damage", tags=["Damage Detection"])

yolo_service = YoloService()


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
# 단일 이미지 파손 탐지 + 주행 가능 판단
# =========================
@router.post("/detect")
async def detect_damage(file: UploadFile = File(...)):
    image_bytes = await file.read()

    # 1. YOLO 파손 탐지
    damage_result = yolo_service.detect(image_bytes)
    detections = damage_result["detections"]

    if not detections:
        return {
            "status": "UNSURE",
            "message": "파손을 탐지하지 못했습니다",
            "detections": [],
            "drivable": None
        }

    # 2. CNN 주행 가능 판단 (1회)
    drive_result = drive_cnn_service.judge(
        detections=detections,
        image_bytes=image_bytes
    )

    # 3. 차량 수 기준 탐지
    car_result = yolo_service.detect_with_car_crop(image_bytes)

    # 4. 사고 유형 판단
    accident = estimate_accident_type(
        detections,
        car_count=car_result["car_count"]
    )

    return {
        "status": "DETECTED",
        "detections": detections,
        "accident": accident,
        "car_count": car_result["car_count"],
        "drivable": drive_result
    }


# =========================
# 이미지 + 시각화 (YOLO 전용)
# =========================
@router.post("/detect/image")
async def detect_damage_image(
    file: UploadFile = File(...),
    conf: float = Query(0.4, ge=0.0, le=1.0)
):
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
# 멀티 이미지 파손 탐지 (CNN 1회)
# =========================
@router.post("/detect/multi")
async def detect_damage_multi(files: list[UploadFile] = File(...)):
    all_detections = []
    image_results = []
    total_car_count = 0
    first_image_bytes = None

    for idx, file in enumerate(files):
        image_bytes = await file.read()

        if first_image_bytes is None:
            first_image_bytes = image_bytes

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
            "drivable": None,
            "images": image_results
        }

    # CNN 주행 가능 판단 (1회)
    drive_result = drive_cnn_service.judge(
        detections=all_detections,
        image_bytes=first_image_bytes
    )

    accident = estimate_accident_type(
        all_detections,
        car_count=total_car_count
    )

    return {
        "status": "DETECTED",
        "accident": accident,
        "drivable": drive_result,
        "total_detection_count": len(all_detections),
        "images": image_results
    }
