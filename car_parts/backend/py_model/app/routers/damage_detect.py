from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from app.services.yolo_service import YoloService
from fastapi.responses import Response, StreamingResponse, JSONResponse
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


@router.post("/detect")
async def detect_damage(file: UploadFile = File(...)):
    if not yolo_service.is_ready():
        raise HTTPException(status_code=500, detail="YOLO model not loaded")

    image_bytes = await file.read()
    detections = yolo_service.detect(image_bytes)

    if not detections:
        return {
            "status": "UNSURE",
            "message": "파손 여부를 판단하기 어렵습니다",
            "detections": []
        }

    return {
        "status": "DETECTED",
        "primary_damage": detections[0],
        "detections": detections
    }

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
                "confidence_threshold": conf,
                "message": "파손 여부를 판단하기 어렵습니다",
                "detections": []
            }
        )

    return StreamingResponse(
        io.BytesIO(img_bytes),
        media_type="image/jpeg",
        headers={
            "X-Detection-Status": "DETECTED",
            "X-Detection-Count": str(len(detections)),
            "X-Primary-Damage": detections[0]["class_name"],
            "X-Confidence-Threshold": str(conf)
        }
    )


@router.post("/detect/multipart")
async def detect_damage_multipart(file: UploadFile = File(...)):
    if not yolo_service.is_ready():
        raise HTTPException(status_code=500, detail="YOLO model not loaded")

    image_bytes = await file.read()

    img_bytes, detections = yolo_service.detect_and_draw(image_bytes)

    boundary = f"boundary-{uuid.uuid4().hex}"

    json_part = json.dumps({
        "count": len(detections),
        "detections": detections
    })

    body = (
        f"--{boundary}\r\n"
        f"Content-Type: image/jpeg\r\n"
        f"Content-Disposition: inline; filename=\"result.jpg\"\r\n\r\n"
    ).encode() + img_bytes + (
        f"\r\n--{boundary}\r\n"
        f"Content-Type: application/json\r\n\r\n"
        f"{json_part}\r\n"
        f"--{boundary}--\r\n"
    ).encode()

    return Response(
        content=body,
        media_type=f"multipart/mixed; boundary={boundary}"
    )
