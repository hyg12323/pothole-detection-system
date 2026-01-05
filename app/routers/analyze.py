import os
import uuid
from fastapi import APIRouter, UploadFile, File

from app.services.accident_service import predict_accident
from app.services.vehicle_service import predict_vehicle_type

router = APIRouter(
    prefix="/analyze",
    tags=["Analysis"]
)

BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, "..", "..", "temp")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("")
async def analyze_image(file: UploadFile = File(...)):
    filename = f"{uuid.uuid4()}.jpg"
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    accident_result = predict_accident(file_path)

    response = {
        "accident": accident_result
    }

    if accident_result["is_accident"]:
        response["vehicle"] = predict_vehicle_type(file_path)

    os.remove(file_path)

    return response
