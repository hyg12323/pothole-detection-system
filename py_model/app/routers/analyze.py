import os
import uuid
from typing import List
from fastapi import APIRouter, UploadFile, File

from app.services.accident_service import predict_accident
from app.services.vehicle_service import predict_vehicle_type

# =========================
# Router
# =========================
router = APIRouter(
    prefix="/analyze",
    tags=["Analysis"]
)

# =========================
# Temp 저장 경로
# =========================
BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, "..", "..", "temp")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ============================================================
# 단일 이미지 분석
# - 사고 여부 판단
# - 차량 종류 분류
# - 결과와 상관없이 damage로 전달
# ============================================================
@router.post("/single")
async def analyze_single(file: UploadFile = File(...)):
    filename = f"{uuid.uuid4()}.jpg"
    file_path = os.path.join(UPLOAD_DIR, filename)

    # 이미지 저장
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # 사고 여부 판단
    accident_result = predict_accident(file_path)

    # 차량 종류 분류 (항상 수행)
    vehicle_result = predict_vehicle_type(file_path)

    return {
        "accident": accident_result,          # T/F + confidence
        "vehicle": vehicle_result,            # 차종 분류
        "next": "/damage/detect",              # 다음 파이프라인
        "image_path": file_path
    }


# ============================================================
# 다중 이미지 분석
# - 대표 이미지 기준 사고 판단
# - 대표 이미지 기준 차량 분류
# - 모든 이미지는 damage/multi로 전달
# ============================================================
@router.post("/multi")
async def analyze_multi(files: List[UploadFile] = File(...)):
    image_paths = []
    first_image_path = None

    for idx, file in enumerate(files):
        filename = f"{uuid.uuid4()}.jpg"
        file_path = os.path.join(UPLOAD_DIR, filename)

        with open(file_path, "wb") as f:
            f.write(await file.read())

        image_paths.append(file_path)

        # 첫 번째 이미지를 대표 이미지로 사용
        if idx == 0:
            first_image_path = file_path

    # 사고 여부 판단 (대표 이미지 기준)
    accident_result = predict_accident(first_image_path)

    # 차량 종류 분류 (대표 이미지 기준)
    vehicle_result = predict_vehicle_type(first_image_path)

    return {
        "accident": accident_result,
        "vehicle": vehicle_result,
        "next": "/damage/detect/multi",
        "image_paths": image_paths
    }
