from fastapi import APIRouter, UploadFile, File
from typing import List
import os
import uuid

from app.services.pipeline_service import (
    analyze_single_pipeline,
    analyze_multi_pipeline
)

router = APIRouter(
    prefix="/pipeline",
    tags=["Pipeline"]
)

BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, "..", "..", "temp")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/single")
async def pipeline_single(file: UploadFile = File(...)):
    filename = f"{uuid.uuid4()}.jpg"
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    try:
        result = analyze_single_pipeline(file_path)
        return result
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@router.post("/multi")
async def pipeline_multi(files: List[UploadFile] = File(...)):
    image_paths = []

    try:
        for file in files:
            filename = f"{uuid.uuid4()}.jpg"
            file_path = os.path.join(UPLOAD_DIR, filename)

            with open(file_path, "wb") as f:
                f.write(await file.read())

            image_paths.append(file_path)

        result = analyze_multi_pipeline(image_paths)
        return result

    finally:
        for path in image_paths:
            if os.path.exists(path):
                os.remove(path)
