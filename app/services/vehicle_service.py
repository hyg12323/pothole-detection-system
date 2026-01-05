import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# =========================
# 설정
# =========================
IMG_SIZE = 224

CLASS_NAMES = [
    "convertible",
    "coupe",
    "hatchback",
    "pickuptruck",
    "sedan",
    "suv"
]

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "..",
    "model",
    "vehicle_model.h5"
)

# =========================
# 모델 로드 (서버 시작 시 1회)
# =========================
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"차종 분류 모델 로드 실패: {e}")

# =========================
# 이미지 전처리
# =========================
def _preprocess_image(img_path: str) -> np.ndarray:
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    x = image.img_to_array(img) / 255.0
    return np.expand_dims(x, axis=0)

# =========================
# 차종 예측
# =========================
def predict_vehicle_type(img_path: str) -> dict:
    x = _preprocess_image(img_path)

    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))

    return {
        "vehicle_type": CLASS_NAMES[idx],
        "confidence": float(probs[idx]),
        "probabilities": {
            CLASS_NAMES[i]: float(probs[i])
            for i in range(len(CLASS_NAMES))
        }
    }
