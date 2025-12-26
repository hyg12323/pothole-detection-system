from ultralytics import YOLO
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np

# =========================
# Model & Threshold Config
# =========================

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "best.pt"

DEFAULT_CONF_THRESHOLD = 0.3

# 클래스별 confidence threshold
CLASS_CONF_THRESHOLD = {
    "Bumper": 0.4,
    "Door": 0.4,
    "Fender": 0.4,
    "Bonnet": 0.4,
    "Windshield": 0.35,
    "Light": 0.25,   # 작고 중요한 부위
}

# 클래스별 색상 (BGR)
CLASS_COLORS = defaultdict(lambda: (0, 255, 0))
CLASS_COLORS.update({
    "Bumper": (255, 0, 0),
    "Door": (0, 0, 255),
    "Light": (0, 255, 255),
})


class YoloService:
    def __init__(self):
        self.model = None
        if MODEL_PATH.exists():
            self.model = YOLO(str(MODEL_PATH))

    def is_ready(self) -> bool:
        return self.model is not None

    # =========================
    # Core Inference Logic
    # =========================
    def _infer(self, img, default_threshold: float):
        """
        공통 추론 로직
        - 클래스별 threshold 적용
        - 클래스별 최고 confidence 1개만 유지
        """
        results = self.model(img)[0]
        detections = []

        if results.boxes is None:
            return []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = self.model.names[cls_id]

            threshold = CLASS_CONF_THRESHOLD.get(
                class_name,
                default_threshold
            )

            if conf < threshold:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detections.append({
                "class_id": cls_id,
                "class_name": class_name,
                "confidence": round(conf, 4),
                "bbox": [x1, y1, x2, y2],
            })

        # 클래스별 대표 1개만 유지
        best_by_class = {}
        for d in detections:
            name = d["class_name"]
            if name not in best_by_class or d["confidence"] > best_by_class[name]["confidence"]:
                best_by_class[name] = d

        # confidence 기준 정렬
        return sorted(
            best_by_class.values(),
            key=lambda x: x["confidence"],
            reverse=True
        )

    # =========================
    # Detect (JSON only)
    # =========================
    def detect(self, image_bytes: bytes, conf_threshold: float = DEFAULT_CONF_THRESHOLD):
        if not self.model:
            raise RuntimeError("YOLO model is not loaded")

        np_img = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if img is None:
            raise RuntimeError("Invalid image file")

        return self._infer(img, conf_threshold)

    # =========================
    # Detect + Draw (Image)
    # =========================
    def detect_and_draw(self, image_bytes: bytes, conf_threshold: float = DEFAULT_CONF_THRESHOLD):
        if not self.model:
            raise RuntimeError("YOLO model is not loaded")

        np_img = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if img is None:
            raise RuntimeError("Invalid image file")

        detections = self._infer(img, conf_threshold)

        for idx, d in enumerate(detections):
            x1, y1, x2, y2 = d["bbox"]
            class_name = d["class_name"]
            conf = d["confidence"]

            label = f"{class_name} {conf:.2f}"
            color = CLASS_COLORS[class_name]

            # 대표 파손(가장 confidence 높은 1개) 강조
            thickness = 4 if idx == 0 else 2

            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(
                img,
                label,
                (x1, max(y1 - 10, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                thickness
            )

        _, encoded = cv2.imencode(".jpg", img)
        return encoded.tobytes(), detections
