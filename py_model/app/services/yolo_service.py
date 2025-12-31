from ultralytics import YOLO
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np

# =========================
# Paths & Models
# =========================
BASE_DIR = Path(__file__).resolve().parents[2]
DAMAGE_MODEL_PATH = BASE_DIR / "models" / "best.pt"   # 파손 탐지 모델
CAR_MODEL_PATH = "yolov8n.pt"                          # COCO 차량 탐지 모델

# =========================
# Threshold Config
# =========================
DEFAULT_CONF_THRESHOLD = 0.3

CLASS_CONF_THRESHOLD = {
    # 핵심 파손 (중간)
    "Bumper": 0.2,
    "Bonnet": 0.2,
    "Fender": 0.2,

    # 데이터 적음 → 오탐 방지
    "Door": 0.25,

    # 레어 + background로 빠짐 → 살려야 함
    "Trunk": 0.1,
    "Windshield": 0.1,
    "Light": 0.1,
}

CLASS_COLORS = defaultdict(lambda: (0, 255, 0))
CLASS_COLORS.update({
    "Bumper": (255, 0, 0),
    "Bonnet": (255, 128, 0),
    "Fender": (128, 0, 255),
    "Door": (0, 0, 255),
    "Light": (0, 255, 255),
    "Trunk": (255, 0, 255),        # 새로 추가
    "Windshield": (0, 255, 128),
})

# COCO 차량 클래스
CAR_CLASSES = {
    "car",
    "truck",
    "bus",
}

PART_IMPORTANCE = {
    "Windshield": 1.5,
    "Trunk": 1.4,
    "Bonnet": 1.3,
    "Door": 1.2,
    "Fender": 1.1,
    "Bumper": 1.0,
    "Light": 0.9,
}

REGION_IMPORTANCE = {
    "front": 1.2,
    "rear": 1.2,
    "side": 1.0,
}

# =========================
# Utils
# =========================
def compute_region(cx: float, image_width: int) -> str:
    if cx < image_width * 0.33:
        return "front"
    elif cx < image_width * 0.66:
        return "side"
    return "rear"


class YoloService:
    def __init__(self):
        self.damage_model = None
        self.car_model = None

        if DAMAGE_MODEL_PATH.exists():
            self.damage_model = YOLO(str(DAMAGE_MODEL_PATH))

        # COCO pretrained (차량 탐지용)
        self.car_model = YOLO(CAR_MODEL_PATH)

    def is_ready(self) -> bool:
        return self.damage_model is not None and self.car_model is not None

    # ==========================================================
    #  사고 판단용 파손 추론 (대표 1개 제한 없음)
    # ==========================================================
    def _infer_all_damage(self, img, default_threshold: float):
        results = self.damage_model(img)[0]
        detections = []

        if results.boxes is None:
            return []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = self.damage_model.names[cls_id]

            threshold = CLASS_CONF_THRESHOLD.get(class_name, default_threshold)
            if conf < threshold:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) / 2

            detections.append({
                "class_id": cls_id,
                "class_name": class_name,
                "confidence": round(conf, 4),
                "bbox": [x1, y1, x2, y2],
                "cx": round(cx, 1),
            })

            print(class_name, conf)

        return detections

    # ==========================================================
    #  시각화용 파손 추론 (클래스별 대표 1개만 유지)
    # ==========================================================
    # def _infer_visual_damage(self, img, default_threshold: float):
    #     detections = self._infer_all_damage(img, default_threshold)

    #     best_by_class = {}
    #     for d in detections:
    #         name = d["class_name"]
    #         if name not in best_by_class or d["confidence"] > best_by_class[name]["confidence"]:
    #             best_by_class[name] = d

    #     return sorted(
    #         best_by_class.values(),
    #         key=lambda x: x["confidence"],
    #         reverse=True
    #     )

    def _infer_visual_damage(self, img, default_threshold: float):
    # 모든 파손 그대로 사용
        return self._infer_all_damage(img, default_threshold)

    # ==========================================================
    #  Detect (사고 판단용 / JSON)
    # ==========================================================
    def detect(self, image_bytes: bytes, conf_threshold: float = DEFAULT_CONF_THRESHOLD):
        np_img = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("Invalid image file")

        h, w = img.shape[:2]
        detections = self._infer_all_damage(img, conf_threshold)

        for d in detections:
            d["region"] = compute_region(d["cx"], w)

        return {
            "detections": detections,
            "car_count": 1
        }

    # ==========================================================
    #  Detect with car-crop fallback (사고 판단용)
    # ==========================================================
    def detect_with_car_crop(self, image_bytes: bytes):
        np_img = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("Invalid image file")

        h, w = img.shape[:2]

        # 차량 탐지
        car_results = self.car_model(img)[0]
        car_boxes = []

        if car_results.boxes:
            for box in car_results.boxes:
                cls_id = int(box.cls[0])
                class_name = self.car_model.names[cls_id].lower()

                if class_name not in CAR_CLASSES:
                    continue

                conf = float(box.conf[0])
                if conf < 0.5:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                car_boxes.append((x1, y1, x2, y2))

        # 차량 못 찾으면 파손 기반으로 보정
        if not car_boxes:
            result = self.detect(image_bytes)

            damage_count = len(result["detections"])
            has_strong_parts = any(
                d["class_name"] in {"Trunk", "Bonnet", "Windshield"}
                for d in result["detections"]
            )

            # 파손이 충분하면 차량은 존재한다고 판단
            if damage_count >= 2 or has_strong_parts:
                result["car_count"] = 1

            return result

        all_detections = []

        for (x1, y1, x2, y2) in car_boxes:
            car_width = max(x2 - x1, 1)

            crop = img[y1:y2, x1:x2]
            detections = self._infer_all_damage(crop, DEFAULT_CONF_THRESHOLD)

            for d in detections:
                bx1, by1, bx2, by2 = d["bbox"]

                d["bbox"] = [bx1 + x1, by1 + y1, bx2 + x1, by2 + y1]

                global_cx = d["cx"] + x1
                d["cx"] = round(global_cx, 1)

                relative_cx = (global_cx - x1) / car_width

                if relative_cx < 0.33:
                    d["region"] = "front"
                elif relative_cx < 0.66:
                    d["region"] = "side"
                else:
                    d["region"] = "rear"


            all_detections.extend(detections)

        return {
            "detections": all_detections,
            "car_count": len(car_boxes)
        }

    def _filter_for_visual(self, detections: list):
        best = {}
        for d in detections:
            name = d["class_name"]
            if name not in best or d["confidence"] > best[name]["confidence"]:
                best[name] = d
        return list(best.values())

    # ==========================================================
    #  Detect + Draw (시각화 전용)
    # ==========================================================
    def detect_and_draw(self, image_bytes: bytes, conf_threshold: float = DEFAULT_CONF_THRESHOLD):
        np_img = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("Invalid image file")

        h, w = img.shape[:2]

        # 파손 추론
        detections = self._infer_visual_damage(img, conf_threshold)

        # 🔹 시각화 전용 필터 (중복 제거)
        detections = self._filter_for_visual(detections)

        for idx, d in enumerate(detections):
            d["region"] = compute_region(d["cx"], w)

            x1, y1, x2, y2 = d["bbox"]
            label = f'{d["class_name"]} {d["confidence"]:.2f} ({d["region"]})'
            color = CLASS_COLORS[d["class_name"]]
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

def compute_damage_score(d: dict) -> float:
    conf = d.get("confidence", 0.0)
    part = d.get("class_name")
    region = d.get("region")

    part_w = PART_IMPORTANCE.get(part, 1.0)
    region_w = REGION_IMPORTANCE.get(region, 1.0)

    return conf * part_w * region_w

def select_primary_damage(detections: list) -> dict | None:
    if not detections:
        return None

    scored = []
    for d in detections:
        score = compute_damage_score(d)
        scored.append((score, d))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]
