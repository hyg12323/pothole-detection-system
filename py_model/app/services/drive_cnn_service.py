import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
from pathlib import Path

# =========================
# Model Path
# =========================
BASE_DIR = Path(__file__).resolve().parents[2]
CNN_MODEL_PATH = BASE_DIR / "models" / "cnn" / "cnn_severity_resnet18.pt"


# =========================
# CNN Model Service
# =========================
class CNNService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ResNet18 구조
        self.model = models.resnet18(pretrained=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 2)

        # 가중치 로드
        state_dict = torch.load(CNN_MODEL_PATH, map_location=self.device)
        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

        # 전처리
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.class_names = ["normal", "severe"]

    # =========================
    # numpy / cv2 → PIL
    # =========================
    def _to_pil(self, img):
        if isinstance(img, np.ndarray):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return Image.fromarray(img)
        return img

    # =========================
    # 단일 crop 예측
    # =========================
    def predict(self, img):
        pil_img = self._to_pil(img)
        x = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)
            conf, pred = torch.max(probs, 1)

        return {
            "label": self.class_names[pred.item()],
            "confidence": round(conf.item(), 4)
        }

    # =========================
    # 주행 가능 판단 (YOLO 결과 기반)
    # =========================
    def judge(self, detections, image_bytes):
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

            cnn_result = self.predict(crop)

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
