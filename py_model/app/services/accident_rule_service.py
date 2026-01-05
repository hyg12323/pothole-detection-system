# app/services/accident_rule_service.py
from collections import defaultdict

ACCIDENT_RELATED_CLASSES = {
    "Bumper",
    "Fender",
    "Light",
    "Bonnet",
    "Windshield",
    "Door",
    "CAR_TRUNK-KP48",
    "Dickey",
}

MIN_CONF_FOR_ACCIDENT = 0.2

MULTI_COLLISION = "MULTI_COLLISION"
COMPLEX_DAMAGE = "COMPLEX_DAMAGE"  # 동일 차량 다각도(전/후 동시 등) 복합 파손

SCORE_RULES = {
    ("rear", "Trunk"): {"REAR_COLLISION": 2.5},
    ("rear", "Bumper"): {"REAR_COLLISION": 2.0},
    ("rear", "Light"): {"REAR_COLLISION": 1.0},
    ("rear", "Fender"): {"REAR_COLLISION": 1.0},

    ("side", "Door"): {"SIDE_COLLISION": 1.5},
    ("side", "Fender"): {"SIDE_COLLISION": 1.5},
    ("side", "Bumper"): {"SIDE_COLLISION": 1.0},
    ("side", "Light"): {"SIDE_COLLISION": 1.0},

    ("front", "Bumper"): {"FRONT_COLLISION": 0.8},
    ("front", "Fender"): {"FRONT_COLLISION": 1.0},
    ("front", "Light"): {"FRONT_COLLISION": 1.0},
}

PART_ONLY_RULES = {
    "Bonnet": {"FRONT_COLLISION": 4.0},
    "Windshield": {"FRONT_COLLISION": 4.0},
    "Trunk": {"REAR_COLLISION": 4.0},
    "Door": {"SIDE_COLLISION": 2.0},
}

SIDE_PART_BONUS = {
    "Door": 1.5,
    "Fender": 1.2,
}


def normalize_part_name(part: str) -> str:
    if part in {"CAR_TRUNK-KP48", "Dickey"}:
        return "Trunk"
    return part


def estimate_accident_scores(detections: list) -> dict:
    scores = defaultdict(float)

    for d in detections:
        raw_part = d.get("class_name")
        part = normalize_part_name(raw_part)
        region = d.get("region")
        conf = d.get("confidence", 0.0)

        if not part or not region:
            continue

        if part in PART_ONLY_RULES:
            for t, base in PART_ONLY_RULES[part].items():
                scores[t] += base * conf

        key = (region, part)
        if key in SCORE_RULES:
            for t, base in SCORE_RULES[key].items():
                scores[t] += base * conf

        # side bonus는 side에서만
        if part in SIDE_PART_BONUS and region == "side":
            scores["SIDE_COLLISION"] += SIDE_PART_BONUS[part] * conf

    return dict(scores)


def can_decide_direction(scores: dict) -> bool:
    direction_scores = {
        k: v for k, v in scores.items()
        if k in {"FRONT_COLLISION", "REAR_COLLISION", "SIDE_COLLISION"}
    }

    if not direction_scores:
        return False

    values = sorted(direction_scores.values(), reverse=True)

    if values[0] < 2.0:
        return False

    if len(values) >= 2 and (values[0] - values[1]) < 1.0:
        return False

    return True


def estimate_accident_type(
    detections: list,
    car_count: int = 1,
    mode: str = "single"  # "single" | "multi"
) -> dict:
    valid_detections = [
        d for d in detections
        if d.get("class_name") in ACCIDENT_RELATED_CLASSES
        and d.get("confidence", 0.0) >= MIN_CONF_FOR_ACCIDENT
    ]

    damage_count = len(valid_detections)

    if damage_count == 0:
        return {
            "accident_detected": False,
            "accident_state": "NO_ACCIDENT",
            "accident_type": "UNKNOWN",
            "confidence_level": "LOW",
            "scores": {},
            "message": "파손이 명확하게 탐지되지 않았습니다."
        }

    scores = estimate_accident_scores(valid_detections)

    regions = {d.get("region") for d in valid_detections if d.get("region")}
    has_front_damage = "front" in regions
    has_rear_damage = "rear" in regions

    # 사고 상태
    if damage_count >= 2:
        state = "CONFIRMED_ACCIDENT"
    else:
        state = "SUSPECTED_ACCIDENT"

    # 사고 유형
    if state != "CONFIRMED_ACCIDENT":
        accident_type = "UNKNOWN"
    else:
        # 멀티 차량은 car_count로만 확정
        if car_count >= 2:
            accident_type = MULTI_COLLISION

        # multi 모드(동일 차량 여러각도)에서 front+rear 동시 파손이면 COMPLEX
        elif mode == "multi" and car_count == 1 and has_front_damage and has_rear_damage:
            accident_type = COMPLEX_DAMAGE

        # 방향 결정 가능하면 방향 선택
        elif can_decide_direction(scores):
            direction_scores = {
                k: v for k, v in scores.items()
                if k in {"FRONT_COLLISION", "REAR_COLLISION", "SIDE_COLLISION"}
            }
            accident_type = max(direction_scores, key=direction_scores.get)

        else:
            accident_type = "UNKNOWN"

    # 신뢰도
    if damage_count >= 3:
        confidence_level = "HIGH"
    elif damage_count == 2:
        confidence_level = "MEDIUM"
    else:
        confidence_level = "LOW"

    # 메시지
    if state == "CONFIRMED_ACCIDENT":
        if accident_type == MULTI_COLLISION:
            message = "여러 차량이 연루된 사고로 판단됩니다."
        elif accident_type == COMPLEX_DAMAGE:
            message = "동일 차량에서 여러 방향 파손이 확인되어 복합 파손으로 판단됩니다."
        elif accident_type != "UNKNOWN":
            message = "사고 방향이 비교적 명확하게 확인되었습니다."
        else:
            message = "사고는 확인되었으나 방향 판단에는 추가 정보가 필요합니다."
    else:
        message = "파손은 확인되었으나 사고 여부는 불확실합니다."

    return {
        "accident_detected": state == "CONFIRMED_ACCIDENT",
        "accident_state": state,
        "accident_type": accident_type,
        "confidence_level": confidence_level,
        "scores": scores,
        "message": message
    }
