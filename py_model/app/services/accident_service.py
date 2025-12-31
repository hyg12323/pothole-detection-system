from collections import defaultdict

# =========================
# 사고 관련 클래스
# =========================
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

# =========================
# region + part 보조 규칙
# =========================
SCORE_RULES = {
    # =================
    # REAR
    # =================
    ("rear", "Trunk"): {"REAR_COLLISION": 2.5},   # 핵심
    ("rear", "Bumper"): {"REAR_COLLISION": 2},
    ("rear", "Light"): {"REAR_COLLISION": 1},
    ("rear", "Fender"): {"REAR_COLLISION": 1},

    # =================
    # SIDE
    # =================
    ("side", "Door"): {"SIDE_COLLISION": 1.5},    # 약화
    ("side", "Fender"): {"SIDE_COLLISION": 1.5},
    ("side", "Bumper"): {"SIDE_COLLISION": 1},
    ("side", "Light"): {"SIDE_COLLISION": 1},

    # =================
    # FRONT
    # =================
    ("front", "Bumper"): {"FRONT_COLLISION": 0.8},
    ("front", "Fender"): {"FRONT_COLLISION": 1},
    ("front", "Light"): {"FRONT_COLLISION": 1},
}

# =========================
# part 단독 규칙 (강한 증거)
# =========================
PART_ONLY_RULES = {
    # FRONT 확정 증거
    "Bonnet": {"FRONT_COLLISION": 4},
    "Windshield": {"FRONT_COLLISION": 4},

    # REAR 확정 증거
    "Trunk": {"REAR_COLLISION": 4},

    # SIDE 단독 증거는 약화
    "Door": {"SIDE_COLLISION": 2},   # 4 → 2
}

# 위치 무관 보조 규칙
SIDE_PART_BONUS = {
    "Door": 1.5,
    "Fender": 1.2,
}

def normalize_part_name(part: str) -> str:
    if part in {"CAR_TRUNK-KP48", "Dickey"}:
        return "Trunk"
    return part

# =========================
# 점수 계산
# =========================
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

        if part in SIDE_PART_BONUS and region != "front":
            scores["SIDE_COLLISION"] += SIDE_PART_BONUS[part] * conf

    return dict(scores)

# =========================
# 방향 확정 가능 여부
# =========================
def can_decide_direction(scores: dict) -> bool:
    if not scores:
        return False

    values = sorted(
        v for k, v in scores.items()
        if k != MULTI_COLLISION
        for v in [v]
    )

    # 최소 점수 조건
    if not values or values[0] < 2.0:
        return False

    # 1위-2위 차이
    if len(values) >= 2 and (values[0] - values[1]) < 1.0:
        return False

    return True

# =========================
# 통합 함수 (최종)
# =========================
def estimate_accident_type(detections: list, car_count: int = 1) -> dict:
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

    # 점수 계산 (먼저!)
    scores = estimate_accident_scores(valid_detections)

    # FRONT/REAR 동시 파손 여부 (valid 기준)
    regions = {d.get("region") for d in valid_detections if d.get("region")}
    has_front_damage = "front" in regions
    has_rear_damage = "rear" in regions

    # MULTI 점수 보너스 (front + rear 동시일 때)
    if has_front_damage and has_rear_damage:
        scores[MULTI_COLLISION] = scores.get(MULTI_COLLISION, 0.0) + 3.0

    # 다중 추돌 조건
    is_multi_collision = car_count >= 2

    if "FRONT_COLLISION" in scores:
        has_strong_front = any(
            d["class_name"] in {"Bonnet", "Windshield"}
            for d in valid_detections
        )
        if not has_strong_front:
            scores["FRONT_COLLISION"] *= 0.6

    # =========================
    # 사고 상태 판단
    # =========================
    if damage_count >= 2:
        state = "CONFIRMED_ACCIDENT"
    elif damage_count == 1:
        state = "SUSPECTED_ACCIDENT"
    else:
        state = "NO_ACCIDENT"

    # =========================
    # 사고 유형 판단
    # =========================
    if state != "CONFIRMED_ACCIDENT":
        accident_type = "UNKNOWN"

    else:
        # =========================
        # SIDE 보정 규칙 (★ 추가)
        # =========================
        parts = {
            normalize_part_name(d["class_name"])
            for d in valid_detections
        }

        has_side_part = bool(parts & {"Door", "Fender"})
        has_strong_front = bool(parts & {"Bonnet", "Windshield"})
        has_strong_rear = "Trunk" in parts

        front_score = scores.get("FRONT_COLLISION", 0.0)
        rear_score = scores.get("REAR_COLLISION", 0.0)

        if (
            has_side_part
            and not has_strong_front
            and not has_strong_rear
            and front_score < 1.0
            and rear_score < 1.0
        ):
            accident_type = "SIDE_COLLISION"

        elif is_multi_collision:
            accident_type = MULTI_COLLISION

        elif can_decide_direction(scores):
            direction_scores = {
                k: v for k, v in scores.items()
                if k != MULTI_COLLISION
            }
            accident_type = max(direction_scores, key=direction_scores.get)

        else:
            accident_type = "UNKNOWN"

    # =========================
    # 신뢰도
    # =========================
    if damage_count >= 3:
        confidence_level = "HIGH"
    elif damage_count == 2:
        confidence_level = "MEDIUM"
    else:
        confidence_level = "LOW"

    # =========================
    # 메시지
    # =========================
    if state == "CONFIRMED_ACCIDENT":
        if accident_type == MULTI_COLLISION:
            message = "다수 차량이 연루된 대형 사고로 판단됩니다."
        elif accident_type != "UNKNOWN":
            message = "사고 방향이 비교적 명확하게 확인되었습니다."
        else:
            message = "사고는 확인되었으나 방향을 단정하기는 어렵습니다."
    else:
        message = "파손은 확인되었으나 사고 여부 판단에는 추가 정보가 필요합니다."

    return {
        "accident_detected": state == "CONFIRMED_ACCIDENT",
        "accident_state": state,
        "accident_type": accident_type,
        "confidence_level": confidence_level,
        "scores": scores,
        "message": message
    }
