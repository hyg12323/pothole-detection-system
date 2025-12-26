🚗 Automobile Accident Analysis & Impact Prediction System
자동차 사고 분석 및 사고 영향 예측 시스템
1. 프로젝트 개요

본 프로젝트는 자동차 사고 이미지 및 사고 상황 데이터를 기반으로
사고 유형을 분석하고, 차량 파손 상태를 인식하며,
이를 종합해 사고 영향 및 주행 가능 여부를 예측하는
AI 기반 사고 분석 시스템을 개발하는 것을 목표로 한다.

본 시스템은 단일 모델이 아닌,
여러 개의 AI 모델을 역할별로 분리하여 구성한 것이 특징이며,
각 모델의 분석 결과를 종합하여 하나의 사고 분석 결과를 제공한다.

2. 프로젝트 전체 목표 (상위 개념)

주제

AI 기반 자동차 사고 분석 및 사고 영향 예측 시스템

시스템이 다루는 사고 분석 영역

사고 유형 분류

차량 파손 부위 탐지

사고 후 주행 가능 여부 예측

3. 시스템 내 AI 모델 구성

본 프로젝트는 총 3개의 AI 분석 모듈로 구성된다.

① 사고 유형 분류 모델

입력: 사고 차량 이미지

출력: 충돌 유형(전면 / 측면 / 후면 등)

역할: 사고 상황 이해

② 차량 파손 부위 탐지 모델 (본인이 담당한 모델)

입력: 사고 차량 이미지

출력: 파손 부위(Bumper, Door, Light 등) + Bounding Box

역할: 차량 손상 상태 정밀 분석

③ 주행 가능 여부 예측 모델

입력: 사고 유형 + 파손 부위 정보

출력: 주행 가능 / 점검 필요 / 운행 불가

역할: 사고 영향 판단

4. 내가 담당한 영역 (핵심)
🔹 차량 파손 부위 탐지 AI 모델

본인은 본 프로젝트에서
YOLO 기반 차량 파손 부위 탐지 모델의 설계·학습·서빙을 담당하였다.

주요 구현 내용

YOLO 기반 객체 탐지 모델 학습

차량 파손 부위 8개 클래스 탐지

클래스별 Confidence Threshold 적용

파손 부위 Bounding Box 시각화

FastAPI 기반 AI 추론 서버 구현

Docker 기반 모델 서버 컨테이너화

5. 기술 스택 (Tech Stack)
🔹 AI / Model Server

Language: Python

Model: YOLO (Object Detection)

Framework: FastAPI

Libraries: Ultralytics, OpenCV, NumPy

🔹 Web Backend

Language: Java

Framework: Spring Boot

Server: Tomcat

🔹 Frontend

Framework: Vue.js

🔹 Database

DB: MongoDB

🔹 DevOps

Containerization: Docker

6. 시스템 아키텍처
[Client (Browser)]
        ↓
[Vue Frontend]
        ↓
[Java / Spring]
        ↓
 ┌─────────────── AI Analysis Layer ───────────────┐
 │ ① 사고 유형 분류 모델                          │
 │ ② 차량 파손 부위 탐지 모델 (YOLO)               │
 │ ③ 주행 가능 여부 예측 모델                      │
 └─────────────────────────────────────────────────┘
        ↓
[MongoDB]

7. 사고 파손 탐지 모델 동작 흐름

사용자가 사고 차량 이미지 업로드

Java 서버가 이미지를 AI 서버로 전달

YOLO 모델이 파손 부위 탐지 수행

파손 부위 Bounding Box 및 Confidence 반환

결과를 상위 사고 분석 로직에 전달

8. 프로젝트의 특징

사고 분석을 단일 모델이 아닌 다중 AI 모델 구조로 설계

AI 모델과 웹 서버 완전 분리

REST API 기반 모델 서빙

Docker 기반 재현 가능한 실행 환경

확장 가능한 사고 분석 파이프라인

9. 향후 확장 방향

파손 부위 조합 기반 수리 비용 예측

GPT API 연동을 통한 사고 분석 설명 자동 생성

사고 데이터 누적 기반 통계 분석

클라우드 환경 배포 및 확장

10. 요약

본 프로젝트는
자동차 사고를 다각도로 분석하는 AI 기반 사고 분석 시스템이며,
그중 차량 파손 부위 탐지 모델은 핵심 분석 모듈 중 하나이다.
