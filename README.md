🚗 Automobile Accident Analysis System

자동차 사고 분석 및 사고 영향 예측 시스템

1. 프로젝트 개요

본 프로젝트는 자동차 사고 이미지를 기반으로
사고를 다각도로 분석하고 사고 영향을 예측하는 AI 기반 사고 분석 시스템이다.

시스템은 단일 모델이 아닌,
사고 유형 분석 · 차량 파손 탐지 · 주행 가능 여부 판단 등
여러 AI 모델의 결과를 종합하여 사고 상태를 해석하는 구조로 설계되었다.

2. 시스템 구성 (요약)

본 프로젝트는 다음과 같은 AI 분석 모듈로 구성된다.

사고 유형 분류 모델

차량 파손 부위 탐지 모델 (YOLO 기반)

주행 가능 여부 예측 모델

각 모델은 독립적으로 동작하며,
웹 서버를 통해 통합된 사고 분석 결과를 제공한다.

3. 내가 담당한 핵심 기능
🔹 차량 파손 부위 탐지 AI 모델

본인은 프로젝트에서 차량 파손 부위 탐지 모델을 담당하였다.

YOLO 기반 객체 탐지 모델 학습

차량 파손 부위 8개 클래스 탐지

Bounding Box 시각화

클래스별 Confidence Threshold 적용

FastAPI 기반 AI 추론 서버 구현

Docker 기반 모델 서버 컨테이너화

4. 기술 스택

AI / Model Server: Python, YOLO, FastAPI

Web Backend: Java, Spring Boot

Frontend: Vue.js

Database: MongoDB

DevOps: Docker

5. 주요 기능 흐름 (간단)

사고 차량 이미지 업로드

AI 모델을 통한 파손 부위 탐지

탐지 결과(JSON 및 시각화 이미지) 반환

상위 사고 분석 로직에서 활용

6. 프로젝트 특징

사고 분석을 위한 다중 AI 모델 구조

AI 서버와 웹 서버 분리 설계

REST API 기반 AI 서빙

Docker 기반 재현 가능한 실행 환경

7. 향후 확장 방향

사고 영향 예측 고도화

수리 비용 예측 모델 추가

GPT API 연동 사고 분석 설명 생성

클라우드 환경 배포
