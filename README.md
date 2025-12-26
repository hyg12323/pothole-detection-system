🚗 Automobile Accident Analysis System

자동차 사고 분석 및 사고 영향 예측 시스템

프로젝트 개요

본 프로젝트는 자동차 사고 이미지를 기반으로
사고 상태를 분석하고 사고 영향을 예측하는 AI 기반 사고 분석 시스템이다.

사고 유형 분석, 차량 파손 부위 탐지, 주행 가능 여부 판단 등
여러 AI 모델의 결과를 종합하여 사고 상태를 해석한다.

시스템 구성

사고 유형 분류 모델

차량 파손 부위 탐지 모델 (YOLO 기반)

주행 가능 여부 예측 모델

담당 역할
차량 파손 부위 탐지 AI 모델

YOLO 기반 파손 부위 탐지

Bounding Box 시각화

Confidence Threshold 적용

FastAPI 기반 추론 서버

Docker 컨테이너화

기술 스택

AI: Python, YOLO, FastAPI

Backend: Java, Spring Boot

Frontend: Vue.js

DB: MongoDB

DevOps: Docker

특징

다중 AI 모델 기반 사고 분석

AI 서버 · 웹 서버 분리 구조

REST API 기반 AI 서빙
