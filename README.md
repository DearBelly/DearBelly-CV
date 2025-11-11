# 🩹 Wound Analysis Module 

## 📘 개요

본 리포지토리는 DearBelly의 약 이미지 인식 기반의 임산부 복용 자문 시스템을 구현한 것입니다.
경구약 이미지를 분류하여 약품명을 식별하고, 그 결과를 바탕으로 LLM을 통해 임산부 복용 가능 여부와 주의사항을 안내합니다.
전체 파이프라인은 데이터 전처리 → 모델 학습(SimpleCNN) → 추론(SimpleCNN / LightCNN) → LLM 자문(OpenAI API) 순으로 구성됩니다.

이미지 데이터셋 전처리부터 CNN 학습, 추론, 유틸리티 테스트까지 하나의 구조로 통합되어 있으며,  
Google Colab / 로컬 환경에서 동일하게 재현 가능합니다.

---

## 🧠 주요 기능

| 구분 | 설명 |
|------|------|
| **데이터 전처리(data_prep)** | ZIP 자동 해제, 이미지 개수 검사, 중앙 크롭 및 리사이즈 자동화 |
| **데이터셋 로더(dataset_pill.py)** | 이미지-라벨 매핑 자동 생성 및 PyTorch Dataset 객체 생성 |
| **모델(SimpleCNN)** | 경량 CNN(2 conv + 2 fc) 기반 약 이미지 분류 모델 |
| **학습(train.py)** | YAML 설정 기반 학습 루프 (train/val 분할, 체크포인트 저장) |
| **추론(predict.py)** | 학습된 모델로 단일 이미지 예측 수행 |
| **테스트(test_training_smoke.py)** | 모델 forward 검증용 스모크 테스트 |
| **유틸(seed, paths)** | 시드 고정, 경로 관리 클래스 제공 |

---

## 🧩 프로젝트 구조

    ai_modules/
    ├── src/
    │   ├── data_prep/
    │   │   ├── dataset_pill.py
    │   │   ├── extract_archives.py
    │   │   ├── count_images.py
    │   │   └── center_crop_resize.py
    │   │
    │   ├── models/
    │   │   ├── simple_cnn.py
    │   │   └── model_lightcnn.py
    │   │
    │   ├── inference/
    │   │   └── predict.py
    │   │
    │   ├── services/
    │   │   └── pregnancy_advice.py
    │   │
    │   ├── utils/
    │   │   ├── seed.py
    │   │   ├── paths.py
    │   │   └── idx2label.py
    │   │
    │   ├── train.py
    │   └── predict_and_advise.py
    │
    ├── configs/
    │   └── baseline.yaml
    │
    ├── tests/
    │   └── test_training_smoke.py
    │
    ├── README.md
    ├── requirements.txt
    ├── .gitignore
    ├── .gitattributes
    └── __init__.py


---

## ⚙️ 설치 및 실행

### 1️⃣ 필수 라이브러리 설치

    pip install -r requirements.txt

---

### 2️⃣ 데이터 전처리
압축된 TS/TL 데이터를 자동으로 해제하고 크롭/리사이즈를 수행함.

    # 압축 해제
    python -m ai_modules.wound_analysis.src.data_prep.extract_archives \
        --img-zip-base "/path/to/원천데이터/단일경구약제 5000종" \
        --lbl-zip-base "/path/to/라벨링데이터/단일경구약제 5000종" \
        --targets 39,41,42,43,46,48,51,54

    # 이미지 개수 확인
    python -m ai_modules.wound_analysis.src.data_prep.count_images \
        --root "/path/to/TS_57_단일"

    # 중앙 크롭 후 리사이즈
    python -m ai_modules.wound_analysis.src.data_prep.center_crop_resize \
        --input  "/path/to/TS_54_단일" \
        --output "/path/to/TS_54_단일crop128" \
        --crop-size 512 --resize-size 128

---

### 3️⃣ 학습 실행

    python -m ai_modules.wound_analysis.src.train \
        --config ai_modules/wound_analysis/configs/baseline.yaml

**baseline.yaml 예시**

    image_root: "/content/gdrive/MyDrive/.../TS_81_단일crop128"
    label_root: "/content/gdrive/MyDrive/.../TL_81_단일"
    label_key: "dl_name"

    img_size: 128
    batch_size: 32
    epochs: 5
    lr: 0.001
    save_dir: "runs/exp001"
    seed: 42

---

### 4️⃣ 추론 실행

    python -m ai_modules.wound_analysis.src.inference.predict \
        --weights runs/exp001/best.pt \
        --image path/to/sample.jpg \
        --num-classes 492 \
        --img-size 128

출력 예시:

    {'pred_index': 27, 'probs_top5': [0.99, 0.87, 0.12, 0.08, 0.05]}

---

## 🧾 라이선스

본 모듈의 코드는 학습 및 개인 프로젝트 목적으로 공개되며,  
실제 의료 진단/처방에는 사용할 수 없음.

    © 2025 DearBelly Project (Mom4U)
    Author: hjjummy ,sangeun