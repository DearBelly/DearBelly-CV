# 🩹 Pill Analysis Module

## 📘 개요

본 리포지토리는 DearBelly의 약 이미지 인식 기반의 임산부 복용 자문 시스템을 구현한 것입니다.
경구약 이미지를 분류하여 약품명을 식별하고, 그 결과를 바탕으로 LLM을 통해 임산부 복용 가능 여부와 주의사항을 안내합니다.
전체 개발 파이프라인은 데이터 전처리 → 데이터 증강 → 모델 학습(SimpleCNN/LightCNN/EfficientNet-B3) → 추론 → LLM 자문(OpenAI API) 순으로 구성하였습니다.

이미지 데이터셋 전처리부터 CNN 학습, 추론, 유틸리티 테스트까지 하나의 구조로 통합되어 있으며,  
Google Colab / 로컬 환경에서 동일하게 재현 가능합니다.

---

## 🧠 주요 기능

| 구분 | 설명 |
|------|------|
| **데이터 전처리(data_prep)** | ZIP 해제, 이미지 개수 검사, 중앙 크롭 및 리사이즈 자동화 |
| **데이터 증강(data_augmentation)** | 노이즈 추가, Shear, 밝기 조절 등 이미지 다양화 기능 |
| **데이터셋 로더(dataset_precomputed.py)** | JSON 기반 이미지·라벨 매핑 자동 생성 및 PyTorch Dataset 구성 |
| **모델(models)** | SimpleCNN, LightCNN, EfficientNet-B3 등 다중 백본 모델 지원 |
| **학습(trainers)** | LightCNN / EfficientNet / TIMM 백본별 학습 루프 및 ArcFace, Mixup 옵션 제공 |
| **옵티마이저(optimizer)** | SGD / Momentum / Adam 등 비교 실험용 최적화 모듈 포함 |
| **추론(inference/predict.py)** | 학습된 모델로 단일 이미지 예측 및 Top-k 확률 출력 |
| **통합 실행(predict_and_advise.py)** | 이미지 추론 후 LLM 기반 복용 자문까지 한 번에 실행 |
| **서비스(services/pregnancy_advice.py)** | LLM(OpenAI API) 기반 임산부 복용 가능 여부 및 주의사항 안내 |
| **테스트(tests/test_training_smoke.py)** | 모델 학습 및 추론 스모크 테스트 |
| **유틸(utils)** | 시드 고정(seed), 경로 관리(paths), 라벨 매핑(idx2label), EarlyStopping 등 공통 유틸 |

---

## 🧩 프로젝트 구조
```markdown
    ai_modules/
        ├── src/
        │   ├── data_prep/
        │   │   ├── dataset_precomputed.py        # JSON 기반 데이터셋 로더 (image_path + label)
        │   │   ├── extract_archives.py           # ZIP 압축 자동 해제 스크립트
        │   │   ├── count_images.py               # 폴더별 이미지 개수 검사
        │   │   └── center_crop_resize.py         # 중앙 크롭 및 리사이즈 자동화
        │   │
        │   ├── data_augmentation/                # 데이터 증강 (개별 실행형)
        │   │   ├── add_noise.py                  # 가우시안 노이즈 추가
        │   │   ├── shear_images.py               # Shear(기울이기) 변형
        │   │   └── adjust_brightness.py          # 밝기 조절
        │   │
        │   ├── models/
        │   │   ├── simple_cnn.py                 # 2 conv + 2 fc 기반 경량 CNN
        │   │   ├── model_lightcnn.py             # LightCNN (AdaptiveAvgPool 포함)
        │   │   └── efficientnet_baseline.py      # EfficientNet-B3 백본 모델
        │   │
        │   ├── trainers/
        │   │   ├── train_light_cnn.py            # LightCNN 학습/평가 루프
        │   │   ├── train_efficientnet_baseline.py# EfficientNet-B3 베이스라인 학습 스크립트
        │   │   └── train_timm.py                 # TIMM 백본 학습 (ArcFace/Mixup 옵션 지원)
        │   │
        │   ├── optimizer/
        │   │   ├── __init__.py
        │   │   ├── optim_experiment.py           # run_experiment_for, plot_from_csvs 등 공통 로직
        │   │   └── main_lightcnn_optim.py        # SGD/Momentum/Adam 비교 실행 엔트리
        │   │
        │   ├── inference/
        │   │   └── predict.py                    # 단일 이미지 추론 (Top-k 결과 출력)
        │   │
        │   ├── services/
        │   │   └── pregnancy_advice.py           # LLM 기반 임산부 복용 자문 모듈 (OpenAI API)
        │   │
        │   ├── utils/
        │   │   ├── seed.py                       # 시드 고정 유틸
        │   │   ├── paths.py                      # 경로 관리 클래스
        │   │   ├── idx2label.py                  # 라벨 매핑 유틸
        │   │   └── early_stopping.py             # EarlyStopping 클래스
        │   │
        │   ├── predict_and_advise.py             # CNN 추론 + LLM 자문 통합 실행 스크립트
        │   └── README.md                         # 서브모듈용 설명 문서
        │
        ├── configs/
        │   └── baseline.yaml                     # 학습 기본 설정 (경로, 배치, 에폭, 러닝레이트 등)
        │
        ├── tests/
        │   └── test_training_smoke.py            # 모델 학습 검증용 스모크 테스트
        │
        ├── README.md                             # 리포지토리 전체 문서 (본 파일)
        ├── requirements.txt                      # 의존 패키지 리스트
        ├── .gitignore                            # 무시 설정
        └── __init__.py                          
```
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