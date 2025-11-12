# 🧠 Optimizer Benchmarks (LightCNN)

LightCNN(64×64) 기반의 경량 CNN 모델에서 **SGD / Momentum / Adam** 옵티마이저를 동일 조건에서 비교하는 실험 모듈임.  
데이터셋 로드 → 학습/검증/얼리스탑 → 테스트 → CSV·로그·모델 저장 → 비교 그래프 출력 순으로 구성되어 있음.

---

## ⚙️ 주요 구성 요소

| 구분 | 설명 |
|------|------|
| `optim_experiment.py` | 학습, 검증, 테스트, CSV 저장, 로그, 얼리스탑, 그래프 출력 등 전체 로직 포함 |
| `main_lightcnn_optim.py` | CLI(명령행) 엔트리. Argparse로 JSON 경로·파라미터 입력 후 실행 |
| `__init__.py` | 패키지 초기화 및 함수 export |
| `README.md` | 폴더 설명 및 실행 가이드 |

---

## 📦 의존 모듈

| 모듈 경로 | 역할 |
|------------|------|
| `ai_modules/src/data_prep/dataset_precomputed.py` | `PrecomputedPillDataset` 정의 |
| `ai_modules/src/models/model_lightcnn.py` | `LightCNN` 모델 정의 |
| `ai_modules/src/utils/early_stopping.py` | EarlyStopping 클래스 |
| `ai_modules/src/utils/seed.py` | 시드 고정 함수 (`set_seed`) |

## 🚀 실행 예시

```bash
python -m ai_modules.src.optimizer.main_lightcnn_optim \
  --train_json "/content/gdrive/MyDrive/Matched/matched_train_90_original_noisy_sheared_bright.json" \
  --test_json1 "/content/gdrive/MyDrive/Matched/fortest.json" \
  --test_json2 "/content/gdrive/MyDrive/Matched/matched_test_18_deduped.json" \
  --save_dir   "/content/gdrive/MyDrive/ModelCheckpoints2" \
  --img_size 64 --batch_size 32 --num_workers 0 --plot