# ============================================================
# 📄 파일명: predict.py
# 📁 위치: ai_modules/src/inference/predict.py
# 📘 목적:
#   - 학습된 가중치(.pt)를 로드하여 단일 이미지에 대해 예측을 수행하는 CLI 유틸임.
#   - 기본 모델은 SimpleCNN(입력 128×128)임.
#   - 선택적으로 idx2label JSON을 받아 클래스 인덱스를 라벨 문자열로 매핑함.
#
# 🧪 사용 예시:
#   1) 인덱스만 출력:
#      python -m ai_modules.src.inference.predict \
#        --weights runs/exp001/best.pt \
#        --image /path/sample.jpg \
#        --num-classes 492
#
#   2) 라벨 문자열까지 출력:
#      python -m ai_modules.src.inference.predict \
#        --weights runs/exp001/best.pt \
#        --image /path/sample.jpg \
#        --num-classes 492 \
#        --idx2label-json /path/matched_all.json
#
# ✅ 특징:
#   - top-5 인덱스/확률을 함께 출력함.
#   - --img-size로 입력 크기 변경 가능(기본 128).
# ============================================================

from __future__ import annotations
import argparse
from typing import Optional, Dict, Any
from PIL import Image
import torch
from torchvision import transforms

from ai_modules.src.models.simple_cnn import SimpleCNN

# idx2label JSON은 선택 사항임
def _try_load_idx2label(path: Optional[str]) -> Optional[Dict[str, str]]:
    if not path:
        return None
    try:
        from ai_modules.src.utils.idx2label import load_idx2label_from_json
        return load_idx2label_from_json(path)
    except Exception as e:
        print(f"[WARN] idx2label 로드 실패: {e}")
        return None

def _build_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

def predict_single(
    weights: str,
    image_path: str,
    num_classes: int,
    img_size: int = 128,
    idx2label_json: Optional[str] = None,
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tfm = _build_transform(img_size)

    # 모델 로드
    model = SimpleCNN(num_classes=num_classes).to(device)
    ckpt = torch.load(weights, map_location=device)
    # state_dict 또는 { "model_state_dict": ... } 모두 지원함
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    # 이미지 로드
    x = tfm(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        topk = min(5, num_classes)
        conf_vals, conf_idx = torch.topk(probs, k=topk)

    idx2label = _try_load_idx2label(idx2label_json)

    result = {
        "pred_index": int(conf_idx[0].item()),
        "pred_confidence": round(float(conf_vals[0].item()), 4),
        "topk_indices": [int(i.item()) for i in conf_idx.tolist()],
        "topk_probs":   [round(float(p.item()), 4) for p in conf_vals.tolist()],
    }

    if idx2label:
        # 문자열 키 우선, 없으면 정수 키 시도함
        def _m(i: int) -> str:
            return idx2label.get(str(i)) or idx2label.get(i) or f"{i}"
        result["pred_label"] = _m(result["pred_index"])
        result["topk_labels"] = [_m(i) for i in result["topk_indices"]]

    return result

def main():
    ap = argparse.ArgumentParser(description="SimpleCNN 단일 이미지 추론 스크립트임.")
    ap.add_argument("--weights", required=True, help="학습된 가중치 경로(.pt)")
    ap.add_argument("--image",   required=True, help="추론할 이미지 경로")
    ap.add_argument("--num-classes", type=int, required=True, help="클래스 개수")
    ap.add_argument("--img-size", type=int, default=128, help="입력 리사이즈 크기(기본 128)")
    ap.add_argument("--idx2label-json", type=str, default=None, help="선택: idx2label 매핑 JSON 경로")
    args = ap.parse_args()

    out = predict_single(
        weights=args.weights,
        image_path=args.image,
        num_classes=args.num_classes,
        img_size=args.img_size,
        idx2label_json=args.idx2label_json,
    )
    print(out)

if __name__ == "__main__":
    main()
