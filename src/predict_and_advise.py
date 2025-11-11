# ============================================================
# 📄 파일명: predict_and_advise.py
# 📁 위치: ai_modules/src/predict_and_advise.py
# 📘 목적:
#   - 경량 LightCNN(64x64)으로 단일 이미지 분류 후,
#     선택적으로 LLM을 호출해 임산부 복용 가능 여부 안내문을 생성함.
#
# 🧪 사용 예시:
#   # 예측만
#   python -m ai_modules.src.predict_and_advise \
#     --weights /path/best_model.pth \
#     --image   /path/sample.jpg \
#     --idx2label-json /path/matched_all.json \
#     --img-size 64
#
#   # 예측 + LLM 자문
#   python -m ai_modules.src.predict_and_advise \
#     --weights /path/best_model.pth \
#     --image   /path/sample.jpg \
#     --idx2label-json /path/matched_all.json \
#     --img-size 64 --ask-llm --openai-model gpt-4o
#
# ✅ 특징:
#   - 체크포인트가 state_dict 또는 {"model_state_dict": ...} 모두 지원됨.
#   - idx2label JSON이 없으면 라벨 문자열 없이 인덱스만 반환함.
# ============================================================

from __future__ import annotations
import argparse, os
from typing import Dict, Any, Optional
from PIL import Image
import torch
from torchvision import transforms

from ai_modules.src.models.model_lightcnn import LightCNN
from ai_modules.src.utils.idx2label import load_idx2label_from_json, map_index
from ai_modules.src.services.pregnancy_advice import ask_pregnancy_safety

def _build_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

def _load_model(weights: str, num_classes: int, device: torch.device) -> LightCNN:
    model = LightCNN(num_classes=num_classes).to(device)
    ckpt = torch.load(weights, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model

@torch.no_grad()
def run_once(
    weights: str,
    image: str,
    img_size: int = 64,
    idx2label_json: Optional[str] = None,
    num_classes: Optional[int] = None,
    ask_llm: bool = False,
    openai_model: Optional[str] = None,
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tfm = _build_transform(img_size)

    # idx2label 로드(선택)
    idx2label = None
    if idx2label_json:
        try:
            idx2label = load_idx2label_from_json(idx2label_json)
        except Exception as e:
            print(f"[WARN] idx2label 로드 실패: {e}")

    # 클래스 수 결정
    inferred_nc = num_classes or (len(idx2label) if idx2label else None)
    if inferred_nc is None:
        raise ValueError("num_classes를 지정하거나 idx2label_json을 제공해야 함.")

    # 모델 로드 및 추론
    model = _load_model(weights, inferred_nc, device)
    x = tfm(Image.open(image).convert("RGB")).unsqueeze(0).to(device)
    logits = model(x)[0]
    probs = torch.softmax(logits, dim=0)
    pred_idx = int(torch.argmax(probs).item())
    conf = float(probs[pred_idx].item())

    out: Dict[str, Any] = {
        "pred_index": pred_idx,
        "confidence": round(conf, 4),
    }

    if idx2label:
        out["pred_label"] = map_index(idx2label, pred_idx)

    # LLM 자문(선택)
    if ask_llm:
        if openai_model:
            os.environ["OPENAI_MODEL"] = openai_model
        pill_name = out.get("pred_label", f"Label-{pred_idx}")
        out["llm_advice"] = ask_pregnancy_safety(pill_name)

    return out

def main():
    ap = argparse.ArgumentParser(description="LightCNN 예측 + 임산부 복용 자문(선택) 통합 스크립트임.")
    ap.add_argument("--weights", required=True, help="모델 가중치(.pt)")
    ap.add_argument("--image",   required=True, help="추론할 이미지 경로")
    ap.add_argument("--img-size", type=int, default=64, help="입력 크기(기본 64)")
    ap.add_argument("--idx2label-json", type=str, default=None, help="라벨 매핑 JSON(선택)")
    ap.add_argument("--num-classes", type=int, default=None, help="클래스 수(미지정 시 idx2label 길이)")

    ap.add_argument("--ask-llm", action="store_true", help="LLM 자문 실행")
    ap.add_argument("--openai-model", type=str, default=None, help="예: gpt-4, gpt-4o, gpt-4o-mini")
    args = ap.parse_args()

    out = run_once(
        weights=args.weights,
        image=args.image,
        img_size=args.img_size,
        idx2label_json=args.idx2label_json,
        num_classes=args.num_classes,
        ask_llm=args.ask_llm,
        openai_model=args.openai_model,
    )
    print(out)

if __name__ == "__main__":
    main()
