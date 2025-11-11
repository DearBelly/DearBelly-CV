# ============================================================
# 📄 파일명: idx2label.py
# 📁 위치: ai_modules/src/utils/idx2label.py
# 📘 목적:
#   - idx↔label 매핑을 JSON에서 불러오거나, 없을 경우 자동 생성하는 유틸임.
#   - 분류 결과의 인덱스를 사람이 읽을 수 있는 라벨 문자열로 변환하는 데 사용함.
#
# 🔎 기대 JSON 형태 예시:
#   {
#     "idx2label": { "0": "K-000001", "1": "K-000002", ... },
#     "samples": [
#       {"path": "...", "label": 38954},
#       {"path": "...", "label": 12685}
#     ]
#   }
#
# 🧪 사용 예시:
#   from ai_modules.src.utils.idx2label import load_idx2label_from_json, map_index
#   mapping = load_idx2label_from_json("/path/matched_all.json")
#   label = map_index(mapping, 27)   # -> "K-000027" 또는 사전에 정의된 라벨
#
# ✅ 특징:
#   - "idx2label" 키가 있으면 그대로 사용함.
#   - 없으면 samples[].label을 기반으로 정렬 후 자동 생성함.
#   - 키가 문자열/정수 혼재여도 안전하게 접근하도록 보조 함수 제공함.
# ============================================================

from __future__ import annotations
from typing import Dict, Any, Optional
import json

def load_idx2label_from_json(json_path: str) -> Dict[str, str]:
    """
    JSON에서 idx2label 매핑을 불러오되, 없으면 samples[].label 기반으로 자동 생성함.
    자동 생성 시 정렬된 순서로 "K-<6자리>" 포맷을 기본으로 부여함.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data: Any = json.load(f)

    idx2label = data.get("idx2label")
    if idx2label:
        # 키/값을 모두 문자열화하여 일관성 보장
        return {str(k): str(v) for k, v in idx2label.items()}

    samples = data.get("samples", [])
    if not samples:
        raise ValueError("samples가 비어 있어 idx2label을 생성할 수 없음임.")

    # label 필드 수집 후 중복 제거 + 정렬
    uniq = sorted({str(s.get("label")) for s in samples if s.get("label") is not None})

    # "K-<6자리>" 기본 포맷. label이 숫자형 문자열이면 포맷 적용, 아니면 원문 유지함.
    gen: Dict[str, str] = {}
    for i, lbl in enumerate(uniq):
        if lbl.isdigit():
            gen[str(i)] = f"K-{int(lbl):06d}"
        else:
            gen[str(i)] = lbl
    return gen

def map_index(idx2label: Dict[str, str], index: int) -> str:
    """
    정수 인덱스를 라벨 문자열로 안전하게 매핑함.
    문자열 키 우선 조회 → 정수 키 시도 → 실패 시 원본 인덱스 반환함.
    """
    return idx2label.get(str(index)) or idx2label.get(index) or f"{index}"

def map_indices(idx2label: Dict[str, str], indices: list[int]) -> list[str]:
    """여러 인덱스를 일괄 매핑하여 문자열 리스트로 반환함."""
    return [map_index(idx2label, i) for i in indices]
