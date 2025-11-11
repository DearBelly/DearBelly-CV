# ============================================================
# 📄 파일명: pregnancy_advice.py
# 📁 위치: ai_modules/src/services/pregnancy_advice.py
# 📘 목적:
#   - (정보 제공용) 약 이름을 입력받아 임산부 복용 가능 여부에 대해
#     LLM(OpenAI)을 통해 간결한 안내문을 생성하는 유틸임.
#
# 🔑 환경변수:
#   - OPENAI_API_KEY : OpenAI API 키
#   - OPENAI_MODEL   : 기본 모델명(없으면 "gpt-4" 사용)
#
# 🧪 사용 예시:
#   from ai_modules.src.services.pregnancy_advice import ask_pregnancy_safety
#   text = ask_pregnancy_safety("한림모사프리드정5밀리그램")
#   print(text)
#
# ⚠️ 주의:
#   - 본 답변은 의학적 진단/처방이 아님. 실제 복용은 반드시 의료진과 상의해야 함.
# ============================================================

from __future__ import annotations
import os
from typing import Optional
from openai import OpenAI

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")

def ask_pregnancy_safety(
    pill_name: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 600,
) -> str:
    api_key = api_key or os.getenv("OPENAI_API_KEY", "")
    model = model or DEFAULT_MODEL
    if not api_key:
        return ("OPENAI_API_KEY가 설정되지 않아 LLM 질의를 생략함.\n"
                "※ 본 기능은 정보 제공용이며, 실제 복용 여부는 의료진과 상의가 필요함.")

    client = OpenAI(api_key=api_key)

    prompt = (
        f"약 이름: {pill_name}\n"
        "질문: 이 약은 임산부가 복용해도 안전한가?\n"
        "요청: 공적 가이드라인 중심으로 한국어로 간략히 정리하고, 문장마다 줄바꿈을 넣을 것. "
        "마지막에 반드시 '의료진과 상의 필요' 문구를 포함할 것. "
        "과도한 확신이나 단정적 표현은 피할 것."
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""
