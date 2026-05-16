from __future__ import annotations

import json
import os
from typing import Any


ALLOWED_VIOLATIONS = [
    "10km/h이상 20km/h 미만의 제한속도 위반",
    "20km/h 이상의 제한속도 위반",
    "일방통행 위반",
    "졸음운전",
    "음주운전(혈중농도 0.03%미만)",
    "음주운전(혈중농도 0.03%이상)",
    "마약 등의 약물운전",
    "무면허운전",
    "차량 유리의 암도가 높은 경우",
    "운전 중 휴대전화 사용",
    "운전 중 영상표시장치 시청,조작",
    "서행",
]


SYSTEM_PROMPT = """당신은 교통사고 과실 비율 산정을 위해 사용자의 진술을 분석하는 AI입니다.
사용자의 진술을 읽고, A차량(상대차량)과 B차량(블랙박스 차량, 즉 사용자)의 위반 사항을 아래 [허용된 위반 리스트]에서만 찾아 JSON으로만 출력하세요.

[출력 규칙]
1. JSON만 출력하세요.
2. 설명, 부연, 인사말을 절대 출력하지 마세요.
3. 각 차량에 해당 위반이 없으면 반드시 빈 리스트 []를 반환하세요.
4. 허용된 위반 리스트에 없는 항목은 절대 생성하지 마세요.
5. 명확한 근거가 없으면 절대 추측하지 마세요.

[차량 구분 규칙]
- B차량: "나", "저", "저는", "내 차", "제 차", "블랙박스 차량", "사용자"
- A차량: "상대", "상대방", "상대 차량", "상대차"
- 한 문장에서 주어가 한 번 정해지면, 이후 별도 주어 변경이 없는 한 같은 차량의 행동으로 판단하세요.

[속도 위반 판단 규칙]
1. 초과속도가 직접 주어지면 그 값을 사용하세요.
2. 제한속도와 실제속도가 같이 주어지면 초과속도 = 실제속도 - 제한속도 로 계산하세요.
3. 초과속도가 10 이상 20 미만이면 "10km/h이상 20km/h 미만의 제한속도 위반"입니다.
4. 초과속도가 20 이상이면 "20km/h 이상의 제한속도 위반"입니다.
5. 초과속도가 10 미만이면 속도위반으로 판단하지 마세요.

[음주운전 판단 규칙]
1. 혈중 알코올 농도 또는 혈중농도가 숫자로 명시된 경우에만 판단하세요.
2. 혈중농도가 0.03 미만이면 "음주운전(혈중농도 0.03%미만)"입니다.
3. 혈중농도가 0.03 이상이면 "음주운전(혈중농도 0.03%이상)"입니다.
4. 숫자 없이 "술을 마셨다", "음주 상태였다"만 있으면 음주운전으로 판단하지 마세요.

[기타 매핑 규칙]
- "휴대전화", "핸드폰", "스마트폰", "전화" 사용 → "운전 중 휴대전화 사용"
- "영상통화", "화상통화", "영상전화" → "운전 중 영상표시장치 시청,조작"

[허용된 위반 리스트]
- 10km/h이상 20km/h 미만의 제한속도 위반
- 20km/h 이상의 제한속도 위반
- 일방통행 위반
- 졸음운전
- 음주운전(혈중농도 0.03%미만)
- 음주운전(혈중농도 0.03%이상)
- 마약 등의 약물운전
- 무면허운전
- 차량 유리의 암도가 높은 경우
- 운전 중 휴대전화 사용
- 운전 중 영상표시장치 시청,조작
- 서행
"""


class ViolationParser:
    def __init__(self, provider: str = "groq", model: str = "llama-3.1-8b-instant", api_key: str | None = None):
        self.provider = provider
        self.model = model
        self.api_key = api_key or os.getenv("GROQ_API_KEY")

    def parse(self, statement: str) -> dict[str, list[str]]:
        if not statement.strip():
            return {"A": [], "B": []}
        if self.provider != "groq":
            raise ValueError(f"Unsupported provider: {self.provider}")
        try:
            from groq import Groq
        except ImportError as exc:
            raise ImportError("Install LLM extras: pip install -e '.[llm]'") from exc
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is required for Groq violation parsing")

        client = Groq(api_key=self.api_key)
        completion = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"사용자 진술: {statement}"},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        return normalize_result(json.loads(completion.choices[0].message.content))


def normalize_result(result: dict[str, Any]) -> dict[str, list[str]]:
    normalized = {"A": [], "B": []}
    allowed = set(ALLOWED_VIOLATIONS)
    for actor in ["A", "B"]:
        values = result.get(actor, [])
        if not isinstance(values, list):
            values = []
        normalized[actor] = sorted({str(v) for v in values if str(v) in allowed})
    return normalized
