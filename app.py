"""
교통사고 과실비율 자동 분석 시스템 — Gradio 데모 앱
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

import cv2
import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from accident_liability.perception.detector import FasterRCNNDetector

from accident_liability.pipeline.service import LiabilityPipeline
from accident_liability.rules.adjustment import AdjustmentModel
from accident_liability.rules.base_ratio import BaseRatioLookup
from accident_liability.scene.class_maps import decode
from accident_liability.scene.video_classifier import VideoClassifier
from accident_liability.schemas import CaseInput, Evidence, SceneAnchor, TrackedObject


# ── 기본 경로 ──────────────────────────────────────────────────────

CLASSIFIER_WEIGHTS = PROJECT_ROOT / "classification" / "video_classification_outputs" / "best.pth"
DETECTOR_WEIGHTS   = PROJECT_ROOT / "detect" / "detection_outputs" / "checkpoints" / "faster_rcnn_baseline" / "best.pth"
BASE_RATIO_CSV     = PROJECT_ROOT / "data" / "lookup" / "base_ratio_table.csv"
ADJUSTMENT_MODEL   = PROJECT_ROOT / "outputs" / "adjustment" / "adjustment_model.joblib"

ACCIDENT_PLACE_OPTIONS = [
    "직선 도로", "사거리교차로(신호등 없음)", "사거리교차로(신호등 있음)",
    "T자형 교차로", "차도와 차도가 아닌 장소", "주차장(또는 차도가 아닌 장소)",
    "회전교차로", "횡단보도(신호등 없음)", "횡단보도(신호등 있음)",
    "횡단보도 없음", "횡단보도(신호등 없음) 부근", "횡단보도(신호등 있음) 부근",
    "육교 및 지하도 부근", "고속도로(자동차 전용도로)포함", "자전거 도로",
]

REPORT_SYSTEM_PROMPT = """당신은 교통사고 과실비율 산정 전문 AI 시스템입니다.
제공된 사고 분석 데이터를 바탕으로 전문적인 한국어 분석 보고서를 작성하세요.

[작성 지침]
- 전문적이고 신뢰감 있는 어조를 사용하세요
- 과실 판단의 근거를 구체적으로 설명하세요
- 관련 교통법규(도로교통법)를 적절히 언급하세요
- 마크다운 형식으로 작성하세요 (##, **bold**, - 리스트 등 활용)
- 한국어로만 작성하세요"""


# ── 모델 로드 (앱 시작 시 1회) ────────────────────────────────────

def load_models(device: str = "cpu") -> dict:
    models: dict = {}

    if CLASSIFIER_WEIGHTS.exists():
        print("[모델] 분류기 로드 중...")
        models["classifier"] = VideoClassifier(weights_path=CLASSIFIER_WEIGHTS, device=device)
        print("[모델] 분류기 OK")
    else:
        print(f"[경고] 분류기 가중치 없음: {CLASSIFIER_WEIGHTS}")

    if DETECTOR_WEIGHTS.exists():
        print("[모델] 디텍터 로드 중...")
        models["detector"] = FasterRCNNDetector(
            weights_path=DETECTOR_WEIGHTS,
            score_threshold=0.5,
            device=device,
            allowed_labels={"vehicle", "two-wheeled-vehicle", "bike"},
        )
        print("[모델] 디텍터 OK")
    else:
        print(f"[경고] 디텍터 가중치 없음: {DETECTOR_WEIGHTS}")

    if BASE_RATIO_CSV.exists() and ADJUSTMENT_MODEL.exists():
        models["pipeline"] = LiabilityPipeline(
            base_lookup=BaseRatioLookup.from_csv(BASE_RATIO_CSV),
            adjustment_model=AdjustmentModel.load(ADJUSTMENT_MODEL),
        )
        print("[모델] 파이프라인 OK")

    return models


# ── 영상 주석 처리 ─────────────────────────────────────────────────

def convert_to_h264(video_path: str | None) -> str | None:
    """업로드된 영상을 H.264로 변환 (브라우저 재생 보장)."""
    if video_path is None:
        return None
    import subprocess
    vp = Path(video_path)
    out = vp.with_name(vp.stem + "_h264.mp4")
    if out.exists():
        return str(out)
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(vp),
         "-vcodec", "libx264", "-preset", "fast", "-crf", "23",
         "-acodec", "aac", "-movflags", "faststart", str(out)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return str(out) if result.returncode == 0 else video_path


def create_annotated_video(video_path: str, detector: FasterRCNNDetector | None) -> str | None:
    if detector is None:
        return None

    vp = Path(video_path)
    cap = cv2.VideoCapture(str(vp))
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    detections_by_frame = detector.detect_video(vp)

    raw_tmp = tempfile.NamedTemporaryFile(suffix=".avi", delete=False)
    raw_path = raw_tmp.name
    raw_tmp.close()

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(raw_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        return None

    cap = cv2.VideoCapture(str(vp))
    fi  = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for det in detections_by_frame[fi] if fi < len(detections_by_frame) else []:
            x1, y1, x2, y2 = (int(v) for v in det.bbox_xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 50), 2)
            label = f"{det.label} {det.score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), (0, 200, 50), -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        writer.write(frame)
        fi += 1

    cap.release()
    writer.release()

    # ffmpeg로 H.264 재인코딩 → 브라우저 재생 보장
    import subprocess
    out_tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out_path = out_tmp.name
    out_tmp.close()
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", raw_path,
         "-vcodec", "libx264", "-preset", "fast", "-crf", "23",
         "-an", "-movflags", "faststart", out_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    os.remove(raw_path)
    return out_path if result.returncode == 0 else None


# ── LLM 보고서 생성 ────────────────────────────────────────────────

def generate_llm_report(anchor, base, adjustment, evidence, violations, api_key: str) -> str:
    from groq import Groq

    violations_a = ", ".join(violations.get("A", [])) or "없음"
    violations_b = ", ".join(violations.get("B", [])) or "없음"

    data = f"""
사고 분석 데이터:
- 사고 장소: {anchor.accident_place}
- 사고 유형(장소 특징): {anchor.accident_place_feature}
- A 차량 행동: {anchor.vehicle_a_progress_info}
- B 차량 행동: {anchor.vehicle_b_progress_info}
- 분류 신뢰도: {anchor.confidence:.1%}
- 기준 과실비율: A {base.ratio_a:.0f}% / B {base.ratio_b:.0f}%
- 가감 보정: {adjustment.adjustment:+.1f}%p
- 최종 과실비율: A {adjustment.final_ratio_a}% / B {adjustment.final_ratio_b}%
- 진입 순서: {evidence.entry_order}
- A 차량 위반사항: {violations_a}
- B 차량 위반사항: {violations_b}
"""

    client = Groq(api_key=api_key)
    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": REPORT_SYSTEM_PROMPT},
            {"role": "user", "content": data},
        ],
        temperature=0.3,
    )
    return res.choices[0].message.content


# ── 메인 분석 함수 ─────────────────────────────────────────────────

def analyze(video_path, statement, accident_place, groq_api_key, models):
    if video_path is None:
        return None, "⚠️ 영상을 먼저 업로드하세요."

    # 1. 영상 분류 → anchor
    clf: VideoClassifier | None = models.get("classifier")
    if clf is None:
        return None, "⚠️ 분류기 가중치가 없습니다."

    raw = clf.predict(video_path)
    def resolve(key):
        code = str(raw[key]["class_name"])
        return decode(key, code) or code

    anchor = SceneAnchor(
        accident_place=accident_place,
        accident_place_feature=resolve("accident_place_feature"),
        vehicle_a_progress_info=resolve("vehicle_a_progress_info"),
        vehicle_b_progress_info=resolve("vehicle_b_progress_info"),
        confidence=min(v["score"] for v in raw.values()),
    )

    # 2. 영상 주석 처리 (A/B 박스)
    annotated = create_annotated_video(video_path, models.get("detector"))

    # 3. 진술 위반사항 파싱
    violations: dict[str, list[str]] = {"A": [], "B": []}
    if statement.strip() and groq_api_key.strip():
        try:
            from accident_liability.llm.violation_parser import ViolationParser
            violations = ViolationParser(api_key=groq_api_key.strip()).parse(statement)
        except Exception as e:
            print(f"[ViolationParser 오류] {e}")

    # 4. 파이프라인 실행
    pipeline: LiabilityPipeline | None = models.get("pipeline")
    if pipeline is None:
        return annotated, "⚠️ 파이프라인 모델이 로드되지 않았습니다."

    result = pipeline.run_with_ready_inputs(
        case=CaseInput(video_path=Path(video_path), statement=statement),
        anchor=anchor,
        evidence=Evidence(entry_order="unknown"),
    )

    # 5. 보고서 생성
    if groq_api_key.strip():
        try:
            report = generate_llm_report(
                anchor, result["base"], result["adjustment"],
                result["evidence"], violations, groq_api_key.strip()
            )
        except Exception as e:
            print(f"[LLM 보고서 오류] {e}")
            report = result["report"]
    else:
        report = result["report"]

    return annotated, report


# ── Gradio UI ──────────────────────────────────────────────────────

def build_ui(models: dict):
    with gr.Blocks(title="교통사고 과실비율 분석") as demo:

        gr.Markdown("""
# 🚗 교통사고 과실비율 자동 분석 시스템
영상과 사고 정보를 입력하면 AI가 과실비율을 자동으로 분석합니다.
""")

        with gr.Row():
            # ── 입력 컬럼 ──────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### 📥 입력")

                video_input = gr.Video(label="사고 영상 업로드", height=260)

                accident_place_input = gr.Dropdown(
                    choices=ACCIDENT_PLACE_OPTIONS,
                    value="직선 도로",
                    label="사고 장소",
                )

                statement_input = gr.Textbox(
                    label="사용자 진술 (선택)",
                    placeholder="예) 상대방이 신호를 무시하고 과속으로 진입했습니다.",
                    lines=3,
                )

                groq_key_input = gr.Textbox(
                    label="Groq API Key (LLM 보고서 생성에 필요)",
                    placeholder="gsk_...",
                    type="password",
                    value=os.getenv("GROQ_API_KEY", ""),
                )

                analyze_btn = gr.Button("🔍 분석 시작", variant="primary", size="lg")

            # ── 출력 컬럼 ──────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### 📤 분석 결과")

                video_output = gr.Video(
                    label="객체 탐지 결과",
                    height=260,
                )

                report_output = gr.Markdown(
                    label="사고 분석 보고서",
                    value="분석 결과가 여기에 표시됩니다.",
                )

        video_input.upload(
            fn=convert_to_h264,
            inputs=video_input,
            outputs=video_input,
        )

        analyze_btn.click(
            fn=lambda vp, st, ap, key: analyze(vp, st, ap, key, models),
            inputs=[video_input, statement_input, accident_place_input, groq_key_input],
            outputs=[video_output, report_output],
        )

        gr.Markdown("""
---
> **A 차량** = 빨간 박스 | **B 차량** = 파란 박스
> Groq API Key 없이도 기본 템플릿 보고서를 제공합니다.
""")

    return demo


# ── 실행 ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--share", action="store_true", help="공개 URL 생성 (RunPod에서 사용)")
    args = parser.parse_args()

    models = load_models(device=args.device)
    demo   = build_ui(models)

    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        font=gr.themes.GoogleFont("Noto Sans KR"),
    )
    demo.launch(
        share=args.share,
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=False,
        theme=theme,
    )
