"""
data/VL_차대차_영상_직선도로/ 의 라벨 JSON과
data/VS_차대차_영상_직선도로/ 의 영상을 매칭해서
build_adjustment_csv.py 용 input JSON 생성.

출력 형식:
[
  {
    "video_path": "data/VS_.../xxx.mp4",
    "accident_place": "직선 도로",
    "true_ratio_a": 30
  },
  ...
]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ACCIDENT_PLACE = {
    0:  "직선 도로",
    1:  "사거리교차로(신호등 없음)",
    2:  "사거리교차로(신호등 있음)",
    3:  "T자형 교차로",
    4:  "차도와 차도가 아닌 장소",
    5:  "주차장(또는 차도가 아닌 장소)",
    6:  "회전교차로",
    7:  "횡단보도(신호등 없음)",
    8:  "횡단보도(신호등 있음)",
    9:  "횡단보도 없음",
    10: "횡단보도(신호등 없음) 부근",
    11: "횡단보도(신호등 있음) 부근",
    12: "육교 및 지하도 부근",
    13: "고속도로(자동차 전용도로)포함",
    14: "자전거 도로",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="라벨 JSON → build_adjustment_csv 입력 JSON 생성")
    parser.add_argument("--label_dir", type=Path,
                        default=PROJECT_ROOT / "data" / "VL_차대차_영상_직선도로")
    parser.add_argument("--video_dir", type=Path,
                        default=PROJECT_ROOT / "data" / "VS_차대차_영상_직선도로")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "data" / "adjustment_input.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    video_map = {p.stem: p for p in args.video_dir.glob("*.mp4")}
    label_files = sorted(args.label_dir.glob("*.json"))

    print(f"라벨 JSON: {len(label_files)}개")
    print(f"영상 파일: {len(video_map)}개")

    items = []
    skipped = 0

    for label_path in label_files:
        stem = label_path.stem
        video_path = video_map.get(stem)

        if video_path is None:
            print(f"  [skip] 영상 없음: {stem}")
            skipped += 1
            continue

        with open(label_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        video_info = data.get("video", data)
        place_idx = video_info.get("accident_place")
        true_ratio_a = video_info.get("accident_negligence_rateA")

        if place_idx is None or true_ratio_a is None:
            print(f"  [skip] 필드 누락: {stem}")
            skipped += 1
            continue

        accident_place = ACCIDENT_PLACE.get(int(place_idx))
        if accident_place is None:
            print(f"  [skip] 알 수 없는 accident_place 인덱스 {place_idx}: {stem}")
            skipped += 1
            continue

        items.append({
            "video_path": str(video_path),
            "accident_place": accident_place,
            "true_ratio_a": int(true_ratio_a),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"\n완료: {len(items)}개 저장 → {args.output}")
    if skipped:
        print(f"스킵: {skipped}개")


if __name__ == "__main__":
    main()
