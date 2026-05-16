# Accident Liability AI

블랙박스 영상과 사용자 진술을 입력으로 받아 사고 유형/장소 후보, 궤적 evidence, 기준 과실비율, 가감 사유, 최종 과실비율 리포트를 만드는 MVP 코드베이스입니다.

## Recommended Directory Structure

```text
accident-liability-ai/
  configs/                 # YAML 설정
  data/
    raw/                   # 원본 AIHub/실제 사고 데이터
    processed/             # 전처리 산출물
    lookup/                # base ratio table, class maps, adjustment table
  scripts/                 # CLI entrypoints
  src/accident_liability/
    scene/                 # VTN 기반 case anchor
    data/                  # AIHub sample collection, datasets
    models/                # VTN 등 모델 정의
    training/              # train/eval loop
    perception/            # YOLO, A/B assignment, Norfair tracking
    trajectory/            # bbox track -> evidence
    rules/                 # base ratio lookup, residual adjustment
    llm/                   # 사용자 진술 violation parser
    report/                # 리포트 생성
    pipeline/              # end-to-end orchestration
  tests/
```

## Where Your Current Files Went

- `train (1).py`
  - `src/accident_liability/data/aihub.py`
  - `src/accident_liability/data/video_dataset.py`
  - `src/accident_liability/models/vtn.py`
  - `src/accident_liability/training/loops.py`
  - `scripts/train_vtn.py`

- `AdjustmentTable_ResidualTraining_synthetic_demo...ipynb`
  - synthetic demo generation은 제외했습니다.
  - 실제 입력 CSV 기반 lookup/residual 학습으로 `src/accident_liability/rules/base_ratio.py`, `src/accident_liability/rules/adjustment.py`에 반영했습니다.

- `accidentmap (1).ipynb`
  - Groq/Llama 호출과 JSON 방어 로직을 `src/accident_liability/llm/violation_parser.py`로 옮겼습니다.

## Data Contracts

### Base Ratio CSV

`data/lookup/base_ratio_table.csv`

```csv
accident_place,accident_place_feature,vehicle_a_progress_info,vehicle_b_progress_info,ratio_a,ratio_b,ratio_class
사거리 교차로(신호등 없음),동일폭 도로,오른쪽에서 직진,왼쪽에서 직진,40,60,21
```

### Class Map CSV

`data/lookup/class_maps/{name}.csv`

```csv
class_id,label
0,직선 도로
1,사거리 교차로(신호등 없음)
```

Supported names:

- `accident_place`
- `accident_place_feature`
- `vehicle_a_progress_info`
- `vehicle_b_progress_info`

### Residual Training CSV

`data/processed/residual_cases.csv`

```csv
case_id,base_ratio_a,true_ratio_a,entry_order,first_entry_strength,first_entry_conf,A_no_deceleration,A_no_deceleration_strength,A_no_deceleration_conf,B_no_deceleration,B_no_deceleration_strength,B_no_deceleration_conf,A_evasive_action,A_evasive_action_strength,A_evasive_action_conf,B_evasive_action,B_evasive_action_strength,B_evasive_action_conf
case-001,40,30,A_first,strong,0.9,false,,0,true,medium,0.7,false,,0,true,weak,0.55
```

The target is `true_ratio_a - base_ratio_a`.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,perception,llm]"
```

Train VTN:

```bash
python scripts/train_vtn.py \
  --data_root data/raw \
  --output_dir outputs/vtn_place \
  --label_key accident_place \
  --num_classes 15
```

Train adjustment model from real residual cases:

```bash
python scripts/train_adjustment.py \
  --input_csv data/processed/residual_cases.csv \
  --output_dir outputs/adjustment
```

Run one pipeline pass:

```bash
python scripts/run_case.py \
  --video_path data/raw/sample.mp4 \
  --statement "상대 차량이 오른쪽에서 들어왔고 저는 서행하지 못했습니다." \
  --base_ratio_csv data/lookup/base_ratio_table.csv \
  --adjustment_model outputs/adjustment/adjustment_model.joblib
```

## Notes

Perception modules use optional dependencies. `ultralytics` and `norfair` are imported only when those classes are instantiated, so the rules/VTN/report layers can still be developed without GPU tooling installed.
