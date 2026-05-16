from __future__ import annotations

from accident_liability.schemas import AdjustmentResult, BaseRatioResult, Evidence, SceneAnchor


class ReportGenerator:
    def generate(
        self,
        anchor: SceneAnchor,
        evidence: Evidence,
        base: BaseRatioResult,
        adjustment: AdjustmentResult,
        violations: dict[str, list[str]] | None = None,
    ) -> str:
        violations = violations or {"A": [], "B": []}
        lines = [
            "사고 과실비율 분석 결과",
            "",
            f"- 사고 장소 후보: {anchor.accident_place or '불명'}",
            f"- 사고 유형 후보: {anchor.accident_place_feature or '불명'}",
            f"- A 진행 정보: {anchor.vehicle_a_progress_info or '불명'}",
            f"- B 진행 정보: {anchor.vehicle_b_progress_info or '불명'}",
            "",
            f"- 기준 과실비율: A {base.ratio_a:.0f}% / B {base.ratio_b:.0f}%",
            f"- 가감 보정: A 기준 {adjustment.adjustment:+.1f}%p",
            f"- 최종 과실비율: A {adjustment.final_ratio_a}% / B {adjustment.final_ratio_b}%",
            "",
            "근거",
            f"- 진입 순서: {evidence.entry_order}, 신뢰도 {evidence.first_entry_conf:.2f}",
            f"- 진행 방향: A={evidence.heading.get('A', 'unknown')}, B={evidence.heading.get('B', 'unknown')}",
            f"- 접근 방향: A={evidence.side_approach.get('A', 'unknown')}, B={evidence.side_approach.get('B', 'unknown')}",
            f"- 충돌 직전 상대 위치: {evidence.collision_relative_position or '불명'}",
        ]
        if violations["A"] or violations["B"]:
            lines.extend(
                [
                    "",
                    "사용자 진술 기반 위반 사항",
                    f"- A: {', '.join(violations['A']) if violations['A'] else '없음'}",
                    f"- B: {', '.join(violations['B']) if violations['B'] else '없음'}",
                ]
            )
        if adjustment.reasons:
            lines.append("")
            lines.append("주요 가감 사유")
            for reason in adjustment.reasons:
                lines.append(
                    f"- {reason['feature']}: contribution {reason['contribution']:+.2f}"
                )
        lines.extend(
            [
                "",
                "불확실성",
                "- VTN 후보, 객체 추적, A/B annotation 품질에 따라 결과가 달라질 수 있습니다.",
                "- 실제 서비스에서는 사람이 base ratio 매칭과 주요 evidence를 검수하는 단계를 권장합니다.",
            ]
        )
        return "\n".join(lines)
