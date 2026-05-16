from __future__ import annotations

from accident_liability.llm.violation_parser import ViolationParser
from accident_liability.report.generator import ReportGenerator
from accident_liability.rules.adjustment import AdjustmentModel
from accident_liability.rules.base_ratio import BaseRatioLookup
from accident_liability.schemas import CaseInput, Evidence, SceneAnchor


class LiabilityPipeline:
    def __init__(
        self,
        base_lookup: BaseRatioLookup,
        adjustment_model: AdjustmentModel,
        violation_parser: ViolationParser | None = None,
        report_generator: ReportGenerator | None = None,
    ):
        self.base_lookup = base_lookup
        self.adjustment_model = adjustment_model
        self.violation_parser = violation_parser
        self.report_generator = report_generator or ReportGenerator()

    def run_with_ready_inputs(
        self,
        case: CaseInput,
        anchor: SceneAnchor,
        evidence: Evidence,
    ) -> dict:
        base = self.base_lookup.get_from_anchor(anchor)
        adjustment = self.adjustment_model.apply(base.ratio_a, evidence)
        violations = {"A": [], "B": []}
        if self.violation_parser is not None:
            violations = self.violation_parser.parse(case.statement)
        report = self.report_generator.generate(anchor, evidence, base, adjustment, violations)
        return {
            "anchor": anchor,
            "evidence": evidence,
            "base": base,
            "adjustment": adjustment,
            "violations": violations,
            "report": report,
        }
