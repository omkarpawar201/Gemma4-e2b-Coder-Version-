"""Simple human-evaluation rubric for GemmaCode E2B benchmarks.

Scores must not be fabricated. This module provides the rubric definition and
helpers for creating manual evaluation records.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


DIMENSIONS = [
    "technical_correctness",
    "code_quality",
    "maintainability",
    "instruction_following",
    "debugging_accuracy",
    "architectural_reasoning",
    "hallucination_resistance",
]

SCORE_SCALE = {
    "min": 1,
    "max": 5,
    "labels": {
        1: "Poor",
        2: "Weak",
        3: "Adequate",
        4: "Good",
        5: "Excellent",
    },
}

MODEL_VARIANTS = [
    "original_gemma4_e2b",
    "finetuned_gemma4_e2b",
    "finetuned_gemma4_e2b_with_rag",
]


@dataclass
class DimensionScore:
    dimension: str
    score: Optional[int]  # 1-5, or None if not yet evaluated
    notes: str = ""


@dataclass
class TaskEvaluation:
    task_id: str
    model_variant: str
    evaluator: str
    scores: List[DimensionScore]
    overall_notes: str = ""
    evaluated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def make_empty_evaluation(task_id: str, model_variant: str, evaluator: str = "") -> TaskEvaluation:
    """Create an empty (unscored) evaluation template for manual scoring."""
    if model_variant not in MODEL_VARIANTS:
        # Allow custom variants for future flexibility, but callers should prefer known ones.
        pass
    return TaskEvaluation(
        task_id=task_id,
        model_variant=model_variant,
        evaluator=evaluator,
        scores=[DimensionScore(dimension=d, score=None, notes="") for d in DIMENSIONS],
        overall_notes="",
        evaluated_at=datetime.now(timezone.utc).isoformat(),
    )
