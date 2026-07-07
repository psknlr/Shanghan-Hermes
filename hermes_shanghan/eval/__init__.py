"""Evaluation suites: prescription-cloze (LOCO), historical case replay,
and evidence-grounding metrics. See runner.run_suites."""
from .cases import CaseBenchmark  # noqa: F401
from .cloze import ClozeBenchmark  # noqa: F401
from .grounding import GroundingBenchmark, build_question_bank  # noqa: F401
from .runner import run_suites  # noqa: F401
