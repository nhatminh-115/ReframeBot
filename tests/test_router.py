"""Unit tests for router.resolve_task().

No models are loaded — this is pure routing logic.
"""
import pytest
from reframebot.router import resolve_task

THRESHOLD = 0.90

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _history(*messages: tuple[str, str]) -> list[dict]:
    """Build a history list from (role, content) pairs."""
    return [{"role": role, "content": content} for role, content in messages]


def route(label: str, score: float, history: list[dict]) -> str:
    return resolve_task(
        guardrail_label=label,
        guardrail_score=score,
        crisis_confidence_threshold=THRESHOLD,
        history=history,
    )


# ---------------------------------------------------------------------------
# Priority 0: Follow-up in academic context
# ---------------------------------------------------------------------------

class TestFollowUpInAcademicContext:
    def test_short_followup_after_exam_mention(self):
        history = _history(
            ("user", "I'm really stressed about my exams"),
            ("assistant", "I hear you. What specifically worries you most?"),
            ("user", "idk"),
        )
        assert route("TASK_3", 0.95, history) == "TASK_1"

    def test_confused_response_after_academic_context(self):
        history = _history(
            ("user", "I'm overwhelmed by my deadlines"),
            ("assistant", "That sounds really tough. Can you tell me more?"),
            ("user", "i can't understand what you mean"),
        )
        assert route("TASK_2", 0.50, history) == "TASK_1"

    def test_assistant_question_triggers_override(self):
        history = _history(
            ("user", "I failed my midterm"),
            ("assistant", "That must feel discouraging. What thoughts are going through your mind?"),
            ("user", "not sure"),
        )
        assert route("TASK_3", 0.80, history) == "TASK_1"


# ---------------------------------------------------------------------------
# Priority 1: Academic keyword present
# ---------------------------------------------------------------------------

class TestAcademicKeywordOverride:
    @pytest.mark.parametrize("keyword", [
        "exam", "finals", "gpa", "thesis", "assignment",
        "deadline", "studying", "presentation", "burnout",
    ])
    def test_keyword_forces_task1(self, keyword: str):
        history = _history(("user", f"I'm struggling with my {keyword}"))
        assert route("TASK_3", 0.95, history) == "TASK_1"

    def test_keyword_in_earlier_turn_also_applies(self):
        history = _history(
            ("user", "I have an exam tomorrow"),
            ("assistant", "Tell me more."),
            ("user", "yes please"),
        )
        assert route("TASK_3", 0.95, history) == "TASK_1"


# ---------------------------------------------------------------------------
# Priority 2 & 3: Guardrail TASK_2 with no crisis signal detected
# ---------------------------------------------------------------------------

class TestGuardrailTask2Override:
    def test_high_confidence_false_alarm_becomes_task1(self):
        history = _history(("user", "I want to die of embarrassment after that presentation"))
        assert route("TASK_2", 0.95, history) == "TASK_1"

    def test_low_confidence_task2_becomes_task1_not_task3(self):
        # Key regression test: previously this was incorrectly TASK_3
        history = _history(("user", "I feel really hopeless about everything"))
        result = route("TASK_2", 0.60, history)
        assert result == "TASK_1", (
            f"Low-confidence TASK_2 should route to TASK_1 (empathetic), got {result!r}. "
            "Routing to TASK_3 (chit-chat) is unsafe for ambiguous distress signals."
        )

    def test_task2_any_score_without_academic_keyword_becomes_task1(self):
        history = _history(("user", "I feel like giving up"))
        assert route("TASK_2", 0.75, history) == "TASK_1"


# ---------------------------------------------------------------------------
# Priority 4: Guardrail label trusted as-is
# ---------------------------------------------------------------------------

class TestPassthroughLabels:
    def test_task1_is_kept(self):
        history = _history(("user", "Can you help me manage procrastination?"))
        assert route("TASK_1", 0.92, history) == "TASK_1"

    def test_task3_is_kept_when_no_academic_context(self):
        history = _history(("user", "What's the best recipe for pasta?"))
        assert route("TASK_3", 0.88, history) == "TASK_3"

    def test_task1_low_score_still_kept(self):
        history = _history(("user", "I feel a bit stressed"))
        assert route("TASK_1", 0.55, history) == "TASK_1"
