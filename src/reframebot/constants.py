from __future__ import annotations

import re
from typing import List

# ---------------------------------------------------------------------------
# Crisis response
# ---------------------------------------------------------------------------

VIETNAMESE_HOTLINES: str = (
    "Please reach out to these resources in Vietnam:\n\n"
    "**1. National Protection Hotline:** 1900 1267\n"
    "**2. 'Ngay Mai' Hotline (Depression & Suicide Prevention):** 096 306 1414\n"
    "**3. Emergency Services:** 113 or 115\n"
    "**4. Depression Emergency Hotline:** 1900 1267\n\n"
    "Please reach out for help immediately. There are people who care about you."
)

# ---------------------------------------------------------------------------
# Academic keyword override list
# ---------------------------------------------------------------------------

ACADEMIC_KEYWORDS: List[str] = [
    # CBT / study techniques
    "pomodoro", "cbt", "cognitive behavioral therapy",
    "smart goals", "mind map", "active recall",
    "spaced repetition", "feynman", "imposter syndrome",
    "burnout", "distortion", "catastrophizing",
    # Common academic stress terms (helps override false TASK_2)
    "exam", "exams", "final", "finals", "midterm", "quiz",
    "assignment", "assignments", "homework", "deadline", "deadlines",
    "presentation", "presenting", "thesis", "dissertation",
    "gpa", "grade", "grades", "course", "class", "lecture",
    "study", "studying", "revision", "project", "projects",
]

# Compiled word-boundary patterns for efficient matching
ACADEMIC_KEYWORD_PATTERNS: List[re.Pattern] = [
    re.compile(r"(?<!\w)" + re.escape(kw.strip().lower()) + r"(?!\w)", flags=re.IGNORECASE)
    for kw in ACADEMIC_KEYWORDS
    if kw.strip()
]

# ---------------------------------------------------------------------------
# Follow-up / clarification phrases (used to detect short follow-up turns)
# ---------------------------------------------------------------------------

FOLLOWUP_PHRASES: List[str] = [
    "i don't know", "idk", "what do you mean", "i cannot understand",
    "i can't understand", "can you explain", "huh", "sorry", "i'm confused",
    "confused", "i don't get it", "i dont get it", "not sure",
]

# ---------------------------------------------------------------------------
# Semantic prototype sentences for cosine-similarity crisis detection
# ---------------------------------------------------------------------------

CRISIS_PROTOTYPES: List[str] = [
    "I want to kill myself.",
    "I'm going to end my life.",
    "I have a plan to commit suicide.",
    "I want to die and I plan to do it tonight.",
    "I am going to hurt myself.",
    "I will overdose on pills.",
    "I want to self-harm.",
    "Tôi muốn tự tử.",
    "Tôi muốn kết thúc cuộc đời.",
    "Tôi muốn làm hại bản thân.",
    "Tôi có kế hoạch tự tử.",
]

ACADEMIC_STRESS_PROTOTYPES: List[str] = [
    "I'm stressed about my exams and grades.",
    "I'm overwhelmed by assignments and deadlines.",
    "I'm anxious about a presentation at school.",
    "I'm worried about failing my course.",
    "I'm burned out from studying.",
    "I'm struggling with coursework and pressure.",
    "Tôi đang stress vì bài tập và deadline.",
    "Tôi lo lắng vì kỳ thi và điểm số.",
    "Tôi hoảng vì thuyết trình ở trường.",
    "Tôi kiệt sức vì học hành.",
]

# ---------------------------------------------------------------------------
# Hard crisis regex patterns (language-level signals)
# ---------------------------------------------------------------------------

_REAL_CRISIS_PATTERN_STRINGS: List[str] = [
    r"\b(suicide|suicidal)\b",
    r"\b(kill myself|end my life|take my life)\b",
    r"\b(hurt myself|harm myself|self[-\s]?harm)\b",
    r"\b(overdose)\b",
    r"\b(i\s*(?:want|wanna)\s*to\s*die)\b",
    r"\b(i\s*(?:don\s*'?t|do not)\s*want\s*to\s*live)\b",
    r"\b(no\s+reason\s+to\s+live)\b",
    r"\b(wish\s+i\s+(?:were|was)\s+dead)\b",
    r"\b(t\s*ô\s*i\s*(?:mu\s*ô\s*\s*n|muốn)\s*t\s*ự\s*t\s*ử)\b",
    r"\b(mu\s*ô\s*\s*n|muốn)\s*ch\s*ế\s*t\b",
    r"\b(k\s*ế\s*t\s*th\s*ú\s*c)\s*(?:cu\s*ộ\s*c)\s*(?:đ\s*ờ\s*i)\b",
]

_BENIGN_METAPHOR_PATTERN_STRINGS: List[str] = [
    r"\bdie\s+of\s+(?:embarrassment|laughter)\b",
    r"\bdying\s+of\s+(?:embarrassment|laughter)\b",
    r"\bthat\s+killed\s+me\b",
    r"\bkill\s+it\b",
]

REAL_CRISIS_RE: List[re.Pattern] = [
    re.compile(p, flags=re.IGNORECASE) for p in _REAL_CRISIS_PATTERN_STRINGS
]
BENIGN_METAPHOR_RE: List[re.Pattern] = [
    re.compile(p, flags=re.IGNORECASE) for p in _BENIGN_METAPHOR_PATTERN_STRINGS
]

# ---------------------------------------------------------------------------
# Strings that indicate the LLM accidentally produced a crisis-style response
# for a non-crisis task (used as a safeguard post-generation)
# ---------------------------------------------------------------------------

ACCIDENTAL_CRISIS_TRIGGERS: List[str] = [
    "1-800-273", "741741", "hotline", "lifeline",
]
