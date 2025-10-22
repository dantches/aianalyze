"""Heuristic vulnerability remediation suggestions."""

from __future__ import annotations

import re
from typing import List


_PATTERNS = [
    (re.compile(r"\beval\s*\("), "Avoid using eval on untrusted input; consider ast.literal_eval or a safer parser."),
    (re.compile(r"\bexec\s*\("), "Avoid exec with dynamic strings; refactor into explicit function calls."),
    (re.compile(r"subprocess\.(Popen|run|call)\s*\(.*shell\s*=\s*True"), "Disable shell=True in subprocess calls or sanitise inputs to prevent command injection."),
    (re.compile(r"pickle\.loads\s*\("), "Do not load pickles from untrusted sources; prefer json or custom serializers."),
    (re.compile(r"yaml\.load\s*\("), "Use yaml.safe_load to avoid arbitrary code execution when parsing YAML."),
    (re.compile(r"hashlib\.md5\s*\("), "MD5 is cryptographically broken; use hashlib.sha256 or stronger algorithms."),
    (re.compile(r"random\.(random|randrange|randint)\s*\("), "Python's random module is not cryptographically secure; use secrets or os.urandom for security tokens."),
]


def suggest_remediations(code: str) -> List[str]:
    """Return textual suggestions based on simple pattern matching."""

    suggestions: List[str] = []
    for pattern, message in _PATTERNS:
        if pattern.search(code):
            suggestions.append(message)

    if not suggestions:
        suggestions.append(
            "Review input validation, authentication checks, and error handling; tighten any unsafe patterns before deployment."
        )
    return suggestions
