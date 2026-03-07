"""
Enterprise Security Manager
============================
Provides input validation, context sanitization, and output guardrails
for the RAG pipeline. Ports the EnterpriseSecurityManager from the
baseline codebase with enhancements for financial document security.
"""

import re
import logging
import html
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class EnterpriseSecurityManager:
    """
    Multi-layer security guardrail system for the RAG pipeline.

    Layer 1: Input validation — regex-based jailbreak/prompt-injection detection.
    Layer 2: Context sanitization — strip HTML/XML/script tags from retrieved chunks.
    Layer 3: Output sanitization — redact sensitive patterns before returning to user.
    """

    # ------------------------------------------------------------------ #
    #  Jailbreak / Prompt Injection Patterns
    # ------------------------------------------------------------------ #
    JAILBREAK_PATTERNS: List[re.Pattern] = [
        re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
        re.compile(r"ignore\s+(all\s+)?above\s+instructions", re.IGNORECASE),
        re.compile(r"disregard\s+(all\s+)?(previous|prior|above)", re.IGNORECASE),
        re.compile(r"you\s+are\s+now\s+(a|an|the)\s+", re.IGNORECASE),
        re.compile(r"pretend\s+(you|to)\s+(are|be)\s+", re.IGNORECASE),
        re.compile(r"act\s+as\s+(a|an|if)\s+", re.IGNORECASE),
        re.compile(r"system\s*prompt", re.IGNORECASE),
        re.compile(r"reveal\s+(your|the)\s+(system|internal|hidden)", re.IGNORECASE),
        re.compile(r"override\s+(safety|security|content)\s*(filter|policy)?", re.IGNORECASE),
        re.compile(r"<\s*(script|img|iframe|object|embed|form)", re.IGNORECASE),
        re.compile(r"javascript\s*:", re.IGNORECASE),
        re.compile(r"data\s*:\s*text/html", re.IGNORECASE),
        re.compile(r"\{\{.*\}\}", re.IGNORECASE),  # template injection
        re.compile(r"\$\{.*\}", re.IGNORECASE),      # expression injection
    ]

    # Patterns that should be stripped from retrieved context
    CONTEXT_STRIP_PATTERNS: List[re.Pattern] = [
        re.compile(r"<[^>]+>", re.DOTALL),           # HTML/XML tags
        re.compile(r"<script[\s\S]*?</script>", re.IGNORECASE | re.DOTALL),
        re.compile(r"<style[\s\S]*?</style>", re.IGNORECASE | re.DOTALL),
        re.compile(r"<!--[\s\S]*?-->", re.DOTALL),   # HTML comments
        re.compile(r"\x00", re.DOTALL),               # null bytes
    ]

    # Sensitive data patterns to redact in outputs
    SENSITIVE_PATTERNS: Dict[str, re.Pattern] = {
        "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "CREDIT_CARD": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
        "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
        "API_KEY": re.compile(r"(?:api[_-]?key|token|secret)\s*[:=]\s*['\"]?[\w\-]{20,}", re.IGNORECASE),
    }

    MAX_INPUT_LENGTH = 5000   # characters
    MAX_CONTEXT_LENGTH = 50000

    def __init__(self, enable_pii_redaction: bool = True):
        self.enable_pii_redaction = enable_pii_redaction
        logger.info("EnterpriseSecurityManager initialized (PII redaction=%s)", enable_pii_redaction)

    # ------------------------------------------------------------------ #
    #  Layer 1 — Input Validation
    # ------------------------------------------------------------------ #
    def validate_input(self, user_input: str) -> Tuple[bool, str]:
        """
        Validate user input for jailbreak attempts and injection attacks.

        Returns:
            Tuple of (is_safe, sanitized_input_or_error_message)
        """
        if not user_input or not user_input.strip():
            logger.warning("Security: Empty input received.")
            return False, "Input cannot be empty."

        if len(user_input) > self.MAX_INPUT_LENGTH:
            logger.warning("Security: Input exceeds max length (%d > %d).",
                           len(user_input), self.MAX_INPUT_LENGTH)
            return False, f"Input exceeds maximum allowed length of {self.MAX_INPUT_LENGTH} characters."

        # Check for jailbreak patterns
        for pattern in self.JAILBREAK_PATTERNS:
            match = pattern.search(user_input)
            if match:
                logger.warning(
                    "Security: Jailbreak attempt detected — pattern '%s' matched: '%s'",
                    pattern.pattern, match.group()
                )
                return False, "Your query was flagged by our security system. Please rephrase."

        # Sanitize: HTML-escape any residual special characters
        sanitized = html.escape(user_input.strip())
        logger.debug("Security: Input validated and sanitized successfully.")
        return True, sanitized

    # ------------------------------------------------------------------ #
    #  Layer 2 — Context Sanitization
    # ------------------------------------------------------------------ #
    def sanitize_context(self, context_chunks: List[str]) -> List[str]:
        """
        Strip potentially dangerous HTML/XML/script content from retrieved
        context chunks before they are injected into the LLM prompt.
        """
        sanitized_chunks = []
        for i, chunk in enumerate(context_chunks):
            if not chunk:
                continue

            sanitized = chunk
            for pattern in self.CONTEXT_STRIP_PATTERNS:
                sanitized = pattern.sub("", sanitized)

            # Collapse excessive whitespace created by tag removal
            sanitized = re.sub(r"\s{3,}", "  ", sanitized).strip()

            if len(sanitized) > self.MAX_CONTEXT_LENGTH:
                sanitized = sanitized[: self.MAX_CONTEXT_LENGTH]
                logger.warning("Security: Context chunk %d truncated to max length.", i)

            sanitized_chunks.append(sanitized)

        logger.debug("Security: Sanitized %d context chunks.", len(sanitized_chunks))
        return sanitized_chunks

    # ------------------------------------------------------------------ #
    #  Layer 3 — Output Sanitization
    # ------------------------------------------------------------------ #
    def sanitize_output(self, llm_output: str) -> str:
        """
        Redact sensitive data patterns (SSN, credit cards, API keys) from
        the LLM output before returning it to the user.
        """
        if not self.enable_pii_redaction:
            return llm_output

        sanitized = llm_output
        for label, pattern in self.SENSITIVE_PATTERNS.items():
            matches = pattern.findall(sanitized)
            if matches:
                logger.warning("Security: Redacting %d instance(s) of %s from output.", len(matches), label)
                sanitized = pattern.sub(f"[REDACTED-{label}]", sanitized)

        return sanitized

    # ------------------------------------------------------------------ #
    #  Convenience — Full Pipeline Guard
    # ------------------------------------------------------------------ #
    def guard_pipeline(
        self,
        user_input: str,
        context_chunks: List[str],
    ) -> Tuple[bool, str, List[str]]:
        """
        Run full input validation + context sanitization in one call.

        Returns:
            (is_safe, sanitized_query_or_error, sanitized_context_chunks)
        """
        is_safe, result = self.validate_input(user_input)
        if not is_safe:
            return False, result, []

        safe_chunks = self.sanitize_context(context_chunks)
        return True, result, safe_chunks
