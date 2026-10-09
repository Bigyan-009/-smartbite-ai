"""LLM provider abstraction.

Supports Google Gemini (default) and OpenAI, selected via ``LLM_PROVIDER``.
API keys are read from the environment only - nothing is hardcoded.

If no key is configured the provider reports ``is_available() == False`` and
callers fall back to offline rule-based logic, so the app always runs.
"""

import json
import re
from typing import Any, Dict, Optional

from config import settings

_llm = None
_init_attempted = False


def is_available() -> bool:
    """True when an LLM is configured and could be initialised."""
    return settings.llm_is_configured() and _get_llm() is not None


def _get_llm():
    """Lazily build the LangChain chat model for the configured provider."""
    global _llm, _init_attempted
    if _init_attempted:
        return _llm
    _init_attempted = True

    if not settings.llm_is_configured():
        return None

    try:
        if settings.LLM_PROVIDER == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI

            _llm = ChatGoogleGenerativeAI(
                model=settings.GEMINI_MODEL,
                google_api_key=settings.GEMINI_API_KEY,
                temperature=0.3,
            )
        elif settings.LLM_PROVIDER == "openai":
            from langchain_openai import ChatOpenAI

            _llm = ChatOpenAI(
                model=settings.OPENAI_MODEL,
                api_key=settings.OPENAI_API_KEY,
                temperature=0.3,
            )
    except Exception:
        _llm = None
    return _llm


def generate(prompt: str, system: Optional[str] = None) -> Optional[str]:
    """Return the model's text response, or None if no LLM is available."""
    llm = _get_llm()
    if llm is None:
        return None
    try:
        messages = []
        if system:
            messages.append(("system", system))
        messages.append(("human", prompt))
        response = llm.invoke(messages)
        return getattr(response, "content", str(response))
    except Exception:
        return None


def generate_json(prompt: str, system: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Generate a response and parse it as JSON. Returns None on failure."""
    text = generate(prompt, system=system)
    if not text:
        return None
    return extract_json(text)


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort JSON extraction from an LLM response (handles code fences)."""
    if not text:
        return None
    # Strip common markdown code fences.
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Fall back to the first {...} block found in the text.
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None
