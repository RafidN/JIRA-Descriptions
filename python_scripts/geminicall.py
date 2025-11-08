import os
import logging
import google.generativeai as genai

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior staff engineer aiding a product manager.
Given a JIRA ticket title and repository context, produce a crisp, step-by-step, developer-ready action plan.

Output Markdown with sections:
- Summary
- Assumptions
- Risks/Unknowns
- Action Items (checkbox list with owner placeholders)
- Acceptance Criteria
- Test Plan
- Files to Touch

Be concise but specific; reference files by path when possible.
"""

_GENAI_READY = False
_RAW_MODELS: list[str] = []
_STRIPPED_TO_RAW: dict[str, str] = {}

_GEN_CFG = {
    "temperature": 0.2,
    "max_output_tokens": 1200,
}

def _ensure_genai_configured() -> None:
    global _GENAI_READY, _RAW_MODELS, _STRIPPED_TO_RAW
    if _GENAI_READY:
        return
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        try:
            from dotenv import load_dotenv, find_dotenv  # type: ignore
            load_dotenv(find_dotenv(usecwd=True) or ".env")
            api_key = os.getenv("GOOGLE_API_KEY")
        except Exception:
            pass
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set")
    genai.configure(api_key=api_key)

    _GENAI_READY = True

def _normalize_requested(name: str) -> str:
    if not name:
        return ""
    # If user provided full raw name that exists
    if name in _RAW_MODELS:
        return name
    # If user omitted prefix
    stripped = name.split("/")[-1]
    if stripped in _STRIPPED_TO_RAW:
        return _STRIPPED_TO_RAW[stripped]
    # If prefix missing but raw starts with 'models/' add it
    candidate = f"models/{stripped}"
    if candidate in _RAW_MODELS:
        return candidate
    return name  # will likely fail and raise clear error

def _pick_model() -> str:
    env_model = os.getenv("GEMINI_MODEL")
    if env_model:
        nm = _normalize_requested(env_model)
        if nm in _RAW_MODELS:
            return nm
        logger.warning(
            f"GEMINI_MODEL='{env_model}' not available for this account. "
            f"Falling back to preference order. Available: {_RAW_MODELS}"
        )

    return "models/gemini-flash-latest"

def call_gemini(jira_title: str, context_markdown: str) -> str:
    _ensure_genai_configured()
    model_name = _pick_model()
    if _RAW_MODELS and model_name not in _RAW_MODELS:
        logger.warning(f"Selected '{model_name}' not in available list; trying anyway.")

    logger.info(f"Using Gemini model={model_name}")
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=SYSTEM_PROMPT,
    )
    user_content = (
        f"# JIRA Ticket Title\n{jira_title}\n\n"
        f"# Repo Context (Top-K)\n{context_markdown}"
    )
    try:
        resp = model.generate_content(
            contents=[{"role": "user", "parts": [user_content]}],
            generation_config=_GEN_CFG,
        )
    except Exception as e:
        raise RuntimeError(f"Gemini generateContent failed for '{model_name}': {e}") from e

    text = getattr(resp, "text", "") or ""
    if not text.strip():
        return (
            "## Summary\nNo output generated.\n\n"
            "## Action Items\n- [ ] Retry with different model or adjust retrieval."
        )
    return text
