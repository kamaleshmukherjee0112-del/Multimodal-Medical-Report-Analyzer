"""
llm_explainer.py

Unified LLM explanation & summarization layer
for Multimodal Medical Report Analyzer.
"""

from typing import Dict, Literal
import json
import requests

# ============================================================
# OLLAMA CONFIG (WINDOWS-SAFE)
# ============================================================

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = "phi3:mini"

# ============================================================
# PUBLIC API (DO NOT CHANGE)
# ============================================================

def explain_report(
    parsed_report: Dict,
    mode: Literal["patient", "clinician"] = "patient",
) -> Dict:
    """
    Generate a safe explanation or summary for a medical report.
    """

    if not parsed_report or not isinstance(parsed_report, dict):
        return _fallback_response()

    prompt = _build_prompt(parsed_report, mode)
    explanation = _run_llm(prompt)

    return {
        "summary": _generate_deterministic_summary(parsed_report),
        "explanation": explanation.strip(),
        "disclaimer": STANDARD_DISCLAIMER,
    }

# ============================================================
# PROMPT CONSTRUCTION
# ============================================================

def _build_prompt(parsed_report: Dict, mode: str) -> str:
    """
    Build a strict, role-specific prompt.
    """

    safe_json = json.dumps(parsed_report, indent=2)

    if mode == "patient":
        return f"""
You are a medical report explanation assistant.

Your task is to explain a medical report to a patient
who has received the report but has not yet consulted a doctor.

STRICT RULES (DO NOT BREAK):
- Do NOT diagnose any disease.
- Do NOT prescribe medications or treatments.
- Do NOT invent medical facts or reference ranges.
- Do NOT contradict the doctor’s report.
- Do NOT exaggerate severity.
- Do NOT add findings not present in the report.
- If the report says "no abnormality", respect it.
- Ignore administrative, junk, or non-medical text.
- Use simple, reassuring language.

YOU MAY:
- Summarize the report in simple terms.
- Highlight abnormal findings ONLY if already flagged.
- Explain what medical terms generally refer to.
- Mention general, non-medical precautions.
- Encourage consulting a qualified doctor.

Here is the structured medical report JSON:
{safe_json}

Write a clear, patient-friendly summary.
End by clearly stating that a doctor must review the report.
""".strip()

    else:  # clinician
        return f"""
You are a clinical report summarization assistant.

Your task is to summarize a medical report
in concise, professional medical language.

STRICT RULES (DO NOT BREAK):
- Do NOT diagnose.
- Do NOT recommend treatment.
- Do NOT infer missing information.
- Do NOT invent reference ranges.
- Do NOT override radiology or lab conclusions.
- Ignore administrative or non-medical noise.

YOU MAY:
- Summarize findings and impressions.
- Highlight reported abnormalities.
- Use standard clinical terminology.
- Recommend clinical correlation (without diagnosis).

Here is the structured medical report JSON:
{safe_json}

Write a brief, neutral clinical summary.
End by stating that findings should be correlated clinically.
""".strip()

# ============================================================
# OLLAMA HEALTH CHECK
# ============================================================

def _ollama_is_running() -> bool:
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

# ============================================================
# LLM EXECUTION (LOCAL OLLAMA)
# ============================================================

def _run_llm(prompt: str) -> str:
    if not _ollama_is_running():
        return (
            "The local language model service is not available. "
            "Please ensure the system is running and consult a healthcare professional."
        )

    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 800,
                    "temperature": 0.2,
                },
            },
            timeout=120,
        )

        resp.raise_for_status()
        data = resp.json()

        response_text = data.get("response")

        if not response_text or not isinstance(response_text, str):
            raise ValueError("Empty or invalid LLM response")

        return response_text.strip()

    except Exception as e:
        print("❌ LLM ERROR:", repr(e))
        return (
            "An explanation could not be generated at this time. "
            "Please consult a qualified healthcare professional for interpretation."
        )

# ============================================================
# DETERMINISTIC SUMMARY (NO LLM)
# ============================================================

def _generate_deterministic_summary(parsed_report: Dict) -> str:
    """
    Generate a short deterministic summary without LLM hallucination.
    """

    exam_type = parsed_report.get("examination", {}).get("exam_type", "report")
    test_results = parsed_report.get("test_results", {})

    abnormal_tests = []

    for row in test_results.get("from_table", []):
        if row.get("flag") in ("Low", "High"):
            abnormal_tests.append(row.get("test_name", "Unknown Test"))

    if abnormal_tests:
        return (
            f"{len(abnormal_tests)} notable finding(s) detected: "
            + ", ".join(abnormal_tests)
            + "."
        )

    return f"No explicitly flagged abnormalities detected in this {exam_type} report."

# ============================================================
# FALLBACK
# ============================================================

def _fallback_response() -> Dict:
    return {
        "summary": "Unable to generate report summary.",
        "explanation": (
            "The report could not be summarized at this time. "
            "Please consult a qualified healthcare professional."
        ),
        "disclaimer": STANDARD_DISCLAIMER,
    }

# ============================================================
# DISCLAIMER
# ============================================================

STANDARD_DISCLAIMER = (
    "This explanation is for educational purposes only. "
    "It does not constitute medical advice. "
    "A qualified healthcare professional must interpret these results."
)
