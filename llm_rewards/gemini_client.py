from __future__ import annotations

import json
import os
import re
import time
import random
from typing import Dict, Tuple, Any, List

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from google import genai


# ---- Load env (.env) ----
load_dotenv()
if not os.getenv("GEMINI_API_KEY"):
    raise RuntimeError(
        "Missing GEMINI_API_KEY. Put it in .env or set it as an environment variable."
    )

client = genai.Client()

# Like your blueprint: try common identifiers to avoid NOT_FOUND formatting issues. :contentReference[oaicite:3]{index=3}
MODEL_CANDIDATES = ["models/gemini-2.5-flash", "gemini-2.5-flash"]


def pick_gemini_25_flash() -> str:
    last_err = None
    for m in MODEL_CANDIDATES:
        try:
            _ = client.models.generate_content(model=m, contents="ping").text
            return m
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not use Gemini 2.5 Flash. Last error: {last_err}")


MODEL_NAME = pick_gemini_25_flash()


class PlanJSON(BaseModel):
    actions: List[int] = Field(min_length=1, max_length=5, description="List of actions 0..3")
    reason: str = Field(description="Short reason (<= 40 words)")


class EvalJSON(BaseModel):
    failure_modes: List[str] = Field(default_factory=list)
    notes: str = Field(default="")


def safe_parse_json(text: str) -> Any:
    """
    Strict parsing with a backup extraction if Gemini wraps JSON in extra text.
    """
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


def gemini_plan(prompt: str, temperature: float = 0.4, max_retries: int = 3) -> Tuple[Dict, bool]:
    """
    Returns (plan_dict, used_fallback)
    """
    last_err = None
    for k in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config={
                    "temperature": float(temperature),
                    "response_mime_type": "application/json",
                    "response_json_schema": PlanJSON.model_json_schema(),
                },
            )
            raw = resp.text
            obj = safe_parse_json(raw)
            plan = PlanJSON.model_validate(obj).model_dump()
            plan["raw"] = raw
            return plan, False
        except Exception as e:
            last_err = e
            time.sleep(0.6 * (k + 1))

    # fallback: random short plan
    plan = {
        "actions": [random.randint(0, 3) for _ in range(3)],
        "reason": "FALLBACK_RANDOM (Gemini error)",
        "raw": repr(last_err),
    }
    return plan, True


def gemini_eval(prompt: str, temperature: float = 0.2, max_retries: int = 2) -> Tuple[Dict, bool]:
    last_err = None
    for k in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config={
                    "temperature": float(temperature),
                    "response_mime_type": "application/json",
                    "response_json_schema": EvalJSON.model_json_schema(),
                },
            )
            raw = resp.text
            obj = safe_parse_json(raw)
            ev = EvalJSON.model_validate(obj).model_dump()
            ev["raw"] = raw
            return ev, False
        except Exception as e:
            last_err = e
            time.sleep(0.5 * (k + 1))

    ev = {
        "failure_modes": ["gemini_fallback"],
        "notes": f"Eval fallback due to error: {last_err}",
        "raw": repr(last_err),
    }
    return ev, True