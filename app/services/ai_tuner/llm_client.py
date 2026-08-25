"""Multimodal Vision LLM client for OpenRouter and OpenAI-compatible Vision APIs."""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from typing import Any

from app.services.ai_tuner.models import EvaluationReport, TunerConfig
from app.services.auto_label.models import AutoLabelClass

LOGGER = logging.getLogger(__name__)


class OpenRouterVisionClient:
    """Sends multimodal error diagnosis requests to OpenRouter / OpenAI Vision endpoints."""

    def __init__(self, config: TunerConfig) -> None:
        self.config = config
        self.api_key = config.api_key or ""
        self.base_url = config.base_url.rstrip("/")
        self.model = config.model_name

    @property
    def is_configured(self) -> bool:
        """Check if API key or local endpoint is configured."""
        is_local = "localhost" in self.base_url or "127.0.0.1" in self.base_url
        return bool(self.api_key.strip()) or is_local

    def refine_prompts_with_vision(
        self,
        classes: list[AutoLabelClass],
        report: EvaluationReport,
    ) -> tuple[dict[str, str], str]:
        """Send visual error crops and diagnostics to Vision LLM and return optimized prompts."""
        if not self.is_configured:
            raise ValueError(
                "OpenRouter API key is missing. Please set OPENROUTER_API_KEY in your .env file."
            )

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/Marsel204/VisionLab",
            "X-Title": "TrafficAnnotator-AITuner",
        }

        system_instruction = (
            "You are an elite Vision AI Prompt Engineer specializing in Grounding DINO.\n"
            "Analyze detection errors (missed objects, false positives, confusion) and craft "
            "precise visual prompts for each class to maximize object detection F1-score.\n\n"
            "Guidelines:\n"
            "1. Use natural visual descriptors and synonyms (e.g. 'sedan, suv, taxi').\n"
            "2. For missed objects (low recall), include distinctive visual attributes.\n"
            "3. For confused classes, add disambiguating features.\n"
            "4. Respond STRICTLY in JSON format matching the schema."
        )

        user_content: list[dict[str, Any]] = []

        classes_summary = "\n".join(
            f"- Class '{c.name}': Current Prompt = \"{c.prompt or c.name}\""
            for c in classes
        )
        diagnostics_summary = (
            "\n".join(f"- {d}" for d in report.semantic_diagnostics)
            or "- No specific diagnostics."
        )

        text_prompt = (
            f"=== CURRENT DETECTION PERFORMANCE ===\n"
            f"Overall Macro F1: {int(round(report.overall_macro_f1 * 100))}%\n"
            f"Precision: {int(round(report.overall_precision * 100))}%\n"
            f"Recall: {int(round(report.overall_recall * 100))}%\n\n"
            f"=== CURRENT CLASSES & PROMPTS ===\n{classes_summary}\n\n"
            f"=== ERROR DIAGNOSTICS ===\n{diagnostics_summary}\n\n"
            f"Below are visual crops of error cases. "
            f"Inspect visual traits and provide optimized prompt rewrites in JSON.\n"
        )
        user_content.append({"type": "text", "text": text_prompt})

        for i, crop in enumerate(report.error_crops[:6]):
            if crop.base64_jpeg:
                label_note = f"Error Crop #{i + 1}: [{crop.error_type}] '{crop.class_name}'"
                user_content.append({"type": "text", "text": label_note})
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{crop.base64_jpeg}",
                        "detail": "low",
                    },
                })

        json_schema_prompt = (
            "\nOutput valid JSON:\n"
            "{\n"
            '  "analysis": "Visual critique of failure modes and added keywords.",\n'
            '  "prompt_updates": {\n'
            '    "class_name": "optimized prompt..."\n'
            "  }\n"
            "}"
        )
        user_content.append({"type": "text", "text": json_schema_prompt})

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.3,
            "max_tokens": 1200,
            "response_format": {"type": "json_object"},
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                body = response.read().decode("utf-8")
                response_json = json.loads(body)
                content = response_json["choices"][0]["message"]["content"]
                return self._parse_llm_json_response(content, classes)
        except urllib.error.HTTPError as err:
            err_body = err.read().decode("utf-8", errors="ignore")
            LOGGER.error("OpenRouter API HTTP error: %s - %s", err.code, err_body)
            msg = f"OpenRouter API request failed (HTTP {err.code}): {err_body}"
            raise RuntimeError(msg) from err
        except urllib.error.URLError as err:
            LOGGER.error("OpenRouter API URL error: %s", err.reason)
            msg = f"Could not connect to OpenRouter API at {url}: {err.reason}"
            raise RuntimeError(msg) from err

    def _parse_llm_json_response(
        self, content: str, classes: list[AutoLabelClass]
    ) -> tuple[dict[str, str], str]:
        """Safely parse JSON from LLM response text, with regex markdown block fallback."""
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as err:
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
            if match:
                parsed = json.loads(match.group(1))
            else:
                msg = f"LLM response was not valid JSON: {content[:100]}..."
                raise ValueError(msg) from err

        reasoning = parsed.get("analysis", "Optimized prompts based on visual error analysis.")
        prompt_updates = parsed.get("prompt_updates", {})

        cleaned_updates: dict[str, str] = {}
        for cls_item in classes:
            if cls_item.name in prompt_updates:
                val = str(prompt_updates[cls_item.name]).strip()
                if val:
                    cleaned_updates[cls_item.name] = val

        return cleaned_updates, reasoning
