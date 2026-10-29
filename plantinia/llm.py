from __future__ import annotations
import json
from .config import LLM_MODEL, KB


class PlantBrain:
    """Local GGUF assistant with deterministic knowledge-base fallback."""

    def __init__(self):
        self.llm = None
        self.llm_error = None
        self.kb = json.loads(KB.read_text(encoding="utf-8")) if KB.exists() else {}
        if LLM_MODEL.exists():
            try:
                from llama_cpp import Llama
                self.llm = Llama(
                    model_path=str(LLM_MODEL),
                    n_ctx=2048,
                    n_threads=4,
                    verbose=False,
                )
            except Exception as exc:
                self.llm_error = f"Local model could not load: {exc}"
        else:
            self.llm_error = "Local GGUF model not found."

    @property
    def status(self) -> str:
        return "Ready • local GGUF" if self.llm else "Fallback • local knowledge base"

    def context(self, vision=None, sensors=None):
        return json.dumps(
            {
                "plant_knowledge": self.kb,
                "vision": vision or [],
                "sensors": sensors or {},
            },
            ensure_ascii=False,
        )

    def _knowledge_fallback(self, question, vision=None, sensors=None):
        q = question.casefold()
        matches = []
        for key, entry in self.kb.items():
            names = [key, entry.get("common_name", ""), entry.get("scientific_name", "")]
            if any(name and name.casefold() in q for name in names):
                matches.append((key, entry))
        if not matches:
            matches = [
                (key, entry)
                for key, entry in self.kb.items()
                if any(word in q for word in key.casefold().replace("_", " ").split())
            ]
        if matches:
            key, plant = matches[0]
            lines = [
                f"Plant knowledge base: {plant.get('common_name', key)}",
                f"Scientific name: {plant.get('scientific_name', 'not listed')}",
            ]
            for field, label in [
                ("water", "Water"),
                ("light", "Light"),
                ("soil", "Soil"),
                ("nutrition", "Nutrition"),
            ]:
                if plant.get(field):
                    lines.append(f"{label}: {plant[field]}")
            if plant.get("problems"):
                lines.append("Known problems: " + ", ".join(plant["problems"]))
            lines.append("Local knowledge-base lookup only; confirm disease and fertiliser decisions with evidence.")
            return "\n".join(lines)

        names = sorted({entry.get("common_name", key) for key, entry in self.kb.items()})
        lines = [
            "The local language model is unavailable, so a generated answer cannot be provided.",
            "Try a plant covered by the local knowledge base: " + ", ".join(names),
        ]
        if vision:
            best = vision[0]
            lines.append(
                f"Latest vision result: {best.get('label', 'unknown')} "
                f"({best.get('confidence', 0):.1%}, {best.get('state', 'unknown')})."
            )
        if sensors:
            known = [f"{key}={value}" for key, value in sensors.items() if value is not None]
            if known:
                lines.append("Entered sensor readings: " + ", ".join(known))
        return "\n".join(lines)

    def ask(self, question: str, vision=None, sensors=None) -> str:
        question = question.strip()
        if not question:
            return "Ask a plant-care question to use Plant Brain."
        if not self.llm:
            return self._knowledge_fallback(question, vision, sensors)

        prompt = (
            "You are Plantinia, an offline plant-care assistant. Use the supplied local context for "
            "plant-specific claims. Be practical and cautious. Never claim a visual prediction is a "
            "certain diagnosis. Do not invent sensor readings, treatments, or sources. If the local "
            "context is insufficient, say so.\n\n"
            f"LOCAL CONTEXT:\n{self.context(vision, sensors)}\n\nUSER:\n{question}"
        )
        try:
            output = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": "You are Plantinia Offline."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=350,
                temperature=0.2,
            )
            text = output["choices"][0]["message"]["content"].strip()
            return text or self._knowledge_fallback(question, vision, sensors)
        except Exception as exc:
            self.llm_error = f"Local model failed during generation: {exc}"
            return self._knowledge_fallback(question, vision, sensors)
