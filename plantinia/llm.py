from __future__ import annotations
import json
from .config import LLM_MODEL, KB


class PlantBrain:
    def __init__(self):
        self.llm = None
        self.kb = json.loads(KB.read_text(encoding="utf-8")) if KB.exists() else {}
        if LLM_MODEL.exists():
            try:
                from llama_cpp import Llama
                self.llm = Llama(model_path=str(LLM_MODEL), n_ctx=2048, n_threads=4, verbose=False)
            except Exception:
                self.llm = None

    def context(self, vision=None, sensors=None):
        return json.dumps({"plant_knowledge": self.kb, "vision": vision or [], "sensors": sensors or {}}, ensure_ascii=False)

    def ask(self, question: str, vision=None, sensors=None) -> str:
        if not self.llm:
            return "Local language model is not installed yet. Run bootstrap_models.ps1. The plant knowledge base and sensor system are available offline."
        prompt = ("You are Plantinia, an offline plant-care assistant. Use only the supplied local context for plant-specific claims. "
                  "Be practical, cautious, and concise. Never claim a visual diagnosis is certain.\n\n"
                  f"LOCAL CONTEXT:\n{self.context(vision, sensors)}\n\nUSER:\n{question}")
        out = self.llm.create_chat_completion(messages=[
            {"role":"system","content":"You are Plantinia Offline."},
            {"role":"user","content":prompt}
        ], max_tokens=350, temperature=0.2)
        return out["choices"][0]["message"]["content"].strip()
