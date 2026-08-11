import json
import os
import re
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from .pipeline import clean_model_output


class DecomposerAgent:
    """
    Conversational agent that clarifies a user's task and emits a structured
    3-agent pipeline spec (JSON) for MARC.
    """

    def __init__(
        self,
        model_name: str = "gemma4:e4b",
        ollama_base_url: Optional[str] = None,
        temperature: float = 0.7,
        num_predict: int = 2048,
        num_ctx: int = 8192,
        prompt_file: str = "prompts/decomp.txt",
    ):
        self.ollama_base_url = (
            ollama_base_url
            or os.getenv("OLLAMA_BASE_URL")
            or "http://localhost:11434"
        )

        system_prompt = self._load_prompt(prompt_file)

        print("[Decomposer] Initializing model:", model_name)
        self.llm = ChatOllama(
            model=model_name,
            base_url=self.ollama_base_url,
            temperature=temperature,
            num_predict=num_predict,
            num_ctx=num_ctx,
        )

        self.messages = [SystemMessage(content=system_prompt)]
        self._decomposition: Optional[dict] = None

    def _load_prompt(self, path: str) -> str:
        if os.path.exists(path):
            return open(path).read()
        raise FileNotFoundError(f"Decomposer prompt not found: {path}")

    def chat(self, user_message: str) -> str:
        self.messages.append(HumanMessage(content=user_message))
        response = self.llm.invoke(self.messages)
        ai_text = clean_model_output(
            response.content if hasattr(response, "content") else str(response)
        )
        self.messages.append(AIMessage(content=ai_text))

        decomp = self._extract_json(ai_text)
        if decomp and "agents" in decomp and len(decomp["agents"]) == 3:
            self._decomposition = decomp

        return ai_text

    def _extract_json(self, text: str) -> Optional[dict]:
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            try:
                return self._fix_templates(json.loads(match.group(1)))
            except json.JSONDecodeError:
                pass

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return self._fix_templates(json.loads(self._sanitize_json(match.group(0))))
            except json.JSONDecodeError as e:
                print(f"[Decomposer] JSON parse failed: {e}")

        return None

    def _sanitize_json(self, text: str) -> str:
        result = []
        in_string = False
        escaped = False
        for char in text:
            if escaped:
                result.append(char)
                escaped = False
            elif char == "\\":
                result.append(char)
                escaped = True
            elif char == '"':
                result.append(char)
                in_string = not in_string
            elif in_string and char == "\n":
                result.append("\\n")
            elif in_string and char == "\r":
                result.append("\\r")
            elif in_string and char == "\t":
                result.append("\\t")
            else:
                result.append(char)
        return "".join(result)

    _VALID_VARS = {"input", "previous_agent_output", "context"}

    def _fix_templates(self, decomp: dict) -> dict:
        required = [
            ["{input}"],
            ["{input}", "{previous_agent_output}"],
            ["{previous_agent_output}"],
        ]
        for i, agent in enumerate(decomp.get("agents", [])):
            if i >= len(required):
                break
            tmpl = agent.get("prompt_template", "")
            tmpl = re.sub(
                r"\{([^}]+)\}",
                lambda m: "{" + m.group(1) + "}" if m.group(1) in self._VALID_VARS else "",
                tmpl,
            )
            for var in required[i]:
                if var not in tmpl:
                    tmpl += f"\n\n{var}"

            if i == 1:
                tmpl = re.sub(
                    r"VERDICT\s*=\s*<[^>]*>",
                    "VERDICT = [write the exact answer label here]",
                    tmpl,
                )

            tmpl += "\n\nDo not repeat these instructions. Output only your answer."
            agent["prompt_template"] = tmpl
        return decomp

    def has_decomposition(self) -> bool:
        return self._decomposition is not None

    def get_decomposition(self) -> Optional[dict]:
        return self._decomposition
