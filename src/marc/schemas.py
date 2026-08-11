from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class AgentConfig(BaseModel):
    name: str
    model: Optional[str] = None
    prompt_file: str
    context_files: List[str] = Field(default_factory=list)
    temperature: float = 0.0
    num_predict: int = 768
    num_ctx: int = 3072

    @field_validator("name", "prompt_file")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("must not be blank")
        return v


class PipelineConfig(BaseModel):
    agents: List[AgentConfig]

    @field_validator("agents")
    @classmethod
    def _at_least_one_agent(cls, v: List[AgentConfig]) -> List[AgentConfig]:
        if not v:
            raise ValueError("pipeline config must define at least one agent")
        return v
