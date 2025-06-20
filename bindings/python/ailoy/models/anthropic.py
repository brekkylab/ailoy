from typing import Literal

from pydantic import BaseModel, Field

ClaudeModelId = (
    Literal[
        "claude-sonnet-4-20250514",
        "claude-3-7-sonnet-20250219",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-20240620",
        "claude-opus-4-20250514",
        "claude-3-opus-20240229",
        "claude-3-5-haiku-20241022",
        "claude-3-haiku-20240307",
    ]
    | str
)


class ClaudeModel(BaseModel):
    component_type: str = Field("claude", frozen=True, init=False)
    id: ClaudeModelId
    api_key: str

    def to_attrs(self):
        return {
            "model": self.id,
            "api_key": self.api_key,
        }
