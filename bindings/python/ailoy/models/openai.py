from typing import Literal

from pydantic import BaseModel, Field

OpenAIModelId = (
    Literal[
        "o4-mini",
        "o3",
        "o3-pro",
        "o3-mini",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
    ]
    | str
)


class OpenAIModel(BaseModel):
    component_type: str = Field("openai", frozen=True, init=False)
    id: OpenAIModelId
    api_key: str

    def to_attrs(self):
        return {
            "model": self.id,
            "api_key": self.api_key,
        }
