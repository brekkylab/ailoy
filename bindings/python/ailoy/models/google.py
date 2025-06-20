from typing import Literal

from pydantic import BaseModel, Field

GeminiModelId = (
    Literal[
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ]
    | str
)


class GeminiModel(BaseModel):
    component_type: str = Field("gemini", frozen=True, init=False)
    id: GeminiModelId
    api_key: str

    def to_attrs(self):
        return {
            "model": self.id,
            "api_key": self.api_key,
        }
