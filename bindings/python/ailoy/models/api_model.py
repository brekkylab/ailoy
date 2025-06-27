from typing import Literal, Optional, Self, get_args

from pydantic import model_validator
from pydantic.dataclasses import dataclass

OpenAIModelId = Literal[
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

GeminiModelId = Literal[
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

ClaudeModelId = Literal[
    "claude-sonnet-4-20250514",
    "claude-3-7-sonnet-20250219",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-20240620",
    "claude-opus-4-20250514",
    "claude-3-opus-20240229",
    "claude-3-5-haiku-20241022",
    "claude-3-haiku-20240307",
]

APIModelProvider = Literal["openai", "gemini", "claude"]


@dataclass
class APIModel:
    id: OpenAIModelId | GeminiModelId | ClaudeModelId | str
    api_key: str
    provider: Optional[APIModelProvider] = None

    @model_validator(mode="after")
    def validate_provider(self) -> Self:
        if self.provider is None:
            if self.id in get_args(OpenAIModelId):
                self.provider = "openai"
            elif self.id in get_args(GeminiModelId):
                self.provider = "gemini"
            elif self.id in get_args(ClaudeModelId):
                self.provider = "claude"
            else:
                raise ValueError(
                    f'Failed to infer the model provider based on the model id "{self.id}". '
                    "Please provide an explicit model provider."
                )

        return self

    @property
    def component_type(self) -> str:
        return self.provider

    def to_attrs(self):
        return {
            "model": self.id,
            "api_key": self.api_key,
        }
