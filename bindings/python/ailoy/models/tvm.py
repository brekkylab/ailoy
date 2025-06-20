from typing import Literal, Optional

from pydantic import BaseModel, Field

TVMModelId = Literal["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B", "Qwen/Qwen3-8B"]

Quantization = Literal["q4f16_1"]


class TVMModel(BaseModel):
    component_type: str = Field("tvm_language_model", frozen=True, init=False)
    id: TVMModelId
    quantization: Quantization = "q4f16_1"
    device: int = 0

    @property
    def default_system_message(self) -> Optional[str]:
        if self.id.startswith("Qwen"):
            return "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        return None

    def to_attrs(self):
        return {
            "model": self.id,
            "quantization": self.quantization,
            "device": self.device,
        }
