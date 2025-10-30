from typing import Literal, Optional

from pydantic.dataclasses import dataclass

LocalModelBackend = Literal["tvm"]
LocalModelId = Literal[
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-30B-A3B",
]
Quantization = Literal["q4f16_1"]


@dataclass
class LocalModel:
    id: LocalModelId
    backend: LocalModelBackend = "tvm"
    quantization: Quantization = "q4f16_1"
    device: int = 0

    @property
    def default_system_message(self) -> Optional[str]:
        if self.id.startswith("Qwen"):
            return "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        return None

    @property
    def component_type(self) -> str:
        if self.backend == "tvm":
            return "tvm_language_model"
        raise ValueError(f"Unknown local model backend: {self.backend}")

    def to_attrs(self) -> dict:
        if self.backend == "tvm":
            return {
                "model": self.id,
                "quantization": self.quantization,
                "device": self.device,
            }
        raise ValueError(f"Unknown local model backend: {self.backend}")
