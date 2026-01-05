import pytest

import ailoy as ai

pytestmark = [pytest.mark.asyncio]


def _check_inference_config_values(config: ai.LangModelInferConfig, data: dict):
    for k, v in data.items():
        if k == "document_polyfill" and v == "Qwen3":
            assert (
                config.document_polyfill.system_message_template
                == "{{- text }}\n# Knowledges\nAfter the user's question, a list of documents retrieved from the knowledge base may appear. Try to answer the user’s question based on the provided knowledges."
            )
            assert (
                config.document_polyfill.query_message_template
                == '{{- text }}\n{%- if documents %}\n    {{- "<documents>\\n" }}\n    {%- for doc in documents %}\n    {{- "<document>\\n" }}\n        {{- doc.text + \'\\n\' }}\n    {{- "</document>\\n" }}\n    {%- endfor %}\n    {{- "</documents>\\n" }}\n{%- endif %}'
            )
        else:
            assert getattr(config, k) == v


def _check_knowledge_config_values(config: ai.KnowledgeConfig, data: dict):
    for k, v in data.items():
        assert getattr(config, k) == v


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"temperature": 0.0},
        {"temperature": 0.0, "top_p": 0.0},
        {"max_tokens": 16384},
        {"document_polyfill": "Qwen3"},
        {"document_polyfill": "Qwen3", "think_effort": "enable"},
        {
            "document_polyfill": "Qwen3",
            "think_effort": "enable",
            "temperature": 0.6,
            "top_p": 0.9,
            "max_tokens": 16384,
        },
    ],
)
async def test_inference_config(data):
    config = ai.LangModelInferConfig.from_dict(data)
    _check_inference_config_values(config, data)


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"top_k": 3},
    ],
)
async def test_knowledge_config(data):
    config = ai.KnowledgeConfig.from_dict(data)
    _check_knowledge_config_values


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"inference": {}},
        {"knowledge": {"top_k": 3}},
        {"inference": {}, "knowledge": {}},
        {"inference": {"document_polyfill": "Qwen3"}, "knowledge": {"top_k": 3}},
    ],
)
async def test_agent_config(data):
    config = ai.AgentConfig.from_dict(data)
    _check_inference_config_values(config.inference, data.get("inference", {}))
    _check_knowledge_config_values(config.knowledge, data.get("knowledge", {}))
