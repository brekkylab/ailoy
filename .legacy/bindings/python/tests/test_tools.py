from typing import Literal

from ailoy.agent import ToolDescription
from ailoy.tools import get_json_schema


def test_tool_description():
    def fn(
        temperature_format: Literal["celsius", "fahrenheit"],
        literal: int | float | bool | str | None = None,
        bool_literal: Literal[0, 1, True, False, "true", "false", "T", "F", "y", "n"] = False,
    ):
        """
        Test function
        Args:
            temperature_format: The temperature format to use
            literal: Any literal value LLM want to put
            bool_literal: A literal value that can be interpreted as boolean value
        Returns:
            The temperature
        """
        return -40.0

    # Let's see if that gets correctly parsed as an enum
    schema = get_json_schema(fn)
    expected_schema = {
        "name": "fn",
        "description": "Test function",
        "parameters": {
            "type": "object",
            "properties": {
                "temperature_format": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature format to use",
                },
                "literal": {
                    "type": ["boolean", "integer", "number", "string"],
                    "nullable": True,
                    "description": "Any literal value LLM want to put",
                },
                "bool_literal": {
                    "type": ["integer", "boolean", "string"],
                    "enum": [0, 1, True, False, "true", "false", "T", "F", "y", "n"],
                    "description": "A literal value that can be interpreted as boolean value",
                },
            },
            "required": ["temperature_format"],
        },
    }

    assert schema["function"] == expected_schema
    ToolDescription.model_validate(schema.get("function", {}))
