"""
Structured Output Prompting

Techniques for constraining LLM outputs to structured formats
(JSON, XML, schema-validated) for reliable programmatic consumption.

https://platform.openai.com/docs/guides/structured-outputs
"""

from __future__ import annotations

import json
import re
from typing import Any


class StructuredOutputPrompt:
    def __init__(self, schema: dict[str, str]):
        self.schema = schema

    def build_prompt(self, instruction: str) -> str:
        schema_str = json.dumps(self.schema, indent=2)
        return (
            f"{instruction}\n\n"
            f"Respond with valid JSON matching this schema:\n{schema_str}\n\n"
            f"JSON response:"
        )

    def validate(self, output: str) -> dict[str, Any] | None:
        try:
            data = json.loads(output)
        except json.JSONDecodeError:
            return None
        for key, expected_type in self.schema.items():
            if key not in data:
                return None
            if not isinstance(data[key], eval(expected_type)):
                return None
        return data


class FunctionCallingTemplate:
    def __init__(self, name: str, parameters: dict[str, dict]):
        self.name = name
        self.parameters = parameters

    def build_prompt(self, query: str) -> str:
        params_str = json.dumps(self.parameters, indent=2)
        return (
            f"{query}\n\n"
            f"Call the function `{self.name}` with the required parameters:\n"
            f"{params_str}\n\n"
            f"Function call:"
        )

    def parse_call(self, text: str) -> dict[str, Any] | None:
        match = re.search(
            rf'{self.name}\((.*?)\)', text, re.DOTALL
        )
        if not match:
            return None
        args_str = match.group(1)
        try:
            return json.loads(f"{{{args_str}}}")
        except json.JSONDecodeError:
            return None


def extract_code_block(text: str, language: str = "") -> str | None:
    pattern = rf"```{language}\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None


if __name__ == "__main__":
    schema = {
        "name": "str",
        "age": "int",
        "email": "str",
        "is_active": "bool",
    }
    sop = StructuredOutputPrompt(schema)
    prompt = sop.build_prompt("Create a user profile for John Doe, age 30.")
    print("=== Structured output prompt ===")
    print(prompt)

    valid_json = '{"name": "John Doe", "age": 30, "email": "john@example.com", "is_active": true}'
    result = sop.validate(valid_json)
    print(f"\nValidation result: {result}")

    fc = FunctionCallingTemplate(
        "get_weather",
        {
            "location": {"type": "string"},
            "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
    )
    fc_prompt = fc.build_prompt("What's the weather in Paris?")
    print("\n=== Function calling template ===")
    print(fc_prompt)
    print("Structured output prompting demo complete.")
