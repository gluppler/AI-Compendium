"""
Function Calling (Tool Use Protocol)

Implements the function calling pattern used by LLMs to invoke
external tools, APIs, and functions with structured parameters.

https://platform.openai.com/docs/guides/function-calling
"""

from __future__ import annotations

import json
import re
from typing import Any


class FunctionDefinition:
    def __init__(self, name: str, description: str, parameters: dict[str, dict]):
        self.name = name
        self.description = description
        self.parameters = parameters

    def to_openai_schema(self) -> dict:
        props = {}
        required = []
        for param_name, param_info in self.parameters.items():
            props[param_name] = {
                "type": param_info.get("type", "string"),
                "description": param_info.get("description", ""),
            }
            if param_info.get("required", False):
                required.append(param_name)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": required,
                },
            },
        }


class FunctionRegistry:
    def __init__(self):
        self.functions: dict[str, FunctionDefinition] = {}
        self.handlers: dict[str, callable] = {}

    def register(self, fn_def: FunctionDefinition, handler: callable) -> None:
        self.functions[fn_def.name] = fn_def
        self.handlers[fn_def.name] = handler

    def parse_call(self, text: str) -> tuple[str, dict[str, Any]] | None:
        for name in self.functions:
            pattern = rf"{re.escape(name)}\((.*?)\)"
            match = re.search(pattern, text, re.DOTALL)
            if match:
                args_str = f"{{{match.group(1)}}}"
                args_str = re.sub(r"(\w+):", r'"\1":', args_str)
                args_str = args_str.replace("'", '"')
                try:
                    args = json.loads(args_str)
                except json.JSONDecodeError:
                    args = {}
                return name, args
        return None

    def execute(self, name: str, args: dict) -> str:
        handler = self.handlers.get(name)
        if handler:
            result = handler(**args)
            return json.dumps({"result": result})
        return json.dumps({"error": f"Function {name} not found"})


def get_weather(location: str, units: str = "celsius") -> str:
    return f"Weather in {location}: 22°{ 'C' if units == 'celsius' else 'F' }, partly cloudy"


if __name__ == "__main__":
    registry = FunctionRegistry()
    registry.register(
        FunctionDefinition(
            "get_weather",
            "Get current weather for a location",
            {
                "location": {"type": "string", "description": "City name", "required": True},
                "units": {"type": "string", "description": "Temperature units", "required": False},
            },
        ),
        get_weather,
    )

    schema = registry.functions["get_weather"].to_openai_schema()
    print("OpenAI function schema:")
    print(json.dumps(schema, indent=2))

    call_text = 'get_weather(location: "Paris", units: "celsius")'
    parsed = registry.parse_call(call_text)
    if parsed:
        name, args = parsed
        result = registry.execute(name, args)
        print(f"\nParsed call: {name}({args})")
        print(f"Result: {result}")
    print("Function calling protocol demo complete.")
