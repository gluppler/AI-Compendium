"""
Structured Context

Techniques for structuring prompts using templates, XML, and
JSON schemas to improve LLM comprehension and output reliability.
"""

from __future__ import annotations

import json


class StructuredContext:
    def __init__(self, template: str):
        self.template = template
        self.sections: dict[str, str] = {}

    def add_section(self, name: str, content: str) -> None:
        self.sections[name] = content

    def remove_section(self, name: str) -> None:
        self.sections.pop(name, None)

    def render(self) -> str:
        context = self.template
        for name, content in self.sections.items():
            placeholder = f"{{{{{name}}}}}"
            context = context.replace(placeholder, content)
        return context


class XMLContextBuilder:
    def __init__(self):
        self.elements: list[tuple[str, str, dict]] = []

    def add_element(self, tag: str, content: str, attrs: dict | None = None) -> None:
        self.elements.append((tag, content, attrs or {}))

    def build(self) -> str:
        parts = []
        for tag, content, attrs in self.elements:
            attr_str = " ".join(f'{k}="{v}"' for k, v in attrs.items())
            if attr_str:
                opening = f"<{tag} {attr_str}>"
            else:
                opening = f"<{tag}>"
            parts.append(f"{opening}{content}</{tag}>")
        return "\n".join(parts)


class JSONContextBuilder:
    def __init__(self):
        self.data: dict = {}

    def set(self, key: str, value: object) -> None:
        self.data[key] = value

    def build(self, indent: int = 2) -> str:
        return json.dumps(self.data, indent=indent)

    def add_array_item(self, key: str, item: object) -> None:
        if key not in self.data:
            self.data[key] = []
        self.data[key].append(item)


def section_header(name: str, level: int = 2) -> str:
    return f"{'#' * level} {name}"


if __name__ == "__main__":
    template = """System: {{system}}
Context: {{context}}
User: {{user_query}}
Assistant: """
    sc = StructuredContext(template)
    sc.add_section("system", "You are a helpful AI assistant.")
    sc.add_section("context", "The user is asking about Python programming.")
    sc.add_section("user_query", "How do I use list comprehensions?")
    print("=== Template-based context ===")
    print(sc.render())

    xml = XMLContextBuilder()
    xml.add_element("system", "You are a helpful assistant", {"role": "system"})
    xml.add_element("message", "List comprehensions in Python", {"role": "user"})
    print("\n=== XML context ===")
    print(xml.build())

    jc = JSONContextBuilder()
    jc.set("model", "gpt-4")
    jc.set("temperature", 0.7)
    jc.add_array_item("messages", {"role": "user", "content": "Hello"})
    print("\n=== JSON context ===")
    print(jc.build())

    print("\n" + section_header("Structured Context Demo"))
    print("Structured context demo complete.")
