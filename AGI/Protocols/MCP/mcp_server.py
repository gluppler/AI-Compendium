"""
Model Context Protocol (MCP) Server

A simplified implementation of the Model Context Protocol server,
which exposes resources, tools, and prompts to LLM clients.

https://modelcontextprotocol.io
"""

from __future__ import annotations

import json
from typing import Any


class MCPResource:
    def __init__(self, uri: str, name: str, content: str, mime_type: str = "text/plain"):
        self.uri = uri
        self.name = name
        self.content = content
        self.mime_type = mime_type

    def to_dict(self) -> dict:
        return {"uri": self.uri, "name": self.name, "mimeType": self.mime_type}

    def read(self) -> dict:
        return {"uri": self.uri, "content": self.content}


class MCPTool:
    def __init__(self, name: str, description: str, parameters: dict):
        self.name = name
        self.description = description
        self.parameters = parameters

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.parameters,
        }

    def execute(self, args: dict) -> str:
        return json.dumps({"result": f"Executed {self.name} with {args}"})


class MCPServer:
    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.resources: dict[str, MCPResource] = {}
        self.tools: dict[str, MCPTool] = {}

    def add_resource(self, resource: MCPResource) -> None:
        self.resources[resource.uri] = resource

    def add_tool(self, tool: MCPTool) -> None:
        self.tools[tool.name] = tool

    def handle_request(self, request: dict) -> dict:
        method = request.get("method", "")
        params = request.get("params", {})

        if method == "resources/list":
            return {"resources": [r.to_dict() for r in self.resources.values()]}
        elif method == "resources/read":
            uri = params.get("uri", "")
            resource = self.resources.get(uri)
            return resource.read() if resource else {"error": f"Resource {uri} not found"}
        elif method == "tools/list":
            return {"tools": [t.to_dict() for t in self.tools.values()]}
        elif method == "tools/call":
            tool = self.tools.get(params.get("name", ""))
            if tool:
                return {"content": [{"type": "text", "text": tool.execute(params.get("arguments", {}))}]}
            return {"error": f"Tool {params.get('name')} not found"}
        return {"error": f"Unknown method: {method}"}


if __name__ == "__main__":
    server = MCPServer("demo-server")
    server.add_resource(MCPResource("docs://readme", "README", "# Demo Server\nThis is a test."))
    server.add_tool(MCPTool(
        "get_weather", "Get weather for a location",
        {"type": "object", "properties": {"location": {"type": "string"}}}
    ))

    requests = [
        {"method": "resources/list", "params": {}},
        {"method": "tools/list", "params": {}},
        {"method": "tools/call", "params": {"name": "get_weather", "arguments": {"location": "NYC"}}},
    ]
    for req in requests:
        resp = server.handle_request(req)
        print(f"Request: {req['method']} -> Response: {json.dumps(resp, indent=2)}")
    print("MCP server demo complete.")
