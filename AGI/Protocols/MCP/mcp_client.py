"""
Model Context Protocol (MCP) Client

A simplified implementation of the MCP client that connects to an
MCP server to discover and use resources and tools.

https://modelcontextprotocol.io
"""

from __future__ import annotations

import json
from typing import Any


class MCPClient:
    def __init__(self):
        self.server = None

    def connect(self, server) -> None:
        self.server = server

    def list_resources(self) -> list[dict]:
        response = self._send({"method": "resources/list", "params": {}})
        return response.get("resources", [])

    def read_resource(self, uri: str) -> dict | None:
        response = self._send({"method": "resources/read", "params": {"uri": uri}})
        return response if "error" not in response else None

    def list_tools(self) -> list[dict]:
        response = self._send({"method": "tools/list", "params": {}})
        return response.get("tools", [])

    def call_tool(self, name: str, arguments: dict) -> str:
        response = self._send({
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        })
        content = response.get("content", [])
        return content[0].get("text", "") if content else response.get("error", "Unknown error")

    def _send(self, request: dict) -> dict:
        if not self.server:
            return {"error": "Not connected"}
        return self.server.handle_request(request)


if __name__ == "__main__":
    from protocols.mcp_server import MCPServer, MCPResource, MCPTool

    server = MCPServer("demo-server")
    server.add_resource(MCPResource("docs://manual", "Manual", "User manual content."))
    server.add_tool(MCPTool(
        "calculate", "Perform calculation",
        {"type": "object", "properties": {"expr": {"type": "string"}}}
    ))

    client = MCPClient()
    client.connect(server)

    print("Resources:", [r["name"] for r in client.list_resources()])
    print("Tools:", [t["name"] for t in client.list_tools()])
    print("Read resource:", client.read_resource("docs://manual"))
    print("Call tool:", client.call_tool("calculate", {"expr": "2+2"}))
    print("MCP client demo complete.")
