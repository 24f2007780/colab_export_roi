"""Model Context Protocol (MCP) Client for Google Colab and Jupyter Notebooks.

This module provides a lightweight, reusable MCP client wrapper allowing Python
code (e.g. running in Google Colab) to connect to MCP servers over stdio, SSE,
or Streamable HTTP transports.

Typical use cases:
1. Local stdio servers (e.g. official MCP Everything server via npx):
    async with MCPClient.from_everything_server() as client:
        tools = await client.list_tools()
        result = await client.call_tool("echo", {"message": "Hello from Colab!"})

2. Remote SSE servers (e.g. SGBC MCP server over HTTP/SSE):
    async with MCPClient.from_sse(url="https://api.sgbc.iitm.ac.in/mcp/sse") as client:
        tools = await client.list_tools()

3. Quick diagnostics / connectivity test:
    report = await MCPClient.test_connection()
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
import sys
from typing import Any, Dict, List, Optional, Union

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client

try:
    from mcp.client.streamable_http import streamable_http_client
    HAS_STREAMABLE_HTTP = True
except ImportError:
    HAS_STREAMABLE_HTTP = False


class MCPClient:
    """Lightweight, transport-agnostic Model Context Protocol (MCP) client.

    Supports both async context manager (`async with MCPClient(...) as client:`)
    and explicit connect/close lifecycle (`await client.connect()` / `await client.close()`).
    """

    def __init__(
        self,
        transport: str = "stdio",
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        url: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 10.0,
        errlog: Optional[Any] = None,
    ) -> None:
        """Initialize MCPClient configuration.

        Parameters
        ----------
        transport : str
            Transport protocol: 'stdio', 'sse', or 'streamable_http' / 'http'.
        command : str, optional
            Command binary for stdio transport (e.g. 'npx', 'python', 'docker').
        args : list of str, optional
            Arguments for the stdio command (e.g. ['-y', '@modelcontextprotocol/server-everything']).
        env : dict, optional
            Environment variables for stdio subprocess.
        url : str, optional
            Target URL for SSE or Streamable HTTP transport.
        headers : dict, optional
            HTTP request headers for SSE or HTTP transport (e.g. auth tokens).
        timeout : float
            Connection timeout in seconds.
        errlog : file-like, optional
            Stream for stderr. Defaults to safe OS stderr in Jupyter/Colab environments.
        """
        self.transport = transport.lower()
        self.command = command
        self.args = args or []
        self.env = env
        self.url = url
        self.headers = headers
        self.timeout = timeout
        self.errlog = errlog

        self._exit_stack: Optional[AsyncExitStack] = None
        self.session: Optional[ClientSession] = None
        self.init_result: Any = None

    @staticmethod
    def _resolve_errlog(custom_errlog: Optional[Any] = None) -> Any:
        """Resolve a safe stream for subprocess stderr with a valid OS file descriptor.

        In Jupyter, IPython, and Google Colab, `sys.stderr` is wrapped by `ipykernel.iostream.OutStream`,
        whose `.fileno()` raises `io.UnsupportedOperation('fileno')` when passed to subprocesses.
        We fall back to `sys.__stderr__` (the true OS stderr stream with fileno=2) or `/dev/null`.
        """
        if custom_errlog is not None:
            return custom_errlog

        # 1. Try sys.stderr if it has a real working fileno
        try:
            if sys.stderr is not None and hasattr(sys.stderr, "fileno"):
                sys.stderr.fileno()
                return sys.stderr
        except Exception:
            pass

        # 2. In Jupyter/IPython/Colab: fallback to sys.__stderr__ (original OS stderr)
        try:
            if sys.__stderr__ is not None and hasattr(sys.__stderr__, "fileno"):
                sys.__stderr__.fileno()
                return sys.__stderr__
        except Exception:
            pass

        # 3. Fallback to /dev/null
        try:
            import os
            return open(os.devnull, "w")
        except Exception:
            return sys.__stderr__

    @classmethod
    def from_stdio(
        cls,
        command: str = "npx",
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        errlog: Optional[Any] = None,
    ) -> "MCPClient":
        """Create an MCPClient configured for local stdio transport."""
        return cls(transport="stdio", command=command, args=args, env=env, errlog=errlog)

    @classmethod
    def from_sse(
        cls,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 10.0,
    ) -> "MCPClient":
        """Create an MCPClient configured for remote Server-Sent Events (SSE) transport."""
        return cls(transport="sse", url=url, headers=headers, timeout=timeout)

    @classmethod
    def from_http(
        cls,
        url: str,
        headers: Optional[Dict[str, str]] = None,
    ) -> "MCPClient":
        """Create an MCPClient configured for Streamable HTTP transport."""
        return cls(transport="streamable_http", url=url, headers=headers)

    @classmethod
    def from_everything_server(cls) -> "MCPClient":
        """Convenience factory for the official MCP Everything reference server."""
        return cls.from_stdio("npx", ["-y", "@modelcontextprotocol/server-everything"])

    async def connect(self) -> "MCPClient":
        """Establish connection to the MCP server and initialize the session."""
        if self.session is not None:
            return self

        self._exit_stack = AsyncExitStack()
        try:
            if self.transport == "stdio":
                if not self.command:
                    raise ValueError("stdio transport requires 'command'")
                server_params = StdioServerParameters(
                    command=self.command,
                    args=self.args,
                    env=self.env,
                )
                safe_err = self._resolve_errlog(self.errlog)
                read, write = await self._exit_stack.enter_async_context(
                    stdio_client(server_params, errlog=safe_err)
                )
            elif self.transport == "sse":
                if not self.url:
                    raise ValueError("sse transport requires 'url'")
                read, write = await self._exit_stack.enter_async_context(
                    sse_client(self.url, headers=self.headers, timeout=self.timeout)
                )
            elif self.transport in ("streamable_http", "http"):
                if not HAS_STREAMABLE_HTTP:
                    raise RuntimeError("streamable_http transport is not supported by installed mcp version")
                if not self.url:
                    raise ValueError("streamable_http transport requires 'url'")
                read, write = await self._exit_stack.enter_async_context(
                    streamable_http_client(self.url)
                )
            else:
                raise ValueError(f"Unsupported transport: {self.transport}")

            self.session = await self._exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            self.init_result = await self.session.initialize()
            return self

        except Exception:
            await self.close()
            raise

    async def close(self) -> None:
        """Close session and release transport resources."""
        if self._exit_stack:
            try:
                await self._exit_stack.aclose()
            finally:
                self._exit_stack = None
                self.session = None
                self.init_result = None

    async def __aenter__(self) -> "MCPClient":
        return await self.connect()

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    def _ensure_connected(self) -> ClientSession:
        if self.session is None:
            raise RuntimeError("MCPClient is not connected. Call 'await connect()' or use 'async with'.")
        return self.session

    @property
    def server_info(self) -> Any:
        """Return server information metadata from initialize response."""
        if self.init_result and hasattr(self.init_result, "server_info"):
            return self.init_result.server_info
        return None

    @property
    def server_capabilities(self) -> Any:
        """Return server capabilities from initialize response."""
        if self.init_result and hasattr(self.init_result, "capabilities"):
            return self.init_result.capabilities
        return None

    @property
    def protocol_version(self) -> Optional[str]:
        """Return negotiated protocol version."""
        if self.init_result and hasattr(self.init_result, "protocol_version"):
            return self.init_result.protocol_version
        return None

    @property
    def instructions(self) -> Optional[str]:
        """Return server instructions if provided by the server."""
        if self.init_result and hasattr(self.init_result, "instructions"):
            return self.init_result.instructions
        return None

    async def list_tools(self) -> List[Any]:
        """List all tools exposed by the MCP server."""
        session = self._ensure_connected()
        result = await session.list_tools()
        return list(result.tools)

    async def list_resources(self) -> List[Any]:
        """List all resources exposed by the MCP server."""
        session = self._ensure_connected()
        result = await session.list_resources()
        return list(result.resources)

    async def list_prompts(self) -> List[Any]:
        """List all prompts exposed by the MCP server."""
        session = self._ensure_connected()
        result = await session.list_prompts()
        return list(result.prompts)

    async def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Any:
        """Call an MCP tool by name with arguments.

        Parameters
        ----------
        name : str
            Tool identifier name.
        arguments : dict, optional
            Arguments matching tool's input schema.

        Returns
        -------
        CallToolResult
            MCP CallToolResult containing content, is_error flag, and optional metadata.
        """
        session = self._ensure_connected()
        return await session.call_tool(name=name, arguments=arguments or {})

    async def call_tool_text(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> str:
        """Call an MCP tool and return the combined text representation of the response."""
        result = await self.call_tool(name, arguments)
        return self.extract_text(result)

    @staticmethod
    def extract_text(call_result: Any) -> str:
        """Extract clean text content from an MCP CallToolResult or content block."""
        if call_result is None:
            return ""
        if isinstance(call_result, str):
            return call_result
        content = getattr(call_result, "content", call_result)
        if not isinstance(content, (list, tuple)):
            content = [content]

        texts: List[str] = []
        for item in content:
            if hasattr(item, "text") and item.text is not None:
                texts.append(str(item.text))
            elif isinstance(item, dict) and "text" in item:
                texts.append(str(item["text"]))
            else:
                texts.append(str(item))
        return "\n".join(texts)

    async def tools_summary(self) -> List[Dict[str, Any]]:
        """Return a simplified list of dicts describing available tools."""
        tools = await self.list_tools()
        summary = []
        for t in tools:
            summary.append({
                "name": getattr(t, "name", ""),
                "description": getattr(t, "description", ""),
                "input_schema": getattr(t, "input_schema", {}),
            })
        return summary

    async def tools_dataframe(self) -> Any:
        """Return available tools as a pandas DataFrame (if pandas is installed)."""
        summary = await self.tools_summary()
        try:
            import pandas as pd
            return pd.DataFrame(summary)[["name", "description"]]
        except ImportError:
            return summary

    @classmethod
    async def test_connection(
        cls,
        transport: str = "stdio",
        command: str = "npx",
        args: Optional[List[str]] = None,
        url: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        test_tool: str = "echo",
        test_args: Optional[Dict[str, Any]] = None,
        errlog: Optional[Any] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Perform end-to-end connectivity check to an MCP server.

        Connects, inspects capabilities, enumerates tools/resources/prompts,
        executes a test tool, prints human-readable diagnostics, and returns
        a structured report dict.
        """
        if args is None and transport == "stdio":
            args = ["-y", "@modelcontextprotocol/server-everything"]
        if test_args is None:
            test_args = {"message": "Hello from Google Colab!"}

        report: Dict[str, Any] = {
            "status": "pending",
            "transport": transport,
            "endpoint": f"{command} {' '.join(args)}" if transport == "stdio" else str(url),
        }

        try:
            client = cls(
                transport=transport,
                command=command,
                args=args,
                url=url,
                headers=headers,
                errlog=errlog,
            )
            async with client:
                info = client.server_info
                server_name = getattr(info, "name", "unknown") if info else "unknown"
                server_version = getattr(info, "version", "unknown") if info else "unknown"
                server_title = getattr(info, "title", None) if info else None

                tools = await client.list_tools()
                resources = await client.list_resources()
                prompts = await client.list_prompts()

                tool_result_obj = None
                tool_output_text = None
                if test_tool and any(getattr(t, "name", None) == test_tool for t in tools):
                    tool_result_obj = await client.call_tool(test_tool, test_args)
                    tool_output_text = client.extract_text(tool_result_obj)

                report.update({
                    "status": "connected",
                    "server_name": server_name,
                    "server_title": server_title,
                    "server_version": server_version,
                    "protocol_version": client.protocol_version,
                    "tools_count": len(tools),
                    "tools": [{"name": getattr(t, "name", ""), "description": getattr(t, "description", "")} for t in tools],
                    "resources_count": len(resources),
                    "resources": [{"uri": getattr(r, "uri", ""), "name": getattr(r, "name", "")} for r in resources],
                    "prompts_count": len(prompts),
                    "prompts": [{"name": getattr(p, "name", ""), "description": getattr(p, "description", "")} for p in prompts],
                    "test_tool": test_tool,
                    "test_tool_result": tool_output_text,
                })

            if verbose:
                _print_diagnostic_banner(report)

            return report

        except Exception as exc:
            report["status"] = "error"
            report["error"] = str(exc)
            if verbose:
                print(f"\n❌ MCP Connection FAILED: {exc}\n")
            return report


def _print_diagnostic_banner(report: Dict[str, Any]) -> None:
    """Print formatted diagnostic summary for console or Colab output."""
    status = report.get("status", "unknown").upper()
    sep = "=" * 60
    subsep = "-" * 60

    print(f"\n{sep}")
    print(f"  MCP Server Connectivity Test: {status}")
    print(f"{subsep}")
    print(f"  Transport:        {report.get('transport')} ({report.get('endpoint')})")
    print(f"  Server Name:      {report.get('server_name')} (v{report.get('server_version')})")
    if report.get("server_title"):
        print(f"  Server Title:     {report.get('server_title')}")
    print(f"  Protocol Version: {report.get('protocol_version')}")
    print(f"  Available Tools:  {report.get('tools_count')} tools")
    print(f"  Resources:        {report.get('resources_count')} resources")
    print(f"  Prompts:          {report.get('prompts_count')} prompts")

    test_tool = report.get("test_tool")
    if test_tool:
        result_text = report.get("test_tool_result", "")
        print(f"{subsep}")
        print(f"  Test Tool Call:   '{test_tool}'")
        print(f"  Result Output:    {result_text}")

    print(f"{subsep}")
    tools_list = report.get("tools", [])
    if tools_list:
        print("  Discovered Tools Preview:")
        for t in tools_list[:5]:
            print(f"    • {t['name']}: {t['description']}")
        if len(tools_list) > 5:
            print(f"    ... and {len(tools_list) - 5} more tools")

    print(f"{sep}\n")


async def run_quick_test() -> Dict[str, Any]:
    """Convenience helper to test the Everything server with default settings."""
    return await MCPClient.test_connection()


if __name__ == "__main__":
    asyncio.run(run_quick_test())
