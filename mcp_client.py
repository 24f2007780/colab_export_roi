"""Model Context Protocol (MCP) Client for Google Colab and Jupyter Notebooks.

This module provides a lightweight, reusable MCP client and interactive notebook
interface allowing researchers to connect to MCP servers (over stdio, SSE, or HTTP)
and interact with tools, resources, and prompts using simple requests.

Quick Start in Google Colab:
----------------------------
# 1. Setup & connect (run once):
from mcp_client import connect_mcp, ask_mcp
await connect_mcp()

# 2. Interact naturally:
await ask_mcp("Use the echo tool to say hello from Colab")
await ask_mcp("add 15 and 27")
await ask_mcp("what is the weather in Chicago?")
await ask_mcp("read resource features.md")
await ask_mcp("get prompt args-prompt city=Boston state=MA")
await ask_mcp("what tools are available?")
await ask_mcp("help")
"""

from __future__ import annotations

import asyncio
import base64
from contextlib import AsyncExitStack
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client

try:
    from mcp.client.streamable_http import streamable_http_client
    HAS_STREAMABLE_HTTP = True
except ImportError:
    HAS_STREAMABLE_HTTP = False

try:
    from IPython.display import display, Markdown, Image, HTML
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False


def _resolve_safe_errlog(custom_errlog: Optional[Any] = None) -> Any:
    """Resolve a stream with a valid OS file descriptor for subprocess stderr.

    In Jupyter/Colab notebooks, `sys.stderr` is wrapped by `ipykernel.iostream.OutStream`
    whose `.fileno()` raises `io.UnsupportedOperation('fileno')` when passed to subprocesses.
    We fall back to `sys.__stderr__` (underlying OS stream with fileno=2) or `/dev/null`.
    """
    if custom_errlog is not None:
        return custom_errlog

    for stream in (sys.stderr, sys.__stderr__):
        if stream is not None and hasattr(stream, "fileno"):
            try:
                stream.fileno()
                return stream
            except Exception:
                pass

    try:
        return open(os.devnull, "w")
    except Exception:
        return sys.__stderr__


class MCPResult:
    """Rich container for MCP query results and tool outputs in Jupyter/Colab."""

    def __init__(
        self,
        action: str,
        target: str,
        text: str = "",
        params: Optional[Dict[str, Any]] = None,
        raw: Any = None,
        structured_data: Optional[Any] = None,
        image_data: Optional[str] = None,
        image_mime: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        self.action = action  # "tool", "resource", "prompt", "discovery", "error"
        self.target = target  # tool name, resource uri, prompt name, or query
        self.text = text
        self.params = params or {}
        self.raw = raw
        self.structured_data = structured_data
        self.image_data = image_data
        self.image_mime = image_mime
        self.error = error

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        status = f"error='{self.error}'" if self.error else "status=success"
        return f"<MCPResult action='{self.action}' target='{self.target}' {status}>"

    def _format_markdown(self) -> str:
        """Format the result as rich GitHub-style Markdown."""
        lines: List[str] = []

        if self.action == "tool":
            lines.append(f"### 🛠️ MCP Tool: `{self.target}`")
            if self.params:
                lines.append(f"**Arguments:** `{json.dumps(self.params)}`\n")
            if self.structured_data:
                lines.append("**Structured Output:**")
                if isinstance(self.structured_data, dict):
                    for k, v in self.structured_data.items():
                        lines.append(f"- **{k}**: `{v}`")
                else:
                    lines.append(f"```json\n{json.dumps(self.structured_data, indent=2)}\n```")
                lines.append("")
            if self.text:
                lines.append("**Response:**")
                lines.append(self.text)

        elif self.action == "resource":
            lines.append(f"### 📄 MCP Resource: `{self.target}`")
            lines.append(f"```markdown\n{self.text}\n```")

        elif self.action == "prompt":
            lines.append(f"### 💬 MCP Prompt: `{self.target}`")
            if self.params:
                lines.append(f"**Arguments:** `{json.dumps(self.params)}`\n")
            lines.append("**Prompt Content:**")
            lines.append(self.text)

        elif self.action == "discovery":
            lines.append(self.text)

        elif self.action == "error":
            lines.append(f"### ❌ MCP Error ({self.target})")
            lines.append(f"**Message:** {self.error or self.text}")

        else:
            lines.append(self.text)

        return "\n".join(lines)

    def _repr_markdown_(self) -> str:
        return self._format_markdown()

    def display(self) -> None:
        """Render the output in Jupyter / Colab notebooks (including inline images)."""
        md_text = self._format_markdown()
        if HAS_IPYTHON:
            display(Markdown(md_text))
            if self.image_data:
                try:
                    img_bytes = base64.b64decode(self.image_data)
                    display(Image(data=img_bytes, format="png"))
                except Exception:
                    pass
        else:
            print(md_text)


class MCPClient:
    """Lightweight Model Context Protocol (MCP) Client for notebooks and services."""

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
    def from_everything_server(cls, errlog: Optional[Any] = None) -> "MCPClient":
        """Convenience factory for the official MCP Everything reference server."""
        return cls.from_stdio("npx", ["-y", "@modelcontextprotocol/server-everything"], errlog=errlog)

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
                safe_err = _resolve_safe_errlog(self.errlog)
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
        """Call an MCP tool by name with arguments."""
        session = self._ensure_connected()
        return await session.call_tool(name=name, arguments=arguments or {})

    async def call_tool_text(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> str:
        """Call an MCP tool and return the combined text representation of the response."""
        result = await self.call_tool(name, arguments)
        return self.extract_text(result)

    async def read_resource(self, uri: str) -> Any:
        """Read a resource by URI from the MCP server."""
        session = self._ensure_connected()
        return await session.read_resource(uri)

    async def read_resource_text(self, uri: str) -> str:
        """Read a resource and extract its text representation."""
        res = await self.read_resource(uri)
        texts: List[str] = []
        for c in getattr(res, "contents", []):
            if hasattr(c, "text") and c.text is not None:
                texts.append(str(c.text))
            elif hasattr(c, "blob") and c.blob is not None:
                texts.append(f"[Binary Blob: {getattr(c, 'mime_type', 'application/octet-stream')}]")
        return "\n".join(texts)

    async def get_prompt(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Any:
        """Retrieve an MCP prompt template by name with arguments."""
        session = self._ensure_connected()
        str_args = {k: str(v) for k, v in (arguments or {}).items()}
        return await session.get_prompt(name, arguments=str_args)

    async def get_prompt_text(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> str:
        """Retrieve an MCP prompt and format its messages as text."""
        res = await self.get_prompt(name, arguments)
        lines: List[str] = []
        for msg in getattr(res, "messages", []):
            role = getattr(msg, "role", "message")
            content = getattr(msg, "content", msg)
            text = self.extract_text(content)
            lines.append(f"**[{role}]** {text}")
        return "\n\n".join(lines)

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

    async def resources_dataframe(self) -> Any:
        """Return available resources as a pandas DataFrame."""
        resources = await self.list_resources()
        rows = [
            {
                "name": getattr(r, "name", ""),
                "uri": getattr(r, "uri", ""),
                "mime_type": getattr(r, "mime_type", ""),
                "description": getattr(r, "description", ""),
            }
            for r in resources
        ]
        try:
            import pandas as pd
            return pd.DataFrame(rows)
        except ImportError:
            return rows

    async def prompts_dataframe(self) -> Any:
        """Return available prompts as a pandas DataFrame."""
        prompts = await self.list_prompts()
        rows = [
            {
                "name": getattr(p, "name", ""),
                "description": getattr(p, "description", ""),
                "arguments": ", ".join(getattr(a, "name", "") for a in getattr(p, "arguments", []) or []),
            }
            for p in prompts
        ]
        try:
            import pandas as pd
            return pd.DataFrame(rows)
        except ImportError:
            return rows

    async def ask(self, query: str = "") -> MCPResult:
        """Interpret a natural-language request, resolve intent, and execute via MCP."""
        q = (query or "").strip()
        ql = q.lower()

        # 1. Overview / Help / What can you do
        if not q or ql in ["help", "overview", "capabilities", "what can you do", "features"] or any(k in ql for k in ["what can you do", "server overview", "server capabilities", "how to use"]):
            return await self._build_overview_result()

        # 2. List tools
        if any(k in ql for k in ["what tools", "list tools", "show tools", "available tools"]) or ql in ["tools", "list_tools"]:
            tools = await self.list_tools()
            lines = [f"### 🛠️ Available MCP Tools ({len(tools)} total)\n"]
            for t in tools:
                lines.append(f"- **`{t.name}`**: {t.description}")
            lines.append("\n*Try calling one:* `await ask_mcp(\"Use the echo tool to say hello from Colab\")`")
            return MCPResult(action="discovery", target="tools", text="\n".join(lines))

        # 3. List resources
        if any(k in ql for k in ["what resources", "list resources", "show resources", "available resources"]) or ql in ["resources", "list_resources"]:
            resources = await self.list_resources()
            lines = [f"### 📄 Available MCP Resources ({len(resources)} total)\n"]
            for r in resources:
                lines.append(f"- **`{r.name}`** (`{r.uri}`): {r.description}")
            lines.append("\n*Try reading one:* `await ask_mcp(\"read resource features.md\")`")
            return MCPResult(action="discovery", target="resources", text="\n".join(lines))

        # 4. List prompts
        if any(k in ql for k in ["what prompts", "list prompts", "show prompts", "available prompts"]) or ql in ["prompts", "list_prompts"]:
            prompts = await self.list_prompts()
            lines = [f"### 💬 Available MCP Prompts ({len(prompts)} total)\n"]
            for p in prompts:
                args = ", ".join(f"`{getattr(a, 'name', '')}`" for a in getattr(p, "arguments", []) or [])
                args_str = f" (args: {args})" if args else " (no args)"
                lines.append(f"- **`{p.name}`**{args_str}: {p.description}")
            lines.append("\n*Try retrieving one:* `await ask_mcp(\"get prompt args-prompt city=Boston state=MA\")`")
            return MCPResult(action="discovery", target="prompts", text="\n".join(lines))

        # 5. Resource Reading
        if any(w in ql for w in ["read resource", "read doc", "get resource", "view resource"]) or (ql.startswith("read ") and any(ext in ql for ext in [".md", ".txt", "demo://"])):
            resources = await self.list_resources()
            matched_uri = None
            for r in resources:
                r_name = getattr(r, "name", "").lower()
                r_uri = getattr(r, "uri", "").lower()
                if r_name and r_name in ql:
                    matched_uri = r.uri
                    break
                if r_uri and r_uri in ql:
                    matched_uri = r.uri
                    break

            # Check dynamic resource pattern: "dynamic [text|blob] <id>"
            if not matched_uri:
                dyn_match = re.search(r'dynamic\s+(text|blob)?\s*(\d+)', ql)
                if dyn_match:
                    res_type = dyn_match.group(1) or "text"
                    res_id = dyn_match.group(2)
                    matched_uri = f"demo://resource/dynamic/{res_type}/{res_id}"

            # Fallback to direct URI in query
            if not matched_uri:
                uri_match = re.search(r'demo://[^\s"\']+', q)
                if uri_match:
                    matched_uri = uri_match.group(0)

            if matched_uri:
                try:
                    content = await self.read_resource_text(matched_uri)
                    return MCPResult(action="resource", target=matched_uri, text=content)
                except Exception as e:
                    return MCPResult(action="error", target=matched_uri, error=str(e), text=f"Failed to read resource: {e}")

        # 6. Prompt Retrieval
        if "prompt" in ql:
            prompts = await self.list_prompts()
            for p in prompts:
                if p.name.lower() in ql:
                    args: Dict[str, Any] = {}
                    for m in re.finditer(r'(\w+)=["\']?([^"\'\s]+)["\']?', q):
                        args[m.group(1)] = m.group(2)
                    try:
                        prompt_text = await self.get_prompt_text(p.name, args)
                        return MCPResult(action="prompt", target=p.name, text=prompt_text, params=args)
                    except Exception as e:
                        return MCPResult(action="error", target=p.name, error=str(e), text=f"Failed to get prompt: {e}")

        # 7. Tool Execution - Explicit Syntax "call <tool_name> <arg1>=<val1>..."
        if ql.startswith("call "):
            parts = q[5:].split()
            tool_name = parts[0]
            args = {}
            for p_str in parts[1:]:
                if "=" in p_str:
                    k, v = p_str.split("=", 1)
                    try:
                        v_parsed = float(v) if "." in v else int(v)
                    except ValueError:
                        v_parsed = v.strip("'\"")
                    args[k] = v_parsed
            return await self._execute_tool_and_wrap(tool_name, args)

        # 8. Tool Execution - Natural Language Intent Parsing
        # 8a. Echo Tool
        if "echo" in ql:
            clean = re.sub(r'^(?:please\s+)?(?:use\s+(?:the\s+)?echo\s+tool\s+(?:to\s+)?)?', '', q, flags=re.I).strip()
            clean = re.sub(r'^(?:say|echo|message\s*[:=]?)\s*', '', clean, flags=re.I).strip().strip("'\"")
            msg = clean or "Hello from Google Colab!"
            return await self._execute_tool_and_wrap("echo", {"message": msg})

        # 8b. Math / Sum Tool
        if any(w in ql for w in ["sum", "add", "plus", "+", "calculate"]):
            nums = [float(n) if "." in n else int(n) for n in re.findall(r'-?\d+(?:\.\d+)?', q)]
            if len(nums) >= 2:
                return await self._execute_tool_and_wrap("get-sum", {"a": nums[0], "b": nums[1]})

        # 8c. Weather / Structured Content Tool
        if any(w in ql for w in ["weather", "structured content", "temperature", "humidity"]):
            city = "Chicago"
            for c in ["New York", "Chicago", "Los Angeles"]:
                if c.lower() in ql:
                    city = c
                    break
            return await self._execute_tool_and_wrap("get-structured-content", {"location": city})

        # 8d. Annotated Message Tool
        if "annotated" in ql:
            msg_type = "error"
            for t in ["success", "debug", "error"]:
                if t in ql:
                    msg_type = t
                    break
            inc_img = "image" in ql
            return await self._execute_tool_and_wrap("get-annotated-message", {"messageType": msg_type, "includeImage": inc_img})

        # 8e. Tiny Image / Logo Tool
        if any(w in ql for w in ["tiny image", "mcp logo", "image", "logo"]):
            return await self._execute_tool_and_wrap("get-tiny-image", {})

        # 8f. Environment Variables Tool
        if any(w in ql for w in ["environment", "env", "get-env"]):
            return await self._execute_tool_and_wrap("get-env", {})

        # 8g. Resource Links Tool
        if "resource link" in ql or "resource links" in ql:
            count = 3
            num_match = re.search(r'\b([1-9]|10)\b', q)
            if num_match:
                count = int(num_match.group(1))
            return await self._execute_tool_and_wrap("get-resource-links", {"count": count})

        # 8h. Resource Reference Tool
        if "resource reference" in ql:
            res_type = "Blob" if "blob" in ql else "Text"
            res_id = 1
            num_match = re.search(r'\b(\d+)\b', q)
            if num_match:
                res_id = int(num_match.group(1))
            return await self._execute_tool_and_wrap("get-resource-reference", {"resourceType": res_type, "resourceId": res_id})

        # 9. Generic Fallback: Match against all available tool names
        tools = await self.list_tools()
        for t in tools:
            if t.name.lower() in ql:
                args = {}
                for m in re.finditer(r'(\w+)=["\']?([^"\'\s]+)["\']?', q):
                    args[m.group(1)] = m.group(2)
                return await self._execute_tool_and_wrap(t.name, args)

        # 10. Could not match - return helpful guidance
        lines = [
            f"ℹ Could not automatically map request: *\"{query}\"*\n",
            "Here are examples of requests you can try:",
            "- `await ask_mcp(\"Use the echo tool to say hello from Colab\")`",
            "- `await ask_mcp(\"add 15 and 27\")`",
            "- `await ask_mcp(\"what is the weather in Chicago?\")`",
            "- `await ask_mcp(\"read resource features.md\")`",
            "- `await ask_mcp(\"get prompt args-prompt city=Boston state=MA\")`",
            "- `await ask_mcp(\"what tools are available?\")`",
            "- `await ask_mcp(\"help\")`",
        ]
        return MCPResult(action="unknown", target=query, text="\n".join(lines))

    async def _execute_tool_and_wrap(self, tool_name: str, arguments: Dict[str, Any]) -> MCPResult:
        """Call tool and package output into an MCPResult with text, structured data, and images."""
        try:
            res = await self.call_tool(tool_name, arguments)
            text = self.extract_text(res)
            structured_data = getattr(res, "structured_content", None)

            # Check if any image is present in content blocks
            img_data = None
            img_mime = None
            content_list = getattr(res, "content", []) or []
            for item in content_list:
                if getattr(item, "type", "") == "image" or hasattr(item, "data"):
                    img_data = getattr(item, "data", None)
                    img_mime = getattr(item, "mime_type", "image/png")
                    break

            return MCPResult(
                action="tool",
                target=tool_name,
                text=text,
                params=arguments,
                raw=res,
                structured_data=structured_data,
                image_data=img_data,
                image_mime=img_mime,
            )
        except Exception as e:
            return MCPResult(
                action="error",
                target=tool_name,
                params=arguments,
                error=str(e),
                text=f"Error executing tool '{tool_name}': {e}",
            )

    async def _build_overview_result(self) -> MCPResult:
        """Construct a comprehensive capabilities overview for researchers."""
        info = self.server_info
        tools = await self.list_tools()
        resources = await self.list_resources()
        prompts = await self.list_prompts()

        server_name = getattr(info, "name", "MCP Server")
        server_ver = getattr(info, "version", "unknown")

        lines = [
            f"## 🌐 MCP Server Capabilities: `{server_name}` (v{server_ver})\n",
            f"**Negotiated Protocol:** `{self.protocol_version}`  ",
            f"**Totals:** `{len(tools)} tools` | `{len(resources)} resources` | `{len(prompts)} prompts`\n",
            "---",
            "### 🛠️ Key Tools",
            "- **`echo`**: Echoes input (`ask_mcp(\"Use the echo tool to say hello from Colab\")`)",
            "- **`get-sum`**: Computes mathematical addition (`ask_mcp(\"add 15 and 27\")`)",
            "- **`get-structured-content`**: Weather with schema validation (`ask_mcp(\"weather for Chicago\")`)",
            "- **`get-annotated-message`**: Priority annotations (`ask_mcp(\"annotated error message\")`)",
            "- **`get-tiny-image`**: Renders embedded PNG MCP logo (`ask_mcp(\"get tiny image\")`)",
            "- **`get-env`**: Dumps server environment JSON (`ask_mcp(\"get environment variables\")`)",
            "- **`get-resource-links`**: Generates links (`ask_mcp(\"get 5 resource links\")`)",
            "\n### 📄 Key Resources",
            "- `architecture.md`, `features.md`, `how-it-works.md`, `instructions.md`",
            "- *Read any document:* `await ask_mcp(\"read resource features.md\")`",
            "\n### 💬 Key Prompts",
            "- `simple-prompt`, `args-prompt` (city, state), `completable-prompt` (department, name)",
            "- *Retrieve a prompt:* `await ask_mcp(\"get prompt args-prompt city=Boston state=MA\")`",
            "\n---",
            "💡 *Tip: You can also use explicit tool calls like:* `await ask_mcp(\"call get-sum a=24 b=18\")`",
        ]
        return MCPResult(action="discovery", target="overview", text="\n".join(lines))

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
        """Perform end-to-end connectivity check to an MCP server."""
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


# ==============================================================================
# Global Researcher API: connect_mcp, ask_mcp, disconnect_mcp
# ==============================================================================

_ACTIVE_CLIENT: Optional[MCPClient] = None


async def connect_mcp(
    server: str = "everything",
    command: Optional[str] = None,
    args: Optional[List[str]] = None,
    url: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    verbose: bool = True,
) -> MCPClient:
    """Connect once to an MCP server and cache the connection for `ask_mcp`.

    Parameters
    ----------
    server : str
        Server target. Defaults to 'everything' (stdio Everything server).
        Or pass an SSE/HTTP URL (e.g. 'https://api.sgbc.iitm.ac.in/mcp/sse').
    command : str, optional
        Custom executable command for stdio (e.g. 'npx').
    args : list of str, optional
        Custom arguments for stdio command.
    url : str, optional
        Target SSE URL if not passed in `server`.
    headers : dict, optional
        HTTP request headers for authentication.
    verbose : bool
        Whether to print connection confirmation.
    """
    global _ACTIVE_CLIENT

    if _ACTIVE_CLIENT is not None and _ACTIVE_CLIENT.session is not None:
        if verbose:
            print("✓ MCP Client is already connected.")
        return _ACTIVE_CLIENT

    if url or server.startswith("http://") or server.startswith("https://"):
        target_url = url or server
        client = MCPClient.from_sse(url=target_url, headers=headers)
    elif command or args:
        client = MCPClient.from_stdio(command=command or "npx", args=args or [])
    else:
        client = MCPClient.from_everything_server()

    await client.connect()
    _ACTIVE_CLIENT = client

    if verbose:
        info = client.server_info
        s_name = getattr(info, "name", "MCP Server")
        s_ver = getattr(info, "version", "unknown")
        tools = await client.list_tools()
        resources = await client.list_resources()
        prompts = await client.list_prompts()
        print(f"✓ Connected to MCP Server: {s_name} (v{s_ver})")
        print(f"  Capabilities: {len(tools)} tools | {len(resources)} resources | {len(prompts)} prompts")
        print("  Ready! Use: await ask_mcp(\"Use the echo tool to say hello from Colab\")\n")

    return _ACTIVE_CLIENT


async def ask_mcp(query: str = "", verbose: bool = True) -> MCPResult:
    """Send a natural-language request, tool invocation, or resource inquiry to the connected MCP server.

    If not already connected, connects automatically to the default MCP server.
    """
    global _ACTIVE_CLIENT
    if _ACTIVE_CLIENT is None or _ACTIVE_CLIENT.session is None:
        await connect_mcp(verbose=False)

    assert _ACTIVE_CLIENT is not None
    result = await _ACTIVE_CLIENT.ask(query)
    if verbose:
        result.display()
    return result


async def disconnect_mcp() -> None:
    """Disconnect and close the active MCP server session."""
    global _ACTIVE_CLIENT
    if _ACTIVE_CLIENT:
        await _ACTIVE_CLIENT.close()
        _ACTIVE_CLIENT = None
        print("✓ Disconnected from MCP server.")


def get_active_client() -> Optional[MCPClient]:
    """Return the currently connected global MCPClient instance if any."""
    return _ACTIVE_CLIENT


if __name__ == "__main__":
    async def _main():
        print("Testing MCPClient connectivity and interactive ask_mcp...")
        await connect_mcp(verbose=True)
        await ask_mcp("Use the echo tool to say hello from Colab")
        await ask_mcp("add 15 and 27")
        await ask_mcp("read resource architecture.md")
        await ask_mcp("what is the weather in Chicago?")
        await disconnect_mcp()

    asyncio.run(_main())
