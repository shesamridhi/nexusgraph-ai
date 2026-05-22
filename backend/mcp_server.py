"""
NexusGraph AI — MCP Server
Claude Desktop / Cursor seedha NexusGraph se baat kar sakta hai.
"""
import asyncio
import json
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import httpx

API_BASE = "http://localhost:8000"
app = Server("nexusgraph")

@app.list_tools()
async def list_tools():
    return [
        Tool(name="search_graph", description="NexusGraph mein query karo — Neo4j + ChromaDB se answer milega", inputSchema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}),
        Tool(name="fetch_document", description="ChromaDB se direct semantic search", inputSchema={"type": "object", "properties": {"query": {"type": "string"}, "top_k": {"type": "integer", "default": 5}}, "required": ["query"]}),
        Tool(name="graph_stats", description="NexusGraph ka graph stats dekho", inputSchema={"type": "object", "properties": {}}),
        Tool(name="ingest_url", description="Koi bhi URL ka content NexusGraph mein add karo", inputSchema={"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}),
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    async with httpx.AsyncClient(timeout=60) as client:
        if name == "search_graph":
            r = await client.post(f"{API_BASE}/query", json={"question": arguments["query"], "max_hops": 2})
            data = r.json()
            return [TextContent(type="text", text=f"**Answer:** {data.get('answer', 'No answer')}\n\n**Confidence:** {data.get('confidence', 0):.2f}\n\n**Query Type:** {data.get('query_type', 'unknown')}")]

        elif name == "fetch_document":
            r = await client.get(f"{API_BASE}/vector/search", params={"q": arguments["query"], "top_k": arguments.get("top_k", 5)})
            data = r.json()
            results = data.get("results", [])
            text = "\n\n".join([f"[{i+1}] (score={r['score']:.3f})\n{r['content'][:300]}" for i, r in enumerate(results)])
            return [TextContent(type="text", text=text or "No results found")]

        elif name == "graph_stats":
            r = await client.get(f"{API_BASE}/graph/stats")
            data = r.json()
            return [TextContent(type="text", text=json.dumps(data, indent=2))]

        elif name == "ingest_url":
            url = arguments["url"]
            async with httpx.AsyncClient(timeout=30) as fetch_client:
                page = await fetch_client.get(url)
                text = page.text[:5000]
            r = await client.post(f"{API_BASE}/ingest/text", json={"content": text, "title": url, "metadata": {"source": url, "type": "web"}})
            data = r.json()
            return [TextContent(type="text", text=f"Ingested: {data.get('doc_id', 'done')} — {data.get('chunks_created', 0)} chunks created")]

        return [TextContent(type="text", text="Unknown tool")]

async def main():
    async with stdio_server() as (r, w):
        await app.run(r, w, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())