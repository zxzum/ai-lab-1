"""Запуск MCP-сервера как дочернего процесса: python -m app.mcp_server (транспорт stdio)."""

from app.mcp_server.server import mcp

if __name__ == "__main__":
    mcp.run(transport="stdio")
