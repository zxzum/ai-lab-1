"""Оценка маршрутизации: первое решение модели на фиксированном наборе запросов.

    python -m scripts.eval_routing --label v1 [--dataset docs/evaluation/lab2-routing.json]

Формат набора: {"now": ISO с поясом, "timezone": IANA, "cases": [{"id", "query",
"expected_action": имя инструмента | "respond" | "clarify", "expected_args": {...}}]}.
В expected_args значение "~слово" означает «текст содержит слово». Инструменты не вызываются.
"""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from app.agent.loop import build_system_prompt
from app.agent.tools import CLARIFY, RATIONALE, expose_tools, localize_reminder
from app.config import Settings
from app.llm.client import LLMClient
from app.llm.prompts import get_system_prompt
from app.mcp_client.client import MCPToolbox, server_params


def args_match(expected: dict, actual: dict) -> bool:
    for key, want in expected.items():
        got = str(actual.get(key, "")).strip().casefold()
        if want.startswith("~"):
            if want[1:].casefold() not in got:
                return False
        elif got != want.strip().casefold():
            return False
    return True


async def decide(llm, tools, system, query, timezone, now):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": query}]
    turn = await llm.chat_with_tools(messages, tools, 0.0)
    if not turn.tool_calls:
        return "respond", {}
    call = turn.tool_calls[0]
    arguments = json.loads(call.arguments)
    arguments.pop(RATIONALE, None)
    if call.name == "add_reminder":  # как в агенте: время трактуется в зоне пользователя
        arguments = localize_reminder(arguments, timezone, now)
    return ("clarify" if call.name == CLARIFY else call.name), arguments


async def evaluate(settings, dataset) -> list[dict]:
    now, timezone = datetime.fromisoformat(dataset["now"]), dataset["timezone"]
    llm = LLMClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        timeout=settings.llm_timeout_seconds,
        proxy=settings.llm_proxy_url,
    )
    toolbox = MCPToolbox(server_params(settings))
    await toolbox.start()
    try:
        tools = expose_tools(toolbox.tools)
        system = build_system_prompt(get_system_prompt("study"), now, timezone, timezone, True)
        rows = []
        for case in dataset["cases"]:
            action, arguments = await decide(llm, tools, system, case["query"], timezone, now)
            needs_tool = case["expected_action"] not in {"respond", "clarify"}
            rows.append(
                {
                    **case,
                    "actual": action,
                    "arguments": arguments,
                    "action_ok": action == case["expected_action"],
                    "args_ok": needs_tool and args_match(case["expected_args"], arguments),
                }
            )
        return rows
    finally:
        await llm.close()
        await toolbox.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="docs/evaluation/lab2-routing.json")
    parser.add_argument("--label", default="run")
    args = parser.parse_args()
    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    settings = Settings.load(require_llm=True)

    rows = asyncio.run(evaluate(settings, dataset))

    print("| № | Запрос | Ожидаемое | Фактическое | Аргументы | Результат |")
    print("|---:|---|---|---|---|---|")
    for r in rows:
        needs_tool = r["expected_action"] not in {"respond", "clarify"}
        args_cell = ("да" if r["args_ok"] else "нет") if needs_tool else "—"
        ok = r["action_ok"] and (r["args_ok"] or not needs_tool)
        row = (r["id"], r["query"], r["expected_action"], r["actual"], args_cell)
        print("| " + " | ".join(map(str, row)) + f" | {'✓' if ok else '✗'} |")
    routing = sum(r["action_ok"] for r in rows)
    tool_rows = [r for r in rows if r["expected_action"] not in {"respond", "clarify"}]
    summary = {
        "label": args.label,
        "model": settings.llm_model,
        "routing_accuracy": f"{routing}/{len(rows)}",
        "argument_accuracy": f"{sum(r['args_ok'] for r in tool_rows)}/{len(tool_rows)}",
    }
    print(json.dumps(summary, ensure_ascii=False))
    out = Path("docs/evaluation/results") / f"{args.label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**summary, "rows": rows}, ensure_ascii=False, indent=1), "utf-8")


if __name__ == "__main__":
    main()
