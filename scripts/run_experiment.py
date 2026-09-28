"""Проводит воспроизводимый эксперимент по влиянию temperature на ответы LLM."""

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import Settings
from app.llm.client import LLMClient
from app.llm.prompts import get_system_prompt

PROMPT_UNDER_TEST = "Объясни, что такое стек и где он применяется в реальном программировании."
TEMPERATURES = [0.0, 0.3, 0.7, 1.0]
RUNS_PER_TEMP = 3
TOP_P = 1.0
MAX_TOKENS = 1000


def format_ok(text: str) -> bool:
    return all(marker in text for marker in ("💡", "💻", "⚠️", "❓"))


async def run_experiment(env_file: str) -> dict:
    settings = Settings.load(env_file, require_llm=True)
    client = LLMClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        timeout=settings.llm_timeout_seconds,
        proxy=settings.llm_proxy_url,
    )
    system_prompt = get_system_prompt("study")
    results: list[dict] = []

    try:
        for temperature in TEMPERATURES:
            previous_response = None
            for _attempt in range(1, RUNS_PER_TEMP + 1):
                response = await client.complete(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": PROMPT_UNDER_TEST},
                    ],
                    temperature=temperature,
                    top_p=TOP_P,
                    max_tokens=MAX_TOKENS,
                )
                if previous_response is None:
                    observation = "Первый ответ группы."
                elif response == previous_response:
                    observation = "Совпадает с предыдущим ответом группы."
                else:
                    observation = "Отличается от предыдущего ответа группы."
                result = {
                    "run": len(results) + 1,
                    "temperature": temperature,
                    "length": len(response),
                    "format": format_ok(response),
                    "observation": observation,
                    "response": response,
                    "usage": client.last_usage,
                }
                results.append(result)
                previous_response = response
                print(
                    f"Запуск #{result['run']:02d} | Temp: {temperature:.1f} | "
                    f"Длина: {len(response)} симв. | "
                    f"Формат: {'Да' if result['format'] else 'Нет'} | {observation}"
                )
    finally:
        await client.close()

    experiment = {
        "date": datetime.now(UTC).isoformat(),
        "model": settings.llm_model,
        "prompt": PROMPT_UNDER_TEST,
        "system_prompt": system_prompt,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
        "results": results,
    }
    return experiment


def main() -> int:
    parser = argparse.ArgumentParser(description="Провести эксперимент с temperature")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--output", type=Path, help="Путь для JSON с сырыми результатами")
    args = parser.parse_args()
    experiment = asyncio.run(run_experiment(args.env_file))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(experiment, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Результаты сохранены: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
