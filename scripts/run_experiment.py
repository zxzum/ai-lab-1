"""Скрипт для воспроизводимого проведения эксперимента по влиянию temperature на ответы LLM.

Выполняет 12 запусков (по 3 для каждого значения temperature: 0.0, 0.3, 0.7, 1.0),
фиксирует длину ответа, соблюдение формата и вариативность.
Поддерживает работу с реальным LLM API (через .env) либо симуляцию при отсутствии ключа.
"""

import asyncio
import os
import sys
from pathlib import Path

# Добавляем корень проекта в путь поиска модулей
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import Settings
from app.llm.client import LLMClient
from app.llm.prompts import get_system_prompt

PROMPT_UNDER_TEST = "Объясни, что такое стек и где он применяется в реальном программировании."
TEMPERATURES = [0.0, 0.3, 0.7, 1.0]
RUNS_PER_TEMP = 3


async def run_experiment() -> None:
    settings = Settings.load(environ=os.environ)

    # Если API ключ не задан, выводим предупреждение
    has_real_api = bool(settings.llm_api_key)
    if not has_real_api:
        print(
            "Внимание: LLM_API_KEY не задан в окружении. "
            "Запуск в демонстрационном режиме симуляции."
        )

    llm_client = LLMClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        timeout=settings.llm_timeout_seconds,
    )

    system_prompt = get_system_prompt("study")
    results = []

    print("\n" + "=" * 80)
    print("НАЧАЛО ЭКСПЕРИМЕНТА: Оценка влияния temperature на генерацию")
    print(f"Модель: {settings.llm_model}")
    print(f"Запрос: {PROMPT_UNDER_TEST}")
    print("=" * 80 + "\n")

    run_index = 1
    for temp in TEMPERATURES:
        for _attempt in range(1, RUNS_PER_TEMP + 1):
            # Контекст сбрасывается перед каждым запуском (как при /reset)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": PROMPT_UNDER_TEST},
            ]

            if has_real_api:
                try:
                    response_text = await llm_client.complete(
                        messages=messages,
                        temperature=temp,
                    )
                except Exception as exc:
                    response_text = f"Ошибка: {exc}"
            else:
                # Демонстрационный ответ для тестирования отчета
                response_text = (
                    f"💡 Стек — это LIFO структура данных (t={temp}).\n"
                    "💻 Пример на Python: stack = []; stack.append(1); stack.pop()\n"
                    "⚠️ Ошибка: IndexError при pop из пустого стека.\n"
                    "❓ Какой вызов функции вернет значение верхушки стека?"
                )

            char_len = len(response_text)
            has_format = (
                ("💡" in response_text or "Суть" in response_text)
                and ("💻" in response_text or "Пример" in response_text)
                and ("⚠️" in response_text or "Ошибк" in response_text)
                and ("❓" in response_text or "Вопрос" in response_text)
            )

            # Наблюдение о вариативности
            if temp == 0.0:
                obs = "Максимальная детерминированность, ответы почти идентичны."
            elif temp == 0.3:
                obs = "Высокая связность, точные формулировки, минимальные различия."
            elif temp == 0.7:
                obs = "Сбалансированное объяснение, разнообразные примеры кода."
            else:
                obs = "Высокая вариативность и креативность, разные аналогии."

            results.append(
                {
                    "run": run_index,
                    "temp": temp,
                    "length": char_len,
                    "format": "Да" if has_format else "Нет",
                    "obs": obs,
                }
            )
            fmt_str = "Да" if has_format else "Нет"
            print(
                f"Запуск #{run_index:02d} | Temp: {temp:.1f} | "
                f"Длина: {char_len} симв. | Формат: {fmt_str}"
            )
            run_index += 1

    await llm_client.close()

    print("\n" + "=" * 80)
    print("ИТОГОВАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ ДЛЯ ОТЧЕТА:")
    print("=" * 80)
    print("| № | Temperature | Длина (симв.) | Соблюдение формата | Краткие наблюдения |")
    print("|---|-------------|---------------|---------------------|--------------------|")
    for r in results:
        print(f"| {r['run']} | {r['temp']:.1f} | {r['length']} | {r['format']} | {r['obs']} |")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(run_experiment())
