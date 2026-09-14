"""Утилита для разбиения длинных сообщений и безопасного форматирования в Telegram."""

import html
import re


def split_message(text: str, max_length: int = 4096) -> list[str]:
    """Разбивает текст на части длиной не более max_length символов.

    Пытается аккуратно разбивать по переносам строк или пробелам,
    чтобы не разрезать слова. Если подходящего разделителя нет,
    выполняет жесткий срез по max_length.
    Порядок частей сохраняется, текст не теряется.
    """
    if not text:
        return []
    if len(text) <= max_length:
        return [text]

    chunks: list[str] = []
    remaining = text

    while len(remaining) > max_length:
        chunk_candidate = remaining[:max_length]

        # 1. Пробуем перенос абзаца
        split_idx = chunk_candidate.rfind("\n\n")
        if split_idx > 0:
            split_point = split_idx + 2
        else:
            # 2. Пробуем одиночный перенос строки
            split_idx = chunk_candidate.rfind("\n")
            if split_idx > 0:
                split_point = split_idx + 1
            else:
                # 3. Пробуем пробел
                split_idx = chunk_candidate.rfind(" ")
                if split_idx > 0:
                    split_point = split_idx + 1
                else:
                    # 4. Жесткий срез
                    split_point = max_length

        chunks.append(remaining[:split_point])
        remaining = remaining[split_point:]

    if remaining:
        chunks.append(remaining)

    return chunks


def markdown_to_telegram_html(text: str) -> str:
    """Преобразует Markdown от LLM в безопасный HTML, поддерживаемый Telegram.

    Поддерживает:
    - Блоки кода: ```lang ... ``` -> <pre><code class="language-lang">...</code></pre>
    - Инлайн-код: `code` -> <code>code</code>
    - Заголовки: # ... -> <b>...</b>
    - Жирный: **text** -> <b>text</b>
    - Курсив: *text* или _text_ -> <i>text</i>
    - Списки: * item или - item -> • item
    - Экранирование спецсимволов HTML (&, <, >) в обычном тексте.
    """
    if not text:
        return ""

    # Автоматически закрываем непарный блок кода, если он был оборван
    if text.count("```") % 2 == 1:
        text += "\n```"

    # 1. Защищаем блоки кода от обработки другими правилами
    code_blocks: list[str] = []

    def save_code_block(match: re.Match[str]) -> str:
        lang = (match.group(1) or "").strip().lower()
        code = html.escape(match.group(2).strip("\n"))
        cls = f' class="language-{lang}"' if lang else ""
        placeholder = f"\x00CB{len(code_blocks)}\x00"
        code_blocks.append(f"<pre><code{cls}>{code}</code></pre>")
        return placeholder

    text = re.sub(r"```([a-zA-Z0-9_\-+]*)\n?(.*?)```", save_code_block, text, flags=re.DOTALL)

    # 2. Защищаем инлайн-код
    if text.count("`") % 2 == 1:
        text += "`"

    inline_codes: list[str] = []

    def save_inline_code(match: re.Match[str]) -> str:
        code = html.escape(match.group(1))
        placeholder = f"\x00IC{len(inline_codes)}\x00"
        inline_codes.append(f"<code>{code}</code>")
        return placeholder

    text = re.sub(r"`([^`\n]+)`", save_inline_code, text)

    # 3. Экранируем HTML-спецсимволы в остальном тексте
    text = html.escape(text)

    # 4. Заголовки: ### Header -> <b>Header</b>
    text = re.sub(r"^(#{1,6})\s+(.+)$", r"<b>\2</b>", text, flags=re.MULTILINE)

    # 5. Жирный текст: **text** -> <b>text</b>
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)

    # 6. Списки: * item или - item -> • item (до курсива, чтобы * не стали <i>)
    text = re.sub(r"^\s*[\*\-\+]\s+", "• ", text, flags=re.MULTILINE)

    # 7. Курсив: *text* или _text_ -> <i>text</i>
    text = re.sub(r"(?<!\w)\*(\S.*?\S|\S)\*(?!\w)", r"<i>\1</i>", text)
    text = re.sub(r"(?<!\w)_(\S.*?\S|\S)_(?!\w)", r"<i>\1</i>", text)

    # 8. Восстанавливаем сохраненные блоки кода
    for i, block in enumerate(code_blocks):
        text = text.replace(f"\x00CB{i}\x00", block)
    for i, code in enumerate(inline_codes):
        text = text.replace(f"\x00IC{i}\x00", code)

    return text
