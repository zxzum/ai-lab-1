"""Утилита для разбиения длинных сообщений перед отправкой в Telegram."""


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
