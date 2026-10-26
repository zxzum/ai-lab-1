import asyncpg

from app.config import Settings


async def create_pool(settings: Settings) -> asyncpg.Pool:
    # Небольшого пула достаточно для учебной ВМ. Таблицы добавляют студенты.
    pool = await asyncpg.create_pool(
        host=settings.postgres_host,
        port=settings.postgres_port,
        database=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password,
        min_size=1,
        max_size=5,
        timeout=5,
        command_timeout=5,
    )
    try:
        await pool.fetchval("SELECT 1", timeout=3)
    except BaseException:
        await pool.close()
        raise
    return pool


async def init_db(pool: asyncpg.Pool) -> None:
    """Создает необходимые таблицы и индексы в PostgreSQL при запуске."""
    async with pool.acquire() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_settings (
                user_id BIGINT PRIMARY KEY,
                mode VARCHAR(32) NOT NULL DEFAULT 'study',
                temperature REAL NOT NULL DEFAULT 0.7,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS dialog_messages (
                id BIGSERIAL PRIMARY KEY,
                user_id BIGINT NOT NULL,
                role VARCHAR(16) NOT NULL,
                content TEXT NOT NULL,
                mode VARCHAR(32) NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );

            CREATE INDEX IF NOT EXISTS idx_dialog_messages_user_mode_created
                ON dialog_messages (user_id, mode, created_at ASC);

            ALTER TABLE user_settings ADD COLUMN IF NOT EXISTS timezone VARCHAR(64);

            CREATE TABLE IF NOT EXISTS reminders (
                id BIGSERIAL PRIMARY KEY,
                owner_id BIGINT NOT NULL,
                text TEXT NOT NULL,
                remind_at TIMESTAMPTZ NOT NULL,
                timezone VARCHAR(64) NOT NULL,
                idempotency_key VARCHAR(64) NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE (owner_id, idempotency_key)
            );

            -- Подготовленное действие, ожидающее подтверждения. UNIQUE (user_id, update_id)
            -- не даёт повторной доставке одного Telegram update создать вторую операцию.
            CREATE TABLE IF NOT EXISTS pending_actions (
                id VARCHAR(32) PRIMARY KEY,
                user_id BIGINT NOT NULL,
                update_id BIGINT NOT NULL,
                request_id VARCHAR(16) NOT NULL,
                tool_name VARCHAR(64) NOT NULL,
                arguments JSONB NOT NULL,
                timezone VARCHAR(64) NOT NULL,
                status VARCHAR(16) NOT NULL DEFAULT 'pending',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                expires_at TIMESTAMPTZ NOT NULL,
                UNIQUE (user_id, update_id)
            );

            CREATE TABLE IF NOT EXISTS agent_audit (
                id BIGSERIAL PRIMARY KEY,
                user_id BIGINT NOT NULL,
                request_id VARCHAR(16) NOT NULL,
                action VARCHAR(16) NOT NULL,
                tool_name VARCHAR(64),
                args_summary TEXT NOT NULL DEFAULT '',
                validation VARCHAR(16) NOT NULL,
                execution VARCHAR(24) NOT NULL,
                reason TEXT NOT NULL DEFAULT '',
                duration_ms INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );

            CREATE INDEX IF NOT EXISTS idx_agent_audit_user_id ON agent_audit (user_id, id DESC);
            """
        )
