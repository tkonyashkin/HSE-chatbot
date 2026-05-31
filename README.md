# HSE Admissions RAG Chatbot

Чат-бот для ответов на вопросы о поступлении на бакалавриат НИУ ВШЭ. В репозитории есть:

- парсер страниц приемной кампании;
- подготовленные JSON-данные за 2026 год;
- chunking + локальный Qdrant-индекс;
- retrieval/generation/agentic RAG;
- Telegram-бот;
- golden set и результаты экспериментов.

## Установка

Нужен Python 3.11 и `uv`.

```bash
uv sync
```

Создайте `.env`:

```bash
cp .env.example .env
```

Заполните:

```dotenv
OPENROUTER_API_KEY=...
OPENROUTER_MODEL=anthropic/claude-haiku-4.5
TELEGRAM_BOT_TOKEN=...
HSE_BOT_CONVERSATION_DB=data/bot/conversations.db
HSE_KNOWLEDGE_DIR=data/knowledge/2026
```

`OPENROUTER_API_KEY` нужен для парсинга через LLM, генерации ответов, agentic RAG и LLM-экспериментов. Для чистого chunking/index/retrieval без генерации он не нужен.

## Индекс

В репозитории уже есть распарсенные данные в `data/programs/2026` и `data/knowledge/2026`. Локальный индекс не коммитится, его нужно собрать:

```bash
uv run hse-rag chunk
uv run hse-rag index
```

После этого можно проверить поиск и ответ:

```bash
uv run hse-rag query "Какие экзамены нужны на программную инженерию?"
uv run hse-rag agent "Сравни программную инженерию и прикладную математику"
```

## Telegram-бот

Убедитесь, что заполнены `TELEGRAM_BOT_TOKEN`, `OPENROUTER_API_KEY`, `HSE_BOT_CONVERSATION_DB` и собран индекс.

```bash
uv run hse-bot
```

SQLite-файл истории будет создан по пути из `HSE_BOT_CONVERSATION_DB`.

## Обновление данных

Если нужно заново собрать данные с сайта:

```bash
uv run hse-parser crawl
uv run hse-parser parse-knowledge
uv run hse-parser parse-programs
```

Полезные варианты:

```bash
uv run hse-parser parse-programs --limit 5
uv run hse-parser parse-programs --slugs se,ami,math
uv run hse-parser parse-programs --rerun-failed
```

После обновления данных пересоберите индекс:

```bash
uv run hse-rag chunk
uv run hse-rag index
```

## Эксперименты

Golden set:

- `experiments/golden_set/validation.yaml`
- `experiments/golden_set/test.yaml`
- source labels: `experiments/golden_set/sources/*.tsv`

Перед retrieval-экспериментами соберите индекс:

```bash
uv run hse-rag chunk
uv run hse-rag index
```

Воспроизвести validation-эксперименты:

```bash
uv run hse-rag experiment preprocessing
uv run hse-rag experiment reranker
uv run hse-rag experiment chunking
uv run hse-rag experiment embedder
uv run hse-rag experiment hybrid
```

LLM-варианты для preprocessing/reranker:

```bash
uv run hse-rag experiment preprocessing --llm
uv run hse-rag experiment reranker --llm
```

Эксперимент генерации на test split:

```bash
RERANKER_MODEL=deepseek/deepseek-v4-flash \
RERANKER_EXTRA_BODY='{"reasoning":{"enabled":false}}' \
uv run hse-rag experiment generation_method \
  --queries-path experiments/golden_set/test.yaml \
  --sources-path experiments/golden_set/sources/test.tsv \
  --output-dir experiments/results/test
```

Standalone-эксперименты:

```bash
uv run python -m rag.experiments.eval_plan_accuracy
uv run python -m rag.experiments.eval_executor_model
```

Результаты пишутся в `experiments/results/validation` или в указанный `--output-dir`.

## Проверки

```bash
uv run ruff check .
uv run mypy .
uv run python -m compileall -q bot parser rag
```
