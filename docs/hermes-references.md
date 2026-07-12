# Hermes — где искать документацию и исходники

Плагин живёт внутри Hermes Agent. Когда нужен API хоста (хуки, инструменты, скиллы,
платформы, `ctx`) — сначала сюда.

## Онлайн-доки (developer-guide)

База: <https://hermes-agent.nousresearch.com/docs/>

| Ссылка | О чём |
|---|---|
| [developer-guide/plugins](https://hermes-agent.nousresearch.com/docs/developer-guide/plugins) | Обзор плагин-системы: `register(ctx)`, полный список хуков, `register_hook`. Хук `pre_llm_call` — единственный канал инъекции контекста в **реактивный** ход (возвращает `{"context": …}`, вклеивается в user-сообщение, НЕ в системный промпт → кэш-префикс цел). |
| [developer-guide/adding-tools](https://hermes-agent.nousresearch.com/docs/developer-guide/adding-tools) | LLM-инструменты: `ctx.register_tool(name, toolset, schema, handler, …)`. `schema` — JSON Schema; `handler(args, **kwargs)` → **JSON-строка** (`json.dumps`, не dict), ошибки как `{"error": …}` и **не кидать**. Попадает в tool-list системного промпта — существо зовёт сам. |
| [developer-guide/creating-skills](https://hermes-agent.nousresearch.com/docs/developer-guide/creating-skills) | Скиллы (`SKILL.md` + `ctx.register_skill(name, path, description)`). Плагинные скиллы **не** индексируются в `<available_skills>` — только явная загрузка `plugin:name` через `skill_view()`. Инвокация: моделью / `/slash` / cron-blueprint. |
| [developer-guide/adding-platform-adapters](https://hermes-agent.nousresearch.com/docs/developer-guide/adding-platform-adapters) | Платформенные адаптеры: `ctx.register_platform(name, adapter_factory, …)`, наследуют `BasePlatformAdapter`, реализуют `connect()`/`disconnect()`/`send()`. Так плагин хостит существо как supervised-платформу (`adapters/being_platform.py`). |

## Локальные исходники (source of truth — читать при расхождении с доками)

- **Hermes agent:** `~/.hermes/hermes-agent/`
  - `hermes_cli/plugins.py` — `PluginContext` (`register_tool` ~:389, `register_command` ~:527, `register_platform` ~:929, `register_hook` ~:1156, `register_skill` ~:1196), `VALID_HOOKS` (~:135), `invoke_hook` (~:1890).
  - `agent/turn_context.py` — где `pre_llm_call` реально срабатывает (`build_turn_context`, ~:478); какие kwargs получает колбэк (`user_message`, `conversation_history`, `session_id`, `platform`, `sender_id`, `is_first_turn`, `model`) и как `{"context": …}` вклеивается.
  - `plugins/` — эталонные плагины (memory/retaindb, observability/langfuse, platforms/raft) — рабочие примеры хуков/инструментов.
- **Live-клон нашего плагина:** `~/.hermes/plugins/lifemodel` (из `origin/main`; `gateway_core.py` с `inject_proactive_turn` — проактивный ход).

> ⚠️ Доки — вторичны; при сомнении смотри исходник в `~/.hermes/hermes-agent/`. Плагин крутится в **venv Hermes** (`~/.hermes/hermes-agent/venv`) — рантайм-код только stdlib + что даёт Hermes.
