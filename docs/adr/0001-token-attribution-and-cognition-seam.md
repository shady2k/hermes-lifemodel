# ADR-0001 — Атрибуция токенов и шов когниции: отложить до приватной когниции

- **Статус:** Принято
- **Дата:** 2026-07-11
- **Контекст-беды:** `lm-fib.7.8` (токен-метрика `llm_observed_*`), `lm-fib.7` (эпик телеметрии), `lm-fgs` (апстрим `inject_proactive_turn`/`register_gateway_service`)
- **Участники:** shady + исследование с ассистентом, критический разбор — codex

---

## Контекст

`lm-fib.7.8` просил domain-метрику расхода токенов существом: `llm_observed_calls_total`,
`llm_observed_tokens_total{kind=prompt|completion|total, model}`, `llm_observed_missing_usage_total`.
Бид был написан в предположении, что токены доступны на хуке `post_llm_call`. Разведка показала,
что это не так, и что «правильная» атрибуция токенов упирается в **структуру текущего механизма
пробуждения когниции** (reach-in), а не в отсутствие данных.

Цель существа шире проактива: оно должно будет ещё **думать / мечтать / спать** — часть этих
активностей будит когницию (LLM-вызовы). Мы хотим мерить токены, которые тратит **именно наш
плагин, когда будит когницию**, с разбивкой по виду когниции (`reason=proactive|dream|sleep|think`).

## Как сейчас работает пробуждение когниции (reach-in)

Существо — 0-LLM автономный мозг (`BeingAdapter`, платформенный адаптер), тикает ~60с через
`SupervisedLoop`. Когда решает потянуться к владельцу:

```
proactive_tick → ReachInEgress.reach_out (adapters/reachin.py:49)
              → gateway_core.inject_proactive_turn (gateway_core.py:75)
              → adapter.handle_message(MessageEvent(text=impulse, internal=True))  [чужой DM-адаптер]
```

Свойства, определяющие всё остальное:

- **Fire-and-forget.** `inject_proactive_turn` планирует `handle_message` на gateway-loop через
  `asyncio.run_coroutine_threadsafe` и **сразу** возвращает `ReachOutcome` (DELIVERED/UNAVAILABLE/FAILED,
  `gateway_core.py:132-134`). Future планирования выбрасывается. Тёрн исполняется асинхронно на
  **сессии владельца** (тот же `session_id`, что реальный DM).
- **Приватные внутренности.** Reach-in лезет в `gateway.run._gateway_runner_ref()` +
  `_gateway_loop` / `_build_process_event_source` / `_running_agents` (version-guard
  `_REQUIRED_RUNNER_ATTRS`). Это НЕ публичный plugin-API, но ту же поверхность использует
  собственный `tools/send_message_tool.py` Hermes.
- **Исход наблюдается хуком.** Sent vs `[SILENT]` ловится асинхронно `make_post_llm_observer`
  (`hooks.py`), который стартует `ASYNC_COMPLETION`-фрейм и разрешает pending-желание. Опознание
  «наш ход» — по `IMPULSE_LABEL_PREFIX` в `user_message`.

## Что где лежит (карта хуков и токенов в Hermes)

| Что | Где | Несёт |
|---|---|---|
| `post_llm_call` (эмит) | `agent/turn_finalizer.py:393` | `session_id, user_message, assistant_response, conversation_history, model, platform` — **токенов НЕТ** |
| `post_api_request` (эмит) | `agent/conversation_loop.py:4283` | `usage` (dict токенов) + `model` + `provider` + `turn_id` + ... — **токены ЗДЕСЬ** |
| форма `usage` | `run_agent.py:2242` → `asdict(CanonicalUsage)` (`agent/usage_pricing.py:31`) | `input/output/cache_read/cache_write/reasoning_tokens`, `request_count`, `prompt_tokens`, `total_tokens` |
| `run_conversation` (публичный headless API) | `run_agent.py:5787` → `agent/conversation_loop.py` | возвращает dict; `agent/turn_finalizer.py:426-447` кладёт `final_response` + `prompt/completion/total_tokens` (из `agent.session_*_tokens`) |
| `append_message` (запись в транскрипт) | `hermes_state.py:3441` | дописать строку в сессию + инкремент `message_count` |

`AIAgent` как публичная библиотека подтверждён докой (`docs/guides/python-library`): `from run_agent import AIAgent`;
`agent.run_conversation(...)`/`chat(...)` headless. **Level-1 спайк** (scratchpad) подтвердил: агент
конструируется in-process, `run_conversation`/`chat` на месте, токен-аккумуляторы `session_*_tokens`
существуют и стартуют с 0.

## Рассмотренные варианты

### A. `turn_id`-джойн хуков (метрика поверх текущего reach-in)
Наблюдатель на `post_api_request` буферит `usage` по `turn_id`; `post_llm_call` классифицирует
(наш проактив по `IMPULSE_LABEL_PREFIX`) и флэшит атрибуцию. **Работает на текущей архитектуре**,
не трогает доставку. Минусы: (1) опознание «наш ход» по текстовому префиксу — **хрупко**;
(2) stateful-буфер; (3) едем на механизме, который всё равно заменим.

### B. Свой шов когниции (headless `AIAgent`)
Существо строит **свой** `AIAgent` (публичный API), гоняет `run_conversation` headless, получает
`{final_response, tokens}` напрямую. Атрибуция тривиальна и робастна (`reason=<вид>` мы знаем сами),
`dream/sleep/think` ложатся как виды когниции со своим system-prompt/toolset/(деш.)моделью.
**Чист для ПРИВАТНОЙ когниции (без доставки владельцу).**

Но для **owner-facing** проактива B упирается в стену **«доставить владельцу» ⇔ «мутировать историю
его сессии» — нельзя расцепить, если нужна непрерывность** (существо должно помнить своё обращение —
бывший баг `lm-pbm`):
- Ручной `append_message` + `adapter.send()` — **неправильно**: split-brain с in-memory историей
  живого кешированного агента `_running_agents[session_key]`; обходит атомарную turn-финализацию.
- headless-когниция + отдельный gateway-тёрн на доставку — **удвоение LLM-стоимости**.
- «Позвать то, что зовёт reach-in» — не даёт токенов: слои с возвратом отдают только текст
  (`runner._handle_message → Optional[str]`, `run.py:8853`) или `None` (`adapter.handle_message`,
  `base.py:4585`), а токен-несущий `run_conversation` (`run.py:18621`) погребён под session-lifecycle,
  который безопасно не обойти.

### Критика codex (ключевое)
- `adapter.send()` — неверный примитив доставки, если нужна непрерывность.
- Отдельный `AIAgent` не повторяет живой gateway-тёрн (context-обёртки, resume/interruption,
  model-override, transcript-гигиена) — ок для private, риск для owner-facing.
- Токены `session_*` кумулятивные → свежий агент на пробуждение ИЛИ diff.
- Ретайр `post_llm_call`-observer'а требует переноса его НЕ-метрик обязанностей (классификация,
  трейсы, `ASYNC_COMPLETION`-фрейм, suppression). И: headless `run_conversation` **всё равно фаярит
  `post_llm_call`** — риск двойного разрешения pending-проактива, если старый observer жив.
- Silence-классификатор `lifemodel` (`hooks.py:102`, substring-`[SILENT]`) ≠ gateway
  (`gateway/response_filters.py:56`, whole-response).

## Решение

**Отложить и `lm-fib.7.8` (токен-метрика), и подход B (свой шов когниции) до появления первого
приватного вида когниции.** Закрыть `lm-fib.7.8` и эпик `lm-fib.7` (ядро телеметрии доставлено:
registry + manifest + сэмплер + `metrics.sqlite` + `/lifemodel stats`).

Обоснование: на reach-in-архитектуре **чистой атрибуции токенов не существует** — оба пути упираются
в структуру (нет возвращаемого значения; хрупкий текстовый префикс; owner-facing нельзя расцепить).
Делать хрупкую P3-метрику на механизме, который заменим, — не окупается. Токены существа становятся
осмысленными и **достаются даром и робастно** ровно там, где появится **приватная когниция** (свой
`run_conversation` → токены в результате).

## Последствия

- Токен-метрики нет до приватной когниции — приемлемо (P3, наблюдаемость, не поведение).
- Owner-facing проактив **остаётся на reach-in** — он корректен по непрерывности; его единственный
  «грех» (приватные внутренности) — отдельная тема апстрима `lm-fgs`, не повод расцеплять.
- Ядро телеметрии (`lm-fib.7`) доставлено и закрыто; будущая токен-ось встроится в тот же
  `MetricRegistry` (`reason` уже в закрытом наборе лейблов).

## Триггер пересмотра

Вернуться к этому ADR, когда заводится **первый приватный вид когниции** (`think`/`dream`/`sleep`):
- реализовать его через B (свой `AIAgent` + `run_conversation` headless, штатный runtime —
  `_credential_pool_for_provider` / `get_default_model_for_provider`, либо публичный
  `hermes_cli.runtime_provider.resolve_runtime_provider`);
- токен-атрибуция приезжает из результата (`prompt/completion/total_tokens`), помеченная `reason=<вид>`
  + `model`; свежий агент на пробуждение → итог = per-cognition;
- owner-facing проактив мигрирует на чистый шов только после `lm-fgs` (апстрим first-class API),
  с явной стратегией записи хода в сессию (не ручной `append_message`).
