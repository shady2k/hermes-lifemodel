# Наблюдаемость как принудительный инвариант ядра — дизайн

**Дата:** 2026-07-09 · **Статус:** GO (2 codex-ревью пройдены), готов к плану · **Версия:** v1.3
**Беды:** `lm-edu` (наблюдаемость вшита в ядро) · мотивирующий потребитель — `lm-uft` (отладка проактивного контакта). Parent-эпик `lm-md6`.
**Родословная:** конкретизирует `docs/hla.md` §4.1 (W3C trace-context, `trace_id` = ключ соединения durable-провенанса и эфемерных спанов) и NFR9. Доводит до полноты фундамент ребилда (`docs/superpowers/specs/2026-07-08-core-single-spine-design.md` §5). Ревью: codex (независимый, с доступом к репо); v1.1 инкорпорирует его must-fix'ы + решение убрать `events.jsonl`; v1.2 — по итогу разведки `~/.hermes`: текстовый лог идёт через нативный stdlib `logging`, как у всех плагинов Hermes; structlog/loguru удалены как зависимости.

> **Что изменилось в v1.3 (2-й codex-ревью, must-fix'ы):**
> 1. **Ретенция не режет in-flight-трассу** — защита `trace_id` с живым якорем в state / `resolved_at IS NULL` + grace-окно (иначе половинка при позднем async-исходе).
> 2. **`agent.log ⊆ sqlite` даже под перегрузкой** — durable-first: проекции пишем только после успешного enqueue; при `queue.Full` — счётчик дропов, без проекций.
> 3. **`loglevel` без structlog** — `setLevel` на `logging.getLogger("lifemodel")`; никогда `basicConfig`/хендлеры (setup владеет Hermes); `record_id` для дедупа deque-оверлея; State — поле dataclass, не миграция таблицы; очистка якоря во всех clear-sites.

> **Что изменилось в v1.2 (после разведки `~/.hermes`):**
> 1. **structlog/loguru — вон.** В venv Hermes их нет; **каждый** плагин Hermes (встроенные `image_gen/*`, `platforms/*`) логирует через голый stdlib `logging.getLogger(__name__)`, а Hermes (`hermes_logging.py::setup_logging`) сам роутит в `~/.hermes/logs/agent.log` с ротацией, **редакцией секретов**, async-`QueueListener`, session-тегом. Наш structlog-сетап переизобретал это и хуже (не установлен → тихий фолбэк). `configure()`/`EventTee`/`_StdlibEventLogger`/`get_logger`-дуализм — удаляются.
> 2. **Человеческий лог = stdlib `logging.getLogger("lifemodel")`** (нативная конвенция Hermes). Durable-трасса = наш `observability.sqlite` (Hermes даёт текст-логи, но не trace-store по `trace_id` — вот это законно наше).

> **Что изменилось в v1.1 (после ревью codex + решения владельца):**
> 1. `events.jsonl` **удаляется** — в проде его никто не читает (только тест-ассерты), лишнее «второе место». `EventSink` → in-memory `deque`. **Одно** durable-структурное место — `observability.sqlite`.
> 2. Мост корреляции **load-bearing → живёт в драгоценном `runtime_state`** (не в одноразовом trace DB). `trace_correlations` — лишь удобный индекс.
> 3. Через async-запуск едет **`traceparent` (trace_id + span_id + flags)**, не голый `trace_id`.
> 4. `ActiveSpan` (мутабельный handle) отделён от frozen `TraceContext`.
> 5. Определена **политика miss** для `post_llm` (`orphan_async_outcome`, никогда не цеплять к чужой трассе).
> 6. Определён **read-your-writes**, идемпотентный жизненный цикл писателя, порядок always-green, allowlist AST-гарда.

> **Эволюция относительно §5 ребилда (не противоречие).** Ребилд: «источник истины — структурные логи, OTel best-effort» — чтобы не зависеть от опционального OTel. lm-edu сохраняет ту же цель, но источником истины делает **durable-локальный sqlite** (stdlib, не OTel), а текстовый человеческий хвост — нативный stdlib `logging` → `agent.log` (проекция того же потока со `trace_id`, не отдельный источник для сшивки).

---

## 1. Проблема (что чиним и почему по-настоящему)

Ребилд (T1, `lm-dge`) заложил фундамент: обязательный trace+span на компонент тика, suppression-span с закрытым enum из 12 причин, «no-log-without-span», structlog. **Но фундамент не доведён — данные размазаны.** Чтобы объяснить ОДНУ проактивную попытку, приходится вручную сшивать **пять** источников: `events.jsonl` + sqlite `runtime_state` + `memory_records` + `agent.log` (structlog INFO) + внешний hindsight (куда уходят reasoning/verdict). Это ровно то, что обещал убрать NFR9.

Разрыв структурный (подтверждён чтением кода):

1. **`trace_id` доходит до `events.jsonl` ТОЛЬКО на `suppression`-событиях.** `EventTee._emit` (`log.py:109`) передаёт в сток только явные kwargs; связанный trace-контекст (structlog contextvars: `trace_id/span_id/tick`) до JSONL-стока не доходит. `tick`, `proactive_tick`, `proactive_outcome` ложатся **без** `trace_id` → сгруппировать по трассе нельзя.
2. **Нет моста `correlation_id → trace_id`.** Запуск минтит `correlation_id = f"proactive-{now}"` и кладёт `pending_proactive_id` в `runtime_state` (`core/cognition.py:151`), но trace не несёт. `LaunchProactive` (`core/intents.py:53`) полей trace не имеет. reach-in (`gateway_core.inject_proactive_turn`) traceparent не передаёт. `post_llm` observer (`hooks.py:215`) читает `pending_proactive_id` из state (`hooks.py:222`) и корреллирует по нему; **известного/используемого поля traceparent на входе нет** (`**_ignored` может нести произвольные host-поля, но traceparent среди них не заявлен). Итог: исходный тик, доставка, async-ход вердикта, разрешающий тик — **четыре разрозненные трассы**.
3. **Span несёт только W3C-идентификаторы.** `TraceContext` (`ports/tracer.py:37`) — frozen, атрибутного мешка нет. Значения решения (`u`, `effective_pressure`, `energy`, гейты) живут в `State`/transient-сигналах и до span'а не доходят → span **не само-объясним**.
4. **`trace_id` в `memory_records` погребён внутри `payload_json._provenance`** (`domain/objects/base.py:212`) — невидим для SQL.
5. **`events.jsonl` в проде не читается** (`debug.py` строит из state/memory; `why` — из графа памяти; `EventSink.read()` зовут только тесты). Вьюера трассы нет; чистая точка врезки есть (`_SUBCOMMANDS` `__init__.py:57`, `dispatch` `:199`, образец — `why_for_dir` `state_commands.py:729`).

**Управляющий закон:** HLA §4.1 «`trace_id` — ключ соединения durable-провенанса и эфемерных спанов; why-graph»; §9.4 «ничего невидимого»; NFR9. Требование владельца (2026-07-09): «Наблюдаемость должна быть вшита в ядро. Не городить наблюдаемость для конкретного кейса.» → решение **generic, не per-case**; и «два места не нужны» → `events.jsonl` убрать.

## 2. Рамка (что делаем и чего НЕ делаем)

**Делаем** наблюдаемость такой, что **уронить trace/span структурно нельзя**, значения решения едут атрибутами span'а, весь durable-трейс лежит в **одном** запрашиваемом стоке, а любой «почему» (в т.ч. async-исход проактивного хода) читается по одному `trace_id`. Проактивная отладка `lm-uft` — тривиальный побочный эффект.

**НЕ делаем (YAGNI, §8):** хард-зависимость OTel; in-band W3C-проброс через границу Hermes (её канал заголовок не несёт — §6); внутренний LLM-ход сущности как child нашего тика (это трасса Hermes); не оставляем `events.jsonl`; не выносим `memory_records.trace_id` в SQL-колонку в этой фазе.

## 3. Четыре закона инварианта

1. **Trace/span нельзя уронить — by construction.** Три замка (§4.1): тип (обязательный `trace`/`ActiveSpan` в контексте + обязательный `origin_traceparent` на async-интентах) · API (обработчику доступен только span-bound логгер) · enforcement-гард (AST-проверка escape-hatch'ей в CI, §4.5).
2. **State DB драгоценен + fail-closed; Trace DB одноразов + fail-open.** Durable-трасса — в **отдельном** sqlite-файле; её потеря не меняет поведение сущности. **Load-bearing якорь связности (`origin_traceparent`) живёт в драгоценном `runtime_state`** рядом с `pending_proactive_id` — чтобы потеря trace DB не рвала связь трасс.
3. **Запись трасс асинхронна, неблокирующа, fail-open.** Тик никогда не ждёт I/O; не записалось — молча игнорируем (§4.2). Исключение — якорь связности (закон 2): он коммитится синхронно вместе с состоянием запуска.
4. **Generic, не per-case.** Ни одного проактив-специфичного поля в схеме трасс; мост работает для любого async-запуска; вьюер рендерит любую трассу одинаково.

## 4. Архитектура — шесть пилонов

### 4.1 `ActiveSpan` + атрибуты · non-optional trace · SpanLogger (no-log-without-span)

- **Два типа, разделены (правка codex).** `TraceContext` (`ports/tracer.py:37`) остаётся **frozen** носителем W3C-идентификаторов (`trace_id/span_id/parent_span_id/flags`). Новый **`ActiveSpan`** — мутабельный handle: оборачивает `TraceContext` + `attrs: dict`, `status`, `started_at/ended_at`. Компоненты получают `ActiveSpan`; `set(**attrs)` кладёт значения решения; `end(status=...)` закрывает.
- **`TickContext.trace`** (`core/component.py:51`) → обязательный (не `| None`). mypy гарантирует span в каждом обработчике.
- **Главный замок — `SpanLogger`.** Обработчик **не получает «голый» логгер**. `TickContext.logger` — `SpanLogger` над `(ActiveSpan, sink, writer)`, каждый `.info/.debug` САМ проставляет `trace_id/span_id/tick`. Голого логгера в тик-пайплайне взять неоткуда → баг §1.1 становится **невыразимым** (чиним корень, а не патчим `EventTee`).
- `emit_suppression_span` (`core/suppression.py:89`) → на `SpanLogger` + значения атрибутами. Четыре пока не эмитящихся reason (`ACT_GATE_SILENT`, `EGRESS_UNAVAILABLE`, `EGRESS_FAILED`, `COMPONENT_FAILED`) подключаются к живым эмиттерам; `component_failed` (`coreloop.py:267`, сейчас голый INFO) → suppression-span.

### 4.2 Три «стока», один поток записи · read-your-writes · жизненный цикл

Тик-поток трогает только дешёвое; всё I/O — асинхронно.

- **In-memory ring (`deque`).** `EventSink` (`events.py:55`) перепрофилируется: **вместо файлового JSONL — потокобезопасный bounded `collections.deque`** (сохраняем интерфейс `emit`/`read`). Синхронный append — O(1), не I/O. Роли: свежесть для вьюера (read-your-writes) + эргономика тест-ассертов. `EVENTS_FILENAME`/`events.jsonl` **удаляются**.
- **Durable trace DB** — `observability.sqlite` (§4.3), запись **асинхронна** через один daemon writer-поток + bounded `queue.Queue`. Каждой записи присваивается **монотонный process-local `record_id`** (счётчик/UUID) — для дедупа при deque-оверлее (правка codex #5). `SpanLogger.emit` — **durable-first порядок** (правка codex #2): (1) `put_nowait` в очередь (async); (2) **только если enqueue удался** → append в `deque` + человеческая строка в `agent.log`; (3) очередь полна (`queue.Full`) → инкремент `observability_dropped_count`, **проекции НЕ пишем** (ни `agent.log`, ни `deque`), опц. rate-limited lifecycle-warning (не притворяется частью трассы). Так `agent.log ⊆ sqlite` даже под перегрузкой — «один источник» не рушится. sqlite-соединение живёт только в writer-потоке (thread-affine), свой WAL, батч-коммит, per-record swallow.
- **Человеческий live-хвост** — нативный stdlib `logging.getLogger("lifemodel")` → Hermes (`hermes_logging.py::setup_logging`) сам роутит в `~/.hermes/logs/agent.log` (ротация, редакция секретов, async-`QueueListener`, session-тег). Строка — **урезанная сводка** со `trace_id`, отсылающим к полной трассе в sqlite. structlog/loguru **не тянем** (§v1.2). **Мы НИКОГДА не зовём `logging.basicConfig` и не добавляем/снимаем хендлеры** — setup владеет Hermes; мы только `getLogger`.
- **Уровень логирования (правка codex #3).** Существующая команда `loglevel` (сейчас через `config.py:27`→`configure()`, `__init__.py:172`) перестраивается: `LOG_LEVEL_NAMES`/`parse_log_level` остаются простыми хелперами; применение уровня = `logging.getLogger("lifemodel").setLevel(level)` (+ уровень трейс-детализации для sqlite). Голый test/CLI-путь может быть без хендлеров — это ОК; тесты используют `caplog`.
- **Read-your-writes (правка codex).** `/lifemodel trace` перед чтением зовёт `writer.flush(timeout=…)` (дренаж очереди) **и** оверлеит `deque`, **дедуп по `record_id`** (флашнутое, но ещё живущее в deque, не задваивается) — иначе `last N` флапает/двоит.
- **Жизненный цикл (правка codex).** Писатель — **singleton per db-path** с refcount/идемпотентным стартом на `BeingAdapter.connect()` (`adapters/being_platform.py:95`), `flush`+stop на disconnect, защита от двойного старта, **reconnect-безопасность** (тест на reconnect). Крэш теряет хвост очереди — приемлемо (одноразов).

**Один источник истины + проекции — НЕ два места (инвариант против регресса в «5 источников»).** В `observability.sqlite` пишется **ВСЁ** (каждый спан/событие/значение решения/async-исход, keyed by `trace_id`) — он **самодостаточен**: любой «почему» отвечается чтением **только** его (`/lifemodel trace <id>`). `agent.log` и `deque` — **проекции того же потока `SpanLogger`**, не независимые источники: `agent.log` — урезанная человеческая сводка со ссылкой-`trace_id`, `deque` — эфемерный не-слитый хвост. Тест на регресс: *надо ли читать ДВА места, чтобы ответить на вопрос?* → **нет** (sqlite полон; проекции избыточны by design и одноразовы). Это категорически отличается от старой беды, где 5 источников несли **разные непересекающиеся куски** и их приходилось сшивать. Воркер **не** должен заводить второй durable-сток или писать в `agent.log` то, чего нет в sqlite.

### 4.3 `observability.sqlite` — отдельный, одноразовый

Инвариант закона 2. Отдельный файл рядом с `lifemodel.sqlite` в workspace-каталоге, своё подключение/WAL → тяжёлая запись не лочит state DB. Удалять/ротировать/`VACUUM` — когда угодно. Плоские строки, дерево по `parent_span_id`:

```sql
-- observability.sqlite (disposable, fail-open, WAL)
CREATE TABLE schema_version (version INTEGER NOT NULL);  -- даже одноразовому нужен (правка codex)

CREATE TABLE trace_spans (
  trace_id TEXT NOT NULL, span_id TEXT NOT NULL, parent_span_id TEXT,  -- NULL = корень
  component TEXT, tick INTEGER, started_at TEXT, ended_at TEXT,
  status TEXT,                          -- ok | suppressed | failed
  attrs_json TEXT,                      -- мешок значений решения
  PRIMARY KEY (trace_id, span_id)
);
CREATE INDEX ix_spans_trace ON trace_spans(trace_id);
CREATE INDEX ix_spans_tick  ON trace_spans(tick);

CREATE TABLE trace_events (               -- строки лога при span'е; ВКЛЮЧАЯ DEBUG-detail
  record_id INTEGER PRIMARY KEY,          -- монотонный process-local; дедуп deque-оверлея
  trace_id TEXT NOT NULL, span_id TEXT, tick INTEGER,
  event TEXT NOT NULL, ts TEXT NOT NULL, fields_json TEXT
);
CREATE INDEX ix_events_trace ON trace_events(trace_id);

CREATE TABLE trace_correlations (         -- удобный ИНДЕКС (не источник истины — тот в runtime_state)
  correlation_id TEXT PRIMARY KEY, origin_trace_id TEXT NOT NULL,
  origin_traceparent TEXT, kind TEXT,     -- generic: "proactive", но схема не per-case
  created_at TEXT NOT NULL, resolved_at TEXT
);
```

- **`trace_events` вбирает DEBUG-detail** (`proactive_reasoning`, `proactive_verdict_detail` — сейчас DEBUG-логи, `hooks.py:89/164`) под origin-трассой → схлопывает 5-й источник (за reasoning/verdict больше не идём в hindsight/DEBUG-логи).
- **Очистка старых логов (обязательный механизм, не «руками»).** Writer оппортунистически подрезает trace DB по **трём осям**, любая срабатывает первой (границы — конфигом, дефолты консервативные):
  - **по дням** — `max_age_days` (напр. 14): удалить трассы, где `started_at` старше порога;
  - **по количеству** — `max_traces` (напр. 5000): держать N последних корневых трасс, старшие — прочь;
  - **по размеру** — `max_bytes` (напр. 256 МБ): при превышении размера файла сносить старейшие трассы, пока не влезет.
  - **Единица удаления — целая трасса** (все `trace_spans`/`trace_events`/`trace_correlations` по `trace_id`), чтобы не оставлять «половинки». Прогон — редкий, из writer-потока (напр. раз в N коммитов / раз в M минут по счётчику), с `VACUUM`/`PRAGMA incremental_vacuum` по потребности. Fail-open: сбой очистки не роняет тик. `agent.log` чистит сам Hermes (`RotatingFileHandler`) — не наша забота.
  - **НИКОГДА не резать in-flight/неразрешённую трассу (правка codex #1, load-bearing).** Иначе ретенция снесёт корень/launch трассы `T` до прихода async-исхода, а `post_llm` допишет outcome под `T` → снова «половинка». **Защищённые от prune `trace_id`:** (а) `trace_id` из `runtime_state.pending_proactive_origin_traceparent`; (б) все `trace_correlations` с `resolved_at IS NULL`; (в) **grace-окно** для только что разрешённых (конфиг `resolved_grace_days`, дефолт +1 день после `resolved_at`; **не** привязано к `max_age_days` — иначе при большом age резолвнутые висят слишком долго под size-давлением) — на случай поздних дописей. Прунится только трасса, прошедшая порог **И** не защищённая. Тест: старая-за-лимитом + неразрешённая корреляция → НЕ прунится; после resolve+grace → прунится.

### 4.4 Мост корреляции: якорь в state (драгоценный), индекс в trace DB (одноразовый)

Разрыв §1.2 — из потери trace-контекста (никто не несёт), не из невозможности.

- **Внутри нашего кода** (тик → launch → доставка): `LaunchProactive` получает **обязательный `origin_traceparent`** (полный W3C: `trace_id`+`span_id`+flags — не голый `trace_id`, иначе не сделать дочерний span; правка codex). `proactive_tick`/reach-in его несут. Тип запрещает запуск без origin-трассы.
- **Якорь связности — в `runtime_state` (правка codex, критично).** При коммите запуска атомарно, рядом с `pending_proactive_id`, пишется **`pending_proactive_origin_traceparent`**. Это единственный durable-якорь, переживающий async-границу (state DB драгоценен). `trace_correlations` в trace DB — лишь **индекс/зеркало** для вьюера; его потеря не рвёт связь (origin берётся из state).
- **Очистка якоря — везде, где чистится `pending` (правка codex #4).** Любой путь, обнуляющий `pending_proactive_id`, обязан обнулять и `pending_proactive_origin_traceparent` (и `resolved_at` в индексе): rollback доставки (`core/proactive.py:87`), разрешение вердикта в агрегации, `force_wake`/`satiate`/`reset` в state-командах, factory-wipe. Иначе stale-якорь путает и инвариант, и защиту ретенции (§4.3).
- **На границе Hermes** in-band W3C-проброс недоступен: `MessageEvent` не имеет канала метаданных, который Hermes вернул бы; `post_llm` не заявляет traceparent на входе. Поэтому **out-of-band через state**: observer читает `pending_proactive_id` + `pending_proactive_origin_traceparent`, `child_of(parse_traceparent(...))`, ре-биндит `SpanLogger` и эмитит `proactive_outcome`/reasoning/verdict под origin-трассой. `verdict_signal` несёт `correlation_id` → агрегация следующего тика так же поднимает origin из state и эмитит span разрешения; `resolved_at` в индекс.
- **Политика miss (правка codex, load-bearing).** Если `origin_traceparent` в state есть → weave. Если **нет** (state потерял `pending` — это уже ломает и доменный вердикт, крупнее наблюдаемости) → **никогда не цеплять исход к свежей чужой трассе**; эмитить explicit `orphan_async_outcome` с `correlation_id`; вьюер показывает «async correlation missing». Потеря trace DB на weave **не влияет** (origin в state).

**Честный нюанс:** внутренний LLM-ход сущности остаётся под трассой **Hermes** — child'ом нашего тика не делаем. Трасса решения объясняет НАШЕ решение и его исход-как-мы-наблюдаем; наблюдаем в `post_llm` (наш код), он штампует origin.

### 4.5 Enforcement — AST-гард + allowlist

Типы/API закрывают штатный путь; escape-hatch'и (`get_logger()` напрямую — есть в `gateway_core.py:84`, `hooks.py:67`, `being_platform.py:58`; `logging`/`print`) — нет. Ruff сторонние плагины не исполняет → **AST-гард на stdlib**, gate в `make check` (`tests/architecture/test_trace_invariants.py`). Правила + **явная граница миграции (правка codex):**
- **`core/`-компоненты (тик):** только `SpanLogger`; `get_logger`/`logging`/`print` — запрещены.
- **Async-boundary handlers** (`hooks.py`, `gateway_core.py`, `core/proactive.py`): обязаны нести/ре-биндить трассу, где она есть (traceparent из launch / origin из state). Модульный логгер допустим только для pre-trace-сбоев (напр. `reachin_unavailable` до всякого span'а).
- **Lifecycle/registration** (`being_platform.py` connect/disconnect, `__init__.py` register): **allowlisted** модульный `logging.getLogger("lifemodel.*")` (ambient-трассы нет по природе; Hermes роутит в `agent.log`).
- Везде: нет `print`, нет `import structlog`/`import loguru` (§v1.2 — их в venv Hermes нет). Текстовый лог — только stdlib `logging.getLogger`; внутри тика — только через `SpanLogger` (он сам эмитит человеческую строку через stdlib `logging`).

Негативные фикстуры проверяют, что заведомо нарушающий модуль ловится.

### 4.6 Вьюер `/lifemodel trace <trace_id | last N>`

Generic. Врезка: `_SUBCOMMANDS` (`__init__.py:57`) + `dispatch` (`:199`), образец — `why_for_dir` (`state_commands.py:729`). `writer.flush()` → читает trace DB, оверлеит `deque`, восстанавливает дерево по `parent_span_id`, рендерит **тик → компоненты → решения (атрибуты) → launch → async-исход → разрешение** одинаково для любой трассы. `last N` — последние N корневых трасс по `started_at`. Миссы кореляции показываются как `orphan_async_outcome`.

## 5. Поток данных одной проактивной попытки (end-to-end по одному trace_id)

1. **Тик N.** `CoreLoop.tick` минтит root → `trace_id=T`. Компоненты — дочерние `ActiveSpan`; решения оседают атрибутами. Durable-first: enqueue → `trace_spans`/`trace_events` под `T`; при успехе enqueue → проекция в `deque`.
2. **Launch.** `CognitionLauncher` минтит `correlation_id=C`, эмитит `LaunchProactive(origin_traceparent=T/span, correlation_id=C)`, **атомарно** пишет в state `pending_proactive_id=C` + `pending_proactive_origin_traceparent`; в trace DB — индекс `trace_correlations(C → T)`.
3. **Доставка.** `proactive_tick`/reach-in несут traceparent → spans доставки под `T`.
4. **Async-ход сущности** — трасса Hermes (не наша).
5. **Read-back (`post_llm`).** Observer читает `pending_proactive_id=C` + origin из state, `child_of`, эмитит `proactive_outcome`/reasoning/verdict под `T`. Нет origin → `orphan_async_outcome`.
6. **Тик N+k (разрешение).** Агрегация потребляет `verdict_signal(C)`, поднимает origin из state, эмитит span разрешения под `T`; `resolved_at` в индекс.
7. **Отладка.** `/lifemodel trace T` — весь путь из **одного** стока. Пять источников схлопнуты.

## 6. Изменения по файлам + порядок always-green

**Порядок (правка codex — не ломать зелёное):**
1. **Новый код, без правок сигнатур.** `ActiveSpan` (`ports/tracer.py`/`adapters/tracer.py`); `state/trace_store.py` — `observability.sqlite` (схема §4.3) + async writer (thread+queue, flush, singleton, retention по count/size/days); `SpanLogger` (`log.py`) — фанаут в `deque`+sqlite+stdlib `logging`. Тест-фейки: `FakeActiveSpan`/`FakeSpanLogger` (`testing/fakes.py`).
2. **Миграция фикстур** на span/logger (сломаются `tests/test_component.py:22`, `test_aggregation.py:52`, `test_cognition.py:39` при ужесточении типа — сперва дать хелперы).
3. **Эмиттеры → `SpanLogger`**; `emit_suppression_span` + значения атрибутами; 4 недостающих reason; `component_failed`→span.
4. **Ужесточение типа** `TickContext.trace: TraceContext` (не `| None`); **ретайр** `creation_provenance()` `TraceContext | None`-фолбэка (`core/trace.py:25`) — свести к тестам.
5. **Async-мост.** `LaunchProactive.origin_traceparent` (`core/intents.py`); **`State.pending_proactive_origin_traceparent`** — это поле dataclass `State` + `to_dict`/`from_dict` (`state/model.py`), **не** миграция таблицы (`runtime_state` — JSON-блоб; правка codex #6); запись в `cognition.py` атомарно с `pending_proactive_id`; **обнуление во ВСЕХ clear-sites** `pending` (§4.4, правка codex #4); ре-бинд в `hooks.py` + miss-политика; span разрешения в агрегации.
6. **AST-гард** (`tests/architecture/test_trace_invariants.py`).
7. **Вьюер** `trace` (`__init__.py` + `state_commands.py`/новый `trace_view.py`).
8. **Удаление `events.jsonl` + structlog-костыля.** `EventSink`→`deque` (`events.py`), убрать `EVENTS_FILENAME`; из `log.py` убрать `configure()`/structlog-pipeline/`_StdlibEventLogger`-shim/`EventTee`/`get_logger`-дуализм → человеческая строка эмитится `SpanLogger`'ом через stdlib `logging.getLogger`. Перепроводка `loglevel`: `__init__.py:172` (`configure()`-вызов убрать), `config.py:27` (импорт `configure` → `setLevel`-хелпер), `LOG_LEVEL_NAMES`/`parse_log_level` оставить хелперами. Мигрировать тест-ассерты `sink.read()` на in-memory ring / trace store + `caplog` для уровня (`test_events.py`, `test_logging.py`, `test_hooks.py`, `test_core_proactive.py`, `test_plugin.py:371`).

## 7. Чего НЕ делаем (YAGNI)

Хард-зависимость OTel (опциональный экспортёр с no-op остаётся) · **structlog/loguru как зависимость** (нет в venv Hermes; текст-лог = stdlib `logging`) · in-band W3C через Hermes · внутренний LLM-ход как child нашего тика · сохранение `events.jsonl` · второй durable-сток / запись в `agent.log` того, чего нет в sqlite · вынос `memory_records.trace_id` в SQL-колонку (отдельной бедой по нужде).

## 8. Тестирование

- **Unit:** `SpanLogger` всегда штампует `trace_id/span_id`; `ActiveSpan.attrs` сериализуются; writer дропает при полной очереди без исключения; fail-open при ошибке записи; `flush` детерминирует read-your-writes; singleton/reconnect идемпотентны.
- **AST-гард** — тест (+ негативные фикстуры).
- **Integration** (реальный `CoreLoop` + fake-порты): полная попытка → `trace T` содержит tick+launch+outcome+resolution под одним `T`; miss → `orphan_async_outcome`; suppression-тик → reason+атрибуты.
- **Один источник истины (правка codex #2):** проекция в `agent.log`/`deque` пишется ТОЛЬКО после успешного enqueue; при `queue.Full` — инкремент `observability_dropped_count`, ни `agent.log`, ни `deque` не пишутся (`agent.log ⊆ sqlite` под перегрузкой).
- **Ретенция:** тесты каждой оси (age/count/size) режут по границе и сносят трассу целиком; **защита in-flight** — старая-за-лимитом + `resolved_at IS NULL` (или живой якорь в state) → НЕ прунится; после resolve+grace → прунится (правка codex #1).
- **Дедуп вьюера (правка codex #5):** flush+deque-оверлей не задваивает флашнутые записи (ключ `record_id`).
- **Clear-sites (правка codex #4):** каждый путь очистки `pending_proactive_id` обнуляет и `pending_proactive_origin_traceparent`.
- **Stdlib-only:** нет импортов `structlog`/`loguru` нигде в рантайме; человеческий лог идёт в stdlib `logging` (рантайм-инвариант CLAUDE.md + доки Hermes).

## 9. Развёртывание

`make check` (ruff/mypy/pytest + AST-гард) → commit+push → `make deploy` (плагин из git; command/adapter — после рестарта gateway). Оба sqlite создаются лениво; отсутствие trace DB — не ошибка. Миграция state-схемы (`pending_proactive_origin_traceparent`) — аддитивная, обратно совместимая.

## 10. Зафиксированные решения

Отдельный одноразовый `observability.sqlite` (**единственный durable-источник истины**; `agent.log`/`deque` — проекции, не второе место); `sqlite3` — stdlib, не зависимость; **structlog/loguru удалены** — человеческий лог через нативный stdlib `logging` → Hermes-`agent.log` (доки Hermes: `lazy_deps`/pip-extras/`check_fn` — не подходят git-directory-плагину для сквозного логгера); `events.jsonl` удалён, `EventSink`→in-memory `deque`; async writer thread+queue fail-open; **якорь связности `origin_traceparent` в драгоценном `runtime_state`**, `trace_correlations` — одноразовый индекс; через launch едет полный **traceparent**; `ActiveSpan` отделён от frozen `TraceContext`; miss → `orphan_async_outcome` (никогда не цеплять к чужой трассе); read-your-writes через `writer.flush`+deque-оверлей; singleton-писатель, reconnect-safe; **очистка старых логов по age/count/size, удаление трассы целиком**; by-construction через тип+API+AST-гард с allowlist'ом; внутренний LLM-ход остаётся под трассой Hermes; generic-схема.
