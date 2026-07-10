# Телеметрия ядра — метрики компонентов, реестр, сэмплинг, вывод в чат — дизайн

**Дата:** 2026-07-10 · **Статус:** GO (brainstorming + 3 codex-ревью), готов к беды/плану · **Версия:** v1.1
**Беды:** новый эпик «Телеметрия ядра» (parent `lm-fib`); поглощает stats-ветку `lm-9zj`. Связан с `lm-otq` (discovery полей `post_llm` для токен-метрик) и `lm-fgs` (`register_gateway_service` — для поздней фазы экспозиции).
**Родословная:** дополняет `docs/superpowers/specs/2026-07-09-observability-core-invariant-design.md` (эпик `lm-edu`). Явное разделение: **`observability.sqlite` (lm-edu) = event-driven «почему произошло ЭТО» (per-trace); `metrics.sqlite` (эта спека) = time-series «сколько / как загружено ВО ВРЕМЕНИ» (агрегаты).** Разные модели данных, разные базы, не смешивать.

> **Что изменилось в v1.1 (3-й codex-ревью артефакта, must-fix'ы):** (1) `ctx.observe` рантайм — fail-open (no-op + `metrics_emit_errors_total`), валидация — fail-fast только на register/тестах; (2) `MetricRegistry`+сэмплер — singleton **per state_dir**, idempotent-acquire, один instance для tick/hooks/`stats` через `get_metric_registry(base_dir)` в composition; (3) `metric_defs`/`metric_samples` дополнены (`label_keys_json`/`export`/`schema_version`/**`run_id`**); (4) counter reset/rate определены через `run_id` (delta<0 = reset, не склеивать процессы); (5) read-алгоритм для skip-unchanged (последняя точка ≤ границы окна + heartbeat); (6) writer-счётчики — снимок absolute-since-boot, не `inc()`; (7) `suppressions_total` — на choke-point `emit_suppression_span` (все suppression, in+out-of-tick); (8) `component_runs_total{status}` — деривация статуса записана; (9) `accepts_signals` — registry/manifest-gauge, не domain-пример; (10) Prometheus — stdlib-рендерер по умолчанию (снято противоречие §2↔§9/§12).

> **Продуктовая рамка (владелец, 2026-07-10):** базовая телеметрия — это **ядро метрик + вывод в чат**, БЕЗ HTTP-сервера. Полная экспозиция (HTTP, Prometheus `/metrics`, страница состояния) — **отдельная поздняя фаза**. Ключевой инвариант: **невозможно создать компонент, который не отдаёт свою статистику.**

---

## 1. Проблема

Движок AUTONOMIC → AGGREGATION → COGNITION тикает ~раз в 60с. Наблюдаемость «почему сущность приняла ЭТО решение» закрыта `lm-edu` (per-trace `observability.sqlite` + `/lifemodel trace`/`debug`/`why`). Но нет ответа на **операционный** вопрос: **насколько слой загружен / перегружен во времени** — длительность тика, события по фазам, вызовы LLM и токены, принимает ли слой сигналы или отключён, backpressure (shedding/drops), tick-lag. Это агрегаты во времени, а не разбор одной трассы.

Дополнительно — структурный риск: наблюдаемость компонента сейчас держится **снаружи** (CoreLoop минтит спан на компонент), но нет гарантии, что компонент отдаёт **метрики**. Владелец требует: *«не может существовать компонент, который не отдаёт статистику»* — инвариант, а не договорённость.

## 2. Рамка

**Делаем (фаза «Ядро»):** структурно-принудительную инструментовку каждого компонента; process-local `MetricRegistry` как источник текущих значений; периодический сэмплинг в отдельную `metrics.sqlite`; вывод в чат `/lifemodel stats`. Плюс предпосылка — фикс span-timing в `CoreLoop` (без него латентность фейковая).

**НЕ делаем сейчас (поздняя фаза «Экспозиция», §9):** HTTP-сервер; Prometheus `/metrics`; страницу состояния. Формат экспозиции решён по умолчанию (OpenMetrics-текст поверх нашего registry, stdlib-рендерер — §9/§12), но код рендерера и HTTP — в поздней фазе. **YAGNI (§10):** высокая кардинальность лейблов; Hermes-wide учёт токенов (только наблюдаемые `post_llm` ходы); хранение вычисленных перцентилей в sqlite (храним бакеты, перцентиль считаем на чтении).

## 3. Инвариант: компонент не может не отдавать статистику

Реализуется НЕ обязательным базовым классом (в Python ABC не даёт настоящей гарантии и превращается в god-object: логи+метрики+registry+tracer+config). Собирается из **двух замков** — «instrumented scheduler contract»:

1. **Харнесс мерит снаружи (обойти нельзя).** `Component` остаётся `Protocol` (`id`+`step`, `core/component.py`). `CoreLoop` оборачивает каждый `step()` и авто-снимает universal-метрики. Единственный prod-путь исполнения компонента — через `CoreLoop` (`core/coreloop.py:205`), так что для универсальных метрик компоненту **не надо сотрудничать** — он не может выполниться неизмеренным.
2. **Registry отказывает в регистрации без манифеста.** `ComponentManifest` (`core/registry.py:28`, сейчас бедный — только `id/type/enabled/version/config`) расширяется: **`layer` · `phase` · `metric_surface` · `accepts_signals`**. `ComponentRegistry.register()` **фейлит fast**, если `layer`/`metric_surface` не заявлены. Тест на composition root: у каждого зарегистрированного компонента есть `layer` + метрик-спеки.

**Domain-метрики** (то, что знает только компонент: токены в cognition, «слой принимает сигналы / отключён», уровни драйва) — добровольные, но **объявленные** в `metric_surface`, эмитятся через новый канал `ctx.observe` (§4.3). Итог: «нельзя быть зарегистрированным без манифеста И нельзя бежать вне wrapper'а» = требуемый инвариант, enforced и для того, что видит харнесс, и для того, что видит только компонент.

## 4. Архитектура

### 4.1 `MetricRegistry` — источник ТЕКУЩИХ значений

Process-local, потокобезопасный **singleton per state_dir/profile**. Не «истина вообще» — durable-истина причин остаётся в `observability.sqlite`; registry держит **current metrics state**.

**Lifecycle (правка codex).** `register()` плагина может зваться повторно и **не имеет teardown** — поэтому acquire **идемпотентен**: `get_metric_registry(base_dir)` возвращает один и тот же instance (и не плодит второй сэмплер-тред). Единый instance получают ВСЕ три потребителя, иначе метрики разъедутся: (а) тик — `BeingAdapter._tick()` строит graph каждый тик через `build_lifemodel()` (`adapters/being_platform.py:82`), поэтому `build_lifemodel(..., metrics=get_metric_registry(base_dir))` в composition-root; (б) хуки — graph хуков строится отдельно в `register()` (`__init__.py:264`), тот же `get_metric_registry(base_dir)`; (в) `/lifemodel stats` — читает тот же singleton (команда бежит В ТОМ ЖЕ gateway-процессе, что и тик/хуки, — как `/lifemodel trace` через `peek_trace_writer`). НЕ на `BeingAdapter.connect()`: хуки `post_llm` пишут метрики вне тик-петли.

Свои типы, stdlib-only:
- **`Counter`** — монотонный кумулятивный.
- **`Gauge`** — устанавливаемое текущее значение.
- **`Histogram`** — фиксированные бакеты + `count` + `sum`. Нужен даже при тике 60с — чтобы стандартный Prometheus-консьюмер (поздняя фаза) понимал латентность штатно; бакеты грубые: `0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30` (сек).

**Жёсткое правило — низкая кардинальность лейблов.** Только закрытый набор: `component`, `layer`, `phase`, `reason`, `outcome`, `model`. НИКОГДА `trace_id`, промпт, текст сообщения, произвольная строка. `MetricSpec` (name/kind/unit/help/label-keys) объявляется декларативно; эмиссия с незаявленным лейблом — ошибка.

### 4.2 Инструментовка `CoreLoop` (universal-метрики)

Wrapper вокруг каждого `step()` (`core/coreloop.py:205`, цикл по `self._registry.enabled()`) авто-эмитит в registry:
- `lifemodel_tick_duration_seconds` (Histogram) · `lifemodel_tick_lag_seconds` (Gauge, `now - last_tick_at`)
- `lifemodel_component_duration_seconds{component,layer}` (Histogram)
- `lifemodel_component_runs_total{component,layer,status}` (Counter). **Деривация статуса (правка codex — записать явно, чтобы воркер не парсил trace_events):** `failed` — если `step()` бросил (exception-путь CoreLoop); иначе `suppressed` — если `span.status == "suppressed"` после `step()`; иначе `ok`.
- `lifemodel_signals_intake_total{lane,result}` (Counter; result = kept/shed/coalesced — из `IntakeResult`/`apply_intake`, `core/intake.py:34/42`)
- `lifemodel_layer_accepts_signals{layer}` (Gauge 0/1) — **из манифеста `accepts_signals`** (это знание registry, не компонента — правка codex; эмитит харнесс/composition, НЕ `ctx.observe`).

**Suppressions — на choke-point, не в CoreLoop-wrapper (правка codex #8).** `lifemodel_suppressions_total{component,reason}` (Counter) эмитится в **`emit_suppression_span` (`core/suppression.py:90`)** — единой точке рождения suppression-спанов, — чтобы считались ВСЕ suppression (in-tick + out-of-tick: `proactive`/egress, `hooks`), и статы никогда не расходились с трассой. Reason — закрытый enum (`core/suppression.py:39`).

**Writer-счётчики — снимок absolute, не `inc()` (правка codex #7).** `dropped_count`/`write_errors` (`TraceWriter` через `peek_trace_writer`) — это **абсолютные process-local значения**. Сэмплер СНИМАЕТ их (записывает как sample с `run_id`, поведение counter-within-run), НЕ вызывает `inc(value)` — иначе двойной учёт. Экспонируются как `lifemodel_trace_writer_dropped_records` / `_write_errors` (absolute since boot).

Длительности требуют реального `ended_at` — см. §5 (span-timing fix), иначе Histogram'ы пустые.

### 4.3 `ctx.observe` — канал domain-метрик

`TickContext` (`core/component.py:28`; `Component` Protocol — `:72`) получает `observe: ComponentObserver` — тонкий типизированный хэндл: `ctx.observe.inc(name, **labels)`, `ctx.observe.set(name, value, **labels)`, `ctx.observe.observe(name, value, **labels)`. Это НЕ источник universal-метрик (их снимает харнесс), а канал того, что знает только компонент.

**Валидация ≠ рантайм-крэш (правка codex #1, снять противоречие с §7).** Проверка «`name`/лейбл заявлен в `metric_surface`» — **fail-fast на composition/register и в тестах** (замыкает инвариант §3). Но в **проде** эмиссия незаявленной/битой метрики через `ctx.observe` **не валит тик**: инкремент `lifemodel_metrics_emit_errors_total{reason}` и **no-op** (метрики fail-open, §7). В bare unit-тесте — no-op.

Первый живой domain-пример (действительно знание компонента, не манифеста): нейрон/драйв эмитит свой вычисленный уровень — напр. `lifemodel_contact_drive_u` (Gauge) из `contact_neuron`/`solitude_drive`. (`accepts_signals` — НЕ здесь: это знание манифеста, эмитит харнесс, §4.2.)

### 4.4 Сэмплер → `metrics.sqlite` (отдельная база, time-series)

Отдельный файл рядом с `lifemodel.sqlite`/`observability.sqlite`. Daemon-тред раз в **N секунд** (конфиг, дефолт напр. 15с) снимает срез registry → строки. Fail-open (потеря метрик не меняет поведение сущности), свой WAL, thread-affine соединение — как `TraceWriter`. Тред стартует/останавливается вместе с singleton'ом registry (§4.1, idempotent — не плодить второй тред при повторном `register()`). Каждый старт процесса минтит **`run_id`** (process-local, уникальный) — ключ различения перезапусков.

Схема (по codex — плоская `(ts,metric,value,labels)` станет болью):
```sql
CREATE TABLE schema_version (version INTEGER NOT NULL);   -- как в trace store
CREATE TABLE metric_defs (
  name TEXT PRIMARY KEY, kind TEXT NOT NULL,       -- counter|gauge|histogram
  unit TEXT, help TEXT,
  label_keys_json TEXT NOT NULL,                   -- закрытый набор лейблов метрики
  export INTEGER NOT NULL DEFAULT 1,               -- сэмплить в sqlite?
  created_at INTEGER, updated_at INTEGER
);
CREATE TABLE metric_samples (
  ts INTEGER NOT NULL, run_id TEXT NOT NULL,        -- run_id: рестарт-граница (правка codex #5)
  name TEXT NOT NULL,
  label_key TEXT NOT NULL,      -- канонический стабильный хэш отсортированных лейблов
  value REAL NOT NULL, labels_json TEXT
);
CREATE INDEX ix_samples_name_label_ts ON metric_samples(name, label_key, ts);
```
- `label_key` — канонический стабильный хэш из отсортированных пар лейблов, чтобы запрос не парсил JSON на каждой строке.
- **Histogram'ы** — как Prometheus-состояние: `name_bucket{le=...}` / `name_count` / `name_sum` (кумулятивные), `le` хранится как лейбл; НЕ вычисленный «p95».
- **Ретенция:** по age + max rows/size (как в `observability.sqlite`); fail-open.
- **Не «всё всегда»:** сэмплятся только `export=1`; sample не пишется, если значение не изменилось.
- **Counter reset/rate (правка codex #4/#5).** Counters process-local, при рестарте сбрасываются. Rate считается **только внутри одного `run_id`**: delta по соседним точкам; `delta < 0` (или смена `run_id`) = reset, старый и новый процесс НЕ склеиваются.
- **Read-алгоритм для skip-unchanged (правка codex #6).** Раз sample пишется только при изменении, `/lifemodel stats` для окна `[t0,t1]` берёт **последнюю точку с `ts ≤ t0`** и **последнюю с `ts ≤ t1`** (в пределах run) — иначе при стабильных значениях окно пустое. Плюс сэмплер пишет **heartbeat-точку раз в M циклов** (даже неизменную) — ограничить давность и подтверждать «сэмплер жив».

### 4.5 `/lifemodel stats` — вывод в чат

Новая подкоманда (регистрация в `__init__.py` `_SUBCOMMANDS`+dispatch, зеркало `trace`/`why`; хендлер в новом `stats_view.py`). Read-only, **fail-soft** (нет/локнут/битый `metrics.sqlite` → дружелюбное сообщение, не падать). Секции:
- **NOW (live из registry):** tick_lag, tick_duration (последняя), writer drops/errors, per-layer accepts_signals, текущие counters.
- **WINDOW (история из `metrics.sqlite`, `last N` сэмплов/минут):** rate событий/суппрессий по причинам, shedding, approx p95 из бакетов Histogram.
- Свёртка `component → layer` статическим маппингом (personality/neuron/drive→AUTONOMIC, aggregation→AGGREGATION, launcher→COGNITION, proactive/egress/writer→INFRA; неизвестный→`other`, не падать).

## 5. Предпосылка: фикс span-timing в `CoreLoop`

Сейчас `core/coreloop.py:155` `started = now.isoformat()`, и этот же `started` пишется как `ended_at` для component- и root-спанов (`:238/:252/:279`) → длительность каждого спана = 0. **Без фикса нет никакой латентной метрики.** Фикс: закрывать спаны реальным wall-clock после `component.step()`; elapsed мерить через `time.monotonic()` (монотонность, не скачет на смене системного времени). Трогает горячий тик-путь → отдельное ревьюируемое изменение, идёт **первым**. Побочно чинит и `/lifemodel trace` (там длительности тоже сейчас нулевые).

## 6. Поток данных

Тик: `CoreLoop` wrapper — `t0 = monotonic()` до `step()`, `dt` после → `observe(component_duration)`, `inc(component_runs{status})`; intake-результаты → `inc(signals_intake)`; суппрессии → `inc(suppressions)`. Компонент внутри `step()` может звать `ctx.observe.*` для domain-метрик. Хуки (`post_llm`) пишут в тот же registry вне тика. Сэмплер раз в N сек снимает registry → `metrics.sqlite`. `/lifemodel stats` читает live registry + историю sqlite.

## 7. Ошибки / устойчивость

Registry-операции — in-memory, **best-effort на горячем пути**: эмиссия метрики (в т.ч. через `ctx.observe` с незаявленным именем/лейблом) **не валит тик** — инкремент `lifemodel_metrics_emit_errors_total{reason}` и no-op (§4.3). Валидация метрик-поверхности — fail-fast, но только на composition/register/тестах, не в проде. Сэмплер/`metrics.sqlite` — fail-open (тик не ждёт, ошибки глотаются, счётчик дропов). `/lifemodel stats` — fail-soft на каждом источнике (нет sqlite → только NOW из registry; нет registry в bare CLI → дружелюбно). `metrics.sqlite` создаётся лениво; отсутствие — не ошибка.

## 8. Тестирование

- **Unit:** типы метрик (Counter монотонность, Gauge set, Histogram бакеты+count+sum); валидация fail-fast на register + **рантайм-эмиссия незаявленной метрики = no-op + `metrics_emit_errors_total`, тик не падает**; `registry.register()` фейлит без `layer`/`metric_surface`; layer-rollup; сэмплер пишет только `export=1` и пропускает неизменившиеся; **rate внутри `run_id` (delta<0/смена run = reset, не склеивать процессы)**; **read-алгоритм окна: последняя точка ≤ границы + heartbeat**; **writer-счётчики — снимок absolute, без double-count**; fail-soft `/lifemodel stats` на отсутствующем/локнутом sqlite.
- **Enforcement-тест (инвариант §3):** composition root — у КАЖДОГО зарегистрированного компонента есть `layer` + `metric_surface`; негативная фикстура (компонент без манифеста) ловится.
- **Integration** (реальный `CoreLoop` + fake-порты): тик → registry содержит tick/component duration (>0 после span-timing fix), runs, intake, suppressions; сэмплер → `metrics.sqlite` содержит ожидаемые строки; `/lifemodel stats` рендерит.
- **Stdlib-only:** нет сторонних импортов в рантайме.

## 9. Поздняя фаза «Экспозиция» (НЕ сейчас)

HTTP-сервер для просмотра состояния агента — **через `register_gateway_service` (`lm-fgs`), НЕ сырой `http.server`-тред** (у сырого треда тяжёлый lifecycle: reload/двойная регистрация/порт занят/multi-profile). Если всё же сырой stdlib: bind только `127.0.0.1`, порт конфигом (default off), без мутаций, без PII в лейблах/HTML, один сервер на процесс, lifecycle на plugin-register. `/metrics` — OpenMetrics-текст поверх того же registry (pull; НЕ дублирует `metrics.sqlite` — разные консьюмеры: live-scrape vs локальная история). Страница состояния — вторична, read-only, не дашборд.

**Prometheus: принимаем стандарт (модель данных + текст-формат), библиотеку — нет.** Прецедент — `lm-edu` принял модель W3C trace-context без SDK OTel. Причины: (1) CLAUDE.md требует stdlib-фолбэк для любой доп-зависимости — stdlib-путь пишется всё равно; (2) registry мы строим сами (для §3 и `/lifemodel stats`+sqlite), так что ценность `prometheus_client` схлопывается до ~20-строчного рендера текста; (3) прецедент сквозного концерна. **По умолчанию — stdlib-рендерер** OpenMetrics-текста поверх registry; `prometheus_client` рассматриваем только если в фазе экспозиции всплывёт конкретная нужда, которую stdlib не покрывает (не «открытый вопрос», а дефолт с правом пересмотра).

## 10. Токены LLM (domain-метрика, зависит от `lm-otq`)

`lifemodel_llm_observed_calls_total`, `lifemodel_llm_observed_tokens_total{kind=prompt|completion|total, model}`, `lifemodel_llm_observed_missing_usage_total`. Честная маркировка `*_observed_*`: это наблюдаемые `post_llm` ходы (проактив act-gate + inbound), НЕ все LLM-вызовы; Hermes-wide учёт требует host-контракта. Извлечение полей (`model id`/`token counts` сейчас в `**_ignored`, `hooks.py:102/294`) — **блокировано discovery `lm-otq`**.

## 11. Разбивка на беды (фазы)

Эпик «Телеметрия ядра» (parent `lm-fib`):
1. **span-timing fix в `CoreLoop`** (предпосылка, первым; blocks длительности). P2.
2. **`MetricRegistry` + типы (Counter/Gauge/Histogram) + `MetricSpec` + низко-кардинальные лейблы.** P2.
3. **Расширение `ComponentManifest` (`layer/phase/metric_surface/accepts_signals`) + registry-валидация + enforcement-тест.** P2. Зависит от 2.
4. **Инструментовка `CoreLoop`** (universal-метрики §4.2). P2. Зависит от 2,3 и (для длительности) 1.
5. **`ctx.observe` канал domain-метрик** (рантайм fail-open) + первый пример — component-internal (`contact_drive_u` из нейрона/драйва; `accepts_signals` — НЕ здесь, он в беде 4). P2/P3. Зависит от 2,3.
6. **Сэмплер → `metrics.sqlite`** (схема §4.4, ретенция, export-whitelist). P2. Зависит от 2.
7. **`/lifemodel stats`** (live+история §4.5). P2. Зависит от 2,6.
8. **Токен-метрика `llm_observed_*`** (§10). P3. Зависит от `lm-otq`.
9. **[Поздняя фаза «Экспозиция»]** HTTP через `register_gateway_service` + `/metrics` + страница + lib-vs-handroll (§9). Отдельный эпик/беда, отложено.

## 12. Зафиксированные решения

Enforcement = manifest + registry-валидация + харнесс-wrapper (НЕ обязательный ABC); `Component` остаётся Protocol. `MetricRegistry` process-local **singleton per state_dir, idempotent-acquire**, lifecycle на plugin-register (не на `connect`); один instance для tick/hooks/`stats` через `get_metric_registry(base_dir)`. Свои типы Counter/Gauge/Histogram, stdlib. Низкая кардинальность закрытых лейблов. **Валидация метрик fail-fast на register/тестах; рантайм `ctx.observe` fail-open (no-op + `metrics_emit_errors_total`), тик не падает.** Отдельная `metrics.sqlite` (time-series) ≠ `observability.sqlite` (event-driven); схема `schema_version`+`metric_defs(label_keys_json,export)`+`metric_samples(run_id,label_key)`. Сэмплинг раз в N сек, export-whitelist, skip-unchanged + read последней точки ≤ границы окна + heartbeat, ретенция age/size, fail-open. **Rate — только внутри `run_id`** (delta<0/смена run = reset). **`suppressions_total` — на choke-point `emit_suppression_span`** (все suppression, in+out-of-tick); **writer-счётчики — снимок absolute (без `inc()`-double-count)**; **`component_runs_total{status}`** — failed(exception)/suppressed(`span.status`)/ok; **`accepts_signals`** — registry/manifest-gauge (харнесс), не `ctx.observe`. `/lifemodel stats` — вывод в чат, fail-soft. span-timing fix (`time.monotonic()`) — первым. HTTP/Prometheus/страница — поздняя фаза через `register_gateway_service`; Prometheus: модель+формат приняты, **stdlib-рендерер по умолчанию**, библиотека — только по конкретной нужде (прецедент lm-edu/W3C). Токены — `*_observed_*`, зависят от `lm-otq`.
