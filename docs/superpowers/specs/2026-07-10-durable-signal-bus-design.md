# Durable Signal Bus — курсорный ingress с инъектируемой backpressure-политикой

**Дата:** 2026-07-10. **Статус:** дизайн утверждён, реализация не начата.
**Источник инвариантов:** [core-engine-design §5.1/§7.4](2026-07-06-core-engine-design.md). **Трекер:** эпик `lm-s56`.
**Ревьюеры:** codex (два прохода в этой сессии).

---

## 1. Проблема

Текущая шина (`adapters/signal_bus.py:FileSignalBus`) — «глупый лог»: `consume_unprocessed()` **читает весь лог, вычитает весь вечный леджер `signals.consumed`**, возвращает остаток и **помечает consumed сразу весь батч**. Ограничение `MAX_CONTROL=256` применяет **отдельная** функция `core/intake.py:apply_intake` уже **после** пометки → overflow помечен потреблённым и **потерян навсегда**. Docstring `intake.py` при этом **утверждает**, что overflow остаётся на шине — это ложь относительно кода.

Два дефекта:
1. **Нарушен §5.1 «control lossless»:** overflow control-сигналов теряется молча (низко-рисково в v1: 256 vs 1-2 сигнала/тик, но инвариант нарушен и docstring врёт).
2. **Нарушен SRP:** backpressure / ограничение батча / решение «что заакать» — это оркестрация, которую сегодня ведёт `CoreLoop` (`consume → apply_intake → …`), а не шина. Плюс `_unprocessed()` перечитывает **всю историю** каждый тик (`adapters/signal_bus.py:90`) и лог растёт неограниченно.

---

## 2. Скоуп — ЧТО строим и что осознанно НЕ строим

> **КРИТИЧНО (требование владельца): «построено не всё».** §5.1 описывает богатую механику под будущие шумные сенсоры. Сегодня **durable-шина несёт только control** (`exchange`, `verdict` из хуков). Всё «сенсорное» (`contact_presence`/`contact_pressure`) — это **внутренняя intra-tick лента через транзиентный `EmitSignal`**, на durable-шину **не идёт** (`core/contact_neuron.py:39`, `core/solitude_drive.py:45`, `core/aggregation.py:120`). Значит sensor-полоса — это **каркас под будущее**, а не рабочий код. Ниже — явная граница.

**Строим сейчас (v1):**
- `DurableSignalBus` (ABC + `FileSignalBus`): append-лог + durable-**курсор** + `claim`/`commit` + read-only `inspect_pending` + dedup-по-claim-окну.
- `BackpressurePolicy` (core-level, инъектируемая): классификация полос, caps, ack/defer/drop, per-lane stats.
- **Control-полоса целиком**: lossless, bounded-prefix `MAX_CONTROL`, overflow **не потребляется** (курсор не двигается за хвост) → доезжает следующим тиком.
- `commit` **после** успешного `StateActor.apply()`, gated на успех durable-ingress consumer-ов.
- Идемпотентность `exchange` reducer по `Signal.timestamp`.
- Миграция CoreLoop + тестов + composition; удаление `consume_unprocessed`/`signals.consumed`.
- `backpressure_stats` (backlog, overflow-тики, shed-count, watermark) — **шина только МЕРИТ и отдаёт**.

**Осознанно НЕ строим (каркас есть, реализация отложена — трекается бидами):**
1. **Sensor-полоса**: резервуар per-kind (`latest/max/count/first_ts/last_ts`), salience-shed, high/low watermark гистерезис. Ждёт **durable sensor-продюсеров** (ping/presence/мысли). Policy sensor-ready, но путь пустой. → `lm-s56`-child.
2. **Настоящий salience**: Weber-Fechner + EMA + арузал. v1 — заглушка (`Signal.salience` или константа `1.0`). → там же.
3. **Арузал-связь**: energy-модель читает `backpressure_stats` и поднимает `A` при затяжной перегрузке. Сейчас — только stats-hook; **шина к `core/energy` не цепляется**. → отдельный бид.
4. **`in_flight`→latest компакция и `delivery_result` final-state**: control-суб-политика для флагов состояния. Нет durable-продюсеров `in_flight`/`delivery_result` сегодня → каркас, не код. → отдельный бид.
5. **Полная компакция лога** (обрезка консумленного префикса по порогу как отдельный maintenance-шаг). v1-лог крошечный; курсор даёт корректность, рост — отдельная забота (ср. `lm-fib.6.5`). → отдельный бид.

---

## 3. Архитектура — три слоя с чистыми границами

```
producers (hooks) ──publish──▶  DurableSignalBus  ──claim()──▶  CoreLoop tick
                                 (лог+курсор+dedup)                 │
                                        ▲                           ├─ BackpressurePolicy (классифицирует/бьёт/коалесит)
                                        │                           ├─ domain reducers (agg/drive) — СМЫСЛ, не трогаем
                                        └──────commit(batch)────────┘  (после StateActor.apply)
```

- **`DurableSignalBus`** — durable-механика: append, курсор, bounded-claim, dedup-по-окну, crash/retry, (позже) компакция. **Не знает** ни полос, ни домена.
- **`BackpressurePolicy`** — core-level, но **не доменная**: знает `kind→lane`, caps, coalesce, ack/defer/drop. **Не знает** desire/energy/thresholds.
- **Доменный reducer** (`aggregation`/`solitude_drive`) — семантика exchange/verdict, drive, желания. **Не меняем** (кроме timestamp-идемпотентности §7).

Граница «что сворачивает шина vs что фолдит домен»: шина/policy сворачивают то, что сворачивается **без доменного состояния** (dedup-повторов, позже — `in_flight`→latest, sensor-агрегаты). Доменный компонент фолдит то, что требует состояния (матчинг `verdict`↔`desire` по `correlation_id` — знание про желание живёт в State, шине недоступно и не должно быть). Если заставить шину свернуть вердикты — она полезет в домен и **сломает SRP в другую сторону**.

---

## 4. Интерфейс

```python
class DurableSignalBus(ABC):
    def publish(self, signal: Signal) -> None: ...          # producer append (immutable event, §7.4)
    def claim(self, policy: BackpressurePolicy) -> WorkBatch: ...  # bounded, classified, НЕ двигает курсор
    def commit(self, batch: WorkBatch) -> None: ...         # двинуть курсор по потреблённому префиксу
    def inspect_pending(self) -> list[Signal]: ...          # read-only (debug/тесты), курсор не трогает
```

`WorkBatch` (минимум):
- `signals: tuple[Signal, ...]` — durable-вход, который CoreLoop раздаёт компонентам;
- `cursor_commit_until` — позиция-граница префикса к потреблению (монотонная **по позиции записи**, не по `origin_id`);
- `ack_ids` / `deferred_ids` — для observability;
- `stats: BackpressureStats` — kept/shed/coalesced/deferred per lane + backlog.

**Инварианты контракта (явно в ABC):**
- **Единственный consumer.** Курсор-модель корректна только при одном потребителе durable-шины.
- **Commit по позиции, не по id.** Иначе partial-defer ломает будущую компакцию.
- `commit` идемпотентен (повторный `commit` того же batch — no-op).

---

## 5. Хранилище: append-лог + курсор + dedup-по-окну

- **Append-лог** `signals.log` — как сейчас (одна JSON-строка на сигнал, fsync, torn-write-tail отбрасывается).
- **Durable-курсор** — маленький файл (позиция последней потреблённой записи). **Заменяет** вечный леджер `signals.consumed`.
- **`claim`** читает вперёд от курсора **ограниченный префикс** (не весь лог), policy выбирает `cursor_commit_until`. Всё **до** границы — processed/(позже)compacted/dropped по policy; всё **после** — остаётся.
- **Backpressure = не-потребление.** Overflow control не входит в потреблённый префикс → курсор его не переходит → следующий тик его видит.
- **Dedup — только по claim-окну** (до commit-курсора): один `origin_id`, легший дважды **в текущем окне**, обрабатывается один раз, курсор после commit покрывает обе записи. **Вечный dedup не нужен** — за курсором записи не перечитываются.
  - *Почему dedup, а не только offset:* двойной выстрел хука (ретрай/повторный `post_llm`) даёт двойной `exchange` → двойной satiate (`solitude_drive.py:55`) и порча счётчика `unanswered_outbound_count` (`aggregation.py:188`). Идемпотентность **значения** это не спасает — нужен ingress-dedup. Оба механизма ортогональны: dedup против двойного **append**, идемпотентность против **replay** (§7).

---

## 6. Полосы и политика

**Control (lossless) — строим:**
- Обрабатывается первой, по важности **не** sheddится.
- **Суб-политика по виду** (не глобальная компакция!): `exchange`/`verdict`/`delivery_result` — **события**, держим каждое и в порядке; `in_flight` — **флаг состояния**, latest-wins *(каркас: durable-продюсера нет)*; `delivery_result` — per-delivery-id final-state *(каркас)*.
- Bounded-prefix `MAX_CONTROL=256`; overflow — не потребляем.

**Sensor (latest-wins/aggregate) — каркас:**
- API policy sensor-ready (`kind→lane=sensor`, cap-by-count `MAX_SENSOR=64`, salience-hook).
- Реализация отложена (нет durable sensor-продюсеров). До тех пор путь пустой.

---

## 7. Семантика commit и крэша (at-least-once)

- Курсор двигается **только после** успешного `StateActor.apply()` (`core/coreloop.py:286`), **до** best-effort trace-export.
- **Commit gated на durable-consumer, не на «тик вернулся».** Если durable-ingress reducer (агрегация) упал — курсор **не** двигаем (сейчас fail-soft продолжает, `coreloop.py:260`; для durable-consumer это надо ужесточить). Падение **опционального** пост-компонента коммит не блокирует.
- Крэш в окне «State закоммичен, курсор не сдвинут» → сигнал **переиграется** (at-least-once) → reducers обязаны быть идемпотентны.
- **Идемпотентность `exchange`:** `last_exchange_at` брать из `max(Signal.timestamp)` валидных real-exchange (fallback `now` только для отсутствующего/битого ts), а не `now` (`aggregation.py:183`). Это чинит replay-дрейф **и вернее по сути**: окно тишины якорится на факт-время обмена, иначе при backlog курсор-модель искусственно «освежает» старые сообщения. Затрагивает якорь `lm-md6.1` — приемлемо и правильно.

---

## 8. Backpressure-stats (для будущего арузала — только hook)

Шина считает и отдаёт `BackpressureStats`: длина backlog, overflow-тики, shed-count, состояние watermark. **Энерго-модель их ЧИТАЕТ** (отдельным компонентом/future-hook) и интерпретирует (затяжная перегрузка → арузал `A` → выше эффективный порог, §5.1). **Шина от `core/energy` не зависит** — она мерит давление, energy интерпретирует. Не наоборот.

---

## 9. Миграция (чистый разрыв)

Плагин в dev, back-compat не нужен, старый ABC врёт.
- Удалить `consume_unprocessed` + `signals.consumed` (вечный леджер).
- Мигрировать `CoreLoop.tick` на `claim`/`commit`.
- Обновить ~20 тестов (`tests/test_signal_bus.py`, `tests/test_fakes.py`, `tests/test_composition.py`) и `FakeSignalBus` (`testing/fakes.py`).
- **Сохранить read-only** путь как `inspect_pending` (debug/hooks/тесты используют `peek`).
- Никакого переходного слоя — он только раздувает поверхность багов.

---

## 10. Тестирование (инварианты)

- **«Слой не падает»:** `N ≫ MAX_CONTROL` control-сигналов в один тик → тик за ограниченное время; control не потерян; исключений нет.
- **Overflow доживает:** `MAX_CONTROL=1`, два `exchange` → после тика 1 второй **не потреблён** (`inspect_pending` его видит) → обработан тиком 2.
- **Dedup-по-окну:** один `origin_id` дважды в окне → обработан один раз; счётчик не задвоился.
- **Idempotent replay:** State закоммичен, `commit` не вызван (крэш-эмуляция) → следующий тик переигрывает сигнал; `last_exchange_at` стабилен (из timestamp сигнала).
- **Commit gating:** durable-consumer кинул → курсор не сдвинут → сигнал доступен следующему тику.
- **Single-consumer / commit-by-position:** partial-defer оставляет хвост; курсор монотонен по позиции.

---

## 11. Открытые вопросы (решить при написании плана)

- Формат курсора: номер строки vs байт-offset (склонность — номер committed-строки, согласуется с torn-tail-обработкой).
- `BackpressureStats` — минимальный набор полей на v1 (backlog + overflow-count достаточно?).
- Компакция лога (п.2.5): совсем отложить или включить минимальный порог-триггер уже сейчас (склонность — отложить, курсор даёт корректность, v1-лог крошечный).
