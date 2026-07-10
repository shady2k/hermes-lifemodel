# Пересборка сенсорного ядра — эфемерный нервный поток по биологии

**Дата:** 2026-07-11. **Статус:** дизайн утверждён (владелец + 2 прохода codex), реализация не начата.
**Отменяет:** `2026-07-10-durable-signal-bus-design.md` (durable-курсорная шина — **признана неверным фундаментом**).
**Трогает:** HLA §1/§2/§4 (резолв противоречия §2↔§4), glossary. **Трекер:** новый эпик «привести ядро в порядок».

---

## 1. Почему пересобираем

Мы построили durable append-only «signal bus» с курсором/дедупом/компакцией (эпик `lm-s56`) — фактически **мини-Kafka в однопроцессном плагине**. Это неверный фундамент:

- **Биологически:** нервная система **не пишет каждый импульс в долговечный лог**. Афферентный поток эфемерен; долговечен только **след в теле (гомеостаз), памяти и записи-наблюдаемости**.
- **HLA сам себе противоречил:** §2 (таблица) — *«Signal bus — durable append-only лог»*; §4 (дома состояния) — *«сигналы — in-memory, эфемерные, TTL, не персистятся»*. Реализация встала на §2. **Резолвим в сторону §4.**
- **Смешали под словом «сигнал» четыре разных вещи:** сырой ввод органа, выход рецептора, промежуточное вычисление слоёв, выход драйва, и (через хуки) исход действия.

Правильная модель ниже.

---

## 2. Инвариант ядра

> **Нервный поток — эфемерен. Durable — только там, где и биология держит след: тело (`AgentState`), память (`Memory`/BDI), запись (`Trace`). Модель Другого (`UserModel`) — наш выведенный кэш. Мутации — только через интенты; сериализованы одним state-actor.**

Отсюда: **шина in-memory, пустая после рестарта** («потеряли сознание → не доигрываем старьё» — это фича).

---

## 3. `ExecutionFrame` — единица обработки

Единица работы — **не «cron-тик», а кадр исполнения (`ExecutionFrame`)**. Frame запускается любым из событий:

- **heartbeat** — периодический тик (пустой мир: драйв растёт, агрегация смотрит порог);
- **входящее событие Hermes** — юзер написал;
- **завершение async-когниции** — исход хода готов (`sent`/`silent`/…);
- **admin/debug-команда**, если мутирует state.

Каждый frame — самодостаточный цикл:
```
событие → снапшот state → SignalFrame (in-memory) → компоненты сворачивают →
  интенты → state-actor атомарно коммитит → SignalFrame умер
```
**Сигнал живёт ≤ один frame** (а не «≤ один тик»). Frame обрабатывается **в момент события** (не ждёт таймер). Все frame-ы **сериализованы одним state-actor** — split-brain невозможен даже при совпадении heartbeat и входящего.

*Почему это критично (codex):* async-исход когниции возникает **между** heartbeat-ами. Если оказия обработки только heartbeat — исходу «некуда лечь» (болтается в эфемерной шине, теряется при рестарте; а он safety-критичный — разрешает pending). Frame даёт оказию **в момент завершения хода** → коммит сразу.

---

## 4. Слои (афферентная половина)

```
мир → Организм/Канал(Organ) → Сенсор → сигнал → SignalFrame(шина) → Драйв → Агрегация → Когниция
```

- **`OrganAdapter`/`ChannelAdapter`** = интеграция Hermes (Telegram…). **Двунаправлен**: афферентная сторона (сенсоры) + эфферентная (мотор/egress). *(Двунаправлен канал, НЕ сенсор.)*
- **`Sensor`** = афферентный трансдьюсер. Читает вход канала → сигнал в SignalFrame. **Band-pass фильтр на самом сенсоре** (control-команды не проходят — как ухо не слышит ультразвук). Тупой.
- **`Drive`** = гомеостатический/аффективный интегратор (Панксепп PANIC/GRIEF). Копит `u` во времени (**растёт в тишине по умолчанию**, `satiate` на контакт). **НЕ сенсор/нейрон** — мотивационный центр с durable-состоянием. `u` в `AgentState`. Остаётся `Component` (нынешний `SolitudeDrive` — хороший, только переименовать «Drive, не neuron»).
- **`Neuron`** — для v1-контакта НЕ нужен (сенсор→драйв почти напрямую). Если появится — читает **шину**, тупой (`порог→emit`, без логики).
- **`Aggregation`** = таламический **гейт**. Читает SignalFrame + state. salience, жизненный цикл желания (рождение/ack/defer/release), **защищает когницию** (backpressure). Решает `LAUNCH` когниции.
- **`Cognition`** = кора/LLM. **Единственный, кто действует в мире.** Async (7-116с). Сочиняет обращение → Hermes доставляет (reach-in) **в рамках хода**, либо `[SILENT]`. Финальный act-gate.

**Тишина = отсутствие сигнала контакта** (инверсия), не событие. Драйв растёт сам, контакт сбрасывает.

---

## 5. Эфферентная половина (мотор) + замыкание петли

- Действует только **когниция**, **в своём ходе** (её вывод доставляет Hermes). 0-LLM слои (сенсор/драйв/агрегация) только сигналят + мутируют **своё** состояние; максимум агрегации — `LAUNCH`.
- **Исход хода (`proactive_outcome`: `sent | silent | failed | stale | superseded`)** — не «verdict», а **факт «что я сделал»** (эференс-копия). Когда ход завершился (async) → **свой frame** → исход в SignalFrame (агрегация разрешает pending) + запись в `AgentState` (`action_pending`, backoff). **Сразу, не ждём heartbeat.**
- **Протухшее обращение** (юзер написал во время сочинения) — **принятое ограничение**: running LLM-ход надёжно не подавить; максимум — Hermes-FIFO может прервать (не гарантируем). `stale`/`superseded` фиксируем как исход, но доставку не отменяем.

---

## 6. Сигналы vs Интенты; закрытая таксономия

- **Сигнал** — афферентный поток, **только чтение**, эфемерен (≤ frame), в SignalFrame.
- **Интент** — канал **мутаций/действий** (`UpdateState`/`PutRecord`/`TransitionRecord`/`LaunchProactive`/`SendMessage`); собирает coreloop, применяет **state-actor** атомарно в конце **frame** (не «в конце cron-тика»).

**Закрытая таксономия сигналов** (не размывать «универсальную шину»):
- `ObservationSignal` — контакт/доставка/канал/время (`contact_observed`…);
- `DriveSignal` — `u`/effective_pressure;
- `CognitiveOutcome` — `sent/silent/failed/stale`;
- `ProposalSignal` — мысль предлагает контакт (позже, Ф6).

---

## 7. Backpressure — в АГРЕГАЦИИ, не в шине

Эфемерная шина ⇒ «отложить overflow на след. тик» **невозможно**. Backpressure переезжает к консюмеру (агрегация) и работает **priority-классами**:
- `must_process` — `contact_observed`, `proactive_outcome`, safety/backstop — **никогда не sheddить**;
- `best_effort` — sensor-шум — коалесить/сбрасывать по salience.

Для v1 (один сенсор) `MAX_INTAKE` почти не нужен — достаточно классификации.

---

## 8. Durable — домены и физика

| домен | что | физически |
|---|---|---|
| **`AgentState`** («я», авторитетно, без TTL) | скаляры сейчас: `u`, energy, fatigue, mood/emotions **+ контакт-бухгалтерия драйва** (`last_exchange_at`, backoff, `unanswered_outbound_count`, `action_pending`, `silence_anchor_at`) **+ idempotency-ring** `processed_external_event_ids` (TTL) | `runtime_state` (`StatePort`) в `lifemodel.sqlite` |
| **`Memory`** (мой BDI) | Desire/Intention/Thought — объекты с жизненным циклом | `memory_records` (`MemoryPort`) в `lifemodel.sqlite` |
| **`UserModel`** («другой», вывод/кэш, TTL) | mood/receptivity/каденс, пер-поле `{value, inferred_at, ttl, confidence?}`; протухло → «неизвестно» + триггер ре-инференса | `memory_records` (`MemoryPort`), но семантически — модель Другого, не мой BDI |
| **`Trace`** (наблюдаемость) | спаны frame-ов, для дебага | отдельная `observability.sqlite` |
| провайдер (hindsight/Honcho) | сырьё истории, из которого выводим `UserModel` — **не наш durable-дубль** | внешний, не owned |

**Idempotency ≠ bus-dedup.** Bus-курсор/дедуп выкинуты. Но **внешние события идемпотентны** через ring `processed_external_event_ids` (TTL) в `AgentState` — Telegram/Hermes ретраят, иначе двойной сброс `u` / двойное закрытие pending.

---

## 9. Наблюдаемость

Каждый **frame** = своя трасса (root-span + дочерние спаны компонентов). Сенсор **штампует `trace_id`** на сигнал (провенанс). Между frame-ами (async-разрыв) — **корреляция** по `trace_id` (`trace_correlations`), не один span сквозь разрыв. Сигналы эфемерны, **спаны durable**.

---

## 10. Что переименовать, что оставить, что выкинуть

**Переименовать (сохранив поведение — НЕ удалять факты):**
- `PresenceNeuron` → `ContactSensor`; `exchange` → `contact_observed`; `verdict` → `proactive_outcome`; `Relationship` → `UserModel` (миграция **отдельно** от хирургии шины, **последней**).

**Оставить (это тело/предохранители, переживают миграцию):**
- `SolitudeDrive` как `Component` (только «Drive, не neuron»); `pending_proactive_id`, `action_pending_since`, backoff, `unanswered_outbound_count`, `silence_anchor_at`, global backstop; `last_exchange_at` (иммунная запись реального обмена).

**Выкинуть:** `DurableSignalBus`/`FileSignalBus`/`signals.log`/`signals.meta`/`signal_records`, cursor/claim/commit, bus-level dedup, компакция, backpressure-в-шине; слово «signal» как универсальное для сырого-ввода/выхода-драйва/мутации/трейса.

---

## 11. Work-items «привести ядро в порядок» (порядок гибкий)

> **Директивы владельца (2026-07-11):** порядок исполнения **не важен**, нужен **конечный результат** (промежуточное не ревьюится); **никакой обратной совместимости** — старый/мёртвый код **удаляем**, не «рядом со старым»; **существо можно сбросить** (factory-reset) — миграция живого state НЕ требуется. Ниже — список работ (не строгий порядок), а **критерий готовности** — §12.

- **`SignalFrame`/`ExecutionFrame`:** in-memory шина внутри frame; CoreLoop принимает `initial_signals`; `EmitSignal` только in-memory; коммит в конце frame; сериализация одним state-actor.
- **Backpressure в агрегации** (priority-классы `must_process`/`best_effort`), НЕ в шине.
- **Хуки переделать:** `pre_gateway_dispatch` → запуск frame с `contact_observed` (не `bus.publish`); `post_llm_call` → запуск frame с `proactive_outcome`. *(Обязательно вместе с удалением шины — иначе потеря async-исхода/inbound.)*
- **Переименования (факты сохранить, поведение не терять):** `PresenceNeuron→ContactSensor`, `exchange→contact_observed`, `verdict→proactive_outcome`; `SolitudeDrive` оставить Component, но «Drive, не neuron».
- **Удалить полностью (no back-compat):** `DurableSignalBus`/`FileSignalBus`/`signals.log`/`signals.meta`/`signal_records`, cursor/claim/commit, bus-dedup, компакция; **мёртвые ABC** (`Neuron`/`Aggregator`/`Layer`/`ActGate` — ср. `lm-u5b`); `signals.consumed`.
- **`AgentState`-схема:** убрать bus-dedup комментарии; **добавить idempotency-ring** `processed_external_event_ids` (TTL) для внешних событий.
- **`UserModel` — последним** (отдельный слой; трогает cognition/inference/prompt/TTL; `Relationship→UserModel`).
- **Живое существо — factory-reset** после выката (можно сбросить, миграция не нужна).

## 12. Критерий готовности (регресс-сценарии — это и есть приёмка)

Конечный результат должен проходить (0-LLM, детерминированно на fake-портах):
1. реальный inbound (`contact_observed`) гасит `u`, ставит `last_exchange_at`, разрешает pending-желание → SATISFIED;
2. control-команда (`/…`) **не** считается контактом (band-pass фильтр сенсора);
3. `u≥θ` + receptivity → агрегация `LAUNCH` когниции;
4. когниция in-flight → повторный frame **не** будит второй ход;
5. `proactive_outcome: sent` → `action_pending`/backoff; `silent` → decline-backoff; pending чистится корректно;
6. дубль внешнего события (тот же `origin_id`) → idempotency-ring **не** гасит `u` дважды;
7. `ExecutionFrame` от завершения async-когниции коммитит исход **сразу**, не ждёт heartbeat;
8. рестарт → шина пуста, durable-state (AgentState/Memory) цел, поведение продолжается;
9. `make check` зелёный (ruff+mypy+pytest); durable-bus/cursor/dead-ABC артефактов в коде нет.
