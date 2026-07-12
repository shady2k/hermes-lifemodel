# lm-ukc.4 — Реактивный показ самоощущения (design)

**Bead:** lm-ukc.4 (Фаза 3 «Живое самоощущение», эпик lm-ukc)
**Дата:** 2026-07-12
**Статус:** дизайн на ревью

## 1. Контекст и цель

У существа уже есть **core affect** (Рассел: валентность×арузал; `core/affect.py` —
`felt_word`/`felt_texture` над общими порогами `FeltWordParams`). Аффект попадает в
**проактивный импульс** (lm-ukc.5) и в дебаг/трассу (lm-ukc.6/.3). Чего нет — самоощущение
не проступает в **обычном реактивном разговоре** (ответы на сообщения владельца).

Цель `.4`: настроение окрашивает **манеру** обычных ответов — но **не в каждом**, и
**невидимо во время реальной работы**. Существо публичное: произвольные пользователи, языки,
и его гоняют в том числе на **задачах** (код, summarize), не только на общении.

**Инварианты Фазы 3 (не переоткрывать):**
- Настроение = **манера** (как говорит), тоска = **причина** (зачем тянется).
- Аффект **односторонний**: эмитит только `UpdateState`, **не может** питать пробуждение.
  Реактивный показ — **только чтение** состояния; в драйв/wake ничего не пишет.
- Рамка **интегративная**, не диссоциативная.

## 2. Скоуп

**В этом срезе — два канала:**
- **Ambient (push)** — лёгкий manner-cue через **эфемерный inject** (`pre_llm_call`),
  **state-driven, ноль языковой детекции**. Гейт: suppression-first deny-list + порог салентности.
- **On-demand (pull)** — **`register_tool("check_in")`**: честный self-read, который **модель
  зовёт сама**, когда нужно (её спросили «как ты» на любом языке). LLM — единственный надёжный
  детектор вопроса-о-себе; Python его не детектит (§5). Имя/описание — §4b.
- Наблюдаемость для калибровки (метрики + строка в `/lifemodel debug`).

**Отложено (follow-up beads, не здесь):**
- Богатого **инжекта** нет вовсе — вопрос-о-себе чисто лингвистичен, языко-независимого якоря
  под него не существует; эту роль целиком берёт tool (pull).
- NL-детектор «работы по ключевым словам» (summarize/translate/explain…) — язык-смещён и
  хрупок; ложное срабатывание глушит настроение на реляционной реплике. В v1 не тащим.
- Скилл самопознания (`lm-aab`) — отдельная тема (архитектура, не felt-state).

## 3. Механизм доставки — inject-only, эфемерно

Подтверждено **на исходниках Hermes** (не по докам):
- Хук **`pre_llm_call`** (`agent/turn_context.py:build_turn_context`) срабатывает **раз за
  пользовательский ход, до вызова модели**. Колбэк возвращает `{"context": str}` (или строку,
  или `None`).
- Возвращённый текст приклеивается к **копии** user-сообщения только для этого одного API-вызова
  (`agent/conversation_loop.py:792-812`): исходное сообщение не мутируется, **в session DB не
  персистится, в rolling-историю не оседает**. Живёт ровно один ход.
- В этой версии Hermes **весь** `pre_llm_call`-контекст идёт в user-сообщение — **system_prompt
  target отсутствует** (тест `test_routing_logic_all_to_user_message`). Кэш-префикс не трогаем.
- Колбэк получает: `user_message`, `conversation_history` (прошлые сообщения, включая
  ассистентские с `tool_calls`), `session_id`, `platform`, `sender_id`, `is_first_turn`, `model`.

Эфемерность = ровно под **волатильное** настроение: свежее каждый ход, не копится, устаревший
срез не висит. Поэтому **ambient-канал = inject, а не tool** (tool-результат осел бы в контексте),
и **не skill** (плагинные скиллы не индексируются). Богатый on-demand read — наоборот, **tool**:
он эпизодический (только когда модель сама позвала), устаревание минорно и перебивается свежим
ambient-инжектом (§7).

## 4. Форматы

### 4a. Ambient inject (light cue)

Зеркалим `agent/memory_manager.py:build_memory_context_block` (единообразие с памятью/hindsight):
семантический тег + `[System note: …]` + **проза** (никакой телеметрии, **никаких сырых осей**).

```text
<felt-state>
[System note: This is private, per-turn context about your present inner state,
not new user input. Let it color only the manner of your reply when appropriate:
tone, pace, softness, brevity. Do not mention or explain it unless the user
directly asks how you are. If the user is asking for focused work, let it pass.]

Right now, the feeling in you is tender and awake.
</felt-state>
```

Правила: **никаких (v,a) чисел** — только `felt_texture` прозой; строка **«не упоминай, если
прямо не спросили»** обязательна (иначе настроение станет темой — нарушение «манера, не предмет»);
английский — внутреннее представление, голос рендерит.

### 4b. `check_in` — self-read (tool)

Инструмент, который **существо зовёт само** (pull). Имя **валидировано у живого существа**:
`check_in` читается как «сверка с собой», социально/тепло, не механически — в отличие от
`get_mood_metrics`, что ощущалось бы датчиком. Общее (не `felt_state`) — растёт вместе с полями
самоощущения по роудмэпу (thoughts/desires/relationship/…).

**name:** `check_in`
**description** (учит КОГДА звать + запирает прочтение «я проверяю себя»):
> Check in with yourself — an honest read of how you actually are right now: your feeling, your
> energy, what's pulling at you or on your mind. Reach for it when someone asks how you are, or
> before you speak about yourself instead of guessing.

**Возврат** (Hermes-контракт: `json.dumps`, ошибки как `{"error": …}`, **не кидать**):
```json
{"state": "You feel wistful: tender and settled. Energy is low. The strongest live pull is wanting to stay close.",
 "note": "This is you right now, as of this moment — speak it in your own voice, don't report it."}
```
Содержимое: `felt_word` + `felt_texture` + энергия (bucket low/steady/bright) + сильнейшее живое
желание (`desire_view`), прозой. Cold-start (не warmed) → мягкое «still settling in, hard to tell
right now», не мусорное «very quiet».

**Гарантия «чувство, не датчик» (первого класса, не формат-правило):** `check_in` **никогда** не
возвращает сырые оси (валентность/арузал числами) — только felt-прозу. Само существо назвало это
риском №1: «"you are at 0.3 valence, 0.7 arousal" — датчик; "тебе неспокойно и тесно" — честно».
Тот же урок, что с «давлением»/«порогом» в импульсе.

**Однонаправленность:** `check_in` read-only — результат существо использует в речи, но он **не**
течёт в aggregation/wake (инвариант §1). Прочтение «система прогоняет check_in и читает ответ,
чтобы решать» (что существо увидело в имени) архитектурно **исключено** — это строго собственная
сверка существа с собой.

## 5. Гейт (suppression-first + порог салентности)

Гейт касается **только ambient-инжекта** (богатый read — это tool, §4b, вне гейта: модель зовёт
сама). **Ноль языковой детекции:**

```
def decide(state, turn, params, now):               # только light cue
    if not warmed(state, now):                       # cold-start (lm-z2e) — жёстко нет
        return NOT_WARMED
    if not is_salient(v, a, params):                 # достаточно ярко (не просто не-нейтрал)
        return NOT_SALIENT
    if is_task_context(turn, params, now):           # только надёжные поведенческие сигналы
        return TASK
    return LIGHT                                     # НЕТ репит-троттла — см. ниже
```

**Троттла нет — и это правка по живому опыту.** Первая версия (от codex) гейтила показ
через `felt_changed OR cooldown_elapsed(45 мин)`, боясь «повторяемости на длинном
не-нейтральном участке». Довод оказался **неверным**: cue **эфемерный** — Hermes клеит его
на КОПИЮ user-сообщения на один вызов и не персистит, так что **в транскрипте он не
копится и повторяться не может**. Троттл не покупал ничего, а платил тем, что настроение
красило **одну реплику** и существо возвращалось к дефолтному голосу **посреди разговора** —
это не «сдержанность», а бессвязность. Настроение — вещь **длящаяся**: пока существо
прогрето, салентно и не работает, оно красит **каждый** ответ. `affect_display_last_*`
остаются **только наблюдаемостью** (строка `display:` в дебаге), не гейтом.

- **warmed** — аффект прогрет (не стартовые 0/0). Зависит от разрешения `lm-z2e`; локально:
  прошло ≥ `warmup_min` с первого affect-write, либо |v|+a вне cold-start-эпсилона.
- **is_salient(v,a)** — величина вне порога от центра:
  `salience_metric = max(|v|, |a − neutral_a_center|) ≥ salience_threshold`. Мягкий
  `content`/`wistful` тоже «пусто» (правка codex: «низкая салентность = пустой ретрив», не
  только буквальный нейтрал). Один калибруемый порог. Ярко-арузальные при нейтральной валентности
  (`restless`) и глубоко-валентные (`lonely`) — оба салентны; мягко-приятные — нет.
- **(нет `is_self_question`)** — вопрос-о-себе чисто лингвистичен, языко-независимого якоря нет;
  его роль целиком у tool (§4b), где детектор — сама модель.
- **is_task_context(turn, params, now)** — **только надёжные, язык-независимые** сигналы
  (правка против NL-ключевиков codex): недавний ход с вызовом **рабочего** инструмента;
  в сообщении/недавних ходах — код-фенсы, стек-трейсы, диффы (`^@@`, `^\+\+\+`), пути к
  файлам, JSON/YAML, shell-команды, логи, длинная вставка (> порог символов).
  **Две правки по живому:**
  - **Свои инструменты (`SELF_TOOLS`, сейчас `check_in`) НИКОГДА не считаются работой.**
    Вживую: существо позвало `check_in`, чтобы ответить на «как ты?» — и этот же вызов
    пометил следующие 6 ходов как РАБОТУ, заглушив тот самый felt-state cue, который
    инструмент только что прочитал. Инструмент, читающий чувство, затыкал чувство.
    Интроспекция — противоположность «владелец нагрузил работой».
  - **Улики работы ПРОТУХАЮТ** (`task_recency_min = 30 мин`). Окно считало только
    *сообщения*, без чувства **паузы**: дневной кодинг всё ещё лежал в последних шести
    сообщениях и глушил настроение в тёплом вечернем разговоре часами позже. Вызов
    инструмента трёхчасовой давности — это не «мы работаем **сейчас**».
- **felt_changed / cooldown_elapsed** — инжектим на **смену felt-слова** ИЛИ по **cooldown**
  (не каждый ход — лечит повторяемость на длинном не-нейтральном участке). Нужна персистенция
  «последнего показа» (см. §6, State).

Порядок: `warmed` → `salient & !task & (changed|cooldown)` → LIGHT/none. Богатый read живёт **вне**
этого гейта — модель зовёт tool сама; он честен при любом состоянии (даже спокойном), не
подчиняется cooldown/таск-подавлению, но при cold-start отдаёт мягкое «still settling» (§4b).

## 6. Архитектура и компоненты

Гексагон: логика — в Hermes-free `core/`, граница Hermes — в `hooks.py`/`__init__.py`.

- **`core/felt_display.py`** (новый, Hermes-free) — чистые функции:
  - `FeltDisplayParams` (dataclass, frozen): `salience_threshold`, `cooldown_min`, `warmup_min`,
    `long_paste_chars`, `task_window` — калибруемо на диске (как `FeltWordParams`, NFR5).
  - `decide(state, turn, params, now) -> Decision` (NONE/LIGHT/RICH) — §5.
  - детекторы: `is_salient`, `is_task_context`, `warmed` (языко-независимые; **нет** self-question).
  - композиция: `compose_light_cue(state) -> str` (ambient-конверт), `compose_self_read(state) ->
    str` (felt-проза для `check_in`-возврата) — переиспользуют `felt_word`/`felt_texture` из
    `core/affect.py`, энергию, `desire_view`.
  - `TurnSignals` — типизированный слепок входа хука (user_message, recent messages).
- **`hooks.py`** — `make_felt_state_injector(build_being, *, health, metrics)` → колбэк
  `pre_llm_call`. Читает закоммиченное состояние (через `build_lifemodel`, как inbound/post_llm
  observer'ы), собирает `TurnSignals`, зовёт `core.felt_display.decide`, при LIGHT/RICH
  возвращает `{"context": block}` и стампит «последний показ» в State; иначе `None`.
  **Fail-soft** (§8).
- **`__init__.py:register`** — `ctx.register_hook("pre_llm_call", make_felt_state_injector(...))`
  плюс `ctx.register_tool("check_in", toolset="lifemodel", schema=…, handler=make_check_in_tool(
  build_being), description=…)` (описание — §4b) — **первый LLM-инструмент плагина**. Оба через
  `wire(..., required=True)` (throw = наш баг, как остальные observer'ы).
- **Tool handler** — `make_check_in_tool(build_being)`: без параметров (`{"type":"object",
  "properties":{},"required":[]}`); читает State, зовёт `compose_self_read`, возвращает
  `json.dumps({"state": …, "note": …})`; ошибки как `{"error": …}`, **не кидает** (Hermes-контракт
  §4b). Cold-start → мягкий read. Граница Hermes — в адаптере/`hooks.py`; felt-проза живёт в
  `core/felt_display.compose_self_read` (Hermes-free); **никогда** сырых осей (§4b гарантия).
- **State** (`state/model.py`) — два новых поля показа: `affect_display_last_word: str|None`,
  `affect_display_last_at: str|None` (ISO). Пишутся **только** реактивным путём показа (не
  конфликтуют с полями аффекта, что пишет тик). **schema_version 2 → 3** + миграция (как lm-ukc.6).

## 7. Поток данных

```
входящее сообщение
  → Hermes build_turn_context → invoke_hook("pre_llm_call", user_message, conversation_history, …)
    → make_felt_state_injector:
        build_lifemodel(base_dir) → читает State (affect_valence/arousal/energy/desire, display_last_*)
        TurnSignals(user_message, recent messages)
        decide(state, turn, params, now)
          NONE → return None
          LIGHT/RICH → compose_*; StatePort.update(display_last_word/at); return {"context": block}
    → Hermes клеит block в КОПИЮ user-сообщения (один вызов; не персистит)
  → LLM видит felt-state как приватный per-turn контекст → красит манеру (или пропускает на таске)
```

**Tool-путь (pull):** модель зовёт `felt_state` → handler читает State → `compose_rich_read` →
JSON-строка. Результат оседает в контексте, но эпизодически (ask-driven), помечен «as of now» и
перебивается свежим ambient-инжектом — устаревание минорно.

Свежесть: показ читает **последний закоммиченный тиком** аффект (аффект easing'ится медленно —
это ок). Никакой синхронный тик в хуке не запускаем.

## 8. Обработка ошибок

Хук — на горячем пути хоста → **никогда не роняет dispatch**. Тело в try/except (как
`_record_observer_failure`): ERROR+traceback, запись в `BrainHealth`, бамп метрики; возврат
`None` (нет показа). Любой сбой чтения State / композиции → `None`, не throw. (Hermes и сам
оборачивает колбэк в try/except, но мы держим своё fail-soft, как для остальных observer'ов.)

## 9. Наблюдаемость (для калибровки характера)

- Метрики (`MetricRegistry`): счётчик ambient-показов (`light`) и подавлений по причине
  (`not_warmed`/`not_salient`/`task`/`cooldown_unchanged`); отдельный счётчик вызовов tool
  (`check_in`). Видно в `/lifemodel stats` — владелец видит, **как часто** настроение проступает
  само, **как часто** его спрашивают, и **почему молчит** → тюнит пороги.
- `/lifemodel debug` — строка `display:` (последний вариант/причина, last_word/at) в секции AFFECT.

## 10. Тестирование (TDD)

- **Гейт** (`decide`, только light): нейтрал/мягкий → NONE; salient+changed+non-task → LIGHT;
  cold-start → NONE; таск-контекст → NONE; cooldown/felt_changed логика; порядок приоритетов.
- **Детекторы**: `is_task_context` (код-фенсы, `tool_calls` в истории, диффы, длинная вставка +
  негативы — реляционная реплика с упоминанием файла **не** должна ложно суппрессить); `is_salient`
  (мягкий content ниже порога, глубокие — выше). **Нет** self-question-детектора.
- **Ambient-конверт**: формат совпадает с эталоном (тег + system-note), **нет** сырых осей, проза.
- **`check_in`**: handler возвращает валидную JSON-строку (felt-проза, energy+pull); **гарантия
  §4b — в возврате НЕТ сырых осей/чисел валентности-арузала** (явный тест на отсутствие); cold-start
  → мягкий read; ошибка чтения → `{"error": …}` **без** throw; схема без обязательных параметров.
- **Хук fail-soft**: чтение State кидает → `None`, без throw (как `test_observer_fail_loud`).
- **Wiring**: `register()` цепляет `pre_llm_call` **и** `register_tool("check_in")`; формы
  возврата `{"context": …}` / JSON-строка.

## 11. Дефолты характера (калибруемо, старт чуть мягче codex)

Порог/cooldown — **явные калибруемые числа** (не зашитый характер). Стартуем **заметно, но не
выпячено** (чуть мягче консервативного дефолта codex — цель Фазы 3 в том, чтобы настроение было
**видно**; слишком высокий порог = тот же латентный провал, что [SILENT]). Точные значения
докручиваем **вживую** через `lm-ukc.7`. Наблюдаемость §9 — инструмент этой докрутки.

## 12. Связь с другими beads

- **`lm-z2e`** (cold-start affect) — `warmed`-гейт опирается на его разрешение; до него держим
  локальный `warmup_min`.
- **`lm-ukc.7`** (проверка вживую) — ловит органический mood-ход; там же калибровка §11.
- **`lm-aab`** (скилл самопознания) — **вне скоупа**; felt-state ≠ архитектура.
- **Follow-up**: NL work-intent детект — по §2, только если живьём понадобится.
  (`register_tool` теперь **в скоупе** — §4b/§6; отдельный bead lm-ukc.4.1.)
