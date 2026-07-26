# Полный список сред в `baba-is-ai`

В репозитории зарегистрировано **53 environment ID**:

- **43 основные задачи**
- **10 специальных OOD-сплитов**

Это процедурные генераторы, а не фиксированные уровни: типы объектов, их цвета и позиции могут случайно меняться.

## Обозначения модификаторов

- `distr_obj` — дополнительный нерелевантный объект.
- `distr_rule` — дополнительный текстовый блок с названием объекта.
- `distr_obj_rule` — дополнительный объект и соответствующий ему текстовый блок.
- `distr_obj-irrelevant_rule` — дополнительный объект и текстовый блок другого, не связанного с ним типа.
- `distr_win_rule` — дополнительное активное, но нерелевантное правило `X IS WIN`.
- `#no_ball_win` — `ball` никогда не является целевым объектом.
- `#only_ball_win` — целевым объектом всегда является `ball`.

| Семейство | Конкретные задачи | Описание семейства | Краткое описание конкретной задачи |
|---|---|---|---|
| **You–Win identification** | `you_win`<br>`you_win-fixed_you` | Минимальная задача на определение управляемого и целевого объектов. Агент должен понять из правил, какой объект является `YOU`, а какой — `WIN`. | `you_win` — позиция управляемого объекта случайна.<br><br>`you_win-fixed_you` — управляемый объект всегда стартует в фиксированной клетке, но его тип и цель остаются случайными. |
| **Single-room Goto Win** | `goto_win`<br>`goto_win-distr_obj`<br>`goto_win-distr_rule`<br>`goto_win-distr_obj_rule`<br>`goto_win-distr_obj-irrelevant_rule`<br>`goto_win-distr_win_rule` | Однокомнатная карта. Правило `X IS WIN` уже активно: агенту нужно определить целевой объект и дойти до него. | `goto_win` — базовая навигация без дистракторов.<br><br>`goto_win-distr_obj` — добавлен лишний объект.<br><br>`goto_win-distr_rule` — добавлен лишний текстовый блок.<br><br>`goto_win-distr_obj_rule` — добавлены лишний объект и связанный с ним блок.<br><br>`goto_win-distr_obj-irrelevant_rule` — добавлены объект и семантически не связанный с ним блок.<br><br>`goto_win-distr_win_rule` — добавлено активное, но нерелевантное правило `X IS WIN`. |
| **Single-room Make Win** | `make_win`<br>`make_win-distr_obj`<br>`make_win-distr_rule`<br>`make_win-distr_obj_rule`<br>`make_win-distr_obj-irrelevant_rule` | Однокомнатная карта. Правило `X IS WIN` изначально разрушено. Нужно передвинуть слово, восстановить правило и затем дойти до объекта `X`. | `make_win` — чистая задача без дистракторов.<br><br>`make_win-distr_obj` — добавлен лишний объект.<br><br>`make_win-distr_rule` — добавлен лишний текстовый блок.<br><br>`make_win-distr_obj_rule` — добавлены лишний объект и его текстовый блок.<br><br>`make_win-distr_obj-irrelevant_rule` — добавлены объект и блок другого типа. |
| **Single-room Make Win: semantic OOD** | Для каждого из пяти вариантов `make_win*`:<br><br>`#no_ball_win`<br>`#only_ball_win` | Специальные train/test-сплиты для проверки генерализации по типу целевого объекта. Всего получается 10 environment ID. | `#no_ball_win` — целью может быть только `key` или `door`.<br><br>`#only_ball_win` — целью всегда является `ball`.<br><br>Пример полного ID: `make_win-distr_obj_rule#only_ball_win`. |
| **Two-room Goto Win** | `two_room-goto_win`<br>`two_room-goto_win-distr_obj`<br>`two_room-goto_win-distr_rule`<br>`two_room-goto_win-distr_obj_rule`<br>`two_room-goto_win-distr_obj-irrelevant_rule`<br>`two_room-goto_win-distr_win_rule` | Двухкомнатная карта. Цель доступна без изменения правил. Основная задача — навигация в более сложной геометрии. | `two_room-goto_win` — базовая навигация.<br><br>Остальные варианты добавляют лишний объект, слово, их комбинацию, несвязанные дистракторы или ложное активное `WIN`-правило. |
| **Two-room Break Stop → Goto** | `two_room-break_stop-goto_win`<br>`two_room-break_stop-goto_win-distr_obj`<br>`two_room-break_stop-goto_win-distr_rule`<br>`two_room-break_stop-goto_win-distr_obj_rule`<br>`two_room-break_stop-goto_win-distr_obj-irrelevant_rule` | Цель находится в другой комнате, но активное правило `WALL IS STOP` блокирует проход. Нужно разрушить правило стены и затем дойти до цели. | `two_room-break_stop-goto_win` — план `break[WALL IS STOP] → goto`.<br><br>Остальные варианты добавляют разные типы объектных и текстовых дистракторов. |
| **Two-room Maybe Break Stop → Goto** | `two_room-maybe_break_stop-goto_win`<br>`two_room-maybe_break_stop-goto_win-distr_obj`<br>`two_room-maybe_break_stop-goto_win-distr_rule`<br>`two_room-maybe_break_stop-goto_win-distr_obj_rule`<br>`two_room-maybe_break_stop-goto_win-distr_obj-irrelevant_rule` | Цель может находиться в любой комнате. Агент должен определить, требуется ли разрушать `WALL IS STOP`. | `two_room-maybe_break-stop-goto-win` — выбор между двумя стратегиями:<br><br>`goto` — если цель доступна напрямую;<br><br>`break → goto` — если цель находится за стеной.<br><br>Остальные варианты добавляют дистракторы. |
| **Two-room Make Win → Goto** | `two_room-make_win`<br>`two_room-make_win-distr_obj`<br>`two_room-make_win-distr_rule`<br>`two_room-make_win-distr_obj_rule`<br>`two_room-make_win-distr_obj-irrelevant_rule`<br>`two_room-make_win-distr_win_rule` | Проход через стену не требуется, но правило `X IS WIN` разрушено. Агент должен восстановить правило и дойти до объекта `X`. | `two_room-make_win` — план `make[X IS WIN] → goto[X]`.<br><br>Остальные варианты добавляют объектные, текстовые, несвязанные или активные `WIN`-дистракторы. |
| **Two-room Break Stop → Make Win → Goto** | `two_room-break_stop-make_win`<br>`two_room-break_stop-make_win-distr_obj`<br>`two_room-break_stop-make_win-distr_rule`<br>`two_room-break_stop-make_win-distr_obj_rule`<br>`two_room-break_stop-make_win-distr_obj-irrelevant_rule` | Композиционная задача: цель находится в другой комнате, стена непроходима, а выигрышное правило разрушено. | `two_room-break_stop-make_win` — план `break[WALL IS STOP] → make[X IS WIN] → goto[X]`.<br><br>Остальные варианты добавляют разные нерелевантные объекты и слова. |
| **Change controllable entity** | `two_room-make_you`<br>`two_room-make_you-make_win`<br>`two_room-make_wall_win` | Семейство задач, где агент должен изменить управляемый объект или семантику стены. | `two_room-make_you` — построить `X IS YOU`, начать управлять новым объектом и достичь цели.<br><br>`two_room-make_you-make_win` — сначала создать нового `YOU`, затем создать `Y IS WIN` и достичь объекта `Y`.<br><br>`two_room-make_wall_win` — разрушить `WALL IS STOP`, построить `WALL IS WIN` и коснуться стены. |

## Иерархия сложности

```text
you_win
→ goto_win
→ make_win
→ two_room-goto_win
→ two_room-break_stop-goto_win
→ two_room-maybe_break_stop-goto_win
→ two_room-make_win
→ two_room-break_stop-make_win
→ two_room-make_you
→ two_room-make_you-make_win