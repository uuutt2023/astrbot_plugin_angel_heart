# Jev 决策后端

秘书判断「该不该接话」默认由轻量模型承担。本插件额外接了 [Jev 决策模型](https://jevmodel.org/zh/docs/)：它不生成文本，只回答类型化问题，因此可以作为秘书的决策后端，也可以作为调用大模型前的廉价门控。

传输层与判读层的实现逻辑对齐 `astrbot_plugin_oven_multi` 的同名模块（两个插件各自独立分发，不跨插件 import）。

## 三种决策方式

配置项 `decision_backend`（插件配置页「秘书决策方式」下拉）：

| 值 | 路径 | 决策来源 |
|----|------|----------|
| `llm` | `Secretary.perform_analysis` → `LLMAnalyzer.analyze_and_decide` | 轻量秘书模型产出完整决策 |
| `jev` | `Secretary.perform_analysis` → `Secretary._decide_by_jev` | 只用 Jev，不调用大模型 |
| `hybrid` | `Secretary.perform_analysis` → `Secretary._jev_gate` → 秘书模型 | Jev 先门控，通过才调用秘书模型 |

判定顺序：`decision_backend` 命中 `jev` / `hybrid` 且 `jev_decision.enabled` 为真且填了 API Key 时才走 Jev；否则一律回落到 `llm`。回落后行为与改造前完全一致。

`hybrid` 的语义是「Jev 与秘书模型都同意才回复」：Jev 门控未通过时直接返回不回复，**不再调用秘书模型**（省 token）；门控通过后仍由秘书模型产出完整决策。门控自身失败（网络、上游故障）不阻断主流程，放行给秘书模型判断。

`must_reply`（点名强制回复）在三条路径下都照旧生效——它在 `perform_analysis` 返回之后统一覆盖 `should_reply`。

## Jev 路径的映射

Jev 的问题集固定 4 道（文档上限 8 道），全部用英文提问（Jev 主训练语言是英文，CJK 精度较低）：

| 问题 | 题型 | 对应决策字段 | 阈值配置 |
|------|------|--------------|----------|
| `should_reply` | noul | `should_reply` | `reply_threshold`（默认 0.6） |
| `is_questioned` | noul | `is_questioned` | `questioned_threshold`（默认 0.5） |
| `is_interesting` | noul | `is_interesting` | `interesting_threshold`（默认 0.5） |
| `reply_strategy` | choice | `reply_strategy` | 枚举，映射回中文 |

noul 答案是「为真」的概率，过阈值才算成立。choice 的选项 key 为英文（`answer_question` / `join_topic` / `offer_help` / `greet` / `stay_silent`），解析边界映射回中文展示；选 `stay_silent` 直接置为不回复。

Jev 不产出自由文本，所以 Jev 路径下的 `topic` 用最新一条消息近似，`reply_target`、`entities`、`facts`、`keywords` 留空。

**正向条件规则与 LLM 路径一致**：`should_reply AND (is_questioned OR is_interesting OR reply_even_not_questioned)`。没有正当理由就不回复，除非开启了「非提问也回复」。

## 传输层

`core/jev_transport.py` 对齐 Jev API 文档：

- **端点**：`POST {base_url}/v1/systemone`，两个网关请求体一致，切换只需换 base URL 与 Bearer key。默认网关是 `https://jevmodel.org`；`https://api.typesafe.ai` 在部分区域会返回 451，按实际可达性选。
- **本地预检**：问题数（≤8）、问题名（标识符，≤64 字符）、state 长度（≤100 万字符）、instructions（≤1800）、criteria 序列化长度（≤2000）、choice 选项数（2–20）、score 级数（2–10）。超限直接拦截，不白发一次请求。
- **Idempotency-Key**：一次逻辑调用一把键（`angel-` + 32 位 hex，38 字符 ≤ 100）。同一次调用内的重试复用它，不会重复扣费；服务端已回过错误时换新键重跑，避免撞上「上一把键仍在处理中」的 409；压根没收到响应时沿用旧键，防止请求已被受理却重复计费。键按调用生成而非按内容派生——内容相同的两条消息曾共用一把键互相冲突。
- **错误码映射**：`401` unauthorized、`402` insufficient_credits、`409` conflict、`422` invalid_request、`429` rate_limited、`502` upstream_error。`409` / `429` / `502` 按退避重试，其余不重试；错误体 `{"error":{"type","message"}}` 会被解析并原样透出。
- **限流**：文档限每 key 每分钟 120 次，传输层本地先拦一道，省掉必然被 429 的往返。
- **余额**：读取响应头 `X-Tokens-Remaining` 记录结算后余额。
- **预算同源**：`budget_seconds()` 是传输层、判读层与调用方共用的口径，避免外层守卫先掐断内层请求、让调用方只看到「超时」而看不到真实原因。

## 可视化配置

「秘书决策方式」在插件配置页与 AstrBot 原生配置页都是下拉框：

- `_conf_schema.json` 用 `options` 存保存值、`labels` 存展示文本、`subtitles` 存选项说明，三者按索引对应（与 AstrBot 原生配置页一致）。
- 插件自带 WebUI 的 `SchemaForm.vue` 增加了 `options` 渲染分支：枚举字段渲染为下拉框，展示文本取 `labels`，选中项的 `subtitles` 显示在控件下方。构建产物在 `pages/chat-config/`。
- `web_api` 的配置校验会拒绝不在 `options` 内的取值。

## 性能

决策路径的本地 CPU 成本（不含网络往返）：

| 环节 | 优化前 | 优化后 |
|------|--------|--------|
| 格式化 50 条历史（LLM 路径） | 610 µs | 82 µs |
| `build_state`（Jev 路径，12+4 条） | 183 µs | 32 µs |
| `JevClient.evaluate()` 请求准备 | 304 µs | 173 µs |
| 一次决策的配置读取（for_chat + 4 字段） | 8.3 µs | 3.2 µs |

要点：

- **昵称不进循环**：`LLMAnalyzer._format_conversation_history` 里 `for_chat()` 原本在每条消息上调一次（最多 50 次），而昵称与整段历史无关——提到循环外。
- **模板覆盖值缓存**：`ChatProfileStore.resolve_override` 按模板缓存过滤后的配置，`ConfigManager` 视图内再缓存一次解析结果；模板更新 / 删除 / 绑定变更时失效。
- **绝对时间按分钟缓存**：输出精确到分钟，同一分钟共用一份结果；`fromtimestamp` 得到的已经是当地时间，不再重复 `astimezone()`。
- **传输层只读不拷贝**：state 与问题在传输层全程只读，不再深拷贝（state 可达百万字符）；请求体只序列化一次，同时用于缓存键、字节计数与实际请求，线上字节与原来完全一致。
- **预检只算必要的长度**：文档按序列化后的字符数设限，而序列化最多把一个字符膨胀成 6 个；raw 长度小到膨胀 6 倍都够不着上限时，不必真去 dumps 一遍。

有一处**刻意不优化**：`FrontDesk.filter_images_for_provider` 里对每条消息的 `copy.deepcopy` 是刻意的隔离屏障——下游会把消息对象原样带进请求体，复用原对象可能污染账本里的消息。

## 相关文件

- `core/jev_transport.py`：Jev 决策端点客户端（预检 / 幂等键 / 错误码 / 限流 / 余额）
- `core/jev_decider.py`：问题集、state 拼装、答案映射、预算内重试
- `roles/secretary.py`：三条决策分支与回落
- `core/config_manager.py`：`decision_backend` / `jev_decision`
- `_conf_schema.json`：「Jev 决策」分组与「秘书决策方式」下拉
- `tests/test_jev_decision_backend.py`：状态拼装、限额预检、概率映射、三条分支、可视化配置
