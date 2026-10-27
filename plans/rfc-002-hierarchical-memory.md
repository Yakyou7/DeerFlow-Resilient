# RFC-002: 分层长程记忆子系统 (Aimem) 设计规范
**Route B: Hierarchical Long-Term Memory Subsystem (`backends/aimem`) Specification**

- **状态 (Status)**: Approved / In Progress
- **作者 (Author)**: DeerFlow Architecture & Optimization Team
- **日期 (Date)**: 2026-10-27
- **关联理论 (Theoretical Foundation)**: 《AI Agent 开发实战》第 3 章（记忆与 RAG）、`ai-memory` 规范

---

## 1. 背景与核心动机 (Motivation & Problem Statement)

在智能体开发中，内存子系统（Long-term Memory）是实现跨会话连续性、用户偏好对齐与领域知识积累的关键。然而，DeerFlow 原生的 `DeerMem` 实现存在以下三大痛点：

### 1.1 痛点一：盲目全量调用 LLM，延迟与成本高昂
- **现状**：原生 DeerMem 在每个会话轮次结束时，无条件在后台发起昂贵的 LLM 事实抽取调用（Prompt 包含完整对话历史及长格式规范）。
- **危害**：对大量普通检索、简单计算或中间过渡轮次，调用 LLM 抽取毫无产出，白白消耗大量 Token 预算，并在后台积压并发任务。

### 1.2 痛点二：黑盒 JSON 存储，缺乏透明性与可编辑性
- **现状**：记忆全部序列化存放在单体 `memory.json` 文件中。
- **危害**：
  1. 人类开发者和业务人员无法直观查看或手动微调/纠偏错误事实；
  2. 随着时间积累，单体 JSON 序列化和反序列化开销急剧攀升；
  3. 无法直接作为 Git 跟踪的 Markdown 知识库协同。

### 1.3 痛点三：平铺事实缺乏重要性分层与热度衰减
- **现状**：所有抽取的事实一律平铺在 `facts: []` 列表中。
- **危害**：缺乏时效性与访问频度加权，多月前的陈旧临时事实与核心长期用户偏好具有相同优先级，在上下文检索注入时容易导致 Context Rot（上下文腐化）。

---

## 2. 架构设计与核心原则 (Architecture & Principles)

针对上述缺陷，本规范设计 `aimem`（AI Hierarchical Memory）后端，遵循以下核心原则：

### 2.1 原则一：Markdown 作为唯一真实事实源 (Single Source of Truth)
* **存储布局**：在 `{storage_path}/aimem/` 目录下，按分类落盘为结构化 Markdown：
  ```
  {storage_path}/aimem/
  ├── users/
  │   └── {user_id}/
  │       ├── profile.md          # 用户画像、偏好、习惯与硬性约束
  │       ├── entities.md         # 提及的关键实体、项目名、常用服务
  │       └── archive.md          # 历史会话关键决策沉淀
  └── derived/
      └── search_index.db         # 本地派生 SQLite FTS5 倒排索引
  ```
* **人类可读与可纠偏**：开发者可以直接用 Markdown 编辑器增删改事实，无需通过任何专有 API。

### 2.2 原则二：本地轻量 SQLite FTS5 毫秒级派生索引
* **架构解耦**：SQLite 数据库仅作为**派生索引（Derived Cache）**存在，丢失或删除后可由 Markdown 实时重建。
* **极速检索**：利用 SQLite 内置的 `FTS5`（全文检索）与 `BM25` 排序算法，支持秒级建立索引、毫秒级（< 3ms）关键词与短语检索，且**零外部依赖、零云端向量计费、零网络延迟**。

### 2.3 原则三：门控式启发式抽取 (Gated Heuristic Extraction)
* **过滤机制**：在调用 LLM 抽取前，增加轻量规则门控（Rule-based Pre-filter）：
  - 过滤短消息（如 `< 15` 字符）；
  - 过滤纯确认/感谢类语气词（`ok`, `thanks`, `好的`, `收到`）；
  - 过滤无实质用户偏好信号的纯代码执行回显。
* **收益**：削减 70%~80% 的无效后台 LLM 请求，彻底降低 API 调用开销与并发阻塞。

### 2.4 原则四：严格遵守 DeerFlow Pluggable MemoryContract
* 继承 `MemoryManager` 抽象基类；
* 遵从不可侵入原则：**零侵入修改 DeerFlow 核心代码**，仅通过 `manager_class: aimem` 即可实现热插拔与一键切换；
* 响应格式完全对齐 DeerMem 契约（`MemoryResponse` 规范），完美兼容前端与管理端点。

---

## 3. 详细接口设计 (Contract Specification)

### 3.1 目录结构与模块划分
```
deer-flow/backend/packages/harness/deerflow/agents/memory/backends/aimem/
├── __init__.py               # 暴露 MANAGER_CLASS = AimemManager
├── config.py                 # AimemConfig (解析 backend_config, 存储路径, 阈值配置)
├── aimem_manager.py          # AimemManager (实现 MemoryManager 接口)
├── storage.py                # Markdown 文件读写、序列化与版本管理
└── indexer.py                # SQLite FTS5 索引构建、更新与 BM25 检索
```

### 3.2 Tier-1 核心接口
1. **`add(thread_id, messages, *, user_id=None, agent_name=None, trace_id=None)`**:
   - 提取会话中的关键交互；
   - 经门控筛选后，若包含有价值的事实，异步/同步更新 Markdown 并刷新 FTS5 索引。
2. **`get_context(user_id, *, agent_name=None, thread_id=None, query=None) -> str`**:
   - 若提供 `query`，通过 FTS5 快速检索最相关的 Top-K 事实；
   - 若未提供 `query`，读取 `profile.md` 中的核心全局偏好；
   - 格式化输出为 `<memory>...</memory>` 上下文注入块。

### 3.3 Tier-2 检索与管理接口
1. **`search(query, top_k=5, *, user_id=None, agent_name=None, category=None) -> list[dict]`**:
   - 执行 FTS5 BM25 全文检索；
   - 返回标准字典列表：`[{"id": ..., "content": ..., "category": ..., "score": ...}]`；
   - 设置 `supports_search: ClassVar[bool] = True`。
2. **`get_memory(*, user_id=None, agent_name=None) -> dict`**:
   - 汇总体系统计（用户偏好、实体数、历史沉淀）；
   - 返回标准 `MemoryResponse` 兼容字典。
3. **`clear_memory(*, user_id=None, agent_name=None) -> dict`**:
   - 清理指定用户的 Markdown 文件与索引条目。

---

## 4. 验收标准与测试矩阵 (Verification & Testing)

### 4.1 单元测试套件 (`test_aimem_backend.py`)
1. **测试用例 1（契约合法性）**：`AimemManager` 实例化成功，`supports_search == True`，无任何未实现抽象方法；
2. **测试用例 2（Markdown 存储与解析）**：事实写入后生成合法 Markdown，手工修改 Markdown 后可无缝重新解析；
3. **测试用例 3（FTS5 检索精度）**：插入若干含关键词（如“咖啡”、“退款规则”、“Python版本”）的记忆，FTS5 检索在 5ms 内准确定位；
4. **测试用例 4（上下文注入格式）**：`get_context` 产出规范文本，未命中时不抛异常返回空串；
5. **测试用例 5（门控过滤）**：无效短会话不触发持久化写入。

### 4.2 集成冒烟测试
- 在 `config.yaml` 中配置 `memory.manager_class: aimem`；
- 执行 `deerflow --json "Remember that my favorite beverage is cold brew"`；
- 紧接着在新会话中提问 `"What beverage do I like?"`；
- 验证 Agent 成功通过 `aimem` 召回用户偏好并准确回答。

---

## 5. 分步实施计划 (Implementation Roadmap)

1. **Step 1: 规范文档确认与骨架搭建**（`config.py`, `aimem_manager.py`, `__init__.py`）；
2. **Step 2: 存储与检索引擎实现**（`storage.py`, `indexer.py`）；
3. **Step 3: 编写完备单元测试**（`backend/tests/test_aimem_backend.py`）并跑通 100% 通过；
4. **Step 4: 配置接入与在线集成验证**（接入 Qoder 极速基座做真实双轮记忆回溯）。
