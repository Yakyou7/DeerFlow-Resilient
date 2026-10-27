# 技术方案规范：DeerFlow 上下文工程与压缩治理重构 (RFC-001)

> **文档状态**：PROPOSED（待评审 / 实施准备）  
> **制定日期**：2026-10-27  
> **对齐理论**：《AI Agent 理论与工程实践》（ai-agent-book）第二章《上下文工程》& 实验 2-10  
> **实验基线**：`experiments/BASELINE.md`（2026-10-21 采集数据，含 t1/t2/t3 任务）  
> **涉及核心代码**：
> - `deerflow/agents/middlewares/summarization_middleware.py`
> - `deerflow/config/summarization_config.py`
> - `deerflow/config.yaml`（`summarization` 配置段）

---

## 1. 背景与问题定义（Problem Statement）

### 1.1 现状实测数据（Baseline Evidence）
在 `experiments/BASELINE.md` 的长任务真实采集运行中，暴露了当前 DeerFlow 原生上下文压缩机制的严重缺陷：

```
t2 运行轨迹（单次请求 Token 变化）：
23,092 → 23,714 → 24,336 → 25,646 → 26,162 → 31,136 → 30,052 → 35,424 
  ➔ [触发摘要] ➔ 31,420 ➔ 33,924 ➔ 34,194 ➔ 34,338
```

* **指标反差**：阈值设定为 `32,000` tokens，当在第 8 步达到峰值 `35,424` 触发压缩后，仅削减到 **`31,420` tokens（仅削减 4,004 tokens，约 11%）**。
* **贴天花板震荡**：随后仅过 1 步，上下文又迅速回弹至 `33,924`，并在此后全程（第 9~12 步）贴着 32K 天花板反复触发压缩（t2 累计触发 24 次非空摘要计算）。
* **资源浪费与注意力腐化（Context Rot）**：
  1. 摘要模型高频空转，浪费大量推理配额与时间（t2 墙钟耗时高达 495 秒）；
  2. 贴着天花板震荡的上下文中充斥着陈旧工具调用返回的原始长文本，加剧长程注意力的“大海捞针”负担，增加模型幻觉和工具重复调用的风险。

### 1.2 根因定位（Root Cause Analysis）
通过剖析 `summarization_middleware.py` 与配置结构，发现三个核心缺陷：
1. **盲目保留重资产（Naive Retention Policy）**：
   - 当前配置为 `keep: {type: messages, value: 10}`。
   - 在 Agent 执行长任务时，最近 10 条消息通常是连续的 `(AIMessage(tool_calls), ToolMessage(content))`。
   - 在多文件读取（`read_file`）或大量命令行（`bash`）操作中，每个 `ToolMessage` 单独占用 500~2000 tokens。保留 10 条消息就保留了高达 **2.5万~3万 tokens 的原始工具输出**，导致无论怎么摘要历史，底部的 10 条消息就直接把窗口占满了。
2. **工具输出缺乏分级治理（No Tool Output Truncation / Offload）**：
   - 现有的工具结果在进入状态图后，未经任何体积削减即作为完整消息永久驻留。
3. **摘要过程脱离任务目标（Agnostic Summarization）**：
   - 默认的 `summary_prompt` 仅按时间线做流水账概括，无法感知“当前用户到底在完成什么任务、需要关注哪些具体指标”，压缩比低且容易遗漏核心线索（对应书里实验 2-10 的低效策略 2/3）。

---

## 2. 目标与非目标（Goals and Non-Goals）

### 2.1 核心目标（Goals）
1. **压缩幅度量化达标**：
   - 触发压缩后，输入上下文峰值从原来的 `>31,000` tokens 压降至 **`<18,000` tokens**（单次压缩释放 45%+ 的上下文空间，消除贴顶震荡）。
2. **总 Token 开销与耗时降低**：
   - 在 t2（30文件逐个读取）等工具重度长任务中，全任务总输入 Token 消耗降低 **30% ~ 45%**；
   - 减少 80% 以上的无效高频重复摘要计算，墙钟耗时明显下降。
3. **零回归（Zero Regression）**：
   - t1、t2、t3 的最终产物（`report.md`, `findings.md`, `analysis.md`）内容完整度与准确率达到 100%，无信息遗失。
4. **建立离线重放闭环（Deterministic Offline Replay Harness）**：
   - 开发零 API 成本、秒级执行的离线压缩重放脚本，使策略迭代周期从 8 分钟缩短至 1 秒以内。

### 2.2 非目标（Non-Goals）
* 本方案不修改 Gateway 对外暴露的 HTTP/SSE 协议；
* 本方案暂不涉及外部持久化记忆库（`backends/aimem`，该工作留待路线 B 推进）；
* 本方案不对模型的权重或推理服务器底层 KV Cache 驱动层进行改动。

---

## 3. 理论依据与设计对齐（Theoretical Grounding）

方案全面对照《AI Agent 理论与工程实践》第二章实验 2-10 的实测结论与 `harness2` 工业级外骨骼设计：

| 本方案设计模块 | 书本/参考项目来源 | 理论机理 |
| :--- | :--- | :--- |
| **工具输出智能截断与外置 (Tool Offload)** | `harness2` Tool-Loop Middleware | 确定性外骨骼硬拦截：超过阈值的工具历史输出强制只保留首尾摘要与落盘指针，从物理源头遏制 Token 膨胀。 |
| **Token-Bounded 双阈值保留策略** | 书本 2-10 策略 6（自适应窗口） | 废除固定 10 条消息保留，改为“动态保留最近 $K$ 个 Token 且工具输出瘦身”，确保压缩后给后续步骤留足爬升缓冲带。 |
| **任务感知上下文摘要 (Task-Aware Prompt)** | 书本 2-10 策略 4（上下文感知摘要） | 将用户的原始 Query/Goal 注入摘要 Prompt，使模型只保留与核心任务强相关的结论与状态，压缩比可达 4:1~6:1。 |

---

## 4. 详细技术方案设计（Detailed Design）

### 4.1 模块架构图

```
                Agent State: messages (32K+ Tokens)
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │ 阶段一：消息切分与预处理 (Cutoff & Split)      │
        │ - 保护系统指令、动态状态栏、当前 User 原始目标 │
        └──────────────────────┬───────────────────────┘
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
 待压缩历史 (messages_to_summarize)      保留区 (preserved_messages)
            │                                     │
            ▼                                     ▼
┌───────────────────────────────┐     ┌───────────────────────────────────┐
│ 阶段二：任务感知稠密摘要       │     │ 阶段三：近期工具输出脱敏瘦身      │
│ - 注入 Current Goal / Query   │     │ - 保留最近 1~2 轮完整输出         │
│ - 提炼关键数据、阻碍与进展     │     │ - 较早轮次的 ToolMessage 内容截断 │
│ - 生成结构化 summary_text     │     │   替换为 [Output pruned: N chars] │
└───────────┬───────────────────┘     └─────────────────┬─────────────────┘
            │                                           │
            └─────────────────┬─────────────────────────┘
                              ▼
        New Agent State (< 18K Tokens, 清爽高密度上下文)
```

### 4.2 核心改动点详述

#### 改动 1：历史工具输出自动瘦身（Tool Message Pruning in Preserved Tail）
* **原理**：即使在保留的消息窗口（`preserved_messages`）内，模型通常只需要**最近 1~2 步的工具原始报错/细节**用于下一步纠错，更早的工具输出（如第 5 步 read 出来的 800 字文件）其关键信息只要已被吸收，原文留存即为负债。
* **机制**：
  - 在 `preserved_messages` 中，对非末轮的 `ToolMessage`：
    - 若长度超过 `max_tool_chars: 400`，将其替换为带元数据的占位符：
      ```markdown
      [Tool Output Snippet (read_file): First 150 chars: "..." ... Full output was 2,400 chars. Status: SUCCESS]
      ```
  - 最新一轮的工具调用保持 100% 原始字节不变，保证即时推理与代码语法纠错能力。

#### 改动 2：配置项升级与 Token 预算驱动保留（Config Schema Extension）
在 `SummarizationConfig`（`deerflow/config/summarization_config.py`）中扩充安全策略：
```yaml
summarization:
  enabled: true
  trigger:
    - type: tokens
      value: 32000
  keep:
    # 策略升级：改为复合保留策略，优先保证保留区 Token 预算不超过 12000
    type: tokens
    value: 12000
  prune_preserved_tools:
    enabled: true
    max_tool_chars: 500       # 保留区中非末轮工具输出的最大字符数
    keep_recent_tool_turns: 2 # 保持完整原始输出的最近轮数
  task_aware_summary:
    enabled: true             # 是否将当前用户目标/Query 显式注入摘要 Prompt
```

#### 改动 3：任务感知摘要提示词（Task-Aware Summary Prompt）
替换原默认的通用摘要提示词，采用书本实验 2-10 策略 4 的模板：
```markdown
You are optimizing the context for an AI Agent working on a specific task.
Target User Goal:
{user_goal}

Previous Progress Summary:
{previous_summary}

New Interactions to Consolidate:
{messages}

Your goal: Produce a highly dense, factual summary.
1. Retain all key facts, numbers, file paths, and conclusions relevant to the Target User Goal.
2. Discard intermediate verbose command outputs, repetitive logs, and failed retries that have already been resolved.
3. State the current pending task status clearly.
```

---

## 5. 验证与评测策略（Verification Loop）

按照 Karpathy 规则《Goal-Driven Execution》，构建两级验证体系：

### 5.1 第一级：离线重放评测（Offline Replay Benchmark，零开销、1 秒反馈）
* **脚本路径**：`experiments/replay_compaction.py`
* **输入数据**：加载现存的真实轨迹 `experiments/raw/t2.ndjson` 与 `t3.ndjson`。
* **执行过程**：
  1. 离线模拟 Agent 逐步累加状态消息；
  2. 当消息达到 32K 触发点时，调用修改前后的切分与保留逻辑；
  3. 计算并对比：
     - 保留区原始 Token 计数；
     - 工具瘦身减少的 Token 量；
     - 压缩后总体积。
* **通过标准**：保留区 Token 从原 `~31,400` 显著压降至 `<= 15,000`。

### 5.2 第二级：在线长任务基准实测（Live Headless Benchmark）
* **执行命令**：`python experiments/run_baseline.py t2`
* **对比指标矩阵**：

| 指标 | 原始基线 (BASELINE.md) | 目标预期 (Target) | 验收判定 |
| :--- | :--- | :--- | :--- |
| **触发摘要次数** | 24 次 | **<= 3 次** | 消除贴顶震荡 |
| **单次压缩后 Token 峰值** | 35,424 ➔ 31,420 | **35,000 ➔ < 18,000** | 降幅 > 45% |
| **全任务输入 Token 合计** | 353,438 tokens | **< 230,000 tokens** | 节省 > 35% |
| **墙钟执行耗时** | 495.1 s | **< 360 s** | 减少无谓调用 |
| **任务结果正确性** | findings.md 精确命中 7 个 refund | findings.md 精确命中 7 个 refund | 100% 一致 |

---

## 6. 分阶段落地实施路线图（Milestones）

```
Step 1: 编写离线重放基准脚本 (experiments/replay_compaction.py)
   ↳ 验证：读取 raw/t2.ndjson，1 秒内精准复现 31.4K 保留区数据。

Step 2: 在 summarization_middleware 中实现工具输出瘦身与 Token 预算裁剪
   ↳ 验证：离线脚本显示保留区 Token 降至 14,000 以内。

Step 3: 引入任务感知摘要提示词与配置支持
   ↳ 验证：离线测试确认 summary 文本保留了销售/退款等关键结论。

Step 4: 运行在线长任务基准测试 (run_baseline.py t2, t1, t3)
   ↳ 验证：产出 report/findings 无回归，记录 OPTIMIZATION_REPORT.md。
```
