# open-jev

[English README](README.md)

> **面向 AI Agent 的类型化、可校准决策层。**

open-jev 是一个开源 PyTorch 决策层，适用于 Agent 路由、工具选择、RAG 检查和评测。它把状态、类型化问题和动态候选集合转换为概率、证据和可重放的决策结果。

[![CI](https://github.com/Eggwardhan/open-jev/actions/workflows/ci.yml/badge.svg)](https://github.com/Eggwardhan/open-jev/actions/workflows/ci.yml)
![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B-ee4c2c)
![License](https://img.shields.io/badge/license-Apache--2.0-green)

## 为什么是 open-jev？

- **类型化决策**：支持动态候选集合
- **概率校准**：输出可测量的概率，而不是没有依据的标签
- **可审计运行**：记录 provenance、证据和 replay key
- **组件可替换**：Tokenizer、Encoder 和评分头可以独立替换
- **可复现实验**：分别报告准确率、校准、延迟和成本

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
```

运行仓库自带的确定性训练实验：

```bash
open-jev train-synthetic --output artifacts/my-run \\
  --groups 600 --epochs 40 --batch-size 32 --seed 7 --device auto
```

使用 `--device cuda` 强制使用 CUDA，使用 `--device cpu` 在本地运行。训练器会生成分组后的 JSONL 数据切分、逐 epoch 日志、checkpoint、留出集预测、指标、哈希和硬件信息。

## 十行决策契约

```python
from open_jev import Example, Question
from open_jev.model import DynamicDecisionModel
from open_jev.tokenization import WhitespaceTokenizer

question = Question(
    type="choice",
    instructions="Choose the safest tool",
    criteria={
        "search": "read-only web lookup",
        "shell": "local command execution",
        "human": "ask the user first",
    },
)
example = Example(
    id="demo-1",
    state={"request": "Inspect a public documentation page"},
    question=question,
    label="search",
)
tokenizer = WhitespaceTokenizer.fit([example])
model = DynamicDecisionModel(len(tokenizer), hidden_size=64)
```

同一套类型化契约支持 `Choice`、`Noul` 和 `Score` 问题。JSONL 示例会保留
`id`、`group`、`source` 和原始 `state`，方便审计；分组切分可以避免相同状态泄漏到不同评测分区。

## 适用场景

- Agent 路由和 Skill 选择
- 工具及动作的执行前检查
- RAG 相关性和证据检查
- 类型化评测信号
- 在确定性策略执行前进行风险分流

## 架构

```text
JSONL -> schema -> grouped split -> tokenizer -> encoder -> dynamic scorer
                                             |-> calibration / audits
                                             |-> benchmark / replay / evidence
                                             |-> checkpoint -> FastAPI
```

默认基线使用确定性的空白分词器、平均池化 Embedding 和置换等变的候选评分器。仓库还包含 provenance 收据、cross-fit 温度校准、选项顺序审计、benchmark 报告、直接 logits 读取契约、TorchScript 导出、证据引用和 replay key。

## 可复现实验结果

仓库内的 [synthetic-v2 实验](examples/synthetic-v2/README.md) 用于验证流水线，不代表真实业务表现，也不代表与 TypeSafe Jev 等价。在合成规则学习任务中，180 条留出数据经过 40 个 epoch 后达到 100% 准确率。使用真实业务数据前，请准备有代表性的标注集，并分别报告准确率、校准、延迟和成本。

参见[中文训练报告](docs/reports/2026-09-21-h800-synthetic.md)和 [20 个相关仓库的集成审计](docs/research/2026-09-22-openjev-20-repo-audit.md)。

## 项目边界

open-jev 是独立的开源实现，不复现 TypeSafe Jev 的私有架构、权重或训练数据。它是一个模块化研究与工程基线，不能单独充当安全边界。涉及重要操作时，仍然需要确定性权限、沙箱、人工确认和领域校验。

## 开发

```bash
ruff check src tests scripts
pytest -q
```

如果新增决策原语或评测路径，请同时提交针对性测试和可复现实例。发布结果前，请阅读 [贡献指南](CONTRIBUTING.md) 和 [引用说明](CITATION.cff)。集成模块的来源和改写记录见 [20 个仓库的审计报告](docs/research/2026-09-22-openjev-20-repo-audit.md)。

本项目使用 Apache-2.0 许可证。
