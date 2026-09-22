# OpenJev 相关仓库代码审查

审查日期：2026-09-22。源码以各仓库浅克隆的当前默认分支为准；本报告记录代码位置和能否在本项目中复用的设计，数字指标仍以原仓库或其 benchmark 的口径为准。

## 审查标准

只有同时满足“代码可定位、行为有测试或可复现产物、与当前 PyTorch 框架边界清楚”中的至少两项，才进入本项目的集成候选。没有许可证、依赖专有权重、或只在 README 中声明的能力不会直接复制。

## 20 个仓库的代码级结论

| # | 仓库 | 代码位置 | 主要特长 | 集成决定 |
|---:|---|---|---|---|
| 1 | [jaredpalmer/kev](https://github.com/jaredpalmer/kev) | `kev/model.py`, `kev/train.py`, `evals/*/manifest.json`, `PLAN.md` | LoRA/pointer readout、冻结数据集、校准和研究日志 | 采用 manifest、冻结 split、温度校准和实验收据 |
| 2 | [TheoLeeCJ/SemIf](https://github.com/TheoLeeCJ/SemIf) | `semif/`, `benchmarks/`, `results/raw/`, `docs/REPRODUCE.md` | 直接读取选项 logits、packed/shared 与 separate 对照、原始结果 | 采用统一 readout 协议、共享状态与干扰审计接口 |
| 3 | [featherless-ai/simple-jev](https://github.com/featherless-ai/simple-jev) | `common/prompt_builder.py`, `common/response_scoring.py`, `hf-server/` | 版本化 prompt、请求校验、服务响应构造 | 采用 prompt/schema 版本和置信度计算；不复制服务实现 |
| 4 | [daseinlabs/open-jev](https://github.com/daseinlabs/open-jev) | `openjev/features.py`, `openjev/head.py`, `openjev/train.py` | 冻结特征缓存、跨注意力 head、shuffle-context 控制 | 采用特征缓存元数据和上下文打乱审计 |
| 5 | [wfzyx/von](https://github.com/wfzyx/von) | `src/von/engine.py`, `src/von/patterns.py`, `training/` | OptionMarker、后端抽象、训练脚本、协议兼容测试 | 采用候选轴约束、后端无关的预测接口和能力矩阵 |
| 6 | [Heman10x-NGU/openJev-verdict-2.0](https://github.com/Heman10x-NGU/openJev-verdict-2.0) | `verdict2/losses.py`, `core/calibration.py`, `artifacts/*`, `tests/` | 双通道校准、选项置换 KL、完整 artifacts | 采用 permutation audit 和校准收据；许可证需人工复核，拒绝复制权重 |
| 7 | [ikermoel/open-alternative-jev](https://github.com/ikermoel/open-alternative-jev) | `so1/prompting.py`, `so1/calibration.py`, `benchmarks/` | ChatML packed prompt、temperature scaling、HF/vLLM 对照 | 采用 prompt builder 的结构与 cross-fit 校准概念 |
| 8 | [razorback16/openjev](https://github.com/razorback16/openjev) | `openjev/engine.py`, `openjev/api.py`, `openjev/mlx_backend.py` | DiffusionGemma typed readout、MLX/vLLM 服务 | 记录为服务参考；当前仓库不引入 DiffusionGemma 依赖 |
| 9 | [ekzhang/openjev-sglang](https://github.com/ekzhang/openjev-sglang) | `src/openjev/scoring.py`, `runtime.py`, `tests/` | prefix cache、分支请求、限流/错误语义和 smoke report | 采用 usage/latency 记录格式和分支级验证思想 |
| 10 | [fstandhartinger/jevbench](https://github.com/fstandhartinger/jevbench) | `jevbench/metrics.py`, `composite_v12.py`, `datasets/manifest.json`, `tests/` | 冻结/held-out 数据、准确率/校准/速度/成本分轴 | 采用分轴 benchmark 报告，不复制其综合排名为单一指标 |
| 11 | [receptron/laya](https://github.com/receptron/laya) | `src/sequence.ts`, `src/types.ts`, `export/export_onnx.py` | ONNX/browser 序列布局、纯模型无关渲染层 | 采用导出契约和序列校验清单，作为可选模块 |
| 12 | [nico-martin/open-jev](https://github.com/nico-martin/open-jev) | `src/questions.ts`, `src/encoding.ts`, `src/utils/math.ts` | typed question builder、选项边界校验、浏览器数学工具 | 采用问题限制和唯一选项校验 |
| 13 | [kyegomez/open-jev](https://github.com/kyegomez/open-jev) | `open_jev/main.py`, `forward.py` | 共享 state encoder、typed heads、信息流隔离的教学实现 | 采用架构文档和 state/question 信息流约束，不采用随机权重实现 |
| 14 | [intikhab49/open-jev-typed-decision-engine](https://github.com/intikhab49/open-jev-typed-decision-engine) | `02_train.py`, `03_calibrate.py`, `04_eval.py`, `06_export_onnx.py` | 校准、ONNX 导出、轻量训练闭环 | 采用校准/导出阶段分离的工作流 |
| 15 | [kshetrajna12/reflex](https://github.com/kshetrajna12/reflex) | `tests/test_pack.py`, `tests/test_ensemble.py`, `docs/results/order-averaging.md` | packed 推理、ensemble、选项顺序研究 | 采用顺序敏感性与 ensemble 审计记录 |
| 16 | [deepanwadhwa/OpenDecision](https://github.com/deepanwadhwa/OpenDecision) | `core/`, `benchmarks/`, `tests/test_evidence_backend.py` | evidence/rules、文档决策和可解释结果 | 采用 evidence 字段的可选扩展，不耦合规则引擎 |
| 17 | [IamBusy/OpenJev-Vision](https://github.com/IamBusy/OpenJev-Vision) | `src/openjev_vision/branch_cache.py`, `tests/test_branch_cache.py`, `docs/REPRODUCING.md` | 图像共享编码、branch cache、重放式实验 | 采用通用 branch-cache 抽象和重放收据，图像编码保持可选 |
| 18 | [SAGAR-TAMANG/sarvam-jev](https://github.com/SAGAR-TAMANG/sarvam-jev) | `src/sarvam_jev/shared.py`, `direct.py`, `tests/test_prompt_parity.py` | shared state、prompt parity、浏览器/服务一致性 | 采用 prompt parity 检查和 shared input 标识 |
| 19 | [mithalouni/system-one-open](https://github.com/mithalouni/system-one-open) | `s1/engine.py`, `train.py`, `evaluate.py`, `results/*.json` | attention LoRA、GPU 训练/评测/报告闭环 | 采用结果报告字段和训练/评测分离，不复制模型权重 |
| 20 | [logicrw/awesome-jev-projects](https://github.com/logicrw/awesome-jev-projects) | `radar/receipts/`, `radar/reviews/`, `VERIFICATION.md` | 来源审查、发现收据、排除清单 | 采用 research receipt 和排除原因记录 |

## 本地集成顺序

1. `provenance.py`：数据、源码、环境和外部仓库的 hash 收据。
2. `calibration.py`：温度缩放、cross-fit、NLL/ECE/Brier。
3. `audits.py`：选项置换、上下文打乱、共享状态/独立问题对照。
4. `benchmark.py`：按类型和维度输出统一报告，并保存原始预测。
5. `readout.py`：可选 Hugging Face next-token readout 接口，不影响当前轻量模型。
6. `export.py`：可选 TorchScript/ONNX 导出和 reload parity。
7. `evidence.py` 与 `replay.py`：可选 evidence 字段和 branch-cache 重放收据。

每个模块都必须先有失败测试，再实现，再运行全量测试；外部项目只作为设计来源，代码保持本仓库自己的实现和许可证边界。

