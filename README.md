# 中文 NER（BiLSTM-CRF）——用于敏感数据脱敏

基于 **人民日报标注语料**（People's Daily NER）训练的字符级 **BiLSTM-CRF** 命名实体识别模型，
识别人名（PER）、机构名（ORG）、地名（LOC），并提供开箱即用的**文本脱敏工具**。

- 数据集：<https://github.com/OYE93/Chinese-NLP-Corpus/tree/master/NER/People's%20Daily>
- 标签体系：`BIO`（`O` / `B,I-PER` / `B,I-ORG` / `B,I-LOC`）
- 字向量：**随机初始化**（不依赖外部预训练向量）
- 训练设备：CUDA GPU（RTX 3050 4GB 可训）

---

## 一、环境准备

需要 Python 3.9+。先装 CUDA 版 PyTorch（CPU 版把 index-url 去掉即可）：

```bash
# CUDA 12.6（示例，按本机驱动选择 cu118 / cu121 / cu126 / cu128）
pip install torch --index-url https://download.pytorch.org/whl/cu126

# 其余依赖
pip install -r requirements.txt
```

验证 GPU：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

> 本机实测环境：Python 3.9.7、torch 2.8.0+cu126、RTX 3050 Laptop 4GB（sm_86）。

---

## 二、目录结构

```
NER_BiLSTM_CRF/
├── data/
│   ├── raw/                 # 原始语料 example.train / example.dev / example.test
│   ├── processed/           # 清洗后语料、词表、数据报告、清洗报告
│   └── samples/             # 脱敏工具示例输入（txt / csv）
├── src/
│   ├── config.py            # 全局配置（路径、清洗参数、模型与训练超参）
│   ├── data_inspect.py      # 数据质量检查（生成 data_report.txt）
│   ├── data_utils.py        # 读取/清洗/长句切分/词表/编码/动态批
│   ├── model.py             # 手写 CRF + BiLSTM-CRF
│   ├── metrics.py           # 实体级 P/R/F1
│   ├── engine.py            # 推理、评测、检查点读写
│   ├── train.py             # 训练（动态批、早停、保存最优）
│   ├── evaluate.py          # 测试集评测
│   ├── predictor.py         # 推理封装（长文本切块）
│   └── anonymize.py         # 脱敏工具（CLI）
├── tests/
│   ├── run_all.py               # 一键运行全部测试
│   ├── test_crf.py              # CRF 与暴力枚举对拍
│   ├── test_data_pipeline.py    # 数据管线、指标、推理对齐回归测试
│   ├── test_anonymize.py        # 脱敏工具测试
│   ├── debug_overfit.py         # 小样本过拟合诊断
│   └── debug_diagnose.py        # 发射分数/对齐诊断
├── models/                  # 训练产出
├── logs/                    # 训练日志
└── requirements.txt
```

---

## 三、快速开始

```bash
cd src

# 1) 数据质量检查（生成 data/processed/data_report.txt）
python data_inspect.py --report

# 2) 数据清洗 + 构建词表（生成 data/processed/train|dev|test.txt、vocab.json）
python data_utils.py

# 3) 训练（约 14s/轮，自动早停并保存最优模型到 models/bilstm_crf_best.pt）
python train.py

# 4) 测试集评测
python evaluate.py --show-errors 10

# 5) 脱敏
python anonymize.py --text "张三在阿里巴巴的杭州总部工作。"
```

快速试跑（小样本冒烟测试）：

```bash
python train.py --limit 2000 --epochs 3
```

### 训练参数

```bash
python train.py --epochs 60 --batch-tokens 12000 --lr 1e-3 --patience 8 --no-constraints
```

| 参数 | 默认 | 说明 |
|:--|:--|:--|
| `--epochs` | 60 | 最大轮数 |
| `--batch-tokens` | 12000 | 动态批的 token 预算（4GB 显存安全值） |
| `--lr` | 1e-3 | Adam 初始学习率 |
| `--patience` | 8 | dev F1 连续多少轮不升则早停（前 8 轮受保护） |
| `--limit` | 0 | 只用前 N 条训练句（冒烟测试） |
| `--no-constraints` | 关 | 关闭 CRF 转移约束做消融对比 |

---

## 四、脱敏工具用法

### 基本用法

```bash
cd src

# 默认：替换为 [人名] / [机构名] / [地名]
python anonymize.py --text "张三在阿里巴巴的杭州总部工作。"

# 只脱敏人名和机构名，用标签形式 [PER]
python anonymize.py --text "..." --types PER,ORG --style tag

# 掩码形式，固定 3 个星号
python anonymize.py --text "..." --style mask --mask-len 3

# 自定义模板 + 同一实体统一编号
python anonymize.py --text "..." --style custom --template "<{name}#{id}>" --with-id --consistent-ids

# 只看识别结果，不改写文本
python anonymize.py --text "..." --entities-only
python anonymize.py --text "..." --json          # 输出实体 JSON
```

### 四种脱敏样式

| `--style` | 效果 |
|:--|:--|
| `placeholder`（默认） | `张三` → `[人名]`，`阿里巴巴` → `[机构名]` |
| `tag` | `张三` → `[PER]`，`阿里巴巴` → `[ORG]` |
| `mask` | `张三` → `**`（`--mask-len 0` 时按原文长度，`--mask-len N` 固定 N 位） |
| `custom` | 按 `--template` 渲染，支持 `{type}` `{name}` `{id}` |

### 批量文件处理

```bash
# 处理整个目录，输出到 dir/masked/
python anonymize.py --input ../data/samples --output ../data/masked

# 单文件
python anonymize.py --input ../data/samples/sample.txt --output out.txt
python anonymize.py --input ../data/samples/sample.csv --output out.csv
```

支持 `.txt`（整篇）、`.csv`（逐单元格，保持表格结构）、`.jsonl` / `.json`（所有字符串字段）。

### 关于拉丁字母误报（已默认处理）

模型可能在邮箱、网址、编号的拉丁字母片段上产生误报，例如把 `zhangsan@example.com`
识别成 `zh[机构名]m`，导致原文被改坏。统计显示语料中**不含汉字**的实体仅 31/45518（0.07%），
因此工具**默认丢弃不含汉字的实体**；含有汉字的实体（如 `TCL集团`）不受影响。
如需保留原始行为，加 `--keep-ascii-entities`。

### 长文本

`predictor.py` 会先把长文本按句末标点切成不超过训练最大句长（200 字符）的块，
分批推理后再把实体偏移还原到原文坐标，因此可以直接处理整篇文档。

---

## 五、训练结果

设置：`EMBED_DIM=128`、`HIDDEN_DIM=256`、1 层 BiLSTM、dropout 0.5、Adam lr=1e-3、
动态批 12000 token、最多 60 轮（第 58 轮最优），单轮约 13.5s，全程约 14 分钟。

**验证集（epoch 58）**

| 类型 | 精确率 | 召回率 | F1 | TP | FP | FN | 支持数 |
|:--|--:|--:|--:|--:|--:|--:|--:|
| PER(人名) | 90.73% | 87.44% | **89.06%** | 773 | 79 | 111 | 884 |
| ORG(机构名) | 82.64% | 80.28% | **81.44%** | 790 | 166 | 194 | 984 |
| LOC(地名) | 88.96% | 88.01% | **88.48%** | 1717 | 213 | 234 | 1951 |
| **总体（micro）** | **87.75%** | **85.89%** | **86.81%** | 3280 | 458 | 539 | 3819 |
| 宏平均 F1 | — | — | 86.33% | — | — | — | — |

**测试集**

| 类型 | 精确率 | 召回率 | F1 | TP | FP | FN | 支持数 |
|:--|--:|--:|--:|--:|--:|--:|--:|
| PER(人名) | 91.97% | 89.11% | **90.52%** | 1661 | 145 | 203 | 1864 |
| ORG(机构名) | 82.33% | 79.95% | **81.12%** | 1747 | 375 | 438 | 2185 |
| LOC(地名) | 87.10% | 87.32% | **87.21%** | 3194 | 473 | 464 | 3658 |
| **总体（micro）** | **86.93%** | **85.66%** | **86.29%** | 6602 | 993 | 1105 | 7707 |
| 宏平均 F1 | — | — | 86.28% | — | — | — | — |

结论：**ORG 明显最难**（F1 81.12%），因为机构名平均长度 5.06 字、最长 33 字，
且常与其他机构名或地名嵌套（如 `中国银行上海分行与华为技术有限公司` 易被合并成一个实体）；
PER（三字人名居多）和 LOC 表现较好。典型错误是**实体边界**问题，例如
`玉峰山` 被切成 `云` + `玉峰`、`宝顶大佛湾` 只识别出 `大佛湾`。

参考：同数据集的字符级 BiLSTM-CRF 论文结果（无预训练向量）大致在 85–91% F1，
本实现处于合理区间。

---

## 六、模型结构

```
输入字符索引 (B, T)
  └─ Embedding(3786, 128)            随机初始化，padding_idx=0
      └─ Dropout(0.5)
          └─ BiLSTM(128 → 256×2, 1 层, 使用 pack_padded_sequence 跳过 padding)
              └─ Dropout(0.5)
                  └─ Linear(512 → 7)  发射分数 (B, T, 7)
                      └─ CRF          转移约束 + log-sum-exp 前向 + Viterbi 解码
```

参数量约 **1.28M**（字符级，无预训练向量）。

CRF 为**手写实现**（不依赖 `torchcrf`），并已用暴力枚举对拍验证正确性：

- 前向归一化 `log Z` 与穷举所有合法路径的结果一致（误差 < 1e-3）
- Viterbi 解码分数与穷举最优分数一致
- 转移约束确实屏蔽了所有非法 BIO 转移

提升方向（当前未启用）：接入预训练字/词向量（如腾讯词向量、`Chinese-Word-Vectors`）
通常可再提升 3–4 个点；`config.py` 已预留 `EMBED_DIM` 等配置位置。

---

## 七、注意事项与已知限制

1. **模型规模小、无预训练向量**：受"纯本地、不下载"约束，F1 低于带预训练向量的方案。
   若允许联网，接入预训练字向量是性价比最高的提升手段。
2. **语料领域限制**：人民日报新闻语料，对口语、网络用语、专业领域文本（医疗/金融术语）
   泛化有限；纯 O 句子占 39% 也说明新闻中有大量无实体句。
3. **单字实体歧义**："日""美""京"等单字地名在非地名语境易误报。
4. **脱敏不是匿名化**：模型输出替换的是**实体文本**，不是稳定的假名映射；
   如需跨文档一致的假名（同一人始终映射到同一代号），可在 `Anonymizer` 上扩展
   `--consistent-ids` 的思路做全局映射表。
5. **数字/身份证/手机号**不在 PER/ORG/LOC 体系内，本模型不识别，
   如需脱敏这类结构化敏感信息应另加正则规则。
6. 训练时 `MAX_SEQ_LEN=200`，超过部分靠推理阶段切块处理；若切成极短碎片可能损失上下文。
