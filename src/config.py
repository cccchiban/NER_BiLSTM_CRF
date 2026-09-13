# -*- coding: utf-8 -*-
"""全局配置：路径、数据清洗参数、模型与训练超参。"""
import os
import random
import sys

import numpy as np
import torch

# ----------------------------------------------------------------------------
# 路径
# ----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")

RAW_FILES = {
    "train": os.path.join(RAW_DIR, "example.train"),
    "dev": os.path.join(RAW_DIR, "example.dev"),
    "test": os.path.join(RAW_DIR, "example.test"),
}
PROCESSED_FILES = {
    "train": os.path.join(PROCESSED_DIR, "train.txt"),
    "dev": os.path.join(PROCESSED_DIR, "dev.txt"),
    "test": os.path.join(PROCESSED_DIR, "test.txt"),
}
VOCAB_FILE = os.path.join(PROCESSED_DIR, "vocab.json")
CLEAN_REPORT = os.path.join(PROCESSED_DIR, "clean_report.txt")

# 最优模型保存路径（state_dict + 词表 + 配置一起打包）
CKPT_FILE = os.path.join(MODEL_DIR, "bilstm_crf_best.pt")
TRAIN_LOG = os.path.join(LOG_DIR, "train_log.csv")

# ----------------------------------------------------------------------------
# 标签体系：PER(人名) / ORG(机构名) / LOC(地名)，BIO 标注
# ----------------------------------------------------------------------------
ENTITY_TYPES = ["PER", "ORG", "LOC"]
ENTITY_NAMES = {"PER": "人名", "ORG": "机构名", "LOC": "地名"}
TAGS = ["O"] + [f"{p}-{t}" for t in ENTITY_TYPES for p in ("B", "I")]
TAG2IDX = {t: i for i, t in enumerate(TAGS)}
IDX2TAG = {i: t for t, i in TAG2IDX.items()}
NUM_TAGS = len(TAGS)
O_TAG_IDX = TAG2IDX["O"]

# ----------------------------------------------------------------------------
# 数据清洗参数（针对本语料实测问题设定，详见 data/processed/data_report.txt）
# ----------------------------------------------------------------------------
# 原始语料是整段合并的非真句子，最长 574 字符；超过该长度按标点安全切分
MAX_SEQ_LEN = 200
# 切分优先级：先硬标点，再软标点
HARD_SPLIT_CHARS = "。！？；!?;…"
SOFT_SPLIT_CHARS = "，、,:）)】》"
# 39% 的句子不含任何实体；保留它们可显著降低误报（对脱敏场景重要）
KEEP_PURE_O_SENTENCES = True
# 单字实体（如 LOC 的"日/美/京"）歧义大，默认保留（过滤会丢真实实体）
FILTER_SINGLE_CHAR_ENTITIES = False
# 非法 BIO（I-x 缺 B / 类型不一致）自动修复：I-x -> B-x
REPAIR_ILLEGAL_BIO = True
# 丢弃清洗后为空的句子
DROP_EMPTY_SENTENCES = True
# 同一 split 内完全重复的句子去重
DEDUPLICATE = True
# 丢弃 dev/test 中与 train 重复的句子（防指标虚高；本语料实测为 0）
DROP_TEST_LEAKAGE = True

# ----------------------------------------------------------------------------
# 词表
# ----------------------------------------------------------------------------
MIN_CHAR_FREQ = 2      # 训练集出现次数低于此值的字归为 <unk>
MAX_VOCAB_SIZE = 20000
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_TOKEN_ID = 0
UNK_TOKEN_ID = 1
# 词表里显式登记的元字符（标签 ID 用 -100 作为 padding 忽略值）
TAG_PAD_ID = -100

# ----------------------------------------------------------------------------
# 模型超参
# ----------------------------------------------------------------------------
EMBED_DIM = 128
HIDDEN_DIM = 256          # 单向隐藏维度，双向输出为 2 * HIDDEN_DIM
NUM_LAYERS = 1
DROPOUT = 0.5
# 用转移约束屏蔽非法 BIO 转移（本语料标注干净，约束安全且收敛更快）
USE_TRANSITION_CONSTRAINTS = True

# ----------------------------------------------------------------------------
# 训练超参
# ----------------------------------------------------------------------------
SEED = 42
BATCH_TOKEN_BUDGET = 12000   # 动态批：每批 token 总数上限（4GB 显存下的安全值）
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 60
PATIENCE = 8                 # 早停：dev F1 连续多少轮不提升则停止
LR_PATIENCE = 3              # ReduceLROnPlateau 耐心
LR_WARMUP_EPOCHS = 5         # 前若干轮不衰减学习率，避免模型还没起步就被降速
MIN_EPOCHS = 8               # 早停保护：至少训练这么多轮
CLIP_GRAD = 5.0
NUM_WORKERS = 0


def setup_console():
    """Windows 控制台默认 GBK，统一改成 UTF-8 以正常打印中文。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def set_seed(seed=None):
    seed = SEED if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # 输入形状基本固定，开启 cuDNN 自动调优可明显加速 LSTM
        torch.backends.cudnn.benchmark = True
    return device
