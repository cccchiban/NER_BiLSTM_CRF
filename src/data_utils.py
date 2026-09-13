# -*- coding: utf-8 -*-
"""数据处理：读取 / 清洗 / 切分 / 词表 / 编码 / 动态批。

语料格式（People's Daily，BIO）:
    海 O
    钓 O
    比 O
    赛 O
    ...
    <空行表示句子结束>
"""
import json
import os
import random
from collections import Counter

import torch

import config as C


# ----------------------------------------------------------------------------
# 读取
# ----------------------------------------------------------------------------
def read_conll(path):
    """读取 `字 标签` 格式语料，返回 [[(char, tag), ...], ...]。"""
    sentences, cur = [], []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n").rstrip("\r")
            if not line.strip():
                if cur:
                    sentences.append(cur)
                    cur = []
                continue
            parts = line.split()
            if len(parts) == 2:
                cur.append((parts[0], parts[1]))
    if cur:
        sentences.append(cur)
    return sentences


def write_conll(path, sentences):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sent in sentences:
            for char, tag in sent:
                f.write(f"{char} {tag}\n")
            f.write("\n")


def sentence_text(sent):
    return "".join(c for c, _ in sent)


# ----------------------------------------------------------------------------
# 清洗
# ----------------------------------------------------------------------------
def is_legal_bio(tag, prev_tag):
    """判断 (prev_tag -> tag) 是否为合法 BIO 转移。"""
    if tag == "O":
        return True
    if tag.startswith("B-"):
        return True
    if tag.startswith("I-"):
        etype = tag[2:]
        if prev_tag == "O":
            return False
        return prev_tag[2:] == etype
    return False


def can_split_at(sent, k):
    """判断能否在位置 k 把句子切成 [0:k] / [k:] 而不切断实体。

    要求 k-1 处实体已闭合（是 I-x 或 O），且 k 处不会以 I-x 开头。
    """
    if k <= 0 or k >= len(sent):
        return False
    prev_tag, next_tag = sent[k - 1][1], sent[k][1]
    prev_ok = prev_tag == "O" or prev_tag.startswith("I-")
    next_ok = next_tag == "O" or next_tag.startswith("B-")
    return prev_ok and next_ok


def split_long_sentence(sent, max_len):
    """把超长句按标点切成多个合法子句，保证不切断实体。

    优先在 max_len 之前最近的硬标点处切分；找不到再用软标点；
    仍找不到则在保证不切断实体的前提下从 max_len 往前找位置兜底。
    """
    if len(sent) <= max_len:
        return [sent]

    # 1) 在 [max_len//2, max_len] 窗口内找硬标点（标点本身留在前一段）
    for chars in (C.HARD_SPLIT_CHARS, C.SOFT_SPLIT_CHARS):
        for k in range(max_len, max(max_len // 2, 1) - 1, -1):
            if k < len(sent) and sent[k - 1][0] in chars and can_split_at(sent, k):
                return split_long_sentence(sent[:k], max_len) + \
                       split_long_sentence(sent[k:], max_len)

    # 2) 兜底：任意合法位置
    for k in range(max_len, 1, -1):
        if can_split_at(sent, k):
            return split_long_sentence(sent[:k], max_len) + \
                   split_long_sentence(sent[k:], max_len)

    # 3) 极端情况（整个 max_len 窗口都在实体内部）：不切，原样返回
    return [sent]


def repair_bio(sent):
    """把非法 BIO 序列修成合法序列，返回 (新句子, 修复次数)。

    规则：出现 I-x 却缺少同类型前驱时，把该 I-x 降级为 B-x。
    """
    fixed, n_fixed = [], 0
    prev_tag = "O"
    for char, tag in sent:
        if tag.startswith("I-") and not is_legal_bio(tag, prev_tag):
            tag = "B-" + tag[2:]
            n_fixed += 1
        fixed.append((char, tag))
        prev_tag = tag
    return fixed, n_fixed


def drop_single_char_entities(sent):
    """把长度为 1 的实体降级为 O。"""
    out = []
    n = len(sent)
    for i, (char, tag) in enumerate(sent):
        if tag.startswith("B-"):
            nxt = sent[i + 1][1] if i + 1 < n else "O"
            if not (nxt.startswith("I-") and nxt[2:] == tag[2:]):
                out.append((char, "O"))
                continue
        out.append((char, tag))
    return out


def clean_split(name, sentences, stats, leakage_keys=None):
    """对单个 split 执行完整清洗流程，返回清洗后的句子列表。"""
    out = []
    seen = set()
    for sent in sentences:
        # 1) 修复非法 BIO
        if C.REPAIR_ILLEGAL_BIO:
            sent, n = repair_bio(sent)
            stats["bio_repaired"] += n

        # 2) 过滤单字实体（可选）
        if C.FILTER_SINGLE_CHAR_ENTITIES:
            sent = drop_single_char_entities(sent)

        # 3) 长句安全切分
        for sub in split_long_sentence(sent, C.MAX_SEQ_LEN):
            if len(sub) > C.MAX_SEQ_LEN:
                stats["unsplittable_long_sent"] += 1

            # 4) 丢弃无实体句（可选）
            if not C.KEEP_PURE_O_SENTENCES and all(t == "O" for _, t in sub):
                stats["dropped_pure_o"] += 1
                continue

            # 5) 空句过滤
            if C.DROP_EMPTY_SENTENCES and not sub:
                stats["dropped_empty"] += 1
                continue

            key = sentence_text(sub)

            # 6) 去重
            if C.DEDUPLICATE and key in seen:
                stats["dropped_duplicate"] += 1
                continue

            # 7) 去除与 train 泄漏的句子
            if C.DROP_TEST_LEAKAGE and leakage_keys is not None and key in leakage_keys:
                stats["dropped_leakage"] += 1
                continue

            seen.add(key)
            out.append(sub)

    stats["kept_sentences"] += len(out)
    return out


def clean_all():
    """清洗 train/dev/test 三个文件并落盘，返回 (datasets, vocab_counter, report)。"""
    stats = Counter()
    raw = {}
    for name, path in C.RAW_FILES.items():
        raw[name] = read_conll(path)
        stats[f"raw_sentences_{name}"] = len(raw[name])
        stats[f"raw_chars_{name}"] = sum(len(s) for s in raw[name])

    train = clean_split("train", raw["train"], stats)
    train_keys = {sentence_text(s) for s in train} if C.DROP_TEST_LEAKAGE else None
    dev = clean_split("dev", raw["dev"], stats, train_keys)
    test = clean_split("test", raw["test"], stats, train_keys)

    datasets = {"train": train, "dev": dev, "test": test}
    char_counter = Counter(c for s in train for c, _ in s)
    return datasets, char_counter, stats


# ----------------------------------------------------------------------------
# 词表
# ----------------------------------------------------------------------------
def build_vocab(char_counter):
    """构建字表：<pad>=0, <unk>=1，其余按频次降序。"""
    chars = [c for c, n in char_counter.items() if n >= C.MIN_CHAR_FREQ and c not in (C.PAD_TOKEN, C.UNK_TOKEN)]
    chars.sort(key=lambda c: (-char_counter[c], c))
    chars = chars[: C.MAX_VOCAB_SIZE - 2]
    char2idx = {C.PAD_TOKEN: C.PAD_TOKEN_ID, C.UNK_TOKEN: C.UNK_TOKEN_ID}
    for c in chars:
        char2idx[c] = len(char2idx)
    return char2idx


def save_vocab(char2idx, path=None):
    path = path or C.VOCAB_FILE
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"char2idx": char2idx, "tags": C.TAGS}, f, ensure_ascii=False, indent=1)


def load_vocab(path=None):
    path = path or C.VOCAB_FILE
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    return obj["char2idx"], obj["tags"]


# ----------------------------------------------------------------------------
# 编码
# ----------------------------------------------------------------------------
def encode_sentence(sent, char2idx):
    ids, tags = [], []
    for char, tag in sent:
        ids.append(char2idx.get(char, C.UNK_TOKEN_ID))
        tags.append(C.TAG2IDX.get(tag, C.O_TAG_IDX))
    return ids, tags


def encode_dataset(sentences, char2idx):
    return [encode_sentence(s, char2idx) for s in sentences]


# ----------------------------------------------------------------------------
# 动态批采样
# ----------------------------------------------------------------------------
class TokenBucketSampler:
    """按长度分桶，令每批 token 总数不超过 budget，减少 padding 浪费。"""

    def __init__(self, lengths, budget=None, shuffle=True, seed=C.SEED):
        self.lengths = list(lengths)
        self.budget = budget or C.BATCH_TOKEN_BUDGET
        self.shuffle = shuffle
        self.rng = random.Random(seed)
        self.batches = self._build()

    def _build(self):
        order = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i])
        batches, cur, cur_max = [], [], 0
        for i in order:
            length = self.lengths[i]
            # 加入后该批所需 padding 后的总格子数
            if cur and max(cur_max, length) * (len(cur) + 1) > self.budget:
                batches.append(cur)
                cur, cur_max = [], 0
            cur.append(i)
            cur_max = max(cur_max, length)
        if cur:
            batches.append(cur)
        return batches

    def __iter__(self):
        order = list(range(len(self.batches)))
        if self.shuffle:
            self.rng.shuffle(order)
        for i in order:
            batch = self.batches[i]
            if self.shuffle:
                batch = batch[:]
                self.rng.shuffle(batch)
            yield batch

    def __len__(self):
        return len(self.batches)


def pad_batch(batch):
    """batch: [(ids, tags), ...] -> (input_ids, tag_ids, mask)，全部为 LongTensor。"""
    lengths = [len(ids) for ids, _ in batch]
    max_len = max(lengths)
    input_ids = torch.full((len(batch), max_len), C.PAD_TOKEN_ID, dtype=torch.long)
    tag_ids = torch.full((len(batch), max_len), C.TAG_PAD_ID, dtype=torch.long)
    mask = torch.zeros((len(batch), max_len), dtype=torch.float)
    for i, (ids, tags) in enumerate(batch):
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        tag_ids[i, : len(ids)] = torch.tensor(tags, dtype=torch.long)
        mask[i, : len(ids)] = 1.0
    return input_ids, tag_ids, mask


def make_batches(encoded, budget=None, shuffle=False, seed=C.SEED):
    sampler = TokenBucketSampler([len(ids) for ids, _ in encoded], budget, shuffle, seed)
    for idx in sampler:
        yield [encoded[i] for i in idx]


# ----------------------------------------------------------------------------
# 报告
# ----------------------------------------------------------------------------
def format_clean_report(datasets, stats, char2idx):
    lines = []
    p = lines.append
    p("=" * 74)
    p("数据清洗报告")
    p("=" * 74)
    for name in ("train", "dev", "test"):
        sents = datasets[name]
        chars = sum(len(s) for s in sents)
        ents = Counter(t[2:] for s in sents for _c, t in s if t.startswith("B-"))
        maxlen = max((len(s) for s in sents), default=0)
        p(f"[{name}] 句子数={len(sents)}  字符数={chars}  最长句={maxlen}  "
          f"PER={ents['PER']}  ORG={ents['ORG']}  LOC={ents['LOC']}")
    p("-" * 74)
    p(f"词表大小: {len(char2idx)}  (min_freq={C.MIN_CHAR_FREQ}, "
      f"unk 命中率见训练日志)")
    p("-" * 74)
    p("清洗动作统计:")
    p(f"  非法 BIO 修复 (I-x -> B-x)     : {stats['bio_repaired']}")
    p(f"  丢弃纯 O 句子                  : {stats['dropped_pure_o']}")
    p(f"  丢弃重复句子                   : {stats['dropped_duplicate']}")
    p(f"  丢弃与 train 泄漏的句子        : {stats['dropped_leakage']}")
    p(f"  丢弃空句                       : {stats['dropped_empty']}")
    p(f"  无法切分的超长句(保持原样)     : {stats['unsplittable_long_sent']}")
    p("=" * 74)
    return "\n".join(lines) + "\n"


def prepare_data(verbose=True):
    """一键完成清洗 -> 词表 -> 编码，返回 (encoded_datasets, char2idx, char_counter)。"""
    datasets, char_counter, stats = clean_all()
    char2idx = build_vocab(char_counter)

    for name in ("train", "dev", "test"):
        write_conll(C.PROCESSED_FILES[name], datasets[name])
    save_vocab(char2idx)

    report = format_clean_report(datasets, stats, char2idx)
    os.makedirs(C.PROCESSED_DIR, exist_ok=True)
    with open(C.CLEAN_REPORT, "w", encoding="utf-8") as f:
        f.write(report)
    if verbose:
        print(report)

    encoded = {name: encode_dataset(datasets[name], char2idx) for name in datasets}
    # OOV 统计（<unk> 占比）
    total = sum(len(ids) for ids, _ in encoded["train"])
    unk = sum(sum(1 for i in ids if i == C.UNK_TOKEN_ID) for ids, _ in encoded["train"])
    if verbose:
        print(f"训练集 <unk> 占比: {unk}/{total} = {unk / max(total, 1) * 100:.3f}%\n")
        print(f"清洗后语料已写入: {C.PROCESSED_DIR}")
    return encoded, char2idx, char_counter


if __name__ == "__main__":
    C.setup_console()
    prepare_data()
