# -*- coding: utf-8 -*-
"""数据质量检查：统计 People's Daily NER 语料的标签分布与常见问题。

只报告 BIO 体系下**真正非法**的序列：
  - I-x 出现在 O 之后（缺少 B）
  - I-x 紧跟 B-y 且 x != y
  - I-x 紧跟 I-y 且 x != y
以及需要留意的标注噪声（合法但易歧义）：
  - B-x 紧跟 B-x（相邻同类实体未分隔）
  - 超长实体
  - 超长句子

用法:
    python src/data_inspect.py            # 打印到 stdout
    python src/data_inspect.py --report   # 同时写 data/processed/data_report.txt
"""
import os
import sys
from collections import Counter

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE, "data", "raw")
OUT_DIR = os.path.join(BASE, "data", "processed")
FILES = ["example.train", "example.dev", "example.test"]
TYPES = ["PER", "ORG", "LOC"]
VALID_TAGS = {"O"} | {f"{p}-{t}" for t in TYPES for p in ("B", "I")}


def read_sentences(path):
    """按空行切分句子，返回 [[(char, tag), ...], ...]"""
    sentences, cur = [], []
    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n").rstrip("\r")
            if not line.strip():
                if cur:
                    sentences.append(cur)
                    cur = []
                continue
            parts = line.split()
            if len(parts) == 2:
                cur.append((parts[0], parts[1]))
            else:
                cur.append((" ".join(parts[:-1]), "<<MALFORMED>>"))
    if cur:
        sentences.append(cur)
    return sentences


def collect_entities(sent):
    """按 BIO 规则抽取实体跨度，同时返回非法转移计数。

    返回 (entities, illegal)；entities 为 [(type, start, end, text), ...]
    """
    entities = []
    illegal = Counter()
    cur_type, cur_start = None, None
    prev_tag = "O"

    for idx, (_char, tag) in enumerate(sent):
        if tag == "<<MALFORMED>>":
            continue
        if tag == "O":
            if cur_type is not None:
                entities.append(_make(sent, cur_type, cur_start, idx))
                cur_type = None
        elif tag.startswith("B-"):
            etype = tag[2:]
            if cur_type is not None:
                entities.append(_make(sent, cur_type, cur_start, idx))
            if prev_tag.startswith("B-") and prev_tag[2:] == etype:
                illegal["B-x 紧跟 B-x（同类相邻实体）"] += 1
            cur_type, cur_start = etype, idx
        elif tag.startswith("I-"):
            etype = tag[2:]
            if prev_tag == "O":
                illegal["I-x 出现在 O 之后（缺 B）"] += 1
                cur_type, cur_start = etype, idx          # 容错：起新实体
            elif prev_tag.startswith("B-") and prev_tag[2:] != etype:
                illegal["I-x 紧跟 B-y 且 x != y"] += 1
                if cur_type is not None:
                    entities.append(_make(sent, cur_type, cur_start, idx))
                cur_type, cur_start = etype, idx
            elif prev_tag.startswith("I-") and prev_tag[2:] != etype:
                illegal["I-x 紧跟 I-y 且 x != y"] += 1
                if cur_type is not None:
                    entities.append(_make(sent, cur_type, cur_start, idx))
                cur_type, cur_start = etype, idx
            elif cur_type is None:
                cur_type, cur_start = etype, idx
        prev_tag = tag

    if cur_type is not None:
        entities.append(_make(sent, cur_type, cur_start, len(sent)))
    return entities, illegal


def _make(sent, etype, start, end):
    text = "".join(c for c, _ in sent[start:end])
    return (etype, start, end, text)


def analyze(name, sentences, out):
    tag_counter = Counter()
    ent_counter = Counter()
    ent_len = {t: Counter() for t in TYPES}
    illegal = Counter()
    illegal_examples = {}
    malformed = 0
    total_chars = 0
    max_len = 0
    only_o = 0
    seen = set()
    dup_internal = 0

    for si, sent in enumerate(sentences):
        max_len = max(max_len, len(sent))
        total_chars += len(sent)
        ents, ill = collect_entities(sent)
        illegal.update(ill)
        for k in ill:
            illegal_examples.setdefault(k, sent)

        if not ents:
            only_o += 1
        key = "".join(c for c, _ in sent)
        if key in seen:
            dup_internal += 1
        seen.add(key)

        for _c, tag in sent:
            tag_counter[tag] += 1
            if tag == "<<MALFORMED>>":
                malformed += 1
        for etype, s, e, _t in ents:
            ent_counter[etype] += 1
            ent_len[etype][e - s] += 1

    p = lambda *a: print(*a, file=out)

    p("=" * 74)
    p(f"[{name}]")
    p(f"  句子数        : {len(sentences)}")
    p(f"  字符数        : {total_chars}")
    p(f"  最长句字符数  : {max_len}")
    p(f"  纯 O 句子数   : {only_o}  ({only_o / max(len(sentences),1) * 100:.1f}%)")
    p(f"  句内重复句    : {dup_internal}")
    p(f"  格式异常行    : {malformed}")
    p("-" * 74)
    p("  标签分布:")
    for tag in sorted(tag_counter):
        p(f"      {tag:<10} {tag_counter[tag]:>8}  ({tag_counter[tag] / max(total_chars,1) * 100:6.2f}%)")
    p("-" * 74)
    p("  实体统计 (按 BIO 跨度解析):")
    for t in TYPES:
        lens = ent_len[t]
        avg = sum(k * v for k, v in lens.items()) / max(sum(lens.values()), 1)
        p(f"      {t}: 数量={ent_counter[t]:<6} 平均长度={avg:5.2f} 最长={max(lens, default=0)}")
        p(f"          长度分布(1-3): " +
          ", ".join(f"{k}字={lens.get(k,0)}" for k in (1, 2, 3)))
    p("-" * 74)
    if illegal:
        p("  !! 非法 BIO 转移:")
        for k, v in illegal.most_common():
            p(f"      {k:<34} {v}")
        for k, ex in illegal_examples.items():
            p(f"      --- 示例 [{k}] ---")
            p("        " + " ".join(f"{c}/{t}" for c, t in ex[:26]))
    else:
        p("  非法 BIO 转移: 无")
    p("=" * 74)
    p("")

    return {
        "sentences": len(sentences),
        "chars": total_chars,
        "tags": tag_counter,
        "ents": ent_counter,
        "illegal": illegal,
        "max_len": max_len,
        "only_o": only_o,
        "malformed": malformed,
        "dup_internal": dup_internal,
    }


def main():
    args = sys.argv[1:]
    report_path = None
    readers = [sys.stdout]
    fh = None
    if "--report" in args:
        os.makedirs(OUT_DIR, exist_ok=True)
        report_path = os.path.join(OUT_DIR, "data_report.txt")
        fh = open(report_path, "w", encoding="utf-8")
        readers.append(fh)

    for out in readers:
        print(f"People's Daily NER 数据集质量报告   (原始目录: {RAW_DIR})", file=out)
        print("", file=out)

    stats = {}
    for fname in FILES:
        path = os.path.join(RAW_DIR, fname)
        if not os.path.exists(path):
            for out in readers:
                print(f"[跳过] 文件不存在: {path}", file=out)
            continue
        sents = read_sentences(path)
        for out in readers:
            stats[fname] = analyze(fname, sents, out)

    for out in readers:
        print("=" * 74, file=out)
        print("汇总与结论", file=out)
        print("=" * 74, file=out)
        # 未知标签
        unknown = Counter()
        for st in stats.values():
            for tag, cnt in st["tags"].items():
                if tag not in VALID_TAGS:
                    unknown[tag] += cnt
        if unknown:
            print(f"  !! 未定义标签: {dict(unknown)}", file=out)
        else:
            print("  标签集合: 全部合法 (O / B,I-PER / B,I-ORG / B,I-LOC)", file=out)

        # 交叉重复
        def key(s):
            return "".join(c for c, _ in s)

        train_keys = {key(s) for s in read_sentences(os.path.join(RAW_DIR, "example.train"))}
        for split in ["example.dev", "example.test"]:
            p = os.path.join(RAW_DIR, split)
            if not os.path.exists(p):
                continue
            ks = [key(s) for s in read_sentences(p)]
            dup = sum(1 for k in ks if k in train_keys)
            print(f"  与 train 完全相同的句子: {split} 有 {dup} / {len(ks)}", file=out)

        malformed_total = sum(st["malformed"] for st in stats.values())
        print(f"  格式异常行合计: {malformed_total}", file=out)
        print("", file=out)

    if fh:
        fh.close()
        print(f"\n[报告已写入] {report_path}")


if __name__ == "__main__":
    sys.exit(main() or 0)
