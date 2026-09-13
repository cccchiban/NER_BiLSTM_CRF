# -*- coding: utf-8 -*-
"""实体级评测：严格匹配（类型与边界均一致才算正确）。"""
from collections import Counter

import config as C


def extract_spans(tag_seq):
    """从标签序列抽取实体跨度，返回 [(type, start, end_exclusive), ...]。

    对非法 BIO 做容错解析：孤立的 I-x 视为新实体的开始。
    """
    spans = []
    cur_type, cur_start = None, None
    for i, tag in enumerate(tag_seq):
        if tag == "O":
            if cur_type is not None:
                spans.append((cur_type, cur_start, i))
                cur_type = None
        elif tag.startswith("B-"):
            if cur_type is not None:
                spans.append((cur_type, cur_start, i))
            cur_type, cur_start = tag[2:], i
        elif tag.startswith("I-"):
            etype = tag[2:]
            if cur_type is None or cur_type != etype:
                if cur_type is not None:
                    spans.append((cur_type, cur_start, i))
                cur_type, cur_start = etype, i
    if cur_type is not None:
        spans.append((cur_type, cur_start, len(tag_seq)))
    return spans


def extract_entities(text, tag_seq):
    """抽取带文本的实体，返回 [{'type','start','end','text'}, ...]。"""
    return [
        {"type": t, "start": s, "end": e, "text": text[s:e]}
        for t, s, e in extract_spans(tag_seq)
    ]


def compute_metrics(gold_tag_seqs, pred_tag_seqs, types=None):
    """计算实体级 P/R/F1。

    返回 {'per_type': {type: {..}}, 'micro': {...}, 'macro_f1': float}
    """
    types = types or C.ENTITY_TYPES
    tp, fp, fn = Counter(), Counter(), Counter()

    assert len(gold_tag_seqs) == len(pred_tag_seqs)
    for gold, pred in zip(gold_tag_seqs, pred_tag_seqs):
        gold_set = Counter(extract_spans(gold))
        pred_set = Counter(extract_spans(pred))
        for span, cnt in gold_set.items():
            common = min(cnt, pred_set.get(span, 0))
            tp[span[0]] += common
            fn[span[0]] += cnt - common
        for span, cnt in pred_set.items():
            common = min(cnt, gold_set.get(span, 0))
            fp[span[0]] += cnt - common

    def prf(t_p, f_p, f_n):
        p = t_p / (t_p + f_p) if t_p + f_p else 0.0
        r = t_p / (t_p + f_n) if t_p + f_n else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return p, r, f

    per_type = {}
    for t in types:
        p, r, f = prf(tp[t], fp[t], fn[t])
        per_type[t] = {
            "precision": p, "recall": r, "f1": f,
            "tp": tp[t], "fp": fp[t], "fn": fn[t],
            "support": tp[t] + fn[t],
        }

    total_tp, total_fp, total_fn = sum(tp.values()), sum(fp.values()), sum(fn.values())
    p, r, f = prf(total_tp, total_fp, total_fn)
    macro_f1 = sum(v["f1"] for v in per_type.values()) / max(len(types), 1)

    return {
        "per_type": per_type,
        "micro": {"precision": p, "recall": r, "f1": f,
                  "tp": total_tp, "fp": total_fp, "fn": total_fn},
        "macro_f1": macro_f1,
    }


def format_metrics(metrics, title="评测结果"):
    lines = []
    p = lines.append
    p("=" * 68)
    p(f"{title}（实体级严格匹配）")
    p("=" * 68)
    p(f"{'类型':<8}{'精确率':>10}{'召回率':>10}{'F1':>10}{'TP':>8}{'FP':>8}{'FN':>8}{'支持数':>8}")
    p("-" * 68)
    for t in C.ENTITY_TYPES:
        m = metrics["per_type"][t]
        name = f"{t}({C.ENTITY_NAMES[t]})"
        p(f"{name:<8}{m['precision'] * 100:>9.2f}%{m['recall'] * 100:>9.2f}%"
          f"{m['f1'] * 100:>9.2f}%{m['tp']:>8}{m['fp']:>8}{m['fn']:>8}{m['support']:>8}")
    p("-" * 68)
    mi = metrics["micro"]
    p(f"{'总体':<8}{mi['precision'] * 100:>9.2f}%{mi['recall'] * 100:>9.2f}%"
      f"{mi['f1'] * 100:>9.2f}%{mi['tp']:>8}{mi['fp']:>8}{mi['fn']:>8}")
    p(f"{'宏平均 F1':<8}{metrics['macro_f1'] * 100:>9.2f}%")
    p("=" * 68)
    return "\n".join(lines) + "\n"
