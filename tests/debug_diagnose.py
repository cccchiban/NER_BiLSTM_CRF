# -*- coding: utf-8 -*-
"""诊断：pack_padded_sequence 是否与普通 LSTM 等价；模型是否真的学到实体。

运行: python tests/debug_diagnose.py
"""
import os
import sys
from collections import Counter

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import config as C          # noqa: E402
import data_utils as D      # noqa: E402
import engine as E          # noqa: E402
from model import BiLSTMCRF  # noqa: E402


def test_packing_equivalence():
    C.set_seed(0)
    model = BiLSTMCRF(50, C.NUM_TAGS)
    model.eval()

    seqs = [([1, 2, 3, 4, 5, 6], [0] * 6), ([7, 8, 9], [0] * 3)]
    ids, _t, mask = D.pad_batch(seqs)

    with torch.no_grad():
        emb = model.embedding(ids)
        plain, _ = model.lstm(emb)
        packed = model._emissions(ids, mask)
        # _emissions 还接了 dropout(评估态无效) 和 fc
        got = model.fc(model.drop(plain))
        # 说明：未打包时双向 LSTM 的反向分支会从 padding 的 0 开始回扫，
        # 从而污染较短序列的有效位置；打包版本才是正确的。
        # 因此两者在有效位置上本就会有差异，这里只报告差异大小。
        valid = mask.bool().unsqueeze(-1).expand_as(packed)
        diff = (packed - got).abs()[valid].max().item()
        padding_effect = (packed - got).abs()[~valid].max().item()
        print(f"  有效位置差异 {diff:.3e}（反向分支读到 padding 导致，打包更正确）")
        print(f"  padding 位置差异 {padding_effect:.3e}（打包版填 0，符合预期）")


def inspect_trained_model():
    if not os.path.exists(C.CKPT_FILE):
        print("  (暂无检查点，跳过)")
        return
    device = C.get_device()
    model, char2idx, meta = E.load_checkpoint(C.CKPT_FILE, device)
    print(f"  检查点: epoch={meta.get('epoch')} dev_F1={meta.get('dev_f1', 0)*100:.2f}%")

    sents = D.read_conll(C.PROCESSED_FILES["dev"])[:300]
    encoded = D.encode_dataset(sents, char2idx)
    ids, tags, mask = D.pad_batch(encoded)
    ids, tags, mask = ids.to(device), tags.to(device), mask.to(device)

    with torch.no_grad():
        emissions = model._emissions(ids, mask)
        preds = model.crf.decode(emissions, mask)

    # 1) 预测标签分布 vs 真实标签分布
    pred_flat = Counter(C.IDX2TAG[t] for p in preds for t in p)
    gold_flat = Counter(C.IDX2TAG[t] for p in tags.tolist() for t in p
                        if t != C.TAG_PAD_ID)
    total_p = sum(pred_flat.values())
    total_g = sum(gold_flat.values())
    print("\n  预测标签分布 vs 真实标签分布:")
    for t in C.TAGS:
        print(f"    {t:<8} 预测 {pred_flat.get(t,0)/total_p*100:6.2f}%   "
              f"真实 {gold_flat.get(t,0)/total_g*100:6.2f}%")

    # 2) 对真实实体位置，模型的发射分数是否偏好实体标签
    ent_pos = (tags != C.TAG2IDX["O"]) & (tags != C.TAG_PAD_ID)
    o_pos = (tags == C.TAG2IDX["O"])
    with torch.no_grad():
        probs = torch.softmax(emissions, dim=-1)
        # 实体位置上，O 标签的平均概率
        o_prob_on_ent = probs[ent_pos][:, C.TAG2IDX["O"]].mean().item()
        # 实体位置上，正确标签的平均概率
        correct = probs[ent_pos].gather(
            1, tags[ent_pos].unsqueeze(1)).squeeze(1).mean().item()
        o_prob_on_o = probs[o_pos][:, C.TAG2IDX["O"]].mean().item()
    print(f"\n  实体位置上 O 标签平均概率 : {o_prob_on_ent * 100:.2f}%")
    print(f"  实体位置上正确标签平均概率: {correct * 100:.2f}%")
    print(f"  O 位置上 O 标签平均概率   : {o_prob_on_o * 100:.2f}%")

    # 3) 样例
    print("\n  样例（真实 -> 预测）:")
    for i in range(3):
        text = "".join(c for c, _ in sents[i])
        g = "".join(C.IDX2TAG[t] if t != C.TAG_PAD_ID else "" for t in tags[i].tolist())
        p = "".join(C.IDX2TAG[t] for t in preds[i])
        print(f"    文本: {text[:50]}")
        print(f"    真实: {g[:50]}")
        print(f"    预测: {p[:50]}")


if __name__ == "__main__":
    C.setup_console()
    print("诊断中 ...")
    test_packing_equivalence()
    inspect_trained_model()
