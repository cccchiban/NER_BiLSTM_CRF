# -*- coding: utf-8 -*-
"""诊断脚本：小样本过拟合测试。

目的：确认 BiLSTM-CRF 管线本身可学习。
  - 用固定学习率、足够多的梯度步，尝试过拟合 N 条句子
  - 对比 torch.nn.CrossEntropyLoss（关掉 CRF）的结果作为基准

运行: python tests/debug_overfit.py [N]
"""
import os
import sys
import time

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import config as C          # noqa: E402
import data_utils as D      # noqa: E402
import metrics as M         # noqa: E402
from model import BiLSTMCRF  # noqa: E402


def tag_distribution(preds):
    from collections import Counter
    c = Counter()
    for p in preds:
        c.update(p)
    total = sum(c.values()) or 1
    return ", ".join(f"{t}:{c.get(t, 0) / total * 100:.1f}%" for t in C.TAGS)


def main():
    C.setup_console()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    C.set_seed(0)
    device = C.get_device()

    encoded, char2idx, _ = D.prepare_data(verbose=False)
    data = encoded["train"][:n]
    text_gold = [[C.IDX2TAG[t] for t in tags] for _ids, tags in data]
    print(f"设备 {device} | 样本 {len(data)} 句 | "
          f"平均长度 {sum(len(i) for i, _ in data) / len(data):.1f}")
    print(f"真实标签分布: {tag_distribution(text_gold)}\n")

    input_ids, tag_ids, mask = D.pad_batch(data)
    input_ids, tag_ids, mask = (input_ids.to(device), tag_ids.to(device),
                                mask.to(device))

    for name, use_crf in (("BiLSTM-CRF", True), ("BiLSTM+CE(无CRF)", False)):
        C.set_seed(0)
        model = BiLSTMCRF(len(char2idx), C.NUM_TAGS,
                          use_constraints=use_crf).to(device)
        if use_crf:
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss_fn = None
        else:
            # 把 CRF 换成普通交叉熵：直接取 emissions
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss_fn = nn.CrossEntropyLoss(ignore_index=C.TAG_PAD_ID)

        print(f"--- {name} ---")
        t0 = time.time()
        for step in range(1, 401):
            model.train()
            if use_crf:
                loss = model(input_ids, tag_ids, mask)
            else:
                emissions = model._emissions(input_ids, mask)
                loss = loss_fn(emissions.reshape(-1, C.NUM_TAGS),
                               tag_ids.reshape(-1))
            opt.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            if step % 50 == 0 or step == 1:
                model.eval()
                preds = model.predict(input_ids, mask) if use_crf else None
                if preds is not None:
                    pred_tags = [[C.IDX2TAG[t] for t in p] for p in preds]
                    mt = M.compute_metrics(text_gold, pred_tags)
                    f1 = mt["micro"]["f1"] * 100
                    info = f"| emF1 {f1:5.2f}%  {tag_distribution(pred_tags)}"
                else:
                    info = ""
                print(f"  step {step:>4}  loss {loss.item():8.4f}  "
                      f"|g| {grad_norm:6.2f} {info}", flush=True)
        print(f"  用时 {time.time() - t0:.1f}s\n")


if __name__ == "__main__":
    main()
