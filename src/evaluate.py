# -*- coding: utf-8 -*-
"""在测试集上评测已训练模型。

用法:
    python src/evaluate.py                      # 评测 test
    python src/evaluate.py --split dev
    python src/evaluate.py --show-errors 10     # 额外打印若干错误样例
    python src/evaluate.py --ckpt models/bilstm_crf_best.pt
"""
import argparse
import os

import torch

import config as C
import data_utils as D
import engine as E
import metrics as M


def parse_args():
    ap = argparse.ArgumentParser(description="评测 BiLSTM-CRF NER 模型")
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"])
    ap.add_argument("--ckpt", default=C.CKPT_FILE)
    ap.add_argument("--batch-tokens", type=int, default=C.BATCH_TOKEN_BUDGET)
    ap.add_argument("--device", default=None)
    ap.add_argument("--show-errors", type=int, default=0,
                    help="打印 N 条预测错误的样例")
    ap.add_argument("--save-pred", action="store_true",
                    help="把预测结果写入 data/processed/pred_<split>.txt")
    return ap.parse_args()


def main():
    C.setup_console()
    args = parse_args()
    device = torch.device(args.device) if args.device else C.get_device()

    if not torch.cuda.is_available():
        torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))

    # 加载词表与数据（复用已清洗语料，不存在则重新清洗）
    if not all(os.path.exists(p) for p in C.PROCESSED_FILES.values()):
        D.prepare_data(verbose=False)
    char2idx, _tags = D.load_vocab()
    sentences = D.read_conll(C.PROCESSED_FILES[args.split])
    encoded = D.encode_dataset(sentences, char2idx)

    model, _c2i, meta = E.load_checkpoint(args.ckpt, device)
    print(f"设备: {device}   检查点: {args.ckpt}")
    if meta:
        print(f"检查点信息: epoch={meta.get('epoch')}, "
              f"dev F1={meta.get('dev_f1', 0) * 100:.2f}%, "
              f"训练句数={meta.get('train_sentences')}")
    print(f"评测集合: {args.split}   句数: {len(sentences)}\n")

    metrics, gold, pred = E.evaluate(model, encoded, device, args.batch_tokens)
    print(M.format_metrics(metrics, f"{args.split} 集评测结果"))

    if args.save_pred:
        out = C.PROCESSED_FILES[args.split].replace(".txt", "_pred.txt")
        with open(out, "w", encoding="utf-8") as f:
            for sent, p in zip(sentences, pred):
                for (char, _gold_tag), ptag in zip(sent, p):
                    f.write(f"{char} {ptag}\n")
                f.write("\n")
        print(f"预测结果已写入: {out}")

    if args.show_errors:
        print_error_samples(sentences, gold, pred, args.show_errors)


def print_error_samples(sentences, gold, pred, n=10):
    """打印实体级预测错误（漏识别 / 误报 / 边界错）的样例。"""
    print("=" * 68)
    print(f"错误样例（最多 {n} 条）")
    print("=" * 68)
    shown = 0
    for sent, g, p in zip(sentences, gold, pred):
        text = D.sentence_text(sent)
        g_spans = set(map(tuple, M.extract_spans(g)))
        p_spans = set(map(tuple, M.extract_spans(p)))
        if g_spans == p_spans:
            continue
        missed = g_spans - p_spans
        spurious = p_spans - g_spans

        def fmt(spans):
            return ", ".join(f"{t}:{text[s:e]}" for t, s, e in sorted(spans, key=lambda x: x[1])) or "-"

        print(f"[{shown + 1}] 文本: {text[:60]}{'...' if len(text) > 60 else ''}")
        print(f"    漏识别: {fmt(missed)}")
        print(f"    误  报: {fmt(spurious)}")
        shown += 1
        if shown >= n:
            break
    if shown == 0:
        print("未发现错误。")
    print("=" * 68)


if __name__ == "__main__":
    main()
