# -*- coding: utf-8 -*-
"""训练 BiLSTM-CRF。

用法:
    python src/train.py                     # 完整训练
    python src/train.py --epochs 3 --limit 2000   # 快速冒烟测试
    python src/train.py --no-constraints    # 关闭转移约束做对比
"""
import argparse
import csv
import os
import time

import torch

import config as C
import data_utils as D
import engine as E
import metrics as M
from model import BiLSTMCRF


def parse_args():
    ap = argparse.ArgumentParser(description="训练 BiLSTM-CRF 中文 NER 模型")
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--batch-tokens", type=int, default=C.BATCH_TOKEN_BUDGET)
    ap.add_argument("--lr", type=float, default=C.LR)
    ap.add_argument("--patience", type=int, default=C.PATIENCE)
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 条训练句（冒烟测试）")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-constraints", action="store_true", help="关闭 CRF 转移约束")
    ap.add_argument("--seed", type=int, default=C.SEED)
    return ap.parse_args()


def main():
    C.setup_console()
    args = parse_args()
    C.set_seed(args.seed)
    device = torch.device(args.device) if args.device else C.get_device()

    print(f"设备: {device}" +
          (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"最大句长(切分阈值): {C.MAX_SEQ_LEN}   批 token 预算: {args.batch_tokens}\n")

    # ---------------- 数据 ----------------
    encoded, char2idx, _counter = D.prepare_data(verbose=True)
    if args.limit:
        for name in ("train", "dev"):
            encoded[name] = encoded[name][: args.limit]
        print(f"[冒烟测试] train/dev 截断为 {args.limit} 句\n")

    train_data, dev_data = encoded["train"], encoded["dev"]
    print(f"训练句数: {len(train_data)}   验证句数: {len(dev_data)}")
    print(f"词表大小: {len(char2idx)}\n")

    # ---------------- 模型 ----------------
    use_constraints = not args.no_constraints
    model = BiLSTMCRF(
        vocab_size=len(char2idx),
        num_tags=C.NUM_TAGS,
        use_constraints=use_constraints,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {n_params:,}   转移约束: {'开' if use_constraints else '关'}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=C.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=C.LR_PATIENCE)

    os.makedirs(C.LOG_DIR, exist_ok=True)
    os.makedirs(C.MODEL_DIR, exist_ok=True)
    log_rows = []
    best_f1, best_epoch, bad_epochs = -1.0, -1, 0
    history = []

    print("=" * 74)
    print(f"{'轮次':>5}{'学习率':>12}{'训练损失':>12}{'精确率':>10}{'召回率':>10}"
          f"{'dev F1':>10}{'耗时':>10}")
    print("=" * 74)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total_loss, n_batches, n_tokens = 0.0, 0, 0

        for batch in D.make_batches(train_data, args.batch_tokens,
                                    shuffle=True, seed=args.seed + epoch):
            input_ids, tag_ids, mask = D.pad_batch(batch)
            input_ids, tag_ids, mask = (input_ids.to(device), tag_ids.to(device),
                                        mask.to(device))
            loss = model(input_ids, tag_ids, mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.CLIP_GRAD)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            n_tokens += int(mask.sum().item())
            if n_batches % 50 == 0:
                print(f"    epoch {epoch} | batch {n_batches} | "
                      f"avg loss {total_loss / n_batches:.4f}", flush=True)

        train_loss = total_loss / max(n_batches, 1)
        metrics, _g, _p = E.evaluate(model, dev_data, device, args.batch_tokens)
        dev_f1 = metrics["micro"]["f1"]
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"{epoch:>5}{lr_now:>12.2e}{train_loss:>12.4f}"
              f"{metrics['micro']['precision'] * 100:>9.2f}%"
              f"{metrics['micro']['recall'] * 100:>9.2f}%"
              f"{dev_f1 * 100:>9.2f}%{elapsed:>9.1f}s", flush=True)

        log_rows.append({
            "epoch": epoch, "lr": f"{lr_now:.6g}", "train_loss": f"{train_loss:.4f}",
            "dev_precision": f"{metrics['micro']['precision']:.4f}",
            "dev_recall": f"{metrics['micro']['recall']:.4f}",
            "dev_f1": f"{dev_f1:.4f}", "macro_f1": f"{metrics['macro_f1']:.4f}",
            "seconds": f"{elapsed:.1f}", "tokens": n_tokens,
        })
        history.append(metrics)

        if dev_f1 > best_f1:
            best_f1, best_epoch, bad_epochs = dev_f1, epoch, 0
            E.save_checkpoint(
                C.CKPT_FILE, model, char2idx,
                {
                    "epoch": epoch,
                    "dev_f1": dev_f1,
                    "dev_precision": metrics["micro"]["precision"],
                    "dev_recall": metrics["micro"]["recall"],
                    "per_type": {k: v["f1"] for k, v in metrics["per_type"].items()},
                    "max_seq_len": C.MAX_SEQ_LEN,
                    "use_constraints": use_constraints,
                    "train_sentences": len(train_data),
                },
            )
            print(f"    ✓ 保存最优模型 (dev F1 = {dev_f1 * 100:.2f}%)")
        else:
            bad_epochs += 1

        # 前 LR_WARMUP_EPOCHS 轮保持学习率不变，让模型先跑起来
        if epoch >= C.LR_WARMUP_EPOCHS:
            scheduler.step(dev_f1)
        if bad_epochs >= args.patience and epoch >= C.MIN_EPOCHS:
            print(f"\n早停：dev F1 连续 {args.patience} 轮未提升。")
            break

    # ---------------- 收尾 ----------------
    with open(C.TRAIN_LOG, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)

    print("=" * 74)
    print(f"训练完成。最优轮次 = {best_epoch}，dev F1 = {best_f1 * 100:.2f}%")
    print(f"最优模型: {C.CKPT_FILE}")
    print(f"训练日志: {C.TRAIN_LOG}")
    if history:
        best_metrics = history[best_epoch - 1]
        print()
        print(M.format_metrics(best_metrics, f"验证集最优结果 (epoch {best_epoch})"))


if __name__ == "__main__":
    main()
