# -*- coding: utf-8 -*-
"""推理 / 评测引擎与模型检查点读写。"""
import os

import torch

import config as C
import data_utils as D
import metrics as M
from model import BiLSTMCRF


def run_inference(model, encoded, device, batch_tokens=None):
    """对编码后的数据集推理，返回预测的标签序列（List[List[str]]）。

    同时返回损失（若需要，用 gold 计算），此处只做前向解码。
    """
    model.eval()
    # 注意：TokenBucketSampler 按长度分桶，批次顺序与原数据顺序不同，
    # 必须按下标回填，否则预测会与句子错位（曾导致 F1 严重偏低）。
    preds = [None] * len(encoded)
    lengths = [len(ids) for ids, _ in encoded]
    sampler = D.TokenBucketSampler(lengths, batch_tokens, shuffle=False)
    for idx in sampler:
        batch = [encoded[i] for i in idx]
        input_ids, _tag_ids, mask = D.pad_batch(batch)
        input_ids, mask = input_ids.to(device), mask.to(device)
        paths = model.predict(input_ids, mask)
        for local_i, data_i in enumerate(idx):
            preds[data_i] = [C.IDX2TAG[t] for t in paths[local_i]]
    return preds


def evaluate(model, encoded, device, batch_tokens=None):
    """返回 (metrics, gold_tag_seqs, pred_tag_seqs)。"""
    gold = [[C.IDX2TAG[t] for t in tags] for _ids, tags in encoded]
    pred = run_inference(model, encoded, device, batch_tokens)
    return M.compute_metrics(gold, pred), gold, pred


def save_checkpoint(path, model, char2idx, meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "char2idx": char2idx,
            "tags": C.TAGS,
            "model_config": {
                "vocab_size": len(char2idx),
                "num_tags": C.NUM_TAGS,
                "embed_dim": C.EMBED_DIM,
                "hidden_dim": C.HIDDEN_DIM,
                "num_layers": C.NUM_LAYERS,
                "dropout": C.DROPOUT,
                "use_constraints": C.USE_TRANSITION_CONSTRAINTS,
            },
            "meta": meta,
        },
        path,
    )


def load_checkpoint(path=None, device=None):
    """加载检查点，返回 (model, char2idx, meta)。"""
    path = path or C.CKPT_FILE
    device = device or C.get_device()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["model_config"]
    model = BiLSTMCRF(
        vocab_size=cfg["vocab_size"],
        num_tags=cfg["num_tags"],
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        use_constraints=cfg.get("use_constraints", True),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt["char2idx"], ckpt.get("meta", {})
