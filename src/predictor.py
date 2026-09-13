# -*- coding: utf-8 -*-
"""推理封装：把训练好的模型包装成可直接对任意文本抽取实体的接口。"""
import torch

import config as C
import data_utils as D
import engine as E
import metrics as M

# 句末标点（长文本按此切块，保证每块不超过训练时的最大句长）
SENT_END_CHARS = "。！？；!?;…\n\r"


def split_units(text, max_len):
    """先把文本切成句子单元；超长单元再硬切。"""
    units, start = [], 0
    for i, ch in enumerate(text):
        if ch in SENT_END_CHARS:
            units.append(text[start: i + 1])
            start = i + 1
    if start < len(text):
        units.append(text[start:])

    out = []
    for u in units:
        if not u:
            continue
        if len(u) <= max_len:
            out.append(u)
        else:
            out.extend(u[k: k + max_len] for k in range(0, len(u), max_len))
    return out


def iter_chunks(text, max_len):
    """把句子单元贪心打包成不超过 max_len 的块，保持原文顺序。"""
    buf = ""
    for unit in split_units(text, max_len):
        if buf and len(buf) + len(unit) > max_len:
            yield buf
            buf = ""
        buf += unit
    if buf:
        yield buf


class NERPredictor:
    """封装模型加载与文本级实体抽取。"""

    def __init__(self, ckpt_path=None, device=None, batch_tokens=None,
                 max_len=None):
        self.device = device or C.get_device()
        self.model, self.char2idx, self.meta = E.load_checkpoint(ckpt_path, self.device)
        self.batch_tokens = batch_tokens or C.BATCH_TOKEN_BUDGET
        self.max_len = max_len or self.meta.get("max_seq_len", C.MAX_SEQ_LEN)
        self.tags = C.TAGS

    # -- 单条文本 ------------------------------------------------------------
    def predict(self, text):
        """返回 [{'type','start','end','text','score_free'}...]，按出现位置排序。"""
        return self.predict_batch([text])[0]

    # -- 批量文本 ------------------------------------------------------------
    def predict_batch(self, texts, progress=False):
        jobs = []            # (text_idx, chunk_offset, chunk_text)
        for ti, text in enumerate(texts):
            offset = 0
            for chunk in iter_chunks(text, self.max_len):
                jobs.append((ti, offset, chunk))
                offset += len(chunk)

        results = [[] for _ in texts]
        if not jobs:
            return results

        # 编码并按 token 预算分批
        encoded = []
        for _ti, _off, chunk in jobs:
            ids = [self.char2idx.get(c, C.UNK_TOKEN_ID) for c in chunk]
            encoded.append((ids, [C.O_TAG_IDX] * len(ids)))

        lengths = [len(ids) for ids, _ in encoded]
        sampler = D.TokenBucketSampler(lengths, self.batch_tokens, shuffle=False)
        n_done = 0
        for idx in sampler:
            batch = [encoded[i] for i in idx]
            input_ids, _tags, mask = D.pad_batch(batch)
            input_ids, mask = input_ids.to(self.device), mask.to(self.device)
            paths = self.model.predict(input_ids, mask)
            for local_i, job_i in enumerate(idx):
                ti, offset, chunk = jobs[job_i]
                tags = [C.IDX2TAG[t] for t in paths[local_i]]
                for ent in M.extract_entities(chunk, tags):
                    ent["start"] += offset
                    ent["end"] += offset
                    results[ti].append(ent)
            n_done += len(idx)
            if progress:
                print(f"    推理进度 {n_done}/{len(jobs)} 块", flush=True)

        for ents in results:
            ents.sort(key=lambda e: e["start"])
        return results
