# -*- coding: utf-8 -*-
"""BiLSTM-CRF 模型。

CRF 层为手写实现（不依赖 torchcrf），包含：
  - 前向归一化 (log-sum-exp) 与负对数似然
  - Viterbi 解码（支持变长序列）
  - 可选非法 BIO 转移约束（把非法转移的分数压到极低）
"""
import torch
import torch.nn as nn

import config as C

NEG_INF = -1e4


class CRF(nn.Module):
    """线性链 CRF（一阶）。transitions[i][j] 表示从标签 i 转移到 j 的分数。"""

    def __init__(self, num_tags, use_constraints=True):
        super().__init__()
        self.num_tags = num_tags
        self.use_constraints = use_constraints

        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))

        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

        # 预计算非法转移掩码（True 表示禁止）
        trans_mask = torch.zeros(num_tags, num_tags, dtype=torch.bool)
        start_mask = torch.zeros(num_tags, dtype=torch.bool)
        end_mask = torch.zeros(num_tags, dtype=torch.bool)
        if use_constraints:
            for i, tag_i in enumerate(C.TAGS):
                # 序列不能以 I-x 开头
                if tag_i.startswith("I-"):
                    start_mask[i] = True
                # 序列不能以 B-x 结尾（实体不完整）
                if tag_i.startswith("B-"):
                    end_mask[i] = True
                for j, tag_j in enumerate(C.TAGS):
                    if tag_j.startswith("I-"):
                        etype = tag_j[2:]
                        legal = tag_i.startswith(("B-", "I-")) and tag_i[2:] == etype
                        if not legal:
                            trans_mask[i, j] = True
        self.register_buffer("trans_mask", trans_mask)
        self.register_buffer("start_mask", start_mask)
        self.register_buffer("end_mask", end_mask)

    # -- 约束后的分数 --------------------------------------------------------
    def _constrained_transitions(self):
        if not self.use_constraints:
            return self.transitions
        return self.transitions.masked_fill(self.trans_mask, NEG_INF)

    def _constrained_start(self):
        if not self.use_constraints:
            return self.start_transitions
        return self.start_transitions.masked_fill(self.start_mask, NEG_INF)

    def _constrained_end(self):
        if not self.use_constraints:
            return self.end_transitions
        return self.end_transitions.masked_fill(self.end_mask, NEG_INF)

    # -- 打分 ----------------------------------------------------------------
    def _score(self, emissions, tags, mask):
        """给定标签序列的对数似然（未归一化）。向量化实现。

        emissions: (B, T, C)   tags: (B, T)   mask: (B, T) float（1=真实 token）
        """
        batch, seq_len, num_tags = emissions.shape
        # padding 位置的标签为 TAG_PAD_ID(-100)，直接索引会越界；
        # 这里先钳到合法范围，这些位置随后会被 mask 置零，不影响结果。
        safe = tags.clamp(min=0)

        emit_all = emissions.gather(2, safe.unsqueeze(2)).squeeze(2)   # (B, T)
        score = self._constrained_start()[safe[:, 0]] + emit_all[:, 0]

        if seq_len > 1:
            trans_all = self._constrained_transitions()               # (C, C)
            trans = trans_all[safe[:, :-1], safe[:, 1:]]              # (B, T-1)
            score = score + (trans + emit_all[:, 1:]).mul(mask[:, 1:]).sum(1)

        lengths = mask.sum(1).long().clamp(min=1) - 1
        last_tags = safe.gather(1, lengths.unsqueeze(1)).squeeze(1)
        return score + self._constrained_end()[last_tags]

    def _normalizer(self, emissions, mask):
        """log Z：所有合法路径的对数和（逐时间步的前向算法）。

        转移矩阵在循环外只构建一次，避免每步重复分配。
        """
        batch, seq_len, num_tags = emissions.shape
        trans = self._constrained_transitions().unsqueeze(0)   # (1, C_src, C_dst)
        alpha = self._constrained_start() + emissions[:, 0]
        for i in range(1, seq_len):
            m = mask[:, i].unsqueeze(1)
            scores = alpha.unsqueeze(2) + trans + emissions[:, i].unsqueeze(1)
            new_alpha = torch.logsumexp(scores, dim=1)
            # padding 位置不更新
            alpha = torch.where(m.bool(), new_alpha, alpha)
        alpha = alpha + self._constrained_end()
        return torch.logsumexp(alpha, dim=1)

    # -- 对外接口 ------------------------------------------------------------
    def neg_log_likelihood(self, emissions, tags, mask):
        """返回 batch 平均负对数似然（标量 tensor）。"""
        log_z = self._normalizer(emissions, mask)
        gold = self._score(emissions, tags, mask)
        return (log_z - gold).mean()

    @torch.no_grad()
    def decode(self, emissions, mask):
        """Viterbi 解码，返回 List[List[int]]（已按真实长度截断）。"""
        batch, seq_len, num_tags = emissions.shape
        score = self._constrained_start() + emissions[:, 0]
        history = []
        trans = self._constrained_transitions().unsqueeze(0)

        for i in range(1, seq_len):
            m = mask[:, i].unsqueeze(1)
            scores = score.unsqueeze(2) + trans
            best_score, best_tag = scores.max(dim=1)
            best_score = best_score + emissions[:, i]
            score = torch.where(m.bool(), best_score, score)
            history.append(torch.where(m.bool(), best_tag,
                                       torch.zeros_like(best_tag)))

        score = score + self._constrained_end()
        best_last = score.argmax(dim=1)
        lengths = mask.sum(1).long().clamp(min=1).tolist()

        results = []
        for b, length in enumerate(lengths):
            last = int(best_last[b].item())
            path = [last]
            j = length - 1
            while j > 0:
                last = int(history[j - 1][b][last].item())
                path.append(last)
                j -= 1
            path.reverse()
            results.append(path)
        return results


class BiLSTMCRF(nn.Module):
    """Embedding -> BiLSTM -> Linear -> CRF。"""

    def __init__(self, vocab_size, num_tags, embed_dim=None, hidden_dim=None,
                 num_layers=None, dropout=None, pad_idx=C.PAD_TOKEN_ID,
                 use_constraints=C.USE_TRANSITION_CONSTRAINTS):
        super().__init__()
        embed_dim = embed_dim or C.EMBED_DIM
        hidden_dim = hidden_dim or C.HIDDEN_DIM
        num_layers = num_layers or C.NUM_LAYERS
        dropout = C.DROPOUT if dropout is None else dropout

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.1)
        with torch.no_grad():
            self.embedding.weight[pad_idx].zero_()

        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim * 2, num_tags)
        self.crf = CRF(num_tags, use_constraints=use_constraints)

    def _emissions(self, input_ids, mask):
        emb = self.drop(self.embedding(input_ids))
        # 用 pack_padded_sequence 跳过长 padding，缩短 CRF 需要展开的时间步
        lengths = mask.sum(1).long().clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False)
        outputs, _ = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True, total_length=emb.size(1))
        return self.fc(self.drop(outputs))

    def forward(self, input_ids, tags, mask):
        """训练用：返回负对数似然。"""
        return self.crf.neg_log_likelihood(self._emissions(input_ids, mask), tags, mask)

    @torch.no_grad()
    def predict(self, input_ids, mask):
        """预测标签索引（List[List[int]]）。"""
        return self.crf.decode(self._emissions(input_ids, mask), mask)
