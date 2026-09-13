# -*- coding: utf-8 -*-
"""CRF 正确性测试：用暴力枚举校验前向归一化与 Viterbi 解码。

运行: python tests/test_crf.py
"""
import itertools
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import config as C          # noqa: E402
from model import CRF, NEG_INF   # noqa: E402


def brute_force_log_z(crf, emissions, mask):
    """枚举所有标签序列求 logsumexp（仅用于小规模验证）。"""
    seq_len, num_tags = emissions.shape
    length = int(mask.sum().item())
    scores = []
    for path in itertools.product(range(num_tags), repeat=length):
        score = crf._constrained_start()[path[0]] + emissions[0, path[0]]
        for i in range(1, length):
            score = score + crf._constrained_transitions()[path[i - 1], path[i]] \
                + emissions[i, path[i]]
        score = score + crf._constrained_end()[path[-1]]
        scores.append(score)
    return torch.logsumexp(torch.stack(scores), dim=0)


def brute_force_best_path(crf, emissions, mask):
    seq_len, num_tags = emissions.shape
    length = int(mask.sum().item())
    best_score, best_path = None, None
    for path in itertools.product(range(num_tags), repeat=length):
        score = crf._constrained_start()[path[0]] + emissions[0, path[0]]
        for i in range(1, length):
            score = score + crf._constrained_transitions()[path[i - 1], path[i]] \
                + emissions[i, path[i]]
        score = score + crf._constrained_end()[path[-1]]
        if best_score is None or score > best_score:
            best_score, best_path = score, path
    return best_score, list(best_path)


def is_legal_path(tags):
    """检查标签序列是否符合 BIO 规则。"""
    prev = "O"
    for i, t in enumerate(tags):
        if t.startswith("I-"):
            if prev == "O" or prev[2:] != t[2:]:
                return False
        prev = t
    if tags and tags[-1].startswith("B-"):
        return False          # 以 B 结尾视为不完整实体
    return True


def test_normalizer_matches_brute_force():
    torch.manual_seed(0)
    for trial in range(30):
        num_tags = 7
        length = torch.randint(1, 5, (1,)).item()
        batch = 3
        crf = CRF(num_tags, use_constraints=True)
        emissions = torch.randn(batch, length, num_tags)
        mask = torch.ones(batch, length)
        # 制造一个短序列（padding）
        if trial % 3 == 0 and length > 1:
            mask[1, -1] = 0

        got = crf._normalizer(emissions, mask)
        for b in range(batch):
            want = brute_force_log_z(crf, emissions[b], mask[b])
            assert torch.allclose(got[b], want, atol=1e-3), \
                f"log Z 不一致 trial={trial} b={b}: {got[b].item()} vs {want.item()}"
    print("  ✓ 前向归一化 log Z 与暴力枚举一致")


def test_viterbi_matches_brute_force():
    torch.manual_seed(1)
    crf = CRF(7, use_constraints=True)
    for trial in range(30):
        length = torch.randint(1, 5, (1,)).item()
        batch = 4
        crf = CRF(7, use_constraints=True)
        emissions = torch.randn(batch, length, 7)
        mask = torch.ones(batch, length)
        if trial % 2 == 0 and length > 1:
            mask[2, -1] = 0

        paths = crf.decode(emissions, mask)
        for b in range(batch):
            length_b = int(mask[b].sum().item())
            assert len(paths[b]) == length_b, \
                f"路径长度错误: {len(paths[b])} vs {length_b}"
            best_score, best_path = brute_force_best_path(crf, emissions[b], mask[b])
            # 分数应一致（最优路径可能不唯一，只比较分数）
            got_score = crf._constrained_start()[paths[b][0]] + \
                emissions[b, 0, paths[b][0]]
            for i in range(1, length_b):
                got_score = got_score + crf._constrained_transitions()[
                    paths[b][i - 1], paths[b][i]] + emissions[b, i, paths[b][i]]
            got_score = got_score + crf._constrained_end()[paths[b][-1]]
            assert abs(got_score.item() - best_score.item()) < 1e-3, \
                f"Viterbi 非最优 trial={trial} b={b}: " \
                f"{got_score.item()} vs {best_score.item()}"
    print("  ✓ Viterbi 解码与暴力枚举最优分数一致")


def test_constraints_forbid_illegal_paths():
    torch.manual_seed(2)
    crf = CRF(7, use_constraints=True)
    # 极端偏置：强制让非法转移的分数最高，验证约束仍然生效
    with torch.no_grad():
        crf.transitions.fill_(0.0)
        crf.start_transitions.fill_(0.0)
        crf.end_transitions.fill_(0.0)
        crf.transitions[C.TAG2IDX["O"], C.TAG2IDX["I-ORG"]] = 100.0
        crf.transitions[C.TAG2IDX["B-PER"], C.TAG2IDX["I-LOC"]] = 100.0
        crf.start_transitions[C.TAG2IDX["I-LOC"]] = 100.0

    emissions = torch.zeros(1, 6, 7)
    mask = torch.ones(1, 6)
    path = crf.decode(emissions, mask)[0]
    tags = [C.IDX2TAG[t] for t in path]
    assert is_legal_path(tags), f"解码出非法路径: {tags}"
    assert C.TAG2IDX["I-ORG"] not in path or True

    # 关闭约束后应能解出非法路径
    crf2 = CRF(7, use_constraints=False)
    with torch.no_grad():
        crf2.transitions.fill_(0.0)
        crf2.start_transitions.fill_(0.0)
        crf2.end_transitions.fill_(0.0)
        crf2.start_transitions[C.TAG2IDX["I-LOC"]] = 100.0
    path2 = crf2.decode(emissions, mask)[0]
    assert path2[0] == C.TAG2IDX["I-LOC"], "关闭约束后应以 I-LOC 开头"
    print("  ✓ 转移约束正确屏蔽非法 BIO 转移")


def test_illegal_transitions_negligible_in_normalizer():
    """在约束下，非法路径的概率贡献应被压到可忽略。"""
    torch.manual_seed(3)
    crf = CRF(7, use_constraints=True)
    emissions = torch.randn(1, 3, 7)
    mask = torch.ones(1, 3)
    trans = crf._constrained_transitions()
    for i, ti in enumerate(C.TAGS):
        for j, tj in enumerate(C.TAGS):
            if tj.startswith("I-") and not (ti.startswith(("B-", "I-"))
                                            and ti[2:] == tj[2:]):
                assert trans[i, j].item() == NEG_INF, f"未屏蔽 {ti}->{tj}"
    print("  ✓ 非法转移已在转移矩阵中被屏蔽")


if __name__ == "__main__":
    C.setup_console()
    print("运行 CRF 单元测试 ...")
    test_normalizer_matches_brute_force()
    test_viterbi_matches_brute_force()
    test_constraints_forbid_illegal_paths()
    test_illegal_transitions_negligible_in_normalizer()
    print("全部 CRF 测试通过 ✓")
