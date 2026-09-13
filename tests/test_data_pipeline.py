# -*- coding: utf-8 -*-
"""数据管线与指标单元测试。

运行: python tests/test_data_pipeline.py
"""
import os
import sys
from collections import Counter

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import config as C            # noqa: E402
import data_utils as D        # noqa: E402
import engine as E            # noqa: E402
import metrics as M           # noqa: E402
from model import BiLSTMCRF    # noqa: E402


def test_split_never_breaks_entities():
    """在真实语料上验证：切分后每个实体的字符与标签完全保持不变。"""
    for name in ("train", "dev", "test"):
        raw = D.read_conll(C.RAW_FILES[name])
        stats = Counter()

        def entity_multiset(sents):
            ms = Counter()
            for s in sents:
                for t, a, b in M.extract_spans([tag for _c, tag in s]):
                    ms[(t, "".join(c for c, _ in s[a:b]))] += 1
            return ms

        before = entity_multiset(raw)
        cleaned = D.clean_split(name, raw, stats)
        after = entity_multiset(cleaned)

        missing = before - after
        added = after - before
        assert not missing, f"{name} 清洗后丢失实体: {list(missing.items())[:5]}"
        assert not added, f"{name} 清洗后多出实体: {list(added.items())[:5]}"
        print(f"  ✓ {name}: 清洗前后实体集合完全一致 ({sum(before.values())} 个)")


def test_split_long_sentence_preserves_length():
    raw = D.read_conll(C.RAW_FILES["test"])
    long_sents = [s for s in raw if len(s) > C.MAX_SEQ_LEN]
    assert long_sents, "测试语料里应存在超长句"
    for sent in long_sents[:200]:
        parts = D.split_long_sentence(sent, C.MAX_SEQ_LEN)
        assert "".join(c for s in parts for c, _ in s) == \
            "".join(c for c, _ in sent), "切分后字符内容发生了变化"
        for p in parts:
            assert len(p) <= C.MAX_SEQ_LEN, f"切分后仍有超长句: {len(p)}"
    print(f"  ✓ 超长句切分保持字符完整，抽查 {min(len(long_sents), 200)} 句")


def test_repair_bio():
    bad = [("张", "O"), ("三", "I-PER"), ("在", "O"), ("北", "B-LOC"), ("京", "I-ORG")]
    fixed, n = D.repair_bio(bad)
    assert n == 2, f"应修复 2 处，实际 {n}"
    tags = [t for _c, t in fixed]
    assert tags == ["O", "B-PER", "O", "B-LOC", "B-ORG"], tags
    # 修复后应全部合法
    prev = "O"
    for t in tags:
        assert D.is_legal_bio(t, prev)
        prev = t
    print("  ✓ 非法 BIO 修复正确")


def test_metrics_known_values():
    gold = [
        ["B-PER", "I-PER", "O", "B-LOC", "O"],
        ["O", "B-ORG", "I-ORG", "I-ORG", "O"],
    ]
    # 模型1：PER 边界错、LOC 全对、ORG 少一个字
    pred = [
        ["B-PER", "O", "O", "B-LOC", "O"],
        ["O", "B-ORG", "I-ORG", "O", "O"],
    ]
    m = M.compute_metrics(gold, pred)
    assert m["per_type"]["PER"]["tp"] == 0 and m["per_type"]["PER"]["fn"] == 1
    assert m["per_type"]["LOC"]["tp"] == 1 and m["per_type"]["LOC"]["fp"] == 0
    assert m["per_type"]["ORG"]["tp"] == 0 and m["per_type"]["ORG"]["fn"] == 1
    assert m["per_type"]["ORG"]["fp"] == 1
    assert abs(m["micro"]["precision"] - 1 / 3) < 1e-9
    assert abs(m["micro"]["recall"] - 1 / 3) < 1e-9

    # 完全正确
    m2 = M.compute_metrics(gold, gold)
    assert abs(m2["micro"]["f1"] - 1.0) < 1e-9
    print("  ✓ 实体级指标计算正确")


def test_adjacent_same_type_entities_not_merged_in_metrics():
    """相邻同类实体应计为两个（如 日/B-LOC 俄/B-LOC）。"""
    gold = [["B-LOC", "B-LOC", "O"]]
    pred = [["B-LOC", "B-LOC", "O"]]
    m = M.compute_metrics(gold, pred)
    assert m["per_type"]["LOC"]["tp"] == 2, m["per_type"]["LOC"]
    print("  ✓ 相邻同类实体正确计为 2 个")


def test_model_handles_padding_like_unpadded():
    """带 padding 的批次，每个样本的损失应与单独前向一致。"""
    C.set_seed(0)
    torch.set_num_threads(1)
    model = BiLSTMCRF(50, C.NUM_TAGS)
    model.eval()
    seqs = [
        ([3, 4, 5, 6, 7], [C.TAG2IDX["B-PER"], C.TAG2IDX["I-PER"],
                           C.TAG2IDX["O"], C.TAG2IDX["O"], C.TAG2IDX["O"]]),
        ([8, 9], [C.TAG2IDX["O"], C.TAG2IDX["O"]]),
    ]
    ids, tags, mask = D.pad_batch(seqs)

    with torch.no_grad():
        single_losses = []
        for i, (s_ids, s_tags) in enumerate(seqs):
            si = torch.tensor([s_ids], dtype=torch.long)
            st = torch.tensor([s_tags], dtype=torch.long)
            sm = torch.ones_like(si, dtype=torch.float)
            single_losses.append(model(si, st, sm).item())

        log_z = model.crf._normalizer(
            model._emissions(ids, mask), mask)
        gold = model.crf._score(model._emissions(ids, mask), tags, mask)
        batch_losses = (log_z - gold).tolist()

    for i in range(len(seqs)):
        assert abs(single_losses[i] - batch_losses[i]) < 1e-3, \
            f"样本 {i} 带 padding 与单独前向损失不一致: " \
            f"{single_losses[i]} vs {batch_losses[i]}"
    print("  ✓ 带 padding 的批次与单样本前向损失一致（TAG_PAD_ID 处理正确）")


def test_decode_respects_true_lengths():
    C.set_seed(1)
    model = BiLSTMCRF(50, C.NUM_TAGS)
    model.eval()
    seqs = [([1, 2, 3, 4, 5, 6], [0] * 6), ([7, 8, 9], [0] * 3)]
    ids, _t, mask = D.pad_batch(seqs)
    with torch.no_grad():
        paths = model.predict(ids, mask)
    assert [len(p) for p in paths] == [6, 3], [len(p) for p in paths]
    print("  ✓ 解码路径长度与真实长度一致")


def test_vocab_and_encode_roundtrip():
    sents = D.read_conll(C.PROCESSED_FILES["dev"])[:50]
    char2idx = {C.PAD_TOKEN: C.PAD_TOKEN_ID, C.UNK_TOKEN: C.UNK_TOKEN_ID}
    for s in sents:
        for c, _t in s:
            char2idx.setdefault(c, len(char2idx))
    encoded = D.encode_dataset(sents, char2idx)
    for sent, (ids, tags) in zip(sents, encoded):
        assert len(ids) == len(sent) == len(tags)
        for i, (c, t) in enumerate(sent):
            assert C.IDX2TAG[tags[i]] == t, f"标签还原错误: {t} vs {C.IDX2TAG[tags[i]]}"
    print("  ✓ 编码/解码标签往返一致")


def test_inference_alignment():
    """回归测试：run_inference 必须把预测放回正确的句子下标。

    动态批采样器按长度分桶，批次顺序与原句顺序不同；
    早期版本用 extend 追加导致预测与句子错位，dev F1 严重偏低。
    """
    data = D.encode_dataset(
        D.read_conll(C.PROCESSED_FILES["dev"])[:137], 
        D.load_vocab()[0],
    )

    class CheatModel:
        """直接返回传入批次的 gold 标签，用于检测对齐是否正确。"""

        def __init__(self, mapping):
            self.mapping = mapping

        def eval(self):
            pass

        def predict(self, input_ids, mask):
            out = []
            for i in range(input_ids.size(0)):
                length = int(mask[i].sum().item())
                out.append(self.mapping[tuple(input_ids[i, :length].tolist())])
            return out

    mapping = {}
    for ids, tags in data:          # 同 key 只保留第一次出现，避免歧义
        mapping.setdefault(tuple(ids), tags)

    preds = E.run_inference(CheatModel(mapping), data, torch.device("cpu"), 512)
    gold = [[C.IDX2TAG[t] for t in tags] for _ids, tags in data]
    assert all(p is not None for p in preds), "部分句子没有被预测到"
    m = M.compute_metrics(gold, preds)
    assert m["micro"]["f1"] == 1.0, f"预测与句子未对齐，F1={m['micro']['f1']}"
    print("  ✓ 推理结果与输入句子按下标正确对齐")


if __name__ == "__main__":
    C.setup_console()
    print("运行数据管线与指标测试 ...")
    if not all(os.path.exists(p) for p in C.PROCESSED_FILES.values()):
        D.prepare_data(verbose=False)
    test_split_never_breaks_entities()
    test_split_long_sentence_preserves_length()
    test_repair_bio()
    test_metrics_known_values()
    test_adjacent_same_type_entities_not_merged_in_metrics()
    test_model_handles_padding_like_unpadded()
    test_decode_respects_true_lengths()
    test_vocab_and_encode_roundtrip()
    test_inference_alignment()
    print("全部数据管线测试通过 ✓")
