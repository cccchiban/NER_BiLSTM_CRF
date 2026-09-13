# -*- coding: utf-8 -*-
"""脱敏工具单元测试（用桩预测器，不依赖真实模型，运行快且确定）。

运行: python tests/test_anonymize.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import config as C                 # noqa: E402
from anonymize import Anonymizer, has_cjk   # noqa: E402


class StubPredictor:
    """按预置实体表返回结果，模拟 NER 模型输出。"""

    def __init__(self, entities):
        self.entities = entities

    def predict(self, text):
        return [e for e in self.entities if e.get("_text") in (None, text)]

    def predict_batch(self, texts):
        return [[e for e in self.entities if e.get("_text") in (None, t)]
                for t in texts]


def ent(text, start, end, etype):
    return {"type": etype, "start": start, "end": end, "text": text}


TEXT = "张三在阿里巴巴的杭州总部工作。"


def make(entities, **kwargs):
    # 默认值与 CLI 默认行为保持一致：过滤不含汉字的实体
    params = dict(types=["PER", "ORG", "LOC"], drop_non_cjk=True)
    params.update(kwargs)
    return Anonymizer(StubPredictor(entities), **params)


def test_basic_placeholder():
    ents = [ent("张三", 0, 2, "PER"), ent("阿里巴巴", 3, 7, "ORG"), ent("杭州", 8, 10, "LOC")]
    masked, used = make(ents).anonymize(TEXT)
    expect = "[人名]在[机构名]的[地名]总部工作。"
    assert masked == expect, f"\n得到: {masked}\n期望: {expect}"
    assert len(used) == 3
    # 未被识别的原文必须原样保留
    assert "总部工作。" in masked
    print("  ✓ 默认占位符替换正确且不破坏非实体文本")


def test_tag_style():
    ents = [ent("张三", 0, 2, "PER"), ent("阿里巴巴", 3, 7, "ORG")]
    masked, _ = make(ents, style="tag").anonymize(TEXT)
    assert masked == "[PER]在[ORG]的杭州总部工作。", masked
    print("  ✓ tag 样式正确")


def test_types_filter():
    ents = [ent("张三", 0, 2, "PER"), ent("阿里巴巴", 3, 7, "ORG"), ent("杭州", 8, 10, "LOC")]
    masked, used = make(ents, types=["PER"]).anonymize(TEXT)
    assert masked == "[人名]在阿里巴巴的杭州总部工作。", masked
    assert len(used) == 1
    print("  ✓ --types 只脱敏指定类型")


def test_mask_style():
    ents = [ent("张三", 0, 2, "PER")]
    masked, _ = make(ents, style="mask").anonymize(TEXT)
    assert masked.startswith("**在阿里巴巴"), masked
    masked2, _ = make(ents, style="mask", mask_len=3).anonymize(TEXT)
    assert masked2.startswith("***在阿里巴巴"), masked2
    print("  ✓ mask 样式：按原长 / 固定长度均正确")


def test_custom_template_with_consistent_ids():
    text = "张三和李四都在北京，张三后来去了上海。"
    ents = [
        ent("张三", 0, 2, "PER"), ent("李四", 3, 5, "PER"),
        ent("北京", 7, 9, "LOC"), ent("张三", 10, 12, "PER"),
        ent("上海", 16, 18, "LOC"),
    ]
    masked, _ = make(ents, style="custom", template="<{name}#{id}>",
                     with_id=True, consistent_ids=True).anonymize(text)
    expect = "<人名#1>和<人名#2>都在<地名#3>，<人名#1>后来去了<地名#4>。"
    assert masked == expect, f"\n得到: {masked}\n期望: {expect}"
    print("  ✓ 自定义模板 + 同一实体一致编号正确")


def test_inconsistent_ids_increment():
    text = "张三和李四。"
    ents = [ent("张三", 0, 2, "PER"), ent("李四", 3, 5, "PER")]
    masked, _ = make(ents, with_id=True, consistent_ids=False).anonymize(text)
    assert masked == "[人名1]和[人名2]。", masked
    print("  ✓ 关闭一致编号时按出现顺序递增编号")


def test_non_cjk_entities_dropped():
    """默认应丢弃纯拉丁字母的实体，避免改坏邮箱/网址。"""
    text = "邮箱 zhangsan@example.com"
    ents = [ent("zhangsan", 3, 11, "ORG")]
    masked, used = make(ents).anonymize(text)          # drop_non_cjk 默认 True
    assert masked == text, f"邮箱被破坏: {masked}"
    assert used == []

    masked2, used2 = make(ents, drop_non_cjk=False).anonymize(text)
    assert masked2 == "邮箱 [机构名]@example.com", masked2
    assert len(used2) == 1
    assert not has_cjk("zhangsan") and has_cjk("张三")
    print("  ✓ 纯拉丁字母实体默认被过滤，可显式关闭")



def test_chinese_entity_kept_even_with_latin():
    """含汉字的实体不应被过滤。"""
    text = "TCL集团在惠州。"
    ents = [ent("TCL集团", 0, 5, "ORG")]
    masked, used = make(ents).anonymize(text)
    assert masked == "[机构名]在惠州。", masked
    assert len(used) == 1
    print("  ✓ 含汉字的实体（如 TCL集团）不被过滤")


def test_overlapping_entities():
    """重叠实体只保留靠前/更长的那个，避免重复遮蔽与越界。"""
    text = "北京市海淀区"
    ents = [ent("北京市", 0, 3, "LOC"), ent("北京市海淀区", 0, 6, "LOC"),
            ent("海淀区", 3, 6, "LOC")]
    masked, used = make(ents).anonymize(text)
    assert masked == "[地名]", masked
    assert len(used) == 1
    print("  ✓ 重叠实体正确处理，不重复遮蔽")


def test_no_entities():
    masked, used = make([]).anonymize(TEXT)
    assert masked == TEXT and used == []
    print("  ✓ 无实体时原文不变")


def test_offset_correctness_with_long_text():
    """长文本中的实体偏移应精确替换，不破坏其上下文。"""
    prefix = "前文内容。" * 20
    text = prefix + "张三是工程师。" + "后文内容。" * 20
    start = len(prefix)
    ents = [ent("张三", start, start + 2, "PER")]
    masked, _ = make(ents).anonymize(text)
    assert masked == prefix + "[人名]是工程师。" + "后文内容。" * 20, masked[:80]
    print("  ✓ 长文本偏移替换正确")


if __name__ == "__main__":
    C.setup_console()
    print("运行脱敏工具测试 ...")
    test_basic_placeholder()
    test_tag_style()
    test_types_filter()
    test_mask_style()
    test_custom_template_with_consistent_ids()
    test_inconsistent_ids_increment()
    test_non_cjk_entities_dropped()
    test_chinese_entity_kept_even_with_latin()
    test_overlapping_entities()
    test_no_entities()
    test_offset_correctness_with_long_text()
    print("全部脱敏工具测试通过 ✓")
