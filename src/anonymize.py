# -*- coding: utf-8 -*-
"""敏感数据脱敏工具：基于训练好的 BiLSTM-CRF 识别 PER/ORG/LOC 并替换为占位符。

用法示例:
    # 单条文本
    python src/anonymize.py --text "张三在阿里巴巴的杭州总部工作。"

    # 只脱敏人名和机构名，用标签形式
    python src/anonymize.py --text "..." --types PER,ORG --style tag

    # 掩码形式（长度固定 3 个 *）
    python src/anonymize.py --text "..." --style mask --mask-len 3

    # 自定义模板 + 同一实体一致编号
    python src/anonymize.py --text "..." --style custom --template "<{name}#{id}>" --with-id

    # 只看实体，不替换
    python src/anonymize.py --text "..." --entities-only
    python src/anonymize.py --text "..." --json

    # 批量处理目录（支持 .txt / .csv / .jsonl）
    python src/anonymize.py --input data/samples --output data/masked
"""
import argparse
import csv
import glob
import itertools
import json
import os
import sys

import config as C

# 内置样式的占位符模板：{type}=类型码, {name}=中文名, {id}=编号
# {id} 在未开启 --with-id 时为空串，因此同一模板可覆盖有无编号两种情况
STYLE_TEMPLATES = {
    "placeholder": "[{name}{id}]",
    "tag": "[{type}{id}]",
}


def has_cjk(text):
    """是否含中日韩汉字。"""
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def parse_args():
    ap = argparse.ArgumentParser(
        description="基于 BiLSTM-CRF 的敏感数据脱敏工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = ap.add_argument_group("输入")
    src.add_argument("--text", help="直接输入一段文本")
    src.add_argument("--input", help="输入文件或目录")
    src.add_argument("--pattern", default="*", help="目录模式下的文件通配符")

    out = ap.add_argument_group("输出")
    out.add_argument("--output", help="输出文件或目录")
    out.add_argument("--suffix", default="_masked", help="批量模式下的文件名后缀")
    out.add_argument("--json", action="store_true", help="额外输出实体 JSON 到 stdout")
    out.add_argument("--json-out", help="把实体 JSON 写入指定文件")
    out.add_argument("--entities-only", action="store_true",
                     help="只输出识别到的实体，不生成脱敏文本")

    sty = ap.add_argument_group("脱敏样式")
    sty.add_argument("--types", default="PER,ORG,LOC",
                     help="需要脱敏的类型，逗号分隔，可选 PER,ORG,LOC")
    sty.add_argument("--style", default="placeholder",
                     choices=["placeholder", "tag", "mask", "custom"],
                     help="placeholder=[人名]  tag=[PER]  mask=***  custom=自定义模板")
    sty.add_argument("--template", default="[{name}]",
                     help="style=custom 时的模板，支持 {type} {name} {id}")
    sty.add_argument("--mask-char", default="*")
    sty.add_argument("--mask-len", type=int, default=0, help="掩码长度，0=按原文长度")
    sty.add_argument("--with-id", action="store_true", help="占位符带编号，如 [人名1]")
    sty.add_argument("--consistent-ids", action="store_true",
                     help="同一实体文本复用同一编号")
    sty.add_argument("--id-start", type=int, default=1)
    sty.add_argument("--keep-ascii-entities", action="store_true",
                     help="保留不含汉字的实体（默认丢弃，避免把邮箱/网址/编号中的"
                          "拉丁字母误当实体而破坏原文）")

    misc = ap.add_argument_group("其他")
    misc.add_argument("--ckpt", default=C.CKPT_FILE)
    misc.add_argument("--device", default=None)
    misc.add_argument("--batch-tokens", type=int, default=C.BATCH_TOKEN_BUDGET)
    misc.add_argument("--encoding", default="utf-8", help="输入文件编码")
    return ap.parse_args()


# ----------------------------------------------------------------------------
# 核心脱敏逻辑
# ----------------------------------------------------------------------------
class Anonymizer:
    """把 NER 结果按指定样式改写文本。"""

    def __init__(self, predictor, types, style="placeholder", template="[{name}]",
                 mask_char="*", mask_len=0, with_id=False, consistent_ids=False,
                 id_start=1, drop_non_cjk=True):
        self.predictor = predictor
        self.types = set(types)
        self.style = style
        self.template = template
        self.mask_char = mask_char
        self.mask_len = mask_len
        self.with_id = with_id
        self.consistent_ids = consistent_ids
        self.id_start = id_start
        # 语料中不含汉字的实体仅 31/45518（0.07%），而推理时这类片段多来自
        # 邮箱/网址/编号里的拉丁字母，默认丢弃可避免把原文改坏
        self.drop_non_cjk = drop_non_cjk

    def _placeholder(self, ent, counter, id_map):
        t = ent["type"]
        name = C.ENTITY_NAMES[t]
        if self.style == "mask":
            length = self.mask_len if self.mask_len > 0 else len(ent["text"])
            return self.mask_char * length

        idx = ""
        if self.with_id:
            if self.consistent_ids:
                if ent["text"] not in id_map:
                    id_map[ent["text"]] = next(counter)
                idx = id_map[ent["text"]]
            else:
                idx = next(counter)

        tmpl = STYLE_TEMPLATES.get(self.style, self.template)
        return tmpl.format(type=t, name=name, id=idx)

    def keep(self, ent):
        if ent["type"] not in self.types:
            return False
        if self.drop_non_cjk and not has_cjk(ent["text"]):
            return False
        return True

    def anonymize(self, text, entities=None):
        """返回 (脱敏后文本, 被替换的实体列表)。"""
        if entities is None:
            entities = self.predictor.predict(text)

        targets = sorted(
            (e for e in entities if self.keep(e)),
            key=lambda e: (e["start"], -(e["end"] - e["start"])),
        )
        counter = itertools.count(self.id_start)
        id_map = {}
        pieces, cursor, used = [], 0, []
        for ent in targets:
            if ent["start"] < cursor:      # 重叠实体，跳过后者
                continue
            pieces.append(text[cursor: ent["start"]])
            pieces.append(self._placeholder(ent, counter, id_map))
            cursor = ent["end"]
            used.append(ent)
        pieces.append(text[cursor:])
        return "".join(pieces), used

    def anonymize_many(self, texts):
        """批量脱敏，返回 ([(masked, used), ...], all_entities)。"""
        all_ents = self.predictor.predict_batch(texts)
        results = [self.anonymize(t, e) for t, e in zip(texts, all_ents)]
        return results, all_ents


# ----------------------------------------------------------------------------
# 文件读写
# ----------------------------------------------------------------------------
def read_text_file(path, encoding="utf-8"):
    """读取文本，编码不匹配时依次回退。"""
    for enc in (encoding, "utf-8", "gbk", "utf-8-sig"):
        try:
            with open(path, encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, LookupError):
            continue
    with open(path, encoding=encoding, errors="replace") as f:
        return f.read()


def mask_txt(path, anon, encoding):
    text = read_text_file(path, encoding)
    masked, used = anon.anonymize(text)
    return masked, used


def mask_csv(path, anon, encoding):
    """逐单元格脱敏，保持表格结构。"""
    with open(path, encoding=encoding, newline="", errors="replace") as f:
        rows = list(csv.reader(f))
    flat = [cell for row in rows for cell in row]
    if not flat:
        return rows, 0
    results, _ = anon.anonymize_many(flat)
    masked_flat = [r[0] for r in results]
    n_replaced = sum(len(r[1]) for r in results)
    it = iter(masked_flat)
    return [[next(it) for _ in row] for row in rows], n_replaced


def _walk_strings(obj, path):
    """递归收集对象里的所有字符串，返回 [(path, str), ...]。"""
    found = []
    if isinstance(obj, str):
        found.append((path, obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            found.extend(_walk_strings(v, path + [k]))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            found.extend(_walk_strings(v, path + [i]))
    return found


def _set_by_path(obj, path, value):
    cur = obj
    for key in path[:-1]:
        cur = cur[key]
    cur[path[-1]] = value


def mask_jsonl(path, anon, encoding):
    """逐行 JSON，对所有字符串字段脱敏，保持键值结构。"""
    objs = []
    with open(path, encoding=encoding, errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                objs.append(json.loads(line))
            except json.JSONDecodeError:
                objs.append(line)     # 非 JSON 行按纯文本处理

    slots, texts = [], []
    for oi, obj in enumerate(objs):
        for path_, s in _walk_strings(obj, []):
            slots.append((oi, path_))
            texts.append(s)

    if not texts:
        return objs, 0

    results, _ = anon.anonymize_many(texts)
    n_replaced = 0
    for (oi, path_), (masked, used) in zip(slots, results):
        n_replaced += len(used)
        if path_:
            _set_by_path(objs[oi], path_, masked)
        else:
            objs[oi] = masked
    return objs, n_replaced


# ----------------------------------------------------------------------------
# 输出辅助
# ----------------------------------------------------------------------------
def selected_types(args):
    types = [t.strip().upper() for t in args.types.split(",") if t.strip()]
    invalid = [t for t in types if t not in C.ENTITY_TYPES]
    if invalid:
        raise SystemExit(f"未知类型: {invalid}，可选 {C.ENTITY_TYPES}")
    if not types:
        raise SystemExit("--types 不能为空")
    return types


def entities_payload(entities, types, drop_non_cjk=False):
    return [
        {"type": e["type"], "type_cn": C.ENTITY_NAMES[e["type"]], "text": e["text"],
         "start": e["start"], "end": e["end"]}
        for e in entities
        if e["type"] in types and (not drop_non_cjk or has_cjk(e["text"]))
    ]


def write_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def print_entities(text, payload):
    print(f"原文: {text}")
    if not payload:
        print("  （未识别到目标类型实体）")
    for e in payload:
        print(f"  [{e['type']}/{e['type_cn']}] {e['text']}  (位置 {e['start']}-{e['end']})")


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main():
    C.setup_console()
    args = parse_args()
    if not args.text and not args.input:
        raise SystemExit("请用 --text 或 --input 指定输入（-h 查看帮助）")

    types = selected_types(args)
    from predictor import NERPredictor
    print(f"[加载模型] {args.ckpt}", file=sys.stderr)
    predictor = NERPredictor(args.ckpt, args.device, args.batch_tokens)
    anon = Anonymizer(
        predictor, types,
        style=args.style, template=args.template, mask_char=args.mask_char,
        mask_len=args.mask_len, with_id=args.with_id,
        consistent_ids=args.consistent_ids, id_start=args.id_start,
        drop_non_cjk=not args.keep_ascii_entities,
    )
    want_json = args.json or args.json_out
    drop = not args.keep_ascii_entities

    def payload_of(ents):
        return entities_payload(ents, types, drop)

    # ---------------- 单条文本 ----------------
    if args.text:
        entities = predictor.predict(args.text)
        if args.entities_only:
            payload = payload_of(entities)
            print_entities(args.text, payload)
            if args.json_out:
                write_json(args.json_out, payload)
            elif args.json:
                print(json.dumps(payload, ensure_ascii=False, indent=2))
            return
        masked, used = anon.anonymize(args.text, entities)
        print(masked)
        if args.json_out:
            write_json(args.json_out, payload_of(used))
        elif args.json:
            print(json.dumps(payload_of(used), ensure_ascii=False, indent=2))
        return

    # ---------------- 目录批量 ----------------
    if os.path.isdir(args.input):
        files = sorted(
            f for f in glob.glob(os.path.join(args.input, args.pattern))
            if os.path.isfile(f) and os.path.splitext(f)[1].lower()
            in (".txt", ".csv", ".jsonl", ".json", ".md")
        )
        if not files:
            raise SystemExit(f"目录 {args.input} 中没有匹配 {args.pattern} 的可处理文件")
        out_dir = args.output or os.path.join(args.input, "masked")
        os.makedirs(out_dir, exist_ok=True)

        for fp in files:
            base, ext = os.path.splitext(os.path.basename(fp))
            ext = ext.lower()
            oname = f"{base}{args.suffix}{ext}"
            opath = os.path.join(out_dir, oname)

            used = None
            if ext == ".csv":
                rows, n = mask_csv(fp, anon, args.encoding)
                with open(opath, "w", encoding="utf-8", newline="") as f:
                    csv.writer(f).writerows(rows)
            elif ext in (".jsonl", ".json"):
                objs, n = mask_jsonl(fp, anon, args.encoding)
                with open(opath, "w", encoding="utf-8") as f:
                    for o in objs:
                        f.write(o if isinstance(o, str) else
                                json.dumps(o, ensure_ascii=False))
                        f.write("\n")
            else:
                masked, used = mask_txt(fp, anon, args.encoding)
                n = len(used)
                with open(opath, "w", encoding="utf-8") as f:
                    f.write(masked)
            print(f"[已脱敏] {os.path.basename(fp)} -> {oname}   替换 {n} 处")
            if args.json_out and used is not None:
                write_json(os.path.join(out_dir, f"{base}.entities.json"),
                           payload_of(used))
        print(f"\n全部输出到: {out_dir}")
        return

    # ---------------- 单文件 ----------------
    if not os.path.exists(args.input):
        raise SystemExit(f"输入不存在: {args.input}")

    ext = os.path.splitext(args.input)[1].lower()
    if ext == ".csv":
        rows, n = mask_csv(args.input, anon, args.encoding)
        if args.output:
            with open(args.output, "w", encoding="utf-8", newline="") as f:
                csv.writer(f).writerows(rows)
            print(f"[已脱敏] {args.input} -> {args.output}   替换 {n} 处")
        else:
            csv.writer(sys.stdout).writerows(rows)
        return
    if ext in (".jsonl", ".json"):
        objs, n = mask_jsonl(args.input, anon, args.encoding)
        text_out = "\n".join(
            o if isinstance(o, str) else json.dumps(o, ensure_ascii=False)
            for o in objs
        )
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(text_out)
            print(f"[已脱敏] {args.input} -> {args.output}   替换 {n} 处")
        else:
            print(text_out)
        return

    text = read_text_file(args.input, args.encoding)
    entities = predictor.predict(text)
    if args.entities_only:
        payload = payload_of(entities)
        print_entities(text, payload)
        if want_json:
            write_json(args.json_out, payload) if args.json_out else \
                print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    masked, used = anon.anonymize(text, entities)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(masked)
        print(f"[已脱敏] {args.input} -> {args.output}   替换 {len(used)} 处")
    else:
        print(masked)
    if want_json:
        payload = payload_of(used)
        write_json(args.json_out, payload) if args.json_out else \
            print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
