# -*- coding: utf-8 -*-
"""统一测试入口：依次运行全部单元测试。

运行: python tests/run_all.py
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

SUITES = [
    ("CRF 正确性（与暴力枚举对拍）", "test_crf.py"),
    ("数据管线 / 指标 / 推理对齐", "test_data_pipeline.py"),
    ("脱敏工具", "test_anonymize.py"),
]


def main():
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    failed = []
    for title, script in SUITES:
        print("=" * 68)
        print(f"▶ {title}   ({script})")
        print("=" * 68)
        result = subprocess.run(
            [sys.executable, os.path.join(HERE, script)],
            cwd=ROOT, env=env,
        )
        if result.returncode != 0:
            failed.append(script)
        print()

    print("=" * 68)
    if failed:
        print(f"✗ 有 {len(failed)} 个测试套件失败: {', '.join(failed)}")
        return 1
    print("✓ 全部测试套件通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
