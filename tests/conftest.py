# -*- coding: utf-8 -*-
"""pytest 全局配置：确保项目根目录可被 import（common_config 等）+ 简洁摘要输出。

推荐用法（输出短、末尾附摘要）:
    python -m pytest tests/ -q --tb=short

测试结束后会自动打印每个测试文件一行的 ✅/❌ 摘要，便于快速定位问题。
"""
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ---------- 简洁摘要输出 ----------
# 收集每个测试文件的结果，测试结束时打印一行一类的汇总
_results = defaultdict(list)


def pytest_runtest_logreport(report):
    """在测试用例执行后收集结果（call 阶段）。"""
    if report.when == "call":
        module = report.nodeid.split("::")[0]
        _results[module].append(report.outcome)


def pytest_sessionfinish(session, exitstatus):
    """测试会话结束后打印简洁摘要（每个测试文件一行）。"""
    if not _results:
        return
    total_p = total_f = total_s = 0
    lines = []
    for module in sorted(_results):
        outcomes = _results[module]
        p = outcomes.count("passed")
        f = outcomes.count("failed")
        s = outcomes.count("skipped")
        total_p += p
        total_f += f
        total_s += s
        mark = "✅" if f == 0 else "❌"
        lines.append(f"  {mark} {module:<40} {p:>2} passed / {f:>2} failed / {s:>2} skipped")
    print("\n" + "=" * 66)
    print("测试摘要（tests/）")
    print("-" * 66)
    print("\n".join(lines))
    print("-" * 66)
    print(f"  总计: {total_p} passed / {total_f} failed / {total_s} skipped"
          + ("  🎉 全部通过" if total_f == 0 else "  ⚠️ 有失败"))
    print("=" * 66)
