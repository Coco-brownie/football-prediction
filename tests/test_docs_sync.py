# -*- coding: utf-8 -*-
"""⑤ 文档同步：结论总览 / 研究索引 / backtest 关键数字逐字一致"""
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read_text(rel):
    with open(os.path.join(ROOT, rel), "r", encoding="utf-8") as f:
        return f.read()


def test_no_fixed_finalized_claim():
    """研究侧红线：结论总览不得再出现「📌已定稿」定性描述"""
    path = os.path.join(ROOT, "【文档】/00_结论总览.md")
    if not os.path.exists(path):
        pytest.skip("研究侧文档未入库（.gitignore 设计），CI 跳过")
    text = _read_text("【文档】/00_结论总览.md")
    assert "📌已定稿" not in text, "❌ 结论总览仍含「📌已定稿」"


def test_falcon_exploratory_candidate():
    """猎鹰当前定位 = 探索性候选 / 待独立期复现"""
    path = os.path.join(ROOT, "【文档】/00_结论总览.md")
    if not os.path.exists(path):
        pytest.skip("研究侧文档未入库（.gitignore 设计），CI 跳过")
    text = _read_text("【文档】/00_结论总览.md")
    assert ("探索性候选" in text) or ("待独立期复现" in text), \
        "❌ 结论总览未标注猎鹰「探索性候选/待复现」"


def test_panshi_113_matches():
    """磐石Pro 统一 113 场口径（研究索引不得再出现 203 场）"""
    path = os.path.join(ROOT, "【策略研究】/00_README_研究索引.md")
    if not os.path.exists(path):
        pytest.skip("研究侧文档未入库（.gitignore 设计），CI 跳过")
    text = _read_text("【策略研究】/00_README_研究索引.md")
    assert "113 场" in text, "❌ 研究索引缺磐石Pro 113 场口径"
    assert "203 场" not in text, "❌ 研究索引仍含旧 203 场口径"


def test_backtest_summary_5175():
    """金标准落盘 wf_fused_summary.txt 含 51.75%"""
    path = os.path.join(ROOT, "backtest", "wf_fused_summary.txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            assert "51.75" in f.read(), "❌ wf_fused_summary.txt 缺 51.75%"


def test_dashboard_no_product_claim():
    """看板不再含「组合出手单」等产品化暗示"""
    text = _read_text("streamlit_dash/⚽_预测中心.py")
    assert "组合出手单" not in text, "❌ 看板仍含「组合出手单」"
    assert "AI 研究参考" in text, "❌ 看板缺「AI 研究参考」研究侧措辞"
