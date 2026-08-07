# -*- coding: utf-8 -*-
"""🔒 看板研究侧红线：AI 卡片描述不得出现定性/暗示性表述（1.4.0 口径收敛）"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DASH_DIR = os.path.join(ROOT, "streamlit_dash")

# 定性/暗示性词表（命中即 FAIL）
BANNED = [
    "已定稿", "可产品化", "放心重仓", "可靠性最高",
    "仓位建议固定", "已验证成立", "可重点参考", "理论最优",
    "存活确认", "团队主力", "底盘主力", "免费彩票",
]


def _iter_py_files():
    for root, _, files in os.walk(DASH_DIR):
        for fn in files:
            if fn.endswith(".py"):
                yield os.path.join(root, fn)


def test_no_qualitative_claims_in_dashboard():
    violations = []
    for path in _iter_py_files():
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        for word in BANNED:
            if word in text:
                violations.append(f"{os.path.relpath(path, ROOT)} → 「{word}」")
    assert not violations, f"❌ 看板仍含定性/暗示性描述：{violations}"


def test_falcon_plus_research_default_off():
    """猎鹰Plus 为研究档且默认关闭"""
    text = open(os.path.join(ROOT, "streamlit_dash", "⚽_预测中心.py"), encoding="utf-8").read()
    assert "猎鹰Plus（研究" in text, "❌ 猎鹰Plus 未标注研究档"
    assert "value=False" in text, "❌ 猎鹰Plus 未默认关闭"
