# -*- coding: utf-8 -*-
"""④ 配置一致性：版本号 1.4.0 / 金标准口径 51.75% 全链路一致"""
import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read_text(rel):
    with open(os.path.join(ROOT, rel), "r", encoding="utf-8") as f:
        return f.read()


def test_config_version_140():
    cfg = json.load(open(os.path.join(ROOT, "config.json"), encoding="utf-8"))
    assert cfg["project"]["version"] == "1.4.0", \
        f"❌ config.json version = {cfg['project']['version']}，权威出口应为 1.4.0"


def test_readme_version_140():
    text = _read_text("README.md")
    assert "1.4.0" in text, "❌ README.md 未含 1.4.0"


def test_changelog_has_140():
    text = _read_text("CHANGELOG.md")
    assert "[1.4.0]" in text, "❌ CHANGELOG.md 未含 [1.4.0]"


def test_dashboard_version_140():
    text = _read_text("streamlit_dash/⚽_预测中心.py")
    assert 'APP_VERSION = "v1.4.0"' in text, "❌ 看板 APP_VERSION 不是 v1.4.0"


def test_gold_accuracy_consistent():
    """金标准 51.75% 必须同时出现在 README / 结论总览 / 研究索引"""
    docs = ("【文档】/00_结论总览.md", "【策略研究】/00_README_研究索引.md")
    missing = [d for d in docs if not os.path.exists(os.path.join(ROOT, d))]
    if missing:
        pytest.skip(f"研究侧文档未入库（.gitignore 设计），CI 跳过: {missing}")
    for rel in ("README.md", *docs):
        text = _read_text(rel)
        assert "51.75%" in text, f"❌ {rel} 缺金标准 51.75%"
