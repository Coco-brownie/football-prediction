# -*- coding: utf-8 -*-
"""③ 置信度查表：阈值配置存在 + 校准产物落盘 + 降级说明"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_confidence_thresholds_configured():
    """config.json 置信阈值配置存在"""
    cfg = json.load(open(os.path.join(ROOT, "config.json"), encoding="utf-8"))
    pred = cfg.get("prediction", {})
    assert "confidence_high" in pred, "❌ 缺 confidence_high"
    assert "confidence_mid" in pred, "❌ 缺 confidence_mid"


def test_wf_confidence_accuracy_exists():
    """WF 置信度-准确率查表产物存在"""
    path = os.path.join(ROOT, "model", "wf_confidence_accuracy.json")
    assert os.path.exists(path), "❌ model/wf_confidence_accuracy.json 缺失"


def test_dashboard_mentions_research():
    """看板 AI 出手参考为研究侧叙述（降级说明存在）"""
    text = open(os.path.join(ROOT, "streamlit_dash", "⚽_预测中心.py"), encoding="utf-8").read()
    assert "不构成任何建议" in text, "❌ 看板缺「不构成任何建议」降级说明"
