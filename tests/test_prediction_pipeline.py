# -*- coding: utf-8 -*-
"""② 预测链路：特征维度 / 降级兜底 / 训练口径一致性"""
import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read_text(rel):
    with open(os.path.join(ROOT, rel), "r", encoding="utf-8") as f:
        return f.read()


def test_match_predict_uses_57dim():
    """match_predict.py 特征 = 57 维（52 基础 + 5 身价）"""
    text = _read_text("match_predict.py")
    assert "BASE_FEATURE_COLS + VALUE_FEATURE_COLS" in text, \
        "❌ match_predict.py 特征拼接非 57 维"


def test_fallback_default_odds():
    """预测降级兜底：config.json 提供默认盘口与置信阈值（真实赔率缺失时）"""
    cfg = json.load(open(os.path.join(ROOT, "config.json"), encoding="utf-8"))
    pred = cfg.get("prediction", {})
    assert pred.get("default_draw_odds") == 3.5, "❌ 默认平局赔率非 3.5"
    assert pred.get("default_away_odds") == 3.0, "❌ 默认客胜赔率非 3.0"
    assert pred.get("confidence_high") == 0.68, "❌ 高置信阈值非 0.68"
    assert pred.get("confidence_mid") == 0.60, "❌ 中置信阈值非 0.60"


def test_train_pipeline_57dim():
    """训练脚本明确 57 维拼接（不依赖 96 维）"""
    path = os.path.join(ROOT, "training", "train_general_elo_v2.py")
    if not os.path.exists(path):
        pytest.skip("训练脚本未入库（training/ 仅本地执行），CI 跳过")
    text = _read_text("training/train_general_elo_v2.py")
    assert "BASE_FEATURE_COLS + VALUE_COLS" in text, \
        "❌ 训练脚本特征拼接非 57 维"
    # 1.4.0 后不得再 import 96 维 getter
    assert "get_context_features" not in text, "❌ 训练脚本仍 import 96 维 getter"


def test_rebuild_all_features_3_steps():
    """特征重建链路收敛为 3 步（不再含 96 维构建）"""
    path = os.path.join(ROOT, "features", "rebuild_all_features.py")
    if not os.path.exists(path):
        pytest.skip("特征重建脚本未入库（features/ 仅本地执行），CI 跳过")
    text = _read_text("features/rebuild_all_features.py")
    assert "build_match_context_features" not in text, \
        "❌ rebuild_all_features.py 仍含 96 维情境构建步骤"
    assert "build_line_movement_features" not in text, \
        "❌ rebuild_all_features.py 仍含 96 维线变构建步骤"
    assert "build_referee_features" not in text, \
        "❌ rebuild_all_features.py 仍含 96 维裁判构建步骤"
