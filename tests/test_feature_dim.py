# -*- coding: utf-8 -*-
"""① 特征维度回归测试：产品口径必须 = 57 维，96 维旧特征禁止出现在 config.json"""
import json
import os

import pytest

from common_config import get_feature_list, get_value_features, get_full_feature_list

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT, "config.json")
EXPECTED_57 = 57


def _load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_config_has_no_96dim_blocks():
    """config.json 不得再含 96 维特征块（1.4.0 已移入 model/_deprecated/）"""
    cfg = _load_config()
    feats = cfg.get("features", {})
    for block in ("context_features", "line_movement_features", "referee_features"):
        assert block not in feats, f"❌ config.json 仍含废弃块 {block}"


def test_feature_dim_is_57():
    """产品特征维度 = 57（52 基础 + 5 身价）"""
    n = len(get_feature_list()) + len(get_value_features())
    assert n == EXPECTED_57, f"❌ 产品特征维度 = {n}，应为 {EXPECTED_57}"


def test_full_feature_list_is_57():
    """get_full_feature_list() 已收敛为 57 维（不再含 96 维）"""
    n = len(get_full_feature_list())
    assert n == EXPECTED_57, f"❌ get_full_feature_list() = {n}，应为 {EXPECTED_57}"


def test_deprecated_96dim_archived():
    """96 维定义已归档至 model/_deprecated/，可追溯"""
    path = os.path.join(ROOT, "model", "_deprecated", "config_features_96dim.json")
    if not os.path.exists(path):
        pytest.skip("96 维归档未入库（model/_deprecated/ 仅本地保留），CI 跳过")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    n = (len(data.get("context_features", [])) +
         len(data.get("line_movement_features", [])) +
         len(data.get("referee_features", [])))
    assert n == 39, f"❌ 归档 96 维旧特征数 = {n}，应为 39（23+9+7）"


def test_deprecated_getters_read_archive():
    """废弃 getter 从归档读取（研究追溯用），不再依赖 config.json"""
    archive = os.path.join(ROOT, "model", "_deprecated", "config_features_96dim.json")
    if not os.path.exists(archive):
        pytest.skip("96 维归档未入库（model/_deprecated/ 仅本地保留），CI 跳过")
    from common_config import get_context_features, get_line_movement_features, get_referee_features
    assert len(get_context_features()) == 23
    assert len(get_line_movement_features()) == 9
    assert len(get_referee_features()) == 7
