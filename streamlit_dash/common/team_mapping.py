"""
球队名称映射公共模块
所有页面统一调用，保证中文映射一致性
"""
import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT_DIR)

from team_mapping_v2 import LEAGUE_TEAM_MAP


def get_team_mappings():
    """
    获取球队中英文映射字典
    返回: (std_2_cn, cn_2_std)
    - std_2_cn: 标准英文名 -> 中文名
    - cn_2_std: 中文名 -> 标准英文名
    """
    std_2_cn = {}
    cn_2_std = {}
    for cfg_code, team_map in LEAGUE_TEAM_MAP.items():
        for eng_std, (full_eng, cn_name) in team_map.items():
            std_2_cn[eng_std] = cn_name
            cn_2_std[cn_name] = eng_std
    return std_2_cn, cn_2_std


def format_team_name(team_std, std_2_cn=None):
    """
    格式化球队名显示：有中文显示「中文名 (英文名)」，无中文只显示英文
    """
    if std_2_cn is None:
        std_2_cn, _ = get_team_mappings()
    cn = std_2_cn.get(team_std, "")
    return f"{cn} ({team_std})" if cn else team_std
