"""
AI出手建议计算模块
对给定比赛，计算三个AI是否建议出手、共识度等级
（模型验证工具，非投注建议）
"""
import sys
import os
import pandas as pd

SCRIPT_PATH = os.path.abspath(__file__)
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_PATH))
sys.path.insert(0, ROOT_DIR)

# 三AI配置（与回测一致）
AI_CONFIGS = {
    "激进AI": {"min_confidence": 0.50, "kelly_fraction": 0.90, "require_value": True, "value_margin": 1.2, "icon": "🔥"},
    "中立AI": {"min_confidence": 0.65, "kelly_fraction": 0.60, "require_value": True, "value_margin": 1.2, "icon": "⚖️"},
    "保守AI": {"min_confidence": 0.70, "kelly_fraction": 0.20, "require_value": True, "value_margin": 1.3, "icon": "🛡️"},
}


def calc_ai_bet_intent(pred_confidence, pred_result, odds=None):
    """
    计算三个AI对单场比赛的出手建议
    
    参数:
        pred_confidence: 模型预测置信度 (0-1)
        pred_result: 预测方向 ('主队胜' / '平局' / '客队胜')
        odds: 对应方向的市场概率（可选，置信度优势判断用）
    
    返回:
        dict: 每个AI的出手建议 + 共识等级
    """
    intents = {}
    bet_count = 0
    
    for ai_name, cfg in AI_CONFIGS.items():
        will_bet = True
        reason = ""
        
        # 置信度门槛
        if pred_confidence < cfg["min_confidence"]:
            will_bet = False
            reason = f"置信度不足（需≥{cfg['min_confidence']:.0%}）"
        
        # 置信度优势判断（需要市场概率）
        if will_bet and cfg["require_value"] and odds and odds > 0:
            implied_prob = 1.0 / odds
            margin = cfg.get("value_margin", 1.0)
            if pred_confidence <= implied_prob * margin:
                will_bet = False
                reason = f"价值不足（需{(margin-1)*100:.0f}%安全边际）"
        
        intents[ai_name] = {
            "will_bet": will_bet,
            "reason": reason,
            "icon": cfg["icon"],
        }
        if will_bet:
            bet_count += 1
    
    # 共识等级
    if bet_count == 3:
        consensus_level = "high"
        consensus_label = "🤝 三AI共识"
    elif bet_count == 2:
        consensus_level = "mid"
        consensus_label = "👥 两AI看好"
    elif bet_count == 1:
        consensus_level = "low"
        consensus_label = "🔥 仅激进关注"
    else:
        consensus_level = "none"
        consensus_label = "❌ 无AI出手"
    
    return {
        "intents": intents,
        "bet_count": bet_count,
        "consensus_level": consensus_level,
        "consensus_label": consensus_label,
    }


def format_intent_badges(intent_result):
    """格式化AI出手意愿为徽章字符串列表"""
    badges = []
    for ai_name, info in intent_result["intents"].items():
        status = "✅" if info["will_bet"] else "❌"
        badges.append(f"{info['icon']} {status} {ai_name}")
    return badges
