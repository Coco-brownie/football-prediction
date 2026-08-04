"""
AI出手建议计算模块
对给定比赛，计算三个AI是否建议出手、共识度等级
（模型验证工具，纯参考；是否出手由用户独立判断，AI 绝不替用户做决定）
【2026-08-08 方案A】置信度口径已改为 WF 分桶历史命中率（见 match_predict._get_confidence）：
  三 AI 阈值适配新口径并拉开梯度（保守优先，宁可不出手），平局整体压一档。
"""
import sys
import os
import pandas as pd

SCRIPT_PATH = os.path.abspath(__file__)
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_PATH))
sys.path.insert(0, ROOT_DIR)

# 三AI配置（与WF回测一致，v1.0.7版本）
# 三个AI的核心区别是定位不同：磐石求稳、天秤均衡、猎鹰求利（冷门猎手）
# 【2026-08-08 方案A】置信度口径改为「WF分桶历史命中率」（41%~87%）后，三AI阈值重新拉开梯度，
#   保守优先、宁可不出手：激进AI 0.50 / 中立AI 0.60 / 保守AI 0.70（历史命中率≥门槛才参考）。
#   【2026-08-18 优化】三档阈值对齐真实命中率梯度：0.50→55.2%、0.60→65.4%、0.70→76.4%。
#   原 0.50/0.55/0.60 中 0.50 与 0.55 同落在 50-60% 档（命中率同为 55.2%），梯度名不副实。
#   【2026-08-18 仓位收紧】kelly_fraction 仅作排序参考，实际仓位建议固定 1%~2%（MC 复核：
#   S1 冷门猎手最安全档=固定1%/2%，凯利0.2已偏险；S3 价值凯利0.2即全爆。激进/中立仓位已压至低位）。
#   【2026-08-19 依据《新口径三AI网格与超级组合复核》回调】激进AI value_margin 1.0→1.2：
#   网格显示 激进0.50×margin1.0 = +3.38%/盈利季56% ⚠️存疑；margin1.2 = +14.51%/盈利季80% ✅（2606场，
#   命中50.8%/赔率2.33=典型冷门猎手风格）。三档 margin 现为 1.2/1.1/1.2，全部 ✅成立。
AI_CONFIGS = {
    "激进AI": {"min_confidence": 0.50, "kelly_fraction": 0.20, "require_value": True, "value_margin": 1.2, "icon": "🦅", "name": "猎鹰"},
    "中立AI": {"min_confidence": 0.60, "kelly_fraction": 0.30, "require_value": True, "value_margin": 1.1, "icon": "⚖️", "name": "天秤"},
    "保守AI": {"min_confidence": 0.70, "kelly_fraction": 0.20, "require_value": True, "value_margin": 1.2, "icon": "🪨", "name": "磐石"},
}

# 参与共识的 AI 键（当前三档；未来新增 AI = AI_CONFIGS 加键 + 本列表登记，
# 共识标签 / 出手参考列 / 角色卡会自动扩展为 4AI/5AI，无需改渲染代码）
CONSENSUS_AI_KEYS = ["激进AI", "中立AI", "保守AI"]

# 价值策略（组合第三腿 · 独立出手逻辑）：EV = 模型概率 × 赔率 - 1
# 研究结论：EV≥0.10 的场次 2012起 9050场 +9.53%（C-014/015），对抽水敏感 → 组合权重 ≤5%
VALUE_STRATEGY_CONFIG = {
    "ev_threshold": 0.10,   # EV ≥ 10% 才算价值机会
    "max_weight": 0.05,     # 组合中 ≤5% 小仓
    "icon": "💎",
    "name": "价值",
}


def compute_value_signal(model_prob, odds):
    """价值策略信号：EV = model_prob × odds - 1；EV≥阈值 出手（独立于三AI共识）"""
    ev = 0.0
    if odds and odds > 0 and model_prob is not None:
        ev = float(model_prob) * float(odds) - 1
    will_bet = ev >= VALUE_STRATEGY_CONFIG["ev_threshold"]
    return {
        "will_bet": will_bet,
        "ev": ev,
        "icon": VALUE_STRATEGY_CONFIG["icon"],
        "display_name": VALUE_STRATEGY_CONFIG["name"],
        "reason": f"EV={ev:+.1%}" + ("" if will_bet else f"（需≥{VALUE_STRATEGY_CONFIG['ev_threshold']:.0%}）"),
    }

# 猎鹰Plus版配置（冷门猎手专精）
# 三个版本：基础版（全联赛）、德甲专精、英超专精
FALCON_PLUS_CONFIG = {
    "基础版": {
        "min_confidence": 0.55,
        "min_odds": 2.5,
        "league_filter": None,
        "icon": "🦅",
        "desc": "全联赛通用，赔率≥2.5+置信≥55%",
    },
    "德甲专精": {
        "min_confidence": 0.50,
        "min_odds": 2.5,
        "league_filter": ["D1"],
        "icon": "🇩🇪",
        "desc": "[实验性 Beta] 仅德甲，赔率≥2.5+置信≥50%【外样本 ⚠️存疑：+6.85%，仅128场/25季≈每季5场，盈利季44%<60%，样本极少勿据此加注】",
    },
    "英超专精": {
        "min_confidence": 0.50,
        "min_odds": 3.0,
        "league_filter": ["E0"],
        "icon": "🏴",
        "desc": "[实验性 Beta] 仅英超，赔率≥3.0+置信≥50%【外样本 ⚠️存疑：+23.51%，仅86场/25季≈每季3.4场，盈利季57%<60%，样本极少勿据此加注】",
    },
}

# 高级模式配置（超级组合策略👑）
# 仅对保守AI生效，开启后升级为超级组合策略
ADVANCED_MODE_CONFIG = {
    "enabled_ai": "保守AI",  # 哪个AI开启高级模式
    "icon": "👑",  # 高级模式标记
    "name_suffix": "Pro",  # 名字后缀
    # 超级组合策略参数
    "min_confidence": 0.55,
    "kelly_fraction": 0.20,
    "require_value": True,
    "value_margin": 1.2,
    # 联赛筛选：只投德甲+意甲
    # 【2026-08-05 修复：意甲用 B体系 db_code I1（原写 SER 旧码，会匹配不上）】
    "league_filter": ["D1", "I1"],
    # 平局策略
    # 【2026-08-19 依据 MC 复核回调】draw_value_margin 1.0→1.2：I1/I3 MC 显示平局组件
    #   不爆仓（flat_1/2 破产0%）但降质（I1 +16.75%/盈利季70% → I3 +11.39%/盈利季52%）。
    #   提高平局价值门槛（EV≈0.20）过滤低质平局，保守方向（更严仅减少出手，不增加风险）。
    "draw_enabled": True,
    "draw_min_confidence": 0.50,
    "draw_kelly_fraction": 0.20,
    "draw_require_value": True,
    "draw_value_margin": 1.2,
    # 置信度打折（磐石最优）
    "confidence_scaling": True,
}


def scale_confidence(conf):
    """置信度打折（磐石最优：保守三档）"""
    if conf < 0.6:
        return conf * 0.98
    elif conf < 0.8:
        return conf * 0.95
    else:
        return conf * 0.90


def calc_ai_bet_intent(pred_confidence, pred_result, odds=None, 
                       league_code=None, draw_prob=None, draw_odds=None,
                       advanced_mode=False, falcon_plus_version=None):
    """
    计算三个AI对单场比赛的出手建议
    
    参数:
        pred_confidence: 模型预测置信度 (0-1)
        pred_result: 预测方向 ('主队胜' / '平局' / '客队胜')
        odds: 对应方向的市场概率（可选，置信度优势判断用）
        league_code: 联赛代码（高级模式联赛筛选用）
        draw_prob: 平局概率（高级模式平局策略用）
        draw_odds: 平局赔率（高级模式平局策略用）
        advanced_mode: 是否开启高级模式（超级组合策略👑）
        falcon_plus_version: 猎鹰Plus版版本（None=关闭，'基础版'/'德甲专精'/'英超专精'）
    
    返回:
        dict: 每个AI的出手建议 + 共识等级
    """
    # 【2026-08-08 方案A】平局整体压一档（+0.05）：1x2 中平局天然低频低准，保守优先
    draw_penalty = 0.05 if pred_result == "平局" else 0.0
    ai_min_conf = {k: cfg["min_confidence"] + draw_penalty for k, cfg in AI_CONFIGS.items()}

    intents = {}
    bet_count = 0
    
    for ai_name, cfg in AI_CONFIGS.items():
        will_bet = True
        reason = ""
        icon = cfg["icon"]
        display_name = ai_name
        is_advanced = False
        
        # 猎鹰Plus版：仅对激进AI生效
        is_falcon_plus = falcon_plus_version and ai_name == "激进AI"
        if is_falcon_plus:
            plus_cfg = FALCON_PLUS_CONFIG.get(falcon_plus_version, {})
            icon = plus_cfg.get("icon", cfg["icon"])
            display_name = f"猎鹰·{falcon_plus_version}"
            
            # 联赛筛选
            if plus_cfg.get("league_filter"):
                if league_code and league_code not in plus_cfg["league_filter"]:
                    will_bet = False
                    reason = "联赛不在专精范围"
            
            # 赔率下限筛选（冷门猎手核心：只买高赔率）
            if will_bet and plus_cfg.get("min_odds") and odds and odds > 0:
                if odds < plus_cfg["min_odds"]:
                    will_bet = False
                    reason = f"赔率不足（需≥{plus_cfg['min_odds']}）"
            
            # 置信度门槛
            if will_bet:
                min_conf = plus_cfg.get("min_confidence", ai_min_conf[ai_name])
                if pred_confidence < min_conf:
                    will_bet = False
                    reason = f"置信度不足（需≥{min_conf:.0%}）"
            
            # 冷门猎手不需要安全边际（高赔率+高置信度天然就有）
            # 所以跳过value_margin检查
            
        # 高级模式：仅对指定AI生效
        elif advanced_mode and ai_name == ADVANCED_MODE_CONFIG["enabled_ai"]:
            is_advanced = True
            icon = ADVANCED_MODE_CONFIG["icon"]
            display_name = f"{ai_name}{ADVANCED_MODE_CONFIG['name_suffix']}"
            
            # 联赛筛选
            if ADVANCED_MODE_CONFIG.get("league_filter"):
                if league_code and league_code not in ADVANCED_MODE_CONFIG["league_filter"]:
                    will_bet = False
                    reason = "联赛不在精选范围（仅德甲+意甲）"
            
            # 置信度打折
            conf = pred_confidence
            if ADVANCED_MODE_CONFIG.get("confidence_scaling"):
                conf = scale_confidence(conf)
            
            # 置信度门槛
            min_conf_adv = ADVANCED_MODE_CONFIG["min_confidence"] + draw_penalty
            if will_bet and conf < min_conf_adv:
                will_bet = False
                reason = f"置信度不足（需≥{min_conf_adv:.0%}）"
            
            # 置信度优势判断
            if will_bet and ADVANCED_MODE_CONFIG["require_value"] and odds and odds > 0:
                implied_prob = 1.0 / odds
                margin = ADVANCED_MODE_CONFIG.get("value_margin", 1.0)
                if conf <= implied_prob * margin:
                    will_bet = False
                    reason = f"价值不足（需{(margin-1)*100:.0f}%安全边际）"
            
            # 平局策略：如果主胜/客胜不出手，看看平局要不要出手
            if not will_bet and ADVANCED_MODE_CONFIG.get("draw_enabled") and draw_prob and draw_odds:
                draw_conf = draw_prob
                # 平局置信度打折？暂时不打
                draw_pass = draw_conf >= ADVANCED_MODE_CONFIG["draw_min_confidence"]
                if draw_pass and ADVANCED_MODE_CONFIG["draw_require_value"] and draw_odds > 0:
                    draw_implied = 1.0 / draw_odds
                    draw_margin = ADVANCED_MODE_CONFIG.get("draw_value_margin", 1.0)
                    draw_pass = draw_conf > draw_implied * draw_margin
                
                if draw_pass:
                    will_bet = True
                    reason = "平局高置信度机会"
                    # 平局也算出手
        else:
            # 普通模式
            # 置信度门槛
            if pred_confidence < ai_min_conf[ai_name]:
                will_bet = False
                reason = f"置信度不足（需≥{ai_min_conf[ai_name]:.0%}）"
            
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
            "icon": icon,
            "display_name": display_name,
            "is_advanced": is_advanced,
            "is_falcon_plus": is_falcon_plus,
        }
        if will_bet:
            bet_count += 1
    
    # 共识等级（基于 CONSENSUS_AI_KEYS 动态计算：三档=三AI共识，未来加AI自动变 4AI/5AI 共识）
    consensus_total = len(CONSENSUS_AI_KEYS)
    consensus_votes = sum(1 for k in CONSENSUS_AI_KEYS if intents.get(k, {}).get("will_bet"))
    if consensus_votes == consensus_total:
        consensus_level = "high"
        consensus_label = "🤝 三AI共识" if consensus_total == 3 else f"🤝 {consensus_total}AI共识"
    elif consensus_votes >= 2:
        consensus_level = "mid"
        consensus_label = "👥 两AI看好" if consensus_total == 3 else f"👥 {consensus_votes}/{consensus_total}AI看好"
    elif consensus_votes == 1 or (consensus_votes == 0 and bet_count > 0):
        consensus_level = "low"
        # 看看是不是只有猎鹰Plus出手
        falcon_plus_only = False
        for ai_name, info in intents.items():
            if info["will_bet"] and info.get("is_falcon_plus"):
                falcon_plus_only = True
                break
        if falcon_plus_only and falcon_plus_version:
            consensus_label = f"🦅 仅猎鹰·{falcon_plus_version}"
        # 看看是不是只有高级AI出手
        advanced_only = False
        for ai_name, info in intents.items():
            if info["will_bet"] and info["is_advanced"]:
                advanced_only = True
                break
        if advanced_only and advanced_mode:
            consensus_label = "👑 仅Pro关注"
        elif not falcon_plus_only:
            consensus_label = "🔥 仅激进关注"
    else:
        consensus_level = "none"
        consensus_label = "❌ 无AI出手"
    
    return {
        "intents": intents,
        "bet_count": bet_count,
        "consensus_level": consensus_level,
        "consensus_label": consensus_label,
        "advanced_mode": advanced_mode,
        "falcon_plus_version": falcon_plus_version,
    }


def format_intent_badges(intent_result):
    """格式化AI出手意愿为徽章字符串列表"""
    badges = []
    for ai_name, info in intent_result["intents"].items():
        status = "✅" if info["will_bet"] else "❌"
        name = info.get("display_name", ai_name)
        badges.append(f"{info['icon']} {status} {name}")
    return badges


# ==================== 产品分级体系（研究→产品映射） ====================
# 对齐金标准《统一策略金标准验证结果》：✅成立 ≠ 可产品化（需 MC 资金曲线安全）
# 等级：stable=✅可产品化 / beta=⚠️实验性(Beta) / watch=❌仅观察
AI_PARAMS_VERSION = "2026-08-19"  # 三AI阈值0.50/0.60/0.70 + 仓位收紧(激进0.20/中立0.30) + 猎鹰专精实验性标注 + margin回调(激进1.2/平局1.2)

PRODUCT_GRADING = {
    "三AI参考": {
        "level": "stable",
        "label": "✅ 可产品化",
        "note": "三AI仅作排序参考；实际仓位建议固定1%~2%（MC：S1冷门猎手/S4磐石最安全档）",
    },
    "猎鹰Plus·基础版": {
        "level": "stable",
        "label": "✅ 可产品化",
        "note": "=S1 冷门猎手（赔率≥2.2 置信≥50%，1759场 +15.79%，盈利季80%），MC 固定1~2%安全",
    },
    "猎鹰Plus·德甲专精": {
        "level": "beta",
        "label": "⚠️ 实验性(Beta)",
        "note": "=C1 德甲（128场 +6.85%，盈利季44%<60%，每季约5场）外样本存疑，样本极少勿据此加注",
    },
    "猎鹰Plus·英超专精": {
        "level": "beta",
        "label": "⚠️ 实验性(Beta)",
        "note": "=C3 英超（86场 +23.51%，盈利季57%<60%，每季约3.4场）外样本存疑，样本极少勿据此加注",
    },
    "超级组合Pro（磐石+平局）": {
        "level": "beta",
        "label": "⚠️ 实验性(Beta)",
        "note": "=I3 超级组合（I1磐石 ✅ + I2平局组件 ⚠️存疑；321场 +11.39% 盈利季52%）\n"
              "【2026-08-19 MC 已验证】含平局 I3：flat_1/2 破产率0%（安全，不爆仓），但盈利季52%<60%、\n"
              "ROI +16.75%→+11.39%，收益低于纯磐石；平局组件已加严价值门槛(1.2)，介意者可关闭平局",
    },
    "平局组件（I2 德甲+意甲）": {
        "level": "beta",
        "label": "⚠️ 实验性(Beta)",
        "note": "=I2 平局·德甲+意甲 概率≥50%（162场 +5.68%，盈利季52%）外样本存疑；勿单独重仓平局",
    },
}


def get_product_grade(key):
    """取某产品的分级（无则默认 watch）"""
    return PRODUCT_GRADING.get(key, {"level": "watch", "label": "❌ 仅观察", "note": "未纳入分级表"})

