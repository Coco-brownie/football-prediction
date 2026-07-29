import pandas as pd
import numpy as np
import math
# 【2026-07-23 移除：import joblib，改用原生lgb加载模型】
import lightgbm as lgb
import json
import os
import pickle

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT_PATH, "model")


# 【2026-07-29 特征泄露修复版：去掉shot_on_diff，52维（含ELO+时间衰减）】
# 特征顺序：基础→ELO直接→ELO扩展→时间衰减→联赛独热（与训练时完全一致）
FEATURE_COLS = [
    "h5_gf","h5_ga","h5_shot","h5_shot_ot",
    "h10_gf","h10_ga",
    "a5_gf","a5_ga","a5_shot","a5_shot_ot",
    "a10_gf","a10_ga",
    "odds_draw_real","odds_lose_real",
    "h2h_cnt", "h2h_home_win_rate", "h2h_draw_rate", "h2h_home_gf_avg", "h2h_home_ga_avg",
    "prob_ratio_ha", "prob_draw_share", "prob_max", "prob_entropy", "prob_home_favorite",
    "home_draw_rate_5", "home_draw_rate_10", "away_draw_rate_5", "away_draw_rate_10",
    "home_elo_before", "away_elo_before", "elo_diff_before",
    "h5_gf_elo_weighted", "h5_ga_elo_weighted",
    "a5_gf_elo_weighted", "a5_ga_elo_weighted",
    "home_w5_elo_trend", "home_w10_elo_trend",
    "away_w5_elo_trend", "away_w10_elo_trend",
    "h5_gf_time_decay", "h5_ga_time_decay",
    "a5_gf_time_decay", "a5_ga_time_decay",
    "h10_gf_time_decay", "h10_ga_time_decay",
    "a10_gf_time_decay", "a10_ga_time_decay",
    "league_SER","league_E0","league_D1","league_LIG","league_LLA",
]

# 泊松模型特征（13维，去掉shot_on_diff）
POISSON_FEATURES = [
    "h5_gf", "h5_ga", "a5_gf", "a5_ga",
    "h10_gf", "h10_ga", "a10_gf", "a10_ga",
    "league_SER", "league_E0", "league_D1", "league_LIG", "league_LLA"
]
# 硬编码索引（与训练时GOAL_FEATURES严格对齐），避免FEATURE_COLS变化导致错位
# 顺序：h5_gf, h5_ga, a5_gf, a5_ga, h10_gf, h10_ga, a10_gf, a10_ga, league_*5
POISSON_FEAT_IDX = [0, 1, 6, 7, 4, 5, 10, 11, 47, 48, 49, 50, 51]

# 融合权重：LGB 55% + 泊松 30% + 平局专项 5%（网格搜索OOS最优保守方案）
FUSION_WEIGHT_LGB = 0.55
FUSION_WEIGHT_POISSON = 0.30
FUSION_WEIGHT_DRAW = 0.05

# 【2026-07-29 特征泄露修复版：去掉shot_on_diff，联赛独立模型47维】
LEAGUE_FEATURE_COLS = [
    "h5_gf","h5_ga","h5_shot","h5_shot_ot",
    "h10_gf","h10_ga",
    "a5_gf","a5_ga","a5_shot","a5_shot_ot",
    "a10_gf","a10_ga",
    "odds_draw_real","odds_lose_real",
    "h2h_cnt", "h2h_home_win_rate", "h2h_draw_rate", "h2h_home_gf_avg", "h2h_home_ga_avg",
    "prob_ratio_ha", "prob_draw_share", "prob_max", "prob_entropy", "prob_home_favorite",
    "home_draw_rate_5", "home_draw_rate_10", "away_draw_rate_5", "away_draw_rate_10",
    "home_elo_before", "away_elo_before", "elo_diff_before",
    "h5_gf_elo_weighted", "h5_ga_elo_weighted",
    "a5_gf_elo_weighted", "a5_ga_elo_weighted",
    "home_w5_elo_trend", "home_w10_elo_trend",
    "away_w5_elo_trend", "away_w10_elo_trend",
]

# 联赛模型缓存
_league_model_cache = {}

# 【2026-07-24 优化：客场模型特征互换不完整，融合反而降准，改用单主场模型】
home_model = lgb.Booster(model_file=os.path.join(MODEL_DIR, "home_model.pkl"))

# 【2026-07-29 修复：泊松模型改用参数化方式，彻底解决pickle跨平台兼容性问题】
# PoissonRegressor = 线性模型，预测公式: lambda = exp(X @ coef + intercept)
with open(os.path.join(MODEL_DIR, "poisson_model_params.json"), "r") as f:
    _poisson_params = json.load(f)
_poisson_home_coef = np.array(_poisson_params["home_coef"])
_poisson_home_intercept = _poisson_params["home_intercept"]
_poisson_away_coef = np.array(_poisson_params["away_coef"])
_poisson_away_intercept = _poisson_params["away_intercept"]

def _predict_poisson_goals(X_poi, is_home=True):
    """手动计算泊松进球预测，不依赖sklearn pickle"""
    coef = _poisson_home_coef if is_home else _poisson_away_coef
    intercept = _poisson_home_intercept if is_home else _poisson_away_intercept
    return np.exp(X_poi @ coef + intercept)

# 【2026-07-29 修复：平局二分类模型改用LightGBM原生格式，解决pickle兼容性】
draw_binary_model = lgb.Booster(model_file=os.path.join(MODEL_DIR, "draw_binary_model.txt"))

# 【2026-07-26 新增：Platt概率校准器】
_calibrator = None
def _get_calibrator():
    global _calibrator
    if _calibrator is None:
        cal_path = os.path.join(MODEL_DIR, "platt_calibrator_home.pkl")
        if os.path.exists(cal_path):
            with open(cal_path, "rb") as f:
                _calibrator = pickle.load(f)
    return _calibrator


def apply_probability_calibration(probs):
    """应用Platt概率校准
    输入：shape=(n, 3)的概率数组 [主胜, 平局, 客胜]
    输出：校准后的概率
    """
    cal = _get_calibrator()
    if cal is None:
        return probs  # 校准器不存在则返回原值
    
    cal_probs = np.zeros_like(probs)
    for i in range(3):
        cal_probs[:, i] = cal[i].predict_proba(probs[:, i].reshape(-1, 1))[:, 1]
    # 归一化
    row_sums = cal_probs.sum(axis=1, keepdims=True)
    # 防止除零
    row_sums = np.where(row_sums == 0, 1, row_sums)
    cal_probs = cal_probs / row_sums
    return cal_probs


def calc_score_matrix(h_lam, a_lam, max_goals=8):
    """计算所有比分的概率矩阵（泊松独立假设）"""
    scores = {}
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            ph = (h_lam ** h) * math.exp(-h_lam) / math.factorial(h)
            pa = (a_lam ** a) * math.exp(-a_lam) / math.factorial(a)
            scores[f"{h}-{a}"] = float(ph * pa)
    return scores


def calc_over_under(h_lam, a_lam, lines=[1.5, 2.5, 3.5, 4.5]):
    """计算多档位大小球概率"""
    total_lam = h_lam + a_lam
    results = {}
    for line in lines:
        under_prob = 0.0
        for goals in range(int(line) + 1):
            under_prob += (total_lam ** goals) * math.exp(-total_lam) / math.factorial(goals)
        results[f"大球{line}"] = float(1 - under_prob)
        results[f"小球{line}"] = float(under_prob)
    return results


def get_top_scores(scores, top_n=5):
    """获取概率最高的TOP N比分"""
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_scores[:top_n]


def calc_poisson_1x2(feature_array):
    """泊松进球预测 → 转胜平负概率"""
    X = np.array(feature_array, dtype=np.float64).reshape(1, -1)
    X_poi = X[:, POISSON_FEAT_IDX]

    h_lam = float(_predict_poisson_goals(X_poi, is_home=True)[0])
    a_lam = float(_predict_poisson_goals(X_poi, is_home=False)[0])
    h_lam = max(min(h_lam, 10.0), 0.3)
    a_lam = max(min(a_lam, 10.0), 0.3)

    p_home = 0.0
    p_draw = 0.0
    p_away = 0.0
    for h in range(9):
        for a in range(9):
            p_h = (h_lam ** h) * math.exp(-h_lam) / math.factorial(h)
            p_a = (a_lam ** a) * math.exp(-a_lam) / math.factorial(a)
            prob = p_h * p_a
            if h > a:
                p_home += prob
            elif h == a:
                p_draw += prob
            else:
                p_away += prob

    total = p_home + p_draw + p_away
    if total < 1e-8:
        # 兜底：泊松计算异常时返回均匀分布
        return np.array([1/3, 1/3, 1/3])
    return np.array([p_home / total, p_draw / total, p_away / total])


def load_league_model(league_cfg_code):
    """加载指定联赛的独立模型（懒加载+缓存）
    返回 (home_model, away_model)，不存在返回 (None, None)
    """
    if league_cfg_code in _league_model_cache:
        return _league_model_cache[league_cfg_code]

    home_path = os.path.join(MODEL_DIR, f"league_{league_cfg_code}_home.pkl")
    away_path = os.path.join(MODEL_DIR, f"league_{league_cfg_code}_away.pkl")

    if not os.path.exists(home_path):
        _league_model_cache[league_cfg_code] = (None, None)
        return None, None

    import pickle
    with open(home_path, "rb") as f:
        h_model = pickle.load(f)
    
    a_model = None
    if os.path.exists(away_path):
        with open(away_path, "rb") as f:
            a_model = pickle.load(f)

    _league_model_cache[league_cfg_code] = (h_model, a_model)
    return h_model, a_model

def predict_match(feature_array, is_home_scene: bool = True):
    # 直接转为2维float numpy数组，彻底规避pandas dtype问题
    X = np.array(feature_array, dtype=np.float64).reshape(1, -1)
    # 【2026-07-23 整改3：空值、正负无穷兜底，避免推理异常】
    X = np.nan_to_num(X, nan=0.0, posinf=999.0, neginf=-999.0)
    # 【2026-07-23 整改4：特征维度强校验，提前拦截特征漏传/多传/错位】
    expect_feature_cnt = len(FEATURE_COLS)
    assert X.shape[1] == expect_feature_cnt, \
        f"特征数量不符，预期{expect_feature_cnt}维，实际{X.shape[1]}维，请核对上游特征构造"
    prob_lgb = home_model.predict(X, num_iteration=home_model.best_iteration)[0]
    prob_poisson = calc_poisson_1x2(feature_array)
    # 平局二分类预测（LightGBM原生格式，predict直接返回正类概率）
    prob_draw_binary = float(draw_binary_model.predict(X)[0])
    
    # 三模型融合：主胜/客胜 = LGB + 泊松；平局 = LGB + 泊松 + 平局专项
    prob_h = prob_lgb[0] * FUSION_WEIGHT_LGB + prob_poisson[0] * FUSION_WEIGHT_POISSON
    prob_a = prob_lgb[2] * FUSION_WEIGHT_LGB + prob_poisson[2] * FUSION_WEIGHT_POISSON
    prob_d = prob_lgb[1] * FUSION_WEIGHT_LGB + prob_poisson[1] * FUSION_WEIGHT_POISSON + prob_draw_binary * FUSION_WEIGHT_DRAW
    
    # 归一化
    tot = prob_h + prob_d + prob_a
    if tot < 1e-8:
        tot = 1.0
    prob_final = np.array([prob_h / tot, prob_d / tot, prob_a / tot])

    # 【2026-07-27 概率校准】Platt缩放，提升置信度可信度
    prob_final = apply_probability_calibration(prob_final.reshape(1, -1))[0]

    # 泊松进球期望值（用于比分/大小球预测）
    X_poi = X[:, POISSON_FEAT_IDX]
    h_lam = float(_predict_poisson_goals(X_poi, is_home=True)[0])
    a_lam = float(_predict_poisson_goals(X_poi, is_home=False)[0])
    h_lam = max(min(h_lam, 10.0), 0.3)
    a_lam = max(min(a_lam, 10.0), 0.3)

    label_idx = int(np.argmax(prob_final))
    label_map = {0:"主胜",1:"平局",2:"客胜"}
    pred_label = label_map[label_idx]
    confidence = round(float(np.max(prob_final)),4)

    # 比分预测 + 大小球
    score_matrix = calc_score_matrix(h_lam, a_lam)
    top_scores = get_top_scores(score_matrix, top_n=5)
    over_under = calc_over_under(h_lam, a_lam)
    expected_goals = {
        'home_expected': round(float(h_lam), 2),
        'away_expected': round(float(a_lam), 2),
        'total_expected': round(float(h_lam + a_lam), 2)
    }

    result = {
        "prob_home_win": round(float(prob_final[0]),4),
        "prob_draw": round(float(prob_final[1]),4),
        "prob_away_win": round(float(prob_final[2]),4),
        "predict_result": pred_label,
        "confidence": confidence,
        "expected_goals": expected_goals,
        "top_scores": top_scores,
        "over_under": over_under,
        "model_detail":{
            "lgb_prob": np.round(prob_lgb, 4).tolist(),
            "poisson_prob": np.round(prob_poisson, 4).tolist(),
            "fusion_weight": f"LGB {FUSION_WEIGHT_LGB:.0%} + 泊松 {FUSION_WEIGHT_POISSON:.0%} + 平局专项 {FUSION_WEIGHT_DRAW:.0%}"
        }
    }
    return result

if __name__ == "__main__":
    # 52维测试样本（英超）
    sample_feature = [
        8, 5, 42, 18,      # h5_gf, h5_ga, h5_shot, h5_shot_ot
        16, 11,             # h10_gf, h10_ga
        6, 7, 36, 14,      # a5_gf, a5_ga, a5_shot, a5_shot_ot
        13, 15,             # a10_gf, a10_ga
        0.25, 0.30,        # odds_draw_real, odds_lose_real
        5, 0.55, 0.25, 1.8, 1.2,  # h2h_cnt, win_rate, draw_rate, gf_avg, ga_avg
        1.5, 0.28, 0.48, 0.9, 1,  # prob_ratio_ha, draw_share, prob_max, entropy, home_fav
        0.3, 0.28, 0.25, 0.27,    # 平局率4维
        0, 1, 0, 0, 0,    # 联赛独热：英超E0
        1600, 1500, 100,  # ELO直接特征
        1.5, 1.2, 1.3, 1.1,  # ELO扩展特征
        5, 10, -3, -5,    # ELO趋势特征
        1.8, 1.5, 1.4, 1.6,  # 时间衰减近5场
        1.7, 1.4, 1.5, 1.5,  # 时间衰减近10场
    ]
    res = predict_match(sample_feature, is_home_scene=True)
    for k,v in res.items():
        print(f"{k}: {v}")

def predict_match_league(feature_array_29, league_cfg_code, is_home_scene: bool = True):
    """使用联赛独立模型预测（29维特征，不含联赛独热）
    若该联赛模型不存在，返回 None，调用方 fallback 到全局模型
    """
    h_model, _ = load_league_model(league_cfg_code)
    if h_model is None:
        return None

    X = np.array(feature_array_29, dtype=np.float64).reshape(1, -1)
    X = np.nan_to_num(X, nan=0.0, posinf=999.0, neginf=-999.0)

    expect_cnt = len(LEAGUE_FEATURE_COLS)
    assert X.shape[1] == expect_cnt, f'联赛模型特征数量不符，预期{expect_cnt}维'

    prob_lgb = h_model.predict_proba(X)[0]

    # 构造泊松模型输入（从47维特征中提取 + 联赛独热）
    league_onehot = {
        "EPL": [0, 1, 0, 0, 0],   # league_E0
        "BUN": [0, 0, 1, 0, 0],   # league_D1
        "LLA": [0, 0, 0, 0, 1],   # league_LLA
        "SER": [1, 0, 0, 0, 0],   # league_SER
        "LIG": [0, 0, 0, 1, 0],   # league_LIG
    }
    feat_47 = feature_array_47
    # 基础特征索引：h5_gf, h5_ga, a5_gf, a5_ga, h10_gf, h10_ga, a10_gf, a10_ga
    base_idx = [0, 1, 6, 7, 4, 5, 10, 11]
    poi_base = [feat_47[i] for i in base_idx]
    poi_feats = poi_base + league_onehot.get(league_cfg_code, [0,0,0,0,0])

    h_lam = max(float(_predict_poisson_goals(np.array([poi_feats]), is_home=True)[0]), 0.3)
    a_lam = max(float(_predict_poisson_goals(np.array([poi_feats]), is_home=False)[0]), 0.3)

    # 泊松转胜平负
    p_h = p_d = p_a = 0.0
    for h in range(9):
        for a in range(9):
            ph = (h_lam ** h) * math.exp(-h_lam) / math.factorial(h)
            pa = (a_lam ** a) * math.exp(-a_lam) / math.factorial(a)
            p = ph * pa
            if h > a: p_h += p
            elif h == a: p_d += p
            else: p_a += p
    tot = p_h + p_d + p_a
    prob_poisson = np.array([p_h/tot, p_d/tot, p_a/tot])

    # 三模型融合：主胜/客胜 = LGB + 泊松；平局 = LGB + 泊松 + 平局专项
    # 构造34维特征给平局二分类模型（29维 + 联赛独热5维）
    draw_feats = list(feature_array_29) + league_onehot.get(league_cfg_code, [0,0,0,0,0])
    prob_draw_binary = float(draw_binary_model.predict(np.array([draw_feats]))[0])
    
    prob_h = prob_lgb[0] * FUSION_WEIGHT_LGB + prob_poisson[0] * FUSION_WEIGHT_POISSON
    prob_a = prob_lgb[2] * FUSION_WEIGHT_LGB + prob_poisson[2] * FUSION_WEIGHT_POISSON
    prob_d = prob_lgb[1] * FUSION_WEIGHT_LGB + prob_poisson[1] * FUSION_WEIGHT_POISSON + prob_draw_binary * FUSION_WEIGHT_DRAW
    
    # 归一化
    tot = prob_h + prob_d + prob_a
    if tot < 1e-8:
        tot = 1.0
    prob_final = np.array([prob_h / tot, prob_d / tot, prob_a / tot])

    label_idx = int(np.argmax(prob_final))
    label_map = {0: '主胜', 1: '平局', 2: '客胜'}
    pred_label = label_map[label_idx]
    confidence = round(float(np.max(prob_final)), 4)

    # 比分预测 + 大小球
    score_matrix = calc_score_matrix(h_lam, a_lam)
    top_scores = get_top_scores(score_matrix, top_n=5)
    over_under = calc_over_under(h_lam, a_lam)
    expected_goals = {
        'home_expected': round(float(h_lam), 2),
        'away_expected': round(float(a_lam), 2),
        'total_expected': round(float(h_lam + a_lam), 2)
    }

    return {
        'prob_home_win': round(float(prob_final[0]), 4),
        'prob_draw': round(float(prob_final[1]), 4),
        'prob_away_win': round(float(prob_final[2]), 4),
        'predict_result': pred_label,
        'confidence': confidence,
        'model_type': 'league_independent',
        'league': league_cfg_code,
        'expected_goals': expected_goals,
        'top_scores': top_scores,
        'over_under': over_under,
        'model_detail': {
            'lgb_prob': np.round(prob_lgb, 4).tolist(),
            'poisson_prob': np.round(prob_poisson, 4).tolist(),
            'fusion_weight': f'LGB {FUSION_WEIGHT_LGB:.0%} + 泊松 {FUSION_WEIGHT_POISSON:.0%} + 平局专项 {FUSION_WEIGHT_DRAW:.0%}'
        }
    }
