import pandas as pd
import numpy as np
import math
# 【2026-07-23 移除：import joblib，改用原生lgb加载模型】
import lightgbm as lgb
import json
import os
import pickle

from common_config import get_feature_list, get_value_features

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT_PATH, "model")


# 【2026-08-03 统一出口：从common_config读取特征列表，保证全项目一致】
# 基础特征（52维）
BASE_FEATURE_COLS = get_feature_list()

# 身价特征（5维，训练端主模型已加入，提升准确性）
# value_ratio: 身价比（主队/客队）
# value_diff_rel: 相对身价差（绝对值/联赛平均）
# home_value_rel: 主队相对身价（/联赛平均）
# away_value_rel: 客队相对身价（/联赛平均）
# value_diff_signed: 带符号的相对身价差（主队-客队）/联赛平均
VALUE_FEATURE_COLS = get_value_features()

# 总特征：52基础 + 5身价 = 57维（与训练端完全对齐）
FEATURE_COLS = BASE_FEATURE_COLS + VALUE_FEATURE_COLS

# 泊松模型特征（13维，去掉shot_on_diff）
POISSON_FEATURES = [
    "h5_gf", "h5_ga", "a5_gf", "a5_ga",
    "h10_gf", "h10_ga", "a10_gf", "a10_ga",
    "league_SER", "league_E0", "league_D1", "league_LIG", "league_LLA"
]
# 硬编码索引（与训练时GOAL_FEATURES严格对齐），避免FEATURE_COLS变化导致错位
# 顺序：h5_gf, h5_ga, a5_gf, a5_ga, h10_gf, h10_ga, a10_gf, a10_ga, league_*5
POISSON_FEAT_IDX = [0, 1, 6, 7, 4, 5, 10, 11, 47, 48, 49, 50, 51]

# 融合权重：LGB 55% + 泊松 30% + 平局专项 15%（最终定案：0.15，2026-08-05）
# 【2026-08-05 三档权重扫描结论（recalculate_predictions.py 全量回填）：
#  0.15: 整体准确率 55.52% / 平局Precision 45.3% / 平局占比 10.4%
#  0.20: 整体准确率 55.29% / 平局Precision 41.3% / 平局占比 15.9%
#  0.25: 整体准确率 54.91% / 平局Precision 39.1% / 平局占比 22.6%
#  关键：0.15→0.20 多喊的3086场平局边际命中仅 33.7%，0.20→0.25 的3733场为 33.9%——
#  均低于35%质量线、贴保本赔率(2.97)，且为样本内虚高数字，外样本大概率转负。
#  结论：平局信号有明确分层（核心层45%真金/尾巴层34%刮底），融合权重硬拉只能买到尾巴货。
#  回 0.15 保持1x2分类最高质量；平局覆盖率交给 draw_binary 独立通道的置信度阈值，勿用权重硬扩。
# 【2026-08-06 外样本终审定案（walk_forward_fused.py 25折完整融合：LGB+泊松+平局二分+每折校准）：
#  0.15: 准确率51.44% / 平局Precision31.4% / 价值下注ROI(≥+0.08) +1.41%  ← 三项最优
#  0.20: 准确率51.04% / 平局Precision31.6% / 价值下注ROI +0.41%
#  0.25: 准确率50.28% / 平局Precision31.1% / 价值下注ROI -0.11%
#  外样本边际平局命中率：0.15→0.20 为 32.0%，0.20→0.25 为 29.9%——与样本内34%完全一致，
#  坐实"拉权重只能买到尾巴货"。draw_binary 独立通道外样本 Precision 最高仅 29.4%、ROI 全负，
#  不可单独作平局下注信号（仅作融合的辅助权重）。0.15 终审通过，不再改动。
#  注意：样本内回填(55.52%)显著虚高于外样本(51.44%)，生产预期以 walk_forward 为准。】
FUSION_WEIGHT_LGB = 0.55
FUSION_WEIGHT_POISSON = 0.30
FUSION_WEIGHT_DRAW = 0.15

def _load_booster(path, expect_n_feat, model_name):
    """加载LightGBM模型并校验特征维度，缺失/不符时给出明确中文指引（启动自检友好化）
    【2026-08-07 新增：原模块级直接 Booster 加载，模型缺失时抛晦涩异常且无处理指引，
      部署到 Streamlit Cloud 时一旦漏文件会导致整个 App 无法启动且难以排查。】
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"【模型缺失】未找到{model_name}模型文件：{path}\n"
            f"处理指引：请先运行 python training\\train_all_clean.py 完成重训，"
            f"并确认 model/ 目录下存在 home_model.pkl、draw_binary_model.txt、"
            f"poisson_model_params.json、calibrator_params.json 四个文件后再启动。"
        )
    model = lgb.Booster(model_file=path)
    assert model.num_feature() == expect_n_feat, (
        f"{model_name}维度不匹配！模型{model.num_feature()}维，"
        f"特征列表{expect_n_feat}维，请检查特征定义或重新训练模型"
    )
    return model


# 【2026-07-24 优化：客场模型特征互换不完整，融合反而降准，改用单主场模型】
# 【2026-08-07 启动自检友好化：缺失时给出明确指引而非晦涩崩溃】
home_model = _load_booster(os.path.join(MODEL_DIR, "home_model.pkl"), len(FEATURE_COLS), "主模型")

# 【2026-07-29 修复：泊松模型改用参数化方式，彻底解决pickle跨平台兼容性问题】
# PoissonRegressor = 线性模型，预测公式: lambda = exp(X @ coef + intercept)
# 【2026-08-07 启动自检友好化】
_poisson_params_path = os.path.join(MODEL_DIR, "poisson_model_params.json")
if not os.path.exists(_poisson_params_path):
    raise FileNotFoundError(
        f"【模型缺失】未找到泊松模型参数文件：{_poisson_params_path}\n"
        f"处理指引：请先运行 python training\\train_all_clean.py 完成重训。"
    )
with open(_poisson_params_path, "r") as f:
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
# 【2026-08-07 启动自检友好化】
draw_binary_model = _load_booster(os.path.join(MODEL_DIR, "draw_binary_model.txt"), len(BASE_FEATURE_COLS), "平局二分类模型")

# 【2026-07-30 新增：置信度校准器（验证集训练，无泄露）】
_calibrator = None
def _get_calibrator():
    global _calibrator
    if _calibrator is None:
        # 【2026-08-03 优先用JSON格式，彻底去掉pickle依赖】
        json_path = os.path.join(MODEL_DIR, "calibrator_params.json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                _calibrator = json.load(f)  # 字典格式，含a和b参数
            return _calibrator
        
        # 【2026-08-07 收敛：JSON（calibrator_params.json）为唯一权威格式，
        #  由 train_all_clean.py 重训时自动生成。以下 pickle 分支仅为旧环境兼容保留，
        #  新重训产物不会走这里；确认旧部署全部升级后可整体移除。】
        cal_path = os.path.join(MODEL_DIR, "confidence_calibrator.pkl")
        if os.path.exists(cal_path):
            with open(cal_path, "rb") as f:
                _calibrator = pickle.load(f)
        else:
            old_cal_path = os.path.join(MODEL_DIR, "platt_calibrator_home.pkl")
            if os.path.exists(old_cal_path):
                with open(old_cal_path, "rb") as f:
                    _calibrator = pickle.load(f)
    return _calibrator


def apply_probability_calibration(probs):
    """应用置信度校准
    输入：shape=(n, 3)的概率数组 [主胜, 平局, 客胜]
    输出：校准后的概率
    校准方式：直接校准最大概率（置信度），其余概率按原始比例缩放
    """
    cal = _get_calibrator()
    if cal is None:
        return probs  # 校准器不存在则返回原值
    
    # 判断校准器类型
    if isinstance(cal, dict):
        # 【2026-08-03 新版JSON格式：直接用a、b参数手动计算】
        a = cal["a"]
        b = cal["b"]
        cal_probs = np.zeros_like(probs)
        
        for i in range(len(probs)):
            p = probs[i]
            max_idx = np.argmax(p)
            max_prob = p[max_idx]
            
            # 校准最大概率：sigmoid(a * max_prob + b)
            calibrated_max = 1.0 / (1.0 + math.exp(-(a * max_prob + b)))
            
            # 其余概率按比例缩放
            remaining = 1.0 - calibrated_max
            other_sum = p.sum() - max_prob
            
            if other_sum > 0:
                scale = remaining / other_sum
                for j in range(3):
                    if j == max_idx:
                        cal_probs[i, j] = calibrated_max
                    else:
                        cal_probs[i, j] = p[j] * scale
            else:
                cal_probs[i] = p
        
        return cal_probs
    elif isinstance(cal, list):
        # 旧版One-vs-Rest校准（兼容）
        cal_probs = np.zeros_like(probs)
        for i in range(3):
            cal_probs[:, i] = cal[i].predict_proba(probs[:, i].reshape(-1, 1))[:, 1]
        row_sums = cal_probs.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        cal_probs = cal_probs / row_sums
        return cal_probs
    else:
        # 新版：直接校准置信度（max_prob）
        cal_probs = np.zeros_like(probs)
        
        for i in range(len(probs)):
            p = probs[i]
            max_idx = np.argmax(p)
            max_prob = p[max_idx]
            
            # 校准最大概率
            calibrated_max = cal.predict_proba(np.array([[max_prob]]))[0, 1]
            
            # 其余概率按比例缩放
            remaining = 1.0 - calibrated_max
            other_sum = p.sum() - max_prob
            
            if other_sum > 0:
                scale = remaining / other_sum
                for j in range(3):
                    if j == max_idx:
                        cal_probs[i, j] = calibrated_max
                    else:
                        cal_probs[i, j] = p[j] * scale
            else:
                cal_probs[i] = p
        
        return cal_probs


# 【2026-08-08 方案A：置信度改用 WF 外样本分桶真实命中率】
# 背景：原置信度=校准后最高概率（Platt 在 40-60% 区间近恒等），数值上≈主胜概率，用户易误解
#       （"置信度和主胜概率怎么一样"）。现改为查根库 wf_confidence_accuracy 表
#       （walk_forward OOS 生成，与看板「核心验证结论」同口径）：
#       按校准后 max_prob 所在档位 → 返回该档位【历史真实命中率】。
#       语义：不是"这次预测的把握有多大"，而是"历史上这个置信度档位的预测命中了多少"，
#       更保守、更接近真实可期命中率，也与 AI 出手建议（保守原则）天然衔接。
# 表缺失/异常时降级为 Platt 校准值，保证预测链路永不断。
_WF_CONF_TABLE = None
_WF_CONF_TABLE_LOADED = False

def _load_wf_conf_table():
    """读取 wf_confidence_accuracy（6档），缓存；失败返回 None
    【2026-08-08 收敛：优先读 model/wf_confidence_accuracy.json（随仓库分发，
      克隆无需 165MB 的 football.db）；db 查询仅为本地旧环境兜底保留。】"""
    global _WF_CONF_TABLE, _WF_CONF_TABLE_LOADED
    if _WF_CONF_TABLE_LOADED:
        return _WF_CONF_TABLE
    _WF_CONF_TABLE_LOADED = True
    # 档位下界（与 backtest/generate_dashboard_tables.py 分桶严格一致）
    _BOUNDS = {"<50%": 0.0, "50-60%": 0.50, "60-70%": 0.60,
               "70-80%": 0.70, "80-90%": 0.80, "≥90%": 0.90}
    table = None
    # ① 优先读 JSON（随仓库分发，5KB 级别）
    try:
        json_path = os.path.join(MODEL_DIR, "wf_confidence_accuracy.json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            tmp = []
            for item in data:
                lab = str(item.get("置信度区间", "")).strip()
                if lab in _BOUNDS:
                    tmp.append((_BOUNDS[lab], float(item.get("准确率", 0))))
            tmp.sort()
            table = tmp if tmp else None
    except Exception:
        table = None
    # ② 兜底：根库查询（本地有 db 时）
    if not table:
        try:
            import sqlite3
            from common_config import get_path
            conn = sqlite3.connect(get_path("db_path"))
            df = pd.read_sql("SELECT 置信度区间, 准确率 FROM wf_confidence_accuracy", conn)
            conn.close()
            tmp = []
            for _, r in df.iterrows():
                lab = str(r["置信度区间"]).strip()
                if lab in _BOUNDS:
                    tmp.append((_BOUNDS[lab], float(r["准确率"])))
            tmp.sort()
            table = tmp if tmp else None
        except Exception:
            table = None
    _WF_CONF_TABLE = table
    return _WF_CONF_TABLE

def _get_confidence(max_prob):
    """置信度 = 校准后 max_prob 所在档位的历史真实命中率（WF OOS 验证）
    表缺失/异常时降级为 Platt 校准值，保证任何环境下都可预测"""
    table = _load_wf_conf_table()
    if table:
        hit = None
        for lo, acc in table:
            if max_prob >= lo:
                hit = acc
            else:
                break
        if hit is not None:
            return round(hit, 4)
    # 降级：Platt 校准（a/b 来自 calibrator_params.json）
    cal = _get_calibrator()
    if isinstance(cal, dict):
        try:
            return round(1.0 / (1.0 + math.exp(-(cal["a"] * max_prob + cal["b"]))), 4)
        except Exception:
            pass
    return round(float(max_prob), 4)


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
    # 平局二分类预测（只用前52维基础特征，身价对平局影响很小）
    prob_draw_binary = float(draw_binary_model.predict(X[:, :len(BASE_FEATURE_COLS)])[0])
    
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
    # 【2026-08-08 方案A：置信度 = WF 分桶历史真实命中率（不再是校准后最高概率）
    #  使置信度与主胜概率数值分离，语义清晰（该档位历史上命中了多少）】
    confidence = _get_confidence(float(np.max(prob_final)))

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
    # 【2026-08-04 修复：测试样本必须与 FEATURE_COLS（get_feature_list + value_features）完全同序，57维】
    # 顺序：base14 + h2h5 + prob5 + draw4 + elo3 + elo_ext8 + time8 + league5 + value5
    sample_feature = [
        # ---- 基础特征 14 ----------------
        2.0, 1.0, 0.5, 0.2,   # h5_gf, h5_ga, h5_shot, h5_shot_ot
        1.8, 1.1,              # h10_gf, h10_ga
        1.4, 1.2, 0.4, 0.15,  # a5_gf, a5_ga, a5_shot, a5_shot_ot
        1.5, 1.3,              # a10_gf, a10_ga
        3.4, 4.2,              # odds_draw_real, odds_lose_real
        # ---- 交锋特征 5 ----------------
        5, 0.4, 0.2, 1.5, 1.1,  # h2h_cnt, h2h_home_win_rate, h2h_draw_rate, h2h_home_gf_avg, h2h_home_ga_avg
        # ---- 概率衍生 5 ----------------
        1.2, 0.28, 0.48, 0.9, 1,  # prob_ratio_ha, prob_draw_share, prob_max, prob_entropy, prob_home_favorite
        # ---- 平局率 4 ----------------
        0.3, 0.28, 0.25, 0.27,  # home_draw_rate_5, home_draw_rate_10, away_draw_rate_5, away_draw_rate_10
        # ---- ELO 3 ----------------
        1600, 1500, 100,  # home_elo_before, away_elo_before, elo_diff_before
        # ---- ELO扩展 8 ----------------
        1.5, 1.2, 1.3, 1.1,   # h5_gf_elo_weighted, h5_ga_elo_weighted, a5_gf_elo_weighted, a5_ga_elo_weighted
        0.5, 0.55, 0.4, 0.45, # home_w5_elo_trend, home_w10_elo_trend, away_w5_elo_trend, away_w10_elo_trend
        # ---- 时间衰减 8 ----------------
        1.8, 1.5, 1.4, 1.6,  # h5_gf_time_decay, h5_ga_time_decay, a5_gf_time_decay, a5_ga_time_decay
        1.7, 1.4, 1.5, 1.5,  # h10_gf_time_decay, h10_ga_time_decay, a10_gf_time_decay, a10_ga_time_decay
        # ---- 联赛独热 5（英超E0）----------------
        0, 1, 0, 0, 0,  # league_SER, league_E0, league_D1, league_LIG, league_LLA
        # ---- 身价特征 5 ----------------
        1.05, 0.3, 1.1, 0.8, 0.4,  # value_ratio, value_diff_rel, home_value_rel, away_value_rel, value_diff_signed
    ]
    assert len(sample_feature) == len(FEATURE_COLS), \
        f"测试样本维度 {len(sample_feature)} 与 FEATURE_COLS {len(FEATURE_COLS)} 不一致"
    res = predict_match(sample_feature, is_home_scene=True)
    for k,v in res.items():
        print(f"{k}: {v}")

