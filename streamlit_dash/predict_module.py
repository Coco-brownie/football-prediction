import streamlit as st
import sys
import os
import numpy as np
import pandas as pd
import sqlite3

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, CUR_DIR)
from match_predict import predict_match, FEATURE_COLS, POISSON_FEAT_IDX, home_model
from feature_auto_build import build_feature_by_teams
from common.style import style_match_result_df
from team_mapping_v2 import LEAGUE_CFG, cfg_to_db_league
from common.usage_tracker import track

# ==================== SHAP 特征贡献解释 ====================
# 特征名中文映射
FEATURE_CN_MAP = {
    "h5_gf": "主队近5场进球",
    "h5_ga": "主队近5场失球",
    "h5_shot": "主队近5场射门",
    "h5_shot_ot": "主队近5场射正",
    "h10_gf": "主队近10场进球",
    "h10_ga": "主队近10场失球",
    "a5_gf": "客队近5场进球",
    "a5_ga": "客队近5场失球",
    "a5_shot": "客队近5场射门",
    "a5_shot_ot": "客队近5场射正",
    "a10_gf": "客队近10场进球",
    "a10_ga": "客队近10场失球",
    "odds_draw_real": "平局赔率偏离",
    "odds_lose_real": "客胜赔率偏离",
    "shot_on_diff": "射正率差异",
    "h2h_cnt": "交锋场次",
    "h2h_home_win_rate": "交锋主队胜率",
    "h2h_draw_rate": "交锋平局率",
    "h2h_home_gf_avg": "交锋主队进球",
    "h2h_home_ga_avg": "交锋主队失球",
    "prob_ratio_ha": "主客胜概率比",
    "prob_draw_share": "平局概率占比",
    "prob_max": "最大概率集中度",
    "prob_entropy": "概率不确定性",
    "prob_home_favorite": "主队热门度",
    "home_draw_rate_5": "主队近5场平局率",
    "home_draw_rate_10": "主队近10场平局率",
    "away_draw_rate_5": "客队近5场平局率",
    "away_draw_rate_10": "客队近10场平局率",
    "league_SER": "联赛-意甲",
    "league_E0": "联赛-英超",
    "league_D1": "联赛-德甲",
    "league_LIG": "联赛-法甲",
    "league_LLA": "联赛-西甲",
    # ELO直接特征
    "home_elo_before": "主队ELO评分",
    "away_elo_before": "客队ELO评分",
    "elo_diff_before": "ELO实力差距",
    # ELO扩展特征
    "h5_gf_elo_weighted": "主队加权进攻效率",
    "h5_ga_elo_weighted": "主队加权防守漏洞",
    "a5_gf_elo_weighted": "客队加权进攻效率",
    "a5_ga_elo_weighted": "客队加权防守漏洞",
    "home_w5_elo_trend": "主队近期状态走势",
    "home_w10_elo_trend": "主队中期状态走势",
    "away_w5_elo_trend": "客队近期状态走势",
    "away_w10_elo_trend": "客队中期状态走势",
    # 时间衰减特征
    "h5_gf_time_decay": "主队近5场进攻(时间加权)",
    "h5_ga_time_decay": "主队近5场防守(时间加权)",
    "a5_gf_time_decay": "客队近5场进攻(时间加权)",
    "a5_ga_time_decay": "客队近5场防守(时间加权)",
    "h10_gf_time_decay": "主队近10场进攻(时间加权)",
    "h10_ga_time_decay": "主队近10场防守(时间加权)",
    "a10_gf_time_decay": "客队近10场进攻(时间加权)",
    "a10_ga_time_decay": "客队近10场防守(时间加权)",
}

# SHAP解释器（延迟初始化）
_shap_explainer = None
_shap_available = None

def _get_shap_explainer():
    global _shap_explainer, _shap_available
    if _shap_available is not None:
        return _shap_explainer
    try:
        import shap
        _shap_explainer = shap.TreeExplainer(home_model)
        _shap_available = True
        return _shap_explainer
    except ImportError:
        _shap_available = False
        return None

def calc_feature_contributions(feature_array):
    """
    计算单场预测的特征贡献度（优先SHAP，降级用增益近似）
    返回: (pos_list, neg_list, summary_text)
    pos_list: [(cn_name, value), ...] 正向Top3
    neg_list: [(cn_name, value), ...] 负向Top3
    summary_text: 一句话总结
    """
    feat_names = FEATURE_COLS
    X = np.array(feature_array, dtype=np.float64).reshape(1, -1)
    
    # 尝试用 SHAP
    explainer = _get_shap_explainer()
    if explainer is not None:
        shap_values = explainer.shap_values(X)
        # 二分类取正类SHAP值
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        else:
            sv = shap_values[0]
        contributions = sv
    else:
        # 降级：用特征重要性 * 特征值符号 近似（不太准但能用）
        gain_imp = home_model.feature_importance(importance_type='gain')
        gain_imp = gain_imp / gain_imp.sum()
        # 简单用特征值正负方向乘权重近似
        contributions = gain_imp * np.sign(X[0])
    
    # 配对中文名
    feat_contrib = []
    for i, name in enumerate(feat_names):
        cn = FEATURE_CN_MAP.get(name, name)
        feat_contrib.append((cn, float(contributions[i])))
    
    # 按贡献值排序
    feat_contrib.sort(key=lambda x: x[1], reverse=True)
    
    # 正向Top3（值>0）
    pos_list = [x for x in feat_contrib if x[1] > 0][:3]
    # 负向Top3（值<0，按绝对值排）
    neg_list = sorted([x for x in feat_contrib if x[1] < 0], key=lambda x: x[1])[:3]
    
    # 生成总结
    if pos_list and neg_list:
        top_pos = pos_list[0][0]
        top_neg = neg_list[0][0]
        summary_text = f"核心驱动：{top_pos}；主要风险：{top_neg}"
    elif pos_list:
        summary_text = f"核心驱动：{pos_list[0][0]}"
    else:
        summary_text = "特征贡献均衡，无明显主导因素"
    
    return pos_list, neg_list, summary_text

# 预测模块独立数据库路径
DB_PATH = os.path.join(ROOT_DIR, "football.db")

# 缓存数据库球队数据，仅首次页面加载读取一次
@st.cache_data
def load_match_base_data(db_path):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM match_feature_final", conn)
    conn.close()
    return df

# 自动迁移：确保字段存在
def _ensure_predictions_schema(cursor):
    cursor.execute('PRAGMA table_info(predictions)')
    cols = [r[1] for r in cursor.fetchall()]
    if 'is_real_match' not in cols:
        cursor.execute('ALTER TABLE predictions ADD COLUMN is_real_match INTEGER DEFAULT 0')
    if 'user_name' not in cols:
        cursor.execute('ALTER TABLE predictions ADD COLUMN user_name TEXT DEFAULT NULL')

# 判断是否为真实赛程中的比赛（不卡日期，只要联赛+对阵在赛程表中存在就算）
def _check_is_real_match(cursor, match_date, home_team, away_team, league_code):
    cursor.execute("""
        SELECT COUNT(*) FROM match_schedule
        WHERE league_code = ? AND home_team = ? AND away_team = ?
    """, (league_code, home_team, away_team))
    return 1 if cursor.fetchone()[0] > 0 else 0

# 保存预测结果到数据库（去重：同联赛+同对阵+同日期+同用户则更新，否则插入）
def save_prediction_to_db(
    match_date, home_team, away_team, league_code,
    prob_home, prob_draw, prob_away, predict_result, confidence,
    predict_source="manual", user_name=None
):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 自动迁移字段
        _ensure_predictions_schema(cursor)
        
        # 判断是否真实比赛
        is_real = _check_is_real_match(cursor, match_date, home_team, away_team, league_code)
        
        # 查找是否已有同一场比赛的预测记录（同用户+同联赛+同对阵+同日期去重）
        cursor.execute("""
            SELECT id FROM predictions
            WHERE league_code = ? AND home_team = ? AND away_team = ? AND match_date = ?
              AND COALESCE(user_name, '') = COALESCE(?, '')
            ORDER BY predict_time DESC LIMIT 1
        """, (league_code, home_team, away_team, match_date, user_name))
        existing = cursor.fetchone()
        
        if existing:
            # 更新已有记录
            pred_id = existing[0]
            cursor.execute("""
                UPDATE predictions SET
                    prob_home = ?, prob_draw = ?, prob_away = ?,
                    predict_result = ?, confidence = ?, predict_source = ?,
                    predict_time = CURRENT_TIMESTAMP, is_real_match = ?,
                    is_verified = 0, actual_result = NULL, is_correct = NULL,
                    user_name = ?
                WHERE id = ?
            """, (prob_home, prob_draw, prob_away, predict_result, confidence,
                  predict_source, is_real, user_name, pred_id))
        else:
            # 插入新记录
            cursor.execute("""
                INSERT INTO predictions
                (match_date, home_team, away_team, league_code,
                 prob_home, prob_draw, prob_away, predict_result, confidence,
                 predict_source, is_real_match, user_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                match_date, home_team, away_team, league_code,
                prob_home, prob_draw, prob_away, predict_result, confidence,
                predict_source, is_real, user_name
            ))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"保存预测结果失败: {e}")
        return False

def render_match_predict_panel(cn_2_std=None, std_2_cn=None, user_name=None):
    st.subheader("🤖 赛事胜负智能预测")
    st.warning("💡 LightGBM三分类 + 泊松进球模型 融合预测，综合攻防数据、市场特征与进球分布，仅供体育数据分析与模型验证参考，严禁用于其他用途")

    # 默认主队主场（日常预测用不到切换）
    is_home = True

    st.divider()
    st.subheader("📋 选择联赛与对阵球队")

    # 1. 联赛选择
    league_keys = list(LEAGUE_CFG.keys())
    sel_league_cfg = st.selectbox(
        "选择联赛",
        options=league_keys,
        index=None,
        placeholder="请先选择联赛",
        format_func=lambda x: LEAGUE_CFG[x]["name"],
        key="predict_league_sel"
    )
    if not sel_league_cfg:
        st.info("👆 请先选择联赛，再选择对阵球队")
        return

    # 转成数据库编码
    curr_league_db = cfg_to_db_league(sel_league_cfg)

    try:
        # 调用缓存函数，不会重复查询数据库
        df_predict_all = load_match_base_data(DB_PATH)
    except Exception as e:
        st.error(f"赛事历史数据读取失败：{str(e)}")
        return

    # 2. 按联赛过滤球队（只显示当前联赛的球队，避免跨联赛不合理对阵）
    league_col = "league_code_raw" if "league_code_raw" in df_predict_all.columns else "league_code"
    home_col = "home_team_std" if "home_team_std" in df_predict_all.columns else "home_team"
    cn_col = "home_team" if "home_team_std" in df_predict_all.columns else None

    df_league = df_predict_all[df_predict_all[league_col] == curr_league_db]
    league_teams_std = sorted(list(set(df_league[home_col].dropna().tolist())))

    if not league_teams_std:
        st.warning("⚠️ 当前联赛暂无可用球队数据")
        return

    # 从数据直接构建 std -> 中文名 映射（100%覆盖，不依赖外部字典）
    std_to_cn_local = {}
    if cn_col:
        for _, row in df_league[[home_col, cn_col]].drop_duplicates().iterrows():
            std_to_cn_local[row[home_col]] = row[cn_col]
    # 合并外部映射作为补充
    if std_2_cn:
        std_to_cn_local.update(std_2_cn)

    # 球队下拉：值=标准英文名，显示=中文名
    def format_team(t):
        cn = std_to_cn_local.get(t, "")
        return f"{cn} ({t})" if cn else t

    c1, c2 = st.columns(2)
    pred_home = c1.selectbox(
        "选择主队", league_teams_std, 
        index=0,
        key="twin_model_home_team",
        format_func=format_team
    )
    # 客队默认选第二个，避免和主队重复
    away_default_idx = 1 if len(league_teams_std) > 1 else 0
    pred_away = c2.selectbox(
        "选择客队", league_teams_std, 
        index=away_default_idx,
        key="twin_model_away_team",
        format_func=format_team
    )

    # 禁止主队、客队选同一支球队
    if pred_home == pred_away:
        st.warning("⚠️ 主队和客队不能选择同一支球队，请重新挑选")
        return

    st.subheader("📊 输入市场概率（可选）")
    col1, col2, col3 = st.columns(3)
    with col1:
        home_odds_input = st.number_input(
            "主胜赔率",
            value=2.50,
            step=0.01,
            min_value=1.01,
            key="input_home_odds"
        )
    with col2:
        draw_odds_input = st.number_input(
            "平局赔率",
            value=3.30,
            step=0.01,
            min_value=1.01,
            key="input_draw_odds"
        )
    with col3:
        away_odds_input = st.number_input(
            "客胜赔率",
            value=3.00,
            step=0.01,
            min_value=1.01,
            key="input_away_odds"
        )

    # 自动计算去水后的真实概率
    inv_sum = 1.0 / home_odds_input + 1.0 / draw_odds_input + 1.0 / away_odds_input
    odds_draw_real = (1.0 / draw_odds_input) / inv_sum
    odds_lose_real = (1.0 / away_odds_input) / inv_sum

    # 高级设置（默认收起）
    with st.expander("⚙️ 高级设置", expanded=False):
        scene_type = st.radio("主客场场景", ["主队主场作战", "客队客场作战"], key="radio_scene_type")
        is_home = True if scene_type == "主队主场作战" else False
        st.caption("默认主队主场即可。切换为「客队客场」可模拟换位预测，日常预测用不到")

        st.divider()
        shot_diff = st.number_input(
            "射正差值（主队 - 客队，默认0表示双方相当）",
            value=0.0,
            step=0.1,
            key="input_shot_diff"
        )
        st.caption("球队近期射正次数差值，影响进球数预测精度，日常使用保持默认即可")

    run_btn = st.button(
        "🚀 预测",
        type="primary",
        use_container_width=True,
        key="btn_run_predict"
    )

    if run_btn:
        with st.spinner("正在抓取球队近期战绩、组装特征、模型推理计算..."):
            try:
                input_values = build_feature_by_teams(
                    df_predict_all, pred_home, pred_away,
                    odds_draw_real, odds_lose_real,
                    league_code=curr_league_db
                )
                pred_res = predict_match(input_values, is_home_scene=is_home)

                # 埋点：手动预测
                track('predict_manual',
                      action_detail=f'{pred_home} vs {pred_away}',
                      page_name='预测中心')

                # 保存预测结果到数据库
                from datetime import datetime
                today_str = datetime.now().strftime("%Y-%m-%d")
                save_prediction_to_db(
                    match_date=today_str,
                    home_team=pred_home,
                    away_team=pred_away,
                    league_code=curr_league_db,
                    prob_home=pred_res["prob_home_win"],
                    prob_draw=pred_res["prob_draw"],
                    prob_away=pred_res["prob_away_win"],
                    predict_result=pred_res["predict_result"],
                    confidence=pred_res["confidence"],
                    predict_source="manual",
                    user_name=user_name
                )
            except Exception as e:
                import traceback
                err_detail = traceback.format_exc()
                st.error(f"特征计算/模型推理失败：{str(e)}")
                st.code(err_detail, language="python")
                return

        st.success("✅ 预测计算完成")

        # 预测结果色块
        result_color = {
            "主胜": "#72d372",
            "平局": "#ffd966",
            "客胜": "#f88379"
        }
        st.markdown(f"""
        <div style="padding:16px;background:{result_color[pred_res['predict_result']]}20;border-left:6px solid {result_color[pred_res['predict_result']]};border-radius:8px;margin-bottom:12px">
            <h3 style="margin:0 0 4px 0">预测结果：{pred_res['predict_result']}</h3>
            <p style="margin:0">置信度：{pred_res['confidence']:.2%}</p>
        </div>
        """, unsafe_allow_html=True)

        # 置信度分级提示
        conf = pred_res["confidence"]
        if conf >= 0.65:
            st.success(f"🔴 高置信预测 — 把握较大，可重点参考")
        elif conf >= 0.50:
            st.warning(f"🟡 中等置信 — 有一定参考价值，建议结合其他分析")
        else:
            st.info(f"🟢 低置信 — 不确定性较大，仅供娱乐参考")

        # 💎 价值决策

        # 市场隐含概率（直接1/赔率）
        market_home = 1.0 / home_odds_input if home_odds_input > 0 else 0
        market_draw = 1.0 / draw_odds_input if draw_odds_input > 0 else 0
        market_away = 1.0 / away_odds_input if away_odds_input > 0 else 0

        # 模型概率
        model_home = pred_res["prob_home_win"]
        model_draw = pred_res["prob_draw"]
        model_away = pred_res["prob_away_win"]

        # 计算期望值和凯利
        def calc_value(model_p, odds):
            ev = model_p * odds - 1
            kelly = 0
            if odds > 1:
                kelly = (model_p * odds - 1) / (odds - 1)
            return ev, max(kelly, 0)

        ev_h, kelly_h = calc_value(model_home, home_odds_input)
        ev_d, kelly_d = calc_value(model_draw, draw_odds_input)
        ev_a, kelly_a = calc_value(model_away, away_odds_input)

        # 详细表格（二级折叠）
        with st.expander("📋 置信度对比分析（点击展开）", expanded=False):
            value_df = pd.DataFrame({
                "赛果": ["主胜", "平局", "客胜"],
            "模型概率": [f"{model_home:.1%}", f"{model_draw:.1%}", f"{model_away:.1%}"],
            "市场隐含": [f"{market_home:.1%}", f"{market_draw:.1%}", f"{market_away:.1%}"],
            "赔率": [f"{home_odds_input:.2f}", f"{draw_odds_input:.2f}", f"{away_odds_input:.2f}"],
            "期望收益": [f"{ev_h:+.1%}", f"{ev_d:+.1%}", f"{ev_a:+.1%}"],
            "凯利仓位": [f"{kelly_h:.1%}", f"{kelly_d:.1%}", f"{kelly_a:.1%}"],
        })
            st.dataframe(value_df, hide_index=True, use_container_width=True)

        # 价值评级 + 仓位建议
        ev_list = [("主胜", ev_h, kelly_h), ("平局", ev_d, kelly_d), ("客队胜", ev_a, kelly_a)]
        best = max(ev_list, key=lambda x: x[1])
        best_name, best_ev, best_kelly = best

        if best_ev > 0.08:
            value_level = "🟢 显著置信度优势"
            value_desc = f"模型显著看好{best_name}，置信度差值+{best_ev:.1%}，验证价值充足"
        elif best_ev > 0.03:
            value_level = "🟡 存在置信度优势"
            value_desc = f"{best_name}方向有正置信度差值，但空间有限，建议低权重验证"
        elif best_ev > 0:
            value_level = "🟡 微弱置信度优势"
            value_desc = f"{best_name}置信度差值略正，接近公允定价，可验证可不验证"
        else:
            value_level = "🔴 无置信度优势"
            value_desc = "三个方向置信度差值均为负，市场定价均高于模型判断，不建议验证"

        # 置信度对比卡片
        if best_ev > 0 and best_kelly > 0:
            position_text = f"保守 {best_kelly*0.25:.1%} ~ 激进 {best_kelly:.1%}"
            position_sub = f"推荐方向：{best_name}"
        else:
            position_text = "不建议"
            position_sub = "无验证价值"

        st.markdown(f"""
        <div style="padding:14px;background:#f0f7ff;border-radius:10px;margin-bottom:8px">
            <div style="display:flex;justify-content:space-between;align-items:center">
                <div>
                    <div style="font-size:0.8em;color:#666;margin-bottom:2px">置信度评级</div>
                    <div style="font-size:1.1em;font-weight:600">{value_level}</div>
                    <div style="font-size:0.75em;color:#888">{value_desc}</div>
                </div>
                <div style="text-align:right">
                    <div style="font-size:0.8em;color:#666;margin-bottom:2px">建议验证权重</div>
                    <div style="font-size:1.1em;font-weight:600">{position_text}</div>
                    <div style="font-size:0.75em;color:#888">{position_sub}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if best_ev > 0:
            st.caption("💡 保守型用1/4凯利，稳健型用1/2凯利，激进型可用全凯利")

        # 概率卡片（6张，预测方向高亮）
        home_not_lose = pred_res["prob_home_win"] + pred_res["prob_draw"]
        away_not_lose = pred_res["prob_away_win"] + pred_res["prob_draw"]
        has_winner = pred_res["prob_home_win"] + pred_res["prob_away_win"]
        pred_dir = pred_res["predict_result"]

        # 第一行：主胜/平局/客胜（预测方向高亮）
        prob_col1, prob_col2, prob_col3 = st.columns(3)
        prob_items = [
            ("主胜", pred_res["prob_home_win"], "主胜", "#e8f5e9", "#43a047"),
            ("平局", pred_res["prob_draw"], "平局", "#fff8e1", "#f9a825"),
            ("客胜", pred_res["prob_away_win"], "客胜", "#ffebee", "#e53935"),
        ]
        for col, (label, prob, dir_key, bg_color, border_color) in zip([prob_col1, prob_col2, prob_col3], prob_items):
            highlight = f"border:2px solid {border_color};" if dir_key == pred_dir else ""
            col.markdown(f"""
            <div style="text-align:center;padding:10px;background:{bg_color};border-radius:8px;{highlight}">
                <div style="font-size:0.8em;color:#666">{label}</div>
                <div style="font-size:1.3em;font-weight:700;color:#333">{prob:.2%}</div>
            </div>
            """, unsafe_allow_html=True)

        # 第二行：衍生概率
        der_col1, der_col2, der_col3 = st.columns(3)
        der_col1.metric("主队不败", f"{home_not_lose:.2%}", help="主胜 + 平局")
        der_col2.metric("客队不败", f"{away_not_lose:.2%}", help="客胜 + 平局")
        der_col3.metric("分胜负", f"{has_winner:.2%}", help="主胜 + 客胜")

        # ==================== SHAP 特征贡献解释 ====================
        st.markdown("### 🔍 模型判断依据")
        try:
            pos_list, neg_list, summary_text = calc_feature_contributions(input_values)
            
            col_pos, col_neg = st.columns(2)
            
            with col_pos:
                st.markdown("**🟢 正向因素**")
                if pos_list:
                    for cn, val in pos_list:
                        pct = val * 100
                        # 用进度条直观展示
                        st.markdown(f"""
                        <div style="margin-bottom:8px">
                            <div style="display:flex;justify-content:space-between;font-size:14px">
                                <span>{cn}</span>
                                <span style="color:#27ae60;font-weight:bold">+{abs(pct):.2f}%</span>
                            </div>
                            <div style="background:#e8f5e9;height:6px;border-radius:3px;margin-top:2px">
                                <div style="background:#27ae60;width:{min(abs(pct)*10, 100)}%;height:100%;border-radius:3px"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.caption("无明显正向因素")
            
            with col_neg:
                st.markdown("**🔴 负向因素**")
                if neg_list:
                    for cn, val in neg_list:
                        pct = val * 100
                        st.markdown(f"""
                        <div style="margin-bottom:8px">
                            <div style="display:flex;justify-content:space-between;font-size:14px">
                                <span>{cn}</span>
                                <span style="color:#e74c3c;font-weight:bold">-{abs(pct):.2f}%</span>
                            </div>
                            <div style="background:#fdecea;height:6px;border-radius:3px;margin-top:2px">
                                <div style="background:#e74c3c;width:{min(abs(pct)*10, 100)}%;height:100%;border-radius:3px;float:right"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.caption("无明显负向因素")
            
            st.caption(f"💡 {summary_text}")
            
        except Exception as e:
            st.caption(f"特征解释暂不可用：{str(e)}")

        # 比分预测（泊松模型）
        exp_h = pred_res["expected_goals"]["home_expected"]
        exp_a = pred_res["expected_goals"]["away_expected"]
        exp_col1, exp_col2 = st.columns(2)
        exp_col1.metric("主队预期进球", f"{exp_h:.2f}")
        exp_col2.metric("客队预期进球", f"{exp_a:.2f}")

        st.markdown("**最可能比分 TOP3**")
        score_cards = st.columns(3)
        for i, (score, prob) in enumerate(pred_res["top_scores"][:3]):
            with score_cards[i]:
                st.markdown(f"""
                <div style="text-align:center;padding:12px;background:#f8f9fa;border-radius:8px">
                    <div style="font-size:1.4em;font-weight:700;color:#333">{score}</div>
                    <div style="font-size:0.8em;color:#888;margin-top:2px">{prob:.1%}</div>
                </div>
                """, unsafe_allow_html=True)

        # 多档位大小球（二级折叠）
        with st.expander("📊 多档位大小球概率（点击展开）", expanded=False):
            ou_lines = [1.5, 2.5, 3.5, 4.5]
            ou_data = []
            for line in ou_lines:
                big = pred_res["over_under"][f"大球{line}"]
                small = pred_res["over_under"][f"小球{line}"]
                ou_data.append({
                    "档位": f"{line}球",
                    "大球": f"{big:.1%}",
                    "小球": f"{small:.1%}"
                })
            st.dataframe(pd.DataFrame(ou_data), use_container_width=True, hide_index=True, height=180)




        # 历史同赔率区间（二级折叠）
        with st.expander("📊 历史同赔率区间参考（点击展开）", expanded=False):
            st.markdown("**历史同赔率区间赛果分布**")
            user_h_prob = 1 - odds_draw_real - odds_lose_real  # 用户输入主胜去水概率
            prob_tolerance = 0.03  # ±3%概率范围

            # 从数据计算主胜真实概率（1 - 平真实 - 客胜真实）
            if "odds_draw_real" in df_predict_all.columns and "odds_lose_real" in df_predict_all.columns:
                real_h_prob = 1 - df_predict_all["odds_draw_real"] - df_predict_all["odds_lose_real"]
                real_d_prob = df_predict_all["odds_draw_real"]
            else:
                # fallback：用原始赔率列
                inv_sum = 1/df_predict_all["odds_win"] + 1/df_predict_all["odds_draw"] + 1/df_predict_all["odds_lose"]
                real_h_prob = (1/df_predict_all["odds_win"]) / inv_sum
                real_d_prob = (1/df_predict_all["odds_draw"]) / inv_sum

            mask_similar = (
                (real_h_prob.between(user_h_prob - prob_tolerance, user_h_prob + prob_tolerance)) &
                (real_d_prob.between(odds_draw_real - prob_tolerance, odds_draw_real + prob_tolerance))
            )
            similar_matches = df_predict_all[mask_similar]
            sim_total = len(similar_matches)

            if sim_total > 0:
                sim_home_win = (similar_matches["match_result"] == "主队胜").mean()
                sim_draw = (similar_matches["match_result"] == "平局").mean()
                sim_away_win = (similar_matches["match_result"] == "客队胜").mean()

                sim_c1, sim_c2, sim_c3, sim_c4 = st.columns(4)
                sim_c1.metric("历史样本数", f"{sim_total} 场")
                sim_c2.metric("实际主胜率", f"{sim_home_win:.1%}")
                sim_c3.metric("实际平局率", f"{sim_draw:.1%}")
                sim_c4.metric("实际客胜率", f"{sim_away_win:.1%}")
                st.caption(f"筛选条件：主胜/平局概率与当前赔率偏差均在 ±{prob_tolerance:.0%} 范围内")
            else:
                st.info("暂无完全匹配的历史样本，可适当放宽赔率范围参考")



        # 双模型概率明细（二级折叠）
        with st.expander("🔬 双模型概率明细（点击展开）", expanded=False):
            detail_df = pd.DataFrame({
                "赛果":["主胜","平局","客胜"],
                "LightGBM": pred_res["model_detail"]["lgb_prob"],
                "泊松模型": pred_res["model_detail"]["poisson_prob"],
            })
            styled_detail = style_match_result_df(detail_df)
            st.dataframe(styled_detail, use_container_width=True)
            st.caption(f"最终结果为融合输出（{pred_res['model_detail']['fusion_weight']}），验证集准确率64.7%")

