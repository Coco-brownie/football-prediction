"""
赛事日历模块
- 赛程浏览、日期/联赛筛选
- 单场比赛预测
"""
import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta

# 导入项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from match_predict import predict_match, FEATURE_COLS
from streamlit_dash.feature_auto_build import build_feature_by_teams
from feature_auto_build import get_team_recent_stats
from team_mapping_v2 import LEAGUE_TEAM_MAP, CFG_2_DB_CODE
from common.usage_tracker import track
from streamlit_dash.predict_module import save_prediction_to_db

# 数据库编码 -> 配置编码反向映射
DB_2_CFG = {v: k for k, v in CFG_2_DB_CODE.items()}

# ========== 配置 ==========
SCRIPT_PATH = os.path.abspath(__file__)
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_PATH))
DB_PATH = os.path.join(ROOT_DIR, "football.db")

# 联赛显示配置
LEAGUE_DISPLAY = {
    "E0": {"name": "英超", "color": "#37003c"},
    "D1": {"name": "德甲", "color": "#d20515"},
    "LLA": {"name": "西甲", "color": "#ee8707"},
    "SER": {"name": "意甲", "color": "#008fd7"},
    "LIG": {"name": "法甲", "color": "#009045"},
    "UCL": {"name": "欧冠", "color": "#0015a8"},
}

# 联赛编码 → 独热位索引（和训练时一致）
LEAGUE_ONEHOT_IDX = {
    "SER": 0,
    "E0": 1,
    "D1": 2,
    "LIG": 3,
    "LLA": 4,
}

# 降级预测默认值（赔率、射门数据缺失时填充）
# 降级预测默认赔率（原始赔率值，会自动去水转概率）
DEFAULT_HOME_ODDS = 2.5
DEFAULT_DRAW_ODDS = 3.5
DEFAULT_AWAY_ODDS = 3.0
DEFAULT_SHOT_DIFF = 0.0
DEFAULT_SHOT_DIFF = 0.0

LEAGUE_OPTIONS = list(LEAGUE_DISPLAY.keys())
LEAGUE_NAMES = {k: v["name"] for k, v in LEAGUE_DISPLAY.items()}


@st.cache_data(ttl=86400)
def load_schedule_data():
    """加载全部赛程数据（缓存1天），自动转换中文队名"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM match_schedule", conn)
    conn.close()
    df["match_date"] = pd.to_datetime(df["match_date"])

    # 英文队名转中文
    def get_cn_name(league_db_code, eng_name):
        cfg_code = DB_2_CFG.get(league_db_code)
        if cfg_code and cfg_code in LEAGUE_TEAM_MAP:
            team_map = LEAGUE_TEAM_MAP[cfg_code]
            if eng_name in team_map:
                return team_map[eng_name][1]
        return eng_name  # 找不到映射保留原名

    df["home_team_cn"] = df.apply(lambda r: get_cn_name(r["league_code"], r["home_team"]), axis=1)
    df["away_team_cn"] = df.apply(lambda r: get_cn_name(r["league_code"], r["away_team"]), axis=1)

    return df


@st.cache_data(ttl=3600)
def load_historical_data():
    """加载历史比赛数据（用于计算球队近期战绩，缓存1小时）"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM match_feature_final", conn)
    conn.close()
    df["match_date"] = pd.to_datetime(df["match_date"])
    return df


def build_pred_feature(df_hist, home_team, away_team, league_code):
    """为单场比赛构建预测特征（34维完整特征，赔率用默认值降级填充）"""
    # 赔率去水转真实概率
    inv_sum = 1.0 / DEFAULT_HOME_ODDS + 1.0 / DEFAULT_DRAW_ODDS + 1.0 / DEFAULT_AWAY_ODDS
    odds_draw_real = (1.0 / DEFAULT_DRAW_ODDS) / inv_sum
    odds_lose_real = (1.0 / DEFAULT_AWAY_ODDS) / inv_sum

    # 复用统一的特征构建函数，保证34维特征完整对齐
    return build_feature_by_teams(
        df_hist, home_team, away_team,
        odds_draw_real, odds_lose_real,
        DEFAULT_SHOT_DIFF, league_code
    )


def can_predict_match(row, df_hist):
    """判断这场比赛是否可以预测（主客队都有历史数据）"""
    home_cn = row.get("home_team_cn")
    away_cn = row.get("away_team_cn")
    league_code = row["league_code"]

    # 欧冠暂时不支持预测（历史数据中没有欧冠）
    if league_code == "UCL":
        return False, "暂无欧冠历史数据"

    # 检查中文名映射
    if pd.isna(home_cn) or home_cn == row["home_team"]:
        return False, "主队暂无映射"
    if pd.isna(away_cn) or away_cn == row["away_team"]:
        return False, "客队暂无映射"

    # 检查历史数据中是否有这两支球队
    home_col = "home_team"
    away_col = "away_team"
    has_home = len(df_hist[df_hist[home_col] == home_cn]) > 0
    has_away = len(df_hist[df_hist[away_col] == away_cn]) > 0

    if not has_home or not has_away:
        return False, "历史数据不足"

    return True, ""


def get_week_range(ref_date=None):
    """获取本周周一到周日的日期范围"""
    if ref_date is None:
        ref_date = datetime.now().date()
    # 周一为一周开始
    start = ref_date - timedelta(days=ref_date.weekday())
    end = start + timedelta(days=6)
    return start, end


def render_schedule_calendar(user_name=None):
    """渲染赛事日历主面板"""
    st.subheader("📅 赛事日历")

    # 加载数据
    df_all = load_schedule_data()
    df_hist = load_historical_data()

    # ========== 筛选区 ==========
    col1, col2, col3 = st.columns([2, 3, 1])

    with col1:
        # 快捷切换
        quick_select = st.radio(
            "时间范围",
            ["今日", "本周", "自定义"],
            horizontal=True,
            label_visibility="collapsed"
        )

    with col2:
        if quick_select == "今日":
            today = datetime.now().date()
            start_date, end_date = today, today
            st.date_input(
                "日期范围",
                value=(start_date, end_date),
                disabled=True,
                label_visibility="collapsed"
            )
        elif quick_select == "本周":
            start_date, end_date = get_week_range()
            st.date_input(
                "日期范围",
                value=(start_date, end_date),
                disabled=True,
                label_visibility="collapsed"
            )
        else:
            min_d = df_all["match_date"].min().date()
            max_d = df_all["match_date"].max().date()
            default_start, default_end = get_week_range()
            date_range = st.date_input(
                "选择日期范围",
                value=(default_start, default_end),
                min_value=min_d,
                max_value=max_d,
                label_visibility="collapsed"
            )
            if len(date_range) == 2:
                start_date, end_date = date_range
            else:
                start_date, end_date = default_start, default_end

    with col3:
        # 联赛筛选
        selected_leagues = st.multiselect(
            "联赛",
            options=LEAGUE_OPTIONS,
            default=LEAGUE_OPTIONS,
            format_func=lambda x: LEAGUE_NAMES[x],
            label_visibility="collapsed"
        )

    st.divider()

    # ========== 数据筛选 ==========
    mask_date = (df_all["match_date"].dt.date >= start_date) & (df_all["match_date"].dt.date <= end_date)
    mask_league = df_all["league_code"].isin(selected_leagues)
    df_filter = df_all[mask_date & mask_league].copy()

    if len(df_filter) == 0:
        st.info("📭 当前筛选条件下暂无比赛")
        return

    # 按日期分组
    df_filter = df_filter.sort_values(["match_date", "match_time"])
    grouped = df_filter.groupby(df_filter["match_date"].dt.date)

    total_matches = len(df_filter)
    played = len(df_filter[df_filter["status"] == "Played"])
    upcoming = total_matches - played

    # 统计概览
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    stat_col1.metric("总场次", total_matches)
    stat_col2.metric("已完赛", played)
    stat_col3.metric("未开赛", upcoming)

    # 批量预测按钮
    batch_pred_key = "batch_pred_result"
    if stat_col4.button("🔮 批量预测", type="primary", use_container_width=True):
        batch_results = []
        track('predict_batch',
              action_detail=f'{len(df_filter)}场赛程',
              page_name='预测中心')
        for _, row in df_filter.iterrows():
            if row["status"] == "Played":
                continue
            can_pred, reason = can_predict_match(row, df_hist)
            if not can_pred:
                continue

            home_name = row["home_team_cn"] if pd.notna(row["home_team_cn"]) else row["home_team"]
            away_name = row["away_team_cn"] if pd.notna(row["away_team_cn"]) else row["away_team"]

            try:
                feature = build_pred_feature(df_hist, home_name, away_name, row["league_code"])
                result = predict_match(feature)
                batch_results.append({
                    "比赛日期": str(row["match_date"])[:10],
                    "联赛": LEAGUE_NAMES.get(row["league_code"], row["league_code"]),
                    "主队": home_name,
                    "客队": away_name,
                    "预测赛果": result["predict_result"],
                    "置信度": result["confidence"],
                    "主胜概率": result["prob_home_win"],
                    "平局概率": result["prob_draw"],
                    "客胜概率": result["prob_away_win"],
                })

                # 写入数据库（统一走保存函数，自动去重+真实比赛标记）
                try:
                    save_prediction_to_db(
                        match_date=str(row["match_date"])[:10],
                        home_team=home_name,
                        away_team=away_name,
                        league_code=row["league_code"],
                        prob_home=result["prob_home_win"],
                        prob_draw=result["prob_draw"],
                        prob_away=result["prob_away_win"],
                        predict_result=result["predict_result"],
                        confidence=result["confidence"],
                        predict_source="schedule_batch",
                        user_name=user_name
                    )
                except:
                    pass
            except:
                continue

        st.session_state[batch_pred_key] = batch_results

    st.markdown("---")

    # 批量预测结果展示
    if batch_pred_key in st.session_state and len(st.session_state[batch_pred_key]) > 0:
        st.subheader(f"📊 批量预测结果（共 {len(st.session_state[batch_pred_key])} 场）")
        df_batch = pd.DataFrame(st.session_state[batch_pred_key])

        # 置信度筛选
        col_f1, col_f2, col_f3 = st.columns([2, 2, 1])
        min_conf = col_f1.slider("最低置信度筛选", 0.35, 0.80, 0.45, 0.05, format="%.0f")
        sort_by = col_f2.selectbox("排序方式", ["置信度从高到低", "比赛日期", "联赛"], index=0)
        col_f3.metric("筛选后场次", f"{len(df_batch[df_batch['置信度'] >= min_conf])}场")

        df_show = df_batch[df_batch['置信度'] >= min_conf].copy()

        # 排序
        if sort_by == "置信度从高到低":
            df_show = df_show.sort_values("置信度", ascending=False)
        elif sort_by == "比赛日期":
            df_show = df_show.sort_values("比赛日期")
        elif sort_by == "联赛":
            df_show = df_show.sort_values("联赛")

        # 格式化百分比列
        for col in ["置信度", "主胜概率", "平局概率", "客胜概率"]:
            df_show[col] = df_show[col].apply(lambda x: f"{x:.1%}")

        # 赛果颜色标注
        def color_result(val):
            if val == "主胜": return "color: #2ecc71; font-weight: bold"
            elif val == "客胜": return "color: #e74c3c; font-weight: bold"
            elif val == "平局": return "color: #f39c12; font-weight: bold"
            return ""

        styled = df_show.style.map(color_result, subset=["预测赛果"])
        st.dataframe(styled, use_container_width=True, hide_index=True, height=420)

        # 统计
        high_conf = len(df_batch[df_batch['置信度'] >= 0.6])
        very_high = len(df_batch[df_batch['置信度'] >= 0.7])
        st.caption(f"💡 ≥60%置信度 {high_conf} 场（预期准确率~85%），≥70%置信度 {very_high} 场（预期准确率~91%）")
        st.caption("⚠️ 赛程预测缺少赔率和射门数据，为降级预测，仅供参考")
        st.markdown("---")

    # ========== 按日期展示 ==========
    for date_val, day_df in grouped:
        day_name = date_val.strftime("%Y-%m-%d %A")
        day_count = len(day_df)

        with st.expander(f"📆 {day_name}（{day_count} 场）", expanded=True):
            for _, row in day_df.iterrows():
                render_match_row(row, df_hist)


def render_match_row(row, df_hist):
    """渲染单场比赛行"""
    league_info = LEAGUE_DISPLAY.get(row["league_code"], {"name": row["league_code"], "color": "#666"})
    match_id = int(row["id"])
    pred_key = f"schedule_pred_{match_id}"

    col_time, col_home, col_vs, col_away, col_status = st.columns([1, 3, 1, 3, 2])

    # 时间
    col_time.markdown(f"<div style='text-align:center;color:#666;font-size:14px;'>{row['match_time']}</div>", unsafe_allow_html=True)

    # 主队
    home_name = row["home_team_cn"] if pd.notna(row.get("home_team_cn")) and row["home_team_cn"] != row["home_team"] else row["home_team"]
    col_home.markdown(f"<div style='text-align:right;font-size:16px;font-weight:500;'>{home_name}</div>", unsafe_allow_html=True)

    # VS
    col_vs.markdown("<div style='text-align:center;color:#999;'>VS</div>", unsafe_allow_html=True)

    # 客队
    away_name = row["away_team_cn"] if pd.notna(row.get("away_team_cn")) and row["away_team_cn"] != row["away_team"] else row["away_team"]
    col_away.markdown(f"<div style='text-align:left;font-size:16px;font-weight:500;'>{away_name}</div>", unsafe_allow_html=True)

    # 状态/比分
    is_played = row["status"] == "Played" and pd.notna(row.get("result"))
    if is_played:
        col_status.markdown(
            f"<div style='text-align:center;'>"
            f"<span style='background:#e8f5e9;color:#2e7d32;padding:2px 8px;border-radius:4px;font-size:13px;'>"
            f"{row['result']}"
            f"</span></div>",
            unsafe_allow_html=True
        )
    else:
        col_status.markdown(
            f"<div style='text-align:center;'>"
            f"<span style='background:#fff3e0;color:#e65100;padding:2px 8px;border-radius:4px;font-size:12px;'>"
            f"未开赛"
            f"</span></div>",
            unsafe_allow_html=True
        )

    # 联赛标签 + 轮次
    st.caption(
        f"<span style='color:{league_info['color']};font-weight:500;'>{league_info['name']}</span> "
        f"· 第 {int(row['matchday'])} 轮"
        + (f" · {row['stadium']}" if pd.notna(row.get('stadium')) else ""),
        unsafe_allow_html=True
    )

    # 预测按钮（仅未开赛且可预测的比赛）
    if not is_played:
        can_pred, reason = can_predict_match(row, df_hist)
        btn_col1, btn_col2 = st.columns([1, 5])

        if can_pred:
            if btn_col1.button("🔮 预测", key=f"btn_pred_{match_id}", type="primary"):
                # 构建特征并预测
                feature = build_pred_feature(df_hist, home_name, away_name, row["league_code"])
                result = predict_match(feature)
                st.session_state[pred_key] = result

                # 保存预测结果到数据库（统一走保存函数，自动去重+真实比赛标记）
                try:
                    save_prediction_to_db(
                        match_date=str(row["match_date"])[:10],
                        home_team=home_name,
                        away_team=away_name,
                        league_code=row["league_code"],
                        prob_home=result["prob_home_win"],
                        prob_draw=result["prob_draw"],
                        prob_away=result["prob_away_win"],
                        predict_result=result["predict_result"],
                        confidence=result["confidence"],
                        predict_source="schedule",
                        user_name=user_name
                    )
                except Exception as e:
                    pass  # 保存失败不影响展示

            # 展示预测结果
            if pred_key in st.session_state:
                pred_result = st.session_state[pred_key]
                st.markdown(
                    f"<div style='background:#f0f7ff;padding:10px;border-radius:8px;margin:8px 0;'>"
                    f"<b>预测结果：{pred_result['predict_result']}</b> "
                    f"<span style='color:#666;'>（置信度 {pred_result['confidence']:.1%}）</span>"
                    f"<br><span style='font-size:13px;color:#888;'>"
                    f"主胜 {pred_result['prob_home_win']:.1%} · "
                    f"平局 {pred_result['prob_draw']:.1%} · "
                    f"客胜 {pred_result['prob_away_win']:.1%}"
                    f"</span>"
                    f"<br><span style='font-size:11px;color:#aaa;'>⚠️ 缺少赔率数据，为降级预测，仅供参考</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
        else:
            btn_col1.button(f"🔮 预测", key=f"btn_pred_{match_id}", disabled=True, help=reason)

    st.markdown("<div style='height:1px;background:#f0f0f0;margin:8px 0;'></div>", unsafe_allow_html=True)
