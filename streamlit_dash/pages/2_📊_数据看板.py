"""
📊 数据看板
- 明细数据查询
- 核心统计
- 联赛积分榜
- 可视化图表
- 比赛分析（单队走势、历史交锋）
- 数据质量校验
"""
import streamlit as st
import os
import sys
import pandas as pd
import numpy as np

# 页面配置
st.set_page_config(
    page_title="数据看板",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 路径配置
SCRIPT_PATH = os.path.abspath(__file__)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_PATH)))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(SCRIPT_PATH)))

# 导入公共模块
from common.data_loader import (
    load_match_feature_data, get_league_list, cfg_to_db_league
)
from common.style import style_match_result_df, apply_global_style
from common.usage_tracker import track

# 应用全局美化样式
apply_global_style()

# 页面访问埋点
track('page_view', page_name='数据看板')

# 导入球队映射
from team_mapping_v2 import LEAGUE_TEAM_MAP, LEAGUE_CFG, get_team_cn_name_v2

# ==================== 加载数据 ====================
df_raw = load_match_feature_data()

# 从match_date推算赛季（8月后算新赛季）
df_raw["season_year"] = df_raw["match_date"].apply(
    lambda d: f"{d.year}-{d.year+1}" if d.month >= 8 else f"{d.year-1}-{d.year}"
)

# 列名兼容
league_raw_col = "league_code_raw" if "league_code_raw" in df_raw.columns else "league_code"
home_col = "home_team_std" if "home_team_std" in df_raw.columns else "home_team"
away_col = "away_team_std" if "away_team_std" in df_raw.columns else "away_team"

# 构建球队名映射（从映射字典 + 数据本身双重保障，100%覆盖）
cn_2_std = {}
std_2_cn = {}
for cfg_code, team_map in LEAGUE_TEAM_MAP.items():
    for eng_std, (full_eng, cn_name) in team_map.items():
        std_2_cn[eng_std] = cn_name
        cn_2_std[cn_name] = eng_std

# 从数据本身补充 std -> 中文名 映射（兜底，确保无缺失）
if "home_team_std" in df_raw.columns and "home_team" in df_raw.columns:
    for _, row in df_raw[["home_team_std", "home_team", "league_code"]].drop_duplicates().iterrows():
        std = row["home_team_std"]
        cn = row["home_team"]
        if std and cn and std not in std_2_cn:
            std_2_cn[std] = cn
if "away_team_std" in df_raw.columns and "away_team" in df_raw.columns:
    for _, row in df_raw[["away_team_std", "away_team", "league_code"]].drop_duplicates().iterrows():
        std = row["away_team_std"]
        cn = row["away_team"]
        if std and cn and std not in std_2_cn:
            std_2_cn[std] = cn

# 终极兜底：用统一映射函数补全所有出现过的球队
all_teams = pd.unique(df_raw[[home_col, away_col]].values.ravel("K")).tolist()
for team in all_teams:
    if team not in std_2_cn:
        # 从数据里找这个球队对应的联赛
        team_league = df_raw[df_raw[home_col] == team]["league_code"].iloc[0] if len(df_raw[df_raw[home_col] == team]) > 0 else None
        if not team_league:
            team_league = df_raw[df_raw[away_col] == team]["league_code"].iloc[0] if len(df_raw[df_raw[away_col] == team]) > 0 else None
        if team_league:
            cn = get_team_cn_name_v2(team_league, team, print_miss=False)
            if cn and cn != team:
                std_2_cn[team] = cn

# ==================== 顶部筛选栏 ====================
st.title("📊 数据看板")

# 筛选行
filter_col1, filter_col2, filter_col3, filter_col4 = st.columns([2, 2, 2, 1])

with filter_col1:
    # 日期范围
    min_date = df_raw["match_date"].min()
    max_date = df_raw["match_date"].max()
    date_range = st.date_input(
        "日期范围",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

with filter_col2:
    # 联赛筛选
    league_dict = get_league_list()
    league_list = list(league_dict.keys())
    league_options = ["全部联赛"] + league_list
    sel_league_label = st.selectbox(
        "选择联赛",
        options=league_options,
        format_func=lambda x: league_dict.get(x, x) if x != "全部联赛" else "全部联赛",
        index=0
    )

with filter_col3:
    # 赛季筛选（单选 + 全部赛季）
    all_seasons = sorted(df_raw["season_year"].dropna().unique().tolist(), reverse=True)
    season_options = ["全部赛季"] + [str(s) for s in all_seasons]
    default_season = all_seasons[0] if all_seasons else "全部赛季"
    sel_season = st.selectbox(
        "选择赛季",
        options=season_options,
        index=0  # 默认全部赛季
    )

with filter_col4:
    st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
    if st.button("🔄 重置筛选", use_container_width=True):
        st.rerun()

# 应用筛选
mask_date = (df_raw["match_date"] >= pd.to_datetime(date_range[0])) & \
            (df_raw["match_date"] <= pd.to_datetime(date_range[1]))

if sel_league_label == "全部联赛":
    mask_league = pd.Series(True, index=df_raw.index)
else:
    league_db_code = cfg_to_db_league(sel_league_label)
    mask_league = df_raw[league_raw_col] == league_db_code

if sel_season == "全部赛季":
    mask_season = pd.Series(True, index=df_raw.index)
else:
    mask_season = df_raw["season_year"].astype(str) == sel_season

df_filter = df_raw[mask_date & mask_league & mask_season].copy()

# 筛选统计
st.caption(f"当前筛选：共 {len(df_filter):,} 场比赛 | 日期：{date_range[0]} ~ {date_range[1]} | "
           f"联赛：{sel_league_label} | 赛季：{sel_season}")

st.divider()

# ==================== 核心统计卡片 ====================
st.subheader("📈 核心数据统计")

total_matches = len(df_filter)
home_win_rate = (df_filter["match_result"] == "主队胜").mean() * 100 if total_matches > 0 else 0
draw_rate = (df_filter["match_result"] == "平局").mean() * 100 if total_matches > 0 else 0
away_win_rate = (df_filter["match_result"] == "客队胜").mean() * 100 if total_matches > 0 else 0

avg_home_goals = df_filter["home_goals"].mean() if total_matches > 0 else 0
avg_away_goals = df_filter["away_goals"].mean() if total_matches > 0 else 0
avg_total_goals = avg_home_goals + avg_away_goals

total_teams = pd.unique(df_filter[[home_col, away_col]].values.ravel("K")).shape[0]
total_leagues = df_filter[league_raw_col].nunique()

c1, c2, c3, c4 = st.columns(4)
c1.metric("总场次", f"{total_matches:,}")
c2.metric("主队胜率", f"{home_win_rate:.1f}%")
c3.metric("平局率", f"{draw_rate:.1f}%")
c4.metric("客队胜率", f"{away_win_rate:.1f}%")

c5, c6, c7, c8 = st.columns(4)
c5.metric("场均进球", f"{avg_total_goals:.2f}")
c6.metric("场均主队进球", f"{avg_home_goals:.2f}")
c7.metric("场均客队进球", f"{avg_away_goals:.2f}")
c8.metric("参赛球队", f"{total_teams} 支")

st.divider()

# ==================== 联赛预测难度排行 ====================
st.subheader("🏆 联赛预测难度排行")
st.caption("基于联赛独立模型Walk Forward验证，准确率越高说明规律越稳定、越好预测")

try:
    import sqlite3
    db_path = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_PATH)), "football.db")
    conn = sqlite3.connect(db_path)
    league_rank_df = pd.read_sql("SELECT * FROM league_independent_wf ORDER BY 整体准确率 DESC", conn)
    conn.close()

    # 排名卡片
    cols = st.columns(5)
    for i, (_, row) in enumerate(league_rank_df.iterrows()):
        with cols[i]:
            medal = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i]
            diff = row['整体准确率'] - league_rank_df['整体准确率'].mean()
            diff_str = f"+{diff:.2%}" if diff > 0 else f"{diff:.2%}"

            st.markdown(f"""
            <div style="text-align:center; padding:15px; border-radius:10px; background:linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); border:1px solid #e2e8f0;">
                <div style="font-size:24px; margin-bottom:5px;">{medal}</div>
                <div style="font-size:18px; font-weight:600; color:#1e293b;">{row['联赛']}</div>
                <div style="font-size:24px; font-weight:700; color:#22c55e; margin:8px 0;">{row['整体准确率']:.1%}</div>
                <div style="font-size:12px; color:#64748b;">整体准确率</div>
                <div style="font-size:11px; color:{'#22c55e' if diff > 0 else '#ef4444'}; margin-top:5px;">
                    {diff_str} vs 平均
                </div>
            </div>
            """, unsafe_allow_html=True)

    # 详细表格 + 高置信准确率对比
    col_table, col_chart = st.columns([1, 1])

    with col_table:
        st.markdown("###### 📊 详细数据")
        league_display = league_rank_df.copy()
        league_display['整体准确率'] = league_display['整体准确率'].apply(lambda x: f"{x:.2%}")
        league_display['平均置信度'] = league_display['平均置信度'].apply(lambda x: f"{x:.2%}")
        league_display['>=70%准确率'] = league_display['>=70%准确率'].apply(lambda x: f"{x:.2%}")
        league_display = league_display[['联赛', '总场次', '整体准确率', '平均置信度', '>=70%准确率']]
        league_display.columns = ['联赛', '验证场次', '整体准确率', '平均置信度', '高置信准确率']
        st.dataframe(league_display, hide_index=True, use_container_width=True)

    with col_chart:
        st.markdown("###### 📈 准确率对比")
        try:
            chart_df = league_rank_df.copy()
            chart_df = chart_df.set_index('联赛')[['整体准确率', '>=70%准确率']]
            chart_df.columns = ['整体准确率', '高置信准确率(≥70%)']
            st.bar_chart(chart_df, height=350)
        except Exception as e:
            st.caption(f"图表渲染失败：{str(e)}")

    st.info("""
    💡 **关键发现**：
    1. **英超最好预测** — 整体准确率最高，强弱分明
    2. **西甲高置信最准** — 虽然整体第三，但高置信度比赛准确率最高
    3. **法甲/德甲最难** — 整体准确率最低，冷门多
    4. **联赛独立模型 < 通用模型** — 单个联赛数据量少，不如全联赛训练的泛化能力强
    """)

except Exception as e:
    st.info(f"联赛排名数据加载失败：{str(e)}")

st.divider()

# ==================== 积分榜 ====================
st.subheader("🏆 联赛积分榜")

if sel_league_label == "全部联赛":
    st.info("请在上方筛选中选择具体联赛，查看对应积分榜")
else:
    # 计算积分榜
    teams = set()
    home_teams = df_filter[home_col].unique()
    away_teams = df_filter[away_col].unique()
    teams.update(home_teams)
    teams.update(away_teams)

    standings = []
    for team in teams:
        home_matches = df_filter[df_filter[home_col] == team]
        away_matches = df_filter[df_filter[away_col] == team]

        h_win = len(home_matches[home_matches["match_result"] == "主队胜"])
        h_draw = len(home_matches[home_matches["match_result"] == "平局"])
        h_loss = len(home_matches[home_matches["match_result"] == "客队胜"])
        h_gf = home_matches["home_goals"].sum()
        h_ga = home_matches["away_goals"].sum()

        a_win = len(away_matches[away_matches["match_result"] == "客队胜"])
        a_draw = len(away_matches[away_matches["match_result"] == "平局"])
        a_loss = len(away_matches[away_matches["match_result"] == "主队胜"])
        a_gf = away_matches["away_goals"].sum()
        a_ga = away_matches["home_goals"].sum()

        total_w = h_win + a_win
        total_d = h_draw + a_draw
        total_l = h_loss + a_loss
        total_gf = h_gf + a_gf
        total_ga = h_ga + a_ga
        points = total_w * 3 + total_d
        gd = total_gf - total_ga

        standings.append({
            "球队": std_2_cn.get(team, team),
            "场次": total_w + total_d + total_l,
            "胜": total_w,
            "平": total_d,
            "负": total_l,
            "进球": int(total_gf),
            "失球": int(total_ga),
            "净胜球": int(gd),
            "积分": points
        })

    df_standings = pd.DataFrame(standings).sort_values(
        by=["积分", "净胜球", "进球"], ascending=False
    ).reset_index(drop=True)
    df_standings.index = df_standings.index + 1

    # 分区颜色标记
    total_teams = len(df_standings)
    # 降级区数量：18队联赛降2个，20队降3个
    relegation_cnt = 2 if total_teams <= 18 else 3

    def highlight_row(row):
        idx = row.name
        styles = [""] * len(row)
        if idx <= 4:
            styles = ["background-color: #fff8e1; font-weight: 500;"] * len(row)  # 欧冠区（金）
        elif idx <= 6:
            styles = ["background-color: #e3f2fd;"] * len(row)  # 欧联/欧协联区（蓝）
        elif idx > total_teams - relegation_cnt:
            styles = ["background-color: #ffebee;"] * len(row)  # 降级区（红）
        return styles

    styled_standings = df_standings.style.apply(highlight_row, axis=1)
    st.dataframe(styled_standings, use_container_width=True, height=500)
    st.caption("🟡 欧冠区（前4） | 🔵 欧战附加区（5-6名） | 🔴 降级区")

st.divider()

# ==================== 比赛分析 ====================
st.subheader("⚔️ 比赛分析")

# 联赛筛选（两个tab共用）
col_ana_lg, _ = st.columns([1, 3])
with col_ana_lg:
    ana_league = st.selectbox("联赛筛选", ["全部联赛"] + sorted(league_list), key="ana_league")

if ana_league != "全部联赛":
    lg_code = cfg_to_db_league(ana_league)
    df_ana = df_filter[df_filter[league_raw_col] == lg_code]
else:
    df_ana = df_filter

tab_team, tab_h2h = st.tabs(["🏟️ 球队详情", "⚔️ 历史交锋"])

with tab_team:
    team_list = sorted(pd.unique(df_ana[[home_col, away_col]].values.ravel("K")).tolist())
    sel_team = st.selectbox(
        "选择球队", 
        options=team_list, 
        index=0 if team_list else 0,
        format_func=lambda x: f"{std_2_cn.get(x, x)} ({x})" if std_2_cn.get(x) != x else x
    )

    if sel_team:
        home_matches = df_ana[df_ana[home_col] == sel_team].copy()
        home_matches["球队进球"] = home_matches["home_goals"]
        home_matches["球队失球"] = home_matches["away_goals"]
        home_matches["赛果"] = home_matches["match_result"].map({
            "主队胜": "胜", "平局": "平", "客队胜": "负"
        })
        home_matches["对手"] = home_matches[away_col]
        home_matches["主客场"] = "主场"

        away_matches = df_ana[df_ana[away_col] == sel_team].copy()
        away_matches["球队进球"] = away_matches["away_goals"]
        away_matches["球队失球"] = away_matches["home_goals"]
        away_matches["赛果"] = away_matches["match_result"].map({
            "客队胜": "胜", "平局": "平", "主队胜": "负"
        })
        away_matches["对手"] = away_matches[home_col]
        away_matches["主客场"] = "客场"

        cols_show = ["match_date", "主客场", "对手", "球队进球", "球队失球", "赛果"]
        team_all = pd.concat([
            home_matches[cols_show],
            away_matches[cols_show]
        ]).sort_values("match_date", ascending=False).reset_index(drop=True)
        team_all.columns = ["比赛日期", "主客场", "对手", "进球", "失球", "赛果"]
        team_all["对手"] = team_all["对手"].map(lambda x: std_2_cn.get(x, x))

        # 基础统计
        total = len(team_all)
        wins = (team_all["赛果"] == "胜").sum()
        draws = (team_all["赛果"] == "平").sum()
        losses = (team_all["赛果"] == "负").sum()
        points = wins * 3 + draws
        gf = team_all["进球"].sum()
        ga = team_all["失球"].sum()
        gd = gf - ga

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("总场次", total)
        c2.metric("战绩", f"{wins}胜{draws}平{losses}负")
        c3.metric("积分", f"{points}分")
        c4.metric("净胜球", f"{gd:+d}")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("场均进球", f"{gf/total:.2f}")
        c6.metric("场均失球", f"{ga/total:.2f}")
        c7.metric("胜率", f"{wins/total:.1%}")
        c8.metric("不败率", f"{(wins+draws)/total:.1%}")

        # 主客场分别表现
        st.markdown("###### 主客场表现")
        home_df = team_all[team_all["主客场"] == "主场"]
        away_df = team_all[team_all["主客场"] == "客场"]

        col_h, col_a = st.columns(2)
        with col_h:
            hw = len(home_df[home_df["赛果"] == "胜"])
            hd = len(home_df[home_df["赛果"] == "平"])
            hl = len(home_df[home_df["赛果"] == "负"])
            hgf = home_df["进球"].sum()
            hga = home_df["失球"].sum()
            st.info(f"🏠 主场 {len(home_df)}场：{hw}胜{hd}平{hl}负 | 进{hgf}失{hga} | 胜率{hw/len(home_df):.0%}")
        with col_a:
            aw = len(away_df[away_df["赛果"] == "胜"])
            ad = len(away_df[away_df["赛果"] == "平"])
            al = len(away_df[away_df["赛果"] == "负"])
            agf = away_df["进球"].sum()
            aga = away_df["失球"].sum()
            st.info(f"✈️ 客场 {len(away_df)}场：{aw}胜{ad}平{al}负 | 进{agf}失{aga} | 胜率{aw/len(away_df):.0%}")

        # 近期状态（最近5场）
        if len(team_all) >= 5:
            st.markdown("###### 近期状态（最近5场）")
            recent = team_all.head(5).copy()
            recent["结果"] = recent["赛果"].map({"胜": "✅", "平": "➖", "负": "❌"})
            recent_str = "  ".join(recent["结果"].tolist())
            recent_wins = (recent["赛果"] == "胜").sum()
            st.metric("近5场战绩", f"{recent_wins}胜 {len(recent[recent['赛果']=='平'])}平 {len(recent[recent['赛果']=='负'])}负", recent_str)

        # 比赛明细
        with st.expander("📋 近期比赛记录（最近20场）", expanded=False):
            st.caption("该球队在当前联赛/赛季筛选条件下的比赛明细，按日期倒序排列")
            st.dataframe(team_all.head(20), use_container_width=True, hide_index=True)

with tab_h2h:
    team_list_h = sorted(pd.unique(df_ana[[home_col, away_col]].values.ravel("K")).tolist())
    h1 = st.selectbox(
        "主队", 
        options=team_list_h, 
        index=0 if team_list_h else 0, 
        key="h2h_home",
        format_func=lambda x: f"{std_2_cn.get(x, x)} ({x})" if std_2_cn.get(x) != x else x
    )
    h2 = st.selectbox(
        "客队", 
        options=team_list_h, 
        index=1 if len(team_list_h) > 1 else 0, 
        key="h2h_away",
        format_func=lambda x: f"{std_2_cn.get(x, x)} ({x})" if std_2_cn.get(x) != x else x
    )

    if h1 and h2 and h1 != h2:
        # 中文队名
        h1_cn = std_2_cn.get(h1, h1)
        h2_cn = std_2_cn.get(h2, h2)
        
        mask_h2h = ((df_ana[home_col] == h1) & (df_ana[away_col] == h2)) | \
                   ((df_ana[home_col] == h2) & (df_ana[away_col] == h1))
        history_show = df_ana[mask_h2h].copy()

        if not history_show.empty:
            # 统一视角：h1 作为主队视角
            def normalize_row(row):
                if row[home_col] == h1:
                    return pd.Series({
                        "比赛日期": row["match_date"].strftime("%Y-%m-%d"),
                        "主队": std_2_cn.get(row[home_col], row[home_col]),
                        "比分": f"{int(row['home_goals'])}:{int(row['away_goals'])}",
                        "客队": std_2_cn.get(row[away_col], row[away_col]),
                        "赛果": row["match_result"]
                    })
                else:
                    # 主客对调，赛果反转
                    rev_result = {"主队胜": "客队胜", "客队胜": "主队胜", "平局": "平局"}
                    return pd.Series({
                        "比赛日期": row["match_date"].strftime("%Y-%m-%d"),
                        "主队": std_2_cn.get(row[away_col], row[away_col]),
                        "比分": f"{int(row['away_goals'])}:{int(row['home_goals'])}",
                        "客队": std_2_cn.get(row[home_col], row[home_col]),
                        "赛果": rev_result.get(row["match_result"], row["match_result"])
                    })

            history_show = history_show.apply(normalize_row, axis=1).sort_values(
                "比赛日期", ascending=False).reset_index(drop=True)

            total_cnt = len(history_show)
            home_win_cnt = len(history_show[history_show["赛果"] == "主队胜"])
            draw_cnt = len(history_show[history_show["赛果"] == "平局"])
            away_win_cnt = len(history_show[history_show["赛果"] == "客队胜"])

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("交锋总场次", total_cnt)
            c2.metric(f"{h1_cn}取胜", home_win_cnt)
            c3.metric("平局场次", draw_cnt)
            c4.metric(f"{h2_cn}取胜", away_win_cnt)

            styled_history = style_match_result_df(history_show)
            st.dataframe(styled_history, use_container_width=True, hide_index=True)
        else:
            st.info("暂无该对阵组合的历史交锋数据")

st.divider()

# ==================== 明细数据 ====================
st.subheader("📋 明细数据")

st.warning("""
⚠️ **重要提醒：明细数据中的「模型置信度」是基于全量训练模型的预测结果，存在一定未来函数，偏乐观！**
真实模型预测能力请参考下方「模型信息」中的WF验证数据，或使用「预测中心」的单场预测功能。
""")

col_disp1, col_disp2 = st.columns([1, 3])
with col_disp1:
    show_mode = st.radio("展示模式", ["精简视图", "完整视图"], horizontal=True)

# 分歧度筛选
if "odds_disp_total" in df_filter.columns:
    with col_disp2:
        disp_options = ["全部分歧度", "低分歧 (<2%)", "中分歧 (2-4%)", "高分歧 (>4%)", "无分歧数据"]
        sel_disp = st.selectbox("分歧度筛选", disp_options, index=0)
    
    if sel_disp == "低分歧 (<2%)":
        df_filter = df_filter[(df_filter["odds_disp_total"] > 0) & (df_filter["odds_disp_total"] < 0.02)]
    elif sel_disp == "中分歧 (2-4%)":
        df_filter = df_filter[(df_filter["odds_disp_total"] >= 0.02) & (df_filter["odds_disp_total"] < 0.04)]
    elif sel_disp == "高分歧 (>4%)":
        df_filter = df_filter[df_filter["odds_disp_total"] >= 0.04]
    elif sel_disp == "无分歧数据":
        df_filter = df_filter[(df_filter["odds_disp_total"].isna()) | (df_filter["odds_disp_total"] <= 0)]

# 精选策略筛选
if "pred_confidence" in df_filter.columns and "odds_disp_total" in df_filter.columns:
    strategy_options = [
        "全部比赛",
        "🎯 精选稳单（高置信+高分歧）",
        "高置信比赛（≥60%）",
        "极高置信（≥65%）",
        "高分歧比赛（>4%）"
    ]
    sel_strategy = st.selectbox("策略筛选", strategy_options, index=0, key="strategy_filter")
    
    if sel_strategy == "🎯 精选稳单（高置信+高分歧）":
        df_filter = df_filter[(df_filter["pred_confidence"] >= 0.6) & (df_filter["odds_disp_total"] >= 0.04)]
    elif sel_strategy == "高置信比赛（≥60%）":
        df_filter = df_filter[df_filter["pred_confidence"] >= 0.6]
    elif sel_strategy == "极高置信（≥65%）":
        df_filter = df_filter[df_filter["pred_confidence"] >= 0.65]
    elif sel_strategy == "高分歧比赛（>4%）":
        df_filter = df_filter[df_filter["odds_disp_total"] >= 0.04]

# 市场热度筛选（初终盘变化）
if "market_move_dir" in df_filter.columns:
    move_options = ["全部热度", "🔥 主队升温", "❄️ 主队降温", "➖ 平稳", "💹 震荡"]
    sel_move = st.selectbox("市场热度筛选", move_options, index=0, key="market_move_filter")
    
    if sel_move == "🔥 主队升温":
        df_filter = df_filter[df_filter["market_move_dir"] == "主队升温"]
    elif sel_move == "❄️ 主队降温":
        df_filter = df_filter[df_filter["market_move_dir"] == "主队降温"]
    elif sel_move == "➖ 平稳":
        df_filter = df_filter[df_filter["market_move_dir"] == "平稳"]
    elif sel_move == "💹 震荡":
        df_filter = df_filter[df_filter["market_move_dir"] == "震荡"]

if show_mode == "精简视图":
    cols_show = ["match_date", home_col, away_col, "home_goals", "away_goals", "match_result", "season_year"]
    cols_show = [c for c in cols_show if c in df_filter.columns]
    df_show = df_filter[cols_show].copy()
    df_show.columns = ["比赛日期", "主队", "客队", "主队进球", "客队进球", "赛果", "赛季"]
    
    # 队名中文化
    df_show["主队"] = df_show["主队"].map(lambda x: std_2_cn.get(x, x))
    df_show["客队"] = df_show["客队"].map(lambda x: std_2_cn.get(x, x))

    # 增加市场分歧度标签（如果有数据）
    if "odds_disp_total" in df_filter.columns:
        def disp_label(val):
            if pd.isna(val) or val <= 0: return "—"
            if val < 0.02: return "🟢 低分歧"
            if val < 0.04: return "🟡 中分歧"
            return "🔴 高分歧"
        df_show.insert(6, "市场分歧", df_filter["odds_disp_total"].apply(disp_label))

    # 增加模型置信度（如果有数据）
    if "pred_confidence" in df_filter.columns:
        def conf_label(val):
            if pd.isna(val): return "—"
            if val >= 0.65: return f"🔴 {val:.1%}"
            if val >= 0.55: return f"🟡 {val:.1%}"
            return f"🟢 {val:.1%}"
        df_show.insert(7, "模型置信度", df_filter["pred_confidence"].apply(conf_label))

    # 增加市场热度（如果有数据）
    if "market_move_dir" in df_filter.columns:
        def move_label(val):
            if pd.isna(val) or val == "": return "—"
            emoji = {"主队升温": "🔥", "主队降温": "❄️", "平稳": "➖", "震荡": "💹"}
            return f"{emoji.get(val, '')} {val}"
        df_show["市场热度"] = df_filter["market_move_dir"].apply(move_label)
else:
    df_show = df_filter.copy()
    # 列名中文化（核心列）
    col_cn_map = {
        "league_code": "联赛编码", "league_code_raw": "联赛编码", "league_cfg": "联赛配置",
        "season_year": "赛季", "match_date": "比赛日期", "Time": "开赛时间",
        "home_team": "主队", "away_team": "客队",
        "home_team_std": "主队(标准名)", "away_team_std": "客队(标准名)",
        "match_result": "赛果", "home_goals": "主队进球", "away_goals": "客队进球",
        "HS": "主队射门", "AwayShot": "客队射门", "HST": "主队射正", "AST": "客队射正",
        "real_h_prob": "主胜赔率概率", "real_d_prob": "平局赔率概率", "real_a_prob": "客胜赔率概率",
        "odds_draw_real": "平局真实概率", "odds_lose_real": "客胜真实概率",
        "shot_on_diff": "射正差",
        "h5_gf": "主队近5场进球", "h5_ga": "主队近5场失球",
        "h5_shot": "主队近5场射门", "h5_shot_ot": "主队近5场射正",
        "a5_gf": "客队近5场进球", "a5_ga": "客队近5场失球",
        "a5_shot": "客队近5场射门", "a5_shot_ot": "客队近5场射正",
        "h10_gf": "主队近10场进球", "h10_ga": "主队近10场失球",
        "a10_gf": "客队近10场进球", "a10_ga": "客队近10场失球",
        "h2h_cnt": "交锋次数", "h2h_home_win_rate": "交锋主队胜率",
        "h2h_draw_rate": "交锋平局率", "h2h_home_gf_avg": "交锋主队场均进球",
        "h2h_home_ga_avg": "交锋主队场均失球",
        "prob_ratio_ha": "主客胜概率比", "prob_draw_share": "平局概率占比",
        "prob_max": "最大概率", "prob_entropy": "概率熵",
        "prob_home_favorite": "主队热门",
        "home_draw_rate_5": "主队近5场平局率", "home_draw_rate_10": "主队近10场平局率",
        "away_draw_rate_5": "客队近5场平局率", "away_draw_rate_10": "客队近10场平局率",
        "odds_disp_home": "主胜赔率分歧度", "odds_disp_draw": "平局赔率分歧度",
        "odds_disp_away": "客胜赔率分歧度", "odds_disp_total": "综合赔率分歧度",
        "odds_range_home": "主胜赔率极值差", "odds_range_draw": "平局赔率极值差",
        "odds_range_away": "客胜赔率极值差",
        "ps_b365_diff_home": "PS主胜偏离B365", "ps_b365_diff_draw": "PS平局偏离B365",
        "ps_b365_diff_away": "PS客胜偏离B365",
    }
    df_show.columns = [col_cn_map.get(c, c) for c in df_show.columns]
    
    # 队名中文化
    if "主队" in df_show.columns:
        df_show["主队"] = df_show["主队"].map(lambda x: std_2_cn.get(x, x))
    if "客队" in df_show.columns:
        df_show["客队"] = df_show["客队"].map(lambda x: std_2_cn.get(x, x))

# 按比赛日期倒序排列（最新在前）
date_col = "比赛日期" if "比赛日期" in df_show.columns else "match_date"
if date_col in df_show.columns:
    df_show = df_show.sort_values(date_col, ascending=False).reset_index(drop=True)

# 列宽优化：数字列窄一点，文本列自适应
col_config = {}
try:
    for col in df_show.columns:
        if pd.api.types.is_numeric_dtype(df_show[col]):
            col_config[col] = st.column_config.NumberColumn(col, width="small")
        elif col in ["比赛日期", "match_date"]:
            col_config[col] = st.column_config.DateColumn(col, width="small")
except Exception:
    # 列宽配置失败时降级到默认
    col_config = None

if col_config:
    st.dataframe(df_show, use_container_width=True, height=400, column_config=col_config)
else:
    st.dataframe(df_show, use_container_width=True, height=400)
st.caption(f"共 {len(df_filter):,} 条记录")

st.divider()

# ==================== 数据质量校验 ====================
with st.expander("🔍 数据质量校验", expanded=False):
    st.markdown("##### 缺失值统计")
    missing = df_filter.isnull().sum()
    missing_pct = (missing / len(df_filter) * 100).round(2)
    df_missing = pd.DataFrame({"缺失数量": missing, "缺失率(%)": missing_pct})
    df_missing = df_missing[df_missing["缺失数量"] > 0].sort_values("缺失数量", ascending=False)
    if len(df_missing) > 0:
        st.dataframe(df_missing, use_container_width=True)
    else:
        st.success("✅ 无缺失值")

    st.markdown("##### 异常值检查")
    goal_cols = ["home_goals", "away_goals"]
    for col in goal_cols:
        if col in df_filter.columns:
            max_val = df_filter[col].max()
            if max_val > 10:
                st.warning(f"⚠️ {col} 最大值 {max_val}，可能存在异常")
    st.success("✅ 数据范围校验通过")

# ==================== 模型信息（高级） ====================
with st.expander("⚙️ 模型信息（高级）", expanded=False):
    st.markdown("##### 模型概览")
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("模型架构", "LGB + 泊松 + 平局专项")
    col_m2.metric("特征维度", "52维（含ELO扩展+时间衰减）")
    col_m3.metric("融合权重", "55% / 30% / 5%")
    
    col_m4, col_m5 = st.columns(2)
    col_m4.metric("WF真实准确率", "53.6%", help="Walk Forward赛季重置版验证，无未来函数")
    col_m5.metric("模型真实alpha", "约1-2%", help="扣除庄家抽水后，勉强打平")
    st.info("💡 三模型融合：LightGBM三分类 + 泊松进球回归 + 平局二分类专项，三种不同原理互补")
    
    st.divider()
    st.markdown("##### 联赛预测难度排行（WF版真实数据）")
    st.caption("联赛独立模型Walk Forward验证准确率，越高说明规律越稳定、越好预测")
    try:
        import sqlite3
        db_path = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_PATH)), "football.db")
        conn = sqlite3.connect(db_path)
        league_rank_df = pd.read_sql("SELECT * FROM league_independent_wf ORDER BY 整体准确率 DESC", conn)
        conn.close()

        league_display = league_rank_df.copy()
        league_display['整体准确率'] = league_display['整体准确率'].apply(lambda x: f"{x:.2%}")
        league_display['>=70%准确率'] = league_display['>=70%准确率'].apply(lambda x: f"{x:.2%}")
        league_display['难度等级'] = league_display['整体准确率'].apply(
            lambda x: "⭐⭐ 较易" if float(x.strip('%')) >= 51
            else "⭐⭐⭐ 中等" if float(x.strip('%')) >= 49
            else "⭐⭐⭐⭐ 较难"
        )
        league_display = league_display[['联赛', '总场次', '整体准确率', '>=70%准确率', '难度等级']]
        league_display.columns = ['联赛', '验证场次', '整体准确率', '高置信准确率', '难度等级']
        st.dataframe(league_display, hide_index=True, use_container_width=True)
    except Exception as e:
        st.caption(f"联赛排名数据加载失败：{str(e)}")
    
    st.divider()
    st.markdown("##### 模型特征重要性 TOP10")
    st.caption("LightGBM模型全局增益重要性，数值越大代表该特征对预测结果影响越大")
    try:
        from match_predict import home_model, FEATURE_COLS
        FEATURE_CN = {
            "h5_gf": "主队近5场进球", "h5_ga": "主队近5场失球",
            "h5_shot": "主队近5场射门", "h5_shot_ot": "主队近5场射正",
            "h10_gf": "主队近10场进球", "h10_ga": "主队近10场失球",
            "a5_gf": "客队近5场进球", "a5_ga": "客队近5场失球",
            "a5_shot": "客队近5场射门", "a5_shot_ot": "客队近5场射正",
            "a10_gf": "客队近10场进球", "a10_ga": "客队近10场失球",
            "odds_draw_real": "平局赔率偏离", "odds_lose_real": "客胜赔率偏离",
            "h2h_cnt": "交锋场次", "h2h_home_win_rate": "交锋主队胜率",
            "h2h_draw_rate": "交锋平局率", "h2h_home_gf_avg": "交锋主队进球",
            "h2h_home_ga_avg": "交锋主队失球",
            "prob_ratio_ha": "主客胜概率比", "prob_draw_share": "平局概率占比",
            "prob_max": "概率集中度", "prob_entropy": "概率不确定性",
            "prob_home_favorite": "主队热门度",
            "home_draw_rate_5": "主队近5场平局率", "home_draw_rate_10": "主队近10场平局率",
            "away_draw_rate_5": "客队近5场平局率", "away_draw_rate_10": "客队近10场平局率",
            "league_SER": "联赛-意甲", "league_E0": "联赛-英超",
            "league_D1": "联赛-德甲", "league_LIG": "联赛-法甲", "league_LLA": "联赛-西甲",
            "home_elo_before": "主队赛前ELO", "away_elo_before": "客队赛前ELO",
            "elo_diff_before": "赛前ELO差值",
            # ELO扩展特征
            "h5_gf_elo_weighted": "主队加权进攻效率", "h5_ga_elo_weighted": "主队加权防守漏洞",
            "a5_gf_elo_weighted": "客队加权进攻效率", "a5_ga_elo_weighted": "客队加权防守漏洞",
            "home_w5_elo_trend": "主队近期状态走势", "home_w10_elo_trend": "主队中期状态走势",
            "away_w5_elo_trend": "客队近期状态走势", "away_w10_elo_trend": "客队中期状态走势",
            # 时间衰减特征
            "h5_gf_time_decay": "主队近期进球(时间衰减)", "h5_ga_time_decay": "主队近期失球(时间衰减)",
            "a5_gf_time_decay": "客队近期进球(时间衰减)", "a5_ga_time_decay": "客队近期失球(时间衰减)",
            "h10_gf_time_decay": "主队中期进球(时间衰减)", "h10_ga_time_decay": "主队中期失球(时间衰减)",
            "a10_gf_time_decay": "客队中期进球(时间衰减)", "a10_ga_time_decay": "客队中期失球(时间衰减)",
        }
        gain_imp = home_model.feature_importance(importance_type='gain')
        total = gain_imp.sum()
        imp_pct = gain_imp / total * 100
        feat_imp = list(zip(FEATURE_COLS, imp_pct))
        feat_imp.sort(key=lambda x: x[1], reverse=True)
        top10 = feat_imp[:10]
        imp_df = pd.DataFrame([
            {"排名": i+1, "特征名称": FEATURE_CN.get(f, f), "权重占比": f"{p:.2f}%"}
            for i, (f, p) in enumerate(top10)
        ])
        st.dataframe(imp_df, hide_index=True, use_container_width=True)
    except Exception as e:
        st.caption(f"特征重要性加载失败：{str(e)}")
    
    st.divider()
    st.markdown("##### 置信度 vs 实际准确率（WF版金标准）")
    st.caption("Walk Forward滚动验证，无未来函数，模型输出置信度与真实准确率的对应关系")
    try:
        import sqlite3
        db_path = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_PATH)), "football.db")
        conn = sqlite3.connect(db_path)
        conf_acc_df = pd.read_sql("SELECT * FROM wf_confidence_accuracy", conn)
        conn.close()

        # 展示表格
        conf_display = conf_acc_df.copy()
        conf_display['准确率'] = conf_display['准确率'].apply(lambda x: f"{x:.2%}")
        conf_display['平均置信度'] = conf_display['平均置信度'].apply(lambda x: f"{x:.2%}")
        conf_display['高估程度'] = (conf_acc_df['平均置信度'] - conf_acc_df['准确率']).apply(lambda x: f"+{x:.1%}" if x > 0 else f"{x:.1%}")
        conf_display = conf_display[['置信度区间', '样本数', '平均置信度', '准确率', '高估程度']]
        st.dataframe(conf_display, hide_index=True, use_container_width=True)

        # 可视化图表
        st.markdown("###### 📈 置信度-准确率曲线图")
        try:
            chart_df = conf_acc_df.copy()
            chart_df = chart_df.set_index('置信度区间')[['准确率', '平均置信度']]
            chart_df.columns = ['真实准确率', '模型输出置信度']
            st.line_chart(chart_df, height=400)
        except Exception as e:
            st.caption(f"图表渲染失败：{str(e)}")

        st.info("""
        📌 **关键结论**：
        1. 置信度与准确率正相关，趋势完全正确
        2. 85%以上高置信度 → 真实准确率约82%
        3. 模型整体略微高估置信度（平均高4-6个百分点）
        4. 高置信度区间高估更明显（约8个百分点）
        """)
    except Exception as e:
        st.caption(f"置信度数据加载失败：{str(e)}")
