"""
🏠 预测中心（主页）
- 手动预测
- 赛事日历
- 预测追踪复盘
- 积分榜速览
"""
import streamlit as st
import os
import sys
import pandas as pd

# 页面配置
st.set_page_config(
    page_title="足球赛事预测中心",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 路径配置
SCRIPT_PATH = os.path.abspath(__file__)
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_PATH))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.dirname(SCRIPT_PATH))

# 导入公共模块
from common.data_loader import (
    load_match_feature_data, load_schedule_data, load_predictions,
    get_league_list, cfg_to_db_league, DB_PATH
)
from common.style import render_confidence_badge, get_result_color, apply_global_style
from common.usage_tracker import track

# 应用全局美化样式
apply_global_style()

# 页面访问埋点
track('page_view', page_name='预测中心')

# 导入功能模块
from predict_module import render_match_predict_panel
from schedule_module import render_schedule_calendar

# 导入球队映射
from team_mapping_v2 import LEAGUE_TEAM_MAP, LEAGUE_CFG

# ==================== 加载数据 ====================
df_all = load_match_feature_data()
df_schedule = load_schedule_data()
df_preds = load_predictions()

# 列名兼容
league_raw_col = "league_code_raw" if "league_code_raw" in df_all.columns else "league_code"
home_col = "home_team_std" if "home_team_std" in df_all.columns else "home_team"
away_col = "away_team_std" if "away_team_std" in df_all.columns else "away_team"

# 构建球队名映射
all_teams_std = pd.unique(df_all[[home_col, away_col]].values.ravel("K"))
cn_2_std = {}
std_2_cn = {}
for cfg_code, team_map in LEAGUE_TEAM_MAP.items():
    for eng_std, (full_eng, cn_name) in team_map.items():
        std_2_cn[eng_std] = cn_name
        cn_2_std[cn_name] = eng_std

# 从数据本身补充映射（兜底，确保无缺失）
home_cn_col = "home_team" if "home_team_std" in df_all.columns else None
if home_cn_col:
    for _, row in df_all[[home_col, home_cn_col]].drop_duplicates().iterrows():
        std = row[home_col]
        cn = row[home_cn_col]
        if std and cn and std not in std_2_cn:
            std_2_cn[std] = cn
    away_cn_col = "away_team" if "away_team_std" in df_all.columns else None
    if away_cn_col:
        for _, row in df_all[[away_col, away_cn_col]].drop_duplicates().iterrows():
            std = row[away_col]
            cn = row[away_cn_col]
            if std and cn and std not in std_2_cn:
                std_2_cn[std] = cn

# ==================== 主区域 ====================
st.title("⚽ 足球赛事预测中心")

# 免责声明
st.warning("""
**⚠️ 免责声明：本系统仅供机器学习研究与教学演示使用，所有预测结果均为模型算法输出，不构成任何投注建议或投资指导。**
博彩有风险，请理性对待。历史数据来自 [football-data.co.uk](https://www.football-data.co.uk/)，模型为自研 LightGBM 三分类。
""")

# 顶部统计卡片
col1, col2, col3, col4 = st.columns(4)
today_str = pd.Timestamp.now().strftime("%Y-%m-%d")
today_matches = len(df_schedule[df_schedule["match_date"].dt.strftime("%Y-%m-%d") == today_str])
total_preds = len(df_preds)
verified_preds = len(df_preds[df_preds["is_verified"] == 1]) if not df_preds.empty else 0
acc = (df_preds[df_preds["is_verified"] == 1]["is_correct"].mean() * 100) if verified_preds > 0 else 0

col1.metric("今日赛事", today_matches, "场")
col2.metric("累计预测", total_preds, "次")
col3.metric("已校验", verified_preds, "场")
col4.metric("预测准确率", f"{acc:.1f}%" if verified_preds > 0 else "--")

# 数据新鲜度提醒
try:
    import sqlite3
    _conn = sqlite3.connect(DB_PATH)
    latest_date = pd.read_sql("SELECT MAX(match_date) as latest FROM match_feature_final", _conn).iloc[0]['latest']
    _conn.close()
    if latest_date:
        latest_dt = pd.to_datetime(latest_date)
        days_old = (pd.Timestamp.now() - latest_dt).days
        latest_str = latest_dt.strftime("%Y-%m-%d")
        
        if days_old <= 7:
            st.success(f"✅ 数据最新 | 最新比赛: {latest_str}（{days_old}天前）")
        elif days_old <= 14:
            st.warning(f"⚠️ 建议更新数据 | 最新比赛: {latest_str}（{days_old}天前）")
        else:
            st.error(f"🔴 数据已过期 | 最新比赛: {latest_str}（{days_old}天前），请及时更新")
except:
    pass

st.divider()

# 核心功能 Tabs
tab_predict, tab_calendar, tab_ai_view, tab_track = st.tabs([
    "🔮 手动预测",
    "📅 赛事日历",
    "🤖 AI今日观点",
    "📈 预测追踪"
])

with tab_predict:
    render_match_predict_panel(cn_2_std=cn_2_std, std_2_cn=std_2_cn)

with tab_calendar:
    render_schedule_calendar()

with tab_ai_view:
    st.subheader("🤖 赛事观点")
    st.caption("基于当前模型批量计算未来赛事的三AI下注意愿，仅供参考")
    
    # 日期范围选择
    col_d1, col_d2 = st.columns([1, 3])
    with col_d1:
        days_ahead = st.slider("未来天数", min_value=1, max_value=14, value=3, key="ai_view_days")
    
    today = pd.Timestamp.now().normalize()
    end_date = today + pd.Timedelta(days=days_ahead)
    
    # 筛选赛程
    df_upcoming = df_schedule[
        (df_schedule["match_date"].dt.normalize() >= today) & 
        (df_schedule["match_date"].dt.normalize() <= end_date)
    ].copy().sort_values("match_date")
    
    # 过滤五大联赛（有模型的）
    valid_leagues = ['E0', 'D1', 'LLA', 'SER', 'LIG']
    df_upcoming = df_upcoming[df_upcoming["league_code"].isin(valid_leagues)]
    
    st.info(f"📅 未来 {days_ahead} 天共 {len(df_upcoming)} 场五大联赛赛事")
    
    if st.button("🔍 批量计算三AI观点", type="primary", key="calc_ai_view"):
        if len(df_upcoming) == 0:
            st.warning("暂无符合条件的赛事")
        else:
            from ai_intent_module import calc_ai_bet_intent
            from match_predict import predict_match
            from streamlit_dash.feature_auto_build import build_feature_by_teams
            
            results = []
            progress_bar = st.progress(0)
            
            for idx, (_, row) in enumerate(df_upcoming.iterrows()):
                home_std = row["home_team"]
                away_std = row["away_team"]
                league = row["league_code"]
                
                try:
                    # 构建特征（用默认赔率降级）
                    feat = build_feature_by_teams(
                        df_all, home_std, away_std,
                        odds_draw_real=0.28, odds_lose_real=0.33,
                        shot_diff=0, league_code=league
                    )
                    # 预测
                    pred = predict_match(feat, is_home_scene=True)
                    conf = pred["confidence"]
                    result = pred["predict_result"]
                    
                    # 计算AI观点
                    intent = calc_ai_bet_intent(conf, result)
                    
                    results.append({
                        "比赛日期": row["match_date"].strftime("%m-%d"),
                        "主队": std_2_cn.get(home_std, home_std),
                        "客队": std_2_cn.get(away_std, away_std),
                        "预测方向": result,
                        "置信度": f"{conf:.1%}",
                        "共识等级": intent["consensus_label"],
                        "激进AI": "✅出手" if intent["intents"]["激进AI"]["will_bet"] else "❌观望",
                        "中立AI": "✅出手" if intent["intents"]["中立AI"]["will_bet"] else "❌观望",
                        "保守AI": "✅出手" if intent["intents"]["保守AI"]["will_bet"] else "❌观望",
                    })
                except Exception as e:
                    results.append({
                        "比赛日期": row["match_date"].strftime("%m-%d"),
                        "主队": std_2_cn.get(home_std, home_std),
                        "客队": std_2_cn.get(away_std, away_std),
                        "预测方向": "—",
                        "置信度": "—",
                        "共识等级": "数据不足",
                        "激进AI": "—",
                        "中立AI": "—",
                        "保守AI": "—",
                    })
                
                progress_bar.progress((idx + 1) / len(df_upcoming))
            
            progress_bar.empty()
            df_ai_view = pd.DataFrame(results)
            st.dataframe(df_ai_view, use_container_width=True, hide_index=True)
            st.caption("💡 赔率使用默认估算值，实际下注意愿以开盘赔率为准；共识度越高可靠性越强")
    else:
        st.info("👆 点击上方按钮批量计算三AI观点")

with tab_track:
    st.subheader("📈 预测追踪与复盘")
    
    if df_preds.empty:
        st.info("暂无预测记录，去手动预测页做一次预测吧～")
    else:
        verified = df_preds[df_preds["is_verified"] == 1].copy()
        total_v = len(verified)
        correct_v = verified["is_correct"].sum() if total_v > 0 else 0
        acc_v = correct_v / total_v if total_v > 0 else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("总预测次数", len(df_preds))
        c2.metric("已完赛校验", total_v)
        c3.metric("预测正确", int(correct_v))
        c4.metric("整体准确率", f"{acc_v:.2%}" if total_v > 0 else "--")

        # 各赛果准确率
        if total_v > 0:
            # 各赛果准确率
            st.markdown("##### 各赛果预测准确率")
            result_acc = []
            result_acc_chart = []
            for res in ["主胜", "平局", "客胜"]:
                subset = verified[verified["predict_result"] == res]
                if len(subset) > 0:
                    r_acc = subset["is_correct"].mean()
                    result_acc.append({
                        "预测赛果": res,
                        "预测场次": len(subset),
                        "正确场次": int(subset["is_correct"].sum()),
                        "准确率": f"{r_acc:.2%}"
                    })
                    result_acc_chart.append({"赛果": res, "准确率": round(r_acc, 3)})

            col_tab1, col_chart1 = st.columns([1, 1])
            col_tab1.dataframe(pd.DataFrame(result_acc), use_container_width=True, hide_index=True)
            if result_acc_chart:
                chart_df = pd.DataFrame(result_acc_chart).set_index("赛果")
                col_chart1.bar_chart(chart_df, color="#3498db")

            # 各联赛准确率对比
            if total_v > 0 and "league_code" in verified.columns:
                st.markdown("##### 🏆 各联赛预测准确率对比")
                db_code_names = {"E0": "英超", "D1": "德甲", "LLA": "西甲", "SER": "意甲", "LIG": "法甲"}
                league_acc_list = []
                league_acc_chart = []
                for lg in sorted(verified["league_code"].dropna().unique()):
                    sub = verified[verified["league_code"] == lg]
                    if len(sub) >= 3:
                        l_acc = sub["is_correct"].mean()
                        lg_name = db_code_names.get(lg, lg)
                        league_acc_list.append({
                            "联赛": lg_name,
                            "预测场次": len(sub),
                            "正确场次": int(sub["is_correct"].sum()),
                            "准确率": f"{l_acc:.2%}"
                        })
                        league_acc_chart.append({"联赛": lg_name, "准确率": round(l_acc, 3)})

                if league_acc_list:
                    col_tab2, col_chart2 = st.columns([1, 1])
                    league_acc_df = pd.DataFrame(league_acc_list).sort_values("准确率", ascending=False)
                    col_tab2.dataframe(league_acc_df, use_container_width=True, hide_index=True)
                    if league_acc_chart:
                        chart_df2 = pd.DataFrame(league_acc_chart).set_index("联赛").sort_values("准确率", ascending=True)
                        col_chart2.bar_chart(chart_df2, color="#2ecc71")
                else:
                    st.caption("各联赛样本不足（需≥3场），暂无法对比")

        # 历史明细（折叠）
        with st.expander(f"📋 预测历史明细（共 {len(df_preds)} 条）", expanded=False):
            # 筛选栏
            filt_col1, filt_col2, filt_col3, filt_col4 = st.columns(4)
            # 联赛筛选
            all_leagues = sorted(df_preds["league_code"].dropna().unique().tolist()) if "league_code" in df_preds.columns else []
            if all_leagues:
                selected_league = filt_col1.selectbox("联赛筛选", ["全部联赛"] + all_leagues, key="filt_league")
            else:
                selected_league = "全部联赛"
            # 预测结果筛选
            selected_result = filt_col2.selectbox("预测赛果", ["全部", "主胜", "平局", "客胜"], key="filt_result")
            # 校验状态
            selected_verified = filt_col3.selectbox("校验状态", ["全部", "已校验", "未校验"], key="filt_verified")
            # 正确与否
            selected_correct = filt_col4.selectbox("预测结果", ["全部", "正确", "错误"], key="filt_correct")

            # 应用筛选
            df_filt = df_preds.copy()
            if selected_league != "全部联赛" and "league_code" in df_filt.columns:
                df_filt = df_filt[df_filt["league_code"] == selected_league]
            if selected_result != "全部":
                df_filt = df_filt[df_filt["predict_result"] == selected_result]
            if selected_verified == "已校验":
                df_filt = df_filt[df_filt["is_verified"] == 1]
            elif selected_verified == "未校验":
                df_filt = df_filt[df_filt["is_verified"] == 0]
            if selected_correct == "正确":
                df_filt = df_filt[df_filt["is_correct"] == 1]
            elif selected_correct == "错误":
                df_filt = df_filt[(df_filt["is_correct"] == 0) & (df_filt["is_verified"] == 1)]

            st.caption(f"筛选后共 {len(df_filt)} 条记录")

            show_cols = ["predict_time", "match_date", "home_team", "away_team",
                         "predict_result", "confidence", "actual_result", "is_correct", "predict_source"]
            col_cn = {
                "predict_time": "预测时间",
                "match_date": "比赛日期",
                "home_team": "主队",
                "away_team": "客队",
                "predict_result": "预测赛果",
                "confidence": "置信度",
                "actual_result": "实际赛果",
                "is_correct": "是否正确",
                "predict_source": "预测来源"
            }
            available_cols = [c for c in show_cols if c in df_filt.columns]
            df_show = df_filt[available_cols].copy()
            df_show.columns = [col_cn.get(c, c) for c in df_show.columns]
            # 预测来源中文化
            if '预测来源' in df_show.columns:
                source_map = {'manual': '手动预测', 'schedule': '赛事日历', 'ai_view': 'AI观点'}
                df_show['预测来源'] = df_show['预测来源'].map(lambda x: source_map.get(str(x), str(x)))
            # 是否正确 & 实际赛果 状态显示
            today = pd.Timestamp.now().normalize()
            if '是否正确' in df_show.columns and '比赛日期' in df_show.columns:
                def format_correct(row):
                    is_ver = df_filt.loc[row.name, 'is_verified'] if 'is_verified' in df_filt.columns else 0
                    is_cor = df_filt.loc[row.name, 'is_correct'] if 'is_correct' in df_filt.columns else None
                    match_dt = pd.to_datetime(row['比赛日期']).normalize()
                    if is_ver == 1:
                        return '✅ 正确' if is_cor == 1 else '❌ 错误'
                    else:
                        return '⏳ 未开赛' if match_dt > today else '🔄 未出结果'
                df_show['是否正确'] = df_show.apply(format_correct, axis=1)
            if '实际赛果' in df_show.columns and '比赛日期' in df_show.columns:
                def format_actual(row):
                    val = row['实际赛果']
                    if pd.isna(val) or val == '' or val is None:
                        match_dt = pd.to_datetime(row['比赛日期']).normalize()
                        return '未开赛' if match_dt > today else '待更新'
                    return val
                df_show['实际赛果'] = df_show.apply(format_actual, axis=1)
            st.dataframe(df_show, use_container_width=True, hide_index=True)
