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

# 使用者昵称（必填，存在 session_state 中）
col_user1, col_user2 = st.columns([1, 4])
with col_user1:
    if "current_user" not in st.session_state:
        st.session_state["current_user"] = ""
    user_input = st.text_input(
        "👤 使用者昵称",
        value=st.session_state["current_user"],
        placeholder="请输入昵称后才能预测",
        max_chars=20,
        key="user_name_input"
    )
    st.session_state["current_user"] = user_input.strip()

current_user = st.session_state["current_user"]
if not current_user:
    st.error("⚠️ 请先在上方输入使用者昵称，才能使用预测功能")
else:
    st.success(f"✅ 当前使用者：{current_user}")

st.divider()

# 顶部信息卡片（引导型，新用户也能看懂）
col1, col2, col3 = st.columns(3)

# 卡片1：今日可预测
today_str = pd.Timestamp.now().strftime("%Y-%m-%d")
today_matches = len(df_schedule[df_schedule["match_date"].dt.strftime("%Y-%m-%d") == today_str])
# 最近开赛日
future_matches = df_schedule[df_schedule["match_date"] >= pd.Timestamp.now().normalize()]
if len(future_matches) > 0:
    next_match_date = future_matches.iloc[0]["match_date"].strftime("%m月%d日")
else:
    next_match_date = "暂无"

with col1:
    st.metric("🎯 今日可预测", f"{today_matches} 场")
    st.caption(f"最近开赛：{next_match_date}")
    st.caption("↓ 下方点击赛事卡片开始预测")

# 卡片2：本周赛事
today = pd.Timestamp.now().normalize()
week_end = today + pd.Timedelta(days=7)
week_matches = len(df_schedule[(df_schedule["match_date"] >= today) & (df_schedule["match_date"] < week_end)])
week_leagues = df_schedule[(df_schedule["match_date"] >= today) & (df_schedule["match_date"] < week_end)]["league_code"].nunique()

with col2:
    st.metric("📅 本周赛事", f"{week_matches} 场")
    st.caption(f"覆盖 {week_leagues} 大联赛")
    st.caption("左右切换查看更多")

# 卡片3：模型验证准确率
with col3:
    st.metric("🤖 模型验证准确率", "53.6%")
    st.caption("基于 3 万场历史数据")
    st.caption("Walk Forward金标准 · 持续优化中")

st.divider()

# 核心功能 Tabs
tab_calendar, tab_predict, tab_track = st.tabs([
    "📅 赛事日历",
    "🔮 单场预测",
    "📋 预测历史"
])

with tab_calendar:
    # 赛事日历公开可见，不需要昵称
    render_schedule_calendar(user_name=current_user if current_user else None)

    # ===== AI 观点区块（折叠式）=====
    st.divider()
    st.info("✨ **AI 出手参考**：三AI共识推荐 + 置信度筛选，点击下方展开查看 ↓")
    with st.expander("🤖 展开 AI 出手参考", expanded=False):
        st.caption("基于当前模型批量计算未来赛事的三AI出手建议，仅供决策参考")
        
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
        
        # 猎鹰Plus版选择（冷门猎手专精）
        col_f1, col_f2 = st.columns([1, 3])
        with col_f1:
            falcon_plus_options = ["关闭", "基础版", "德甲专精", "英超专精"]
            falcon_plus_version = st.selectbox(
                "🦅 猎鹰Plus版",
                options=falcon_plus_options,
                index=0,
                help="冷门猎手专精：基础版全联赛，德甲专精ROI 60%+，英超专精ROI +27%"
            )
        with col_f2:
            if falcon_plus_version != "关闭":
                from ai_intent_module import FALCON_PLUS_CONFIG
                desc = FALCON_PLUS_CONFIG.get(falcon_plus_version, {}).get("desc", "")
                st.caption(f"✅ 已开启猎鹰·{falcon_plus_version}：{desc}")
            else:
                st.caption("💡 猎鹰Plus版：冷门猎手专精，高赔率+高置信度策略")

        if st.button("🔍 批量计算三AI参考", type="primary", key="calc_ai_view"):
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
                            0.28, 0.33,
                            league_code=league
                        )
                        # 预测
                        pred = predict_match(feat, is_home_scene=True)
                        conf = pred["confidence"]
                        result = pred["predict_result"]
                        
                        # 计算AI观点
                        intent = calc_ai_bet_intent(
                            conf, result, 
                            league_code=league,
                            draw_prob=pred.get("prob_draw", 0),
                            draw_odds=None,  # 暂无真实平局赔率
                            advanced_mode=False,
                            falcon_plus_version=None if falcon_plus_version == "关闭" else falcon_plus_version
                        )
                        
                        # 获取显示名称
                        conservative_name = intent["intents"]["保守AI"].get("display_name", "保守AI")
                        conservative_icon = intent["intents"]["保守AI"].get("icon", "🛡️")
                        
                        results.append({
                            "比赛日期": row["match_date"].strftime("%m-%d"),
                            "主队": std_2_cn.get(home_std, home_std),
                            "客队": std_2_cn.get(away_std, away_std),
                            "预测方向": result,
                            "置信度": f"{conf:.1%}",
                            "共识等级": intent["consensus_label"],
                            "激进AI": "✅建议" if intent["intents"]["激进AI"]["will_bet"] else "❌观望",
                            "中立AI": "✅建议" if intent["intents"]["中立AI"]["will_bet"] else "❌观望",
                            f"{conservative_icon} {conservative_name}": "✅建议" if intent["intents"]["保守AI"]["will_bet"] else "❌观望",
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
                
                # 共识等级排序映射
                consensus_order = {
                    "🛡️ 三AI共识": 0,
                    "⚖️ 两AI共识": 1,
                    "🔥 仅激进关注": 2,
                    "无AI出手": 3,
                    "数据不足": 4
                }
                df_ai_view["共识排序"] = df_ai_view["共识等级"].map(consensus_order)
                df_ai_view = df_ai_view.sort_values(["共识排序", "置信度"], ascending=[True, False])
                df_ai_view = df_ai_view.drop(columns=["共识排序"])
                
                # 精选：三AI全票通过
                df_top = df_ai_view[df_ai_view["共识等级"] == "🛡️ 三AI共识"]
                
                if len(df_top) > 0:
                    st.markdown("### 🏆 精选推荐（三AI共识）")
                    st.caption("三个AI同时建议关注，可靠性最高")
                    for _, row in df_top.iterrows():
                        st.markdown(f"""
                        <div style="padding:12px;background:#f0f9eb;border-left:4px solid #67c23a;border-radius:6px;margin-bottom:8px">
                            <b>{row['比赛日期']} · {row['主队']} vs {row['客队']}</b><br>
                            <span style="color:#67c23a">预测：{row['预测方向']}</span> · 
                            置信度：{row['置信度']}
                        </div>
                        """, unsafe_allow_html=True)
                    st.divider()
                
                # 完整表格 + 筛选
                st.markdown("### 📋 全部赛事参考")
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    filter_consensus = st.multiselect(
                        "按共识等级筛选",
                        ["🛡️ 三AI共识", "⚖️ 两AI共识", "🔥 仅激进关注", "无AI出手"],
                        default=["🛡️ 三AI共识", "⚖️ 两AI共识"],
                        key="ai_view_filter"
                    )
                
                df_show = df_ai_view.copy()
                if filter_consensus:
                    df_show = df_show[df_show["共识等级"].isin(filter_consensus)]
                
                st.dataframe(df_show, use_container_width=True, hide_index=True)
                st.caption("💡 市场概率使用默认估算值，实际置信度以真实数据为准；共识度越高参考价值越强")

with tab_predict:
    if not current_user:
        st.info("👆 请先在顶部输入昵称，再使用单场预测功能")
    else:
        render_match_predict_panel(cn_2_std=cn_2_std, std_2_cn=std_2_cn, user_name=current_user)

with tab_track:
    st.subheader("📋 预测历史")
    
    if df_preds.empty:
        st.info("暂无预测记录，去手动预测页做一次预测吧～")
    else:
        # 统计范围选择
        has_user = "user_name" in df_preds.columns
        if has_user:
            col_scope1, col_scope2 = st.columns([1, 1])
            with col_scope1:
                stat_scope = st.radio(
                    "统计范围",
                    ["全部人员", "仅当前用户"],
                    horizontal=True,
                    key="stat_scope",
                    label_visibility="collapsed"
                )
            with col_scope2:
                match_type = st.radio(
                    "比赛类型",
                    ["真实预测", "模拟预测", "全部"],
                    horizontal=True,
                    key="match_type",
                    label_visibility="collapsed"
                )
        else:
            stat_scope = "全部人员"
            match_type = "真实预测"
        
        # 筛选比赛类型 + 人员范围
        df_filtered = df_preds.copy()
        if "is_real_match" in df_preds.columns:
            if match_type == "真实预测":
                df_filtered = df_filtered[df_filtered["is_real_match"] == 1]
            elif match_type == "模拟预测":
                df_filtered = df_filtered[df_filtered["is_real_match"] == 0]
        
        if stat_scope == "仅当前用户" and current_user:
            df_filtered = df_filtered[df_filtered["user_name"] == current_user]
        
        verified = df_filtered[df_filtered["is_verified"] == 1].copy()
        total_v = len(verified)
        correct_v = verified["is_correct"].sum() if total_v > 0 else 0
        acc_v = correct_v / total_v if total_v > 0 else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("预测场次", len(df_filtered))
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
            filt_col1, filt_col2, filt_col3, filt_col4, filt_col5, filt_col6 = st.columns(6)
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
            # 真实比赛筛选
            if "is_real_match" in df_preds.columns:
                selected_real = filt_col5.selectbox("比赛类型", ["全部", "仅真实赛程", "仅娱乐模拟"], key="filt_real")
            else:
                selected_real = "全部"
            # 预测人筛选
            if "user_name" in df_preds.columns:
                all_users = sorted(df_preds["user_name"].dropna().unique().tolist())
                selected_user = filt_col6.selectbox("预测人", ["全部"] + all_users, key="filt_user")
            else:
                selected_user = "全部"

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
            if selected_real == "仅真实赛程" and "is_real_match" in df_filt.columns:
                df_filt = df_filt[df_filt["is_real_match"] == 1]
            elif selected_real == "仅娱乐模拟" and "is_real_match" in df_filt.columns:
                df_filt = df_filt[df_filt["is_real_match"] == 0]
            if selected_user != "全部" and "user_name" in df_filt.columns:
                df_filt = df_filt[df_filt["user_name"] == selected_user]

            st.caption(f"筛选后共 {len(df_filt)} 条记录")

            show_cols = ["predict_time", "match_date", "home_team", "away_team",
                         "predict_result", "confidence", "actual_result", "is_correct",
                         "predict_source", "user_name"]
            col_cn = {
                "predict_time": "预测时间",
                "match_date": "比赛日期",
                "home_team": "主队",
                "away_team": "客队",
                "predict_result": "预测赛果",
                "confidence": "置信度",
                "actual_result": "实际赛果",
                "is_correct": "是否正确",
                "predict_source": "预测来源",
                "user_name": "预测人"
            }
            available_cols = [c for c in show_cols if c in df_filt.columns]
            df_show = df_filt[available_cols].copy()
            # 队名转中文
            df_show["home_team"] = df_show["home_team"].map(lambda x: std_2_cn.get(x, x))
            df_show["away_team"] = df_show["away_team"].map(lambda x: std_2_cn.get(x, x))
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

# 底部更新日志（默认折叠，不占空间）
APP_VERSION = "v1.0.8"
st.divider()

# 免责声明
st.warning("""
**⚠️ 免责声明：本系统仅供机器学习研究与模型验证使用，所有预测结果均为模型算法输出，不构成任何决策建议。严禁用于其他用途。**
历史数据来自 [football-data.co.uk](https://www.football-data.co.uk/)。
""")

with st.expander(f"📝 更新日志 · {APP_VERSION} 🆕", expanded=False):
    st.markdown("""
**v1.0.8 — 2026-07-30（猎鹰Plus版）**
- 猎鹰策略新增双Pro模式：德甲专精/英超专精
- 模型验证页集成Pro版开关，互斥切换
- 策略研究结论全文复核整理

**v1.0.7 — 2026-07-30（猎鹰策略深度优化）**
- 猎鹰策略重新定位：从"最低置信度"到「冷门猎手」
- 二维网格搜索：赔率×置信度35组参数全面测试
- 德甲专精版固定ROI +60.69%，盈利赛季83.3%
- 热门×置信度2×2矩阵验证，确认冷门+高置信有正alpha

**v1.0.6 — 2026-07-30（置信度完美校准）**
- 时间序列交叉验证+Platt缩放，置信度完美校准
- ECE（期望校准误差）0.92%，几乎完美
- 置信度偏差从+4.05%降到-0.00%
- 凯利动态仓位从此可信

**v1.0.5 — 2026-07-30（冷门猎手发现）**
- 发现置信度虚高问题，高置信度区间虚高7-10%
- 发现冷门猎手现象：高赔率+高置信度=正收益
- 反直觉：热门策略跑输市场，冷门策略正收益
- 开始深入研究猎鹰策略优化方向

**v1.0.4 — 2026-07-29（高级模式·超级组合）**
- 👑 新增高级模式，磐石策略升级为超级组合策略
- 🚀 赛季ROI从+0.9%提升至+22.6%
- 🐛 修复实时预测特征维度不匹配Bug（补上时间衰减特征）

**v1.0.3 — 2026-07-29（策略深度探索）**
- 🔍 放开最低出手限制、联赛拆分、德甲专项优化
- 🤖 三大模型横向对比（LightGBM/XGBoost/CatBoost）
- ⚖️ 平局联动策略验证有效

**v1.0.2 — 2026-07-29（随机基准·反直觉研究）**
- 🎲 随机基准对比测试，验证模型真实alpha
- 🧠 三个反直觉现象深入研究
- 📊 置信度校准与分桶分析

**v1.0.1 — 2026-07-29（稳健性·概率校准）**
- 📈 蒙特卡洛稳健性测试（WF版）
- 🎯 Platt概率校准，预测概率更准确
- 🏆 五大联赛独立模型验证

---

**v1.0.0 — 2026-07-29（正式版·泄露完全修复）**

🎉 **里程碑：从实验版到正式版的跨越**
- 两层特征泄露问题完全修复，所有模型全量重建
- 建立完整的三级验证体系，Walk Forward金标准验证上线
- 模型真实能力得到客观评估，结果真实可信

🚨 **重大修复：特征泄露问题完全修复**
- 发现并修复了两层特征泄露：
  - 第一层：shot_on_diff特征（赛后数据被误用为赛前特征）
  - 第二层：h5/a5等基础状态特征（rolling计算未shift，包含本场数据）
- 所有模型已全部重新训练，结果真实可信
- WF赛季重置版验证：模型真实准确率约53.6%，alpha约1-2%

📊 **验证体系升级**
- 建立三级验证体系：全量回测 ⭐ / 赛季重置 ⭐⭐⭐ / WF赛季重置 ⭐⭐⭐⭐⭐
- WF版为金标准，所有策略上线前必须通过WF验证
- 全量回测结果仅供快速筛选，严重偏乐观，不可作为最终结论

💡 **核心发现**
- 足球预测很难，模型alpha有限，符合市场规律
- 安全边际筛选非常有效，能大幅减少亏损
- 出手质量远大于数量，宁可少出手也要选最有把握的

---

**v0.7.1 — 2026-07-29（紧急修复·泄露修复版）**
⚠️ **注意：此版本为过渡修复版，已被v1.0.0取代**
- 发现第二层特征泄露（h5/a5系列rolling未shift）
- 紧急修复12个基础状态特征
- 重新训练所有模型
- 初步验证修复效果

**v0.7 — 2026-07-28（Walk Forward验证体系）**
⚠️ **注意：此版本存在特征泄露，结果不可信**
- 建立Walk Forward滚动验证体系
- 实现赛季重置资金机制
- 策略动物园扩展至36组参数组合
- 初步发现全量回测与WF验证结果差异巨大
- 开始怀疑存在特征泄露问题

**v0.6 — 2026-07-28（特征工程大升级）**
⚠️ **注意：此版本存在特征泄露，结果不可信**
- 新增8个ELO扩展特征（对手加权攻防 + ELO趋势）
- 新增8个时间衰减特征（近期比赛权重更高）
- 特征维度从36维扩展到52维
- 模型准确率进一步提升（当时以为是进步，实际是泄露加重）
- ELO特征总贡献达到16%+

**v0.5 — 2026-07-27（策略动物园上线）**
⚠️ **注意：此版本存在特征泄露，结果不可信**
- 新增策略动物园功能，支持多参数组合回测
- 实现凯利公式动态仓位计算
- 新增三AI策略角色卡（磐石/天秤/猎鹰）
- 策略回测结果看起来非常好（实际是泄露导致的假象）
- 新增共识分析功能

---

**v0.4.4 — 2026-07-27（前端体验大优化）**
- 首屏重构：4张统计卡片 → 3张引导型卡片（今日可预测/本周赛事/模型准确率）
- 赛事日历公开化：无需昵称即可浏览赛程、查看预测结果
- AI出手参考改为折叠式，默认收起，首屏更清爽
- 预测历史增加「真实/模拟」筛选，统计维度更清晰
- 免责声明移至更新日志上方，数据新鲜度移至页面底部
- 修复模型判断依据ELO特征中文映射缺失
- 修复数据看板球队名映射缺失问题

> 🎉 **特别感谢**：「贵阳梅西」提供的新手体验优化建议，本轮首屏引导优化、赛事日历公开化、AI出手参考折叠等功能均来自他的反馈 👏

**v0.4.3 — 2026-07-27（ELO扩展版）**
- 模型准确率提升至 64.47%（较v0.4再提升 0.41%）
- 新增 8 个 ELO 扩展特征（对手加权攻防 + ELO趋势）
- ELO 特征总贡献达 19.97%，实力评估维度更丰富
- 5 大联赛独立模型同步升级，平均准确率 62.80%
- 平局二分类模型同步升级至 45 维

**v0.4.2 — 2026-07-26（架构优化）**
- 功能重定位：预测中心/模型验证/数据看板 三页面职责更清晰
- 赛事日历升级为默认首屏（先粗后精）
- AI出手参考整合进赛事日历tab
- 预测追踪改名为预测历史，弱化博彩感
- 赛事日历增加联赛快速筛选标签（一键切换）
- 赛事日历今明两天默认展开，其他日期折叠，首屏更清爽
- 修复模型验证页多处列名不匹配报错
- 新增版本号显示 + 更新提醒标记
- 批量预测结果可折叠，看完收起不占空间
- 单场卡片增加「查看详情」，展开显示三AI建议

**v0.4.1 — 2026-07-26（体验优化）**
- 赛事日历升级为卡片周视图，支持上/下周快速切换
- 修复赛事日历预测结果全部相同的问题（队名匹配）
- 修复数据看板排序、中文映射、列宽等多项体验问题
- 修复策略回测页列名报错
- 批量预测结果增加去重保护

**v0.4 — 2026-07-26（ELO增强版）**
- 模型准确率提升至 64.06%（累计提升约 2.4%）
- 新增 ELO 评分体系，球队实力评估更精准
- 三模型融合权重优化，泊松模型价值充分释放
- 新增概率校准功能，置信度数值更可信
- 前端文案优化，明确机器学习验证工具定位

**v0.3 — 2026-07-25**
- 新增昵称用户体系
- 真实比赛标记与统计过滤
- 预测去重机制
- 预测历史中文队名修复

**v0.2 — 2026-07-24**
- 联赛独立模型上线
- 泊松进球模型融合
- 平局二分类专项模型
- 数据看板页面

**v0.1 — 初始版本**
- LightGBM 基础预测模型
- 手动预测功能
- 赛事日历模块
- 三AI出手建议计算
    """)
    st.caption("完整开发计划见 DEVELOPMENT_PLAN.md")

# 数据新鲜度提醒（底部小字，仅运维参考）
try:
    import sqlite3
    _conn = sqlite3.connect(DB_PATH)
    latest_date = pd.read_sql("SELECT MAX(match_date) as latest FROM match_feature_final", _conn).iloc[0]['latest']
    _conn.close()
    if latest_date:
        latest_dt = pd.to_datetime(latest_date)
        days_old = (pd.Timestamp.now() - latest_dt).days
        latest_str = latest_dt.strftime("%Y-%m-%d")
        st.caption(f"📊 训练数据截止：{latest_str}（{days_old}天前）")
except:
    pass
