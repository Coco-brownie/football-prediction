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
# 【2026-08-05 修复：联赛码从 LEAGUE_REGISTRY 派生，消除前端旧码（LLA/SER/LIG）漂移】
from common_config import get_all_known_codes, league_name_by_code

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

# 【云端演示模式】部署端无 football.db（165MB 超 GitHub 限制未入库）时历史数据为空：
# 赛事日历/单场预测/预测历史 依赖本地历史数据，云端仅展示项目框架与模型验证结论
IS_CLOUD_DEMO = df_all.empty or df_schedule.empty

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
home_cn_col = "home_team" if "home_team" in df_all.columns else None
if home_cn_col:
    for _, row in df_all[[home_col, home_cn_col]].drop_duplicates().iterrows():
        std = row[home_col]
        cn = row[home_cn_col]
        if std and cn and std not in std_2_cn:
            std_2_cn[std] = cn

away_cn_col = "away_team" if "away_team" in df_all.columns else None
if away_cn_col:
    for _, row in df_all[[away_col, away_cn_col]].drop_duplicates().iterrows():
        std = row[away_col]
        cn = row[away_cn_col]
        if std and cn and std not in std_2_cn:
            std_2_cn[std] = cn

# ==================== 主区域 ====================
st.title("⚽ 足球赛事预测中心")

if IS_CLOUD_DEMO:
    st.warning(
        "**🌐 云端演示模式**：当前环境未包含历史比赛数据库（`football.db` 已瘦身至约 64MB，"
        "随仓库分发；正常部署会自动加载）。\n\n"
        "赛事日历 / 单场预测 / 预测历史 依赖历史数据；若云端仍显示本提示，"
        "说明 `football.db` 未随部署生效，请在本地运行完整功能。\n\n"
        "**完整功能**：`streamlit run streamlit_dash/⚽_预测中心.py` "
        "（本地 `football.db` 含全部历史数据）。"
    )

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
    st.caption("💡 输入昵称后，你的预测会自动保存到**专属预测历史**（含个人准确率、连胜、待开奖提醒）；不输入也能先浏览下方赛事日历。")
else:
    st.success(f"✅ 当前使用者：{current_user}")

st.divider()

# ===== 下一步引导条（新老用户差异化 · 方向1：指引性） =====
if current_user and ("user_name" in df_preds.columns) and not df_preds.empty:
    _my_df = df_preds[df_preds["user_name"] == current_user]
    _my_n = len(_my_df)
    _pending_n = int((_my_df["is_verified"] == 0).sum()) if "is_verified" in _my_df.columns else 0
    if _my_n > 0:
        if _pending_n > 0:
            st.info(f"👋 **{current_user}**，你已有 **{_my_n}** 条预测记录，其中 **{_pending_n}** 场待开奖 → 去 **📋 预测历史** 查看结果")
        else:
            st.info(f"👋 **{current_user}**，你已有 **{_my_n}** 条预测记录 → 去 **📋 预测历史** 回顾战绩")
    else:
        st.info("👋 新朋友？三步上手：**① 输入昵称 ✅ ② 下方赛事日历选一场比赛 ③ 点「🔮 预测」**，预测会自动保存到你的预测历史。")
elif current_user:
    st.info("👋 新朋友？三步上手：**① 输入昵称 ✅ ② 下方赛事日历选一场比赛 ③ 点「🔮 预测」**，预测会自动保存到你的预测历史。")
else:
    st.info("👋 你好！输入上方昵称即可开始预测；未输入昵称也能先浏览下方赛事日历。")

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
    st.metric("🎯 今日赛事", f"{today_matches} 场")
    if today_matches == 0:
        st.caption(f"📭 今日暂无赛程（休赛日）· 最近开赛 {next_match_date}")
    else:
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
    st.metric("🤖 模型验证准确率", "51.75%")
    st.caption("基于 50,752 场外样本 · WF金标准统一验证")
    st.caption("argmax口径 · 主胜基准 45.8% · 详见统一验证报告")

st.divider()

# ===== 📖 一句话看懂本系统策略（普通人版，折叠） =====
with st.expander("📖 一句话看懂本系统策略（给普通人）", expanded=False):
    st.markdown(
        "> **本项目为机器学习研究项目**：用 AI 在五大联赛里识别「模型最有把握 + 赔率给得划算」的比赛，"
        "所有数字均为历史回测/外样本验证数据，不代表未来表现，不构成任何建议。**"
    )
    st.markdown(
        "- 🦅 **猎鹰**（研究·高赔冷门档）：外样本 357 次出手（赔率≥2.2 置信≥55%）样本内 ROI +15.51%；"
        "真实可成交赔率口径（Pinnacle 终盘）下样本偏薄、未过 Bonferroni → **待 2026-2027 独立期复现，当前仅研究参考**。\n"
        "- 🪨 **磐石Pro**（研究·高命中档）：研究样本 113 场 +13.16%，聚类 CI 含 0 → 统计显著不成立，仅研究观察。\n"
        "- ⚖️ **价值**（已移除）：曾有研究 alpha，但无后见之明验证不成立（C-015），**已从组合移除**（C-016）。\n\n"
        "**风险提示**：以上均为历史研究数据，不代表未来收益；任何策略在独立期复现前不构成任何建议。"
    )

# ===== 🎯 今日 AI 看点（方向1：首屏高价值内容，给你回来看的理由） =====
def _compute_ai_focus(days, falcon_plus_version):
    """计算未来days天五大联赛三AI参考（今日看点 / AI出手参考共用，带结果缓存）
    修复：build_feature_by_teams 传【3个默认原始赔率】+ 开身价特征(57维)，
    此前误传 0.28/0.33 两个概率且缺第3个赔率 → TypeError → 全部落"数据不足"。"""
    from ai_intent_module import calc_ai_bet_intent
    from match_predict import predict_match
    from streamlit_dash.feature_auto_build import build_feature_by_teams

    _t = pd.Timestamp.now().normalize()
    _end = _t + pd.Timedelta(days=days)
    df_up = df_schedule[
        (df_schedule["match_date"].dt.normalize() >= _t) &
        (df_schedule["match_date"].dt.normalize() <= _end)
    ].copy().sort_values("match_date")
    df_up = df_up[df_up["league_code"].isin(get_all_known_codes())]
    _rows = []
    for _, r in df_up.iterrows():
        try:
            feat = build_feature_by_teams(
                df_all, r["home_team"], r["away_team"],
                2.5, 3.3, 3.0, r["league_code"], use_value_features=True
            )
            pred = predict_match(feat, is_home_scene=True)
            intent = calc_ai_bet_intent(
                pred["confidence"], pred["predict_result"],
                league_code=r["league_code"],
                draw_prob=pred.get("prob_draw", 0), draw_odds=None,
                advanced_mode=False,
                falcon_plus_version=None if falcon_plus_version == "关闭" else falcon_plus_version
            )
            _rows.append({
                "比赛日期": r["match_date"].strftime("%m-%d"),
                "主队": std_2_cn.get(r["home_team"], r["home_team"]),
                "客队": std_2_cn.get(r["away_team"], r["away_team"]),
                "预测方向": pred["predict_result"],
                "置信度": float(pred["confidence"]),
                "共识等级": intent["consensus_label"],
            })
        except Exception:
            pass
    return pd.DataFrame(_rows)

# 今日看点：默认未来2天；结果缓存到 session_state（交互不丢，跨天自动失效）
_today_tag = pd.Timestamp.now().strftime("%Y%m%d")
_focus_cache_key = f"ai_focus_cache_{_today_tag}"
if _focus_cache_key not in st.session_state:
    st.session_state[_focus_cache_key] = None

with st.expander("🎯 **今日 AI 看点**（未来 2 天 · 三AI共识精选 · 每天自动更新）", expanded=True):
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.caption("三个 AI（🦅猎鹰 / ⚖️天秤 / 🪨磐石）共同看好的赛事（研究参考，不构成任何建议）。基于默认赔率降级预测。"
                   "三 AI 为同一模型按风险偏好分三档的研究性输出；价值策略已从组合移除（C-016），不再纳入。")
    with col_h2:
        refresh_focus = st.button("🔄 刷新今日看点", key="btn_refresh_focus", use_container_width=True)

    if refresh_focus:
        with st.spinner("正在计算今日 AI 看点（约 10~30 秒）..."):
            st.session_state[_focus_cache_key] = _compute_ai_focus(2, "关闭")

    _focus_df = st.session_state.get(_focus_cache_key)
    # 空状态分类：先判断未来2天是否有五大联赛赛程（决定是「暂停更新」还是「有赛程但模型保守」）
    _t_now = pd.Timestamp.now().normalize()
    _upcoming_cnt = len(df_schedule[
        (df_schedule["match_date"].dt.normalize() >= _t_now) &
        (df_schedule["match_date"].dt.normalize() <= _t_now + pd.Timedelta(days=2)) &
        (df_schedule["league_code"].isin(get_all_known_codes()))
    ])
    if _focus_df is not None and len(_focus_df) > 0:
        _top = _focus_df[_focus_df["共识等级"].isin(["🤝 三AI共识", "👥 两AI看好"])].sort_values("置信度", ascending=False).head(5)
        if len(_top) > 0:
            for _, r in _top.iterrows():
                _cons_color = "#67c23a" if r["共识等级"] == "🤝 三AI共识" else "#409eff"
                st.markdown(
                    f"<div style='padding:10px 12px;background:#f7fafd;border-left:4px solid {_cons_color};"
                    f"border-radius:6px;margin-bottom:6px'>"
                    f"<b>{r['比赛日期']} · {r['主队']} vs {r['客队']}</b>"
                    f"<span style='float:right;font-size:12px;color:{_cons_color};font-weight:600'>{r['共识等级']}</span><br>"
                    f"<span style='font-size:13px;color:#555'>预测 <b>{r['预测方向']}</b> · 置信度 {r['置信度']:.1%}</span>"
                    f"</div>", unsafe_allow_html=True)
            st.caption("💡 想要完整版？在下方 **📅 赛事日历** tab 的「AI 出手参考」可查看全部场次 + 三AI明细 + 猎鹰Plus。")
        else:
            st.caption("未来 2 天暂无三AI共识赛事（模型保守优先），可点击「刷新」或到 AI 出手参考查看完整列表。")
    else:
        if _upcoming_cnt == 0:
            st.info("📭 未来 2 天暂无五大联赛赛事，今日 AI 看点暂停更新（有新赛程后自动恢复）。")
        else:
            st.caption("👆 点击右上「刷新今日看点」生成今日精选（首次约需 10~30 秒计算）；若刷新后仍为空，说明近期场次模型评估保守或历史数据不足。")

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
    st.info("✨ **AI 研究参考**：三AI共识 + 磐石Pro，点击下方展开查看 ↓（纯研究输出，不构成任何建议）")
    with st.expander("🤖 展开 AI 研究参考", expanded=False):
        st.caption("基于当前模型批量计算未来赛事的**AI 研究参考**（三AI + 磐石Pro），仅供研究，不构成任何决策建议")
        st.caption("🦅⚖️🪨 **三AI**（猎鹰/天秤/磐石）= 同一模型按风险偏好分三档；👑 **磐石Pro** = 保守AI高级模式（德甲/意甲精选，样本少、统计不显著，仅研究观察）；💎 **价值** 因无后见之明验证不成立已从组合移除（C-015/016）")
        st.caption("⚠️ 以上为历史回测数字（三AI网格样本内固定ROI +14.5%/+9.4%/+15.8%），不代表未来收益；不构成任何仓位建议。")
        
        # 日期范围选择（按周划分，贴合周/半月数据更新节奏；不再用抽象「未来N天」）
        import datetime as _dt
        _today_d = _dt.date.today()
        _monday = _today_d - _dt.timedelta(days=_today_d.weekday())
        _sunday = _monday + _dt.timedelta(days=6)
        _next_monday = _sunday + _dt.timedelta(days=1)
        _next_sunday = _next_monday + _dt.timedelta(days=6)
        range_choice = st.radio(
            "预测范围",
            ["本周", "下周", "本周+下周"],
            index=2,
            horizontal=True,
            key="ai_view_range"
        )
        if range_choice == "本周":
            _start_d, _end_d = _today_d, _sunday
            _range_label = f"本周（{_monday.month}/{_monday.day} ~ {_sunday.month}/{_sunday.day}）"
        elif range_choice == "下周":
            _start_d, _end_d = _next_monday, _next_sunday
            _range_label = f"下周（{_next_monday.month}/{_next_monday.day} ~ {_next_sunday.month}/{_next_sunday.day}）"
        else:
            _start_d, _end_d = _today_d, _next_sunday
            _range_label = f"本周+下周（{_today_d.month}/{_today_d.day} ~ {_next_sunday.month}/{_next_sunday.day}）"

        # 筛选赛程（按日期范围；跨月/跨年用 date 比较天然支持）
        df_upcoming = df_schedule[
            (df_schedule["match_date"].dt.date >= _start_d) &
            (df_schedule["match_date"].dt.date <= _end_d)
        ].copy().sort_values("match_date")

        # 过滤五大联赛（有模型的；兼容 B体系 db_code 与旧翻译码）
        valid_leagues = get_all_known_codes()
        df_upcoming = df_upcoming[df_upcoming["league_code"].isin(valid_leagues)]

        st.info(f"📅 {_range_label} · 共 {len(df_upcoming)} 场五大联赛赛事")
        
        # 猎鹰Plus（研究档：冷门猎手专精，待独立期复现）
        # 【2026-08-21 简化】改为单一 checkbox（默认关闭，研究档）；基础版样本内 +15.79% 但产品口径未过 Bonferroni；
        #  德甲/英超专精为实验性 Beta（外样本存疑、每季仅约3~5场），不作为产品化依据，已从界面移除，
        #  避免用户面对「关闭/基础版/德甲/英超」四选一不知所措。
        enable_falcon_plus = st.checkbox(
            "🦅 猎鹰Plus（研究·高赔冷门精选档）",
            value=False,
            key="ai_view_falcon_plus",
            help="猎鹰Plus（研究档）= 猎鹰（激进档）加强：赔率≥2.5 + 置信≥55% 的冷门猎手精选。"
                 "样本内固定ROI +15.79%，但真实可成交赔率口径（Pinnacle 终盘）下未过 Bonferroni，待 2026-2027 独立期复现；"
                 "开启仅作研究观察，不构成任何建议。德甲/英超专精为实验性 Beta，样本极少，不提供选择。"
        )
        falcon_plus_version = "基础版" if enable_falcon_plus else "关闭"
        st.caption("💡 猎鹰Plus（研究档）= 冷门猎手专精（高赔率+高置信），样本内固定ROI +15.79%（待独立期复现）；"
                   "关闭则退回基础猎鹰。批量场景无真实赔率，实际按置信度门槛（≥55%）生效。")

        # 【方向6-流畅性】结果缓存到 session_state：切tab/滑滑块等交互后不丢，无需重算
        ai_view_cache_key = "ai_view_result_cache"
        _ai_view_sig = (range_choice, falcon_plus_version)

        if st.button("🔍 批量计算 AI 研究参考", type="primary", key="calc_ai_view"):
            if len(df_upcoming) == 0:
                st.warning("暂无符合条件的赛事")
            else:
                from ai_intent_module import calc_ai_bet_intent, CONSENSUS_AI_KEYS, compute_value_signal, AI_CONFIGS
                from match_predict import predict_match
                from streamlit_dash.feature_auto_build import build_feature_by_teams
                
                results = []
                progress_bar = st.progress(0)
                
                for idx, (_, row) in enumerate(df_upcoming.iterrows()):
                    home_std = row["home_team"]
                    away_std = row["away_team"]
                    league = row["league_code"]
                    
                    try:
                        # 【修复】build_feature_by_teams 传 3 个默认原始赔率 + 开身价特征(57维)；
                        #  此前误传 0.28/0.33 两个概率且缺第3个赔率 → TypeError → 全部落"数据不足"
                        feat = build_feature_by_teams(
                            df_all, home_std, away_std,
                            2.5, 3.3, 3.0, league, use_value_features=True
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
                        
                        # 磐石Pro（保守AI + 高级模式：德甲/意甲精选，样本少、仅研究观察）
                        pro_intent = calc_ai_bet_intent(
                            conf, result,
                            league_code=league,
                            draw_prob=pred.get("prob_draw", 0),
                            draw_odds=None,
                            advanced_mode=True,
                            falcon_plus_version=None if falcon_plus_version == "关闭" else falcon_plus_version
                        )
                        pro_status = pro_intent["intents"]["保守AI"]["will_bet"]

                        # 价值策略（独立小仓：EV≥10% 出手，组合权重 ≤5%）
                        val_sig = compute_value_signal(pred.get("prob_home_win", 0), 2.5)

                        # 三AI列名动态生成（注册表驱动；猎鹰Plus开启时激进列自动带版本名）
                        ai_cols = {}
                        for _k in CONSENSUS_AI_KEYS:
                            _info = intent["intents"][_k]
                            _nm = _info["display_name"] if _info.get("is_falcon_plus") else AI_CONFIGS[_k]["name"]
                            ai_cols[_k] = f"{_info['icon']} {_nm}"

                        _row = {
                            "比赛日期": row["match_date"].strftime("%m-%d"),
                            "主队": std_2_cn.get(home_std, home_std),
                            "客队": std_2_cn.get(away_std, away_std),
                            "预测方向": result,
                            "置信度": float(conf),
                            "共识等级": intent["consensus_label"],
                        }
                        for _k in CONSENSUS_AI_KEYS:
                            _row[ai_cols[_k]] = "✅建议" if intent["intents"][_k]["will_bet"] else "❌观望"
                        _row["💎 价值"] = f"✅ EV{val_sig['ev']:+.0%}" if val_sig["will_bet"] else f"— EV{val_sig['ev']:+.0%}"
                        _row["👑 磐石Pro"] = "✅建议" if pro_status else "❌观望"
                        results.append(_row)
                    except Exception:
                        # 与成功分支同构的列（注册表驱动），避免 pd.DataFrame 列错位/NaN 混乱
                        _er_cols = {_k: f"{AI_CONFIGS[_k]['icon']} {AI_CONFIGS[_k]['name']}" for _k in CONSENSUS_AI_KEYS}
                        _er = {
                            "比赛日期": row["match_date"].strftime("%m-%d"),
                            "主队": std_2_cn.get(home_std, home_std),
                            "客队": std_2_cn.get(away_std, away_std),
                            "预测方向": "—",
                            "置信度": None,
                            "共识等级": "数据不足",
                        }
                        for _k in CONSENSUS_AI_KEYS:
                            _er[_er_cols[_k]] = "—"
                        _er["💎 价值"] = "—"
                        _er["👑 磐石Pro"] = "—"
                        results.append(_er)
                    
                    progress_bar.progress((idx + 1) / len(df_upcoming))
                
                progress_bar.empty()
                df_ai_view = pd.DataFrame(results)
                st.session_state[ai_view_cache_key] = {"sig": _ai_view_sig, "df": df_ai_view}

                
        # 【方向6-流畅性】结果从 session_state 读取（交互不丢）；参数变化提示重算
        if ai_view_cache_key in st.session_state and st.session_state[ai_view_cache_key] is not None:
            _cached = st.session_state[ai_view_cache_key]
            if _cached["sig"] != _ai_view_sig:
                st.caption("⚠️ 已调整「预测范围 / 猎鹰Plus版」参数，点击上方按钮重新计算。")
            else:
                df_ai_view = _cached["df"].copy()
                # 共识等级排序映射（对齐 ai_intent_module 实际输出标签）
                consensus_order = {
                    "🤝 三AI共识": 0,
                    "👥 两AI看好": 1,
                    "🔥 仅激进关注": 2,
                    "👑 仅Pro关注": 2,
                    "❌ 无AI出手": 3,
                    "数据不足": 4,
                }
                df_ai_view["共识排序"] = df_ai_view["共识等级"].map(consensus_order)
                df_ai_view = df_ai_view.sort_values(["共识排序", "置信度"], ascending=[True, False])
                df_ai_view = df_ai_view.drop(columns=["共识排序"])

                # ===== 出手统计汇总：让用户一眼知道「AI 评估了全部、只有这些推荐」，避免空表误判 =====
                _n_total = len(df_ai_view)
                _n_cons = int((df_ai_view["共识等级"] == "🤝 三AI共识").sum())
                _n_val = int(df_ai_view["💎 价值"].astype(str).str.startswith("✅").sum())
                _n_pro = int((df_ai_view["👑 磐石Pro"] == "✅建议").sum())
                _n_none = int((df_ai_view["共识等级"] == "❌ 无AI出手").sum())
                _n_skip = int((df_ai_view["共识等级"] == "数据不足").sum())
                sc1, sc2, sc3, sc4, sc5 = st.columns(5)
                sc1.metric("📅 已评估场次", _n_total)
                sc2.metric("🤝 三AI共识", f"{_n_cons} 场")
                sc3.metric("💎 价值出手", f"{_n_val} 场")
                sc4.metric("👑 磐石Pro", f"{_n_pro} 场")
                sc5.metric("❌ 无AI出手", f"{_n_none} 场")
                if _n_skip > 0:
                    st.caption(f"⚠️ 其中 {_n_skip} 场因历史数据不足未能评估（非模型不推荐）；"
                               f"{_n_val} 场触发 💎价值、{_n_pro} 场触发 👑磐石Pro（独立小仓，勿据此下重注）")

                # 精选：三AI全票通过
                df_top = df_ai_view[df_ai_view["共识等级"] == "🤝 三AI共识"]

                if len(df_top) > 0:
                    st.markdown("### 🔬 三AI研究共识")
                    st.caption("三个 AI 研究输出一致（研究参考，不构成任何建议）")
                    for _, row in df_top.iterrows():
                        st.markdown(f"""
                        <div style="padding:12px;background:#f0f9eb;border-left:4px solid #67c23a;border-radius:6px;margin-bottom:8px">
                            <b>{row['比赛日期']} · {row['主队']} vs {row['客队']}</b><br>
                            <span style="color:#67c23a">预测：{row['预测方向']}</span> · 
                            置信度：{row['置信度']}
                        </div>
                        """, unsafe_allow_html=True)
                    st.divider()
                else:
                    # 【2026-08-21 空态明确化：不是程序出错，而是模型保守没出手】
                    st.info("📭 本批**没有「三AI共识」场次**——AI 已评估全部赛事，但三档未同时出手（模型保守优先、宁缺毋滥）。"
                            "各 AI 的出手/观望状态见下方完整表格。")

                # 完整表格 + 筛选（默认全选，杜绝空表让用户误以为程序出错）
                st.markdown("### 📋 全部赛事参考")
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    _all_cons = ["🤝 三AI共识", "👥 两AI看好", "🔥 仅激进关注", "👑 仅Pro关注", "❌ 无AI出手", "数据不足"]
                    filter_consensus = st.multiselect(
                        "按共识等级筛选（默认全选）",
                        _all_cons,
                        default=_all_cons,
                        key="ai_view_filter"
                    )
                
                df_show = df_ai_view.copy()
                if filter_consensus:
                    df_show = df_show[df_show["共识等级"].isin(filter_consensus)]
                if "置信度" in df_show.columns:
                    df_show["置信度"] = df_show["置信度"].apply(
                        lambda x: f"{x:.1%}" if isinstance(x, (int, float)) and not pd.isna(x) else "—")
                
                st.dataframe(df_show, use_container_width=True, hide_index=True)
                st.caption("💡 市场概率使用默认估算值（主胜2.5/平3.3/客3.0），实际以真实赔率为准；💎价值列 EV 基于模型概率×默认赔率，为历史口径研究参考，不构成任何建议；三AI共识度仅供参考")

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
                # 【方向2-与我有关】有昵称时默认只看"我"的预测，而不是所有人的
                _scope_default = 1 if current_user else 0
                stat_scope = st.radio(
                    "统计范围",
                    ["全部人员", "仅当前用户"],
                    index=_scope_default,
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

        # ===== 方向3：待开奖 + 最近开奖（给你回来看的理由） =====
        if stat_scope == "仅当前用户" and current_user and "is_verified" in df_filtered.columns:
            _pending = df_filtered[df_filtered["is_verified"] != 1]
            _verified_user = df_filtered[df_filtered["is_verified"] == 1]
            _recent7 = _verified_user[
                pd.to_datetime(_verified_user["match_date"], errors="coerce").fillna(pd.Timestamp.min)
                >= (pd.Timestamp.now().normalize() - pd.Timedelta(days=7))
            ]
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                if len(_pending) > 0:
                    st.warning(f"🔔 **{len(_pending)} 场待开奖**：开赛后回来刷新即可看到结果")
                    for _, r in _pending.head(3).iterrows():
                        st.caption(f"· {str(r['match_date'])[:10]} {std_2_cn.get(r['home_team'], r['home_team'])} vs {std_2_cn.get(r['away_team'], r['away_team'])} → 预测 **{r['predict_result']}**")
                else:
                    st.success("✅ 当前无待开奖预测")
            with col_p2:
                if len(_recent7) > 0:
                    _rc = int(_recent7["is_correct"].sum())
                    st.success(f"🎉 **近 7 天开奖 {len(_recent7)} 场**：预测正确 **{_rc}** 场")
                else:
                    st.caption("近 7 天暂无开奖记录")

        # ===== 方向2：我的战绩卡片（与我有关） =====
        if stat_scope == "仅当前用户" and current_user and len(df_filtered) >= 1:
            _uv_verified = df_filtered[df_filtered["is_verified"] == 1]
            _uv_recent10 = _uv_verified.sort_values("match_date", ascending=False).head(10) if len(_uv_verified) > 0 else _uv_verified
            _recent_acc = _uv_recent10["is_correct"].mean() if len(_uv_recent10) > 0 else None
            _streak = 0
            for _, r in _uv_verified.sort_values("match_date", ascending=False).iterrows():
                if r["is_correct"] == 1:
                    _streak += 1
                else:
                    break
            _best_league = "—"
            if len(_uv_verified) >= 3 and "league_code" in _uv_verified.columns:
                _lg_acc = _uv_verified.groupby("league_code")["is_correct"].agg(["mean", "count"])
                _lg_acc = _lg_acc[_lg_acc["count"] >= 3]
                if len(_lg_acc) > 0:
                    _best_lg = _lg_acc["mean"].idxmax()
                    _best_league = f"{league_name_by_code(_best_lg)}（{_lg_acc.loc[_best_lg, 'mean']:.0%}）"
            st.markdown("##### 👤 我的战绩")
            gc1, gc2, gc3, gc4 = st.columns(4)
            gc1.metric("📈 近10场准确率", f"{_recent_acc:.0%}" if _recent_acc is not None else "—")
            gc2.metric("🔥 当前连胜", f"{_streak} 连胜" if _streak > 0 else "0")
            gc3.metric("🏆 最准联赛", _best_league)
            gc4.metric("🧮 已完赛校验", len(_uv_verified))

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
                db_code_names = {code: league_name_by_code(code) for code in get_all_known_codes()}
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
APP_VERSION = "v1.4.0"
st.divider()

# 免责声明
st.warning("""
**⚠️ 免责声明：本系统仅供机器学习研究与模型验证使用，所有预测结果均为模型算法输出，不构成任何决策建议。严禁用于其他用途。**
历史数据来自 [football-data.co.uk](https://www.football-data.co.uk/)。
""")

with st.expander(f"📝 更新日志 · {APP_VERSION} 🆕", expanded=False):
    st.markdown("""
**v1.4.0 — 2026-08-07（口径收敛 · 研究侧叙述统一 · 猎鹰降级）**
- 🔬 **猎鹰/磐石/价值 全部降级为研究侧叙述**：移除「定稿/验证/产品化」等定性描述（仅保留研究侧措辞）；猎鹰标注「真实可成交赔率口径（Pinnacle 终盘）未过 Bonferroni，待 2026-2027 独立期复现」；磐石Pro 统一为 113 场口径、标注「统计不显著·仅研究观察」；价值标注「已从组合移除（C-015/016）」
- 🚫 **看板删除产品化暗示**：移除全部暗示性表述（含出手类/仓位类/过度自信类措辞），猎鹰Plus 改为研究档并默认关闭
- 📐 **版本号统一为 1.4.0**（config.json 为权威出口，README/CHANGELOG 同步）
- 📖 单场预测/共识分析/决策参考 全部改为「历史回测观察 · 研究参考 · 不构成任何建议」措辞

**v1.3.3 — 2026-08-05（单场/出手参考体验修正）**
- 🎯 单场预测卡片：模型概率上位为主角（预测方向 + 主胜39%·平28%·客胜33%），「历史命中率」降级为「该档位」小字校准说明——修复旧版用户误读「主胜置信度41.4%」与模型概率对不上的困惑
- 🦅 猎鹰Plus 简化为「启用基础版」单选框（默认开启，已验证+15.8%）：移除「关闭/德甲/英超」四选一，德甲/英超专精（实验性Beta存疑）不再作为选择项
- 📊 出手参考空态根治：批量计算后默认显示全部赛事 + 顶部出手统计汇总（已评估/三AI共识/价值/磐石Pro/无AI出手）；无共识场次明确提示「模型保守没出手」而非空表；数据不足行列名与成功行统一
- 📅 未来天数 → 按周划分：本周 / 下周 / 本周+下周，贴合周/半月数据更新节奏

**v1.3.2 — 2026-08-05（组合参考升级 · AI注册表驱动）**
- 🔧 **出手参考升级为「组合参考」**：不再只是三AI，新增 💎价值（EV≥10% 独立小仓 ≤5%）与 👑磐石Pro（德甲/意甲精选）两条腿，与决策参考的组合规则闭环
- 🧱 **AI 注册表驱动架构**：新增 `CONSENSUS_AI_KEYS` + `compute_value_signal`，共识标签/出手参考列/角色卡全部由注册表自动生成，未来加第四AI/第五AI只需登记一行配置，渲染代码零改动
- 🔄 共识逻辑动态化：三档仍显示「三AI共识/两AI看好」，扩展到 4AI/5AI 时自动变「4AI共识」等

**v1.3.1 — 2026-08-04（看板指引性·流畅性·用户粘性优化）**
- 🎯 新增「今日 AI 看点」：首屏直达未来2天三AI共识精选，每天自动更新，给你回来看的理由
- 👋 新老用户差异化引导条：老用户显示「待开奖 X 场」钩子，新用户三步上手引导
- 🔧 修复 AI 出手参考失效 Bug：特征调用缺第3赔率 → 全部"数据不足"，现已修复；结果持久化，切 tab 不丢
- 👤 预测历史默认切到「仅当前用户」+「我的战绩」卡片（近10场准确率/当前连胜/最准联赛）
- 🔔 预测历史新增「待开奖 + 最近7天开奖」提醒，开赛后来刷新看结果
- 💾 预测保存成功/失败明确反馈 + 批量预测成功/跳过统计

**v1.3.0 — 2026-08-04（看板全面修复 · 方案A：置信度改历史命中率）**
- 🔥 **方案A：置信度 ≠ 主胜概率**——置信度改为查「WF 分桶历史真实命中率」（根库 wf_confidence_accuracy）：拜仁主胜 52.3% → 置信度 55.2%；巴萨主胜 48.6% → 置信度 41.4%。语义＝「这个档位历史上命中了多少」，更保守真实
- 🤖 **AI 出手参考保守化**：三 AI 阈值梯度（激进50% / 中立55% / 保守60%，历史命中率口径）+ 平局整体压一档；纯参考、绝不替您决定出手（UI 明确标注「是否出手由您独立判断」）
- 🧩 预测/批量预测失效根治：match_feature_final 中英混杂队名 → feature_auto_build 中英双向匹配
- 🏷 27 支遗留英文队名映射补齐（Reggina / MGladbach / Cottbus 等），看板中文全覆盖
- 🗄 多库根治：数据看板 3 处误连次库修复，全看板统一连接根库 football.db
- 📐 赛事日历特征 52→57 维对齐（补身价特征），预测不再崩溃
- 📊 新增验证表：wf_confidence_accuracy（置信度分桶 vs 真实准确率）+ league_independent_wf（联赛难度排行）
- 🔴 身价特征 value_ratio 3981 倍漂移修复：ASOF 快照重建 + 500万下限清洗，全模型重训 57 维

**v1.2.3 — 2026-08-03（第三版审计紧急修复）**
- 🔥 P0 特征顺序错位修复：在线端/平局模型/校准器三处顺序统一，联赛独热放最后
- 全链路统一顺序：基础→交锋→概率→平局率→ELO→ELO扩展→时间衰减→联赛→身价
- 统一出口：全部从 common_config.get_feature_list() 读取
- P1：daily_verify 启动即崩、backtest LEAGUE_FIX_COLS 取错列 修复
- P2：身价特征统一出口、前端合规化（凯利弱化为理论参考）、联赛独立模型删除

**v1.2.2 — 2026-08-02（全量数据扩容 + 转会窗alpha验证）**
- 数据扩容：53,308 场（2000-2026），样本量 +32%
- 身价特征 v2：中性默认值方案，覆盖率 68.2%
- 转会窗 alpha：Elo差+身价差双重验证，长期存在（9月 +17.9% vs 其他月 -5.8%）

**v1.2.0 — 2026-07-30（平局专项优化·第一阶段）**
- 平局二分类模型 ROI：-2.23% → +7.32%（17 个最优特征）
- 盈利赛季 6/9，赛季均 ROI +8.09%

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
