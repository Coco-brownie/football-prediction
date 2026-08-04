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
from team_mapping_v2 import LEAGUE_TEAM_MAP, CFG_2_DB_CODE, get_team_cn_name_v2
from common_config import LEAGUE_REGISTRY
from common.usage_tracker import track
from streamlit_dash.predict_module import save_prediction_to_db
from common.data_loader import _empty_df_with, _FEATURE_EMPTY_COLS, _SCHEDULE_EMPTY_COLS

# 数据库编码 -> 配置编码反向映射
DB_2_CFG = {v: k for k, v in CFG_2_DB_CODE.items()}

# ========== 配置 ==========
SCRIPT_PATH = os.path.abspath(__file__)
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_PATH))
DB_PATH = os.path.join(ROOT_DIR, "football.db")

# 联赛显示配置
# 【2026-08-05 修复：以 B体系 db_code（E0/D1/SP1/I1/F1）为主键，并兼容旧翻译码（LLA/SER/LIG），
#  消除与 LEAGUE_REGISTRY 的码漂移；UCL 为赛程独有联赛，单独保留。】
_LEAGUE_NAME_COLOR = {
    "英超": "#37003c", "德甲": "#d20515", "西甲": "#ee8707",
    "意甲": "#008fd7", "法甲": "#009045", "欧冠": "#0015a8",
}
LEAGUE_DISPLAY = {}
for _v in LEAGUE_REGISTRY.values():
    _disp = {"name": _v["name"], "color": _LEAGUE_NAME_COLOR.get(_v["name"], "#666")}
    LEAGUE_DISPLAY[_v["db_code"]] = _disp
    if _v.get("old_db_code"):
        LEAGUE_DISPLAY[_v["old_db_code"]] = _disp
LEAGUE_DISPLAY["UCL"] = {"name": "欧冠", "color": "#0015a8"}


def _league_match_codes(tab_code):
    """联赛tab码 → 该联赛可能出现的全部码（db_code + 旧码），供筛选匹配"""
    if tab_code == "UCL":
        return ["UCL"]
    for v in LEAGUE_REGISTRY.values():
        if tab_code == v["db_code"]:
            codes = [v["db_code"]]
            if v.get("old_db_code") and v["old_db_code"] not in codes:
                codes.append(v["old_db_code"])
            return codes
    return [tab_code]

# 降级预测默认值（赔率、射门数据缺失时填充）
# 降级预测默认赔率（原始赔率值，会自动去水转概率）
DEFAULT_HOME_ODDS = 2.5
DEFAULT_DRAW_ODDS = 3.5
DEFAULT_AWAY_ODDS = 3.0
DEFAULT_SHOT_DIFF = 0.0

LEAGUE_OPTIONS = list(LEAGUE_DISPLAY.keys())
LEAGUE_NAMES = {k: v["name"] for k, v in LEAGUE_DISPLAY.items()}


@st.cache_data(ttl=86400)
def load_schedule_data():
    """加载全部赛程数据（缓存1天），自动转换中文队名
    【云端无 db 降级】表不存在时返回带列空 df，页面显示"无数据"而非崩溃"""
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM match_schedule", conn)
        conn.close()
    except Exception:
        return _empty_df_with(_SCHEDULE_EMPTY_COLS)
    df["match_date"] = pd.to_datetime(df["match_date"])

    # 英文队名转中文（【2026-08-08 修复】改用统一映射接口 get_team_cn_name_v2：
    #  内部完成 db_code→配置键 转换（SP1→LLA/I1→SER/F1→LIG）+ 归一化匹配（变音/前缀/全称/简写），
    #  修复西甲/法甲/意甲仅精确匹配导致的中文队名未映射问题）
    def get_cn_name(league_db_code, eng_name):
        return get_team_cn_name_v2(league_db_code, eng_name, print_miss=False)

    df["home_team_cn"] = df.apply(lambda r: get_cn_name(r["league_code"], r["home_team"]), axis=1)
    df["away_team_cn"] = df.apply(lambda r: get_cn_name(r["league_code"], r["away_team"]), axis=1)

    return df


@st.cache_data(ttl=3600)
def load_historical_data():
    """加载历史比赛数据（用于计算球队近期战绩，缓存1小时）
    【云端无 db 降级】表不存在时返回带列空 df，页面显示"无数据"而非崩溃"""
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM match_feature_final", conn)
        conn.close()
    except Exception:
        return _empty_df_with(_FEATURE_EMPTY_COLS)
    df["match_date"] = pd.to_datetime(df["match_date"])
    return df


def build_pred_feature(df_hist, home_team, away_team, league_code):
    """为单场比赛构建预测特征（52/57维完整特征，赔率用默认值降级填充）。
    【2026-08-07 口径修复：build_feature_by_teams 现接收【原始赔率】(home/draw/away 三个)，
     内部自行去水算概率衍生特征；旧调用只传2个去水概率+误把0.0当 away_odds，会除零崩溃，
     且与训练端 match_feature_final 特征口径失配。现改传三个默认原始赔率。】"""
    # 复用统一的特征构建函数（降级预测用默认赔率，自动去水转概率）
    return build_feature_by_teams(
        df_hist, home_team, away_team,
        DEFAULT_HOME_ODDS, DEFAULT_DRAW_ODDS, DEFAULT_AWAY_ODDS,
        league_code,
        use_value_features=True  # 【2026-08-08 修复：与 predict_module 一致开启身价特征(57维)；否则52维触发 predict_match 维度断言】
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
    # 【2026-08-08 修复】match_feature_final 球队列实为 home_team_std/away_team_std 且【中英混杂】
    #  （约80%中文、20%英文）。旧硬编码 home_team 会 KeyError；须【中英双向匹配】(isin)，
    #   确保无论表内存中文还是英文都能判定"有历史数据"，与 build_feature_by_teams 的双向匹配一致。
    home_col = "home_team_std" if "home_team_std" in df_hist.columns else "home_team"
    away_col = "away_team_std" if "away_team_std" in df_hist.columns else "away_team"
    home_keys = {row["home_team"], home_cn}
    away_keys = {row["away_team"], away_cn}
    has_home = len(df_hist[df_hist[home_col].isin(home_keys)]) > 0
    has_away = len(df_hist[df_hist[away_col].isin(away_keys)]) > 0

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

    # 【云端无 db 降级】无赛程数据时直接提示，不渲染筛选区（避免空 df .dt/ min 报错）
    if df_all.empty:
        st.info("📭 云端暂无赛程数据（本地 `football.db` 含完整赛程，请本地运行查看）")
        return

    # ========== 筛选区 ==========
    col1, col2, col3 = st.columns([2, 3, 1])

    with col1:
        # 快捷切换
        quick_select = st.radio(
            "时间范围",
            ["今日", "本周", "自定义"],
            horizontal=True,
            index=1,  # 默认本周
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

    # 周切换按钮（仅本周模式显示）
    if quick_select == "本周":
        # 用session_state保存周偏移
        if "week_offset" not in st.session_state:
            st.session_state.week_offset = 0

        wc1, wc2, wc3, wc4, wc5 = st.columns([1, 1, 1, 1, 6])
        if wc1.button("← 上一周", key="prev_week"):
            st.session_state.week_offset -= 1
        if wc3.button("下一周 →", key="next_week"):
            st.session_state.week_offset += 1
        if wc4.button("回到本周", key="this_week"):
            st.session_state.week_offset = 0

        # 根据偏移计算日期
        base_monday, _ = get_week_range()
        offset_days = st.session_state.week_offset * 7
        start_date = base_monday + timedelta(days=offset_days)
        end_date = start_date + timedelta(days=6)

        week_label = f"{start_date.strftime('%m/%d')} ~ {end_date.strftime('%m/%d')}"
        if st.session_state.week_offset == 0:
            week_label += " （本周）"
        wc2.markdown(f"<div style='text-align:center;padding:6px 0;font-weight:500'>{week_label}</div>", unsafe_allow_html=True)

    # ========== 联赛快速筛选标签 ==========
    if "calendar_league" not in st.session_state:
        st.session_state.calendar_league = "all"

    league_tabs = st.columns([1, 1, 1, 1, 1, 1, 1, 3])
    # 【2026-08-05 修复：联赛tab改用 B体系 db_code，筛选时同时匹配 db_code 与旧码】
    league_labels = [("all", "全部")] + [(v["db_code"], v["name"]) for v in LEAGUE_REGISTRY.values()] + [("UCL", "欧冠")]

    for i, (code, name) in enumerate(league_labels):
        is_active = st.session_state.calendar_league == code
        btn_type = "primary" if is_active else "secondary"
        if league_tabs[i].button(name, key=f"league_tab_{code}", type=btn_type, use_container_width=True):
            st.session_state.calendar_league = code

    # 根据标签生成筛选列表（每个联赛同时匹配 db_code 与旧码，兼容历史数据）
    if st.session_state.calendar_league == "all":
        selected_leagues = LEAGUE_OPTIONS
    else:
        selected_leagues = _league_match_codes(st.session_state.calendar_league)

    st.divider()

    # ========== 数据筛选 ==========
    mask_date = (df_all["match_date"].dt.date >= start_date) & (df_all["match_date"].dt.date <= end_date)
    mask_league = df_all["league_code"].isin(selected_leagues)
    df_filter = df_all[mask_date & mask_league].copy()

    if len(df_filter) == 0:
        st.info("📭 当前筛选条件下暂无比赛")
        return

    # 按比赛ID去重（防止重复数据）
    df_filter = df_filter.drop_duplicates(subset=["id"]).copy()

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
        with st.spinner("🔮 正在预测中，请稍等..."):
            batch_results = []
            _pred_count = 0
            _skip_count = 0
            track('predict_batch',
                  action_detail=f'{len(df_filter)}场赛程',
                  page_name='预测中心')
            for _, row in df_filter.iterrows():
                if row["status"] == "Played":
                    continue
                can_pred, reason = can_predict_match(row, df_hist)
                if not can_pred:
                    _skip_count += 1
                    continue
                _pred_count += 1

                home_name = row["home_team_cn"] if pd.notna(row["home_team_cn"]) else row["home_team"]
                away_name = row["away_team_cn"] if pd.notna(row["away_team_cn"]) else row["away_team"]
                # 特征构建用英文标准名（历史数据是英文列）
                home_std = row["home_team"]
                away_std = row["away_team"]

                try:
                    feature = build_pred_feature(df_hist, home_std, away_std, row["league_code"])
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

                    # 写入数据库（有昵称才保存）
                    if user_name:
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

            # 结果去重（按日期+主队+客队）
            seen = set()
            unique_results = []
            for r in batch_results:
                key = (r["比赛日期"], r["主队"], r["客队"])
                if key not in seen:
                    seen.add(key)
                    unique_results.append(r)

            st.session_state[batch_pred_key] = unique_results
            # 方向6：批量预测统计反馈（成功/跳过）
            if _pred_count > 0:
                st.success(f"✅ 批量预测完成：成功 **{_pred_count}** 场，跳过 **{_skip_count}** 场（已完赛/无历史数据）")
            else:
                st.warning(f"⚠️ 本批 {len(df_filter)} 场赛事均无法预测（已完赛或历史数据不足）")

    # 批量预测结果展示
    if batch_pred_key in st.session_state and len(st.session_state[batch_pred_key]) > 0:
        with st.expander(f"📊 批量预测结果（共 {len(st.session_state[batch_pred_key])} 场）· 点击标题可折叠收起", expanded=True):
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
            st.caption(f"💡 置信度=该档位历史命中率（WF外样本验证）：≥60%命中率 {high_conf} 场，≥70%命中率 {very_high} 场")
            st.caption("⚠️ 赛程预测缺少赔率和射门数据，为降级预测；AI建议仅供参考，是否出手由您独立判断")

    # ========== 按天卡片式展示 ==========
    WEEKDAY_CN = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    today = datetime.now().date()
    tomorrow = today + timedelta(days=1)

    for date_val, day_df in grouped:
        day_name = date_val.strftime("%m-%d")
        weekday = WEEKDAY_CN[date_val.weekday()]
        day_count = len(day_df)
        is_today = date_val == today
        is_tomorrow = date_val == tomorrow
        is_near = is_today or is_tomorrow
        is_weekend = date_val.weekday() >= 5

        # 日期标题栏
        if is_today:
            header_bg = "#e3f2fd"
            header_border = "3px solid #2196f3"
            today_tag = " 🔴 今天"
        elif is_tomorrow:
            header_bg = "#e8f5e9"
            header_border = "2px solid #66bb6a"
            today_tag = " 🟢 明天"
        elif is_weekend:
            header_bg = "#f5f5f5"
            header_border = "none"
            today_tag = ""
        else:
            header_bg = "#fafafa"
            header_border = "none"
            today_tag = ""

        date_title = f"📅 {day_name} {weekday}（{day_count} 场）{today_tag}"

        # 今明两天直接展开，其他天折叠
        if is_near:
            st.markdown(f"""
            <div style="padding:10px 16px;background:{header_bg};border-radius:8px;
                        margin-bottom:12px;border-left:{header_border}">
                <span style="font-weight:600;font-size:17px">{date_title}</span>
            </div>
            """, unsafe_allow_html=True)

            # 双列卡片布局
            day_list = day_df.sort_values("match_time").to_dict('records')
            col_left, col_right = st.columns(2)

            for i, row in enumerate(day_list):
                col = col_left if i % 2 == 0 else col_right
                with col:
                    render_match_card(row, df_hist, user_name)

            st.markdown("<br>", unsafe_allow_html=True)
        else:
            with st.expander(date_title, expanded=False):
                # 双列卡片布局
                day_list = day_df.sort_values("match_time").to_dict('records')
                col_left, col_right = st.columns(2)

                for i, row in enumerate(day_list):
                    col = col_left if i % 2 == 0 else col_right
                    with col:
                        render_match_card(row, df_hist, user_name)


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
                feature = build_pred_feature(df_hist, row["home_team"], row["away_team"], row["league_code"])
                result = predict_match(feature)
                st.session_state[pred_key] = result

                # 保存预测结果到数据库（有昵称才保存）
                if user_name:
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
                    f"<span style='color:#666;'>（置信度 {pred_result['confidence']:.1%} = 该档位历史命中率）</span>"
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


def render_match_card(row, df_hist, user_name=None):
    """渲染单场比赛卡片（A+方案）"""
    league_info = LEAGUE_DISPLAY.get(row["league_code"], {"name": row["league_code"], "color": "#666"})
    match_id = int(row["id"])
    pred_key = f"schedule_pred_{match_id}"

    home_name = row["home_team_cn"] if pd.notna(row.get("home_team_cn")) and row["home_team_cn"] != row["home_team"] else row["home_team"]
    away_name = row["away_team_cn"] if pd.notna(row.get("away_team_cn")) and row["away_team_cn"] != row["away_team"] else row["away_team"]
    # 特征构建用英文标准名
    home_std = row["home_team"]
    away_std = row["away_team"]

    is_played = row["status"] == "Played" and pd.notna(row.get("result"))

    # 卡片顶部：联赛标签 + 时间
    st.markdown(f"""
    <div style="padding:12px 14px;background:#fff;border:1px solid #e8e8e8;border-radius:10px;
                box-shadow:0 1px 3px rgba(0,0,0,0.04);margin-bottom:10px">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
            <span style="font-size:11px;padding:2px 10px;border-radius:12px;font-weight:500;
                         background:{league_info['color']};color:white">{league_info['name']}</span>
            <span style="font-size:13px;color:#888">⏰ {row['match_time']}</span>
        </div>
        <div style="font-size:15px;font-weight:500;text-align:center;padding:4px 0 8px;">
            {home_name} <span style="color:#bbb;margin:0 10px;font-weight:400">vs</span> {away_name}
        </div>
    """, unsafe_allow_html=True)

    # 比分（已完赛）
    if is_played:
        st.markdown(
            f"<div style='text-align:center;margin-bottom:8px;'>"
            f"<span style='background:#e8f5e9;color:#2e7d32;padding:3px 12px;border-radius:6px;font-size:14px;font-weight:600;'>"
            f"{row['result']}"
            f"</span></div>",
            unsafe_allow_html=True
        )

    # 预测按钮 + 结果
    if not is_played:
        can_pred, reason = can_predict_match(row, df_hist)

        if can_pred:
            if st.button("🔮 预测", key=f"btn_card_pred_{match_id}", type="primary", use_container_width=True):
                feature = build_pred_feature(df_hist, home_std, away_std, row["league_code"])
                result = predict_match(feature)
                st.session_state[pred_key] = result

                # 方向3：保存反馈（None=未保存(无昵称) True=成功 False=失败）
                _saved_flag = None
                if user_name:
                    try:
                        _saved_flag = bool(save_prediction_to_db(
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
                        ))
                    except:
                        _saved_flag = False
                st.session_state[f"card_saved_{match_id}"] = _saved_flag

            # 展示预测结果
            if pred_key in st.session_state:
                pred_result = st.session_state[pred_key]
                result_color = {"主胜": "#2ecc71", "平局": "#f39c12", "客胜": "#e74c3c"}.get(pred_result["predict_result"], "#666")
                # 【2026-08-21 展示口径：模型概率为主角，历史命中率降级为校准辅助】
                #  旧版把「置信度 xx%（历史命中）」高亮在预测方向后，用户会误读成
                #  「主胜本次命中率=xx%」，与模型概率（如 39%）对不上而困惑。
                #  现改为：高亮只显示预测方向 → 模型概率分布独立一行（主角）→
                #  历史命中率明确标注「该档位」小字校准说明（保留但不再喧宾夺主）。
                _ph = pred_result['prob_home_win']
                _pd_ = pred_result['prob_draw']
                _pa = pred_result['prob_away_win']
                st.markdown(
                    f"<div style='margin-top:8px;padding:8px 10px;background:#f8f9fa;border-radius:6px;'>"
                    f"<div style='font-size:14px;font-weight:600;color:{result_color}'>"
                    f"→ 预测 {pred_result['predict_result']}"
                    f"</div>"
                    f"<div style='font-size:12px;color:#444;margin-top:2px'>"
                    f"模型概率：主胜 {_ph:.0%} · 平 {_pd_:.0%} · 客胜 {_pa:.0%}"
                    f"</div>"
                    f"<div style='font-size:11px;color:#999;margin-top:2px'>"
                    f"该档位历史命中率 {pred_result['confidence']:.1%} · 缺赔率降级预测，仅供参考"
                    f"</div></div>",
                    unsafe_allow_html=True
                )
                # 方向3：保存反馈
                _saved_flag = st.session_state.get(f"card_saved_{match_id}")
                if _saved_flag is True:
                    st.caption("✅ 已保存到预测历史")
                elif _saved_flag is False:
                    st.caption("⚠️ 预测已展示，但保存失败")

                # 详情展开：AI出手建议
                show_detail = st.checkbox("📋 查看详情", key=f"detail_{match_id}")
                if show_detail:
                    from ai_intent_module import calc_ai_bet_intent
                    intent = calc_ai_bet_intent(pred_result["confidence"], pred_result["predict_result"])
                    st.markdown(f"**共识等级：{intent['consensus_label']}**")
                    for ai_name, ai_data in intent["intents"].items():
                        status = "✅ 建议关注" if ai_data["will_bet"] else "❌ 观望"
                        reason = ai_data.get("reason", "")
                        st.caption(f"{ai_name}：{status}{f'（{reason}）' if reason else ''}")
                    st.caption(f"模型融合：{pred_result.get('model_detail', {}).get('fusion_weight', 'LGB 55% + 泊松 30% + 平局专项 15%')}")
                    st.caption("⚠️ 置信度=该档位历史命中率（WF外样本验证）；AI 建议仅供数据参考，是否出手由您独立判断，AI 不替您做决定")
        else:
            st.button("🔮 预测", key=f"btn_card_pred_{match_id}", disabled=True, help=reason, use_container_width=True)

    # 卡片底部闭合
    st.markdown("</div>", unsafe_allow_html=True)
