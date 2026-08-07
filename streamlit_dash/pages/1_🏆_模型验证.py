"""
AI策略验证面板 - 三个AI策略对比（模型验证工具，非投注建议）
"""
import sys
import os
sys.stdout.reconfigure(encoding='utf-8')

# 路径配置
SCRIPT_PATH = os.path.abspath(__file__)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_PATH)))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(SCRIPT_PATH)))

import streamlit as st
import pandas as pd
import sqlite3
import numpy as np
import json

# 页面配置
st.set_page_config(page_title="AI策略验证面板", page_icon="🏆", layout="wide")

# 公共样式
from common.style import apply_global_style
apply_global_style()

from common.data_loader import DB_PATH

# 球队中文映射
from team_mapping_v2 import LEAGUE_TEAM_MAP
std_2_cn = {}
for cfg_code, team_map in LEAGUE_TEAM_MAP.items():
    for eng_std, (full_eng, cn_name) in team_map.items():
        std_2_cn[eng_std] = cn_name

# AI配置：角色卡风格 + 人设
AI_PROFILES = {
    "保守AI": {
        "display_name": "磐石#D01",
        "icon": "🪨",
        "color": "#27ae60",
        "style_tag": "防守型",
        "style_desc": "不急，等机会",
        "skill_name": "【稳如泰山】",
        "skill_desc": "高命中率档（研究观察）：历史样本 113 场 +13.16%，聚类 CI 含 0 → 统计不显著，仅研究观察",
        "personality": "研究·高命中档 · 保守偏好",
        "params": "命中率≥70% · 研究档 · 20%安全边际",
        "suitable": "研究观察：偏好高命中率场景",
    },
    "中立AI": {
        "display_name": "天秤#B01",
        "icon": "⚖️",
        "color": "#3498db",
        "style_tag": "均衡型",
        "style_desc": "两边都要，但要算清楚",
        "skill_name": "【均衡之道】",
        "skill_desc": "均衡档（研究观察）：在收益与风险间取平衡，样本内 ROI +9.4% 为历史回测，研究参考",
        "personality": "研究·均衡档 · 覆盖适中",
        "params": "命中率≥60% · 研究档 · 10%安全边际",
        "suitable": "大多数场景的默认选择，攻守兼备",
    },
    "激进AI": {
        "display_name": "猎鹰#A01",
        "icon": "🦅",
        "color": "#e74c3c",
        "style_tag": "攻击型",
        "style_desc": "高赔冷门，精准打击",
        "skill_name": "【疾风突袭】",
        "skill_desc": "冷门猎手，高赔率+高置信度；外样本 2.2/0.55 = 357场 样本内 ROI +15.51%，但真实可成交赔率口径（Pinnacle 终盘）未过 Bonferroni，待 2026-2027 独立期复现（研究档）",
        "personality": "激进派猎手 · 波动较大 · 研究观察",
        "params": "赔率≥2.2 置信≥55% · 研究档 · 每季约14场",
        "suitable": "追求高收益、能承受波动的场景",
    },
    "串关AI": {
        "display_name": "串关参考",
        "icon": "☯️",
        "color": "#9b59b6",
        "style_tag": "风险参考",
        "style_desc": "串关稀释命中率，仅娱乐参考",
        "skill_name": "【串关风险提示】",
        "skill_desc": "2串1命中率≈单场命中率²（单场50%→2串1仅25%），每加一场再砍一半；本系统不将串关作为推荐策略（C-012 存疑）",
        "personality": "参考说明 · 非推荐策略 · 娱乐可用",
        "params": "最多2串1 · 极小注 · 娱乐性质",
        "suitable": "仅娱乐参考，不建议作为主策略",
        "placeholder": True,
    },
}


def get_db():
    return sqlite3.connect(DB_PATH)


def get_all_seasons():
    conn = get_db()
    df = pd.read_sql("SELECT DISTINCT season_year FROM ai_season_summary ORDER BY season_year DESC", conn)
    conn.close()
    return df['season_year'].tolist()


def get_season_summary(season_year):
    conn = get_db()
    df = pd.read_sql(
        "SELECT * FROM ai_season_summary WHERE season_year = ? ORDER BY final_score DESC",
        conn, params=[season_year]
    )
    conn.close()
    return df


def get_betting_log(ai_name=None, season_year=None, limit=100):
    conn = get_db()
    params = []
    where = []
    if ai_name:
        where.append("ai_name = ?")
        params.append(ai_name)
    if season_year:
        where.append("season_year = ?")
        params.append(season_year)
    where_sql = "WHERE " + " AND ".join(where) if where else ""
    params.append(limit)
    query = f"SELECT * FROM ai_betting_log {where_sql} ORDER BY match_date DESC LIMIT ?"
    df = pd.read_sql(query, conn, params=params)
    conn.close()
    return df


def get_score_curve(ai_name, season_year):
    conn = get_db()
    df = pd.read_sql(
        "SELECT match_date, score_after FROM ai_betting_log WHERE ai_name = ? AND season_year = ? ORDER BY match_date ASC",
        conn, params=[ai_name, season_year]
    )
    conn.close()
    return df


def get_confidence_distribution(ai_name, season_year):
    """置信度分布：高/中/低三档"""
    conn = get_db()
    df = pd.read_sql("""
        SELECT 
            SUM(CASE WHEN confidence >= 0.75 THEN 1 ELSE 0 END) as high,
            SUM(CASE WHEN confidence >= 0.6 AND confidence < 0.75 THEN 1 ELSE 0 END) as mid,
            SUM(CASE WHEN confidence < 0.6 THEN 1 ELSE 0 END) as low,
            COUNT(*) as total
        FROM ai_betting_log 
        WHERE ai_name = ? AND season_year = ?
    """, conn, params=[ai_name, season_year])
    conn.close()
    if len(df) == 0 or df.iloc[0]['total'] == 0:
        return {'high': 0, 'mid': 0, 'low': 0, 'total': 0}
    row = df.iloc[0]
    return {
        'high_pct': row['high'] / row['total'],
        'mid_pct': row['mid'] / row['total'],
        'low_pct': row['low'] / row['total'],
        'total': row['total']
    }


def format_growth_multiplier(roi):
    """将收益率格式化为增长倍数（处理天文数字）"""
    if roi is None or np.isnan(roi):
        return "--"
    multiplier = roi + 1  # 收益率转倍数
    if multiplier < 10000:
        return f"{multiplier:.1f}x"
    elif multiplier < 1e8:
        return f"{multiplier/1e4:.1f}万x"
    else:
        # 超大数字用科学计数法
        exp = int(np.floor(np.log10(multiplier)))
        base = multiplier / (10 ** exp)
        return f"{base:.2f}×10^{exp}x"


def get_consensus_analysis(season_year):
    """AI共识分析：按实际出手AI数量精确分级，计算胜率差异"""
    conn = get_db()
    df = pd.read_sql("""
        SELECT match_date, home_team, away_team, ai_name, confidence, win, odds
        FROM ai_betting_log 
        WHERE season_year = ?
    """, conn, params=[season_year])
    conn.close()
    
    if len(df) == 0:
        return None
    
    # 按比赛分组
    pivot = df.pivot_table(
        index=['match_date', 'home_team', 'away_team'],
        columns='ai_name',
        values=['win', 'confidence', 'odds'],
        aggfunc='first'
    ).reset_index()
    
    ai_cols = ['激进AI', '中立AI', '保守AI']
    pivot['bet_count'] = pivot[('win', '激进AI')].notna().astype(int) + \
                         pivot[('win', '中立AI')].notna().astype(int) + \
                         pivot[('win', '保守AI')].notna().astype(int)
    
    stats = []
    for label, count_filter, desc in [
        ("🔥 仅激进出手", 1, "低共识，只有激进AI出手"),
        ("🤝 三AI共识", 3, "高共识，三个AI全部看好"),
    ]:
        subset = pivot[pivot['bet_count'] == count_filter]
        if len(subset) > 0:
            win_rate = subset[('win', '激进AI')].mean()
            avg_odds = subset[('odds', '激进AI')].mean()
            avg_conf = subset[('confidence', '激进AI')].mean()
            stats.append({
                '共识等级': label,
                '说明': desc,
                '场次': len(subset),
                '胜率': f"{win_rate:.1%}",
                '平均置信度': f"{avg_odds:.2f}",
                '平均置信度': f"{avg_conf:.1%}",
            })
    
    return pd.DataFrame(stats) if stats else None


# ========== 页面主体 ==========
st.markdown("# 🏆 模型验证")
st.markdown("Walk Forward 金标准验证 · 无未来函数 · 真实预测能力评估")
st.caption("💡 **说人话**：系统用过去 20 多年的比赛「模拟真实预测」——每场比赛只使用它**之前**已经知道的信息，一场一场往后推着验证，确保结论不是「事后诸葛亮」。下方所有准确率均为这种严格验证下的真实水平（50,752 场外样本，整体 51.75%）。")

# ===== 📖 一句话看懂本系统策略（普通人版，折叠） =====
with st.expander("📖 一句话看懂本系统策略（给普通人）", expanded=False):
    st.markdown(
        "> **用 AI 在五大联赛里挑「模型最有把握 + 赔率给得划算」的比赛，每次只下总资金很小的比例，"
        "靠长期纪律累积优势——不赌单场，赌「做过验证、长期占优」的概率。**"
    )
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.info("**🦅 猎鹰（研究·高赔冷门档）**\n\n外样本 357 次出手（去重后 2.2/0.55）样本内 ROI +15.51%；真实可成交赔率口径样本偏薄、未过 Bonferroni → 待 2026-2027 独立期复现，仅研究参考。")
    with col_s2:
        st.info("**🪨 磐石Pro（研究·高命中档）**\n\n研究样本 113 场 +13.16%，聚类 CI 含 0 → 统计显著不成立，仅研究观察。")
    with col_s3:
        st.warning("**⚖️ 价值（已移除）**\n\n曾有研究 alpha，但无后见之明验证不成立（C-015），已从组合移除（C-016），仅保留研究记录。")
    st.caption("**风险提示**：以上均为历史研究数据，不代表未来收益；任何策略在独立期复现前不构成任何建议。")

# 验证体系说明（折叠式，节省空间）
with st.expander("📊 验证体系说明（点击展开）", expanded=False):
    col_v1, col_v2, col_v3 = st.columns(3)
    
    with col_v1:
        st.info("""
        ⭐ **Level 1：全量回测**
        - 全量数据训练，全量数据测试
        - 用途：快速筛选、方向探索
        - **严重偏乐观，仅供参考**
        """)
    
    with col_v2:
        st.info("""
        ⭐⭐⭐ **Level 2：赛季重置**
        - 全量数据训练，按赛季重置资金
        - 用途：策略稳定性初步评估
        - 仍有未来函数
        """)
    
    with col_v3:
        st.success("""
        ⭐⭐⭐⭐⭐ **Level 3：WF赛季重置**
        - 滚动训练，无未来函数
        - 用途：最终结论、金标准
        - **所有结论以此为准**
        """)

st.divider()

# 赛季选择（如果有数据的话）
seasons = get_all_seasons()
has_season_data = len(seasons) > 0

if has_season_data:
    col_sel, _ = st.columns([1, 3])
    with col_sel:
        selected_season = st.selectbox("选择赛季", seasons, index=0)
else:
    st.info("ℹ️ AI策略赛季数据待生成，先展示策略动物园和其他验证结果")

# ===== 15赛季历史平均战绩 =====
if has_season_data:
    with st.expander("📜 15赛季历史平均战绩（新赛季参考基准）", expanded=False):
        conn = get_db()
        hist_df = pd.read_sql("""
            SELECT 
                ai_name,
                AVG(win_rate) as avg_win_rate,
                AVG(max_drawdown) as avg_max_dd,
                AVG(total_bets) as avg_bets,
                SUM(CASE WHEN is_bankrupt = 1 THEN 1 ELSE 0 END) as bankrupt_seasons,
                COUNT(*) as total_seasons
            FROM ai_season_summary
            GROUP BY ai_name
            ORDER BY avg_win_rate DESC
        """, conn)
        conn.close()
        
        if len(hist_df) > 0:
            display_hist = hist_df.copy()
            # 映射AI展示名
            display_hist['AI'] = display_hist['ai_name'].apply(
                lambda x: f"{AI_PROFILES.get(x, {}).get('icon', '🤖')} {AI_PROFILES.get(x, {}).get('display_name', x)}"
            )
            display_hist['平均胜率'] = display_hist['avg_win_rate'].apply(lambda x: f"{x:.1%}")
            display_hist['平均回撤'] = display_hist['avg_max_dd'].apply(lambda x: f"{x:.1%}")
            display_hist['场均出手'] = display_hist['avg_bets'].round(0).astype(int)
            display_hist['深撤赛季'] = display_hist['bankrupt_seasons'].astype(str) + '/' + display_hist['total_seasons'].astype(str)
            st.dataframe(display_hist[['AI', '平均胜率', '平均回撤', '场均出手', '深撤赛季']], 
                         use_container_width=True, hide_index=True)
        st.caption("💡 新赛季开局参考：基于15个完整赛季的Walk-Forward OOS回测数据")

# ========== 四个AI核心指标卡片（角色卡风格） ==========
if has_season_data:
    st.markdown("## 🎴 AI 策略角色卡")
    st.caption("🪨⚖️🦅 三张 AI 卡 = **同一模型按风险偏好分三档**（求稳→均衡→进取），不是三种独立策略；"
               "均为研究性输出：猎鹰（研究·高赔冷门档）、磐石Pro（研究·高命中档）；**价值策略已从组合移除（C-016），仅保留研究记录**。"
               "串关 / 高级模式（Pro 专精）等**不作为推荐策略**，风险说明见下方「📌 决策参考」。")

    summary_df = get_season_summary(selected_season)
    cols = st.columns(3)

    for idx, (ai_name, profile) in enumerate(AI_PROFILES.items()):
        if profile.get('placeholder'):
            continue  # 串关已撤销 → 风险说明见下方「📌 决策参考」
        ai_data = summary_df[summary_df['ai_name'] == ai_name]
        with cols[idx]:
            if len(ai_data) > 0:
                row = ai_data.iloc[0]
                win_rate = row['win_rate']
                bets = row['total_bets']
                max_dd = row['max_drawdown']
                roi = row['roi']
                bankrupt = row['is_bankrupt']
                
                growth_str = format_growth_multiplier(roi)
                status = "📉 深度回撤" if bankrupt else "✅ 存活"
                
                # 置信度分布
                conf_dist = get_confidence_distribution(ai_name, selected_season)
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {profile['color']}20, {profile['color']}08); 
                            border: 2px solid {profile['color']}40; border-radius: 12px; padding: 16px; margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-size: 22px; font-weight: bold;">{profile['icon']} {profile['display_name']}</span>
                        <span style="font-size: 11px; padding: 3px 8px; background: {'#27ae60' if not bankrupt else '#e74c3c'}20; 
                              color: {'#27ae60' if not bankrupt else '#e74c3c'}; border-radius: 10px; font-weight: bold;">
                            {'✅ 存活' if not bankrupt else '📉 回撤'}
                        </span>
                    </div>
                    <div style="font-size: 12px; color: {profile['color']}; font-weight: bold; margin-bottom: 10px;">
                        {profile['style_tag']} · {profile['personality']}
                    </div>
                    <div style="font-size: 13px; color: #666; margin-bottom: 12px; font-style: italic;">
                        "{profile['style_desc']}"
                    </div>
                    <div style="background: {profile['color']}10; border-radius: 6px; padding: 10px; margin-bottom: 12px;">
                        <div style="font-size: 12px; color: {profile['color']}; font-weight: bold; margin-bottom: 6px;">
                            ⚡ {profile['skill_name']}
                        </div>
                        <div style="font-size: 12px; color: #555; line-height: 1.5;">
                            {profile['skill_desc']}
                        </div>
                    </div>
                    <div style="display: flex; justify-content: space-between; text-align: center; font-size: 12px;">
                        <div>
                            <div style="color: #888; font-size: 11px;">胜率</div>
                            <div style="font-weight: bold; color: #333; font-size: 14px;">{win_rate:.1%}</div>
                        </div>
                        <div>
                            <div style="color: #888; font-size: 11px;">回撤</div>
                            <div style="font-weight: bold; color: #e74c3c; font-size: 14px;">{max_dd:.1%}</div>
                        </div>
                        <div>
                            <div style="color: #888; font-size: 11px;">出手</div>
                            <div style="font-weight: bold; color: #333; font-size: 14px;">{bets}场</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.caption(f"📊 参数：{profile['params']}")
                st.caption(f"🎯 置信分布：高{conf_dist.get('high_pct', 0):.0%} · 中{conf_dist.get('mid_pct', 0):.0%} · 低{conf_dist.get('low_pct', 0):.0%}")
            else:
                # 占位符AI（暂无数据）
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {profile['color']}15, {profile['color']}05); 
                            border: 2px dashed {profile['color']}40; border-radius: 12px; padding: 16px; margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-size: 22px; font-weight: bold;">{profile['icon']} {profile['display_name']}</span>
                        <span style="font-size: 11px; padding: 3px 8px; background: #99999920; 
                              color: #999; border-radius: 10px; font-weight: bold;">
                            🚧 准备中
                        </span>
                    </div>
                    <div style="font-size: 12px; color: {profile['color']}; font-weight: bold; margin-bottom: 10px;">
                        {profile['style_tag']} · {profile['personality']}
                    </div>
                    <div style="font-size: 13px; color: #666; margin-bottom: 12px; font-style: italic;">
                        "{profile['style_desc']}"
                    </div>
                    <div style="background: {profile['color']}10; border-radius: 6px; padding: 10px; margin-bottom: 12px;">
                        <div style="font-size: 12px; color: {profile['color']}; font-weight: bold; margin-bottom: 6px;">
                            ⚡ {profile['skill_name']}
                        </div>
                        <div style="font-size: 12px; color: #555; line-height: 1.5;">
                            {profile['skill_desc']}
                        </div>
                    </div>
                    <div style="display: flex; justify-content: space-between; text-align: center; font-size: 12px;">
                        <div>
                            <div style="color: #888; font-size: 11px;">胜率</div>
                            <div style="font-weight: bold; color: #999; font-size: 14px;">--</div>
                        </div>
                        <div>
                            <div style="color: #888; font-size: 11px;">回撤</div>
                            <div style="font-weight: bold; color: #999; font-size: 14px;">--</div>
                        </div>
                        <div>
                            <div style="color: #888; font-size: 11px;">出手</div>
                            <div style="font-weight: bold; color: #999; font-size: 14px;">--</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if ai_name == '串关AI':
                    st.warning("☯️ **串关风险参考**：2串1 命中率 ≈ 单场命中率²（单场 50% → 2串1 仅 25%），每加一场再砍一半。**本系统不将串关作为推荐策略**（C-012 存疑）；如娱乐使用，请用极小注、最多 2 串 1。")
                else:
                    st.caption(f"📊 参数：{profile['params']}")
                    st.caption("🚧 数据准备中，敬请期待")
else:
    # 没有赛季数据时，展示AI角色卡占位符
    st.markdown("## 🎴 AI 策略角色卡")
    st.caption("🪨⚖️🦅 三张 AI 卡 = **同一模型按风险偏好分三档**（求稳→均衡→进取）；"
               "均为研究性输出；**价值策略已从组合移除（C-016），仅保留研究记录**。串关 / 高级模式（Pro 专精）等**不作为推荐策略**，风险说明见下方「📌 决策参考」。")
    cols = st.columns(3)
    for idx, (ai_name, profile) in enumerate(AI_PROFILES.items()):
        if profile.get('placeholder'):
            continue  # 串关已撤销 → 风险说明见下方「📌 决策参考」
        with cols[idx]:
            # === 磐石AI：支持高级模式切换 ===
            if ai_name == '保守AI':
                # 高级模式状态
                if 'panshi_advanced_mode' not in st.session_state:
                    st.session_state['panshi_advanced_mode'] = False
                
                advanced_mode = st.session_state['panshi_advanced_mode']
                
                if advanced_mode:
                    # 👑 Pro版角色卡（超级组合策略）
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #f39c1225, #f39c1208); 
                                border: 2px solid #f39c1260; border-radius: 12px; padding: 16px; margin-bottom: 10px;
                                box-shadow: 0 4px 12px #f39c1220;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span style="font-size: 22px; font-weight: bold;">👑 磐石#D01 Pro</span>
                            <span style="font-size: 11px; padding: 3px 8px; background: #f39c1230; 
                                  color: #f39c12; border-radius: 10px; font-weight: bold;">
                                👑 Pro
                            </span>
                        </div>
                        <div style="font-size: 12px; color: #f39c12; font-weight: bold; margin-bottom: 10px;">
                            磐石Pro（超级组合）· ⚠️ 金标准存疑（C-012，样本小）
                        </div>
                        <div style="font-size: 13px; color: #666; margin-bottom: 12px; font-style: italic;">
                            "精选联赛，只做最好的机会"
                        </div>
                        <div style="background: #f39c1215; border-radius: 6px; padding: 10px; margin-bottom: 12px;">
                            <div style="font-size: 12px; color: #f39c12; font-weight: bold; margin-bottom: 6px;">
                                ⚡ 【王者之道】
                            </div>
                            <div style="font-size: 12px; color: #555; line-height: 1.5;">
                                德甲+意甲精选联赛 + 高置信度平局机会，收益最大化
                            </div>
                        </div>
                        <div style="display: flex; justify-content: space-between; text-align: center; font-size: 12px;">
                            <div>
                                <div style="color: #888; font-size: 11px;">胜率</div>
                                <div style="font-weight: bold; color: #27ae60; font-size: 14px;">50.2%</div>
                            </div>
                            <div>
                                <div style="color: #888; font-size: 11px;">回撤</div>
                                <div style="font-weight: bold; color: #e74c3c; font-size: 14px;">25%</div>
                            </div>
                            <div>
                                <div style="color: #888; font-size: 11px;">出手</div>
                                <div style="font-weight: bold; color: #333; font-size: 14px;">13场</div>
                            </div>
                        </div>
                        <div style="text-align: center; margin-top: 10px; padding-top: 8px; border-top: 1px solid #eee;">
                            <span style="font-size: 13px; font-weight: bold; color: #f39c12;">
                                ⚠️ 金标准：+11.39%存疑（盈利季52%）
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.caption("📊 参数：联赛筛选(德甲+意甲) · 主胜/客胜≥55% · 平局≥50% · 固定小注（凯利已废弃，见D-004）")
                    
                    # 高级模式开关
                    st.toggle("👑 高级模式", value=advanced_mode, key='panshi_advanced_mode',
                              help="⚠️ 高级模式（Pro 专精）样本小、金标准存疑（C-012），不作为产品化依据，普通用户无需开启。开启后启用超级组合策略：联赛筛选+平局联动+最优参数")
                else:
                    # 🪨 普通版角色卡（A方案基准）
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, {profile['color']}15, {profile['color']}05); 
                                border: 2px solid {profile['color']}40; border-radius: 12px; padding: 16px; margin-bottom: 10px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span style="font-size: 22px; font-weight: bold;">{profile['icon']} {profile['display_name']}</span>
                            <span style="font-size: 11px; padding: 3px 8px; background: {profile['color']}20; 
                                  color: {profile['color']}; border-radius: 10px; font-weight: bold;">
                                🔬 研究档
                            </span>
                        </div>
                        <div style="font-size: 12px; color: {profile['color']}; font-weight: bold; margin-bottom: 10px;">
                            {profile['style_tag']} · {profile['personality']}
                        </div>
                        <div style="font-size: 13px; color: #666; margin-bottom: 12px; font-style: italic;">
                            "{profile['style_desc']}"
                        </div>
                        <div style="background: {profile['color']}10; border-radius: 6px; padding: 10px; margin-bottom: 12px;">
                            <div style="font-size: 12px; color: {profile['color']}; font-weight: bold; margin-bottom: 6px;">
                                ⚡ {profile['skill_name']}
                            </div>
                            <div style="font-size: 12px; color: #555; line-height: 1.5;">
                                {profile['skill_desc']}
                            </div>
                        </div>
                        <div style="display: flex; justify-content: space-between; text-align: center; font-size: 12px;">
                            <div>
                                <div style="color: #888; font-size: 11px;">胜率</div>
                                <div style="font-weight: bold; color: #27ae60; font-size: 14px;">~70%</div>
                            </div>
                            <div>
                                <div style="color: #888; font-size: 11px;">回撤</div>
                                <div style="font-weight: bold; color: #e74c3c; font-size: 14px;">~15%</div>
                            </div>
                            <div>
                                <div style="color: #888; font-size: 11px;">出手</div>
                                <div style="font-weight: bold; color: #333; font-size: 14px;">~13场</div>
                            </div>
                        </div>
                        <div style="text-align: center; margin-top: 10px; padding-top: 8px; border-top: 1px solid #eee;">
                            <span style="font-size: 13px; font-weight: bold; color: #27ae60;">
                                样本内ROI +15.8%（研究）
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.caption(f"📊 参数：{profile['params']}")
                    
                    # 高级模式开关
                    st.toggle("👑 高级模式", value=advanced_mode, key='panshi_advanced_mode',
                              help="⚠️ 高级模式（Pro 专精）样本小、金标准存疑（C-012），不作为产品化依据，普通用户无需开启。开启后启用超级组合策略：联赛筛选+平局联动+最优参数")
            
            elif ai_name == '中立AI':
                # ⚖️ 天秤#B01（均衡型）
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {profile['color']}15, {profile['color']}05); 
                            border: 2px solid {profile['color']}40; border-radius: 12px; padding: 16px; margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-size: 22px; font-weight: bold;">{profile['icon']} {profile['display_name']}</span>
                        <span style="font-size: 11px; padding: 3px 8px; background: #27ae6020; 
                        color: #27ae60; border-radius: 10px; font-weight: bold;">
                        🔬 研究档
                        </span>
                        </div>
                        <div style="font-size: 12px; color: {profile['color']}; font-weight: bold; margin-bottom: 10px;">
                        {profile['style_tag']} · {profile['personality']}
                        </div>
                        <div style="font-size: 13px; color: #666; margin-bottom: 12px; font-style: italic;">
                        "{profile['style_desc']}"
                        </div>
                        <div style="background: {profile['color']}10; border-radius: 6px; padding: 10px; margin-bottom: 12px;">
                        <div style="font-size: 12px; color: {profile['color']}; font-weight: bold; margin-bottom: 6px;">
                        ⚡ {profile['skill_name']}
                        </div>
                        <div style="font-size: 12px; color: #555; line-height: 1.5;">
                        {profile['skill_desc']}
                        </div>
                        </div>
                        <div style="display: flex; justify-content: space-between; text-align: center; font-size: 12px;">
                        <div>
                        <div style="color: #888; font-size: 11px;">胜率</div>
                        <div style="font-weight: bold; color: #333; font-size: 14px;">~60%</div>
                        </div>
                        <div>
                        <div style="color: #888; font-size: 11px;">回撤</div>
                        <div style="font-weight: bold; color: #e74c3c; font-size: 14px;">~20%</div>
                        </div>
                        <div>
                        <div style="color: #888; font-size: 11px;">出手</div>
                        <div style="font-weight: bold; color: #333; font-size: 14px;">~95场</div>
                        </div>
                        </div>
                        <div style="text-align: center; margin-top: 10px; padding-top: 8px; border-top: 1px solid #eee;">
                        <span style="font-size: 13px; font-weight: bold; color: #27ae60;">
                        样本内ROI +9.4%（研究）
                        </span>
                        </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.caption(f"📊 参数：{profile['params']}")
                st.caption("💡 均衡档研究观察，覆盖适中；历史回测数字不代表未来收益")
            
            elif ai_name == '激进AI':
                # 🦅 猎鹰#A01（攻击型）— 支持德甲专精/英超专精双Pro模式（互斥）
                # 初始化session_state（全局开关，其他页面可读取）
                if 'falcon_bundesliga_mode' not in st.session_state:
                    st.session_state['falcon_bundesliga_mode'] = False
                if 'falcon_epl_mode' not in st.session_state:
                    st.session_state['falcon_epl_mode'] = False
                
                bundesliga_mode = st.session_state['falcon_bundesliga_mode']
                epl_mode = st.session_state['falcon_epl_mode']
                
                # 互斥逻辑：同时开启时优先保留德甲专精（表现更好），自动关闭英超专精
                if bundesliga_mode and epl_mode:
                    st.session_state['falcon_epl_mode'] = False
                    epl_mode = False
                
                if bundesliga_mode:
                    # 🇩🇪 德甲专精版（Pro）
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #f39c1225, #f39c1208); 
                                border: 2px solid #f39c1260; border-radius: 12px; padding: 16px; margin-bottom: 10px;
                                box-shadow: 0 4px 12px #f39c1220;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span style="font-size: 22px; font-weight: bold;">👑 猎鹰#A01 Pro</span>
                            <span style="font-size: 11px; padding: 3px 8px; background: #f39c1230; 
                                  color: #f39c12; border-radius: 10px; font-weight: bold;">
                                德甲专精
                            </span>
                        </div>
                        <div style="font-size: 12px; color: #f39c12; font-weight: bold; margin-bottom: 10px;">
                            冷门猎手 · 德甲专精 · ⚠️ 实验性（样本小，金标准存疑）
                        </div>
                        <div style="font-size: 13px; color: #666; margin-bottom: 12px; font-style: italic;">
                            "专门找德甲高赔率但模型很有把握的比赛"
                        </div>
                        <div style="background: #f39c1215; border-radius: 6px; padding: 10px; margin-bottom: 12px;">
                            <div style="font-size: 12px; color: #f39c12; font-weight: bold; margin-bottom: 6px;">
                                ⚡ 【冷门猎手】
                            </div>
                            <div style="font-size: 12px; color: #555; line-height: 1.5;">
                                高赔率+高置信度，市场低估的冷门机会，信息不对称带来超额收益
                            </div>
                        </div>
                        <div style="display: flex; justify-content: space-between; text-align: center; font-size: 12px;">
                            <div>
                                <div style="color: #888; font-size: 11px;">胜率</div>
                                <div style="font-weight: bold; color: #27ae60; font-size: 14px;">53.8%</div>
                            </div>
                            <div>
                                <div style="color: #888; font-size: 11px;">平均赔率</div>
                                <div style="font-weight: bold; color: #333; font-size: 14px;">2.95</div>
                            </div>
                            <div>
                                <div style="color: #888; font-size: 11px;">赛季均</div>
                                <div style="font-weight: bold; color: #333; font-size: 14px;">4.3场</div>
                            </div>
                        </div>
                        <div style="text-align: center; margin-top: 10px; padding-top: 8px; border-top: 1px solid #eee;">
                            <span style="font-size: 13px; font-weight: bold; color: #f39c12;">
                                ⚠️ 金标准打折：+6.85%（26场·存疑）
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.caption("📊 参数：仅德甲 · 赔率≥2.5 · 置信度≥50% · 无最低出手限制")
                    
                    # 两个Pro版开关（互斥）
                    col1, col2 = st.columns(2)
                    with col1:
                        st.toggle("德甲专精", value=bundesliga_mode, key='falcon_bundesliga_mode',
                                  help="⚠️ 实验性 Beta（金标准存疑）！无安全边际、出手太多质量差、亏损严重。开启后仅德甲，赔率≥2.5，置信≥50%。")
                    with col2:
                        st.toggle("英超专精", value=epl_mode, key='falcon_epl_mode',
                                  help="⚠️ 实验性 Beta（金标准存疑）！无安全边际、出手太多质量差、亏损严重。开启后仅英超，赔率≥3.0，置信≥50%。")
                    
                    st.caption("💡 盈利赛季占比：83.3% · 6个赛季中5个盈利 ⚠️ 此为旧口径，已由统一金标准打折至 +6.85%（26场·存疑），详见 10_金标准统一验证 报告")
                
                elif epl_mode:
                    # 🏴 英超专精版（Pro）
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #3498db25, #3498db08); 
                                border: 2px solid #3498db60; border-radius: 12px; padding: 16px; margin-bottom: 10px;
                                box-shadow: 0 4px 12px #3498db20;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span style="font-size: 22px; font-weight: bold;">👑 猎鹰#A01 Pro</span>
                            <span style="font-size: 11px; padding: 3px 8px; background: #3498db30; 
                                  color: #3498db; border-radius: 10px; font-weight: bold;">
                                英超专精
                            </span>
                        </div>
                        <div style="font-size: 12px; color: #3498db; font-weight: bold; margin-bottom: 10px;">
                            冷门猎手 · 英超专精 · 稳健备选
                        </div>
                        <div style="font-size: 13px; color: #666; margin-bottom: 12px; font-style: italic;">
                            "英超冷门更多机会也更多，适合作为德甲专精的补充"
                        </div>
                        <div style="background: #3498db15; border-radius: 6px; padding: 10px; margin-bottom: 12px;">
                            <div style="font-size: 12px; color: #3498db; font-weight: bold; margin-bottom: 6px;">
                                ⚡ 【冷门猎手】
                            </div>
                            <div style="font-size: 12px; color: #555; line-height: 1.5;">
                                高赔率+高置信度，专门捕捉英超被市场低估的冷门比赛
                            </div>
                        </div>
                        <div style="display: flex; justify-content: space-between; text-align: center; font-size: 12px;">
                            <div>
                                <div style="color: #888; font-size: 11px;">胜率</div>
                                <div style="font-weight: bold; color: #27ae60; font-size: 14px;">36.7%</div>
                            </div>
                            <div>
                                <div style="color: #888; font-size: 11px;">平均赔率</div>
                                <div style="font-weight: bold; color: #333; font-size: 14px;">3.50</div>
                            </div>
                            <div>
                                <div style="color: #888; font-size: 11px;">赛季均</div>
                                <div style="font-weight: bold; color: #333; font-size: 14px;">2.7场</div>
                            </div>
                        </div>
                        <div style="text-align: center; margin-top: 10px; padding-top: 8px; border-top: 1px solid #eee;">
                            <span style="font-size: 13px; font-weight: bold; color: #3498db;">
                                ⚠️ 金标准打折：个位数%（样本极少·存疑）
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.caption("📊 参数：仅英超 · 赔率≥3.0 · 置信度≥50% · 无最低出手限制")
                    
                    # 两个Pro版开关（互斥）
                    col1, col2 = st.columns(2)
                    with col1:
                        st.toggle("德甲专精", value=bundesliga_mode, key='falcon_bundesliga_mode',
                                  help="⚠️ 实验性 Beta（金标准存疑）！无安全边际、出手太多质量差、亏损严重。开启后仅德甲，赔率≥2.5，置信≥50%。")
                    with col2:
                        st.toggle("英超专精", value=epl_mode, key='falcon_epl_mode',
                                  help="⚠️ 实验性 Beta（金标准存疑）！无安全边际、出手太多质量差、亏损严重。开启后仅英超，赔率≥3.0，置信≥50%。")
                    
                    st.caption("💡 盈利赛季占比：54.5% · 11个赛季中6个盈利 ⚠️ 此为旧口径，已由统一金标准打折（样本极少·存疑），详见 10_金标准统一验证 报告")
                
                else:
                    # 🦅 普通版猎鹰#A01（攻击型）
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, {profile['color']}15, {profile['color']}05); 
                                border: 2px solid {profile['color']}40; border-radius: 12px; padding: 16px; margin-bottom: 10px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span style="font-size: 22px; font-weight: bold;">{profile['icon']} {profile['display_name']}</span>
                            <span style="font-size: 11px; padding: 3px 8px; background: #27ae6020; 
                            color: #27ae60; border-radius: 10px; font-weight: bold;">
                            🔬 研究档
                            </span>
                            </div>
                            <div style="font-size: 12px; color: {profile['color']}; font-weight: bold; margin-bottom: 10px;">
                            {profile['style_tag']} · {profile['personality']}
                            </div>
                            <div style="font-size: 13px; color: #666; margin-bottom: 12px; font-style: italic;">
                            "{profile['style_desc']}"
                            </div>
                            <div style="background: {profile['color']}10; border-radius: 6px; padding: 10px; margin-bottom: 12px;">
                            <div style="font-size: 12px; color: {profile['color']}; font-weight: bold; margin-bottom: 6px;">
                            ⚡ {profile['skill_name']}
                            </div>
                            <div style="font-size: 12px; color: #555; line-height: 1.5;">
                            {profile['skill_desc']}
                            </div>
                            </div>
                            <div style="display: flex; justify-content: space-between; text-align: center; font-size: 12px;">
                            <div>
                            <div style="color: #888; font-size: 11px;">胜率</div>
                            <div style="font-weight: bold; color: #333; font-size: 14px;">50.8%</div>
                            </div>
                            <div>
                            <div style="color: #888; font-size: 11px;">回撤</div>
                            <div style="font-weight: bold; color: #e74c3c; font-size: 14px;">~25%</div>
                            </div>
                            <div>
                            <div style="color: #888; font-size: 11px;">出手</div>
                            <div style="font-weight: bold; color: #333; font-size: 14px;">~104场</div>
                            </div>
                            </div>
                            <div style="text-align: center; margin-top: 10px; padding-top: 8px; border-top: 1px solid #eee;">
                            <span style="font-size: 13px; font-weight: bold; color: #27ae60;">
                            样本内ROI +14.5%（研究）
                            </span>
                            </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.caption(f"📊 参数：{profile['params']}")
                    
                    # 两个Pro版开关（互斥）
                    col1, col2 = st.columns(2)
                    with col1:
                        st.toggle("德甲专精", value=bundesliga_mode, key='falcon_bundesliga_mode',
                                  help="⚠️ 实验性 Beta（金标准存疑）！无安全边际、出手太多质量差、亏损严重。开启后仅德甲，赔率≥2.5，置信≥50%。")
                    with col2:
                        st.toggle("英超专精", value=epl_mode, key='falcon_epl_mode',
                                  help="⚠️ 实验性 Beta（金标准存疑）！无安全边际、出手太多质量差、亏损严重。开启后仅英超，赔率≥3.0，置信≥50%。")
                    
                    st.caption("⚠️ 无安全边际，出手太多质量差，亏损严重（实验性 Beta，不建议普通用户开启）")
            
            else:
                # ☯️ 八卦#C01（串关AI，待验证）
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {profile['color']}15, {profile['color']}05); 
                            border: 2px dashed {profile['color']}40; border-radius: 12px; padding: 16px; margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-size: 22px; font-weight: bold;">{profile['icon']} {profile['display_name']}</span>
                        <span style="font-size: 11px; padding: 3px 8px; background: #99999920; 
                              color: #999; border-radius: 10px; font-weight: bold;">
                            🚧 待验证
                        </span>
                    </div>
                    <div style="font-size: 12px; color: {profile['color']}; font-weight: bold; margin-bottom: 10px;">
                        {profile['style_tag']} · {profile['personality']}
                    </div>
                    <div style="font-size: 13px; color: #666; margin-bottom: 12px; font-style: italic;">
                        "{profile['style_desc']}"
                    </div>
                    <div style="background: {profile['color']}10; border-radius: 6px; padding: 10px; margin-bottom: 12px;">
                        <div style="font-size: 12px; color: {profile['color']}; font-weight: bold; margin-bottom: 6px;">
                            ⚡ {profile['skill_name']}
                        </div>
                        <div style="font-size: 12px; color: #555; line-height: 1.5;">
                            {profile['skill_desc']}
                        </div>
                    </div>
                    <div style="display: flex; justify-content: space-between; text-align: center; font-size: 12px;">
                        <div>
                            <div style="color: #888; font-size: 11px;">胜率</div>
                            <div style="font-weight: bold; color: #999; font-size: 14px;">--</div>
                        </div>
                        <div>
                            <div style="color: #888; font-size: 11px;">回撤</div>
                            <div style="font-weight: bold; color: #999; font-size: 14px;">--</div>
                        </div>
                        <div>
                            <div style="color: #888; font-size: 11px;">出手</div>
                            <div style="font-weight: bold; color: #999; font-size: 14px;">--</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.caption(f"📊 参数：{profile['params']}")
                st.caption("🚧 串关策略WF验证待生成")


# ========== AI共识分析 ==========
if has_season_data:
    st.markdown("## 🤝 共识分析")
    st.caption("多个AI同时看好的比赛，历史回测中胜率更高 — 15赛季历史数据（研究观察，不构成任何建议）")

    consensus_df = get_consensus_analysis(selected_season)
    if consensus_df is not None and len(consensus_df) > 0:
        st.dataframe(consensus_df, use_container_width=True, hide_index=True)
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.info("""
            **📌 历史回测观察**
            - 三AI共识胜率在历史样本中比单激进高出 30+ 个百分点
            - 15个赛季样本中差值 26%~36%（历史回测，不代表未来）
            - 共识度越高，赔率越低，但胜率提升幅度更大
            """)
        with col_c2:
            st.info("""
            **💡 研究说明**
            - 三AI共识场次：历史样本中命中率更高（研究观察）
            - 仅激进出手场次：历史样本接近抛硬币，谨慎对待
            - 以上均为历史回测规律，不构成任何建议
            """)
    else:
        st.info("暂无共识数据")

# ========== 决策参考 ==========
st.markdown("## 💡 决策参考")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🔬 历史回测观察")
    st.markdown("""
    - **三AI共识**：历史样本中多重信号一致，命中率相对更高（研究观察）
    - **高安全边际**：20%以上价值优势，历史样本中回撤相对更小
    - **强弱分明**：实力差距大的比赛，历史样本中预测准确率更高
    - **模型预测主胜**：历史样本中主胜预测准确率最高
    - **出手质量 > 数量**：历史样本中精选出手的 ROI 优于高频出手
    - ⚠️ 以上均为历史回测规律，不代表未来，不构成任何建议
    """)

with col2:
    st.markdown("### ⚠️ 谨慎参考场景")
    st.markdown("""
    - **模型预测平局**：平局最难预测，谨慎对待
    - **两队状态相当**：接近抛硬币，50%左右
    - **低安全边际**：价值不足，容易亏
    - **高赔率 > 3** 的比赛：冷门难预测
    - **任何比赛都要防平局**：很多翻车都是平局
    """)

# ========== 最近出手记录 ==========
if has_season_data:
    with st.expander("📝 最近出手记录（含筛选）", expanded=False):
        # 筛选器
        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            filt_league = st.selectbox("联赛筛选", ["全部联赛", "英超", "德甲", "西甲", "意甲", "法甲"], key="filt_league")
        with fcol2:
            filt_result = st.selectbox("结果筛选", ["全部", "仅命中", "仅失手"], key="filt_result")
        with fcol3:
            filt_value = st.selectbox("价值筛选", ["全部", "有价值", "无价值"], key="filt_value")

        league_map = {"英超": "E0", "德甲": "D1", "西甲": "LLA", "意甲": "SER", "法甲": "LIG"}
        filt_league_code = league_map.get(filt_league)

        def enrich_bet_df(log_df):
            """给出手记录加派生字段：仓位、共识等级、价值标记"""
            if len(log_df) == 0:
                return log_df
            df = log_df.copy()
            # 仓位比例
            bet_before = df['score_after'] - df['profit']
            df['验证权重'] = (df['bet_amount'] / bet_before).apply(lambda x: f"{x:.1%}")
            # 格式化
            df['confidence_pct'] = df['confidence'].apply(lambda x: f"{x:.1%}")
            df['win_label'] = df['win'].apply(lambda x: "✅ 命中" if x else "❌ 失手")
            df['actual_short'] = df['actual_result'].map({'主队胜': '主胜', '平局': '平局', '客队胜': '客胜'})
            df['pred_short'] = df['prediction'].map({'主队胜': '主胜', '平局': '平局', '客队胜': '客胜'})
            # 价值标记：置信度 > 1/赔率 即为正价值
            implied_prob = 1.0 / df['odds']
            df['value_diff'] = df['confidence'] - implied_prob
            df['价值'] = df['value_diff'].apply(lambda x: "💎 有价值" if x > 0 else "📉 无价值")
            # 共识等级：按比赛分组统计出手AI数量
            df['match_key'] = df['match_date'].astype(str) + '|' + df['home_team'].astype(str) + '|' + df['away_team'].astype(str)
            ai_count = df.groupby('match_key')['ai_name'].nunique()
            df['ai_count'] = df['match_key'].map(ai_count)
            df['共识'] = df['ai_count'].apply(lambda x: "🤝 三AI共识" if x >= 3 else ("⚖️ 两AI" if x == 2 else "🔥 仅激进"))
            return df

        def apply_filters(df):
            if filt_league_code:
                df = df[df['league_code'] == filt_league_code]
            if filt_result == "仅命中":
                df = df[df['win'] == 1]
            elif filt_result == "仅失手":
                df = df[df['win'] == 0]
            if filt_value == "有价值":
                df = df[df['value_diff'] > 0]
            elif filt_value == "无价值":
                df = df[df['value_diff'] <= 0]
            return df

        tab1, tab2, tab3, tab4 = st.tabs(["全部", "🦅 猎鹰#A01", "⚖️ 天秤#B01", "🪨 磐石#D01"])

        with tab1:
            log_df = get_betting_log(season_year=selected_season, limit=200)
            if len(log_df) > 0:
                df = enrich_bet_df(log_df)
                df = apply_filters(df)
                if len(df) > 0:
                    # 映射AI展示名
                    df['AI'] = df['ai_name'].apply(
                        lambda x: f"{AI_PROFILES.get(x, {}).get('icon', '🤖')} {AI_PROFILES.get(x, {}).get('display_name', x)}"
                    )
                    show = df[['match_date', 'AI', 'home_team', 'away_team', 'pred_short', 
                               'confidence_pct', 'odds', '验证权重', '共识', '价值', 'actual_short', 'win_label']]
                    show.columns = ['日期', 'AI', '主队', '客队', '预测', '置信度', '市场概率', '验证权重', '共识', '置信度优势', '赛果', '结果']
                    st.dataframe(show, use_container_width=True, hide_index=True)
                    st.caption(f"共 {len(df)} 条记录 | 胜率: {(df['win'].mean()):.1%}")
                else:
                    st.info("当前筛选条件下无记录")

        for tab_idx, ai_name in enumerate(['激进AI', '中立AI', '保守AI'], start=2):
            with [tab2, tab3, tab4][tab_idx-2]:
                log_df = get_betting_log(ai_name, season_year=selected_season, limit=200)
                if len(log_df) > 0:
                    df = enrich_bet_df(log_df)
                    df = apply_filters(df)
                    if len(df) > 0:
                        show = df[['match_date', 'home_team', 'away_team', 'pred_short', 
                                   'confidence_pct', 'odds', '验证权重', '共识', '价值', 'actual_short', 'win_label']]
                        show.columns = ['日期', '主队', '客队', '预测', '置信度', '市场概率', '验证权重', '共识', '置信度优势', '赛果', '结果']
                        st.dataframe(show, use_container_width=True, hide_index=True)
                        st.caption(f"共 {len(df)} 条记录 | 胜率: {(df['win'].mean()):.1%}")
                    else:
                        st.info("当前筛选条件下无记录")

        # ========== 赔率分布统计 ==========
        st.markdown("---")
        st.markdown("###### 📊 赔率分布统计")
        log_all = get_betting_log(season_year=selected_season, limit=1000)
        if len(log_all) > 0:
            df_all = enrich_bet_df(log_all)
            df_all = apply_filters(df_all)
            if len(df_all) > 0:
                # 赔率分桶
                bins = [0, 1.5, 2.0, 2.5, 3.0, 4.0, 99]
                labels = ["<1.5", "1.5-2.0", "2.0-2.5", "2.5-3.0", "3.0-4.0", "4.0+"]
                df_all['odds_bin'] = pd.cut(df_all['odds'], bins=bins, labels=labels)
                dist = df_all.groupby('odds_bin', observed=True).agg(
                    出手数=('win', 'count'),
                    胜率=('win', 'mean')
                ).reset_index()
                dist['胜率'] = dist['胜率'].apply(lambda x: f"{x:.1%}")
                dist.columns = ['赔率区间', '出手数', '胜率']
                dcol1, dcol2 = st.columns([2, 1])
                dcol1.dataframe(dist, use_container_width=True, hide_index=True)
                dcol2.metric("平均赔率", f"{df_all['odds'].mean():.2f}")
            else:
                st.info("暂无数据")

# ========== 决策参考（组合定位 + 风险提示） ==========
with st.expander("📌 决策参考（研究定位 · 哪些不要碰）", expanded=False):
    st.markdown("""
    **🔬 当前研究定位（P0 产品口径重跑后 · 2026-08-09）：**
    - 🦅 **猎鹰**（研究·高赔冷门档）：外样本 357场 +15.51%（B365 口径样本内）；**真实可成交赔率口径（Pinnacle 终盘）2.2/0.50 n=689 存疑、2.2/0.55 n=176 未过 Bonferroni** → 待 2026-2027 独立期复现，当前仅研究参考
    - 🪨 **磐石Pro**（研究·高命中档）：研究样本 113场 +13.16%，聚类 CI 含 0 → 统计显著不成立，仅研究观察
    - ⚖️ **价值腿已移除**（C-016）：真实赔率复核显示执行层脆弱 + 无后见之明验证不成立（C-015），不再进入任何组合
    - 组合 P(破产)≈0% 为**账面口径**（MC 模拟），不代表真实交易结果

    **⚠️ 明确不要碰（C-012 存疑）：**
    - ☯️ **串关（多场连买）**：2串1 命中率 ≈ 单场命中率²（单场 50% → 2串1 仅 25%），每加一场再砍一半；**不作为推荐策略**
    - 👑 **高级模式（Pro 专精）**：德甲/英超专精、超级组合等样本小、金标准存疑，不作为产品化依据
    - 🎯 **高赔率高置信（≥3.0 且 ≥55%）**：样本少、ROI 转负，雷区
    - 🎲 **平局单吊**：平局组件历史样本中拉低整体（盈利季 52%<60%），勿单独重仓

    **📡 健康监控（C-013）**：系统每天盯猎鹰近 2 年战绩，若滚动 ROI 置信区间下限 < 0 亮黄灯预警，连续下滑即收手。当前状态：🟡 观察（近 3 年 +32.31%，为历史样本 iid CI 显著；产品口径待独立期复现）。
    """)

# ========== 核心验证结论 ==========
st.markdown("## 🎯 核心验证结论")

tab_conf, tab_league = st.tabs(["📈 置信度 vs 准确率", "🏆 联赛难度排行"])

with tab_conf:
    st.caption("Walk Forward滚动验证，无未来函数，模型输出置信度与真实准确率的对应关系")
    conf_acc_df = None
    try:
        # 【2026-08-20】优先读随仓库分发的 model/wf_confidence_accuracy.json（含样本数，
        #   支持低样本降级）；football.db 仅为本地旧环境兑底。
        _json_path = os.path.join(ROOT_DIR, "model", "wf_confidence_accuracy.json")
        if os.path.exists(_json_path):
            with open(_json_path, "r", encoding="utf-8") as f:
                _raw = json.load(f)
            conf_acc_df = pd.DataFrame([d for d in _raw if "置信度区间" in d])
        if conf_acc_df is None or len(conf_acc_df) == 0:
            conn = get_db()
            conf_acc_df = pd.read_sql("SELECT * FROM wf_confidence_accuracy", conn)
            conn.close()
    except Exception as e:
        st.info(f"置信度数据加载失败：{str(e)}")

    if conf_acc_df is not None and len(conf_acc_df) > 0:
        col_c1, col_c2 = st.columns([1, 1])

        with col_c1:
            st.markdown("###### 📊 分桶数据")
            conf_display = conf_acc_df.copy()
            conf_display['准确率'] = conf_display['准确率'].apply(lambda x: f"{x:.2%}")
            if '平均置信度' in conf_display.columns:
                conf_display['平均置信度'] = conf_display['平均置信度'].apply(lambda x: f"{x:.2%}")
                conf_display['高估程度'] = (conf_acc_df['平均置信度'] - conf_acc_df['准确率']).apply(lambda x: f"+{x:.1%}" if x > 0 else f"{x:.1%}")
            else:
                conf_display['平均置信度'] = "—"
                conf_display['高估程度'] = "—"
            # 【2026-08-20 低样本降级】样本<100 的档位（如 ≥90% 仅48场）标注"⚠️样本少"，
            #   避免对 48 场得出的 87.5% 产生过度置信（二项CI约±9pt）。
            if '样本数' in conf_display.columns:
                conf_display['样本提示'] = conf_display['样本数'].apply(lambda n: "⚠️样本少" if n < 100 else "")
                show_cols = ['置信度区间', '样本数', '平均置信度', '准确率', '高估程度', '样本提示']
            else:
                show_cols = ['置信度区间', '平均置信度', '准确率', '高估程度']
            st.dataframe(conf_display[show_cols], hide_index=True, use_container_width=True, height=400)
            st.caption("「高估程度」= 模型报的置信度 − 实际命中率；越接近 0 说明模型越诚实，正值代表模型过于自信（实际没那么准）。")

        with col_c2:
            st.markdown("###### 📈 曲线图")
            try:
                chart_df = conf_acc_df.copy()
                chart_df = chart_df.set_index('置信度区间')[['准确率', '平均置信度']]
                chart_df.columns = ['真实准确率', '模型输出置信度']
                st.line_chart(chart_df, height=400)
            except Exception as e:
                st.caption(f"图表渲染失败：{str(e)}")

        st.success("""
        📌 **关键结论**（数据随 WF 融合金标准自动更新）：
        1. 置信度与准确率正相关，趋势完全正确
        2. 高置信度区间 → 真实准确率显著更高
        3. 校准后模型置信度与真实准确率基本一致
        4. 高置信度区间历史样本中命中率更高（研究观察，不构成下注依据）
        ⚠️ ≥90% 档仅 48 场，87.5% 的置信区间约 ±9pt，仅供参考。
        """)
    else:
        st.info("置信度分桶数据暂不可用（请先运行 features/recalc_confidence_buckets.py 生成）")

with tab_league:
    st.caption("联赛独立模型Walk Forward验证，准确率越高说明规律越稳定、越好预测")
    try:
        conn = get_db()
        league_rank_df = pd.read_sql("SELECT * FROM league_independent_wf ORDER BY 整体准确率 DESC", conn)
        conn.close()

        col_l1, col_l2 = st.columns([1, 1])

        with col_l1:
            st.markdown("###### 📊 详细数据")
            league_display = league_rank_df.copy()
            league_display['整体准确率'] = league_display['整体准确率'].apply(lambda x: f"{x:.2%}")
            league_display['平均置信度'] = league_display['平均置信度'].apply(lambda x: f"{x:.2%}")
            league_display['>=70%准确率'] = league_display['>=70%准确率'].apply(lambda x: f"{x:.2%}")
            league_display = league_display[['联赛', '总场次', '整体准确率', '平均置信度', '>=70%准确率']]
            league_display.columns = ['联赛', '验证场次', '整体准确率', '平均置信度', '高置信准确率']
            st.dataframe(league_display, hide_index=True, use_container_width=True, height=300)

        with col_l2:
            st.markdown("###### 📈 准确率对比")
            try:
                import altair as alt
                _ldata = league_rank_df.copy()
                _lmelt = _ldata.melt(id_vars=['联赛'], value_vars=['整体准确率', '>=70%准确率'],
                                     var_name='指标', value_name='准确率')
                _lchart = alt.Chart(_lmelt).mark_bar().encode(
                    x=alt.X('联赛:N', sort=None, axis=alt.Axis(labelAngle=0, title=None)),
                    xOffset='指标:N',
                    y=alt.Y('准确率:Q', axis=alt.Axis(format='%', title='准确率')),
                    color=alt.Color('指标:N', scale=alt.Scale(scheme='category10'))
                ).properties(height=350)
                st.altair_chart(_lchart, use_container_width=True)
            except Exception as e:
                st.caption(f"图表渲染失败：{str(e)}")

        st.info("""
        💡 **关键发现**（按当前 WF 金标准数据自动排序）：
        1. 整体准确率越高的联赛，规律越稳定、越好预测
        2. 高置信度区间（≥70%）的准确率普遍高于整体
        3. 联赛间差异源于强弱分明程度与冷门密度
        4. **联赛独立模型 < 通用模型** — 单个联赛数据量少，不如全联赛训练的泛化能力强
        """)
    except Exception as e:
        st.info(f"联赛排名数据加载失败：{str(e)}")

st.divider()

# ========== 策略动物园（二级参考，折叠收起） ==========
with st.expander("🦓 策略动物园 · 36组参数回测排行榜（点击展开）"):
    st.warning("⚠️ **数据版本提示**：当前 strategy_zoo 表为身价修复前（v1.0.4）的旧回测数据。"
               "身价特征修复 + 全模型重训（v1.3.0）后如需更新，请重新运行策略回测脚本生成新表。")
    st.caption("探索不同参数组合的历史表现，属于二级参考指标，不影响核心4个AI策略")
    
    tab_wf, tab_full = st.tabs(["⭐⭐⭐⭐⭐ WF赛季重置版（金标准）", "⭐ 全量基准版（偏乐观·参考）"])
    
    with tab_wf:
        st.success("""
        ✅ **金标准：Walk Forward 赛季重置版**
        - 滚动训练，无未来函数，最接近真实情况
        - 每赛季初始积分5000，赛季结束重置
        - 12个测试赛季（2014-2026）
        """)
        conn = get_db()
        wf_df = pd.read_sql("SELECT * FROM strategy_zoo_wf_a_36 ORDER BY sharpe_ratio DESC", conn)
        
        if len(wf_df) > 0:
            # 筛选器
            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                sort_by_wf = st.selectbox("排序方式", ["夏普比率", "胜率", "最大回撤", "出手数", "ROI"], key="zoo_sort_wf")
            with col_f2:
                min_win_wf = st.slider("最低胜率", 0.4, 0.7, 0.5, 0.05, key="zoo_win_wf")
            with col_f3:
                max_dd_wf = st.slider("最大回撤上限", 0.3, 1.0, 0.8, 0.05, key="zoo_dd_wf")
            
            # 应用筛选
            wf_filt = wf_df[wf_df['win_rate'] >= min_win_wf].copy()
            wf_filt = wf_filt[wf_filt['avg_max_drawdown'] <= max_dd_wf].copy()
            
            sort_map_wf = {
                "夏普比率": "sharpe_ratio", 
                "胜率": "win_rate", 
                "最大回撤": "avg_max_drawdown", 
                "出手数": "avg_bets_per_season",
                "ROI": "avg_roi"
            }
            ascending = sort_by_wf == "最大回撤"
            wf_filt = wf_filt.sort_values(sort_map_wf[sort_by_wf], ascending=ascending)
            
            # 核心发现
            st.markdown("### 🔍 WF版核心发现")
            col_d1, col_d2, col_d3 = st.columns(3)
            best_sharpe_wf = wf_df.iloc[0]
            best_win_wf = wf_df.loc[wf_df['win_rate'].idxmax()]
            best_dd_wf = wf_df.loc[wf_df['avg_max_drawdown'].idxmin()]
            
            with col_d1:
                st.metric("最高夏普", f"{best_sharpe_wf['sharpe_ratio']:.2f}", 
                          f"≥{best_sharpe_wf['min_confidence']:.0%}置信 + {(best_sharpe_wf['value_margin']-1)*100:.0f}%边际")
            with col_d2:
                st.metric("最高胜率", f"{best_win_wf['win_rate']:.1%}", 
                          f"≥{best_win_wf['min_confidence']:.0%}置信 + {(best_win_wf['value_margin']-1)*100:.0f}%边际")
            with col_d3:
                st.metric("最小回撤", f"{best_dd_wf['avg_max_drawdown']:.1%}", 
                          f"凯利{best_dd_wf['kelly_fraction']:.1f} + {(best_dd_wf['value_margin']-1)*100:.0f}%边际")
            
            st.info("""
            **规律总结（WF金标准·新口径）：**
            1. **安全边际是第一生产力**：1.2x边际的策略夏普最高，比1.0x高约20%
            2. **置信度甜点在55%~65%**：太高或太低都不好
            3. **质量 > 数量**：精选策略出手少但ROI高（猎鹰/磐石固定ROI +15%~+17%）
            4. **凯利只影响波动**：同样条件下，凯利大小不影响夏普，只放大收益和回撤
            5. ⚠️ 下表为旧版策略动物园数据（v1），新口径金标准结果见上方核心结论
            """)
            
            # 🎲 随机基准对比
            st.markdown("### 🎲 随机基准对比（500次蒙特卡洛）")
            st.caption("和纯随机策略对比，验证AI是否真的有预测能力")
            
            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                st.metric("随机策略平均ROI", "-8.6%", 
                          "6种随机策略平均（输抽水）")
            with col_r2:
                st.metric("最好随机策略", "-4.4%", 
                          "热门猎手")
            with col_r3:
                st.metric("新口径精选AI", "+15%~+17%", 
                          "猎鹰/磐石Pro（金标准）", delta_color="normal")
            
            st.success("""
            ✅ **结论：AI有真实alpha，但需严格筛选**
            - 旧版36组参数中最好的仅+0.9%（参数网格未优化）
            - 新口径金标准：猎鹰/磐石Pro等精选策略固定ROI +15%~+17%，盈利季70%+
            - 关键：安全边际1.2x + 置信度甜点 + 联赛/赔率筛选
            - ⚠️ 下表为旧版数据，新口径结果见上方三AI角色卡和核心结论
            """)
            
            st.markdown("---")
            
            # 格式化展示
            wf_show = wf_filt.head(36).copy()
            wf_show['排名'] = wf_show['rank'] + 1
            wf_show['置信度'] = wf_show['min_confidence'].apply(lambda x: f"≥{x:.0%}")
            wf_show['凯利系数'] = wf_show['kelly_fraction'].apply(lambda x: f"{x:.1f}")
            wf_show['安全边际'] = wf_show['value_margin'].apply(lambda x: f"{(x-1)*100:.0f}%" if x > 1 else "无")
            wf_show['胜率'] = wf_show['win_rate'].apply(lambda x: f"{x:.1%}")
            wf_show['平均赔率'] = wf_show['avg_odds'].apply(lambda x: f"{x:.2f}")
            wf_show['最大回撤'] = wf_show['avg_max_drawdown'].apply(lambda x: f"{x:.1%}")
            wf_show['出手/赛季'] = wf_show['avg_bets_per_season'].astype(int)
            wf_show['赛季ROI'] = wf_show['avg_roi'].apply(lambda x: f"{x:+.1%}")
            wf_show['夏普'] = wf_show['sharpe_ratio'].round(2)
            
            display_cols = ['排名', '置信度', '凯利系数', '安全边际', '出手/赛季', '胜率', '平均赔率', '最大回撤', '赛季ROI', '夏普']
            st.dataframe(wf_show[display_cols], use_container_width=True, hide_index=True, height=520)
        else:
            st.info("WF版策略动物园数据未生成")
        conn.close()
    
    with tab_full:
        st.warning("""
        ⚠️ **注意：全量基准版存在未来函数，严重偏乐观！**
        - 全量数据训练，全量数据测试，相当于开了上帝视角
        - 结果虚高，仅供快速筛选和方向探索参考
        - **绝对不能作为最终结论**，一切以WF版为准
        """)
        conn = get_db()
        full_df = pd.read_sql("SELECT * FROM strategy_zoo_full_a_36 ORDER BY sharpe_ratio DESC", conn)
        
        if len(full_df) > 0:
            # 格式化展示
            full_show = full_df.head(36).copy()
            full_show['排名'] = full_show['rank'] + 1
            full_show['置信度'] = full_show['min_confidence'].apply(lambda x: f"≥{x:.0%}")
            full_show['凯利系数'] = full_show['kelly_fraction'].apply(lambda x: f"{x:.1f}")
            full_show['安全边际'] = full_show['value_margin'].apply(lambda x: f"{(x-1)*100:.0f}%" if x > 1 else "无")
            full_show['胜率'] = full_show['win_rate'].apply(lambda x: f"{x:.1%}")
            full_show['平均赔率'] = full_show['avg_odds'].apply(lambda x: f"{x:.2f}")
            full_show['最大回撤'] = full_show['avg_max_drawdown'].apply(lambda x: f"{x:.1%}")
            full_show['出手/赛季'] = full_show['avg_bets_per_season'].astype(int)
            full_show['赛季ROI'] = full_show['avg_roi'].apply(lambda x: f"{x:+.1%}")
            full_show['夏普'] = full_show['sharpe_ratio'].round(2)
            
            display_cols = ['排名', '置信度', '凯利系数', '安全边际', '出手/赛季', '胜率', '平均赔率', '最大回撤', '赛季ROI', '夏普']
            st.dataframe(full_show[display_cols], use_container_width=True, hide_index=True, height=520)
            
            st.info("💡 对比WF版可以看到：全量版夏普最高3.00（含未来函数，虚高），WF版旧数据夏普约0.09；新口径金标准精选策略夏普0.60~0.68——这就是未来函数和严格筛选的差距！")
        else:
            st.info("全量版策略动物园数据未生成")
        conn.close()

# ========== 最新策略研究发现 ==========
with st.expander("🔬 历史策略研究（v1.0.4 · 基于修复前身价数据）", expanded=False):
    st.warning("⚠️ **本节为 v1.0.4 时代（身价特征修复前）的策略研究存档。** "
               "2026-08 身价特征 value_ratio 漂移修复 + 全模型重训（v1.3.0）后，文中 ROI/回撤 等数字已失效。"
               "当前金标准以「核心验证结论」页的 WF 融合结果为准（整体准确率 51.75%，无未来函数）。")
    st.caption("历史研究成果存档 · 非当前体系结论 · 仅供回溯参考")
    
    col_r1, col_r2 = st.columns(2)
    
    with col_r1:
        st.markdown("#### 1️⃣ 出手越少，ROI越高")
        st.info("""
        **核心发现：完全单调递减！**
        
        | 每周最低出手 | 平均赛季ROI |
        |-------------|-----------|
        | 无限制 | **+9.29%** 🏆 |
        | 1次 | +9.05% |
        | 2次（基准） | +8.47% |
        | 5次 | -0.22% |
        
        **结论**：强制补仓的比赛质量差，拉低收益。宁缺毋滥！
        """)
        
        st.markdown("#### 2️⃣ 联赛筛选效果显著")
        st.success("""
        **核心发现：德甲+意甲精选联赛，回撤更低**
        
        | 方案 | 固定ROI | 盈利赛季 | 最大回撤(MC) |
        |------|---------|---------|-------------|
        | 全联赛价值基准 | +4.79% | 64% | - |
        | 德甲+意甲（磐石） | **+16.75%** ✅ | 70% | ~23% |
        
        **结论**：精选联赛+严格筛选，ROI和稳定性都更好
        """)
        
        st.markdown("#### 3️⃣ 置信度打折验证")
        st.info("""
        **新口径（WF分桶命中率）下打折有效：**
        - 不打折：n=997，命中率49.8%，固定ROI +11.39%，盈利季68%
        - 打折（前端当前）：n=203，命中率58.1%，固定ROI **+16.75%**，盈利季70%
        - 打折后出手更少但更精选，ROI提升约5.4%
        """)
    
    with col_r2:
        st.markdown("#### 4️⃣ 平局组件验证")
        st.warning("""
        **新口径下平局组件对ROI贡献有限：**
        
        | 方案 | 固定ROI | 盈利赛季 | 出手/赛季 |
        |------|---------|---------|----------|
        | 仅磐石（打折） | **+16.75%** ✅ | 70% | ~8 |
        | 磐石+平局（超级组合） | +11.39% | 52% | ~13 |
        
        **说明**：平局组件已加严价值门槛(1.2)，加入后整体ROI反而低于纯磐石，
        盈利赛季也从70%降到52%。介意者可在高级模式关闭平局组件。
        """)
        
        st.markdown("#### 👑 磐石Pro（当前前端版）")
        st.markdown("""
        **当前配置：**
        - ✅ 联赛筛选：德甲 + 意甲
        - ✅ 主胜/客胜：命中率≥55% + 边际1.2（含置信度打折）
        - ✅ 平局组件：可选（默认开启，加严门槛1.2）
        - ✅ 仓位建议：固定1%~2%（凯利仅排序参考）
        
        **金标准成绩（仅磐石）：**
        - 🏆 固定ROI：**+16.75%**（n=203，盈利季70%）
        - 📉 MC最大回撤：**~23%**（固定2%仓位）
        - 🎯 出手/赛季：**~8场**（精选）
        """)
        
        st.warning("""
        ⚠️ **风险提示：**
        - 样本量有限（203场），存在选择偏差可能
        - 联赛筛选有数据窥探风险，未来不一定保持
        - 建议固定小仓位（1%~2%），不要加注
        """)

# ========== 策略说明 ==========
with st.expander("📖 策略规则说明"):
    st.markdown("""
    ### 当前研究定位（P0 产品口径重跑后 · 2026-08-09）
    - 🦅 **猎鹰**（研究·高赔冷门档）：外样本 357场 +15.51%（B365 样本内）；真实可成交赔率口径 2.2/0.55 n=176 未过 Bonferroni → 待 2026-2027 独立期复现，仅研究参考
    - 🪨 **磐石Pro**（研究·高命中档）：研究样本 113场 +13.16%，聚类 CI 含 0 → 统计不显著，仅研究观察
    - ⚖️ **价值腿已移除**（C-015/016）：无后见之明验证不成立 + 真实赔率复核执行层脆弱 → 不再进入任何组合
    - 组合 P(破产)≈0% 为账面口径（MC 模拟），不代表真实交易结果；健康监控（C-013）每天盯战绩，滚动 2 年 CI 下限 <0 亮黄灯预警

    ### 基础规则（赛季重置模拟）
    - 每个赛季初始积分：5000，赛季结束重置，不滚存
    - **宁缺毋滥**：只下模型最有把握的场次，不强制补仓（研究证明强制补仓拉低收益）
    - 单注上限：5%（防止一把梭哈）
    
    ### 三AI策略参数（WF金标准验证）
    | AI | 风格定位 | 置信度门槛(命中率) | 凯利系数(仅排序参考) | 价值安全边际 |
    |----|---------|-----------|---------|-------------|
    | 🔥 激进AI | 高频价值型 | ≥50% | 0.20 | 20% |
    | ⚖️ 中立AI | 均衡精选型 | ≥60% | 0.30 | 10% |
    | 🛡️ 保守AI | 极致稳型 | ≥70% | 0.20 | 20% |
    > 【2026-08-04 回调】依据《新口径三AI网格与超级组合复核》：激进AI 边际 1.0→1.2（原1.0档盈利季56%存疑，1.2档+14.5%/80%盈利季✅成立）；
    > 置信门槛0.50/0.60/0.70 对齐命中率梯度 55.2%/65.4%/76.4%；**历史 MC 复核显示固定 1%~2% 较安全（研究侧，非建议）**（凯利系数仅用于排序参考，MC显示凯利下注有破产风险）。
    
    ### 三级验证体系
    ⭐ **Level 1：全量回测**（最低可信度）
    - 全量数据训练，全量数据测试
    - 用途：快速筛选、方向探索
    - 严重偏乐观，仅供参考
    
    ⭐⭐⭐ **Level 2：赛季重置**（中等可信度）
    - 全量数据训练，按赛季重置资金
    - 用途：策略稳定性初步评估
    - 仍有未来函数
    
    ⭐⭐⭐⭐⭐ **Level 3：WF赛季重置**（最高可信度·金标准）
    - 滚动训练，无未来函数，赛季重置资金
    - 用途：最终结论、策略上线评估
    - 所有策略上线前必须通过此验证
    
    ### 关于积分数字
    足球预测很难，模型alpha有限，真实环境下很难稳定盈利。
    **相对排名和策略特性才是有价值的参考**。
    
    ### 风格定位说明
    三个AI是**按风险偏好**区分的，不是按好坏排名。保守AI仓位最轻，但不代表激进AI"差"——只是风险偏好不同，适合不同决策场景。
    
    ### 数据来源
    - 赔率：B365终盘
    - 模型：Walk-Forward滚动验证（纯OOS数据，12赛季2万+场）
    - 参数优化：策略动物园网格搜索 + WF版验证
    
    ### 🎯 核心结论（WF版验证）
    
    **1. 模型有真实alpha，但需严格筛选**
    - 全量argmax准确率约51.75%（+约6.0pt vs主胜基准45.8%）
    - 精选策略（猎鹰/磐石Pro）固定ROI +15%~+17%，盈利季70%+
    - 但高置信（≥60%）固定ROI仅+0.22%，不是越自信越好
    
    **2. 安全边际非常重要**
    - 1.2x（20%边际）是甜点，夏普最高
    - 1.0x（无边际）基本打平或微亏
    - 价值筛选确实有效，能过滤低质量注单
    
    **3. 质量远大于数量**
    - 精选策略出手少（8~70场/季）但ROI高
    - 高频策略（数千场/季）ROI低甚至为负
    - 宁可少出手，也要选最有把握的
    
    **4. 没有圣杯，注意风险**
    - 策略有波动，单赛季可能亏损（盈利季60%~80%）
    - 建议固定小仓位（1%~2%），不要用凯利重仓
    - 高赔率高置信（≥3.0/≥55%）是雷区，样本少ROI转负
    """)

# ========== 积分与增长倍数（数据记录，仅供参考） ==========
if has_season_data:
    with st.expander("📊 积分走势与增长倍数（数据记录·仅供参考）", expanded=False):
        st.warning("""
        ⚠️ **重要提示：以下数据基于全量训练模型，存在未来函数，仅供研究记录使用**
        
        - 全量训练模型已经见过所有比赛数据，相当于开了上帝视角
        - 凯利公式 + 高频出手的复利效应会导致积分呈指数级增长，数字虚高
        - 真实 Walk Forward（无未来函数）下的增长倍数仅为数千倍，远低于此
        - **核心参考指标：胜率、回撤、夏普，增长倍数仅作数据记录**
        """)
        
        # 积分走势图
        st.markdown("### 📈 积分走势（对数刻度）")
        st.caption("单赛季内积分变化（每个赛季独立重置500），复利效应导致数字虚高，主要看相对节奏和回撤深度")

        chart_data = pd.DataFrame()
        for ai_name, profile in AI_PROFILES.items():
            if profile.get('placeholder'):
                continue
            curve = get_score_curve(ai_name, selected_season)
            if len(curve) > 0:
                curve = curve.reset_index(drop=True)
                curve['idx'] = curve.index  # 用累计序号做x轴，避免日期重复
                display_name = f"{profile['icon']} {profile['display_name']}"
                curve = curve.set_index('idx')[['score_after']].rename(columns={'score_after': display_name})
                chart_data = pd.concat([chart_data, curve], axis=1)

        if len(chart_data) > 0:
            # 对数转换后展示（加1防止log(0)）
            log_data = np.log10(chart_data.clip(lower=1))
            st.line_chart(log_data, height=350)
            st.caption("纵坐标为积分的10为底对数值，数值差异被压缩，主要看趋势和相对位置")
        else:
            st.info("暂无走势数据")
        
        st.markdown("---")
        
        # 详细数据表
        st.markdown("### 📋 赛季详细数据")
        if len(summary_df) > 0:
            display_df = summary_df.copy()
            # 映射AI展示名
            display_df['AI'] = display_df['ai_name'].apply(
                lambda x: f"{AI_PROFILES.get(x, {}).get('icon', '🤖')} {AI_PROFILES.get(x, {}).get('display_name', x)}"
            )
            display_df['增长倍数'] = display_df['roi'].apply(format_growth_multiplier)
            display_df['胜率'] = display_df['win_rate'].apply(lambda x: f"{x:.2%}")
            display_df['最大回撤'] = display_df['max_drawdown'].apply(lambda x: f"{x:.2%}")
            display_df['深度回撤'] = display_df['is_bankrupt'].apply(lambda x: "是" if x else "否")
            st.dataframe(
                display_df[['AI', '增长倍数', '胜率', 'total_bets', '最大回撤', '深度回撤']].rename(columns={'total_bets': '出手数'}), 
                use_container_width=True, hide_index=True
            )
            st.caption("增长倍数为全量训练模型的理论计算值，存在未来函数，仅供参考")
