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
        "skill_desc": "只做有价值的比赛，仓位最轻，抗波动能力最强",
        "personality": "佛系老大哥 · 稳如泰山 · 抗抽水最强",
        "params": "置信度≥55% · 凯利0.20 · 20%安全边际",
        "suitable": "求稳为主、优先控制风险的场景",
    },
    "中立AI": {
        "display_name": "天秤#B01",
        "icon": "⚖️",
        "color": "#3498db",
        "style_tag": "均衡型",
        "style_desc": "两边都要，但要算清楚",
        "skill_name": "【均衡之道】",
        "skill_desc": "在收益和风险之间找最优平衡点，不冒进也不保守",
        "personality": "均衡大师 · 选择困难症救星 · 综合最优",
        "params": "置信度≥55% · 凯利0.60 · 10%安全边际",
        "suitable": "大多数场景的默认选择，攻守兼备",
    },
    "激进AI": {
        "display_name": "猎鹰#A01",
        "icon": "🦅",
        "color": "#e74c3c",
        "style_tag": "攻击型",
        "style_desc": "梭哈，赢了会所嫩模",
        "skill_name": "【疾风突袭】",
        "skill_desc": "有价值就敢上，仓位最重，波动最大",
        "personality": "激进派猎手 · 收益天花板 · 波动大",
        "params": "置信度≥55% · 凯利0.90 · 0%安全边际",
        "suitable": "追求高覆盖、能承受较大波动的场景",
    },
    "串关AI": {
        "display_name": "八卦#C01",
        "icon": "☯️",
        "color": "#9b59b6",
        "style_tag": "策略型",
        "style_desc": "分散风险，稳中求胜",
        "skill_name": "【八卦阵】",
        "skill_desc": "2串1保守型，用串关分散风险，以柔克刚",
        "personality": "防守大师 · 串关狂魔 · 以柔克刚",
        "params": "2串1 · 置信度≥70% · 凯利0.20",
        "suitable": "喜欢串关、追求高赔率的场景",
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

    summary_df = get_season_summary(selected_season)
    cols = st.columns(4)

    for idx, (ai_name, profile) in enumerate(AI_PROFILES.items()):
        ai_data = summary_df[summary_df['ai_name'] == ai_name] if not profile.get('placeholder') else pd.DataFrame()
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
                
                st.caption(f"📊 参数：{profile['params']}")
                st.caption("🚧 数据准备中，敬请期待")
else:
    # 没有赛季数据时，展示AI角色卡占位符
    st.markdown("## 🎴 AI 策略角色卡")
    cols = st.columns(4)
    for idx, (ai_name, profile) in enumerate(AI_PROFILES.items()):
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
                            超级组合 · 精选之王 · 收益天花板
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
                                <div style="font-weight: bold; color: #27ae60; font-size: 14px;">51.9%</div>
                            </div>
                            <div>
                                <div style="color: #888; font-size: 11px;">回撤</div>
                                <div style="font-weight: bold; color: #e74c3c; font-size: 14px;">38.7%</div>
                            </div>
                            <div>
                                <div style="color: #888; font-size: 11px;">出手</div>
                                <div style="font-weight: bold; color: #333; font-size: 14px;">35场</div>
                            </div>
                        </div>
                        <div style="text-align: center; margin-top: 10px; padding-top: 8px; border-top: 1px solid #eee;">
                            <span style="font-size: 13px; font-weight: bold; color: #f39c12;">
                                赛季ROI：+22.6% 👑
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.caption("📊 参数：联赛筛选(德甲+意甲) · 主胜/客胜≥55% · 平局≥50% · 凯利0.2")
                    
                    # 高级模式开关
                    st.toggle("👑 高级模式", value=advanced_mode, key='panshi_advanced_mode',
                              help="开启后启用超级组合策略：联赛筛选+平局联动+最优参数")
                else:
                    # 🪨 普通版角色卡（A方案基准）
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, {profile['color']}15, {profile['color']}05); 
                                border: 2px solid {profile['color']}40; border-radius: 12px; padding: 16px; margin-bottom: 10px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span style="font-size: 22px; font-weight: bold;">{profile['icon']} {profile['display_name']}</span>
                            <span style="font-size: 11px; padding: 3px 8px; background: {profile['color']}20; 
                                  color: {profile['color']}; border-radius: 10px; font-weight: bold;">
                                ✅ 已验证
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
                                <div style="font-weight: bold; color: #333; font-size: 14px;">53.9%</div>
                            </div>
                            <div>
                                <div style="color: #888; font-size: 11px;">回撤</div>
                                <div style="font-weight: bold; color: #e74c3c; font-size: 14px;">56.0%</div>
                            </div>
                            <div>
                                <div style="color: #888; font-size: 11px;">出手</div>
                                <div style="font-weight: bold; color: #333; font-size: 14px;">153场</div>
                            </div>
                        </div>
                        <div style="text-align: center; margin-top: 10px; padding-top: 8px; border-top: 1px solid #eee;">
                            <span style="font-size: 13px; font-weight: bold; color: #27ae60;">
                                赛季ROI：+0.9% ✅
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.caption(f"📊 参数：{profile['params']}")
                    
                    # 高级模式开关
                    st.toggle("👑 高级模式", value=advanced_mode, key='panshi_advanced_mode',
                              help="开启后启用超级组合策略：联赛筛选+平局联动+最优参数")
            
            elif ai_name == '中立AI':
                # ⚖️ 天秤#B01（均衡型）
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {profile['color']}15, {profile['color']}05); 
                            border: 2px solid {profile['color']}40; border-radius: 12px; padding: 16px; margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-size: 22px; font-weight: bold;">{profile['icon']} {profile['display_name']}</span>
                        <span style="font-size: 11px; padding: 3px 8px; background: #f39c1220; 
                              color: #f39c12; border-radius: 10px; font-weight: bold;">
                            ⚠️ 待优化
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
                            <div style="font-weight: bold; color: #333; font-size: 14px;">55.6%</div>
                        </div>
                        <div>
                            <div style="color: #888; font-size: 11px;">回撤</div>
                            <div style="font-weight: bold; color: #e74c3c; font-size: 14px;">69.1%</div>
                        </div>
                        <div>
                            <div style="color: #888; font-size: 11px;">出手</div>
                            <div style="font-weight: bold; color: #333; font-size: 14px;">288场</div>
                        </div>
                    </div>
                    <div style="text-align: center; margin-top: 10px; padding-top: 8px; border-top: 1px solid #eee;">
                        <span style="font-size: 13px; font-weight: bold; color: #e74c3c;">
                            赛季ROI：-30.8% ⚠️
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.caption(f"📊 参数：{profile['params']}")
                st.caption("💡 安全边际偏低，价值筛选不足，有待优化")
            
            elif ai_name == '激进AI':
                # 🦅 猎鹰#A01（攻击型）
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {profile['color']}15, {profile['color']}05); 
                            border: 2px solid {profile['color']}40; border-radius: 12px; padding: 16px; margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-size: 22px; font-weight: bold;">{profile['icon']} {profile['display_name']}</span>
                        <span style="font-size: 11px; padding: 3px 8px; background: #e74c3c20; 
                              color: #e74c3c; border-radius: 10px; font-weight: bold;">
                            ⚠️ 待优化
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
                            <div style="font-weight: bold; color: #333; font-size: 14px;">63.3%</div>
                        </div>
                        <div>
                            <div style="color: #888; font-size: 11px;">回撤</div>
                            <div style="font-weight: bold; color: #e74c3c; font-size: 14px;">79.6%</div>
                        </div>
                        <div>
                            <div style="color: #888; font-size: 11px;">出手</div>
                            <div style="font-weight: bold; color: #333; font-size: 14px;">739场</div>
                        </div>
                    </div>
                    <div style="text-align: center; margin-top: 10px; padding-top: 8px; border-top: 1px solid #eee;">
                        <span style="font-size: 13px; font-weight: bold; color: #e74c3c;">
                            赛季ROI：-64.1% ⚠️
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.caption(f"📊 参数：{profile['params']}")
                st.caption("💡 无安全边际，出手太多质量差，亏损严重")
            
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
    st.caption("多个AI同时看好的比赛，可靠性显著更高 — 15赛季历史数据验证")

    consensus_df = get_consensus_analysis(selected_season)
    if consensus_df is not None and len(consensus_df) > 0:
        st.dataframe(consensus_df, use_container_width=True, hide_index=True)
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.success("""
            **📌 核心规律**
            - 三AI共识胜率比单激进高出 **30+ 个百分点**
            - 15个赛季全部稳定成立，差值26%~36%
            - 共识度越高，赔率越低，但胜率提升幅度更大
            """)
        with col_c2:
            st.info("""
            **💡 决策参考**
            - 三AI共识场次：放心重仓，可靠性接近保守AI
            - 仅激进出手场次：谨慎对待，接近抛硬币
            - 共识度是比单一置信度更可靠的筛选信号
            """)
    else:
        st.info("暂无共识数据")

# ========== 决策参考 ==========
st.markdown("## 💡 决策参考")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### ✅ 相对可靠场景")
    st.markdown("""
    - **三AI共识**：多重验证，可靠性更高
    - **高安全边际**：20%以上价值优势，亏得少
    - **强弱分明**：实力差距大的比赛，预测更准
    - **模型预测主胜**：主胜预测最可靠
    - **出手质量 > 数量**：宁可少出手，也要选最有把握的
    """)

with col2:
    st.markdown("### ⚠️ 谨慎出手场景")
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

        tab1, tab2, tab3, tab4 = st.tabs(["全部", "🦅 猎鹰#S03", "⚖️ 天秤#S02", "🪨 磐石#S01"])

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

# ========== 核心验证结论 ==========
st.markdown("## 🎯 核心验证结论")

tab_conf, tab_league = st.tabs(["📈 置信度 vs 准确率", "🏆 联赛难度排行"])

with tab_conf:
    st.caption("Walk Forward滚动验证，无未来函数，模型输出置信度与真实准确率的对应关系")
    try:
        conn = get_db()
        conf_acc_df = pd.read_sql("SELECT * FROM wf_confidence_accuracy", conn)
        conn.close()

        col_c1, col_c2 = st.columns([1, 1])

        with col_c1:
            st.markdown("###### 📊 分桶数据")
            conf_display = conf_acc_df.copy()
            conf_display['准确率'] = conf_display['准确率'].apply(lambda x: f"{x:.2%}")
            conf_display['平均置信度'] = conf_display['平均置信度'].apply(lambda x: f"{x:.2%}")
            conf_display['高估程度'] = (conf_acc_df['平均置信度'] - conf_acc_df['准确率']).apply(lambda x: f"+{x:.1%}" if x > 0 else f"{x:.1%}")
            conf_display = conf_display[['置信度区间', '样本数', '平均置信度', '准确率', '高估程度']]
            st.dataframe(conf_display, hide_index=True, use_container_width=True, height=400)

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
        📌 **关键结论**：
        1. 置信度与准确率正相关，趋势完全正确
        2. 85%以上高置信度 → 真实准确率约82%
        3. 模型整体略微高估置信度（平均高4-6个百分点）
        4. 高置信度区间高估更明显（约8个百分点）
        """)
    except Exception as e:
        st.info(f"置信度数据加载失败：{str(e)}")

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

# ========== 策略动物园（二级参考，折叠收起） ==========
with st.expander("🦓 策略动物园 · 36组参数回测排行榜（点击展开）"):
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
            **规律总结（WF金标准）：**
            1. **模型alpha有限**：最好的策略夏普也只有0.09，接近随机水平
            2. **安全边际非常重要**：20%安全边际的策略亏得最少，甚至勉强打平
            3. **质量 > 数量**：出手越少，亏得越少；宁可少出手，也要选最有把握的
            4. **36个策略全部存活**：没有破产的，但大部分是亏损的
            5. **凯利只影响波动**：同样条件下，凯利大小不影响夏普，只放大收益和回撤
            """)
            
            # 🎲 随机基准对比
            st.markdown("### 🎲 随机基准对比（500次蒙特卡洛）")
            st.caption("和纯随机策略对比，验证AI是否真的有预测能力")
            
            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                st.metric("随机策略平均ROI", "-8.6%", 
                          "6种随机策略平均")
            with col_r2:
                st.metric("最好随机策略", "-4.4%", 
                          "热门猎手")
            with col_r3:
                st.metric("最好AI策略", "+0.9%", 
                          "磐石#D01", delta_color="normal")
            
            st.success("""
            ✅ **结论：AI确实有alpha，但很微弱**
            - 最好的AI策略（磐石）比最好的随机策略好约5个百分点
            - 但alpha非常有限，远不足以覆盖交易成本
            - 符合市场有效性假说：足球博彩市场接近有效
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
            
            st.info("💡 对比WF版可以看到：全量版夏普最高3.00，WF版只有0.09，差了33倍——这就是未来函数的威力！")
        else:
            st.info("全量版策略动物园数据未生成")
        conn.close()

# ========== 最新策略研究发现 ==========
with st.expander("🔬 最新策略研究发现（v1.0.4）", expanded=False):
    st.caption("基于WF金标准的深度策略探索，属于前沿研究成果")
    
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
        **核心发现：德甲+意甲1+1>2！**
        
        | 方案 | 平均赛季ROI | 最大回撤 |
        |------|-----------|---------|
        | 所有联赛（基准） | +9.29% | 47.57% |
        | 只投德甲 | +10.81% | **17.62%** ✅ |
        | **德甲+意甲** | **+17.91%** 🚀 | 31.99% |
        
        **结论**：分散风险+互补，1+1>2
        """)
        
        st.markdown("#### 3️⃣ 德甲专项优化")
        st.info("""
        **最优参数：**
        - 置信度阈值：50%（比默认低5%）
        - 安全边际：1.2
        - 凯利系数：0.5
        - **ROI：+13.57%**（提升+2.76%）
        """)
    
    with col_r2:
        st.markdown("#### 4️⃣ 平局联动效果惊人")
        st.success("""
        **核心发现：加入平局策略ROI提升4.73%！**
        
        | 方案 | 平均赛季ROI | 总出手/赛季 |
        |------|-----------|------------|
        | 只有磐石 | +17.91% | 32.0 |
        | **磐石+平局** | **+22.64%** 🏆 | 34.9 |
        
        **最优平局参数：**
        - 平局置信度阈值：50%
        - 平局凯利系数：0.2
        - 平局出手只有4.6次/赛季，不多但贡献很大
        """)
        
        st.markdown("#### 👑 超级组合策略（最终版）")
        st.balloons()
        st.markdown("""
        **最终配置：**
        - ✅ 联赛筛选：只投德甲 + 意甲
        - ✅ 主胜/客胜：磐石策略（置信≥55% + 边际1.2 + 凯利0.2）
        - ✅ 平局策略：高置信度平局（置信≥50% + 凯利0.2）
        - ✅ 出手限制：无最低出手，宁缺毋滥
        
        **最终成绩：**
        - 🏆 平均赛季ROI：**+22.64%**
        - 📉 最大回撤：**38.71%**
        - 🎯 出手/赛季：**34.9次**（其中平局4.6次）
        """)
        
        st.warning("""
        ⚠️ **风险提示：**
        - 平局出手较少，样本量有限，可能存在过拟合
        - 联赛筛选可能存在数据窥探偏差
        - 建议作为高级可选策略，谨慎参考
        """)

# ========== 策略说明 ==========
with st.expander("📖 策略规则说明"):
    st.markdown("""
    ### 基础规则
    - 每个赛季初始积分：5000
    - 赛季结束重置，不滚存
    - 每周至少出手2场（无符合条件时强制选置信度最高的）
    - 单注上限：5%（防止一把梭哈）
    
    ### 三AI策略参数（WF金标准验证）
    | AI | 风格定位 | 置信度门槛 | 凯利系数 | 价值安全边际 |
    |----|---------|-----------|---------|-------------|
    | 🔥 激进AI | 高频价值型 | ≥55% | 0.90 | 20% |
    | ⚖️ 中立AI | 均衡精选型 | ≥55% | 0.60 | 20% |
    | 🛡️ 保守AI | 极致稳型 | ≥55% | 0.20 | 20% |
    
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
    
    **1. 模型alpha有限，符合市场规律**
    - 最好的策略赛季ROI也只有+1.2%，几乎不赚
    - 夏普只有0.09，接近随机水平
    - 这才是真实的足球预测，没有圣杯
    
    **2. 安全边际非常重要**
    - 20%安全边际：勉强打平
    - 10%安全边际：亏20-30%
    - 无安全边际：亏30-60%
    - 价值筛选确实有效，能大幅减少亏损
    
    **3. 质量远大于数量**
    - 出手越少，亏得越少
    - 宁可少出手，也要选最有把握的
    - 高置信度不一定好，太低也不行
    
    **4. 没有圣杯，持续优化**
    - 模型是有效的，但很有限
    - 还有很大提升空间
    - 继续优化特征和策略
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
