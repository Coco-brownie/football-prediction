"""
公共样式模块
所有页面共用的样式和格式化函数
"""
import pandas as pd
import streamlit as st


def apply_global_style():
    """应用全局美化样式"""
    st.markdown("""
    <style>
    /* 主标题 */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
    }

    /* 二级标题 */
    h2 {
        border-left: 4px solid #667eea;
        padding-left: 12px;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }

    /* 三级标题 */
    h3 {
        color: #374151;
        font-weight: 600;
    }

    /* metric卡片 */
    [data-testid="stMetric"] {
        background: #f8fafc;
        border-radius: 12px;
        padding: 12px 16px;
        border: 1px solid #e2e8f0;
        transition: all 0.2s ease;
    }
    [data-testid="stMetric"]:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transform: translateY(-1px);
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem;
        color: #64748b;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1e293b;
    }

    /* Tab样式 */
    [data-testid="stTabs"] [role="tab"] {
        font-weight: 500;
    }

    /* 信息框 */
    [data-testid="stInfo"] {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
    }

    /* 警告框 */
    [data-testid="stWarning"] {
        background: #fffbeb;
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
    }

    /* 成功框 */
    [data-testid="stSuccess"] {
        background: #f0fdf4;
        border-left: 4px solid #22c55e;
        border-radius: 8px;
    }

    /* 分隔线 */
    hr {
        border: none;
        border-top: 1px solid #e2e8f0;
        margin: 1.5rem 0;
    }

    /* 按钮 */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }

    /* 数据表格 */
    [data-testid="stDataFrame"] {
        border-radius: 8px;
        overflow: hidden;
    }

    /* ========== 移动端适配 ========== */
    @media (max-width: 768px) {
        h1 { font-size: 1.5rem; }
        h2 { font-size: 1.2rem; padding-left: 8px; border-left-width: 3px; }
        h3 { font-size: 1rem; }

        [data-testid="stMetric"] {
            padding: 8px 10px;
            border-radius: 8px;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.15rem;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.75rem;
        }

        .stButton > button {
            width: 100%;
            padding: 0.5rem;
        }

        [data-testid="stTabs"] [role="tab"] {
            font-size: 0.8rem;
            padding: 0.5rem 0.3rem;
        }
    }

    @media (max-width: 480px) {
        h1 { font-size: 1.25rem; }
        [data-testid="stMetricValue"] {
            font-size: 1rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)


def style_match_result_df(df):
    """赛果数据表格样式"""
    def color_result(val):
        if val == "主队胜":
            return "color: #2e7d32; font-weight: 600"
        elif val == "客胜":
            return "color: #c62828; font-weight: 600"
        elif val == "平局":
            return "color: #f57c00; font-weight: 600"
        return ""

    return df.style.map(color_result, subset=["赛果"] if "赛果" in df.columns else [])


def confidence_level(conf):
    """置信度分级
    返回 (等级标签, 颜色)
    """
    if conf >= 0.65:
        return "高置信", "#e53935"
    elif conf >= 0.50:
        return "中等置信", "#f9a825"
    else:
        return "低置信", "#2e7d32"


def render_confidence_badge(confidence):
    """渲染置信度徽章HTML"""
    level, color = confidence_level(confidence)
    icon_map = {"高置信": "🔴", "中等置信": "🟡", "低置信": "🟢"}
    return f"<span style='color:{color};font-weight:600;'>{icon_map.get(level,'')} {level}（{confidence:.1%}）</span>"


def get_result_color(result):
    """赛果对应的颜色"""
    color_map = {
        "主胜": "#4caf50",
        "主队胜": "#4caf50",
        "平局": "#ffb300",
        "客胜": "#ef5350",
        "客队胜": "#ef5350",
    }
    return color_map.get(result, "#999")
