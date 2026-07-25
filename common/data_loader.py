"""
公共数据加载模块
所有页面共用的数据加载函数
"""
import os
import sys
import sqlite3
import pandas as pd
import streamlit as st

# 项目根目录
SCRIPT_PATH = os.path.abspath(__file__)
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_PATH))
DB_PATH = os.path.join(ROOT_DIR, "football.db")

sys.path.insert(0, ROOT_DIR)
from team_mapping_v2 import LEAGUE_CFG, CFG_2_DB_CODE, LEAGUE_TEAM_MAP

# 数据库编码 -> 配置编码
DB_2_CFG = {v: k for k, v in CFG_2_DB_CODE.items()}


@st.cache_data(ttl=3600)
def load_match_feature_data(db_path=DB_PATH):
    """加载历史比赛特征数据（全量，用于预测和分析）"""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM match_feature_final", conn)
    conn.close()
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    return df


@st.cache_data(ttl=86400)
def load_schedule_data(db_path=DB_PATH):
    """加载赛程数据，自动转换中文队名"""
    conn = sqlite3.connect(db_path)
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
        return eng_name

    df["home_team_cn"] = df.apply(lambda r: get_cn_name(r["league_code"], r["home_team"]), axis=1)
    df["away_team_cn"] = df.apply(lambda r: get_cn_name(r["league_code"], r["away_team"]), axis=1)
    return df


@st.cache_data(ttl=3600)
def verify_predictions(db_path=DB_PATH):
    """自动回填已完赛预测的实际结果"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 查找未验证的预测
    cursor.execute("""
        SELECT id, match_date, home_team, away_team, league_code, predict_result
        FROM predictions 
        WHERE is_verified = 0 OR is_verified IS NULL
    """)
    pending = cursor.fetchall()
    
    updated = 0
    for pred_id, match_date, home_team, away_team, league_code, pred_result in pending:
        # 从match_result表查找实际结果
        cursor.execute("""
            SELECT home_goals, away_goals, match_result
            FROM match_result
            WHERE home_team = ? AND away_team = ? 
              AND league_code = ? AND match_date = ?
        """, (home_team, away_team, league_code, match_date))
        
        res = cursor.fetchone()
        if res:
            hg, ag, actual = res
            is_correct = 1 if actual == pred_result else 0
            
            cursor.execute("""
                UPDATE predictions SET
                    actual_result = ?,
                    actual_home_goals = ?,
                    actual_away_goals = ?,
                    is_correct = ?,
                    is_verified = 1
                WHERE id = ?
            """, (actual, hg, ag, is_correct, pred_id))
            updated += 1
    
    conn.commit()
    conn.close()
    return updated


def load_predictions(db_path=DB_PATH):
    """加载预测记录（自动回填已完赛结果）"""
    # 先自动验证
    try:
        verify_predictions(db_path)
    except:
        pass
    
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql("SELECT * FROM predictions ORDER BY predict_time DESC", conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df


def get_league_list():
    """获取联赛列表（配置编码 -> 中文名）"""
    return {k: v["name"] for k, v in LEAGUE_CFG.items()}


def cfg_to_db_league(cfg_code):
    """配置编码转数据库编码"""
    return CFG_2_DB_CODE.get(cfg_code, cfg_code)
