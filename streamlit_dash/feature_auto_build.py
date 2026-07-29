import pandas as pd
import numpy as np
import sqlite3
import os

# 完整52维特征（去掉shot_on_diff泄露特征），与 match_predict.py 全局统一
FEATURE_COLS = [
    "h5_gf","h5_ga","h5_shot","h5_shot_ot",
    "h10_gf","h10_ga",
    "a5_gf","a5_ga","a5_shot","a5_shot_ot",
    "a10_gf","a10_ga",
    "odds_draw_real","odds_lose_real",
    "h2h_cnt", "h2h_home_win_rate", "h2h_draw_rate", "h2h_home_gf_avg", "h2h_home_ga_avg",
    "prob_ratio_ha", "prob_draw_share", "prob_max", "prob_entropy", "prob_home_favorite",
    "home_draw_rate_5", "home_draw_rate_10", "away_draw_rate_5", "away_draw_rate_10",
    "league_SER","league_E0","league_D1","league_LIG","league_LLA",
    "home_elo_before", "away_elo_before", "elo_diff_before",
    "h5_gf_elo_weighted", "h5_ga_elo_weighted",
    "a5_gf_elo_weighted", "a5_ga_elo_weighted",
    "home_w5_elo_trend", "home_w10_elo_trend",
    "away_w5_elo_trend", "away_w10_elo_trend",
]

# 联赛编码 → 5个独热位的映射（顺序严格对齐训练时的 LEAGUE_FIX_COLS）
LEAGUE_ONEHOT_MAP = {
    "SER": [1, 0, 0, 0, 0],
    "E0":  [0, 1, 0, 0, 0],
    "D1":  [0, 0, 1, 0, 0],
    "LIG": [0, 0, 0, 1, 0],
    "LLA": [0, 0, 0, 0, 1],
}

# ELO缓存
_elo_cache = None
_elo_history_cache = None  # 每支球队的ELO时序列表 [(date, elo), ...]

def _get_elo_cache():
    """懒加载ELO缓存：从match_elo表取每支球队最新ELO"""
    global _elo_cache, _elo_history_cache
    if _elo_cache is not None:
        return _elo_cache
    
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "football.db")
    if not os.path.exists(db_path):
        db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "football.db")
    
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("""
        SELECT league_code, home_team_std as team, home_elo_before as elo, match_date
        FROM match_elo
        UNION ALL
        SELECT league_code, away_team_std as team, away_elo_before as elo, match_date
        FROM match_elo
    """, conn)
    conn.close()
    
    df["match_date"] = pd.to_datetime(df["match_date"])
    df = df.sort_values("match_date")
    
    _elo_cache = {}
    _elo_history_cache = {}
    for league in df["league_code"].unique():
        league_df = df[df["league_code"] == league]
        latest = league_df.groupby("team").last()["elo"].to_dict()
        _elo_cache[league] = latest
        # 时序数据：每支球队按日期排序的ELO列表
        history = {}
        for team in league_df["team"].unique():
            team_df = league_df[league_df["team"] == team].sort_values("match_date")
            history[team] = list(zip(team_df["match_date"], team_df["elo"]))
        _elo_history_cache[league] = history
    
    return _elo_cache


def get_team_elo(team_std, league_code):
    """获取一支球队的当前ELO评分"""
    cache = _get_elo_cache()
    league_elos = cache.get(league_code, {})
    return league_elos.get(team_std, 1500.0)


def get_team_elo_weighted_stats(df_filter, team_name, league_code, recent_n=5):
    """计算球队近N场的ELO加权攻防统计
    核心逻辑：赢强队比赢弱队更有价值，按对手ELO加权
    返回：加权场均进球、加权场均失球
    """
    _get_elo_cache()  # 确保缓存加载
    elo_map = _elo_cache.get(league_code, {})
    
    home_col = "home_team_std" if "home_team_std" in df_filter.columns else "home_team"
    away_col = "away_team_std" if "away_team_std" in df_filter.columns else "away_team"

    home_rec = df_filter[df_filter[home_col] == team_name].copy()
    away_rec = df_filter[df_filter[away_col] == team_name].copy()
    all_rec = pd.concat([home_rec, away_rec]).sort_values("match_date", ascending=False).head(recent_n)

    if len(all_rec) == 0:
        return 0.0, 0.0

    weighted_gf = 0.0
    weighted_ga = 0.0
    total_weight = 0.0

    for _, row in all_rec.iterrows():
        # 确定对手
        if row[home_col] == team_name:
            opponent = row[away_col]
            gf = row["home_goals"]
            ga = row["away_goals"]
        else:
            opponent = row[home_col]
            gf = row["away_goals"]
            ga = row["home_goals"]

        # 对手ELO作为权重（用联赛均值归一化，避免数值过大）
        opp_elo = elo_map.get(opponent, 1500.0)
        weight = opp_elo / 1500.0  # 以1500为基准

        weighted_gf += gf * weight
        weighted_ga += ga * weight
        total_weight += weight

    if total_weight == 0:
        return 0.0, 0.0

    return weighted_gf / total_weight, weighted_ga / total_weight


def get_team_elo_trend(team_std, league_code, recent_n=5):
    """计算球队近N场的ELO变化趋势
    返回：趋势值（每场平均ELO变化，正=上升，负=下降）
    """
    _get_elo_cache()  # 确保缓存加载
    history = _elo_history_cache.get(league_code, {}).get(team_std, [])

    if len(history) < 2:
        return 0.0

    # 取最近N+1个点（N场比赛对应N+1个赛前ELO值）
    recent = history[-(recent_n + 1):] if len(history) > recent_n else history

    if len(recent) < 2:
        return 0.0

    # 简单线性趋势：(最新 - 最早) / 场次数
    first_elo = recent[0][1]
    last_elo = recent[-1][1]
    n_games = len(recent) - 1

    return (last_elo - first_elo) / n_games if n_games > 0 else 0.0

def get_team_recent_stats(df_filter, team_name, recent_n):
    # 自动兼容列名：优先 _std 后缀，回退原生列名
    home_col = "home_team_std" if "home_team_std" in df_filter.columns else "home_team"
    away_col = "away_team_std" if "away_team_std" in df_filter.columns else "away_team"

    home_rec = df_filter[df_filter[home_col] == team_name].copy()
    away_rec = df_filter[df_filter[away_col] == team_name].copy()
    all_rec = pd.concat([home_rec, away_rec]).sort_values("match_date", ascending=False).head(recent_n)

    gf = 0
    ga = 0
    shot_sum = 0
    shot_ot_sum = 0
    for _, row in all_rec.iterrows():
        if row[home_col] == team_name:
            # 主队：进球=home_goals，失球=away_goals
            gf += row["home_goals"]
            ga += row["away_goals"]
            # 射门、射正字段从数据库累加
            if "HS" in row and pd.notna(row["HS"]):
                shot_sum += row["HS"]
            if "HST" in row and pd.notna(row["HST"]):
                shot_ot_sum += row["HST"]
        else:
            # 客队：进球=away_goals，失球=home_goals
            gf += row["away_goals"]
            ga += row["home_goals"]
            if "AwayShot" in row and pd.notna(row["AwayShot"]):
                shot_sum += row["AwayShot"]
            if "AST" in row and pd.notna(row["AST"]):
                shot_ot_sum += row["AST"]
    n = len(all_rec)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    # 返回场均值（与训练数据match_feature_final口径一致）
    return gf / n, ga / n, shot_sum / n, shot_ot_sum / n

def calc_h2h_stats(df_filter, home_team, away_team):
    """计算两队历史交锋统计（近5场，主队视角）"""
    home_col = "home_team_std" if "home_team_std" in df_filter.columns else "home_team"
    away_col = "away_team_std" if "away_team_std" in df_filter.columns else "away_team"

    # 找两队历史交锋（主客场都算）
    past = df_filter[
        ((df_filter[home_col] == home_team) & (df_filter[away_col] == away_team)) |
        ((df_filter[home_col] == away_team) & (df_filter[away_col] == home_team))
    ].sort_values("match_date", ascending=False).head(5)

    if len(past) == 0:
        return 0, 0.0, 0.0, 0.0, 0.0

    home_wins = 0
    draws = 0
    home_gf_total = 0
    home_ga_total = 0

    for _, m in past.iterrows():
        if m[home_col] == home_team:
            hg = m["home_goals"]
            ag = m["away_goals"]
        else:
            hg = m["away_goals"]
            ag = m["home_goals"]

        home_gf_total += hg
        home_ga_total += ag
        if hg > ag:
            home_wins += 1
        elif hg == ag:
            draws += 1

    cnt = len(past)
    return (
        cnt,
        round(home_wins / cnt, 4),
        round(draws / cnt, 4),
        round(home_gf_total / cnt, 4),
        round(home_ga_total / cnt, 4)
    )

def calc_decay_weight(n_games_ago, half_life=10):
    """计算指数衰减权重"""
    import numpy as np
    k = np.log(2) / half_life
    return np.exp(-k * n_games_ago)


def get_team_time_decay_stats(df_full, team, n=5):
    """计算球队近n场的时间衰减加权进球/失球"""
    home_col = "home_team_std" if "home_team_std" in df_full.columns else "home_team"
    away_col = "away_team_std" if "away_team_std" in df_full.columns else "away_team"
    
    # 找到这支球队的所有比赛，按日期倒序
    team_matches = df_full[
        (df_full[home_col] == team) | (df_full[away_col] == team)
    ].sort_values("match_date", ascending=False)
    
    if len(team_matches) == 0:
        return 1.5, 1.5  # 默认值
    
    # 取最近n场
    recent = team_matches.head(n)
    
    # 计算每一场的进球和失球
    goals_for = []
    goals_against = []
    for _, row in recent.iterrows():
        if row[home_col] == team:
            # 主队
            goals_for.append(row["home_goals"])
            goals_against.append(row["away_goals"])
        else:
            # 客队
            goals_for.append(row["away_goals"])
            goals_against.append(row["home_goals"])
    
    # 计算时间衰减权重（越近权重越大）
    m = len(goals_for)
    weights = [calc_decay_weight(m - 1 - i) for i in range(m)]
    total_w = sum(weights)
    
    if total_w == 0:
        return 1.5, 1.5
    
    avg_gf = sum(g * w for g, w in zip(goals_for, weights)) / total_w
    avg_ga = sum(g * w for g, w in zip(goals_against, weights)) / total_w
    
    return avg_gf, avg_ga


def build_feature_by_teams(df_full, home_team, away_team, draw_odds, away_odds, league_code,
                           use_value_features=False, league_cfg_code=None):
    """
    构建完整52维特征（去掉shot_on_diff泄露特征）
    league_code: 数据库联赛编码（SER/E0/D1/LIG/LLA）
    use_value_features: 是否追加身价特征（默认关闭，需补充team_value_data.json后开启）
    league_cfg_code: 联赛配置编码（EPL/LLA/BUN/SER/LIG），身价特征需要
    """
    h5_gf, h5_ga, h5_shot, h5_shot_ot = get_team_recent_stats(df_full, home_team, 5)
    h10_gf, h10_ga, _, _ = get_team_recent_stats(df_full, home_team, 10)
    a5_gf, a5_ga, a5_shot, a5_shot_ot = get_team_recent_stats(df_full, away_team, 5)
    a10_gf, a10_ga, _, _ = get_team_recent_stats(df_full, away_team, 10)

    # 计算历史交锋特征
    h2h_cnt, h2h_win_rate, h2h_draw_rate, h2h_gf_avg, h2h_ga_avg = calc_h2h_stats(df_full, home_team, away_team)

    # 前24维基础特征（12基础 + 5交锋 + 5赔率衍生 + 2赔率）

    # 计算赔率衍生特征（基于去水后的概率）
    p_home = 1.0 - draw_odds - away_odds
    p_draw = draw_odds
    p_away = away_odds

    prob_ratio_ha = p_home / (p_away + 1e-8)
    prob_draw_share = p_draw / (p_home + p_draw + p_away + 1e-8)
    prob_max = max(p_home, p_draw, p_away)
    # 概率熵
    prob_entropy = 0.0
    for p in [p_home, p_draw, p_away]:
        if p > 0.001:
            prob_entropy -= p * np.log2(p)
    prob_home_favorite = 1 if p_home > p_away else 0

    # 计算球队近期平局率
    def calc_draw_rate(team, n):
        home_col = "home_team_std" if "home_team_std" in df_full.columns else "home_team"
        away_col = "away_team_std" if "away_team_std" in df_full.columns else "away_team"
        team_matches = df_full[
            (df_full[home_col] == team) | (df_full[away_col] == team)
        ].sort_values("match_date", ascending=False).head(n)
        if len(team_matches) == 0:
            return 0.25  # 默认值
        draw_count = (team_matches["match_result"] == "平局").sum()
        return draw_count / len(team_matches)

    home_draw_5 = calc_draw_rate(home_team, 5)
    home_draw_10 = calc_draw_rate(home_team, 10)
    away_draw_5 = calc_draw_rate(away_team, 5)
    away_draw_10 = calc_draw_rate(away_team, 10)

    base_feat = [
        h5_gf, h5_ga, h5_shot, h5_shot_ot,
        h10_gf, h10_ga,
        a5_gf, a5_ga, a5_shot, a5_shot_ot,
        a10_gf, a10_ga,
        draw_odds, away_odds,
        h2h_cnt, h2h_win_rate, h2h_draw_rate, h2h_gf_avg, h2h_ga_avg,
        prob_ratio_ha, prob_draw_share, prob_max, prob_entropy, prob_home_favorite,
        home_draw_5, home_draw_10, away_draw_5, away_draw_10
    ]
    # 后5维联赛独热列
    league_feat = LEAGUE_ONEHOT_MAP.get(league_code, [0, 0, 0, 0, 0])

    result = base_feat + league_feat
    
    # ELO特征（3维直接特征）
    home_elo = get_team_elo(home_team, league_code)
    away_elo = get_team_elo(away_team, league_code)
    elo_diff = home_elo - away_elo
    result += [home_elo, away_elo, elo_diff]

    # ELO扩展特征（8维：4个对手加权 + 4个趋势）
    h5_gf_w, h5_ga_w = get_team_elo_weighted_stats(df_full, home_team, league_code, 5)
    a5_gf_w, a5_ga_w = get_team_elo_weighted_stats(df_full, away_team, league_code, 5)
    home_w5_trend = get_team_elo_trend(home_team, league_code, 5)
    home_w10_trend = get_team_elo_trend(home_team, league_code, 10)
    away_w5_trend = get_team_elo_trend(away_team, league_code, 5)
    away_w10_trend = get_team_elo_trend(away_team, league_code, 10)
    result += [h5_gf_w, h5_ga_w, a5_gf_w, a5_ga_w,
               home_w5_trend, home_w10_trend, away_w5_trend, away_w10_trend]
    
    # 时间衰减特征（8维：近5场+近10场，主队+客队，进球+失球）
    h5_gf_td, h5_ga_td = get_team_time_decay_stats(df_full, home_team, 5)
    h10_gf_td, h10_ga_td = get_team_time_decay_stats(df_full, home_team, 10)
    a5_gf_td, a5_ga_td = get_team_time_decay_stats(df_full, away_team, 5)
    a10_gf_td, a10_ga_td = get_team_time_decay_stats(df_full, away_team, 10)
    result += [h5_gf_td, h5_ga_td, a5_gf_td, a5_ga_td,
               h10_gf_td, h10_ga_td, a10_gf_td, a10_ga_td]

    # 可选：追加身价特征（5维）
    if use_value_features and league_cfg_code:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from team_value_feature import calc_value_features
        _, _, value_diff, value_ratio, log_value_ratio = calc_value_features(
            league_cfg_code, home_team, away_team
        )
        result += [value_diff, value_ratio, log_value_ratio]

    return result