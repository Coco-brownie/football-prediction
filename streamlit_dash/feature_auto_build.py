import pandas as pd
import numpy as np

# 完整20维特征，与 match_predict.py / twin_lgb_train.py 全局统一
FEATURE_COLS = [
    "h5_gf","h5_ga","h5_shot","h5_shot_ot",
    "h10_gf","h10_ga",
    "a5_gf","a5_ga","a5_shot","a5_shot_ot",
    "a10_gf","a10_ga",
    "odds_draw_real","odds_lose_real",
    "shot_on_diff",
    "league_SER","league_E0","league_D1","league_LIG","league_LLA"
]

# 联赛编码 → 5个独热位的映射（顺序严格对齐训练时的 LEAGUE_FIX_COLS）
LEAGUE_ONEHOT_MAP = {
    "SER": [1, 0, 0, 0, 0],
    "E0":  [0, 1, 0, 0, 0],
    "D1":  [0, 0, 1, 0, 0],
    "LIG": [0, 0, 0, 1, 0],
    "LLA": [0, 0, 0, 0, 1],
}

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

def build_feature_by_teams(df_full, home_team, away_team, draw_odds, away_odds, shot_diff, league_code,
                           use_value_features=False, league_cfg_code=None):
    """
    构建完整34维特征（开启身价特征后为39维）
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

    # 前25维基础特征（15基础 + 5交锋 + 5赔率衍生）

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
        shot_diff,
        h2h_cnt, h2h_win_rate, h2h_draw_rate, h2h_gf_avg, h2h_ga_avg,
        prob_ratio_ha, prob_draw_share, prob_max, prob_entropy, prob_home_favorite,
        home_draw_5, home_draw_10, away_draw_5, away_draw_10
    ]
    # 后5维联赛独热列
    league_feat = LEAGUE_ONEHOT_MAP.get(league_code, [0, 0, 0, 0, 0])

    result = base_feat + league_feat

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