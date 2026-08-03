import pandas as pd
import numpy as np
import sqlite3
import os
import sys

# 把项目根目录加到path，导入公共配置
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CUR_DIR)
sys.path.insert(0, ROOT_DIR)

from common_config import get_feature_list, get_league_onehot, get_all_db_codes, db_to_onehot, get_path
from team_mapping_v2 import get_team_cn_name_v2

# 【2026-08-08 中英双向匹配辅助：match_feature_final 的球队列（home_team_std 等）为历史构建遗留的
#  中英混杂数据（约80%中文、20%英文）。所有球队匹配统一用本函数解析出【中/英两种写法】，
#  用 isin 匹配，确保无论表内存中文还是英文都能命中。根治方向是重建数据统一英文名
#  （待 feature_engineering 增加 get_standard_team 归一化后再做，代码层先兜底）。】
def _team_keys(team, league_code=None):
    keys = {team} if team else set()
    if league_code and team:
        try:
            cn = get_team_cn_name_v2(str(league_code), str(team), print_miss=False)
            if cn and cn != team:
                keys.add(cn)
        except Exception:
            pass
    return keys

# 【2026-08-03 统一出口：从common_config读取特征列表，保证全项目一致】
# 完整52维特征，与训练/推理端完全对齐
FEATURE_COLS = get_feature_list()

# 联赛编码 → 5个独热位的映射（顺序严格对齐 config.json features.league_onehot）
# 【2026-08-07 修复：旧硬编码用 SER/LIG/LLA 作键，但前端传入的是真实数据库编码
#  (E0/D1/SP1/I1/F1，经 cfg_to_db_league)，导致西甲(SP1)/意甲(I1)/法甲(F1)独热全0——
#  与训练端 common_config.db_to_onehot 失配。现统一从 common_config 注册表派生，杜绝再漂移。】
_ONEHOT_ORDER = get_league_onehot()

def _build_league_onehot_map():
    m = {}
    for _db_code in get_all_db_codes():
        _col = db_to_onehot(_db_code)
        if _col in _ONEHOT_ORDER:
            _vec = [0] * len(_ONEHOT_ORDER)
            _vec[_ONEHOT_ORDER.index(_col)] = 1
            m[_db_code] = _vec
    return m

LEAGUE_ONEHOT_MAP = _build_league_onehot_map()

# ELO缓存
_elo_cache = None
_elo_history_cache = None  # 每支球队的ELO时序列表 [(date, elo), ...]

def _load_elo_frame(conn):
    """加载ELO时序数据：优先独立表 match_elo，回退训练表 match_feature_final（必然含 ELO 列）。
    返回两列：elo_before(赛前) / elo_after(赛后)。
    【2026-08-08 口径修复（两套口径并存）】
      - get_team_elo（直接ELO特征） 取 elo_after 最新："某队某日前最近一场赛后ELO" = 该队本场开赛前ELO
      - get_team_elo_trend（趋势特征） 用 elo_before 时序：与训练端 (最近一场赛前 - N场前赛前)/N 一致
      （若趋势也误用赛后序列，会把本场赛前多算进窗口 → ~1.8 分漂移）
    """
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)["name"].tolist()
    if "match_elo" in tables:
        _cols = pd.read_sql("SELECT * FROM match_elo LIMIT 1", conn).columns.tolist()
        if "home_elo_after" in _cols:
            return pd.read_sql("""
                SELECT league_code, home_team_std as team, match_date,
                       home_elo_before as elo_before, home_elo_after as elo_after
                FROM match_elo
                UNION ALL
                SELECT league_code, away_team_std as team, match_date,
                       away_elo_before as elo_before, away_elo_after as elo_after
                FROM match_elo
            """, conn)
        # 旧表无赛后列：after 回退赛前（历史兼容）
        return pd.read_sql("""
            SELECT league_code, home_team_std as team, match_date,
                   home_elo_before as elo_before, home_elo_before as elo_after
            FROM match_elo
            UNION ALL
            SELECT league_code, away_team_std as team, match_date,
                   away_elo_before as elo_before, away_elo_before as elo_after
            FROM match_elo
        """, conn)
    return pd.read_sql("""
        SELECT league_code, home_team_std as team, match_date,
               home_elo_before as elo_before, home_elo_before as elo_after
        FROM match_feature_final
        WHERE home_elo_before IS NOT NULL
        UNION ALL
        SELECT league_code, away_team_std as team, match_date,
               away_elo_before as elo_before, away_elo_before as elo_after
        FROM match_feature_final
        WHERE away_elo_before IS NOT NULL
    """, conn)


def _get_elo_cache():
    """懒加载ELO缓存：从 match_elo（回退 match_feature_final）取每支球队最新ELO"""
    global _elo_cache, _elo_history_cache
    if _elo_cache is not None:
        return _elo_cache
    
    # 【2026-08-07 修复：统一读根库（与 predict_module.DB_PATH 一致），消除双库不同步风险】
    db_path = get_path("db_path")
    
    conn = sqlite3.connect(db_path)
    df = _load_elo_frame(conn)
    conn.close()
    
    df["match_date"] = pd.to_datetime(df["match_date"])
    df = df.sort_values("match_date")
    
    _elo_cache = {}
    _elo_history_cache = {}
    for league in df["league_code"].unique():
        league_df = df[df["league_code"] == league]
        # 【2026-08-08】直接ELO特征：取该队最新【赛后】ELO（= 该队本场开赛前 ELO，训练端口径）
        latest = league_df.groupby("team").last()["elo_after"].to_dict()
        _elo_cache[league] = latest
        # 【2026-08-08】趋势特征：用【赛前】ELO 时序（与训练端 (最近赛前-N场前赛前)/N 一致，
        #  绝不能用赛后序列——那会把本场赛前多算进窗口，造成 ~1.8 分漂移）
        history = {}
        for team in league_df["team"].unique():
            team_df = league_df[league_df["team"] == team].sort_values("match_date")
            history[team] = list(zip(team_df["match_date"], team_df["elo_before"]))
        _elo_history_cache[league] = history
    
    return _elo_cache


def get_team_elo(team_std, league_code):
    """获取一支球队的当前ELO评分
    【2026-08-03 口径统一：缺省值改为0，与训练端fillna(0)一致】
    【2026-08-08 中英双向：ELO缓存key与match_feature_final一致（中英混杂），查询双向兑底】
    """
    cache = _get_elo_cache()
    league_elos = cache.get(league_code, {})
    if team_std in league_elos:
        return league_elos[team_std]
    for _k in _team_keys(team_std, league_code):
        if _k in league_elos:
            return league_elos[_k]
    return 0.0  # 缺省0，与训练端一致


# ===================== 身价特征缓存 =====================
_value_cache = None
_league_avg_value = None

def _get_value_cache():
    """加载球队最新身价缓存（从team_value_features表）"""
    global _value_cache, _league_avg_value
    if _value_cache is not None:
        return _value_cache, _league_avg_value
    
    # 【2026-08-07 修复：统一读根库（team_value_features 表在根库），消除双库不同步风险】
    db_path = get_path("db_path")
    
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("""
        SELECT league_code, home_team_std as team, home_value as value, match_date
        FROM team_value_features
        WHERE home_value IS NOT NULL
        UNION ALL
        SELECT league_code, away_team_std as team, away_value as value, match_date
        FROM team_value_features
        WHERE away_value IS NOT NULL
    """, conn)
    conn.close()
    
    df["match_date"] = pd.to_datetime(df["match_date"])
    df = df.sort_values("match_date")
    
    _value_cache = {}
    _league_avg_value = {}
    
    for league in df["league_code"].unique():
        league_df = df[df["league_code"] == league]
        # 每支球队取最新的身价
        latest = league_df.groupby("team").last()["value"].to_dict()
        _value_cache[league] = latest
        # 联赛平均身价
        if latest:
            _league_avg_value[league] = sum(latest.values()) / len(latest)
        else:
            _league_avg_value[league] = 10000000.0  # 默认1000万
    
    return _value_cache, _league_avg_value


def get_team_value(team_std, league_code):
    """获取一支球队的最新身价
    【2026-08-08 中英双向：身价缓存key与球队列一致（中英混杂），查询双向兑底】
    """
    cache, _ = _get_value_cache()
    league_values = cache.get(league_code, {})
    if team_std in league_values:
        return league_values[team_std]
    for _k in _team_keys(team_std, league_code):
        if _k in league_values:
            return league_values[_k]
    return 10000000.0  # 默认1000万


def get_team_elo_weighted_stats(df_filter, team_name, league_code, recent_n=5):
    """计算球队近N场的ELO加权攻防统计
    核心逻辑：赢强队比赢弱队更有价值，按对手ELO加权
    返回：加权场均进球、加权场均失球
    """
    _get_elo_cache()  # 确保缓存加载
    elo_map = _elo_cache.get(league_code, {})
    
    home_col = "home_team_std" if "home_team_std" in df_filter.columns else "home_team"
    away_col = "away_team_std" if "away_team_std" in df_filter.columns else "away_team"

    # 【2026-08-08 中英双向匹配：match_feature_final 球队列中英混杂】
    _keys = _team_keys(team_name, league_code)
    home_rec = df_filter[df_filter[home_col].isin(_keys)].copy()
    away_rec = df_filter[df_filter[away_col].isin(_keys)].copy()
    all_rec = pd.concat([home_rec, away_rec]).sort_values("match_date", ascending=False).head(recent_n)

    if len(all_rec) == 0:
        return 0.0, 0.0

    weighted_gf = 0.0
    weighted_ga = 0.0
    total_weight = 0.0

    for _, row in all_rec.iterrows():
        # 确定对手（row[home_col] 可能是中文或英文，统一用 _keys 判定）
        if row[home_col] in _keys:
            opponent = row[away_col]
            gf = row["home_goals"]
            ga = row["away_goals"]
        else:
            opponent = row[home_col]
            gf = row["away_goals"]
            ga = row["home_goals"]

        # 【2026-08-07 口径统一：改用该场对手的赛前ELO（row级 home/away_elo_before），
        #  与训练端 build_elo_extended_features.calc_weighted_stats 完全一致；
        #  该场ELO缺失时兜底用最新ELO，再兜底1500】
        if row[home_col] in _keys:
            opp_elo = row.get("away_elo_before", np.nan)
        else:
            opp_elo = row.get("home_elo_before", np.nan)
        if opp_elo is None or (isinstance(opp_elo, float) and np.isnan(opp_elo)):
            opp_elo = elo_map.get(opponent, 1500.0)
        weight = float(opp_elo) / 1500.0  # 以1500为基准

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
    _hist_map = _elo_history_cache.get(league_code, {})
    history = _hist_map.get(team_std, [])
    if not history:
        for _k in _team_keys(team_std, league_code):
            if _k in _hist_map:
                history = _hist_map[_k]
                break

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

def get_team_recent_stats(df_filter, team_name, recent_n, venue=None, league_code=None):
    """
    获取球队近期场均数据
    venue: None=主客混合（旧行为）, "home"=仅主场, "away"=仅客场
    【2026-08-03 口径统一：与训练端一致，按主客场分别统计】
    【2026-08-08 中英双向：df_filter 球队列中英混杂，用 isin([中文,英文]) 匹配】
    """
    # 自动兼容列名：优先 _std 后缀，回退原生列名
    home_col = "home_team_std" if "home_team_std" in df_filter.columns else "home_team"
    away_col = "away_team_std" if "away_team_std" in df_filter.columns else "away_team"

    _keys = _team_keys(team_name, league_code)
    if venue == "home":
        # 仅取主场比赛
        rec = df_filter[df_filter[home_col].isin(_keys)].copy()
    elif venue == "away":
        # 仅取客场比赛
        rec = df_filter[df_filter[away_col].isin(_keys)].copy()
    else:
        # 主客混合（旧行为）
        home_rec = df_filter[df_filter[home_col].isin(_keys)].copy()
        away_rec = df_filter[df_filter[away_col].isin(_keys)].copy()
        rec = pd.concat([home_rec, away_rec])
    
    rec = rec.sort_values("match_date", ascending=False).head(recent_n)
    
    gf = 0
    ga = 0
    shot_sum = 0
    shot_ot_sum = 0
    for _, row in rec.iterrows():
        if row[home_col] in _keys:
            # 主队：进球=home_goals，失球=away_goals
            gf += row["home_goals"]
            ga += row["away_goals"]
            # 射门、射正字段从数据库累加（用归一化后的列，与训练端一致）
            if "home_shot" in row and pd.notna(row["home_shot"]):
                shot_sum += row["home_shot"]
            if "home_shot_on_target" in row and pd.notna(row["home_shot_on_target"]):
                shot_ot_sum += row["home_shot_on_target"]
        else:
            # 客队：进球=away_goals，失球=home_goals
            gf += row["away_goals"]
            ga += row["home_goals"]
            if "away_shot" in row and pd.notna(row["away_shot"]):
                shot_sum += row["away_shot"]
            if "away_shot_on_target" in row and pd.notna(row["away_shot_on_target"]):
                shot_ot_sum += row["away_shot_on_target"]
    n = len(rec)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    # 返回场均值（与训练数据match_feature_final口径一致）
    return gf / n, ga / n, shot_sum / n, shot_ot_sum / n

def calc_h2h_stats(df_filter, home_team, away_team, league_code=None):
    """计算两队历史交锋统计（近5场，主队视角）
    【2026-08-08 中英双向：df_filter 球队列中英混杂，用 isin([中文,英文]) 匹配】"""
    home_col = "home_team_std" if "home_team_std" in df_filter.columns else "home_team"
    away_col = "away_team_std" if "away_team_std" in df_filter.columns else "away_team"

    h_keys = _team_keys(home_team, league_code)
    a_keys = _team_keys(away_team, league_code)

    # 找两队历史交锋（主客场都算）
    past = df_filter[
        ((df_filter[home_col].isin(h_keys)) & (df_filter[away_col].isin(a_keys))) |
        ((df_filter[home_col].isin(a_keys)) & (df_filter[away_col].isin(h_keys)))
    ].sort_values("match_date", ascending=False).head(5)

    if len(past) == 0:
        return 0, 0.0, 0.0, 0.0, 0.0

    home_wins = 0
    draws = 0
    home_gf_total = 0
    home_ga_total = 0

    for _, m in past.iterrows():
        if m[home_col] in h_keys:
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


def get_team_time_decay_stats(df_full, team, n=5, league_code=None):
    """计算球队近n场的时间衰减加权进球/失球
    【2026-08-08 中英双向：df_full 球队列中英混杂，用 isin([中文,英文]) 匹配】"""
    home_col = "home_team_std" if "home_team_std" in df_full.columns else "home_team"
    away_col = "away_team_std" if "away_team_std" in df_full.columns else "away_team"

    _keys = _team_keys(team, league_code)
    # 找到这支球队的所有比赛，按日期倒序
    team_matches = df_full[
        (df_full[home_col].isin(_keys)) | (df_full[away_col].isin(_keys))
    ].sort_values("match_date", ascending=False)
    
    if len(team_matches) == 0:
        return 1.5, 1.5  # 默认值
    
    # 取最近n场
    recent = team_matches.head(n)
    
    # 计算每一场的进球和失球
    goals_for = []
    goals_against = []
    for _, row in recent.iterrows():
        if row[home_col] in _keys:
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


def build_feature_by_teams(df_full, home_team, away_team, home_odds, draw_odds, away_odds, league_code,
                           use_value_features=False, league_cfg_code=None):
    """
    构建完整52维特征（去掉shot_on_diff泄露特征）
    home_odds/draw_odds/away_odds: 原始十进制赔率（如 2.5 / 3.3 / 3.0）
    【2026-08-07 口径修复：与训练端 match_feature_final 完全对齐——
     特征13/14 odds_draw_real / odds_lose_real 存的是【原始赔率】(范围2~5，
     repair_feature_table.py 从 match_feature_full.odds_draw/odds_lose 原样填入)；
     概率衍生特征19-23 用【去水概率】计算（add_odds_features.py 用 1/原始赔率 归一化）。
     旧实现在这里把【去水概率】直接当特征13/14、并误作基础概率推导，
     与训练数据(原始赔率)系统性失配 → 在线回测准确率仅42% vs 训练52%。
     现在：13/14 放原始赔率，衍生特征内部去水，双向对齐。】
    league_code: 数据库联赛编码（E0/D1/SP1/I1/F1）
    use_value_features: 是否追加身价特征（默认关闭，需补充team_value_data.json后开启）
    league_cfg_code: 联赛配置编码（EPL/LLA/BUN/SER/LIG），身价特征需要
    """
    # 【2026-08-03 口径统一：与训练端一致，主队用主场数据，客队用客场数据】
    # 【2026-08-08 中英双向：传入 league_code 供 _team_keys 解析中/英两种写法】
    h5_gf, h5_ga, h5_shot, h5_shot_ot = get_team_recent_stats(df_full, home_team, 5, venue="home", league_code=league_code)
    h10_gf, h10_ga, _, _ = get_team_recent_stats(df_full, home_team, 10, venue="home", league_code=league_code)
    a5_gf, a5_ga, a5_shot, a5_shot_ot = get_team_recent_stats(df_full, away_team, 5, venue="away", league_code=league_code)
    a10_gf, a10_ga, _, _ = get_team_recent_stats(df_full, away_team, 10, venue="away", league_code=league_code)

    # 计算历史交锋特征
    h2h_cnt, h2h_win_rate, h2h_draw_rate, h2h_gf_avg, h2h_ga_avg = calc_h2h_stats(df_full, home_team, away_team, league_code)

    # 前24维基础特征（12基础 + 5交锋 + 5赔率衍生 + 2赔率）

    # 计算赔率衍生特征（基于去水后的概率；与训练端 add_odds_features.py 同公式）
    # 输入是原始十进制赔率 → 先取倒数再归一化去水
    _inv_h = 1.0 / home_odds
    _inv_d = 1.0 / draw_odds
    _inv_a = 1.0 / away_odds
    _inv_sum = _inv_h + _inv_d + _inv_a
    p_home = _inv_h / _inv_sum
    p_draw = _inv_d / _inv_sum
    p_away = _inv_a / _inv_sum

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
        _keys = _team_keys(team, league_code)
        team_matches = df_full[
            (df_full[home_col].isin(_keys)) | (df_full[away_col].isin(_keys))
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
    # 联赛独热（5维，放在最后，与common_config.get_feature_list()顺序严格一致）
    league_feat = LEAGUE_ONEHOT_MAP.get(league_code, [0, 0, 0, 0, 0])

    result = base_feat.copy()
    
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
    h5_gf_td, h5_ga_td = get_team_time_decay_stats(df_full, home_team, 5, league_code=league_code)
    h10_gf_td, h10_ga_td = get_team_time_decay_stats(df_full, home_team, 10, league_code=league_code)
    a5_gf_td, a5_ga_td = get_team_time_decay_stats(df_full, away_team, 5, league_code=league_code)
    a10_gf_td, a10_ga_td = get_team_time_decay_stats(df_full, away_team, 10, league_code=league_code)
    result += [h5_gf_td, h5_ga_td, a5_gf_td, a5_ga_td,
               h10_gf_td, h10_ga_td, a10_gf_td, a10_ga_td]
    
    # 联赛独热（放在最后，与common_config顺序一致）
    result += league_feat

    # 【2026-08-03 身价特征：完整5维，与训练端完全对齐】
    # 顺序：value_ratio, value_diff_rel, home_value_rel, away_value_rel, value_diff_signed
    if use_value_features and league_code:
        value_cache, league_avg_cache = _get_value_cache()
        league_values = value_cache.get(league_code, {})
        league_avg = league_avg_cache.get(league_code, 10000000.0)
        
        home_val = get_team_value(home_team, league_code)
        away_val = get_team_value(away_team, league_code)
        
        # 计算5个身价特征
        value_ratio = home_val / (away_val + 1e-8)
        value_diff_abs = abs(home_val - away_val)
        value_diff_rel = value_diff_abs / (league_avg + 1e-8)
        home_value_rel = home_val / (league_avg + 1e-8)
        away_value_rel = away_val / (league_avg + 1e-8)
        value_diff_signed = (home_val - away_val) / (league_avg + 1e-8)
        
        result += [value_ratio, value_diff_rel, home_value_rel, away_value_rel, value_diff_signed]

    return result