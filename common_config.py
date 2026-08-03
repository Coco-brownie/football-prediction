"""
统一配置加载工具
所有脚本通过此文件读取配置，杜绝硬编码
"""
import os
import json

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ROOT_DIR, "config.json")

_config_cache = None

def load_config():
    """加载配置文件（带缓存）
    【2026-08-09 加固：config.json 缺失时给出明确中文指引，
      避免部署/克隆环境缺文件时抛裸 FileNotFoundError 难以排查】"""
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(
            f"【配置缺失】未找到配置文件：{CONFIG_PATH}\n"
            f"处理指引：请从仓库获取 config.json（项目根目录）后再启动。"
        )
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        _config_cache = json.load(f)
    return _config_cache

def get_path(key):
    """获取路径配置，自动转为绝对路径"""
    cfg = load_config()
    rel_path = cfg["paths"][key]
    return os.path.join(ROOT_DIR, rel_path)

def get_league_db_code(cfg_key):
    """联赛配置键 → 数据库编码（统一从 LEAGUE_REGISTRY 取，不再读 config.json 以免两处漂移）"""
    info = LEAGUE_REGISTRY.get(cfg_key)
    return info["db_code"] if info else None

def get_all_league_codes():
    """获取所有联赛数据库编码列表（统一从 LEAGUE_REGISTRY 取）"""
    return get_all_db_codes()

def get_feature_list():
    """获取完整特征列表（基础 + 交锋 + 概率衍生 + 平局率 + ELO + 时间衰减 + 联赛独热）"""
    cfg = load_config()
    feat = cfg["features"]
    return (
        feat["base_features"] + 
        feat["h2h_features"] + 
        feat["prob_features"] + 
        feat["draw_features"] + 
        feat["elo_features"] + 
        feat["elo_extended_features"] + 
        feat["time_decay_features"] + 
        feat["league_onehot"]
    )

def get_base_features():
    """获取基础特征列表"""
    cfg = load_config()
    return cfg["features"]["base_features"]

def get_h2h_features():
    """获取交锋特征列表"""
    cfg = load_config()
    return cfg["features"]["h2h_features"]

def get_league_onehot():
    """获取联赛独热特征列表"""
    cfg = load_config()
    return cfg["features"]["league_onehot"]

def get_value_features():
    """获取身价特征列表"""
    cfg = load_config()
    return cfg["features"]["value_features"]

def get_full_feature_list():
    """获取完整特征列表（52维基础 + 5维身价 = 57维）"""
    return get_feature_list() + get_value_features()

def get_model_params():
    """获取模型默认超参数"""
    cfg = load_config()
    return cfg["model"]["default_params"].copy()

# ============================================================
# 联赛统一注册表（League Registry）— 全项目唯一的联赛编码出口
# 每个联赛一条记录，统一多套编码视角。以后新增联赛/数据源只改这里。
#   ⚠️ 本表按 cfg_key 排序（EPL,BUN,LLA,SER,LIG）；特征列的【顺序】权威在
#      config.json features.league_onehot（SER,E0,D1,LIG,LLA），训练/推理一律从
#      config.json 取顺序，本表只提供 码↔列名 的映射。
#   db_code     : 数据库表内实际存储码（match_feature_full/final/elo/value/result）← 主链路唯一事实
#   onehot_col  : 特征独热列名
#   csv_prefix  : football-data CSV 文件名前缀（big5_combined 的 league 列）
#   tm_code     : Transfermarkt competition_code
#   old_db_code : 旧链路 match_result 历史数据用的翻译码（SP1→LLA 等，仅存档说明，不再使用）
# ============================================================
LEAGUE_REGISTRY = {
    "EPL": {"name": "英超", "cfg_key": "EPL", "db_code": "E0",
            "onehot_col": "league_E0", "csv_prefix": "E0",
            "tm_code": "GB1", "old_db_code": "E0"},
    "BUN": {"name": "德甲", "cfg_key": "BUN", "db_code": "D1",
            "onehot_col": "league_D1", "csv_prefix": "D1",
            "tm_code": "L1", "old_db_code": "D1"},
    "LLA": {"name": "西甲", "cfg_key": "LLA", "db_code": "SP1",
            "onehot_col": "league_LLA", "csv_prefix": "SP1",
            "tm_code": "ES1", "old_db_code": "LLA"},  # ← 真实库码是 SP1，不是 LLA！
    "SER": {"name": "意甲", "cfg_key": "SER", "db_code": "I1",
            "onehot_col": "league_SER", "csv_prefix": "I1",
            "tm_code": "IT1", "old_db_code": "SER"},  # ← 真实库码是 I1，不是 SER！
    "LIG": {"name": "法甲", "cfg_key": "LIG", "db_code": "F1",
            "onehot_col": "league_LIG", "csv_prefix": "F1",
            "tm_code": "FR1", "old_db_code": "LIG"},  # ← 真实库码是 F1，不是 LIG！
}

# ---- 便捷查询索引 ----
_DB_TO_CFG = {v["db_code"]: k for k, v in LEAGUE_REGISTRY.items()}
_DB_TO_ONHOT = {v["db_code"]: v["onehot_col"] for k, v in LEAGUE_REGISTRY.items()}
_DB_TO_NAME = {v["db_code"]: v["name"] for k, v in LEAGUE_REGISTRY.items()}
_CSV_TO_DB = {v["csv_prefix"]: v["db_code"] for k, v in LEAGUE_REGISTRY.items()}
_TM_TO_DB = {v["tm_code"]: v["db_code"] for k, v in LEAGUE_REGISTRY.items()}


def get_all_db_codes():
    """所有主链路数据库联赛码（B体系：E0/D1/SP1/I1/F1）"""
    return [v["db_code"] for v in LEAGUE_REGISTRY.values()]


def get_all_onehot_cols():
    """所有联赛独热列名集合（特征顺序以 config.json features.league_onehot 为准）"""
    return [v["onehot_col"] for v in LEAGUE_REGISTRY.values()]


def db_to_onehot(db_code):
    """数据库联赛码 → 独热列名（训练/特征工程用）"""
    return _DB_TO_ONHOT.get(str(db_code))


def db_to_cfg(db_code):
    """数据库联赛码 → 配置键（team_mapping_v2 / 前端下拉用）"""
    return _DB_TO_CFG.get(str(db_code))


def db_to_name(db_code):
    """数据库联赛码 → 中文联赛名"""
    return _DB_TO_NAME.get(str(db_code))


def csv_to_db(prefix):
    """CSV文件名前缀 → 数据库联赛码"""
    return _CSV_TO_DB.get(str(prefix))


def tm_to_db(tm_code):
    """Transfermarkt competition_code → 数据库联赛码"""
    return _TM_TO_DB.get(str(tm_code))


def get_all_known_codes():
    """所有已知联赛码（B体系 db_code + 旧翻译码 old_db_code）
    供前端筛选等场景兼容历史数据，无论表内存哪套码都能匹配。
    例：['E0','D1','SP1','I1','F1','LLA','SER','LIG']
    """
    codes = []
    for v in LEAGUE_REGISTRY.values():
        if v["db_code"] not in codes:
            codes.append(v["db_code"])
        if v.get("old_db_code") and v["old_db_code"] not in codes:
            codes.append(v["old_db_code"])
    return codes


def league_name_by_code(code):
    """任意联赛码（db_code 或 old_db_code）→ 中文联赛名；未知码返回 None"""
    s = str(code)
    for v in LEAGUE_REGISTRY.values():
        if s == v["db_code"] or s == v.get("old_db_code"):
            return v["name"]
    return None


def split_train_val(df, val_ratio=0.2):
    """统一训练/验证切分出口（三个训练脚本共用）
    优先按 config.training.val_split_days（日期窗口切分），未配置则按 val_ratio 比例切分。
    【2026-08-07 统一：消除 train_general_elo_v2 / draw_binary_train / poisson_goals_train 各自硬编码 80/20】
    """
    import pandas as pd
    df_sorted = df.sort_values("match_date").reset_index(drop=True)
    cfg = load_config()
    val_days = cfg.get("training", {}).get("val_split_days", 0)
    if val_days and val_days > 0:
        max_date = df_sorted["match_date"].max()
        cut_date = max_date - pd.Timedelta(days=val_days)
        train_df = df_sorted[df_sorted["match_date"] <= cut_date].copy()
        val_df = df_sorted[df_sorted["match_date"] > cut_date].copy()
        if len(val_df) >= 50:
            return train_df, val_df
    split_idx = int(len(df_sorted) * (1 - val_ratio))
    return df_sorted.iloc[:split_idx].copy(), df_sorted.iloc[split_idx:].copy()


if __name__ == "__main__":
    # 自检
    cfg = load_config()
    print(f"项目: {cfg['project']['name']} v{cfg['project']['version']}")
    print(f"联赛数量: {len(cfg['leagues'])}")
    print(f"特征维度: {len(get_feature_list())} 维")
    print(f"  基础特征: {len(get_base_features())}")
    print(f"  交锋特征: {len(get_h2h_features())}")
    print(f"  联赛独热: {len(get_league_onehot())}")
    print(f"数据库路径: {get_path('db_path')}")

    # 注册表 vs config.json 一致性校验
    print("\n联赛注册表一致性校验：")
    _cfg_leagues = cfg["leagues"]
    for _k, _v in LEAGUE_REGISTRY.items():
        _j = _cfg_leagues.get(_k, {})
        _mark = "✅" if _j.get("db_code") == _v["db_code"] else f"⚠️ config.json={_j.get('db_code')}"
        print(f"  {_k}({_v['name']}): db_code={_v['db_code']} → 独热 {_v['onehot_col']}  {_mark}")
    print("✅ 配置加载正常")
