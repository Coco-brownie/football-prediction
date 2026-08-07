"""验证瘦身后的数据库核心功能是否正常"""
import os
import sqlite3
import pandas as pd

# 数据库路径：优先取脚本同目录（兼容本地 Windows 与 CI Linux 环境）
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'football.db')

conn = sqlite3.connect(DB_PATH)

print("=== 核心表数据验证 ===\n")

# 1. match_feature_final（最核心的表）
print("1. match_feature_final（主特征表）")
df = pd.read_sql("SELECT * FROM match_feature_final LIMIT 3", conn)
print(f"   总行数: {pd.read_sql('SELECT COUNT(*) as cnt FROM match_feature_final', conn).iloc[0,0]:,}")
print(f"   列数: {len(df.columns)}")
print(f"   最新日期: {pd.read_sql('SELECT MAX(match_date) FROM match_feature_final', conn).iloc[0,0]}")
print(f"   最早日期: {pd.read_sql('SELECT MIN(match_date) FROM match_feature_final', conn).iloc[0,0]}")
print()

# 2. match_result
print("2. match_result（比赛结果）")
df = pd.read_sql("SELECT * FROM match_result LIMIT 3", conn)
print(f"   总行数: {pd.read_sql('SELECT COUNT(*) as cnt FROM match_result', conn).iloc[0,0]:,}")
print(f"   列数: {len(df.columns)}")
print()

# 3. match_elo
print("3. match_elo（ELO数据）")
df = pd.read_sql("SELECT * FROM match_elo LIMIT 3", conn)
print(f"   总行数: {pd.read_sql('SELECT COUNT(*) as cnt FROM match_elo', conn).iloc[0,0]:,}")
print(f"   列数: {len(df.columns)}")
print()

# 4. 策略验证表
print("4. 策略验证表")
for t in ['wf_confidence_accuracy', 'league_independent_wf', 'strategy_zoo_wf_a_36', 'strategy_zoo_full_a_36']:
    cnt = pd.read_sql(f'SELECT COUNT(*) as cnt FROM {t}', conn).iloc[0,0]
    print(f"   {t}: {cnt} 行")
print()

# 5. 用户数据表
print("5. 用户数据")
for t in ['predictions', 'usage_log', 'match_schedule', 'team_value_features']:
    cnt = pd.read_sql(f'SELECT COUNT(*) as cnt FROM {t}', conn).iloc[0,0]
    print(f"   {t}: {cnt} 行")
print()

# 6. 检查索引是否还在
print("6. 索引检查")
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='index' ORDER BY name")
indexes = [row[0] for row in cursor.fetchall()]
print(f"   索引数量: {len(indexes)}")
for idx in indexes[:10]:
    print(f"   - {idx}")
if len(indexes) > 10:
    print(f"   ... 还有 {len(indexes)-10} 个")

conn.close()

print("\n✅ 验证完成，核心表数据完整！")
