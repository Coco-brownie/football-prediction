# -*- coding: utf-8 -*-
"""导出 wf_confidence_accuracy 表到 model/wf_confidence_accuracy.json
【背景】football.db 达 165MB 超过 GitHub 100MB 限制，无法入库。
        但方案A置信度查表只有 6 行（置信度区间→历史命中率），导出为 JSON 随仓库分发，
        克隆后无需 db 即可让方案A工作（match_predict.py 优先读 JSON，db 仅本地兜底）。
运行：python export_wf_conf_table.py
"""
import os
import json
import sqlite3

ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(ROOT, "football.db")
OUT_PATH = os.path.join(ROOT, "model", "wf_confidence_accuracy.json")

# 与 backtest/generate_dashboard_tables.py 分桶严格一致
BOUNDS = {"<50%": 0.0, "50-60%": 0.50, "60-70%": 0.60,
          "70-80%": 0.70, "80-90%": 0.80, "≥90%": 0.90}

if not os.path.exists(DB_PATH):
    raise FileNotFoundError(f"未找到数据库：{DB_PATH}")

conn = sqlite3.connect(DB_PATH)
try:
    rows = conn.execute(
        "SELECT 置信度区间, 准确率 FROM wf_confidence_accuracy"
    ).fetchall()
finally:
    conn.close()

if not rows:
    raise RuntimeError("wf_confidence_accuracy 表为空，请先运行【策略研究】/01_基准策略动物园/wf_confidence_accuracy.py")

data = []
for lab, acc in rows:
    lab = str(lab).strip()
    if lab not in BOUNDS:
        print(f"⚠️ 跳过未知档位: {lab}")
        continue
    data.append({"置信度区间": lab, "准确率": float(acc)})

data.sort(key=lambda x: BOUNDS[x["置信度区间"]])
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"✅ 已导出 {len(data)} 条 → {OUT_PATH}")
for d in data:
    print(f"   {d['置信度区间']:<8} {d['准确率']:.2%}")
