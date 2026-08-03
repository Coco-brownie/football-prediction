# ⚽ 足球赛事预测中心 (football_pred)

五大联赛（英超/西甲/德甲/意甲/法甲）足球赛事胜负预测系统。
**LightGBM 三分类 + 泊松进球模型 + 平局专项二分类** 三模型融合 + 概率校准。

- 版本：**1.3.0**（2026-08-08）
- 技术栈：Python 3.10 / LightGBM / scikit-learn / Streamlit / SQLite

---

## ✨ 核心能力

- **赛事预测**：融合 LGB(55%) + 泊松(30%) + 平局专项(15%)，输出 1x2 概率 + 置信度
- **比分/大小球**：泊松模型预测进球期望、TOP 比分、多档位大小球
- **特征解释**：SHAP 单场特征贡献（正向/负向 Top3）
- **价值决策**：模型概率 vs 市场隐含概率，期望值 + 凯利仓位参考
- **模型验证**：严格无未来函数 Walk Forward 金标准，外样本统一验证

## 📊 模型表现（Level 3 金标准）

| 指标 | 数值 |
|---|---|
| 整体准确率（无未来函数） | **52.01%**（三分类随机基线 33.3%） |
| 特征维度 | **57**（52 基础 + 5 身价） |
| 身价特征真实覆盖 | **60.9%**（2008 起，ASOF 无未来值） |
| 身价特征重要性 | 4.71%（健康占比） |
| 投注 ROI（真实赔率扣抽水） | -1.7%~-2.3%（**直接下注不盈利，庄家抽水现实**） |

> ⚠️ 模型用于数据分析/学术研究，**严禁用于任何形式的赌博投注**。

---

## 📁 目录结构

```
football_pred/
├── match_predict.py          # 生产推理入口（57维，三模型融合+校准）
├── common_config.py          # 特征/联赛/模型参数统一配置出口（全模块依赖）
├── config.json               # 特征定义+模型参数权威源（运行必需）
├── team_mapping_v2.py        # 球队/联赛映射
├── streamlit_dash/           # Streamlit 看板（预测中心/模型验证/数据看板）
│   ├── ⚽_预测中心.py         # 主页
│   ├── predict_module.py     # 预测面板
│   ├── feature_auto_build.py # 在线特征构建（ELO/身价/攻防）
│   └── pages/                # 子页面
├── common/                   # 公共模块（数据加载/样式/埋点）
├── model/                    # 模型产物（home_model/draw_binary/poisson/calibrator）
├── features/                 # 离线特征构建脚本（含 build_team_value_features.py）
├── training/                 # 训练脚本（train_all_clean.py 等，仅本地执行）
├── backtest/                 # 回测/验证脚本（walk_forward 金标准）
└── football.db               # SQLite 数据库（本地数据资产，不入 git）
```

## 🚀 快速开始（本地）

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动看板
python run.py          # 或双击 启动看板.bat
# 浏览器打开 http://localhost:8501

# 3. 命令行验证
python backtest\recalculate_predictions.py   # 生产模型全量回填+健康检查
python backtest\walk_forward.py              # WF 金标准验证
```

## 📦 部署说明（重要）

| 资源 | 是否入 git | 说明 |
|---|---|---|
| 代码（根目录核心 + common/ + streamlit_dash/） | ✅ | 完整上传 |
| `model/` 模型产物 | ✅ | 核心产物 |
| `common_config.py` / `config.json` | ✅ | **运行必需，已放行** |
| `football.db` 数据库 | ❌ | 173MB 本地数据资产，部署时单独同步 |
| `features/ backtest/ training/ data/ tools/` | ❌ | 离线维护脚本，仅本地执行 |
| `model/_deprecated/ model_backup_*/` | ❌ | 归档 |

**云端/新环境部署步骤**：
1. `git clone`（拉取代码 + 模型 + 配置）
2. 单独同步 `football.db`（含 `team_value_features` 重建表 + `pred_result/pred_confidence` 回填）
3. `pip install -r requirements.txt`
4. `streamlit run streamlit_dash/⚽_预测中心.py`

> 📖 **详细部署步骤 + 数据库同步手册（表清单/同步方式/部署后验证）见 `DEPLOYMENT.md`**

---

## 🗺️ 开发计划

### ✅ 已完成（v1.3.0）
1. **身价特征数据修复**：ASOF 快照重建 + 500 万下限清洗 + 零未来值
2. **全模型重训**：57 维主模型，OOS 53.39%
3. **统计口径统一**：train/backtest/walk_forward 全部 60.9% 真实覆盖
4. **金标准验证**：WF 52.01% 无未来函数
5. **生产健康检查**：全量回填 52561 条，57 维链路健康
6. **推送健全性**：放行 common_config.py + config.json，新增 CHANGELOG/README

### 🔜 后续规划（策略层）
- [ ] **价值投注策略**：模型概率 vs 赔率隐含概率差值，只下注正期望场次（当前 ROI 为负的改善方向）
- [ ] **平局独立通道**：draw_binary 置信度阈值化输出，避免融合权重硬拉买"尾巴货"
- [ ] **联赛特化模型**：按联赛单独验证/调参（当前 5 联赛 50.3%~53.1% 分布）
- [ ] **在线更新链路**：周更快照 → 自动重建身价特征 → 自动重训 → 自动回填
