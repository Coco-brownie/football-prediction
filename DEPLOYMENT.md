# 🚀 部署手册（DEPLOYMENT）

## 架构概览

```
┌─────────────────────────────────────────────────────────┐
│  Git 仓库（GitHub）                                      │
│  ├─ 代码：根目录核心 + common/ + streamlit_dash/          │
│  ├─ 模型：model/ 核心产物（4 件套 + 重训中间产物）         │
│  └─ 配置：common_config.py + config.json（运行必需）      │
├─────────────────────────────────────────────────────────┤
│  本地数据资产（不入 git，部署时单独同步）                   │
│  └─ football.db（173MB SQLite）                          │
└─────────────────────────────────────────────────────────┘
```

---

## 一、部署环境要求

| 依赖 | 说明 |
|---|---|
| Python | 3.10+ |
| 依赖包 | `pip install -r requirements.txt`（streamlit / lightgbm / scikit-learn / pandas / numpy / joblib / plotly / tqdm） |
| 数据库 | SQLite（`football.db`，免安装服务） |

---

## 二、完整部署步骤

### 第 1 步：拉取代码
```bash
git clone <repo_url> && cd football_pred
```

### 第 2 步：同步数据库 ⚠️ 关键步骤（代码运行必需）

**方式 A：整库直传（最简单，推荐）**
```bash
# 本地执行
scp football.db  user@server:/path/to/football_pred/
```
`football.db` 包含运行必需的表：
- `match_feature_final`：52,561 场比赛特征 + `pred_result/pred_confidence`（已回填）
- `team_value_features`：ASOF 重建身价表（2008 起真实覆盖 60.9%，零未来值）
- `match_feature_full`：赔率源表（odds_win/draw/lose）
- `tm_club_squads`：身价快照源表（重建身价特征用）

**方式 B：只同步关键表（小体积，带宽受限时）**
```bash
# 本地导出
sqlite3 football.db ".dump match_feature_final"  > mff.sql
sqlite3 football.db ".dump team_value_features"  > tvf.sql
# 服务器导入（需先建空库 + 建表结构）
sqlite3 football.db < mff.sql
sqlite3 football.db < tvf.sql
```
> ⚠️ 方式 B 需保证表结构与本地一致（列定义见 `features/` 构建脚本）。

### 第 3 步：安装依赖
```bash
pip install -r requirements.txt
```

### 第 4 步：启动看板
```bash
streamlit run streamlit_dash/⚽_预测中心.py --server.port 8501
```

---

## 三、部署后验证（必须执行）

1. **看板启动**：首页正常加载，无「【模型缺失】」红色报错
2. **生产健康检查**：
   ```bash
   python backtest\recalculate_predictions.py
   ```
   预期输出：`主模型特征数: 57`、`已更新 52561 条记录`、校准器 `a=4.3632, b=-2.2046`
3. **实时预测**：预测一场比赛 → 正常输出（置信度 0.33~0.85）
4. **回填检查**：`match_feature_final` 的 `pred_result/pred_confidence` 非空

---

## 四、模型产物清单（git 内，勿删）

| 文件 | 用途 | 来源 |
|---|---|---|
| `model/home_model.pkl` | 主模型 LGB 三分类（57 维） | 8/3 重训 |
| `model/draw_binary_model.txt` | 平局专项二分类（52 维） | 8/3 重训 |
| `model/poisson_model_params.json` | 泊松进球参数 | 8/3 重训 |
| `model/calibrator_params.json` | 置信度校准器 a/b | 8/3 重训 |
| `model/draw_binary_model.pkl` | **重训中间产物（convert 源，勿删）** | 8/3 |
| `model/poisson_home_goals.pkl` / `poisson_away_goals.pkl` | **重训中间产物（convert 源，勿删）** | 8/3 |
| `model/confidence_calibrator.pkl` | **重训中间产物（convert 源，勿删）** | 8/3 |
| `model/poisson_features.json` | 泊松特征定义 | 8/3 |
| `model/*_feat_import.csv` | 特征重要性（看板展示） | 8/3 |

> 📦 旧模型已归档至 `model/_deprecated/`（gitignore 忽略，不上传）。
> 若需重训，`training/train_all_clean.py` 会从上述中间产物重新生成 4 件套。

---

## 五、完整数据/模型重建（如需在云端从零构建）

```bash
# 1. 导入原始数据
python features\import_full_data.py
# 2. 重建特征表 match_feature_final
python features\rebuild_all_features.py
# 3. ASOF 重建身价特征表 team_value_features
python features\build_team_value_features.py
# 4. 重训全模型（生成 model/ 4 件套）
python training\train_all_clean.py
# 5. 回填预测
python backtest\recalculate_predictions.py
```
> ⚠️ `features/ training/ backtest/` 为离线脚本，**不在 git**（gitignore 约定），云端重建需从本地拷贝或单独维护。

---

## 六、安全与运维

- `config.json` **无敏感信息**（仅路径/特征/参数），可安全入库
- `football.db` 为自有数据资产，注意服务器访问控制（勿公开泄露）
- 每赛季结束后建议例行：**重训 → 回填 → 重启看板**
- 模型仅供数据分析/学术研究，**严禁用于赌博投注**
