# ⚽ football_pred — 部署手册（DEPLOYMENT.md）

> **版本**：1.4.0（2026-08-07）｜**定位**：机器学习模型验证工具，非博彩/投注类产品
> **本文件是部署/运维的唯一权威文档**（置于仓库根目录，随代码入 git）。
> 研究类文档见【文档】/（本地保留，不入 git）。
>
> ⚠️ **免责声明**：本项目仅供机器学习研究与学术交流使用，不构成任何投注建议或投资指导。

---

## 一、项目简介

五大联赛（英超/西甲/德甲/意甲/法甲）足球赛事胜负预测系统。
**LightGBM 三分类 + 泊松进球模型 + 平局专项二分类** 三模型融合 + 概率校准。

- 版本：**1.4.0**（2026-08-07，`config.json` 为权威出口）
- 特征：**57 维**（52 基础 + 5 身价）
- 技术栈：Python 3.10 / Streamlit / LightGBM / scikit-learn / SQLite
- 金标准准确率：**51.75%**（去重后 50752 场唯一比赛 · Walk Forward 无未来函数）

## 二、环境要求

- Python **3.10.x**（建议用 `py -3.10` 启动器）
- 依赖见 `requirements.txt`（宽松版本区间，见第四节）

## 三、部署架构（git 上传清单）

### ✅ 随仓库上传（部署必需）

| 资源 | 说明 |
|---|---|
| 根目录核心模块 | `match_predict.py`（生产推理入口）/ `team_mapping_v2.py`（球队映射）/ `common_config.py`（特征与模型参数统一出口）/ `export_wf_conf_table.py` |
| `config.json` | 特征定义 + 模型参数权威源（运行必需） |
| `football.db` | SQLite 数据库（约 **64MB**，已瘦身，随代码上传） |
| `model/` | 模型产物（核心 pkl + 配置 json；`_deprecated/` 与 `*.bak` 不入库） |
| `common/` | 公共模块（数据加载/样式/埋点） |
| `streamlit_dash/` | Streamlit 看板（入口 `⚽_预测中心.py`） |
| `tests/` | 回归测试（pytest，24 项） |
| `.streamlit/config.toml` | Streamlit 主题/服务器配置 |
| `requirements.txt` / `README.md` / `CHANGELOG.md` / `pytest.ini` / `.gitignore` / `DEPLOYMENT.md` | 项目文档与配置 |

### ❌ 不上传（仅本地维护）

| 资源 | 说明 |
|---|---|
| `data/` `features/` `backtest/` `training/` `tools/` `logs/` | 离线数据导入/特征构建/训练/回测脚本，仅本地执行 |
| `update_weekly.py` / `*.bat` | 本地更新脚本（依赖 `data/` 与 `features/`，云端无意义） |
| `csv_data/` | 原始数据包（football-data zip） |
| `archive/` `model/_deprecated/` `model_backup_*/` | 归档 |
| `【文档】/` `【策略研究】/` `论文区/` | 研究文档（本地保留） |

> 判断依据：`.gitignore` 的放行/忽略规则即部署边界；修改上传清单时请同步修改本表与 `.gitignore`。

## 四、依赖清单（requirements.txt）

```text
streamlit>=1.29,<2.0
pandas>=2.0,<4.0
numpy>=1.26,<3.0
scipy>=1.10,<2.0
lightgbm>=4.0,<5.0
scikit-learn>=1.3,<2.0
joblib>=1.3,<2.0
plotly>=5.0,<6.0
tqdm>=4.66,<5.0

# --- dev / 测试（回归测试 tests/）---
pytest>=8.0,<9.0
```

## 五、部署步骤（3 步）

### 1. 拉取代码

```bash
git clone <your-repo>.git
cd football_pred
```

> `football.db`（64MB）随仓库一起克隆，**无需单独同步**。

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

> 建议在 Python 3.10 虚拟环境中安装。

### 3. 启动看板

```bash
python run.py
# 或直接
streamlit run streamlit_dash/⚽_预测中心.py
```

浏览器打开 **http://localhost:8501**

## 六、部署后验证（必须）

| 步骤 | 命令 | 预期 |
|---|---|---|
| 1. 回归测试 | `python -m pytest tests/ -q --tb=short` | **24 passed** |
| 2. 数据库验证 | `python verify_db.py` | 核心表行数/列数/索引正常 |
| 3. 看板启动 | `python run.py` | 主页正常打开，无报错 |
| 4. 模型链路 | 看板执行一次预测 | 57 维链路正常，输出 1x2 概率 |

> 若部署后任何一步失败，见第九节常见问题。

## 七、数据库 football.db（约 64MB）

随仓库上传的 SQLite 数据库，核心表如下：

| 表 | 说明 |
|---|---|
| `match_feature_final` | **主特征表**（57 维，含 `pred_result` / `pred_confidence` 回填列） |
| `match_feature_full` | 特征全量明细 |
| `match_result` | 比赛结果 |
| `match_elo` | ELO 数据 |
| `team_value_features` | 身价特征（ASOF 无未来值，2008 起真实覆盖 60.9%） |
| `wf_confidence_accuracy` / `league_independent_wf` / `strategy_zoo_*` | 策略验证/回测结果表 |
| `predictions` / `usage_log` / `match_schedule` | 用户预测记录 / 使用日志 / 赛程 |

> 数据库为**静态数据资产**：云端部署后不随新赛果自动更新。更新方式见第八节。

## 八、数据更新（仅本地执行）

> ⚠️ `update_weekly.py`、`data/`、`features/` 均**不入 git**，只在本地运行。

### 周更一键脚本 `update_weekly.py`

```bash
python update_weekly.py                       # 全流程（交互确认）
python update_weekly.py --auto                # 全流程（不暂停，供定时任务调用）
python update_weekly.py --auto --retrain      # 全流程 + 自动重训 + 自动回填（重操作，慎用）
python update_weekly.py --zip "路径\data.zip"  # 先归档新数据包再全流程
python update_weekly.py --check               # 只健康检查（不写库）
python update_weekly.py --force-rebuild       # 即使无新增也重建特征
```

流程：**健康检查 → dry-run 预览 → 增量入库 → 特征重建（3 步）→ 校验报告 →（可选 `--retrain`）模型备份 → 自动重训 → 自动回填 → 回归验证**。

> ⚠️ `--retrain` 为重操作（重训 4 个模型 + 全量回填），默认不启用；与 `--auto` 配合可接入定时任务。重训前自动备份 `model/` 到 `model_backup_时间戳/`，失败可回滚。

### 云端更新方式

本地跑完 `update_weekly.py --auto` → 重新推送 `football.db` 到仓库 → 云端重新部署。

## 九、常见问题

| 问题 | 解决 |
|---|---|
| 部署后 `ModuleNotFoundError` | 确认根目录核心模块（`match_predict.py` 等）已上传，见第三节 |
| 看板报数据库表缺失 | 确认 `football.db` 已完整克隆（约 64MB），跑 `python verify_db.py` |
| 模型加载失败 | 确认 `model/` 下 5 个核心 pkl 齐全（home / draw_binary / poisson×2 / confidence_calibrator），LightGBM 版本兼容 |
| 中文乱码 | 所有 .py 为 UTF-8 编码；终端运行用 `py -3.10` 并确保 `PYTHONIOENCODING=utf-8` |
| 预测结果异常 / 特征列不齐 | 跑 `python -m pytest tests/ -q --tb=short`，确认 24 项全绿 |

## 十、相关文档

- `README.md`：项目总览 / 快速开始 / 开发计划
- `CHANGELOG.md`：版本历史
- `tests/`：回归测试（pytest）
- 【文档】/（本地保留）：研究纲领 / 结论总览 / 验证标准等研究文档
