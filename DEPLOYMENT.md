# 足球赛事预测模型 - Streamlit Cloud 部署指南

## 项目简介
基于 LightGBM + 泊松进球模型的五大联赛足球赛事预测系统，包含手动预测、赛事日历、AI 模拟投注、策略回测、数据看板等功能。

> ⚠️ 免责声明：本项目仅供机器学习研究与学术交流使用，不构成任何投注建议或投资指导。

## 技术栈
- **前端**：Streamlit 1.29.0（多页面应用）
- **主模型**：LightGBM 三分类（主胜/平局/客胜）
- **辅助模型**：泊松进球回归 + 平局二分类专项
- **数据库**：SQLite（football.db，约 58MB）
- **数据范围**：2009-2026 五大联赛共 3 万 + 场比赛

---

## 部署步骤（Streamlit Cloud）

### 1. 准备代码仓库
将以下文件/目录推送到 GitHub 仓库（公开或私有均可）：

```
your-repo/
├── .streamlit/
│   └── config.toml          # Streamlit 配置
├── streamlit_dash/
│   ├── ⚽_预测中心.py        # 主入口文件
│   ├── predict_module.py     # 手动预测模块
│   ├── schedule_module.py    # 赛程模块
│   ├── ai_intent_module.py   # AI 下注意愿模块
│   ├── feature_auto_build.py # 特征自动构建
│   ├── common/               # 公共模块
│   │   ├── data_loader.py
│   │   ├── style.py
│   │   └── usage_tracker.py
│   └── pages/                # 子页面
│       ├── 1_🏆_策略回测.py
│       └── 2_📊_数据看板.py
├── model/                    # 模型文件（3个 .pkl）
│   ├── home_model.pkl
│   ├── poisson_home_goals.pkl
│   ├── poisson_away_goals.pkl
│   └── draw_binary_model.pkl
├── match_predict.py          # 预测核心逻辑
├── team_mapping_v2.py        # 球队中文映射
├── football.db               # SQLite 数据库（约 58MB）
└── requirements.txt          # Python 依赖
```

### 2. 登录 Streamlit Cloud
访问 https://share.streamlit.io ，使用 GitHub 账号登录。

### 3. 创建新 App
1. 点击 **"New app"**
2. 选择你的 GitHub 仓库
3. **Main file path** 填写：`streamlit_dash/⚽_预测中心.py`
4. （可选）自定义 App URL
5. 点击 **"Deploy!"**

### 4. 等待部署完成
首次部署需要安装依赖，约 2-5 分钟。部署成功后会自动打开应用。

---

## 依赖清单（requirements.txt）
```
streamlit==1.29.0
pandas==2.1.4
numpy==1.26.4
lightgbm==4.2.0
scikit-learn==1.3.2
joblib==1.3.2
```

---

## 注意事项

### 数据库大小
- `football.db` 约 58MB，Streamlit Cloud 支持 Git LFS 或直接提交
- 若仓库限制大文件，可使用 Git LFS 管理数据库文件

### 冷启动时间
- 首次访问或长时间闲置后，应用需要冷启动（约 10-30 秒）
- 数据库加载有缓存，热启动速度快

### 数据更新
- 部署后数据为静态，如需更新需重新推送数据库文件
- 增量更新脚本（`incremental_ai_update.py`）需在本地运行后推送

### 隐私与安全
- 所有数据均存储在 SQLite 文件中，无外部 API 调用
- 无用户账号系统，无个人数据收集
- 已内置免责声明提示

---

## 本地运行（测试用）

```bash
pip install -r requirements.txt
streamlit run streamlit_dash/⚽_预测中心.py
```

访问 http://localhost:8501

---

## 常见问题

### Q: 部署后报模块找不到错误？
A: 确认仓库目录结构与上方一致，`match_predict.py` 和 `team_mapping_v2.py` 必须在仓库根目录。

### Q: 数据库加载失败？
A: 确认 `football.db` 已完整上传，检查文件大小是否为 58MB 左右。

### Q: 页面中文显示乱码？
A: Streamlit Cloud 默认支持 UTF-8，确保所有 .py 文件为 UTF-8 编码。

### Q: 模型加载失败？
A: 确认 `model/` 目录下 4 个 .pkl 文件都存在，scikit-learn 和 lightgbm 版本匹配。

---

## 页面结构
1. **⚽ 预测中心**（主页）：手动预测、赛事日历、AI今日观点、预测追踪
2. **🏆 策略回测**：三AI模拟投注、赛季对比、策略动物园排行榜
3. **📊 数据看板**：比赛分析、球队详情、历史交锋数据
