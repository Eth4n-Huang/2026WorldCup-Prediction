<div align="center">

# ⚽ 2026WorldCup-Prediction

**基于 Dixon-Coles 双泊松与动态 Elo 的 2026 世界杯逐场胜平负预测模型**

*A leakage-free Dixon-Coles + dynamic-Elo model for 2026 FIFA World Cup match prediction*

[🔮 在线看板 Live Dashboard](https://eth4n-huang.github.io/2026WorldCup-Prediction/)

</div>

---

## 📖 简介 · Overview

这是一个**严肃、可复现、防数据泄漏**的足球比赛预测模型,用于预测 2026 美加墨世界杯的逐场胜平负。

它不依赖任何"玄学因子"或人工拍定的权重——所有参数都从历史数据估计,并通过**三届世界杯(2014/2018/2022)共 192 场的滚动回测**严格验证。模型在世界杯期间每日滚动产出**前瞻性预测**,所有预测在赛前锁定、只追加不修改,以保证可被真实结果检验。

> This is a serious, reproducible, **leakage-free** football prediction model. No "mystic factors", no hand-picked weights — every parameter is estimated from historical data and validated through **rolling backtests over 3 World Cups (192 matches)**. During the tournament it produces **forward-looking daily predictions**, locked before kickoff and append-only, so they can be honestly checked against real outcomes.

---

## ✨ 亮点 · Highlights

- **🎯 主模型 Dixon-Coles 双泊松**：显式建模低比分相关性(ρ 修正),λ 由历史进球数最大似然估计,而非线性硬拍。
- **📈 动态 Elo 评分**：含主场优势、赛事类型加权(世界杯/预选赛/友谊赛)的时序演化评分,作为核心特征。
- **🔬 防数据泄漏铁律**：特征只用赛前信息;严格按时间切分(禁随机划分/K 折);超参数只在训练期选择。准确率异常跳升一律先怀疑泄漏。
- **📊 三届回测验证**：在 2014/2018/2022 共 192 场上,准确率 **55.2%**、Brier **0.579**,显著优于"Elo 高者胜"基线的概率质量。
- **🤖 XGBoost 对照模型**：作为机器学习对照(含交互特征),与 DC 主模型并列评估。
- **🖥️ 实时滚动看板**：预测看板 + 积分榜(含 2026 新规最佳第三名)+ 赛程(北京时间),每日更新,零服务器静态部署。

---

## 📊 模型表现 · Performance

### 历史回测(主指标) · Backtest (primary metric)

在 2014 / 2018 / 2022 三届世界杯、共 **192 场**比赛上的滚动回测结果:

| 模型 Model | 准确率 ACC | Brier |
|---|---|---|
| **Dixon-Coles(主模型)** | **0.552** | **0.579** |
| XGBoost(对照) | 0.563 | — |
| 基线 a:Elo 高者胜 | ~0.52 | 更高(差) |
| 基线 b:随机猜测 | ~0.33 | — |

> ⚠️ **关于准确率天花板的诚实说明**：足球单场预测的准确率上限约为 **55%**,这是该问题的固有难度——博彩公司用海量资源也难以突破。本模型达到这一水平,且在概率质量(Brier / LogLoss)上显著优于基线。**"准确率 55%"不是缺陷,而是这个问题做到位后该有的样子。**

### 2026 实时预测 · Live (in-tournament)

模型在 2026 世界杯期间每日滚动产出前瞻预测。**截至小组赛前期,实时准确率低于回测值**,主要原因是本届出现了**统计上显著的"平局潮"**(前 14 场平局率达 57%,远超历史均值 25%,Fisher 检验 p≈0.03)。平局是所有预测模型(包括博彩公司)的共同盲区。即便如此,DC 模型在该阶段仍是所有对照中最准的,跑赢傻瓜基线约一倍。

> The live accuracy during the early group stage runs below the backtest figure, driven by a **statistically significant "draw surge"** unique to this tournament (57% draws in the first 14 matches vs. a 25% historical mean, Fisher p≈0.03). Draws are the blind spot of every predictive model. Even so, the DC model remained the most accurate among all baselines in this period. 实时滚动数据见[在线看板](https://eth4n-huang.github.io/2026WorldCup-Prediction/)。

---

## 🧠 方法论 · Methodology

### 核心特征

- **动态 Elo**(主特征):时序演化,含主场优势 H_adv、赛事类型 K 值加权
- **近期状态**:近 5/10 场胜率、进失球
- **疲劳**:休息天数差、加时标记
- **情境**:东道主 / 主场 / 中立场、大洲、赛事类型、阶段
- **世界杯经验、历史交锋**(h2h,低权重)
- **出线形势**:积分、排名、出线状态、必胜指数(含 2026 最佳第三名规则)

### 主模型:Dixon-Coles 双泊松

```
主队期望进球 λ_home、客队期望进球 λ_away 由历史数据 MLE 估计
P(进球数) = Poisson 分布,并对低比分(0-0/1-1/1-0/0-1)施加 ρ 相关性修正
胜平负概率 = 对所有比分组合的概率求和
```

相比朴素 Poisson(假设两队进球独立),Dixon-Coles 的 ρ 修正能更准确地刻画真实足球中低比分的相关性,这是该方法被学界广泛采用的原因。

### 防泄漏设计(本项目的核心严谨性)

- 所有特征只使用**比赛日之前**的信息
- 数据**只按时间切分**,严禁随机划分或 K 折交叉验证
- 超参数(含平局阈值 δ)**只在训练期(1998–2014)选择**,测试期完全不可见
- 实时预测**赛前锁定、只追加不修改**(`is_current` 去重机制),杜绝事后追配
- 开幕日早于流水线上线的比赛**不计入准确率统计**,防止回溯泄漏

---

## 📁 项目结构 · Structure

```
src/
├── step1_load_data.py     # 数据清洗(Kaggle 国际比赛 1872-2025)
├── step2_elo.py           # 动态 Elo 评分
├── step3_features.py      # 特征工程
├── step4_train.py         # 模型训练(DC + XGBoost)
├── step5_backtest.py      # 三届滚动回测
├── step6_ablation.py      # 消融实验
├── step7_predict_2026.py  # 蒙特卡洛模拟
├── daily_update.py        # 每日滚动:录比分→更新动态特征→输出预测→结算
├── build_dashboard.py     # 生成看板 HTML
├── best_thirds.py         # 2026 最佳第三名出线逻辑
└── team_names.py          # 权威队名映射
index.html                 # 三标签页看板(预测/积分榜/赛程)
```

---

## 🚀 使用 · Usage

在线看板无需安装,直接访问：**https://eth4n-huang.github.io/2026WorldCup-Prediction/**

本地运行预测流水线：

```bash
python src/step1_load_data.py     # 准备数据
python src/step5_backtest.py      # 复现三届回测
python src/daily_update.py        # 每日滚动预测与结算
python src/build_dashboard.py     # 重新生成看板
```

---

## 🔭 局限与未来工作 · Limitations & Future Work

本项目秉持诚实的科学态度,明确承认以下边界：

- **准确率天花板**:足球单场预测上限约 55%,本模型无法、也不应声称突破这一物理极限。
- **平局难题**:三分类预测对平局的召回天然偏弱,这是全行业共性问题。
- **球队级建模**:当前模型基于国家队整体战绩(Elo),尚未引入球员级数据(身价、年龄结构、阵容深度)。引入球员特征是明确的未来工作方向——但需保证时点正确以防泄漏。
- **赛制变化的影响**:2026 扩军至 48 队 + 最佳第三名规则可能改变比赛结果分布,本项目正以实时数据研究这一现象。

---

## 📜 数据来源 · Data

- 历史国际比赛结果：Kaggle *International football results 1872–2025*
- 2026 赛程与赛果：公开赛事数据(以 FIFA 官方为准)

---

<div align="center">

如果这个项目对你有帮助,欢迎点一个 ⭐ Star!

*If you find this project useful, a ⭐ would be appreciated!*

</div>
