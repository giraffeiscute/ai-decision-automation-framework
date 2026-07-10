# OHCA AI 決策自動化與 AED 資源配置框架

> [!NOTE]
> **[繁體中文](#繁體中文)** | **[English Version](#english-version)**

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-EB5B26)
![PyTorch](https://img.shields.io/badge/Model-PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![XAI](https://img.shields.io/badge/XAI-SHAP%20%7C%20LIME-6A5ACD)
![Gurobi](https://img.shields.io/badge/Optimization-Gurobi-EE3524)

<a name="繁體中文"></a>

## 📖 專案簡介

本專案建構一套 **AI 決策自動化框架**，整合風險預測、模型解釋與最佳化決策，將院外心臟驟停（OHCA）風險分佈模型轉化為可執行的 AED 資源配置方案。相較於只建立單一預測模型，本專案更重視「從模型輸出到決策方案」的完整自動化流程，讓 AI 結果能真正被決策單位採用。

專案導入可解釋 AI 與模型診斷流程，運用 SHAP、LIME 等方法分析黑盒模型，建立類 Root Cause Analysis（RCA）的特徵歸因流程，用於識別高風險因素、模型盲點與資料偏差，提升模型在真實場景中的可解釋性與穩健性。

在應用落地方面，本專案與深圳市及宜興市政府合作完成跨城市 PoC 驗證與部署策略規劃，將模型風險圖、可解釋 AI 歸因結果與最佳化座標轉化為可評估、可執行的公共安全決策。專案服務情境覆蓋約 **1,868 萬名居民**，並在實驗中使 AED 覆蓋率相較隨機部署提升約 **27%**。

![OHCA 風險預測、模型解釋與 AED 最佳化部署流程](https://github.com/giraffeiscute/ai-decision-automation-framework/blob/main/read_me_chart/Flow_chart.png)

---

## 🎯 專案定位

本倉庫將研究方法整理成可閱讀、可追蹤、可延伸的工程實作，重點不只是模型分數，而是展示一套 **AI workflow automation** 與 **decision support system** 的完整設計。

這個專案雖然應用於 OHCA 與 AED 部署，但核心能力可對應到智慧製造與 AI 軟體工程場景，例如預測分析、異常診斷、資源調度、模型可解釋性、決策最佳化與自動化工作流程。

公開版本以 Virginia Beach 的 OpenStreetMap 與 OHCA 資料驗證核心方法；深圳與宜興的跨城市 PoC、政府協作與部署規劃屬於專案落地經驗，受資料治理與合作規範限制，城市原始資料與部署設定未公開。

### 核心能力

* **端到端 AI 工作流程設計**：從資料處理、模型訓練、模型解釋到最佳化決策。
* **決策自動化**：將預測結果自動轉化為可執行的資源配置方案。
* **模型診斷與可解釋 AI**：透過 SHAP / LIME 分析模型判斷依據與潛在盲點。
* **類 RCA 特徵歸因流程**：識別高風險因素、資料偏差與模型異常行為。
* **最佳化建模**：使用 Gurobi 建立 AED 部署的二元整數規劃模型。
* **PoC 驗證與落地規劃**：支援跨城市驗證、部署策略規劃與決策評估。

---

## 🧩 系統流程

```text
OpenStreetMap POI / building data + historical OHCA records
                            │
                            ▼
Coordinate parsing → H3 spatial aggregation → feature table
                            │
                            ▼
Risk prediction models: XGBoost / PyTorch MLP / SVR / Poisson
                            │
                            ▼
Explainable AI: SHAP / LIME / representative local explanations
                            │
                            ▼
Model diagnosis: risk factors, blind spots, and data bias
                            │
                            ▼
Decision scoring: prediction and attribution to candidate locations
                            │
                            ▼
Gurobi integer programming for AED deployment optimization
                            │
                            ▼
Coverage, survival proxy, baseline comparison, and visualization
```

---

## 🔧 核心模組

### 1. 地理資料處理

* 解析 OpenStreetMap 中的 POI 與 building 資料。
* 將地理座標映射至 H3 spatial grid。
* 聚合區域特徵，建立可供模型訓練的 feature table。
* 支援跨城市資料遷移時替換資料來源並重新建立特徵。

### 2. 風險預測

* 使用 XGBoost、PyTorch MLP、SVR 與 Poisson regression 建立 OHCA 風險預測模型。
* 採用 spatial holdout 評估模型在地理空間上的泛化能力。
* 輸出區域風險分佈，作為後續最佳化決策的基礎。

### 3. 可解釋 AI 與模型診斷

* 使用 SHAP 分析模型對不同地理特徵的依賴。
* 使用 LIME 觀察單一區域的 local explanation。
* 建立類 RCA 的特徵歸因流程，用於識別高風險因素、模型盲點與資料偏差。
* 將模型診斷結果轉化為更容易被決策單位理解的解釋。

### 4. 決策分數轉換

* 將模型預測與 SHAP attribution 轉換為 candidate-level deployment score。
* 將風險分數映射回可部署位置，形成最佳化模型的輸入。
* 支援 prediction-only 與 SHAP-guided 兩種決策策略比較。

### 5. AED 最佳化部署

* 使用 Gurobi 建立二元整數規劃模型。
* 在 AED 數量限制與候選點間距限制下，選擇高價值部署位置。
* 比較最佳化部署與 random baseline 的覆蓋率差異。
* 支援不同 AED 數量、距離限制與模型來源的 sensitivity analysis。

---

## 📊 專案成果

| 指標 | 結果 |
|---|---:|
| PyTorch MLP spatial holdout | Test R² `0.761`, MAE `5.539` |
| XGBoost spatial holdout | Test R² `0.752`, MAE `5.750` |
| Linear SVR spatial holdout | Test R² `0.731`, MAE `5.911` |
| SHAP-guided deployment, `N=100`, `d_min=1.2 km` | Mean covered OHCA `1,385.9 / 1,983` (`69.9%`) |
| Random placement, `N=100` | Mean covered OHCA `1,088.63 / 1,983` (`54.9%`) |
| Coverage uplift | `+27.3%` relative (`+15.0` percentage points) |
| Mean survival proxy | SHAP-guided `14.8%` vs random `12.6%` |

在跨城市 PoC 與部署規劃中，專案將模型風險圖、XAI 特徵歸因與最佳化座標轉換為公共安全單位可評估的部署方案，服務情境覆蓋約 **1,868 萬名居民**。

---

## 📂 專案結構

```text
ai-decision-automation-framework/
├── README.md
├── read_me_chart/
│   └── Flow_chart.png
├── notebook/
│   ├── OpenStreetMap/              # OSM data parsing and preprocessing
│   ├── h3_l7_df process.ipynb      # H3 feature construction
│   ├── ML models.ipynb             # model training and comparison
│   ├── XGB.ipynb                   # XGBoost risk prediction
│   ├── NN.ipynb                    # PyTorch MLP risk prediction
│   ├── SVM.ipynb                   # SVR baseline
│   ├── poisson regression.ipynb    # statistical baseline
│   ├── XGB SHAP.ipynb              # XGBoost SHAP explanation
│   ├── NN SHAP.ipynb               # MLP SHAP explanation
│   ├── SVM SHAP.ipynb              # SVR SHAP explanation
│   ├── XGB_LIME SP.ipynb           # local explanation and representative selection
│   ├── LIME integrate.ipynb        # explanation integration
│   ├── summary.ipynb               # dataset and shift diagnostics
│   └── geo graph.ipynb             # geospatial visualization
└── Optimizor/
    ├── Data_20250720.py            # data loading and distance utilities
    ├── ModelBuilder_20250720.py    # Gurobi optimization model
    ├── run_this_20250720.py        # batch optimization experiments
    ├── run_this_random_*.py        # random deployment baselines
    ├── run_this_*_R+_20251003.py   # revised survival evaluation
    ├── difference table *.ipynb    # result comparison tables
    └── draw_fig_2.ipynb            # reporting figures
```

`notebook/` 負責資料工程、模型訓練與 XAI 分析；`Optimizor/` 負責決策建模、批次實驗與 KPI 評估。模型輸出與最佳化輸入透過 CSV / NumPy artifacts 解耦，因此可以替換風險模型而不必重寫 Gurobi model builder。

---

## 🛠️ 技術棧

| Layer | Technologies |
|---|---|
| Data Processing | Python, NumPy, pandas |
| Geospatial | H3, OSMnx, GeoPandas, Shapely |
| Machine Learning | XGBoost, PyTorch, scikit-learn, statsmodels |
| Explainable AI | SHAP, LIME |
| Optimization | Gurobi, binary integer programming |
| Evaluation | MAE, R², coverage, survival proxy, sensitivity analysis |
| Visualization | Matplotlib, Seaborn, H3 polygons |
| Artifacts | CSV, NumPy `.npy`, Excel tables |

---

## 🚀 安裝與使用

本專案目前以 Jupyter notebooks 與 Python batch scripts 為主，尚未整理成完整 package、CLI 或 production service。建議使用獨立 Python environment：

```bash
pip install jupyter numpy pandas matplotlib seaborn scipy scikit-learn \
  statsmodels xgboost torch shap lime h3 geopandas shapely osmnx tqdm openpyxl gurobipy
```

> [!IMPORTANT]
> 部分 notebooks 使用 h3-py 3.x API，例如 `geo_to_h3`、`h3_to_geo`、`h3_to_geo_boundary`。若使用 h3-py 4.x，需調整為新版 API。執行最佳化流程需要有效的 Gurobi license。

建議執行順序：

1. 準備 OpenStreetMap features 與 OHCA event data。
2. 執行 H3 feature construction notebooks。
3. 訓練 XGBoost / MLP / SVR / Poisson 風險模型。
4. 執行 SHAP / LIME notebooks 產生解釋結果。
5. 建立 candidate-level decision score。
6. 執行 `Optimizor/` 中的 Gurobi 最佳化與 random baseline。
7. 使用 comparison notebooks 彙整 coverage、survival proxy 與 sensitivity analysis。

---

## 📚 研究論文與引用

### 研究論文

論文定義 Learn-Then-Optimize 方法、SHAP-guided Integer Programming（SIP）模型與實驗設計。若引用研究方法或實驗結果，請引用論文。

### 工程實作

本倉庫提供該方法的 notebook 與 Python 實作，包括 OSM / H3 資料處理、多模型風險預測、SHAP / LIME 診斷、Gurobi 決策模型、random baseline、survival proxy、敏感度分析與地理視覺化。


```bibtex
@article{yang2025public,
  title={Public Access Defibrillator Deployment for Cardiac Arrests: A Learn-Then-Optimize Approach with SHAP-based Interpretable Analytics},
  author={Yang, Chih-Yuan and Leong, Keng-Hou and Cao, Kexin and Yang, Mingchuan and Chan, Wai Kin},
  journal={arXiv preprint arXiv:2501.00819},
  year={2025}
}
```

Yang, C.-Y., Leong, K.-H., Cao, K., Yang, M., & Chan, W. K. (2025). *Public Access Defibrillator Deployment for Cardiac Arrests: A Learn-Then-Optimize Approach with SHAP-based Interpretable Analytics*. arXiv:2501.00819.

[完整文章 arXiv 連結](https://arxiv.org/abs/2501.00819)

---

<a name="english-version"></a>

# OHCA AI Decision Automation and AED Deployment Framework

## 📖 Project Overview

This project builds an **AI decision automation framework** that integrates risk prediction, model explanation, and optimization to convert out-of-hospital cardiac arrest (OHCA) risk distributions into actionable AED deployment strategies. Instead of only building a standalone prediction model, this project focuses on the complete automation pipeline from model output to decision-making.

The project introduces explainable AI and model diagnosis using SHAP and LIME to analyze black-box models. It builds an RCA-like feature attribution workflow for identifying key risk factors, model blind spots, and data bias, improving model interpretability and robustness in real-world scenarios.

For real-world validation, the project supported cross-city PoC validation and deployment planning with Shenzhen and Yixing governments. It transformed model risk maps, explainable AI attributions, and optimized coordinates into actionable public-safety decisions. The project context covers approximately **18.68 million residents**, and the optimized AED deployment strategy improved AED coverage by approximately **27%** compared with random placement.

![OHCA risk prediction, model explanation, and AED deployment optimization workflow](https://github.com/giraffeiscute/ai-decision-automation-framework/blob/main/read_me_chart/Flow_chart.png)

---

## 🎯 Project Positioning

This repository translates the research method into a readable, traceable, and extensible engineering implementation. The focus is not only on model scores, but also on the design of an **AI workflow automation** and **decision support system**.

Although the application scenario is OHCA risk prediction and AED deployment, the core workflow is also relevant to smart manufacturing and AI software engineering scenarios, such as predictive analytics, anomaly diagnosis, resource scheduling, model explainability, decision optimization, and automated decision workflows.

The public implementation validates the core method with Virginia Beach OpenStreetMap and OHCA data. Shenzhen and Yixing cross-city PoCs, government collaboration, and deployment planning are project-delivery experience; city-specific source data and deployment configurations are not published because of data-governance and partnership constraints.

### Core Capabilities

* **End-to-end AI workflow design**: from data processing, model training, and model explanation to optimized decision-making.
* **Decision automation**: automatically converts prediction outputs into executable resource allocation strategies.
* **Model diagnosis and explainable AI**: uses SHAP / LIME to analyze model behavior and potential blind spots.
* **RCA-like feature attribution workflow**: identifies risk factors, data bias, and abnormal model behavior.
* **Optimization modeling**: formulates AED deployment as a binary integer programming problem with Gurobi.
* **PoC validation and deployment planning**: supports cross-city validation, deployment strategy planning, and decision evaluation.

---

## 🧩 System Workflow

```text
OpenStreetMap POI / building data + historical OHCA records
                            │
                            ▼
Coordinate parsing → H3 spatial aggregation → feature table
                            │
                            ▼
Risk prediction models: XGBoost / PyTorch MLP / SVR / Poisson
                            │
                            ▼
Explainable AI: SHAP / LIME / representative local explanations
                            │
                            ▼
Model diagnosis: risk factors, blind spots, and data bias
                            │
                            ▼
Decision scoring: prediction and attribution to candidate locations
                            │
                            ▼
Gurobi integer programming for AED deployment optimization
                            │
                            ▼
Coverage, survival proxy, baseline comparison, and visualization
```

---

## 🔧 Core Modules

### 1. Geospatial Data Processing

* Parses OpenStreetMap POI and building data.
* Maps geographic coordinates to H3 spatial grids.
* Aggregates regional features into model-ready feature tables.
* Supports data-source replacement and feature reconstruction for cross-city adaptation.

### 2. Risk Prediction

* Trains OHCA risk prediction models using XGBoost, PyTorch MLP, SVR, and Poisson regression.
* Uses spatial holdout evaluation to test geographic generalization.
* Produces regional risk distributions as input for downstream decision optimization.

### 3. Explainable AI and Model Diagnosis

* Uses SHAP to analyze model dependence on geographic features.
* Uses LIME for local explanations at individual spatial regions.
* Builds an RCA-like attribution workflow to identify high-risk factors, model blind spots, and data bias.
* Converts model diagnosis results into explanations that are easier for decision makers to interpret.

### 4. Decision Score Conversion

* Converts model predictions and SHAP attributions into candidate-level deployment scores.
* Maps risk scores back to deployable locations as optimizer inputs.
* Supports comparison between prediction-only and SHAP-guided decision strategies.

### 5. AED Deployment Optimization

* Formulates a binary integer programming model with Gurobi.
* Selects high-value deployment locations under AED budget and spacing constraints.
* Compares optimized deployment with random-placement baselines.
* Supports sensitivity analysis across AED budgets, distance constraints, and model sources.

---

## 📊 Results

| Metric | Result |
|---|---:|
| PyTorch MLP spatial holdout | Test R² `0.761`, MAE `5.539` |
| XGBoost spatial holdout | Test R² `0.752`, MAE `5.750` |
| Linear SVR spatial holdout | Test R² `0.731`, MAE `5.911` |
| SHAP-guided deployment, `N=100`, `d_min=1.2 km` | Mean covered OHCA `1,385.9 / 1,983` (`69.9%`) |
| Random placement, `N=100` | Mean covered OHCA `1,088.63 / 1,983` (`54.9%`) |
| Coverage uplift | `+27.3%` relative (`+15.0` percentage points) |
| Mean survival proxy | SHAP-guided `14.8%` vs random `12.6%` |

In cross-city PoC and deployment planning, the project translated model risk maps, XAI feature attributions, and optimized coordinates into deployment plans that public-safety teams could evaluate, with a service context covering approximately **18.68 million residents**.

---

## 📂 Repository Structure

```text
ai-decision-automation-framework/
├── README.md
├── read_me_chart/
│   └── Flow_chart.png
├── notebook/
│   ├── OpenStreetMap/              # OSM data parsing and preprocessing
│   ├── h3_l7_df process.ipynb      # H3 feature construction
│   ├── ML models.ipynb             # model training and comparison
│   ├── XGB.ipynb                   # XGBoost risk prediction
│   ├── NN.ipynb                    # PyTorch MLP risk prediction
│   ├── SVM.ipynb                   # SVR baseline
│   ├── poisson regression.ipynb    # statistical baseline
│   ├── XGB SHAP.ipynb              # XGBoost SHAP explanation
│   ├── NN SHAP.ipynb               # MLP SHAP explanation
│   ├── SVM SHAP.ipynb              # SVR SHAP explanation
│   ├── XGB_LIME SP.ipynb           # local explanation and representative selection
│   ├── LIME integrate.ipynb        # explanation integration
│   ├── summary.ipynb               # dataset and shift diagnostics
│   └── geo graph.ipynb             # geospatial visualization
└── Optimizor/
    ├── Data_20250720.py            # data loading and distance utilities
    ├── ModelBuilder_20250720.py    # Gurobi optimization model
    ├── run_this_20250720.py        # batch optimization experiments
    ├── run_this_random_*.py        # random deployment baselines
    ├── run_this_*_R+_20251003.py   # revised survival evaluation
    ├── difference table *.ipynb    # result comparison tables
    └── draw_fig_2.ipynb            # reporting figures
```

`notebook/` contains data engineering, model training, and XAI analysis. `Optimizor/` contains decision modeling, batch experiments, and KPI evaluation. Model outputs and optimization inputs are decoupled through CSV / NumPy artifacts, so the risk model can be replaced without rewriting the Gurobi model builder.

---

## 🛠️ Technology Stack

| Layer | Technologies |
|---|---|
| Data Processing | Python, NumPy, pandas |
| Geospatial | H3, OSMnx, GeoPandas, Shapely |
| Machine Learning | XGBoost, PyTorch, scikit-learn, statsmodels |
| Explainable AI | SHAP, LIME |
| Optimization | Gurobi, binary integer programming |
| Evaluation | MAE, R², coverage, survival proxy, sensitivity analysis |
| Visualization | Matplotlib, Seaborn, H3 polygons |
| Artifacts | CSV, NumPy `.npy`, Excel tables |

---

## 🚀 Installation and Usage

This repository is currently organized as Jupyter notebooks and Python batch scripts. It is not yet packaged as a complete Python package, CLI, or production service. An isolated Python environment is recommended:

```bash
pip install jupyter numpy pandas matplotlib seaborn scipy scikit-learn \
  statsmodels xgboost torch shap lime h3 geopandas shapely osmnx tqdm openpyxl gurobipy
```

> [!IMPORTANT]
> Some notebooks use the h3-py 3.x API, including `geo_to_h3`, `h3_to_geo`, and `h3_to_geo_boundary`. If using h3-py 4.x, the API calls need to be updated. The optimization pipeline also requires a valid Gurobi license.

Suggested workflow:

1. Prepare OpenStreetMap features and OHCA event data.
2. Run H3 feature construction notebooks.
3. Train XGBoost / MLP / SVR / Poisson risk models.
4. Run SHAP / LIME notebooks to generate explanations.
5. Build candidate-level decision scores.
6. Run Gurobi optimization and random baselines under `Optimizor/`.
7. Use comparison notebooks to summarize coverage, survival proxy, and sensitivity analysis.

---

## 📚 Research Paper and Citation

### Research Paper

The paper defines the Learn-Then-Optimize method, SHAP-guided Integer Programming (SIP) formulation, and experimental design. Please cite the paper when referencing the research method or experimental results.

### Engineering Implementation

This repository provides notebook and Python implementations of the method, including OSM / H3 data processing, multi-model risk prediction, SHAP / LIME diagnosis, Gurobi decision modeling, random baselines, survival proxy evaluation, sensitivity analysis, and geospatial visualization.


```bibtex
@article{yang2025public,
  title={Public Access Defibrillator Deployment for Cardiac Arrests: A Learn-Then-Optimize Approach with SHAP-based Interpretable Analytics},
  author={Yang, Chih-Yuan and Leong, Keng-Hou and Cao, Kexin and Yang, Mingchuan and Chan, Wai Kin},
  journal={arXiv preprint arXiv:2501.00819},
  year={2025}
}
```

Yang, C.-Y., Leong, K.-H., Cao, K., Yang, M., & Chan, W. K. (2025). *Public Access Defibrillator Deployment for Cardiac Arrests: A Learn-Then-Optimize Approach with SHAP-based Interpretable Analytics*. arXiv:2501.00819.

[Paper on arXiv](https://arxiv.org/abs/2501.00819)
