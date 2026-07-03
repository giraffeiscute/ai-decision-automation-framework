# OHCA AI 決策自動化與 AED 資源配置框架

# OHCA AI Decision Automation & AED Deployment Framework

> 中文在前，英文緊接於各節之後。
> Chinese first, with English immediately following each section.

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-EB5B26)
![PyTorch](https://img.shields.io/badge/Model-PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![XAI](https://img.shields.io/badge/XAI-SHAP%20%7C%20LIME-6A5ACD)
![Gurobi](https://img.shields.io/badge/Optimization-Gurobi-EE3524)

本專案建構 AI 決策支援流程，整合風險預測、模型解釋與最佳化決策，將院外心臟驟停（OHCA）風險分佈模型自動轉化為 AED 資源配置方案，協助決策單位提升資源配置效率，最終使 AED 覆蓋率提升 27%。

導入可解釋 AI 與模型診斷流程，運用 SHAP、LIME 等方法分析黑盒模型，建立類 Root Cause Analysis（RCA）的特徵歸因流程，用於識別高風險因素、模型盲點與資料偏差，提升模型在真實場景中的可解釋性與穩健性。

與深圳市及宜興市政府合作完成跨城市 PoC 驗證與部署策略規劃，將模型輸出轉化為可執行的公共安全決策，服務覆蓋約 1,868 萬名居民。

Builds an AI-driven decision support workflow that integrates risk prediction, explainable AI, and optimization to automatically convert OHCA risk prediction results into AED deployment strategies, improving resource allocation efficiency and increasing AED coverage by 27%.

Introduces explainable AI and model diagnosis using SHAP and LIME to build an RCA-like feature attribution workflow for identifying key risk factors, model blind spots, and data bias, improving model interpretability and robustness.

Collaborates with Shenzhen and Yixing governments to conduct cross-city PoC validation and deployment planning, transforming AI model outputs into actionable public safety decisions serving approximately 18.68 million residents.

![OHCA risk prediction, explainability, and AED optimization workflow](https://github.com/giraffeiscute/ai-decision-automation-framework/blob/main/read_me_chart/Flow_chart.png)

## 專案定位｜Project Positioning

這不是單一預測模型，而是一個「資料處理 → 風險建模 → 模型診斷 → 決策分數轉換 → 整數規劃 → 成效評估」的 Learn-Then-Optimize 工程流程。公開實作以 Virginia Beach 的 OpenStreetMap 與 OHCA 案例資料驗證核心方法；深圳與宜興的跨城市 PoC、政府協作及部署規劃屬於專案落地經驗，受限於資料治理與合作規範，城市原始資料與部署設定未公開。

This is not a standalone prediction model. It is a Learn-Then-Optimize engineering workflow spanning data processing, risk modeling, model diagnosis, decision-score transformation, integer programming, and outcome evaluation. The public implementation validates the core method with Virginia Beach OpenStreetMap and OHCA case data. Shenzhen and Yixing cross-city PoCs, government collaboration, and deployment planning are project-delivery experience; city-specific source data and deployment configurations are not published because of data-governance and partnership constraints.

## 系統架構｜System Architecture

```text
OpenStreetMap POI / building data + historical OHCA records
                            │
                            ▼
Coordinate parsing → deduplication → H3 level-7 spatial aggregation
                            │
                            ▼
Spatial train/test split (east/west holdout at longitude -76.05)
                            │
             ┌──────────────┼──────────────┐
             ▼              ▼              ▼
          XGBoost       PyTorch MLP      SVR / Poisson
             │              │              │
             └──────────────┼──────────────┘
                            ▼
       SHAP + LIME + representative explanation selection
                            │
                            ▼
Feature attribution → per-POI risk score → coverage-area weighting
                            │
                            ▼
 Gurobi binary integer program: maximize risk score under budget and
                 candidate-pair spacing constraints
                            │
                            ▼
Deployment coordinates + OHCA coverage + survival proxy + sensitivity
```

各階段以明確的 CSV／NumPy 資料契約連接：H3 特徵矩陣提供模型訓練，`total_score` 與 `total_score_mlp` 提供最佳化目標權重，候選集合與距離指標矩陣提供空間限制，部署結果則輸出為 `.npy` 供比較、敏感度分析與地圖視覺化。

Stages are connected through explicit CSV/NumPy contracts: an H3 feature matrix feeds model training; `total_score` and `total_score_mlp` provide optimization weights; candidate sets and distance-indicator matrices encode spatial constraints; and deployment arrays are persisted as `.npy` files for comparison, sensitivity analysis, and map visualization.

## 端到端工作流程｜End-to-End Workflow

### 1. 地理資料處理｜Geospatial Data Processing

- 解析 OpenStreetMap amenity 與 building 幾何欄位，統一為 `lat`、`lon` 與 building/POI 類別。
- 將 POI、建築與 OHCA 座標映射至 H3 resolution 7 網格。
- 依 H3 cell 聚合各類 POI／建築數量，並將去重後的 OHCA 筆數作為監督式學習目標。
- 實作資料包含 177 個 H3 cells、115 個原始地理特徵欄位、1,983 筆去重 OHCA 紀錄與 138,086 筆 POI／建築紀錄；建模前移除 `commercial;yes` 欄位。

- Parses OpenStreetMap amenity/building geometries into normalized `lat`, `lon`, and category fields.
- Maps POIs, buildings, and OHCA coordinates to H3 resolution-7 cells.
- Aggregates POI/building counts by H3 cell and uses deduplicated OHCA counts as the supervised target.
- The implementation data contains 177 H3 cells, 115 raw geographic feature columns, 1,983 deduplicated OHCA records, and 138,086 POI/building records; `commercial;yes` is removed before modeling.

### 2. 風險預測流程｜Risk Prediction Pipeline

- 以經度 `-76.05` 建立空間 holdout：東側 83 個 H3 cells 作為訓練集，西側 94 個 cells 作為測試集，避免只以隨機切分評估空間泛化。
- 使用 min-max normalization 處理 H3 特徵矩陣，並在預測後還原 OHCA 數量尺度。
- 實作並比較 XGBoost regressor、兩層 ReLU PyTorch MLP、linear SVR 與 Poisson GLM。
- XGBoost notebook 同時提供固定參數版本與 `RandomizedSearchCV`（100 組搜尋、5-fold CV、MAE scoring）；MLP 使用 Adam、MSE、L2 weight decay 與 5,000 次迭代。
- 以 MAE、R²、實際值／預測值散點圖與 H3 heatmap 檢查模型品質。

- Creates a spatial holdout at longitude `-76.05`: 83 eastern H3 cells for training and 94 western cells for testing, avoiding a purely random spatial-generalization estimate.
- Applies min-max normalization to the H3 feature matrix and restores predictions to the original OHCA-count scale.
- Implements and compares an XGBoost regressor, a two-hidden-layer ReLU PyTorch MLP, linear SVR, and Poisson GLM.
- The XGBoost notebooks include fixed-parameter and `RandomizedSearchCV` variants (100 trials, 5-fold CV, MAE scoring); the MLP uses Adam, MSE, L2 weight decay, and 5,000 iterations.
- Evaluates models with MAE, R², actual-versus-predicted plots, and H3 heatmaps.

### 3. 可解釋 AI 與模型診斷｜Explainable AI & Model Diagnosis

- XGBoost 使用 `shap.TreeExplainer` 產生 global summary、interaction value、normalized importance，並驗證 SHAP additive reconstruction 是否逼近模型輸出。
- PyTorch MLP 使用 `shap.GradientExplainer`；SVR 使用 `shap.KernelExplainer`，讓不同模型共享相同的特徵歸因檢查方式。
- LIME 對單一 H3 cell 產生 local explanation；自訂 greedy submodular-pick 流程從所有 local explanations 中選出具代表性的網格，支援跨區域模型診斷。
- RCA-like 診斷將建築／設施類別（例如 apartments、house、hospital、pharmacy、clinic、fire station）連回預測貢獻，用於檢查高風險因素、模型盲點與局部異常。
- `summary.ipynb` 比較 train/test feature distributions，實作 Wilcoxon signed-rank test、cosine similarity 與 Jensen-Shannon distance，補足資料偏移診斷。

- XGBoost uses `shap.TreeExplainer` for global summaries, interaction values, normalized importance, and additive-reconstruction checks against model output.
- The PyTorch MLP uses `shap.GradientExplainer`, while SVR uses `shap.KernelExplainer`, providing a common attribution workflow across model families.
- LIME produces local explanations for individual H3 cells; a custom greedy submodular-pick routine selects representative grids from all local explanations for regional diagnosis.
- The RCA-like workflow maps building/facility categories—such as apartments, houses, hospitals, pharmacies, clinics, and fire stations—back to prediction contributions to inspect risk factors, blind spots, and local anomalies.
- `summary.ipynb` compares train/test feature distributions with the Wilcoxon signed-rank test, cosine similarity, and Jensen-Shannon distance to diagnose data shift.

### 4. 解釋結果到決策分數｜Attribution-to-Decision Scoring

- 將 cell-level SHAP value 除以對應類別數量，轉換為每一類 POI／建築的 attribution score；若類別數為 0，保留原 attribution，避免除以零。
- 依 `(h3_l7, building)` 將分數回填至測試區域的 99,724 個候選位置。
- 使用 Haversine 距離、H3 中心點快取與兩圓交集面積，將候選 AED 周邊 1.21 km 範圍內的 attribution 加權聚合為 `total_score`。
- 同時產生直接由模型風險預測聚合而成的 `total_score_mlp`，支援 Explain-Then-Optimize 與 Predict-Then-Optimize 的對照實驗。

- Converts cell-level SHAP values into per-category POI/building attribution scores by dividing by category counts; when a count is zero, the attribution is retained to avoid division by zero.
- Joins scores back to 99,724 test-region candidate locations by `(h3_l7, building)`.
- Uses Haversine distance, cached H3 centers, and circle-intersection area to aggregate attribution within a 1.21 km candidate-AED neighborhood into `total_score`.
- Also produces `total_score_mlp` from directly aggregated model predictions, enabling Explain-Then-Optimize versus Predict-Then-Optimize experiments.

### 5. AED 最佳化決策｜AED Optimization Pipeline

對候選位置集合 \(C\) 建立二元變數 \(x_i\)，以風險分數 \(s_i\) 最大化部署價值：

\[
\max \sum_{i \in C} s_i x_i
\]

subject to：

\[
\sum_{i \in C} x_i \le N, \qquad x_i \in \{0,1\}
\]

預先計算的 pairwise indicator matrix 會標記不可同時選取的候選點，`ModelBuilder_20250720.py` 對每一組標記 pair 加入 \(x_i + x_j \le 1\)。Gurobi 模型啟用 presolve、保留 solver log，並設定三小時 time limit。實驗批次測試：

- AED 數量 \(N \in \{5,10,20,40,60,80,100\}\)
- 候選位置：每組從 99,724 個位置中取 5,000 個，共 10 組
- 間距情境：`0`, `0.6`, `0.8`, `0.96`, `1.0`, `1.2`, `1.4`, `1.6` km
- 目標函數：SHAP-guided `total_score` 或 prediction-only `total_score_mlp`

For candidate set \(C\), the optimizer creates binary variables \(x_i\) and maximizes deployment value using risk score \(s_i\):

\[
\max \sum_{i \in C} s_i x_i
\]

subject to:

\[
\sum_{i \in C} x_i \le N, \qquad x_i \in \{0,1\}
\]

A precomputed pairwise indicator matrix marks candidate pairs that cannot be selected together; `ModelBuilder_20250720.py` adds \(x_i + x_j \le 1\) for every marked pair. The Gurobi model enables presolve and solver logging and applies a three-hour time limit. Batch experiments cover:

- AED budgets \(N \in \{5,10,20,40,60,80,100\}\)
- Ten candidate sets, each sampling 5,000 locations from 99,724 test-region positions
- Spacing scenarios of `0`, `0.6`, `0.8`, `0.96`, `1.0`, `1.2`, `1.4`, and `1.6` km
- SHAP-guided `total_score` and prediction-only `total_score_mlp` objectives

### 6. 決策評估與視覺化｜Decision Evaluation & Visualization

- 以 Haversine distance 計算每筆 OHCA 到最近部署 AED 的距離。
- 以 0.96 km 判斷 OHCA coverage；修正版生存率流程假設移動速度 300 m/min，超過四分鐘則設為 0，四分鐘內套用 logistic survival proxy。
- 每組候選集合執行 10 次 random-placement baseline，輸出 coverage、全部個案平均生存率與 non-zero survival average。
- 比較 SHAP-guided、prediction-only 與 random placement，並對 AED 數量、距離限制及模型來源執行 sensitivity analysis。
- 使用 Matplotlib、Seaborn、GeoPandas、Shapely、OSMnx 與 H3 polygon 繪製 OHCA density、預測 heatmap、候選位置及最終部署點。

- Computes the Haversine distance from each OHCA event to its nearest deployed AED.
- Defines OHCA coverage at 0.96 km; the corrected survival workflow assumes 300 m/min travel speed, assigns zero beyond four minutes, and applies a logistic survival proxy within four minutes.
- Runs ten random-placement baselines per candidate set and persists coverage, mean survival across all events, and non-zero survival averages.
- Compares SHAP-guided, prediction-only, and random placement across AED budget, spacing constraint, and source-model sensitivity.
- Uses Matplotlib, Seaborn, GeoPandas, Shapely, OSMnx, and H3 polygons to render OHCA density, prediction heatmaps, candidate sites, and selected deployments.

## 核心模組｜Core Modules

| 路徑 / Path | 中文職責 | Engineering responsibility |
|---|---|---|
| `notebook/OpenStreetMap/VB_amenity.ipynb` | OSM 幾何資料檢查與座標解析 | OSM geometry inspection and coordinate extraction |
| `notebook/h3_l7_df process.ipynb` | POI／建築與 OHCA 去重、H3 L7 聚合 | POI/building aggregation, OHCA deduplication, H3 feature-table construction |
| `notebook/ML models.ipynb` | XGBoost 調參、固定參數模型與 SVR 比較 | XGBoost search/fixed configurations and SVR comparison |
| `notebook/XGB.ipynb`, `NN.ipynb`, `SVM.ipynb`, `poisson regression.ipynb` | 多模型訓練、指標與空間預測圖 | Multi-model training, metrics, and spatial prediction maps |
| `notebook/XGB SHAP.ipynb`, `NN shap value*.ipynb`, `SVM SHAP.ipynb` | 多模型 SHAP 解釋與 additive check | Cross-model SHAP analysis and additive checks |
| `notebook/XGB_LIME SP.ipynb`, `NN_LIME SP.ipynb`, `LIME integrate.ipynb` | Local explanation、代表性網格選取與特徵縮減實驗 | Local explanations, representative-grid selection, and feature-reduction experiments |
| `notebook/summary.ipynb` | Train/test 統計與資料分佈偏移診斷 | Dataset statistics and train/test shift diagnostics |
| `Optimizor/Data_20250720.py` | 讀取候選分數、OHCA 座標與距離矩陣；Haversine 工具 | Loads candidate scores, OHCA coordinates, distance matrices, and geodesic utilities |
| `Optimizor/ModelBuilder_20250720.py` | Gurobi binary IP 建模與 SHAP／prediction objectives | Gurobi binary-IP construction for SHAP-guided and prediction-only objectives |
| `Optimizor/run_this_20250720.py` | 多預算、多間距批次最佳化與 coverage/survival 評估 | Batch optimization and coverage/survival evaluation across budgets and spacing |
| `Optimizor/run_this_random_*.py` | 隨機部署基準與修正版 survival 指標 | Random-placement baselines and corrected survival metrics |
| `Optimizor/run_this_*_R+_20251003.py` | 重新計算 XGBoost／MLP 部署結果的平均與 non-zero survival | Re-evaluates XGBoost/MLP decisions with mean and non-zero survival metrics |
| `Optimizor/difference table *.ipynb`, `draw_fig_2.ipynb` | 結果矩陣、相對差異、敏感度分析與圖表 | Result matrices, relative comparisons, sensitivity analysis, and reporting plots |
| `notebook/geo graph.ipynb` | 將 H3 風險與部署決策疊加至地圖 | Overlays H3 risk and deployment decisions on maps |

## 資料契約｜Data Contracts

| Artifact | 必要欄位 / Required fields | 用途 / Purpose |
|---|---|---|
| H3 feature table | `id`, POI/building count columns, `ohca` | 模型訓練與空間 holdout / model training and spatial holdout |
| OHCA event table | `Latitude`, `Longitude` | coverage 與 survival 評估 / coverage and survival evaluation |
| Optimizer candidate table | `lat`, `lon`, `h3_l7`, `building`, `total_score`, `total_score_mlp` | 候選位置與兩種最佳化目標 / candidate locations and two optimization objectives |
| Candidate-set array | shape `(10, 5000)` | 固定重複實驗的候選 ID / candidate IDs for repeated experiments |
| Pairwise indicator array | Boolean matrix indexed by candidate IDs | 標記不可同時部署的候選 pair / marks mutually incompatible candidate pairs |
| Deployment outputs | selected IDs, coverage arrays, survival arrays | 比較、敏感度分析與地圖輸出 / comparison, sensitivity analysis, and mapping |

大型原始資料、99,724 × 99,724 距離指標矩陣（約 9.9 GB）、候選分數 CSV 與批次結果不納入目前 Git 追蹤；公開版本保留 notebook、最佳化程式與資料介面。這個邊界避免將敏感／大型衍生資料放入原始碼倉庫，但也代表完整重跑前必須先準備相容的資料 artifacts。

Large raw datasets, the 99,724 × 99,724 distance-indicator matrix (about 9.9 GB), candidate-score CSVs, and batch results are not tracked by the current Git configuration. The public repository retains notebooks, optimization code, and their data interfaces. This boundary keeps sensitive and large derived artifacts out of source control, but compatible artifacts must be prepared before a full rerun.

## 實驗與專案成果｜Results & Project Impact

### 公開實作基準｜Public Implementation Benchmark

| 指標 / Metric | 結果 / Result |
|---|---:|
| PyTorch MLP spatial holdout | Test R² `0.761`, MAE `5.539` |
| XGBoost spatial holdout | Test R² `0.752`, MAE `5.750` |
| Linear SVR spatial holdout | Test R² `0.731`, MAE `5.911` |
| SHAP-guided deployment, `N=100`, `d_min=1.2 km` | Mean covered OHCA `1,385.9 / 1,983` (`69.9%`) |
| Random placement, `N=100` | Mean covered OHCA `1,088.63 / 1,983` (`54.9%`) |
| Coverage uplift | `+27.3%` relative (`+15.0` percentage points) |
| Mean survival proxy | SHAP-guided `14.8%` vs random `12.6%` (about `+17.6%` relative) |

在已儲存的 sensitivity analysis 中，`N=100` 且 `d_min=1.2 km` 的 XGBoost SHAP-guided 方案覆蓋 1,385.9 筆 OHCA，隨機部署平均覆蓋 1,088.63 筆，對應 27.3% 相對提升。測試範圍 `0.6–1.2 km` 內，1.2 km 同時得到最高 coverage 與全部個案平均 survival proxy；修正版 non-zero survival 評估則另外衡量已進入四分鐘服務範圍的個案。

In the persisted sensitivity analysis, the XGBoost SHAP-guided configuration at `N=100` and `d_min=1.2 km` covers 1,385.9 OHCA events versus a random-placement mean of 1,088.63, a 27.3% relative uplift. Within the evaluated `0.6–1.2 km` range, 1.2 km also produces the highest coverage and all-event mean survival proxy; the corrected non-zero survival evaluation separately measures events reached within the four-minute service window.

### 跨城市 PoC 與部署｜Cross-City PoC & Deployment

- 與深圳市、宜興市政府合作進行跨城市 PoC 驗證與部署策略規劃。
- 將模型風險圖、XAI 特徵歸因與最佳化座標轉換為可供公共安全單位評估的部署方案。
- 專案服務情境合計覆蓋約 1,868 萬居民。
- 城市移植流程沿用程式中的模組邊界：替換 OSM／OHCA data adapter、重新建立 H3 特徵、執行空間 holdout 與 XAI 診斷、產生當地候選點與限制矩陣，再比較既有配置、隨機基準與最佳化方案。

- Conducted cross-city PoC validation and deployment-strategy planning with Shenzhen and Yixing governments.
- Converted model risk maps, XAI attributions, and optimized coordinates into deployment plans that public-safety teams could evaluate.
- The combined project-service context covers approximately 18.68 million residents.
- City transfer follows the implementation’s module boundaries: replace the OSM/OHCA data adapter, rebuild H3 features, run spatial holdout and XAI diagnostics, generate local candidates and constraint matrices, then compare current placement, random baselines, and optimized strategies.

## 專案結構｜Project Structure

```text
ai-decision-automation-framework/
├── README.md
├── read_me_chart/
│   └── Flow_chart.png
├── notebook/
│   ├── OpenStreetMap/
│   │   └── VB_amenity.ipynb
│   ├── h3_l7_df process.ipynb
│   ├── ML models.ipynb
│   ├── XGB.ipynb / XGB SHAP.ipynb / XGB_LIME SP.ipynb
│   ├── NN.ipynb / NN shap value*.ipynb / NN_LIME SP.ipynb
│   ├── SVM.ipynb / SVM SHAP.ipynb
│   ├── poisson regression.ipynb
│   ├── LIME integrate.ipynb
│   ├── summary.ipynb
│   └── geo graph.ipynb
└── Optimizor/
    ├── Data_20250720.py
    ├── Data_without_dmin_20251003.py
    ├── ModelBuilder_20250720.py
    ├── run_this_20250720.py
    ├── run_this_MLP_only_20250720.py
    ├── run_this_random_*.py
    ├── run_this_mlp_R+_20251003.py
    ├── run_this_xgb_R+_20251003.py
    ├── difference table coverage.ipynb
    ├── difference table survial.ipynb
    └── draw_fig_2.ipynb
```

`notebook/` 負責資料工程、模型與 XAI；`Optimizor/` 負責決策模型、批次實驗與 KPI 評估。模型輸出與最佳化輸入透過欄位契約解耦，因此可替換風險模型而不必重寫 Gurobi model builder。

`notebook/` owns data engineering, modeling, and XAI; `Optimizor/` owns decision modeling, batch experiments, and KPI evaluation. Model outputs and optimization inputs are decoupled by field contracts, so the risk model can be replaced without rewriting the Gurobi model builder.

## 執行環境與重現方式｜Environment & Reproduction

### 主要技術｜Technologies

| Layer | Technologies actually used |
|---|---|
| Data processing | Python, NumPy, pandas, regex |
| Geospatial | H3, OSMnx, GeoPandas, Shapely |
| Machine learning | XGBoost, PyTorch, scikit-learn, statsmodels |
| Explainability | SHAP, LIME, custom greedy representative-explanation selection |
| Optimization | Gurobi (`gurobipy`), binary integer programming |
| Evaluation | MAE, R², coverage, survival proxy, Wilcoxon, cosine similarity, Jensen-Shannon distance |
| Visualization | Matplotlib, Seaborn, H3 polygons |
| Artifact interface | CSV, NumPy `.npy`, Excel comparison tables |

### 安裝｜Installation

本倉庫目前沒有 pinned `requirements.txt`、package metadata、CLI 或容器設定；實作以 Jupyter notebooks 與 Python batch scripts 為主。建議使用獨立環境，並安裝：

The repository currently has no pinned `requirements.txt`, package metadata, CLI, or container definition; the implementation is organized as Jupyter notebooks and Python batch scripts. Use an isolated environment and install:

```bash
pip install jupyter numpy pandas matplotlib seaborn scipy scikit-learn \
  statsmodels xgboost torch shap lime h3 geopandas shapely osmnx tqdm openpyxl gurobipy
```

> Notebook 使用 `h3.geo_to_h3`、`h3.h3_to_geo` 與 `h3.h3_to_geo_boundary`，對應 h3-py 3.x API；新版 h3-py 4.x 需改用重新命名後的 API。執行最佳化亦需要有效的 Gurobi license。
> The notebooks use the h3-py 3.x API (`geo_to_h3`, `h3_to_geo`, and `h3_to_geo_boundary`); h3-py 4.x requires the renamed API. Optimization also requires a valid Gurobi license.

### 執行順序｜Execution Order

1. 將 OpenStreetMap features 與 OHCA event data 放到 notebooks 使用的相對路徑。
   Place OpenStreetMap features and OHCA event data at the relative paths referenced by the notebooks.

2. 執行 `h3_l7_df process.ipynb`，產生 H3 feature table 與去重 OHCA table。
   Run `h3_l7_df process.ipynb` to produce the H3 feature table and deduplicated OHCA table.

3. 執行 XGBoost／MLP／SVM notebooks，接著執行對應 SHAP 與 LIME notebooks。
   Run the XGBoost/MLP/SVM notebooks, followed by the corresponding SHAP and LIME notebooks.

4. 依資料契約建立 optimizer candidate CSV，至少包含 `lat`, `lon`, `total_score`, `total_score_mlp`；同時準備候選集合與對應距離 indicator matrices。
   Build the optimizer candidate CSV with at least `lat`, `lon`, `total_score`, and `total_score_mlp`, and prepare candidate-set and distance-indicator arrays.

5. 從 `Optimizor/` 工作目錄執行批次最佳化；腳本使用相對路徑，輸出資料夾需預先建立。
   Run batch optimization from the `Optimizor/` working directory; scripts use relative paths and expect output directories to exist.

```bash
cd Optimizor
python run_this_20250720.py
python run_this_random_R+.py
python run_this_mlp_R+_20251003.py
python run_this_xgb_R+_20251003.py
```

6. 使用 `difference table coverage.ipynb`、`difference table survial.ipynb` 與 `draw_fig_2.ipynb` 彙整 coverage、survival 與 sensitivity results。
   Use the difference-table and plotting notebooks to aggregate coverage, survival, and sensitivity results.

目前流程具備明確的模組與 artifact boundary，但仍是研究／PoC 等級的工程實作，不是可直接部署的服務：完整自動化仍需補上版本鎖定、設定檔、資料驗證、CLI／workflow orchestration、測試、模型 registry 與監控。這些邊界在此明確列出，避免將 notebook 實驗環境誤述為 production service。

The workflow has clear module and artifact boundaries, but remains a research/PoC engineering implementation rather than a directly deployable service. Full production automation would still require dependency locking, configuration files, data validation, CLI/workflow orchestration, tests, a model registry, and monitoring. These boundaries are explicit so the notebook environment is not misrepresented as a production service.

## 研究論文與工程實作｜Research Paper vs Engineering Implementation

### 研究論文｜Research Paper

論文定義 Learn-Then-Optimize 方法、SHAP-guided Integer Programming（SIP）模型與實驗設計；若引用研究方法或結果，請引用論文。

The paper defines the Learn-Then-Optimize method, SHAP-guided Integer Programming (SIP) formulation, and experimental design. Cite the paper when referencing the research method or findings.

### 工程實作｜Engineering Implementation (This Repository)

本倉庫提供該方法的 notebook 與 Python 實作，包括 OSM/H3 資料處理、多模型風險預測、SHAP/LIME 診斷、Gurobi 決策模型、random baseline、survival proxy、敏感度分析與地理視覺化。若延伸程式碼，請同時保留對本倉庫與論文的來源說明。

This repository provides the notebook and Python implementation: OSM/H3 processing, multi-model risk prediction, SHAP/LIME diagnostics, Gurobi decision modeling, random baselines, survival proxies, sensitivity analysis, and geospatial visualization. If extending the code, retain attribution to both this repository and the paper.

## 引用方式｜Citation

```bibtex
@article{yang2025public,
  title={Public Access Defibrillator Deployment for Cardiac Arrests: A Learn-Then-Optimize Approach with SHAP-based Interpretable Analytics},
  author={Yang, Chih-Yuan and Leong, Keng-Hou and Cao, Kexin and Yang, Mingchuan and Chan, Wai Kin},
  journal={arXiv preprint arXiv:2501.00819},
  year={2025}
}
```

Yang, C.-Y., Leong, K.-H., Cao, K., Yang, M., & Chan, W. K. (2025). *Public Access Defibrillator Deployment for Cardiac Arrests: A Learn-Then-Optimize Approach with SHAP-based Interpretable Analytics*. arXiv:2501.00819.

[完整文章 arXiv 連結｜Paper on arXiv](https://arxiv.org/abs/2501.00819)
