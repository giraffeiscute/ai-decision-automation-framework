# Predict and Interpret Workspace


## Directory Layout

```text
predict_and_interpret/
  README.md
  notebooks/
    01_data_prep/
    02_eda_geo/
    03_modeling/
    04_interpretability/
  data/
    raw/
      ohca/
      osm/
    interim/
    processed/
    summaries/
  outputs/
    figures/
      shap/
      lime/
      maps/
    tables/
    reports/
```

## Main Data Flow

1. Raw OHCA records and OpenStreetMap features are read from `data/raw/`.
2. Data preparation notebooks convert records to H3 level 7 cells and assemble feature counts.
3. The main model-ready table is `data/processed/h3_l7_df_new.csv`.
4. Modeling notebooks train OHCA prediction models using the H3 feature table.
5. Interpretability notebooks explain trained model behavior with SHAP, LIME, and SP-LIME.
6. Figures and reports are written under `outputs/`.

Most modeling and interpretation notebooks use the same spatial train/test split:

- Train set: H3 cells with longitude greater than `-76.05`.
- Test set: H3 cells with longitude less than or equal to `-76.05`.

## Path Convention

Notebooks are stored under `notebooks/*/`, so relative paths are written from each notebook folder:

- Read raw data with `../../data/raw/...`
- Read processed data with `../../data/processed/...`
- Save figures with `../../outputs/figures/...`
- Save tables or reports with `../../outputs/tables/...` or `../../outputs/reports/...`



## Notebook Responsibilities

| Notebook | Role |
|---|---|
| `01_data_prep/h3_l7_df process.ipynb` | Builds the H3 level 7 modeling table from OSM `A_features.csv`, `B_features.csv`, and raw OHCA records. Writes `data/interim/ohca_df.csv` and the main processed dataset `data/processed/h3_l7_df_new.csv`. |
| `01_data_prep/VB_amenity.ipynb` | Explores Virginia Beach OSM amenity data, extracts latitude/longitude from geometries, and checks amenity categories. |
| `02_eda_geo/summary.ipynb` | Produces dataset summaries for OHCA counts, building counts, feature distributions, and train/test distribution checks. |
| `02_eda_geo/geo graph.ipynb` | Creates geospatial visualizations for H3 cells, OHCA density, train/test regions, AED candidates, and deployment decisions. |
| `03_modeling/ML models.ipynb` | Combined modeling notebook for train/test setup, XGBoost hyperparameter search, and SVM comparison experiments. |
| `03_modeling/XGB.ipynb` | Trains and evaluates the selected XGBoost regression model for OHCA prediction. |
| `03_modeling/SVM.ipynb` | Trains and evaluates the SVR/SVM regression baseline. |
| `03_modeling/poisson regression.ipynb` | Trains a Poisson GLM statistical baseline and reports model summary and prediction metrics. |
| `03_modeling/NN.ipynb` | Trains and evaluates a PyTorch neural network regression model for OHCA prediction. |
| `04_interpretability/XGB SHAP.ipynb` | Uses SHAP to explain the XGBoost model and generate feature attribution plots. |
| `04_interpretability/SVM SHAP.ipynb` | Uses SHAP to explain the SVR/SVM model. |
| `04_interpretability/NN SHAP.ipynb` | Uses SHAP to explain the neural network model and compare important OSM feature categories. |
| `04_interpretability/NN shap value other test.ipynb` | Alternate NN SHAP test notebook for checking SHAP behavior under another test setup. |
| `04_interpretability/XGB_LIME SP.ipynb` | Uses LIME and SP-LIME to explain XGBoost predictions and select representative instances. |
| `04_interpretability/NN_LIME SP.ipynb` | Uses LIME and SP-LIME to explain neural network predictions and export local explanation figures. |
| `04_interpretability/LIME integrate.ipynb` | Integrated LIME experiment notebook covering XGBoost LIME, dimensionality reduction checks, and explanation consolidation. |


## Key Inputs and Outputs

| Artifact | Purpose |
|---|---|
| `data/raw/ohca/OHCAs.csv` | Raw OHCA event input. Preserve as source data. |
| `data/raw/osm/A_features.csv` | Raw OSM amenity feature input. Preserve as source data. |
| `data/raw/osm/B_features.csv` | Raw OSM building feature input. Preserve as source data. |
| `data/interim/ohca_df.csv` | OHCA records with derived H3 cells. |
| `data/processed/h3_l7_df_new.csv` | Main H3 level 7 feature table used by modeling, EDA, and interpretation notebooks. |
| `outputs/figures/maps/` | Geospatial maps and H3 visualizations. |
| `outputs/figures/shap/` | SHAP feature attribution plots. |
| `outputs/figures/lime/` | LIME and SP-LIME explanation figures. |

## Handoff Notes

- Treat files under `data/raw/` as read-only source data.
- Rebuild `h3_l7_df_new.csv` before rerunning model notebooks if raw OHCA or OSM inputs change.
- Keep model-specific notebooks paired with their interpretation notebooks: `XGB.ipynb` with `XGB SHAP.ipynb` and `XGB_LIME SP.ipynb`, `SVM.ipynb` with `SVM SHAP.ipynb`, and `NN.ipynb` with `NN SHAP.ipynb` and `NN_LIME SP.ipynb`.
