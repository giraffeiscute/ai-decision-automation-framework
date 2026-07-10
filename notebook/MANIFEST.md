# Notebook File Manifest

This manifest records the planned and completed relocation of files from the former mixed `notebook/` layout.

| Current path | Planned path | Category | Reproducibility | Notes |
| --- | --- | --- | --- | --- |
| `notebook/OHCAs.csv` | `notebook/data/raw/ohca/OHCAs.csv` | raw | preserve | Original OHCA source data. |
| `notebook/OpenStreetMap/A_features.csv` | `notebook/data/raw/osm/A_features.csv` | raw | preserve | Original OSM amenity/features source data. |
| `notebook/OpenStreetMap/B_features.csv` | `notebook/data/raw/osm/B_features.csv` | raw | preserve | Large original OSM building/features source data. |
| `notebook/ohca_df.csv` | `notebook/data/interim/ohca_df.csv` | interim | reproducible | Cleaned OHCA intermediate data. |
| `notebook/poi_df.csv` | `notebook/data/interim/poi_df.csv` | interim | unknown | POI intermediate data. |
| `notebook/h3_l7_df_new.csv` | `notebook/data/processed/h3_l7_df_new.csv` | processed | reproducible | Main model-ready H3 dataset. |
| `notebook/test_poi_df_total.csv` | `notebook/data/processed/test_poi_df_total.csv` | processed | unknown | Model/geographic test dataset. |
| `notebook/test_poi_df_NNtotal.csv` | `notebook/data/processed/test_poi_df_NNtotal.csv` | processed | unknown | NN test dataset. |
| `notebook/test_poi_df_XGBtotal.csv` | `notebook/data/processed/test_poi_df_XGBtotal.csv` | processed | unknown | XGB test dataset. |
| `notebook/distinct_nonzero_counts.csv` | `notebook/data/summaries/distinct_nonzero_counts.csv` | summary | reproducible | Derived summary table. |
| `notebook/Shap_value_all_building_v1.pdf` | `notebook/outputs/figures/shap/Shap_value_all_building_v1.pdf` | output | reproducible | SHAP figure. |
| `notebook/Shap_value_MLP_10_v1.pdf` | `notebook/outputs/figures/shap/Shap_value_MLP_10_v1.pdf` | output | reproducible | SHAP figure. |
| `notebook/Shap_value_MLP_11-20_v1.pdf` | `notebook/outputs/figures/shap/Shap_value_MLP_11-20_v1.pdf` | output | reproducible | SHAP figure. |
| `notebook/Shap_value_MLP_25_v2.pdf` | `notebook/outputs/figures/shap/Shap_value_MLP_25_v2.pdf` | output | reproducible | SHAP figure. |
| `notebook/Shap_value_POI_38_v1.pdf` | `notebook/outputs/figures/shap/Shap_value_POI_38_v1.pdf` | output | reproducible | SHAP figure. |
| `notebook/Shap_value_POI_39-78_v1.pdf` | `notebook/outputs/figures/shap/Shap_value_POI_39-78_v1.pdf` | output | reproducible | SHAP figure. |
| `notebook/cache/10f77de7c1cda1f177f424a79e2bd7ef6646285b.json` | `notebook/cache/10f77de7c1cda1f177f424a79e2bd7ef6646285b.json` | cache | reproducible | Existing cache file remains in place. |
| `notebook/h3_l7_df process.ipynb` | `notebook/notebooks/01_data_prep/h3_l7_df process.ipynb` | notebook | preserve | Data preparation notebook. |
| `notebook/OpenStreetMap/VB_amenity.ipynb` | `notebook/notebooks/01_data_prep/VB_amenity.ipynb` | notebook | preserve | OSM amenity processing notebook. |
| `notebook/geo graph.ipynb` | `notebook/notebooks/02_eda_geo/geo graph.ipynb` | notebook | preserve | Geographic visualization notebook. |
| `notebook/summary.ipynb` | `notebook/notebooks/02_eda_geo/summary.ipynb` | notebook | preserve | Summary notebook. |
| `notebook/ML models.ipynb` | `notebook/notebooks/03_modeling/ML models.ipynb` | notebook | preserve | General model notebook. |
| `notebook/poisson regression.ipynb` | `notebook/notebooks/03_modeling/poisson regression.ipynb` | notebook | preserve | Poisson regression notebook. |
| `notebook/NN.ipynb` | `notebook/notebooks/03_modeling/NN.ipynb` | notebook | preserve | Neural network notebook. |
| `notebook/SVM.ipynb` | `notebook/notebooks/03_modeling/SVM.ipynb` | notebook | preserve | SVM notebook. |
| `notebook/XGB.ipynb` | `notebook/notebooks/03_modeling/XGB.ipynb` | notebook | preserve | XGBoost notebook. |
| `notebook/LIME integrate.ipynb` | `notebook/notebooks/04_interpretability/LIME integrate.ipynb` | notebook | preserve | LIME integration notebook. |
| `notebook/NN_LIME SP.ipynb` | `notebook/notebooks/04_interpretability/NN_LIME SP.ipynb` | notebook | preserve | NN LIME notebook. |
| `notebook/XGB_LIME SP.ipynb` | `notebook/notebooks/04_interpretability/XGB_LIME SP.ipynb` | notebook | preserve | XGB LIME notebook. |
| `notebook/NN shap value.ipynb` | `notebook/notebooks/04_interpretability/NN shap value.ipynb` | notebook | preserve | NN SHAP notebook. |
| `notebook/NN shap value other test.ipynb` | `notebook/notebooks/04_interpretability/NN shap value other test.ipynb` | notebook | preserve | NN SHAP alternate test notebook. |
| `notebook/SVM SHAP.ipynb` | `notebook/notebooks/04_interpretability/SVM SHAP.ipynb` | notebook | preserve | SVM SHAP notebook. |
| `notebook/XGB SHAP.ipynb` | `notebook/notebooks/04_interpretability/XGB SHAP.ipynb` | notebook | preserve | XGB SHAP notebook. |

## Path Impact Audit

The notebooks previously used these root-relative files from the old `notebook/` working directory:

- `h3_l7_df_new.csv` -> `../../data/processed/h3_l7_df_new.csv`
- `OHCAs.csv` -> `../../data/raw/ohca/OHCAs.csv`
- `OpenStreetMap/A_features.csv` -> `../../data/raw/osm/A_features.csv`
- `OpenStreetMap/B_features.csv` -> `../../data/raw/osm/B_features.csv`
- `test_poi_df_total.csv` -> `../../data/processed/test_poi_df_total.csv`
- `ohca_df.csv` -> `../../data/interim/ohca_df.csv`
- SHAP PDF outputs -> `../../outputs/figures/shap/...`
- LIME PDF outputs -> `../../outputs/figures/lime/...`
- Map PDF/PNG outputs -> `../../outputs/figures/maps/...`
