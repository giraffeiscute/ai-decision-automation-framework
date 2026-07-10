# Notebook Workspace

This directory separates exploratory notebooks, source data, generated data, and analysis outputs.

## Directory Layout

```text
notebook/
  README.md
  MANIFEST.md
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
  cache/
```

## Categories

- `notebooks/`: Jupyter notebooks grouped by workflow stage.
- `data/raw/`: original source data that should be preserved as-is.
- `data/interim/`: intermediate cleaned or joined datasets.
- `data/processed/`: model-ready datasets used by analysis notebooks.
- `data/summaries/`: small summary tables derived from other datasets.
- `outputs/`: generated figures, tables, and reports.
- `cache/`: transient cache files.

## Notebook Groups

- `01_data_prep/`: source data processing and feature assembly.
- `02_eda_geo/`: geographic exploration and summary notebooks.
- `03_modeling/`: model training and baseline model notebooks.
- `04_interpretability/`: SHAP, LIME, and interpretation notebooks.

## Path Convention

Notebooks are stored under `notebooks/*/`, so relative paths should be written from that location:

- Read data with `../../data/...`
- Save figures with `../../outputs/figures/...`
- Use cache files with `../../cache/...`

Avoid reading or writing bare filenames from the notebook root.

## Suggested Execution Order

1. `notebooks/01_data_prep/h3_l7_df process.ipynb`
2. `notebooks/01_data_prep/VB_amenity.ipynb`
3. `notebooks/02_eda_geo/summary.ipynb`
4. `notebooks/02_eda_geo/geo graph.ipynb`
5. `notebooks/03_modeling/*.ipynb`
6. `notebooks/04_interpretability/*.ipynb`

## Notes

- Raw files such as `OHCAs.csv`, `A_features.csv`, and `B_features.csv` are treated as preserve-only inputs.
- Generated CSV and PDF files are organized by their role, not by the notebook that produced them.
- The repository `.gitignore` currently excludes CSV, PDF, and JSON files under `notebook/`; these files remain local data artifacts unless the ignore policy changes.
