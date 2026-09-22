# Contributing to AeroPattern Free

Thank you for helping make aircraft maintenance pattern recognition free and accessible.

## Quick start for contributors

```bash
git clone https://github.com/kiranashokthomas/aero-pattern-free.git
cd aero-pattern-free
pip install -r requirements.txt
streamlit run app.py
```

## Project layout

- `app.py` — entry point (loads `_app_p1.py` + `_app_p2.py`)
- `utils/schema.py` — standard column names and airline aliases
- `utils/data_prep.py` — mapping, validation, cleaning
- `utils/ml_models.py` — RUL, failure risk, anomaly, clustering
- `utils/data_generator.py` — sample fleet data

## Ideas welcome

- More column aliases from real MRO systems (AMOS, Trax, SAP, etc.)
- XGBoost / LightGBM / simple LSTM options
- SHAP explanations
- Multi-file batch upload
- Export of validation reports and predictions as PDF

## License

By contributing, you agree your contributions are under the MIT License.
