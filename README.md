# ✈️ AeroPattern Free

**Open-source aircraft maintenance pattern recognition — free for anyone.**

AeroPattern Free helps airlines, MROs, students and researchers turn messy maintenance / sensor data into actionable patterns:

- **Smart Data Preparation** — map any column names, validate quality, clean automatically
- Predict **Remaining Useful Life (RUL)**
- Classify **imminent failure risk**
- Detect **anomalies**
- Discover **behavioral clusters**

No paid services. No vendor lock-in. Runs on your laptop or free cloud hosting.

---

## Why the Data Preparation step?

Real airline data almost never arrives ready for analysis. Different systems use different names (`Tail_No`, `SerialNumber`, `FC`, `FlightHours`, `EGT`, …), separators, and quality levels. Manually cleaning this in Excel is slow and error-prone.

AeroPattern Free includes a **pre-analysis pipeline** that:

1. Suggests column mappings (known aliases + fuzzy matching)
2. Lets you map arbitrary CSV columns → a standard schema
3. Validates types, missing values and basic consistency
4. Cleans and formats the data
5. Only then unlocks the analysis pages

This saves significant time and prevents garbage-in / garbage-out.

---

## Quick Start

```bash
git clone https://github.com/kiranashokthomas/aero-pattern-free.git
cd aero-pattern-free
pip install -r requirements.txt
streamlit run app.py
```

Open the URL shown in the terminal (usually http://localhost:8501).

1. Go to **🧹 Data Preparation**
2. Load the built-in sample fleet **or** upload your CSV
3. Map columns (suggestions are provided)
4. Click **Apply mapping, validate and clean**
5. When status is OK, use the analysis pages

---

## Features

| Step / Page              | What it does                                      |
|--------------------------|---------------------------------------------------|
| Data Preparation         | Upload → map → validate → clean                   |
| Explore Data             | Distributions, per-unit trends                    |
| Predict RUL              | Random Forest regression                          |
| Failure Risk             | Classify if failure is imminent                   |
| Anomaly Detection        | Isolation Forest                                  |
| Pattern Clusters         | K-Means behavioral regimes                        |
| Maintenance Log Demo     | Sample work-order style data                      |

---

## Standard schema (after mapping)

| Column                    | Required?   | Notes                                      |
|---------------------------|-------------|--------------------------------------------|
| `engine_id`               | Yes         | Unit / engine / component identifier       |
| `cycle`                   | Yes         | Operating cycle or sequential time step    |
| `sensor_1` … `sensor_N`   | Recommended | Any numeric sensor readings                |
| `operational_setting_1/2/3` | Optional  | Flight / operating conditions              |
| `RUL`                     | Optional    | Remaining Useful Life (for supervised)     |
| `failure_imminent`        | Optional    | 0/1 (auto-created from RUL if missing)     |

The models adapt to whatever sensors you map.

---

## Project structure

```
aero-pattern-free/
├── app.py                  # Streamlit application
├── requirements.txt
├── LICENSE                 # MIT
├── README.md
├── .gitignore
├── utils/
│   ├── schema.py           # Standard columns + aliases
│   ├── data_prep.py        # Mapping, validation, cleaning
│   ├── data_generator.py   # Sample fleet + work-order data
│   └── ml_models.py        # RUL, classifier, anomaly, clustering
├── data/                   # Optional place for your CSVs
└── models/                 # Optional saved models
```

---

## Free hosting

- [Streamlit Community Cloud](https://streamlit.io/cloud) — connect GitHub repo
- [Hugging Face Spaces](https://huggingface.co/spaces)
- Render / Railway free tiers
- Any laptop or VPS

---

## Contributing

Pull requests are welcome. Ideas:

- Better auto-mapping / fuzzy matching
- XGBoost / LightGBM / simple LSTM options
- SHAP explanations
- Multi-file batch upload
- Export of validation reports and predictions

---

## License

MIT License — free for commercial and non-commercial use.

---

## Disclaimer

This is an **educational and research tool**.  
It is **not** certified or approved for operational airworthiness or maintenance-release decisions.  
Always follow your approved maintenance program and applicable regulations (FAA, EASA, etc.).

---

Made to be free. Fork it, improve it, share it.
