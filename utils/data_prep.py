"""
Data preparation pipeline for AeroPattern Free.

Handles messy real-world airline CSVs:
- Automatic column suggestions (aliases + fuzzy matching)
- Interactive mapping → standard schema
- Validation (types, missing values, consistency)
- Cleaning so data is analysis-ready
- CSV / TSV / Excel (.xlsx) support
"""

from __future__ import annotations

import io
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.schema import (
    STANDARD_COLUMNS,
    get_aliases_map,
    get_required_columns,
    SENSOR_PREFIX,
)


def _normalize(name: str) -> str:
    if not isinstance(name, str):
        name = str(name)
    name = name.strip().lower()
    name = re.sub(r"[\s\-]+", "_", name)
    name = re.sub(r"[^a-z0-9_]", "", name)
    return name


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def read_uploaded_file(file_bytes: bytes, filename: str = "") -> pd.DataFrame:
    """Read CSV/TSV/Excel from raw bytes. Tries common separators and encodings."""
    name = (filename or "").lower()

    if name.endswith((".xlsx", ".xls", ".xlsm")):
        try:
            return pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")
        except Exception:
            return pd.read_excel(io.BytesIO(file_bytes))

    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        for sep in (",", ";", "\t", "|"):
            try:
                df = pd.read_csv(io.BytesIO(file_bytes), sep=sep, encoding=encoding)
                if len(df.columns) > 1:
                    return df
            except Exception:
                continue

    return pd.read_csv(io.BytesIO(file_bytes))


def suggest_column_mapping(raw_columns: List[str]) -> Dict[str, Optional[str]]:
    """Suggest which raw column maps to each standard column."""
    aliases = get_aliases_map()
    norm_to_raw = {_normalize(c): c for c in raw_columns}
    used_raw: set = set()
    suggestions: Dict[str, Optional[str]] = {}

    for std_name in STANDARD_COLUMNS:
        best = None
        if _normalize(std_name) in norm_to_raw:
            best = norm_to_raw[_normalize(std_name)]
        else:
            for alias, target in aliases.items():
                if target == std_name and alias in norm_to_raw:
                    best = norm_to_raw[alias]
                    break
        if best and best not in used_raw:
            suggestions[std_name] = best
            used_raw.add(best)
        else:
            suggestions[std_name] = None

    for std_name, current in list(suggestions.items()):
        if current is not None:
            continue
        best_raw = None
        best_score = 0.55
        for raw in raw_columns:
            if raw in used_raw:
                continue
            score = _similarity(_normalize(std_name), _normalize(raw))
            for alias in STANDARD_COLUMNS[std_name]["aliases"]:
                score = max(score, _similarity(_normalize(alias), _normalize(raw)))
            if score > best_score:
                best_score = score
                best_raw = raw
        if best_raw:
            suggestions[std_name] = best_raw
            used_raw.add(best_raw)

    return suggestions


def suggest_sensor_mapping(raw_columns: List[str], already_mapped: List[str]) -> List[str]:
    """Rank remaining columns that look like sensors (aviation-oriented)."""
    candidates = []
    for col in raw_columns:
        if col in already_mapped:
            continue
        n = _normalize(col)
        score = 0.0

        if re.search(
            r"egt|n1|n2|n3|oil_?temp|oil_?press|fuel_?flow|bleed|vibration|"
            r"temp|press|rpm|sensor|accel|flow|current|voltage|torque|thrust",
            n,
        ):
            score += 0.7
        if re.search(r"sensor_?\d+|s\d+$|_\d+$", n):
            score += 0.35
        if re.match(r"^[stpfv]\d+", n):
            score += 0.25
        if re.match(r"^[a-z]{0,4}\d{1,3}$", n):
            score += 0.15
        if re.search(r"note|comment|remark|unnamed|index|id$|name$|date|time$", n):
            score -= 0.5

        if score > 0.1:
            candidates.append((score, col))

    candidates.sort(key=lambda x: (-x[0], x[1]))
    return [c for _, c in candidates]


def apply_mapping(
    df: pd.DataFrame,
    column_map: Dict[str, Optional[str]],
    sensor_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Rename columns to standard schema; keep only mapped + sensor columns."""
    rename_dict: Dict[str, str] = {}
    for std, raw in column_map.items():
        if raw and raw in df.columns:
            rename_dict[raw] = std

    if sensor_map:
        for std_sensor, raw in sensor_map.items():
            if raw and raw in df.columns:
                rename_dict[raw] = std_sensor

    out = df.rename(columns=rename_dict)
    keep = [c for c in out.columns if c in STANDARD_COLUMNS or c.startswith(SENSOR_PREFIX)]
    out = out.loc[:, ~out.columns.duplicated()]
    return out[[c for c in keep if c in out.columns]].copy()


def validate_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate prepared data. Returns status ok | warning | error."""
    report: Dict[str, Any] = {
        "status": "ok",
        "messages": [],
        "stats": {},
        "column_status": {},
    }

    if df is None or len(df) == 0:
        report["status"] = "error"
        report["messages"].append("Dataset is empty.")
        return report

    report["stats"]["n_rows"] = int(len(df))
    report["stats"]["n_cols"] = int(len(df.columns))

    required = get_required_columns()
    missing_req = [c for c in required if c not in df.columns]
    if missing_req:
        report["status"] = "error"
        report["messages"].append(f"Missing required columns: {', '.join(missing_req)}")

    for col in df.columns:
        info: Dict[str, Any] = {"present": True}
        series = df[col]
        null_pct = float(series.isna().mean() * 100)
        info["null_pct"] = round(null_pct, 1)
        info["n_unique"] = int(series.nunique(dropna=True))

        if col in STANDARD_COLUMNS:
            expected = STANDARD_COLUMNS[col]["dtype"]
        elif col.startswith(SENSOR_PREFIX):
            expected = "numeric"
        else:
            expected = "unknown"

        if expected == "numeric":
            numeric = pd.to_numeric(series, errors="coerce")
            coerced = int(numeric.isna().sum() - series.isna().sum())
            info["dtype_ok"] = coerced == 0
            info["extra_nulls_from_coerce"] = coerced
            if coerced > 0:
                report["messages"].append(
                    f"Column '{col}' has {coerced} non-numeric values (will become missing)."
                )
                if report["status"] == "ok":
                    report["status"] = "warning"
            if numeric.notna().any():
                info["min"] = float(numeric.min())
                info["max"] = float(numeric.max())
        else:
            info["dtype_ok"] = True

        if null_pct > 30:
            report["messages"].append(f"Column '{col}' has high missing rate ({null_pct:.0f}%).")
            if report["status"] == "ok":
                report["status"] = "warning"
        if null_pct > 80 and col in required:
            report["status"] = "error"
            report["messages"].append(
                f"Required column '{col}' is mostly empty ({null_pct:.0f}% missing)."
            )

        report["column_status"][col] = info

    if "cycle" in df.columns:
        cyc = pd.to_numeric(df["cycle"], errors="coerce")
        if cyc.notna().any() and (cyc < 0).any():
            report["messages"].append("Some cycle values are negative.")
            if report["status"] == "ok":
                report["status"] = "warning"

    if "RUL" in df.columns:
        rul = pd.to_numeric(df["RUL"], errors="coerce")
        if (rul < 0).any():
            report["messages"].append("Some RUL values are negative.")
            if report["status"] == "ok":
                report["status"] = "warning"

    if "engine_id" in df.columns:
        n_engines = int(df["engine_id"].nunique(dropna=True))
        report["stats"]["n_engines"] = n_engines
        if n_engines == 0:
            report["status"] = "error"
            report["messages"].append("No valid engine_id values found.")

    sensor_cols = [c for c in df.columns if c.startswith(SENSOR_PREFIX)]
    report["stats"]["n_sensors"] = len(sensor_cols)
    if len(sensor_cols) == 0:
        report["messages"].append(
            "No sensor columns detected. Anomaly detection and clustering will be limited."
        )
        if report["status"] == "ok":
            report["status"] = "warning"

    if not report["messages"]:
        report["messages"].append("All checks passed. Data looks ready for analysis.")

    return report


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Standard cleaning: types, IDs, sort, dedupe."""
    out = df.copy()
    out = out.dropna(how="all")

    if "engine_id" in out.columns:
        out["engine_id"] = out["engine_id"].astype(str).str.strip()
        out.loc[out["engine_id"].isin(["nan", "None", "NaN", ""]), "engine_id"] = np.nan

    numeric_candidates = ["cycle", "RUL", "failure_imminent"] + [
        c for c in out.columns
        if c.startswith("operational_setting") or c.startswith(SENSOR_PREFIX)
    ]
    for col in numeric_candidates:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "failure_imminent" in out.columns:
        out["failure_imminent"] = (
            out["failure_imminent"].fillna(0).astype(int).clip(0, 1)
        )

    if "engine_id" in out.columns and "cycle" in out.columns:
        out = out.sort_values(["engine_id", "cycle"])
        out = out.drop_duplicates(subset=["engine_id", "cycle"], keep="last")

    return out.reset_index(drop=True)


def create_failure_imminent_from_rul(
    df: pd.DataFrame, threshold: int = 30
) -> pd.DataFrame:
    """Create failure_imminent from RUL when missing."""
    out = df.copy()
    if "RUL" in out.columns and "failure_imminent" not in out.columns:
        out["failure_imminent"] = (
            pd.to_numeric(out["RUL"], errors="coerce") <= threshold
        ).astype(int)
    return out


def get_analysis_ready_sensors(df: pd.DataFrame) -> List[str]:
    sensors = []
    for c in df.columns:
        if not c.startswith(SENSOR_PREFIX):
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() > 0.5:
            sensors.append(c)
    return sensors
