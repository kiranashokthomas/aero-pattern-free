"""
Data preparation pipeline for AeroPattern Free.

Handles:
- Automatic column name suggestions (fuzzy + alias matching)
- Interactive mapping of arbitrary airline CSV columns → standard schema
- Validation (types, missing values, ranges, consistency)
- Cleaning & formatting so the data is analysis-ready
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.schema import (
    STANDARD_COLUMNS,
    get_aliases_map,
    get_required_columns,
    SENSOR_PREFIX,
    MAX_SENSORS,
)


def _normalize(name: str) -> str:
    """Lowercase, replace spaces/hyphens, strip non-alphanumeric (keep underscore)."""
    if not isinstance(name, str):
        name = str(name)
    name = name.strip().lower()
    name = re.sub(r"[\s\-]+", "_", name)
    name = re.sub(r"[^a-z0-9_]", "", name)
    return name


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def suggest_column_mapping(raw_columns: List[str]) -> Dict[str, Optional[str]]:
    """
    Suggest which raw column should map to each standard column.
    Returns {standard_name: best_raw_column_or_None}
    """
    aliases = get_aliases_map()
    norm_to_raw = {_normalize(c): c for c in raw_columns}
    used_raw = set()
    suggestions: Dict[str, Optional[str]] = {}

    # 1) Exact / alias match first
    for std_name in STANDARD_COLUMNS:
        best = None
        # try standard name itself
        if _normalize(std_name) in norm_to_raw:
            best = norm_to_raw[_normalize(std_name)]
        else:
            # try known aliases
            for alias, target in aliases.items():
                if target == std_name and alias in norm_to_raw:
                    best = norm_to_raw[alias]
                    break
        if best and best not in used_raw:
            suggestions[std_name] = best
            used_raw.add(best)
        else:
            suggestions[std_name] = None

    # 2) Fuzzy match for anything still missing
    for std_name, current in list(suggestions.items()):
        if current is not None:
            continue
        best_raw = None
        best_score = 0.55  # threshold
        for raw in raw_columns:
            if raw in used_raw:
                continue
            score = _similarity(_normalize(std_name), _normalize(raw))
            # also try against aliases
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
    """
    Return a list of remaining raw columns that look like sensors
    (numeric-ish names or containing sensor-like keywords), ordered by likelihood.
    """
    candidates = []
    for col in raw_columns:
        if col in already_mapped:
            continue
        n = _normalize(col)
        score = 0.0
        if re.search(r"sensor|temp|press|vib|rpm|egt|oil|fuel|bleed|accel|flow|n1|n2|n3", n):
            score += 0.6
        if re.search(r"_\d+$|s\d+$|sensor\d+", n):
            score += 0.3
        if re.match(r"^[stpfv]\d+", n):
            score += 0.2
        # pure numbers or short codes often sensors
        if re.match(r"^[a-z]{0,3}\d{1,3}$", n):
            score += 0.15
        if score > 0:
            candidates.append((score, col))
    candidates.sort(key=lambda x: -x[0])
    return [c for _, c in candidates]


def apply_mapping(
    df: pd.DataFrame,
    column_map: Dict[str, Optional[str]],
    sensor_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Rename columns according to the mapping and keep only mapped ones + sensors.
    sensor_map: {standard_sensor_name: raw_column}
    """
    rename_dict = {}
    for std, raw in column_map.items():
        if raw and raw in df.columns:
            rename_dict[raw] = std

    if sensor_map:
        for std_sensor, raw in sensor_map.items():
            if raw and raw in df.columns:
                rename_dict[raw] = std_sensor

    out = df.rename(columns=rename_dict)
    # keep only the columns we care about
    keep = [c for c in out.columns if c in STANDARD_COLUMNS or c.startswith(SENSOR_PREFIX)]
    return out[keep].copy()


def validate_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run a battery of checks. Returns a report dict with:
    - status: "ok" | "warning" | "error"
    - messages: list of human-readable issues
    - stats: basic numbers
    - column_status: per-column info
    """
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

    report["stats"]["n_rows"] = len(df)
    report["stats"]["n_cols"] = len(df.columns)

    required = get_required_columns()
    missing_req = [c for c in required if c not in df.columns]
    if missing_req:
        report["status"] = "error"
        report["messages"].append(f"Missing required columns: {', '.join(missing_req)}")

    # Per-column checks
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
            coerced_nulls = numeric.isna().sum() - series.isna().sum()
            info["dtype_ok"] = coerced_nulls == 0
            info["extra_nulls_from_coerce"] = int(coerced_nulls)
            if coerced_nulls > 0:
                report["messages"].append(
                    f"Column '{col}' has {coerced_nulls} non-numeric values that will become missing."
                )
                if report["status"] == "ok":
                    report["status"] = "warning"
            # basic range sanity (very loose)
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
            report["messages"].append(f"Required column '{col}' is mostly empty ({null_pct:.0f}% missing).")

        report["column_status"][col] = info

    # Logical checks
    if "cycle" in df.columns:
        cyc = pd.to_numeric(df["cycle"], errors="coerce")
        if cyc.notna().any() and (cyc < 0).any():
            report["messages"].append("Some cycle values are negative.")
            if report["status"] == "ok":
                report["status"] = "warning"

    if "RUL" in df.columns and "cycle" in df.columns:
        # RUL should generally decrease or stay sensible; just check non-negative
        rul = pd.to_numeric(df["RUL"], errors="coerce")
        if (rul < 0).any():
            report["messages"].append("Some RUL values are negative.")
            if report["status"] == "ok":
                report["status"] = "warning"

    if "engine_id" in df.columns:
        n_engines = df["engine_id"].nunique()
        report["stats"]["n_engines"] = int(n_engines)
        if n_engines == 0:
            report["status"] = "error"
            report["messages"].append("No valid engine_id values found.")

    sensor_cols = [c for c in df.columns if c.startswith(SENSOR_PREFIX)]
    report["stats"]["n_sensors"] = len(sensor_cols)
    if len(sensor_cols) == 0:
        report["messages"].append(
            "No sensor columns detected. Anomaly detection and pattern clustering will be limited."
        )
        if report["status"] == "ok":
            report["status"] = "warning"

    if not report["messages"]:
        report["messages"].append("All checks passed. Data looks ready for analysis.")

    return report


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply standard cleaning:
    - Drop completely empty rows
    - Convert numeric columns
    - Strip string IDs
    - Sort by engine_id + cycle when possible
    - Drop duplicate (engine_id, cycle) keeping last
    """
    out = df.copy()

    # Drop rows that are entirely NaN
    out = out.dropna(how="all")

    # engine_id as clean string
    if "engine_id" in out.columns:
        out["engine_id"] = out["engine_id"].astype(str).str.strip()
        out.loc[out["engine_id"].isin(["nan", "None", ""]), "engine_id"] = np.nan

    # Numeric conversions
    numeric_candidates = ["cycle", "RUL", "failure_imminent"] + [
        c for c in out.columns if c.startswith("operational_setting") or c.startswith(SENSOR_PREFIX)
    ]
    for col in numeric_candidates:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # failure_imminent to 0/1 if present
    if "failure_imminent" in out.columns:
        out["failure_imminent"] = out["failure_imminent"].fillna(0).astype(int).clip(0, 1)

    # Sort and deduplicate
    if "engine_id" in out.columns and "cycle" in out.columns:
        out = out.sort_values(["engine_id", "cycle"])
        out = out.drop_duplicates(subset=["engine_id", "cycle"], keep="last")

    out = out.reset_index(drop=True)
    return out


def create_failure_imminent_from_rul(df: pd.DataFrame, threshold: int = 30) -> pd.DataFrame:
    """If RUL exists but failure_imminent does not, create it."""
    out = df.copy()
    if "RUL" in out.columns and "failure_imminent" not in out.columns:
        out["failure_imminent"] = (pd.to_numeric(out["RUL"], errors="coerce") <= threshold).astype(int)
    return out


def get_analysis_ready_sensors(df: pd.DataFrame) -> List[str]:
    """Return list of sensor_* columns that are usable (mostly numeric, low missing)."""
    sensors = []
    for c in df.columns:
        if not c.startswith(SENSOR_PREFIX):
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() > 0.5:
            sensors.append(c)
    return sensors
