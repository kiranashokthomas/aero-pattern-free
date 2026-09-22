"""
Synthetic aircraft maintenance / sensor data generator.
Inspired by NASA C-MAPSS turbofan degradation datasets.
Completely free to use and modify.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_sample_fleet_data(
    n_engines: int = 20,
    cycles_per_engine: int = 150,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate realistic-looking run-to-failure style data for multiple engines.

    Columns:
    - engine_id
    - cycle
    - operational_setting_1,2,3
    - sensor_1 ... sensor_14  (temperatures, pressures, speeds, etc.)
    - RUL (remaining useful life in cycles)  ← target for supervised learning
    - failure_imminent (1 if RUL <= 30)
    """
    rng = np.random.default_rng(seed)
    rows = []

    for eng in range(1, n_engines + 1):
        # Each engine has a different total life
        max_life = rng.integers(120, 280)
        # Degradation start point
        degradation_start = rng.integers(40, max_life // 2)

        for cycle in range(1, max_life + 1):
            # Operational settings (altitude, speed, etc. – simplified)
            op1 = rng.normal(0.0, 0.3)
            op2 = rng.normal(0.0, 0.2)
            op3 = rng.choice([0, 1, 2])

            # Base sensor values
            base = {
                "sensor_1": 518.0 + rng.normal(0, 0.5),      # temperature
                "sensor_2": 642.0 + rng.normal(0, 1.2),
                "sensor_3": 1580.0 + rng.normal(0, 5),
                "sensor_4": 1400.0 + rng.normal(0, 8),
                "sensor_5": 14.6 + rng.normal(0, 0.05),
                "sensor_6": 21.6 + rng.normal(0, 0.1),
                "sensor_7": 553.0 + rng.normal(0, 2),
                "sensor_8": 2388.0 + rng.normal(0, 3),
                "sensor_9": 9050.0 + rng.normal(0, 20),
                "sensor_10": 1.3 + rng.normal(0, 0.01),
                "sensor_11": 47.5 + rng.normal(0, 0.3),
                "sensor_12": 521.0 + rng.normal(0, 1.5),
                "sensor_13": 2388.0 + rng.normal(0, 2),
                "sensor_14": 8130.0 + rng.normal(0, 15),
            }

            # Add progressive degradation after start point
            if cycle > degradation_start:
                progress = (cycle - degradation_start) / (max_life - degradation_start)
                # Some sensors rise, some fall with wear
                base["sensor_2"] += progress * 25 * rng.uniform(0.8, 1.2)
                base["sensor_3"] += progress * 40 * rng.uniform(0.7, 1.3)
                base["sensor_4"] += progress * 50 * rng.uniform(0.8, 1.2)
                base["sensor_7"] -= progress * 15 * rng.uniform(0.7, 1.1)
                base["sensor_11"] += progress * 8 * rng.uniform(0.9, 1.1)
                base["sensor_12"] -= progress * 12 * rng.uniform(0.8, 1.2)

            rul = max_life - cycle
            failure_imminent = 1 if rul <= 30 else 0

            row = {
                "engine_id": f"ENG-{eng:03d}",
                "cycle": cycle,
                "operational_setting_1": round(op1, 4),
                "operational_setting_2": round(op2, 4),
                "operational_setting_3": op3,
                **{k: round(v, 3) for k, v in base.items()},
                "RUL": rul,
                "failure_imminent": failure_imminent,
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def generate_maintenance_log(n_events: int = 80, seed: int = 42) -> pd.DataFrame:
    """Simple maintenance work-order style log."""
    rng = np.random.default_rng(seed)
    components = [
        "Engine Core", "High Pressure Turbine", "Low Pressure Compressor",
        "Fan Blade", "Landing Gear Actuator", "Hydraulic Pump",
        "APU", "Fuel Control Unit", "Bleed Valve"
    ]
    actions = ["Inspection", "Repair", "Replace", "Overhaul", "Clean"]
    findings = [
        "Normal wear", "Minor crack", "Excessive vibration", "Leak detected",
        "Corrosion", "Within limits", "Requires further monitoring", "Failed"
    ]

    start = datetime(2023, 1, 1)
    rows = []
    for i in range(n_events):
        days = rng.integers(0, 700)
        date = start + timedelta(days=int(days))
        rows.append({
            "work_order_id": f"WO-{1000 + i}",
            "aircraft_tail": f"N{rng.integers(100, 999)}AB",
            "component": rng.choice(components),
            "action": rng.choice(actions),
            "flight_hours_at_event": round(rng.uniform(500, 12000), 1),
            "cycles_at_event": rng.integers(200, 4500),
            "finding": rng.choice(findings),
            "downtime_hours": round(rng.exponential(8), 1),
            "date": date.strftime("%Y-%m-%d"),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_sample_fleet_data(n_engines=5, cycles_per_engine=100)
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Engines: {df['engine_id'].nunique()}")
