"""
Standard schema and common column name aliases for airline maintenance data.
Helps map messy real-world CSVs into a clean format the models understand.
"""

from typing import Dict, List

# Canonical column names used by the models
STANDARD_COLUMNS = {
    "engine_id": {
        "description": "Unique identifier for the engine / unit / aircraft component",
        "required": True,
        "dtype": "string",
        "aliases": [
            "engine_id", "engineid", "engine", "unit", "unit_id", "unitid",
            "asset_id", "asset", "serial", "serial_number", "serial_no", "sn",
            "tail", "tail_number", "tail_no", "aircraft", "aircraft_id",
            "ac_id", "equipment_id", "eq_id", "component_id", "part_id",
        ],
    },
    "cycle": {
        "description": "Operating cycle / flight cycle number (or sequential time step)",
        "required": True,
        "dtype": "numeric",
        "aliases": [
            "cycle", "cycles", "flight_cycle", "flight_cycles", "fc",
            "operating_cycle", "op_cycle", "time_cycle", "cycle_number",
            "cycle_no", "flight_hours", "hours", "fh", "time", "timestep",
            "step", "sequence", "seq",
        ],
    },
    "operational_setting_1": {
        "description": "Operational setting / condition 1 (e.g. altitude, Mach, throttle)",
        "required": False,
        "dtype": "numeric",
        "aliases": [
            "operational_setting_1", "op_setting_1", "setting_1", "os_1",
            "altitude", "alt", "mach", "throttle", "power_setting",
            "condition_1", "op1", "setting1",
        ],
    },
    "operational_setting_2": {
        "description": "Operational setting / condition 2",
        "required": False,
        "dtype": "numeric",
        "aliases": [
            "operational_setting_2", "op_setting_2", "setting_2", "os_2",
            "condition_2", "op2", "setting2", "speed", "airspeed",
        ],
    },
    "operational_setting_3": {
        "description": "Operational setting / condition 3",
        "required": False,
        "dtype": "numeric",
        "aliases": [
            "operational_setting_3", "op_setting_3", "setting_3", "os_3",
            "condition_3", "op3", "setting3", "mode", "flight_mode",
        ],
    },
    "RUL": {
        "description": "Remaining Useful Life (cycles or hours until end-of-life / failure)",
        "required": False,
        "dtype": "numeric",
        "aliases": [
            "rul", "remaining_useful_life", "remaining_life", "life_remaining",
            "ttf", "time_to_failure", "cycles_to_failure", "hours_to_go",
            "remaining_cycles", "ttl",
        ],
    },
    "failure_imminent": {
        "description": "Binary flag: 1 if failure is imminent (e.g. RUL <= threshold), else 0",
        "required": False,
        "dtype": "numeric",
        "aliases": [
            "failure_imminent", "imminent_failure", "will_fail", "fail_soon",
            "failure_flag", "is_failure", "label", "target", "y",
        ],
    },
}

# Sensor columns – we accept any number; models currently expect sensor_1 … sensor_14
# but the prep layer can rename/map flexible sensor columns.
SENSOR_PREFIX = "sensor_"
MAX_SENSORS = 21  # generous limit

# Common sensor-like names seen in aviation / industrial data
SENSOR_ALIASES = [
    "sensor", "s", "temp", "temperature", "t", "press", "pressure", "p",
    "vibration", "vib", "speed", "rpm", "n1", "n2", "n3", "egt", "egt_temp",
    "oil_temp", "oil_press", "fuel_flow", "ff", "bleed", "vibration_x",
    "vibration_y", "vibration_z", "accel", "current", "voltage", "flow",
]


def get_all_standard_names() -> List[str]:
    return list(STANDARD_COLUMNS.keys())


def get_required_columns() -> List[str]:
    return [k for k, v in STANDARD_COLUMNS.items() if v["required"]]


def get_aliases_map() -> Dict[str, str]:
    """Return a flat dict: alias (lowered) → standard name."""
    mapping = {}
    for std_name, meta in STANDARD_COLUMNS.items():
        for alias in meta["aliases"]:
            mapping[alias.lower().replace(" ", "_").replace("-", "_")] = std_name
        # also map the standard name itself
        mapping[std_name.lower()] = std_name
    return mapping
