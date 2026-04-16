from typing import Dict, List

REWARD_SCHEMES = ("default", "uf_conc_pump", "fill")

DEFAULT_REWARD_SCHEMA: List[Dict[str, str]] = [
    {"key": "progress", "label": "产出", "color": "#22c55e"},
    {"key": "process", "label": "过程质量", "color": "#38bdf8"},
    {"key": "operation", "label": "运行代价", "color": "#f97316"},
    {"key": "safety", "label": "安全约束", "color": "#ef4444"},
    {"key": "terminal", "label": "终局", "color": "#eab308"},
]

UF_CONC_PUMP_SCHEMA: List[Dict[str, str]] = [
    {"key": "progress", "label": "产出", "color": "#22c55e"},
    {"key": "process", "label": "浓度引导", "color": "#38bdf8"},
    {"key": "operation", "label": "运行代价", "color": "#f97316"},
    {"key": "safety", "label": "安全约束", "color": "#ef4444"},
    {"key": "terminal", "label": "终局", "color": "#eab308"},
]

FILL_SCHEMA: List[Dict[str, str]] = [
    {"key": "process", "label": "稳定控制", "color": "#22c55e"},
    {"key": "operation", "label": "运行代价", "color": "#f97316"},
    {"key": "safety", "label": "安全约束", "color": "#ef4444"},
]

REWARD_BREAKDOWN_SCHEMAS: Dict[str, List[Dict[str, str]]] = {
    "default": DEFAULT_REWARD_SCHEMA,
    "uf_conc_pump": UF_CONC_PUMP_SCHEMA,
    "fill": FILL_SCHEMA,
}


def normalize_reward_scheme(scheme: str) -> str:
    key = (scheme or "default").strip().lower()
    return key if key in REWARD_SCHEMES else "default"


def get_reward_schema(scheme: str) -> List[Dict[str, str]]:
    key = normalize_reward_scheme(scheme)
    schema = REWARD_BREAKDOWN_SCHEMAS.get(key, DEFAULT_REWARD_SCHEMA)
    return [dict(item) for item in schema]
