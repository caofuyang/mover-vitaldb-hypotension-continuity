"""Locked surgical taxonomy helpers shared by both feature builders."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


RULE_FILE = Path(__file__).resolve().parent.parent / "config" / "SURGICAL_CATEGORY_RULES_v4.csv"


def load_rules(rule_file: Path = RULE_FILE) -> pd.DataFrame:
    rules = pd.read_csv(rule_file).sort_values("priority")
    rules["compiled"] = [re.compile(pattern, re.I) for pattern in rules.pattern]
    return rules


def classify(text: object, scope: str, rules: pd.DataFrame) -> str:
    value = "" if pd.isna(text) else str(text)
    for row in rules.itertuples():
        if row.scope not in {"both", scope}:
            continue
        if row.compiled.search(value):
            return row.category
    return "unmapped"


def egfr_2021(creatinine: float, age: float, female: bool) -> float:
    kappa = 0.7 if female else 0.9
    alpha = -0.241 if female else -0.302
    return (
        142
        * min(creatinine / kappa, 1) ** alpha
        * max(creatinine / kappa, 1) ** -1.2
        * 0.9938**age
        * (1.012 if female else 1)
    )
