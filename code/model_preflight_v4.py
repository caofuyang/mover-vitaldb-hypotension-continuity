"""Audit the locked model matrix without fitting an exposure-outcome association."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import patsy

from fit_primary_model_v4 import PRIMARY_TERMS, prepare


def matrix_audit(frame: pd.DataFrame, terms: list[str]) -> dict:
    formula = " + ".join(terms)
    matrix = patsy.dmatrix(formula, frame, return_type="dataframe")
    rank = int(np.linalg.matrix_rank(matrix.to_numpy(float)))
    return {
        "rows": int(len(frame)),
        "columns": int(matrix.shape[1]),
        "rank": rank,
        "full_rank": rank == matrix.shape[1],
        "events": int(frame.early_aki_48h.sum()),
        "events_per_matrix_column": float(frame.early_aki_48h.sum() / matrix.shape[1]),
        "column_names": matrix.columns.tolist(),
        "category_counts": {
            column: {str(key): int(value) for key, value in frame[column].value_counts(dropna=False).items()}
            for column in ["center", "sex_group", "asa_group", "surgical_category"]
        },
        "category_event_counts": {
            column: {
                str(key): int(value)
                for key, value in frame.groupby(column, dropna=False).early_aki_48h.sum().items()
            }
            for column in ["center", "sex_group", "asa_group", "surgical_category"]
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    data = prepare(pd.read_csv(args.input_csv))
    numeric = ["age10", "egfr10", "anesthesia_hours", "total_low10", "excess5_10", "mean_depth5"]
    data = data[data[numeric].notna().all(axis=1)].copy()
    observed = data[data.outcome_observed_48h & data.early_aki_48h.notna()].copy()
    audits = {
        "pooled_primary": matrix_audit(observed, PRIMARY_TERMS),
        "pooled_common_surgical_support": matrix_audit(
            observed[observed.common_surgical_support.astype(bool)], PRIMARY_TERMS
        ),
        "center_specific": {},
    }
    center_terms = [term for term in PRIMARY_TERMS if term != "C(center)"]
    for center, group in observed.groupby("center"):
        audits["center_specific"][center] = matrix_audit(group, center_terms)
    result = {
        "purpose": "Locked model-matrix preflight; no exposure-outcome association fitted",
        "status": "pass" if all(
            audit["full_rank"]
            for audit in [audits["pooled_primary"], audits["pooled_common_surgical_support"], *audits["center_specific"].values()]
        ) else "fail",
        "audits": audits,
        "interpretation": {
            "pooled_primary": "Confirmatory analysis; event-per-column ratio is a diagnostic, not a selection rule.",
            "center_specific": "Secondary transportability estimates; sparse-event centers require cautious interval interpretation and are not required to be significant.",
        },
    }
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
