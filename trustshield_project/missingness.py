"""Missing value handling and indicator generation for TrustShield."""

import pandas as pd
from typing import List, Optional, Dict


def add_missingness_indicators(
    df: pd.DataFrame, columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """Add explicit binary indicator columns '{col}__missing' for columns with nulls or specified columns.

    In fraud detection, whether a field (e.g. device_id or tracking number) is missing
    is often an informative signal in itself.
    """
    out = df.copy()
    target_cols = columns if columns is not None else [c for c in out.columns if bool(out[c].isna().any())]
    for col in target_cols:
        if col in out.columns:
            out[f"{col}__missing"] = out[col].isna().astype("int8")
    return out


def audit_missingness(df: pd.DataFrame) -> Dict[str, float]:
    """Return dictionary of missingness fraction for each column with missing values."""
    null_counts = df.isna().mean()
    return {str(col): float(rate) for col, rate in null_counts.items() if rate > 0.0}
