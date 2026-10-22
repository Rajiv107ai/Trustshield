"""Canonical temporal utilities and leakage guards for TrustShield.

Enforces the strict temporal invariance principle:
    event_time < decision_time
An event is legally observable at prediction/decision time if and only if
its timestamp is strictly prior to the decision timestamp.
"""

from __future__ import annotations
import pandas as pd
from typing import Any, cast


def is_strictly_before(event_time: Any, decision_time: Any) -> bool:
    """Canonical temporal guard: event is available at decision_time iff event_time < decision_time."""
    return bool(pd.to_datetime(event_time) < pd.to_datetime(decision_time))


def is_available(event_time: Any, decision_time: Any) -> bool:
    """Alias for is_strictly_before for consistent cross-module usage."""
    return is_strictly_before(event_time, decision_time)


def filter_historical_events(
    df: pd.DataFrame,
    timestamp_col: str,
    cutoff_time: Any,
) -> pd.DataFrame:
    """Returns rows where df[timestamp_col] < cutoff_time.

    Strict historical filtering: equality is excluded to prevent same-timestamp lookahead.
    """
    if df.empty or timestamp_col not in df.columns or cutoff_time is None:
        return df
    cutoff_ts = pd.to_datetime(cutoff_time)
    return cast(pd.DataFrame, df[pd.to_datetime(df[timestamp_col]) < cutoff_ts].copy())
