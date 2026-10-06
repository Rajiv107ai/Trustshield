"""Split validation and entity isolation guards for TrustShield."""

import pandas as pd


def assert_disjoint_ids(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    id_col: str,
) -> None:
    """Verify that identity keys (e.g. order_id) never overlap across train, validation, and test splits."""
    train_ids = set(train[id_col].dropna())
    val_ids = set(validation[id_col].dropna())
    test_ids = set(test[id_col].dropna())

    pairs = [
        ("train", "validation", train_ids & val_ids),
        ("train", "test", train_ids & test_ids),
        ("validation", "test", val_ids & test_ids),
    ]

    for name1, name2, overlap in pairs:
        if overlap:
            raise AssertionError(
                f"Data leakage detected: {name1} and {name2} share {len(overlap)} IDs on '{id_col}'."
            )


def validate_temporal_boundaries(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    time_col: str = "order_date",
) -> None:
    """Verify strict chronological sequence: max(train) <= min(val) and max(val) <= min(test)."""
    train_max = pd.to_datetime(train[time_col]).max()
    val_min = pd.to_datetime(val[time_col]).min()
    val_max = pd.to_datetime(val[time_col]).max()
    test_min = pd.to_datetime(test[time_col]).min()

    if train_max > val_min:
        raise AssertionError(
            f"Temporal boundary violation: train max ({train_max}) > val min ({val_min})"
        )
    if val_max > test_min:
        raise AssertionError(
            f"Temporal boundary violation: val max ({val_max}) > test min ({test_min})"
        )
