"""TrustShield AI — Point-in-Time Transaction History & Edge Weight Service.

Provides point-in-time historical replay and live decision-time transaction counts
for buyer-seller pairs without data leakage or static snapshot substitution.
"""

from __future__ import annotations

import bisect
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

logger = logging.getLogger(__name__)


class PointInTimeHistoryStore:
    """Manages timestamped transaction history for point-in-time graph feature calculation.

    Principles:
    1. Historical replay: count of transactions between (buyer_id, seller_id) strictly before
       the order timestamp (t_hist < t_req). Orders with identical timestamps do NOT count
       each other (matches allow_exact_matches=False). Future orders are strictly excluded.
    2. Live scoring: count of confirmed transactions for (buyer_id, seller_id) recorded prior
       to decision time t_decision. The current pending transaction is not counted.
    3. Honesty & Safe Fallback: If transaction history is unavailable or pair is unknown,
       return 0.0 with explicit provenance note; NEVER silently substitute static cumulative snapshot.
    """

    def __init__(self, orders_csv_path: Optional[str] = None):
        self.is_loaded: bool = False
        self.source_path: Optional[str] = None
        # Map: (buyer_id, seller_id) -> sorted list of pd.Timestamp
        self._history: Dict[Tuple[str, str], List[pd.Timestamp]] = {}
        # Live transaction ledger: (buyer_id, seller_id) -> list of pd.Timestamp
        self._live_history: Dict[Tuple[str, str], List[pd.Timestamp]] = {}
        self.total_historical_orders: int = 0
        self.unique_pairs: int = 0

        if orders_csv_path:
            self.load_from_csv(orders_csv_path)

    def load_from_csv(self, orders_csv_path: str) -> bool:
        """Loads and indexes timestamped orders table into sorted in-memory pair history."""
        try:
            import os
            if not os.path.isfile(orders_csv_path):
                logger.warning("Point-in-time orders table not found at: %s", orders_csv_path)
                self.is_loaded = False
                return False

            df = pd.read_csv(orders_csv_path, usecols=["order_id", "buyer_id", "seller_id", "order_date"])
            df["order_date"] = pd.to_datetime(df["order_date"])
            # Mergesort guarantees stable chronological ordering
            df_sorted = df.sort_values(by="order_date", kind="mergesort")

            history: Dict[Tuple[str, str], List[pd.Timestamp]] = {}
            for _, row in df_sorted.iterrows():
                b_id = str(row["buyer_id"]).strip()
                s_id = str(row["seller_id"]).strip()
                history.setdefault((b_id, s_id), []).append(row["order_date"])

            self._history = history
            self.is_loaded = True
            self.source_path = orders_csv_path
            self.total_historical_orders = len(df_sorted)
            self.unique_pairs = len(history)
            logger.info(
                "Loaded PointInTimeHistoryStore from %s (%d orders, %d unique pairs)",
                orders_csv_path, self.total_historical_orders, self.unique_pairs
            )
            return True
        except Exception as exc:
            logger.error("Failed to load PointInTimeHistoryStore from %s: %s", orders_csv_path, exc)
            self.is_loaded = False
            return False

    def get_prior_edge_weight(
        self,
        buyer_id: Optional[str],
        seller_id: Optional[str],
        order_date: Optional[Union[str, datetime, pd.Timestamp]] = None,
        order_id: Optional[str] = None,
    ) -> Tuple[float, str, str]:
        """Calculates point-in-time prior transaction count.

        Returns:
            (edge_weight, provenance_source, provenance_note)
            where provenance_source is one of:
            - 'server_point_in_time': exact count from timestamped historical ledger or live history
            - 'fallback_unobserved_zero': entity pair has 0 prior transactions in history
            - 'fallback_history_unavailable': reliable history source unavailable; safe 0.0 default
        """
        if not buyer_id or not seller_id:
            return 0.0, "fallback_unobserved_zero", "Buyer or seller ID not provided"

        b_clean = str(buyer_id).strip()
        s_clean = str(seller_id).strip()
        pair_key = (b_clean, s_clean)

        # 1. Historical replay mode: order_date is provided
        if order_date is not None:
            try:
                target_dt = pd.to_datetime(order_date)
            except Exception:
                target_dt = None

            if target_dt is not None:
                if not self.is_loaded:
                    return (
                        0.0,
                        "fallback_history_unavailable",
                        "Server point-in-time history unavailable; defaulted to 0.0 (lifetime count NOT substituted)",
                    )

                pair_ts = self._history.get(pair_key)
                if not pair_ts:
                    # Check live history
                    pair_ts_live = self._live_history.get(pair_key)
                    if not pair_ts_live:
                        return 0.0, "fallback_unobserved_zero", "First transaction between buyer and seller (count=0)"
                    pair_ts = pair_ts_live

                # Strict point-in-time causality: count only transactions strictly before target_dt
                # bisect_left locates first index where t >= target_dt.
                # Therefore all indices < idx have t < target_dt.
                # Duplicate timestamps: any transaction at t == target_dt is at index >= idx, so excluded!
                idx = bisect.bisect_left(pair_ts, target_dt)
                return (
                    float(idx),
                    "server_point_in_time",
                    f"Point-in-time prior order count before {target_dt.strftime('%Y-%m-%d %H:%M:%S')}",
                )

        # 2. Live decision-time scoring mode: order_date not provided
        # Definition: all confirmed prior transactions for this pair recorded prior to decision time.
        # The pending order being scored is not in prior history.
        if not self.is_loaded and not self._live_history:
            return (
                0.0,
                "fallback_history_unavailable",
                "Server transaction history store unavailable; defaulted to 0.0 (lifetime count NOT substituted)",
            )

        # Total confirmed prior count = historical count + any live confirmed count
        hist_count = len(self._history.get(pair_key, [])) if self.is_loaded else 0
        live_count = len(self._live_history.get(pair_key, []))
        total_prior = float(hist_count + live_count)

        if total_prior == 0.0:
            return 0.0, "fallback_unobserved_zero", "No prior recorded transactions for buyer-seller pair at decision time (count=0)"

        return (
            total_prior,
            "server_point_in_time",
            f"Live decision-time confirmed prior transactions: {int(total_prior)}",
        )

    def record_live_transaction(
        self,
        buyer_id: str,
        seller_id: str,
        timestamp: Optional[Union[str, datetime, pd.Timestamp]] = None,
        order_id: Optional[str] = None,
    ) -> None:
        """Records a committed live transaction into the live history."""
        b_clean = str(buyer_id).strip()
        s_clean = str(seller_id).strip()
        pair_key = (b_clean, s_clean)
        ts = pd.to_datetime(timestamp) if timestamp is not None else pd.Timestamp.now(timezone.utc)

        live_list = self._live_history.setdefault(pair_key, [])
        live_list.append(ts)
        live_list.sort()

    def reset_live_history(self) -> None:
        """Clears live session transactions (for testing and isolation)."""
        self._live_history.clear()
