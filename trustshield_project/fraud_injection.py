"""Fraud pattern injection for synthetic marketplace: fake listings, return abuse, coordinated rings, and collusion."""

import pandas as pd
from datetime import timedelta

from entity_generator import rng, SIM_END, build_base_entities
from product_listing_generator import build_catalog_and_listings
from order_return_generator import build_orders_and_returns, RETURN_REASONS, RETURN_REASON_WEIGHTS

TOTAL_ORDERS_ASSUMED = 50000
TARGET_FRAUD_RATE = 0.07

FRAUD_TYPE_SHARE = {
    "fake_listing": 0.40,
    "return_abuse": 0.30,
    "coordinated_fraud": 0.20,
    "seller_buyer_collusion": 0.10,
}
FAKE_LISTING_RATE = 0.025


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


# ---------------------------------------------------------------------------
# Scenario 1: Fake Listing (~40% of fraud orders)
# ---------------------------------------------------------------------------

def inject_fake_listings(listings_df, orders_df, products_df, target_n_listings, n_traffic_tiers=5):
    """
    Picks target_n_listings listings, STRATIFIED across order-traffic tiers
    (including zero/low-traffic listings) rather than only the highest-
    traffic ones. Pure order-count-driven selection technically hits an
    order target efficiently, but concentrates fraud into a handful of
    already-popular listings — leaving the listing-level classifier's
    positive class too thin to train or evaluate meaningfully. Real fake
    listings also span the traffic spectrum (many get caught fast with
    almost no orders; a few slip through and rack up sales) — stratifying
    is the more realistic choice, not just the more ML-convenient one.

    Applies a price anomaly (way underpriced vs. its own catalog price) and
    an image-text mismatch (the listing's *displayed* product image is
    swapped for an unrelated product's image — simulating a copied/stolen
    image — while product_id itself, the ground-truth actual item, stays
    untouched), then flags every order on a chosen listing as fraud.

    Each rogue seller's flagged listings become one ring (a scam seller
    running several fake listings at once, not just one).
    """
    listings_df = listings_df.copy()
    orders_df = orders_df.copy()
    listings_df["is_fraudulent"] = False
    listings_df["fraud_type"] = None
    listings_df["price_anomaly"] = False
    listings_df["image_mismatch"] = False
    listings_df["displayed_product_id"] = listings_df["product_id"]  # default: matches actual product

    order_counts = orders_df.groupby("listing_id").size()
    traffic = listings_df["listing_id"].map(order_counts).fillna(0)
    tiers = pd.qcut(traffic.rank(method="first"), q=n_traffic_tiers, labels=False)

    per_tier_target = target_n_listings // n_traffic_tiers
    remainder = target_n_listings % n_traffic_tiers

    chosen_listing_ids = []
    for tier in range(n_traffic_tiers):
        tier_ids = listings_df.loc[tiers == tier, "listing_id"].to_numpy()
        n_pick = per_tier_target + (1 if tier < remainder else 0)
        n_pick = min(n_pick, len(tier_ids))
        if n_pick > 0:
            chosen_listing_ids.extend(rng.choice(tier_ids, size=n_pick, replace=False))

    chosen_mask = listings_df["listing_id"].isin(chosen_listing_ids)
    n_chosen = chosen_mask.sum()

    # Only a subset gets a price anomaly — a mixed signal (some scams are
    # underpriced, some rely purely on stolen images/description) both
    # avoids perfect price-only separability and is more realistic than
    # every single fake listing being drastically cheap.
    price_anomaly_subset = rng.random(n_chosen) < 0.6
    chosen_idx = listings_df.index[chosen_mask]
    price_anomaly_idx = chosen_idx[price_anomaly_subset]

    # Price anomaly: factor overlaps the tail of normal price variance
    # (normal price noise is lognormal sigma=0.15, i.e. roughly 0.65-1.55x)
    # rather than being a cleanly separable band — a real detector has to
    # work a bit for this signal, not just threshold a disjoint range.
    price_factor = rng.uniform(0.3, 0.7, size=len(price_anomaly_idx))
    listings_df.loc[price_anomaly_idx, "price"] = (
        listings_df.loc[price_anomaly_idx, "price"].values * price_factor
    ).round(2)
    listings_df.loc[price_anomaly_idx, "price_anomaly"] = True

    # Propagate the discounted price to orders_df so the model actually sees it!
    price_anomaly_listing_ids = listings_df.loc[price_anomaly_idx, "listing_id"]
    updated_prices = listings_df.loc[price_anomaly_idx].set_index("listing_id")["price"]
    affected_orders_mask = orders_df["listing_id"].isin(price_anomaly_listing_ids)
    orders_df.loc[affected_orders_mask, "amount"] = orders_df.loc[affected_orders_mask, "listing_id"].map(updated_prices)

    # Image mismatch: the listing shows a *different* product's image/text
    # than what product_id (the real catalog item) actually is.
    all_product_ids = products_df["product_id"].values
    swapped_product_ids = rng.choice(all_product_ids, size=n_chosen, replace=True)
    listings_df.loc[chosen_mask, "displayed_product_id"] = swapped_product_ids
    listings_df.loc[chosen_mask, "image_mismatch"] = True

    listings_df.loc[chosen_mask, "is_fraudulent"] = True
    listings_df.loc[chosen_mask, "fraud_type"] = "fake_listing"

    # Ring assignment: group by seller (a rogue seller's fake listings = one ring)
    seller_of_chosen = listings_df.loc[chosen_mask, ["listing_id", "seller_id"]]
    ledger_rows = []
    for seller_id, group in seller_of_chosen.groupby("seller_id"):
        ring_id = f"RING_FAKE_{seller_id}"
        for lid in group["listing_id"]:
            ledger_rows.append({"fraud_ring_id": ring_id, "fraud_type": "fake_listing",
                                 "entity_type": "listing", "entity_id": lid})

    fraud_order_ids = set(orders_df.loc[orders_df["listing_id"].isin(chosen_listing_ids), "order_id"])
    return listings_df, orders_df, fraud_order_ids, pd.DataFrame(ledger_rows)


# ---------------------------------------------------------------------------
# Scenario 2: Return Abuse (~30% of fraud orders)
# ---------------------------------------------------------------------------

def inject_return_abuse(buyers_df, orders_df, returns_df, address_sharing_log,
                         already_fraud_order_ids, target_fraud_orders):
    """
    Selects a pool of "abuser" buyers and forces a high fraction of their
    remaining (not already fraud-tagged) orders into abusive returns —
    fast turnaround (1-5 days, vs. the organic 1-21 day window) and a
    dedicated 'used_and_returned' reason, approved regardless.

    If a buyer already shares an address with another buyer in a
    fraud_linked pairing, both are pulled into the same ring — return
    abuse is often a shared-address household running the same scam
    together, not just a lone buyer.
    """
    orders_df = orders_df.copy()
    eligible_orders = orders_df[~orders_df["order_id"].isin(already_fraud_order_ids)]
    orders_per_buyer = eligible_orders.groupby("buyer_id").size()
    eligible_buyer_set = set(orders_per_buyer[orders_per_buyer >= 2].index)

    fraud_linked_pairs = address_sharing_log[address_sharing_log["share_type"] == "fraud_linked"]
    partner_of = dict(zip(fraud_linked_pairs["buyer_id"], fraud_linked_pairs["shared_with_buyer_id"]))

    # Prioritize BOTH members of a fraud-linked address-sharing pair as
    # co-abusers, so return_abuse actually produces multi-buyer rings for
    # Phase 3 graph detection — previously every ring ended up solo-buyer
    # (verified empirically: 151/151 rings were size 1) because pair
    # overlap with the general candidate pool was left to chance instead
    # of being deliberately selected for.
    paired_candidates = []
    seen = set()
    for b1, b2 in zip(fraud_linked_pairs["buyer_id"], fraud_linked_pairs["shared_with_buyer_id"]):
        if b1 in eligible_buyer_set and b2 in eligible_buyer_set and b1 not in seen and b2 not in seen:
            paired_candidates.append(b1)
            paired_candidates.append(b2)
            seen.add(b1)
            seen.add(b2)

    solo_candidates = sorted(eligible_buyer_set - seen)
    rng.shuffle(solo_candidates)

    # Paired buyers go first (as a block, so both halves of a pair land
    # near each other and both get picked before budget runs out), then
    # solo buyers fill the remaining budget.
    candidate_buyers = paired_candidates + solo_candidates

    new_return_rows = []
    updated_return_ids_to_fraud = []
    ledger_rows = []
    fraud_order_ids = set()
    running_total = 0
    return_counter = len(returns_df)

    for buyer_id in candidate_buyers:
        if running_total >= target_fraud_orders:
            break

        buyer_orders = eligible_orders[eligible_orders["buyer_id"] == buyer_id]
        if len(buyer_orders) == 0:
            continue

        abuse_fraction = rng.uniform(0.6, 0.9)
        n_abuse = max(1, int(round(len(buyer_orders) * abuse_fraction)))
        targeted_orders = buyer_orders.sample(
            n=min(n_abuse, len(buyer_orders)),
            random_state=int(rng.integers(0, 2**31)),
        )

        already_returned_ids = set(returns_df["order_id"])
        for _, order in targeted_orders.iterrows():
            fraud_order_ids.add(order["order_id"])
            running_total += 1

            if order["order_id"] in already_returned_ids:
                updated_return_ids_to_fraud.append(order["order_id"])
            else:
                delay = int(rng.integers(1, 6))
                # Reason drawn from the SAME pool as organic returns — a
                # real abusive buyer states a normal-sounding reason, not
                # something that gives away the fraud itself.
                reason = rng.choice(RETURN_REASONS, p=RETURN_REASON_WEIGHTS)
                new_return_rows.append({
                    "return_id": f"RETURN_FRAUD_{return_counter:06d}",
                    "order_id": order["order_id"],
                    "buyer_id": order["buyer_id"],
                    "seller_id": order["seller_id"],
                    "return_date": order["order_date"] + timedelta(days=delay),
                    "reason": reason,
                    "status": "approved",
                })
                return_counter += 1

        ring_id = f"RING_ABUSE_{min(buyer_id, partner_of.get(buyer_id, buyer_id))}" \
            if buyer_id in partner_of else f"RING_ABUSE_{buyer_id}"
        ledger_rows.append({"fraud_ring_id": ring_id, "fraud_type": "return_abuse",
                             "entity_type": "buyer", "entity_id": buyer_id})

    returns_df = returns_df.copy()
    returns_df["is_fraudulent"] = returns_df.get("is_fraudulent", False)
    returns_df["fraud_type"] = returns_df.get("fraud_type", None)
    if updated_return_ids_to_fraud:
        mask = returns_df["order_id"].isin(updated_return_ids_to_fraud)
        returns_df.loc[mask, "is_fraudulent"] = True
        returns_df.loc[mask, "fraud_type"] = "return_abuse"
        # Reason left as originally recorded — a relabeled organic return
        # keeps its stated reason, since a real abuser wouldn't announce
        # the fraud through the reason field.

    if new_return_rows:
        new_df = pd.DataFrame(new_return_rows)
        new_df["is_fraudulent"] = True
        new_df["fraud_type"] = "return_abuse"
        returns_df = pd.concat([returns_df, new_df], ignore_index=True)

    return returns_df, fraud_order_ids, pd.DataFrame(ledger_rows)


# ---------------------------------------------------------------------------
# Scenario 3: Coordinated Fraud — device-sharing rings (~20% of fraud orders)
# ---------------------------------------------------------------------------

def inject_coordinated_fraud(orders_df, listings_df, buyers_df, returns_df, device_sharing_log,
                              already_fraud_order_ids, target_fraud_orders):
    """
    Groups buyers connected by fraud_linked device sharing into rings
    (connected components), then for each ring reschedules a subset of
    members' existing orders into a short burst window (3-7 days) — a
    sybil / multi-accounting pattern where linked accounts transact in a
    tight cluster rather than spread naturally across the year.

    The burst window for a ring is always chosen *after* both the latest
    listing_date AND the latest buyer signup_date among the orders being
    moved, so neither order_date >= listing_date nor order_date >= buyer
    signup_date is ever violated by the reschedule. Orders that already
    have a return on file are excluded from the reschedule pool entirely —
    moving order_date forward without also shifting that return's date
    would leave a return dated before its own order.
    """
    fraud_pairs = device_sharing_log[device_sharing_log["share_type"] == "fraud_linked"]

    uf = UnionFind()
    for _, row in fraud_pairs.iterrows():
        uf.union(row["buyer_id"], row["shared_with_buyer_id"])

    rings = {}
    # sorted() here is not cosmetic — Python's string-set iteration order is
    # hash-randomized per process (PYTHONHASHSEED), so without sorting, ring
    # processing order (and therefore every downstream rng draw) silently
    # differed between runs even with a fixed rng seed. Verified empirically:
    # two runs produced different is_fraudulent counts before this fix.
    for buyer_id in sorted(set(fraud_pairs["buyer_id"]) | set(fraud_pairs["shared_with_buyer_id"])):
        root = uf.find(buyer_id)
        rings.setdefault(root, set()).add(buyer_id)
    rings = {rid: sorted(members) for rid, members in rings.items() if len(members) >= 2}

    listing_dates = listings_df.set_index("listing_id")["listing_date"]
    signup_dates = buyers_df.set_index("buyer_id")["signup_date"]
    already_returned_order_ids = set(returns_df["order_id"])

    orders_df = orders_df.copy()
    eligible_orders = orders_df[
        (~orders_df["order_id"].isin(already_fraud_order_ids)) &
        (~orders_df["order_id"].isin(already_returned_order_ids))
    ]

    ledger_rows = []
    fraud_order_ids = set()
    running_total = 0
    reschedule_map = {}  # order_id -> new order_date

    ring_ids = list(rings.keys())
    rng.shuffle(ring_ids)

    for root in ring_ids:
        if running_total >= target_fraud_orders:
            break
        members = rings[root]
        ring_orders = eligible_orders[eligible_orders["buyer_id"].isin(members)]
        if len(ring_orders) == 0:
            continue

        # Cap how many of this ring's orders get pulled into the burst,
        # so one large ring doesn't single-handedly blow past the target.
        n_take = min(len(ring_orders), max(2, int(rng.integers(2, 6))))
        taken = ring_orders.sample(n=n_take, random_state=int(rng.integers(0, 2**31)))

        listing_dates_for_taken = taken["listing_id"].map(listing_dates)
        signup_dates_for_taken = taken["buyer_id"].map(signup_dates)
        burst_start_floor = max(listing_dates_for_taken.max(), signup_dates_for_taken.max())
        max_start_offset = max(1, (SIM_END - burst_start_floor).days - 7)
        burst_start = burst_start_floor + timedelta(days=int(rng.integers(1, max_start_offset + 1)))
        burst_span_days = int(rng.integers(3, 8))

        for order_id in taken["order_id"]:
            offset = int(rng.integers(0, burst_span_days))
            reschedule_map[order_id] = burst_start + timedelta(days=offset)
            fraud_order_ids.add(order_id)
            running_total += 1

        ring_id_label = f"RING_COORD_{root}"
        for member in members:
            ledger_rows.append({"fraud_ring_id": ring_id_label, "fraud_type": "coordinated_fraud",
                                 "entity_type": "buyer", "entity_id": member})

    orders_df["is_fraudulent"] = orders_df.get("is_fraudulent", False)
    orders_df["fraud_type"] = orders_df.get("fraud_type", None)
    if reschedule_map:
        mask = orders_df["order_id"].isin(reschedule_map.keys())
        orders_df.loc[mask, "order_date"] = orders_df.loc[mask, "order_id"].map(reschedule_map)
        orders_df.loc[mask, "is_fraudulent"] = True
        orders_df.loc[mask, "fraud_type"] = "coordinated_fraud"

    return orders_df, fraud_order_ids, pd.DataFrame(ledger_rows)


# ---------------------------------------------------------------------------
# Scenario 4: Seller-Buyer Collusion (~10% of fraud orders)
# ---------------------------------------------------------------------------

def inject_seller_buyer_collusion(orders_df, returns_df, listings_df, buyers_df,
                                   already_fraud_order_ids, target_fraud_orders):
    """
    Picks a seller and a small group (3-6) of their repeat buyers, then
    concentrates a burst of orders + near-automatic 'approved' refund
    returns between just that group and that seller — simulating a seller
    and a handful of buyer accounts self-dealing to fabricate revenue and
    then extract refunds, rather than genuine independent customers.

    Like coordinated_fraud, the group's chosen orders are rescheduled into
    a short burst window (5-14 days — slightly longer than coordinated
    fraud's 3-7, since a collusion "campaign" plausibly runs a bit longer
    before the seller account gets flagged) so this scenario leaves an
    actual detectable temporal/structural signature — a seller receiving
    a burst of repeat business from the same small buyer set — rather than
    just being scattered pre-existing orders with extra returns attached.
    The burst window respects both listing_date and buyer signup_date, and
    orders that already have a return on file are excluded from the pool
    entirely (same reasoning as coordinated_fraud: rescheduling without
    also shifting an existing return would leave it dated before its own
    order).
    """
    orders_df = orders_df.copy()
    already_returned_ids = set(returns_df["order_id"])
    eligible_orders = orders_df[
        (~orders_df["order_id"].isin(already_fraud_order_ids)) &
        (~orders_df["order_id"].isin(already_returned_ids))
    ]

    buyers_per_seller = eligible_orders.groupby("seller_id")["buyer_id"].nunique()
    candidate_sellers = buyers_per_seller[buyers_per_seller >= 3].index.to_numpy()
    rng.shuffle(candidate_sellers)

    listing_dates = listings_df.set_index("listing_id")["listing_date"]
    signup_dates = buyers_df.set_index("buyer_id")["signup_date"]

    ledger_rows = []
    new_return_rows = []
    fraud_order_ids = set()
    running_total = 0
    return_counter = len(returns_df) + 100000  # offset to avoid id collision with return_abuse block
    reschedule_map = {}

    for seller_id in candidate_sellers:
        if running_total >= target_fraud_orders:
            break

        seller_orders = eligible_orders[eligible_orders["seller_id"] == seller_id]
        buyer_pool = seller_orders["buyer_id"].unique()
        if len(buyer_pool) < 3:
            continue

        group_size = min(len(buyer_pool), int(rng.integers(3, 7)))
        colluding_buyers = rng.choice(buyer_pool, size=group_size, replace=False)

        group_orders = seller_orders[seller_orders["buyer_id"].isin(colluding_buyers)]
        n_take = min(len(group_orders), max(3, int(rng.integers(3, 10))))
        taken = group_orders.sample(n=n_take, random_state=int(rng.integers(0, 2**31)))

        listing_dates_for_taken = taken["listing_id"].map(listing_dates)
        signup_dates_for_taken = taken["buyer_id"].map(signup_dates)
        burst_start_floor = max(listing_dates_for_taken.max(), signup_dates_for_taken.max())
        max_start_offset = max(1, (SIM_END - burst_start_floor).days - 14)
        burst_start = burst_start_floor + timedelta(days=int(rng.integers(1, max_start_offset + 1)))
        burst_span_days = int(rng.integers(5, 15))

        for order_id in taken["order_id"]:
            offset = int(rng.integers(0, burst_span_days))
            reschedule_map[order_id] = burst_start + timedelta(days=offset)

        for _, order in taken.iterrows():
            fraud_order_ids.add(order["order_id"])
            running_total += 1
            new_order_date = reschedule_map[order["order_id"]]

            if rng.random() < 0.8:
                delay = int(rng.integers(1, 4))
                reason = rng.choice(RETURN_REASONS, p=RETURN_REASON_WEIGHTS)
                new_return_rows.append({
                    "return_id": f"RETURN_COLLUSION_{return_counter:06d}",
                    "order_id": order["order_id"],
                    "buyer_id": order["buyer_id"],
                    "seller_id": order["seller_id"],
                    "return_date": new_order_date + timedelta(days=delay),
                    "reason": reason,
                    "status": "approved",
                })
                return_counter += 1

        ring_id = f"RING_COLLUSION_{seller_id}"
        ledger_rows.append({"fraud_ring_id": ring_id, "fraud_type": "seller_buyer_collusion",
                             "entity_type": "seller", "entity_id": seller_id})
        for b in colluding_buyers:
            ledger_rows.append({"fraud_ring_id": ring_id, "fraud_type": "seller_buyer_collusion",
                                 "entity_type": "buyer", "entity_id": b})

    orders_df["is_fraudulent"] = orders_df.get("is_fraudulent", False)
    orders_df["fraud_type"] = orders_df.get("fraud_type", None)
    mask = orders_df["order_id"].isin(fraud_order_ids)
    orders_df.loc[mask, "is_fraudulent"] = True
    orders_df.loc[mask, "fraud_type"] = "seller_buyer_collusion"
    if reschedule_map:
        resched_mask = orders_df["order_id"].isin(reschedule_map.keys())
        orders_df.loc[resched_mask, "order_date"] = orders_df.loc[resched_mask, "order_id"].map(reschedule_map)

    returns_df = returns_df.copy()
    if new_return_rows:
        new_df = pd.DataFrame(new_return_rows)
        new_df["is_fraudulent"] = True
        new_df["fraud_type"] = "seller_buyer_collusion"
        returns_df = pd.concat([returns_df, new_df], ignore_index=True)

    return orders_df, returns_df, fraud_order_ids, pd.DataFrame(ledger_rows)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def inject_all_fraud(listings_df, orders_df, returns_df, buyers_df, products_df,
                      address_sharing_log, device_sharing_log,
                      total_orders=None, target_rate=TARGET_FRAUD_RATE):
    total_orders = total_orders or len(orders_df)
    total_fraud_target = int(total_orders * target_rate)
    target_n_fake_listings = int(len(listings_df) * FAKE_LISTING_RATE)

    print(f"Fraud injection targets: {total_fraud_target} fraud orders total ({target_rate:.0%} of "
          f"{total_orders}); fake_listing sized by listing count ({target_n_fake_listings} listings, "
          f"{FAKE_LISTING_RATE:.1%} of {len(listings_df)})")

    print("Injecting fake listings...")
    listings_df, orders_df, fake_ids, ledger1 = inject_fake_listings(
        listings_df, orders_df, products_df, target_n_fake_listings
    )

    # fake_listing's actual order count is a byproduct of listing selection
    # now, not a direct target — rebalance the other 3 types' order budgets
    # off the REMAINING total so overall order-level fraud rate still lands
    # close to target_rate, keeping their 30:20:10 relative proportions.
    remaining_budget = max(0, total_fraud_target - len(fake_ids))
    other_share_sum = FRAUD_TYPE_SHARE["return_abuse"] + FRAUD_TYPE_SHARE["coordinated_fraud"] \
        + FRAUD_TYPE_SHARE["seller_buyer_collusion"]
    targets = {
        k: int(remaining_budget * (FRAUD_TYPE_SHARE[k] / other_share_sum))
        for k in ["return_abuse", "coordinated_fraud", "seller_buyer_collusion"]
    }
    print(f"  fake_listing produced {len(fake_ids)} fraud orders; remaining budget {remaining_budget} "
          f"rebalanced across other 3 types: {targets}")

    print("Injecting return abuse...")
    returns_df, abuse_ids, ledger2 = inject_return_abuse(
        buyers_df, orders_df, returns_df, address_sharing_log, fake_ids, targets["return_abuse"]
    )

    print("Injecting coordinated fraud...")
    already = fake_ids | abuse_ids
    orders_df, coord_ids, ledger3 = inject_coordinated_fraud(
        orders_df, listings_df, buyers_df, returns_df, device_sharing_log, already, targets["coordinated_fraud"]
    )

    print("Injecting seller-buyer collusion...")
    already = already | coord_ids
    orders_df, returns_df, collusion_ids, ledger4 = inject_seller_buyer_collusion(
        orders_df, returns_df, listings_df, buyers_df, already, targets["seller_buyer_collusion"]
    )

    # Merge fake-listing / return-abuse fraud flags onto orders_df too, so
    # orders_df.is_fraudulent reflects ALL four scenario types in one place.
    orders_df["is_fraudulent"] = orders_df.get("is_fraudulent", False)
    orders_df["fraud_type"] = orders_df.get("fraud_type", None)
    fake_mask = orders_df["order_id"].isin(fake_ids)
    orders_df.loc[fake_mask, "is_fraudulent"] = True
    orders_df.loc[fake_mask, "fraud_type"] = "fake_listing"
    abuse_mask = orders_df["order_id"].isin(abuse_ids)
    orders_df.loc[abuse_mask, "is_fraudulent"] = True
    orders_df.loc[abuse_mask, "fraud_type"] = "return_abuse"

    fraud_ground_truth = pd.concat([ledger1, ledger2, ledger3, ledger4], ignore_index=True)

    # --- Leakage guard: fraud_ring_id must never touch the feature tables ---
    for name, df in [("listings", listings_df), ("orders", orders_df), ("returns", returns_df)]:
        assert "fraud_ring_id" not in df.columns, f"LEAKAGE: fraud_ring_id found in {name}_df"

    return {
        "listings": listings_df,
        "orders": orders_df,
        "returns": returns_df,
        "fraud_ground_truth": fraud_ground_truth,
    }


if __name__ == "__main__":
    base = build_base_entities()
    catalog = build_catalog_and_listings(base["sellers"])
    txn = build_orders_and_returns(
        buyers_df=base["buyers"], sellers_df=catalog["sellers"],
        listings_df=catalog["listings"], device_mapping_df=base["device_mapping"],
    )

    result = inject_all_fraud(
        listings_df=catalog["listings"],
        orders_df=txn["orders"],
        returns_df=txn["returns"],
        buyers_df=txn["buyers"],
        products_df=catalog["products"],
        address_sharing_log=base["address_sharing_log"],
        device_sharing_log=base["device_sharing_log"],
    )

    orders_df = result["orders"]
    print(f"\n✅ Overall order fraud rate: {orders_df['is_fraudulent'].mean():.3%} "
          f"(target {TARGET_FRAUD_RATE:.0%})")

    fraud_only = orders_df[orders_df["is_fraudulent"]]
    pcts = (fraud_only["fraud_type"].value_counts(normalize=True) * 100).round(1).to_dict()
    pcts_str = ", ".join(f"{k}: {v}%" for k, v in pcts.items())
    print(f"✅ Fraud type breakdown: {pcts_str}")

    print(f"\nFake listings flagged: {result['listings']['is_fraudulent'].sum()} "
          f"/ {len(result['listings'])} listings")
    print(f"Fraudulent returns flagged: {result['returns']['is_fraudulent'].sum()} "
          f"/ {len(result['returns'])} returns")

    print(f"\nGround-truth ledger: {len(result['fraud_ground_truth'])} rows, "
          f"{result['fraud_ground_truth']['fraud_ring_id'].nunique()} unique rings")
    print(result["fraud_ground_truth"]["fraud_type"].value_counts())

    print("\n✅ Time-burst check (coordinated_fraud order date spread per ring in days):")
    coord_orders = orders_df[orders_df["fraud_type"] == "coordinated_fraud"].merge(
        result["fraud_ground_truth"][result["fraud_ground_truth"]["fraud_type"] == "coordinated_fraud"]
        [["entity_id", "fraud_ring_id"]].drop_duplicates(),
        left_on="buyer_id", right_on="entity_id", how="inner"
    )
    if len(coord_orders):
        spread = coord_orders.groupby("fraud_ring_id")["order_date"].agg(lambda x: (x.max() - x.min()).days)
        desc = spread.describe()
        print(f"   Spread (days): mean={desc['mean']:.1f}, min={desc['min']:.0f}, max={desc['max']:.0f}")

    print("\nTemporal integrity re-check after rescheduling (order_date >= listing_date):")
    merged_check = orders_df.merge(result["listings"][["listing_id", "listing_date"]], on="listing_id")
    violations = (merged_check["order_date"] < merged_check["listing_date"]).sum()
    print(f"  violations found: {violations} (should be 0)")

    print("\nLeakage guard: fraud_ring_id absent from listings/orders/returns columns — passed (assertions above).")
