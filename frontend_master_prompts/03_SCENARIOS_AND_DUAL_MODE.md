# 03 — DUAL-MODE UX & QUICK SCENARIO PRESETS

## 1. Quick Scenario Presets (1-Click Interactive Evaluation)

Provide these 4 clickable presets in a sticky banner at the top of the Transaction Analyzer (`/transactions`):

```typescript
export const SCENARIO_PRESETS = [
  {
    id: "preset_normal_buyer",
    label: "1. Verified Repeat Buyer",
    badge: "LOW RISK (ALLOW)",
    color: "emerald",
    description: "Account active for 180 days, 15 prior successful orders, zero returns, single device.",
    payload: {
      order_id: "ORD_SIM_SAFE_01",
      buyer_id: "BUYER_VERIFIED_77",
      seller_id: "SELLER_REPUTABLE_12",
      amount: 65.50,
      base_price: 65.00,
      category_median_price: 70.00,
      buyer_age_days: 180,
      buyer_orders_before: 15,
      buyer_returns_before: 0,
      device_shared_buyer_count: 1.0,
      share_degree: 0.0,
    }
  },
  {
    id: "preset_price_arbitrage",
    label: "2. New Account Price Anomaly",
    badge: "SUSPICIOUS (REVIEW)",
    color: "amber",
    description: "New account (1 day old) ordering at 3.5x normal catalog price.",
    payload: {
      order_id: "ORD_SIM_ANOMALY_02",
      buyer_id: "BUYER_NEWBIE_99",
      seller_id: "SELLER_UNKNOWN_44",
      amount: 450.00,
      base_price: 120.00,
      category_median_price: 110.00,
      buyer_age_days: 1,
      buyer_orders_before: 0,
      buyer_returns_before: 0,
      device_shared_buyer_count: 1.0,
      share_degree: 1.0,
    }
  },
  {
    id: "preset_device_farm",
    label: "3. Device Farm Collusion Ring",
    badge: "HIGH RISK (BLOCK)",
    color: "rose",
    description: "Shared hardware device used across 9 distinct buyer accounts in 24 hours.",
    payload: {
      order_id: "ORD_SIM_FARM_03",
      buyer_id: "BUYER_RING_MEMBER_04",
      seller_id: "SELLER_RING_LEADER_01",
      amount: 890.00,
      base_price: 850.00,
      category_median_price: 800.00,
      buyer_age_days: 4,
      buyer_orders_before: 1,
      buyer_returns_before: 1,
      device_shared_buyer_count: 9.0,
      share_degree: 8.5,
      seller_buyer_concentration_hhi: 0.88,
    }
  },
  {
    id: "preset_serial_returner",
    label: "4. Serial Return Abuse",
    badge: "HOLD (INVESTIGATE)",
    color: "orange",
    description: "Buyer with 80% historical return rate submitting an expensive claim.",
    payload: {
      order_id: "ORD_SIM_RETURN_04",
      buyer_id: "BUYER_REFUND_ABUSER",
      seller_id: "SELLER_ELECTRONICS_09",
      amount: 320.00,
      base_price: 320.00,
      category_median_price: 300.00,
      buyer_age_days: 60,
      buyer_orders_before: 5,
      buyer_returns_before: 4,
      device_shared_buyer_count: 2.0,
      share_degree: 1.5,
    }
  }
];
```

---

## 2. Reason Code Human Translation Dictionary

Translate raw backend strings into plain-English bullet points with icons:

```typescript
export const REASON_CODE_TRANSLATIONS: Record<string, { title: string; explanation: string; severity: "critical" | "warning" | "info" }> = {
  HIGH_DEVICE_COLLISION: {
    title: "Shared Device Alert",
    explanation: "This computer or mobile phone has been used by multiple separate buyer accounts recently.",
    severity: "critical"
  },
  PRICE_OUTLIER_99TH_PCT: {
    title: "Unusual Price Alert",
    explanation: "The purchase price is significantly higher or lower than market standards for this category.",
    severity: "warning"
  },
  COLD_START_BUYER_HIGH_VALUE: {
    title: "New Account High Spend",
    explanation: "This customer signed up recently and is making an unusually large first purchase.",
    severity: "warning"
  },
  RING_COLLUSION_SUSPECT: {
    title: "Coordinated Ring Signal",
    explanation: "Graph analysis detected a dense relationship cluster between this buyer and merchant.",
    severity: "critical"
  },
  CROSS_SELLER_IMAGE_REUSE: {
    title: "Potential Counterfeit / Photo Theft",
    explanation: "The product photo matches an existing image from an unrelated top-rated seller.",
    severity: "critical"
  },
  HIGH_RETURN_VELOCITY: {
    title: "Serial Return Risk",
    explanation: "This buyer frequently purchases items and requests refunds or claims missing packages.",
    severity: "warning"
  }
};
```
