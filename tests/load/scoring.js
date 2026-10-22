import http from "k6/http";
import { check } from "k6";

// This script intentionally has no production URL or credential baked in.
// Example:
//   BASE_URL=https://staging.example API_KEY=... TARGET_RPS=5 \
//   TARGET_P95_MS=250 k6 run tests/load/scoring.js
const BASE_URL = (__ENV.BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
const API_KEY = __ENV.API_KEY || __ENV.TRUSTSHIELD_API_KEY || "";

function numberEnv(name, fallback) {
  const value = Number(__ENV[name]);
  return Number.isFinite(value) ? value : fallback;
}

// These are deployment-specific targets, not claims about universal capacity.
const TARGET_RPS = Math.max(1, Math.floor(numberEnv("TARGET_RPS", 1)));
const TARGET_P95_MS = Math.max(1, numberEnv("TARGET_P95_MS", 500));
const TARGET_P99_MS = Math.max(TARGET_P95_MS, numberEnv("TARGET_P99_MS", 1000));
const MAX_ERROR_RATE = Math.min(1, Math.max(0, numberEnv("MAX_ERROR_RATE", 0.01)));
const MIN_CHECK_RATE = Math.min(1, Math.max(0, numberEnv("MIN_CHECK_RATE", 0.99)));
const DURATION = __ENV.DURATION || "30s";
const PREALLOCATED_VUS = Math.max(1, Math.floor(numberEnv("PREALLOCATED_VUS", 2)));
const MAX_VUS = Math.max(PREALLOCATED_VUS, Math.floor(numberEnv("MAX_VUS", 10)));

export const options = {
  discardResponseBodies: true,
  scenarios: {
    scoring: {
      executor: "constant-arrival-rate",
      rate: TARGET_RPS,
      timeUnit: "1s",
      duration: DURATION,
      preAllocatedVUs: PREALLOCATED_VUS,
      maxVUs: MAX_VUS,
    },
  },
  thresholds: {
    checks: [`rate>${MIN_CHECK_RATE}`],
    http_req_failed: [`rate<${MAX_ERROR_RATE}`],
    http_req_duration: [`p(95)<${TARGET_P95_MS}`, `p(99)<${TARGET_P99_MS}`],
  },
};

function headers() {
  const result = { "Content-Type": "application/json" };
  if (API_KEY) result["X-API-Key"] = API_KEY;
  return result;
}

export default function () {
  const payload = JSON.stringify({
    order_id: `k6-score-${__VU}-${__ITER}`,
    buyer_id: "k6-buyer",
    seller_id: "k6-seller",
    amount: 150.0,
    base_price: 150.0,
    category_median_price: 150.0,
  });
  const response = http.post(`${BASE_URL}/transaction/score`, payload, {
    headers: headers(),
    tags: { endpoint: "transaction_score" },
  });

  check(response, {
    "scoring response is successful": (res) => res.status === 200,
    "scoring response is JSON": (res) =>
      String(res.headers["Content-Type"] || "").includes("application/json"),
  });
}
