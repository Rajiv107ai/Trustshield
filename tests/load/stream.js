import http from "k6/http";
import { check } from "k6";

const BASE_URL = (__ENV.BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
const API_KEY = __ENV.API_KEY || __ENV.TRUSTSHIELD_API_KEY || "";

function numberEnv(name, fallback) {
  const value = Number(__ENV[name]);
  return Number.isFinite(value) ? value : fallback;
}

// Streaming targets are deliberately configurable because event cadence and
// connection budgets vary by deployment.  A stream request uses a bounded
// event count so the load test cannot leave unbounded connections open.
const TARGET_RPS = Math.max(1, Math.floor(numberEnv("TARGET_RPS", 1)));
const TARGET_P95_MS = Math.max(1, numberEnv("TARGET_P95_MS", 5000));
const TARGET_P99_MS = Math.max(TARGET_P95_MS, numberEnv("TARGET_P99_MS", 10000));
const MAX_ERROR_RATE = Math.min(1, Math.max(0, numberEnv("MAX_ERROR_RATE", 0.01)));
const MIN_CHECK_RATE = Math.min(1, Math.max(0, numberEnv("MIN_CHECK_RATE", 0.99)));
const DURATION = __ENV.DURATION || "30s";
const PREALLOCATED_VUS = Math.max(1, Math.floor(numberEnv("PREALLOCATED_VUS", 1)));
const MAX_VUS = Math.max(PREALLOCATED_VUS, Math.floor(numberEnv("MAX_VUS", 5)));
const STREAM_INTERVAL = Math.max(0.5, numberEnv("STREAM_INTERVAL", 0.5));
const STREAM_LIMIT = Math.max(1, Math.floor(numberEnv("STREAM_LIMIT", 1)));
const STREAM_TIMEOUT = __ENV.STREAM_TIMEOUT || "20s";

export const options = {
  scenarios: {
    stream: {
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
  const result = { Accept: "text/event-stream" };
  if (API_KEY) result["X-API-Key"] = API_KEY;
  return result;
}

export default function () {
  const url = `${BASE_URL}/stream/transactions?interval=${STREAM_INTERVAL}&limit=${STREAM_LIMIT}`;
  const response = http.get(url, {
    headers: headers(),
    timeout: STREAM_TIMEOUT,
    tags: { endpoint: "transaction_stream" },
  });
  const body = String(response.body || "");
  const eventCount = body.split("\n").filter((line) => line.startsWith("data: ")).length;

  check(response, {
    "stream response is successful": (res) => res.status === 200,
    "stream content type is SSE": (res) =>
      String(res.headers["Content-Type"] || "").includes("text/event-stream"),
    "stream returned a bounded event": () => eventCount >= STREAM_LIMIT,
  });
}
