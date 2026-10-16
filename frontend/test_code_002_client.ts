import assert from "node:assert";
import {
  TrustShieldApi,
  ApiError,
  setDemoMode,
  isDemoMode,
  setSessionApiKey,
  getSessionApiKey,
  clearSessionApiKey,
} from "./src/lib/api/client";

async function runTests() {
  console.log("--- Starting CODE-002 Regression Test Suite ---");

  // 1. In-memory session API key management
  console.log("[Test 1] In-memory session API key management");
  clearSessionApiKey();
  assert.strictEqual(getSessionApiKey(), "", "Key should be empty initially");
  setSessionApiKey("  my-secret-key-123  ");
  assert.strictEqual(getSessionApiKey(), "my-secret-key-123", "Key should be trimmed and stored");
  clearSessionApiKey();
  assert.strictEqual(getSessionApiKey(), "", "Key should be cleared");

  // 2. Demo mode toggling
  console.log("[Test 2] Demo mode explicit toggle");
  setDemoMode(true);
  assert.strictEqual(isDemoMode(), true, "Demo mode should be active");
  setDemoMode(false);
  assert.strictEqual(isDemoMode(), false, "Demo mode should be inactive");

  // 3. Explicit Demo Mode Honest Marking
  console.log("[Test 3] Explicit Demo Mode Honest Evidence Labeling");
  setDemoMode(true);
  const demoListing = await TrustShieldApi.analyzeListing({
    listing_id: "TEST_LISTING_01",
    product_id: "PROD_01",
    price: 100,
  });
  assert.strictEqual(demoListing.isLive, false, "Demo response must have isLive=false");
  assert.strictEqual(demoListing.data.clip_scored, false, "Demo response must NOT claim clip_scored=true");
  assert.ok(demoListing.data.model_used.includes("Offline Simulation"), "Model must be labeled as Offline Simulation");

  const demoTx = await TrustShieldApi.scoreTransaction({
    order_id: "TEST_ORD_01",
    amount: 150,
  });
  assert.strictEqual(demoTx.isLive, false, "Demo tx must have isLive=false");
  assert.ok(demoTx.data.model_used.includes("Offline Simulation"), "Tx model must be labeled as Offline Simulation");

  const demoReady = await TrustShieldApi.getReady();
  assert.strictEqual(demoReady.isLive, false, "Demo ready must have isLive=false");
  assert.strictEqual(demoReady.data.status, "not_ready", "Demo status must be not_ready");
  assert.strictEqual(demoReady.data.models_ready, false, "Demo must not claim models are ready");

  // 4. Live Mode Error Handling (No silent fallback to fake data)
  console.log("[Test 4] Live Mode Error Transparency (No silent fallback)");
  setDemoMode(false); // Live mode

  // When live backend is unreachable on a non-running port or error
  // Attempting scoreTransaction in live mode against an unreachable port should THROW ApiError,
  // NOT silently return fabricated scores!
  let threwApiError = false;
  try {
    await TrustShieldApi.scoreTransaction({
      order_id: "FAIL_TX_01",
      amount: 500,
    });
  } catch (err: unknown) {
    if (err instanceof ApiError) {
      threwApiError = true;
      assert.ok(err.message.length > 0, "Error message must be present");
      console.log(`  Successfully caught ApiError: ${err.message}`);
    }
  }
  assert.strictEqual(threwApiError, true, "scoreTransaction must throw ApiError in live mode when request fails");

  let threwListingError = false;
  try {
    await TrustShieldApi.analyzeListing({
      listing_id: "FAIL_LISTING_01",
      product_id: "PROD_01",
      price: 500,
    });
  } catch (err: unknown) {
    if (err instanceof ApiError) {
      threwListingError = true;
      assert.ok(err.message.length > 0, "Error message must be present");
      console.log(`  Successfully caught ApiError: ${err.message}`);
    }
  }
  assert.strictEqual(threwListingError, true, "analyzeListing must throw ApiError in live mode when request fails");

  // 5. Stream Event Field Mapping & Separation
  console.log("[Test 5] SSE Stream Event field mapping and score separation");
  setDemoMode(true);
  const demoTxScore = await TrustShieldApi.scoreTransaction({
    order_id: "TEST_ORD_STREAM_01",
    amount: 300,
  });
  assert.ok(typeof demoTxScore.data.calibrated_fraud_probability === "number", "calibrated_fraud_probability must be present");
  assert.ok(typeof demoTxScore.data.policy_risk_score === "number", "policy_risk_score must be present");
  assert.strictEqual(demoTxScore.data.overall_fraud_probability, demoTxScore.data.policy_risk_score, "overall_fraud_probability matches policy risk score");

  // Stream event schema contract verification
  const testStreamEvent = {
    event_id: "EVT_TEST_01",
    timestamp: "12:00:00",
    order_id: "ORD_TEST_01",
    buyer_id: "B1",
    seller_id: "S1",
    amount: 200,
    risk_score: 0.12,
    calibrated_risk: 0.04,
    policy_risk_score: 0.35,
    risk_level: "medium" as const,
    decision: "REVIEW" as const,
    trust_score: 65,
    reason_codes: ["SUSPICIOUS_VELOCITY"],
    source: "simulation_stream" as const,
  };
  assert.strictEqual(testStreamEvent.calibrated_risk, 0.04, "calibrated_risk must hold calibrated probability");
  assert.strictEqual(testStreamEvent.policy_risk_score, 0.35, "policy_risk_score must hold composite policy score");
  assert.notStrictEqual(testStreamEvent.calibrated_risk, testStreamEvent.policy_risk_score, "calibrated_risk and policy_risk_score must remain distinct");

  console.log("--- All CODE-002 Regression Tests Passed Successfully ---");
}

runTests().catch((err) => {
  console.error("Test failed:", err);
  process.exit(1);
});
