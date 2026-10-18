"use client";

import React, { useState } from "react";
import { TrustShieldApi, ApiError } from "@/lib/api/client";
import { ReturnScoreRequest, ReturnScoreResponse } from "@/lib/types/api";
import { DecisionBadge } from "@/components/ui/DecisionBadge";
import {
  RotateCcw,
  Zap,
  AlertTriangle,
  ShieldAlert,
} from "lucide-react";

export default function ReturnAbusePage() {
  /* BEGIN GENERATED */
  const [formData, setFormData] = useState<ReturnScoreRequest>({
    return_id: "RET_CLAIM_9021",
    order_id: "ORD_SIM_RETURN_04",
    buyer_id: "BUYER_REFUND_ABUSER",
    seller_id: "SELLER_ELECTRONICS_09",
    order_amount: 320.00,
    days_to_return: 28,
    buyer_prior_returns: 4,
    buyer_orders_before_return: 5,
    buyer_return_rate_before: 0.80,
    seller_prior_returns: 2,
    seller_orders_before_return: 120,
    seller_return_rate_before: 0.016,
    reason: "wrong_item_received",
  });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ReturnScoreResponse | null>({
    return_id: "RET_CLAIM_9021",
    return_fraud_probability: 0.825,
    risk_label: "high",
    decision: "BLOCK",
    model_used: "Return Abuse Gradient Boosted Classifier (Phase 3)",
    reason_codes: ["HIGH_RETURN_VELOCITY", "SUSPICIOUS_REFUND_CLAIM"],
  });
  /* END GENERATED */

  const [errorState, setErrorState] = useState<{ message: string; isAuth?: boolean; status?: number } | null>(null);

  const handleAnalyze = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setErrorState(null);
    try {
      const res = await TrustShieldApi.analyzeReturn(formData);
      setResult(res.data);
    } catch (err: unknown) {
      console.error("Return analysis failed:", err);
      setResult(null);
      const isAuth = err instanceof ApiError ? err.isAuthError : false;
      const status = err instanceof ApiError ? err.status : undefined;
      const message = err instanceof Error ? err.message : String(err);
      setErrorState({ message, isAuth, status });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-[var(--text-primary)] flex items-center gap-2">
            <RotateCcw className="w-5 h-5 text-orange-500" />
            <span>Specialized Return Abuse &amp; Wardrobing Studio</span>
          </h1>
          <p className="text-xs text-[var(--text-secondary)] mt-1">
            Detects serial return abuse, wardrobing, empty-box claims, and coordinated return fraud patterns.
          </p>
        </div>

        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[var(--surface)] border border-[var(--border)] text-xs font-mono">
          <span className="text-[var(--text-muted)]">Model:</span>
          <span className="text-orange-500 font-semibold">Phase 3 Return Classifier</span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Form Inputs */}
        <div className="p-5 rounded-xl border border-[var(--border)] bg-[var(--surface)] shadow-md space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold uppercase tracking-wider text-[var(--text-primary)]">
              Claim Parameters
            </span>
            <span className="text-[10px] font-mono text-orange-500">POST /return/analyze</span>
          </div>

          <form onSubmit={handleAnalyze} className="space-y-3 text-xs font-mono">
            <div>
              <label className="text-[11px] text-[var(--text-secondary)] block mb-1">Return Claim ID</label>
              <input
                type="text"
                value={formData.return_id || ""}
                onChange={(e) => setFormData({ ...formData, return_id: e.target.value })}
                className="w-full bg-[var(--surface-elevated)] border border-[var(--border)] rounded-md px-3 py-1.5 text-[var(--text-primary)] focus-ring"
              />
            </div>

            <div>
              <label className="text-[11px] text-[var(--text-secondary)] block mb-1">Buyer ID</label>
              <input
                type="text"
                value={formData.buyer_id || ""}
                onChange={(e) => setFormData({ ...formData, buyer_id: e.target.value })}
                className="w-full bg-[var(--surface-elevated)] border border-[var(--border)] rounded-md px-3 py-1.5 text-[var(--text-primary)] focus-ring"
              />
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="text-[11px] text-[var(--text-secondary)] block mb-1">Order Amount ($)</label>
                <input
                  type="number"
                  value={formData.order_amount || 0}
                  onChange={(e) => setFormData({ ...formData, order_amount: parseFloat(e.target.value) || 0 })}
                  className="w-full bg-[var(--surface-elevated)] border border-[var(--border)] rounded-md px-3 py-1.5 text-[var(--text-primary)] focus-ring"
                />
              </div>

              <div>
                <label className="text-[11px] text-[var(--text-secondary)] block mb-1">Days to Return</label>
                <input
                  type="number"
                  value={formData.days_to_return || 0}
                  onChange={(e) => setFormData({ ...formData, days_to_return: parseInt(e.target.value, 10) || 0 })}
                  className="w-full bg-[var(--surface-elevated)] border border-[var(--border)] rounded-md px-3 py-1.5 text-[var(--text-primary)] focus-ring"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="text-[11px] text-[var(--text-secondary)] block mb-1">Prior Returns</label>
                <input
                  type="number"
                  value={formData.buyer_prior_returns || 0}
                  onChange={(e) => {
                    const retVal = Math.max(0, parseInt(e.target.value, 10) || 0);
                    const ordVal = formData.buyer_orders_before_return || 1;
                    const calculatedRate = ordVal > 0 ? Math.min(1.0, Math.round((retVal / ordVal) * 1000) / 1000) : 0;
                    setFormData({
                      ...formData,
                      buyer_prior_returns: retVal,
                      buyer_return_rate_before: calculatedRate,
                    });
                  }}
                  className="w-full bg-[var(--surface-elevated)] border border-[var(--border)] rounded-md px-3 py-1.5 text-[var(--text-primary)] focus-ring"
                />
              </div>

              <div>
                <label className="text-[11px] text-[var(--text-secondary)] block mb-1">Total Orders</label>
                <input
                  type="number"
                  value={formData.buyer_orders_before_return || 0}
                  onChange={(e) => {
                    const ordVal = Math.max(0, parseInt(e.target.value, 10) || 0);
                    const retVal = formData.buyer_prior_returns || 0;
                    const calculatedRate = ordVal > 0 ? Math.min(1.0, Math.round((retVal / ordVal) * 1000) / 1000) : 0;
                    setFormData({
                      ...formData,
                      buyer_orders_before_return: ordVal,
                      buyer_return_rate_before: calculatedRate,
                    });
                  }}
                  className="w-full bg-[var(--surface-elevated)] border border-[var(--border)] rounded-md px-3 py-1.5 text-[var(--text-primary)] focus-ring"
                />
              </div>
            </div>

            <div>
              <label className="text-[11px] text-[var(--text-secondary)] block mb-1">Claimed Return Reason</label>
              <select
                value={formData.reason}
                onChange={(e) => setFormData({ ...formData, reason: e.target.value })}
                className="w-full bg-[var(--surface-elevated)] border border-[var(--border)] rounded-md px-3 py-1.5 text-[var(--text-primary)] focus-ring"
              >
                <option value="wrong_item_received">Wrong Item Received</option>
                <option value="defective">Defective / Damaged</option>
                <option value="changed_mind">Changed Mind</option>
                <option value="size_issue">Size / Fit Issue</option>
              </select>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full mt-2 py-2.5 rounded-lg bg-orange-600 hover:bg-orange-500 text-xs font-bold text-white shadow-md shadow-orange-600/20 transition-all flex items-center justify-center gap-2 focus-ring"
            >
              <Zap className="w-4 h-4" />
              <span>{loading ? "Analyzing Return Risk..." : "Analyze Return Claim"}</span>
            </button>
          </form>
        </div>

        {/* Results Overview */}
        {errorState && (
          <div className="lg:col-span-2 p-6 rounded-xl border border-rose-500/30 bg-rose-500/10 space-y-3">
            <div className="flex items-center gap-2 text-rose-600 dark:text-rose-400 font-bold">
              <ShieldAlert className="w-5 h-5" />
              <span>
                {errorState.isAuth
                  ? "Authentication Required (HTTP 401 / 403)"
                  : errorState.status
                  ? `Return Analysis Failed (HTTP ${errorState.status})`
                  : "Backend Service Unavailable"}
              </span>
            </div>
            <p className="text-xs text-[var(--text-primary)] font-mono leading-relaxed">{errorState.message}</p>
            <p className="text-[11px] text-[var(--text-secondary)]">
              The return abuse analysis endpoint is unreachable or experienced a server error.
            </p>
          </div>
        )}

        {/* BEGIN GENERATED */}
        {result && (
          <div className="lg:col-span-2 space-y-6">
            <div className="p-6 rounded-xl border border-[var(--border)] bg-[var(--surface)] shadow-md">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                <div>
                  <div className="flex items-center gap-2">
                    <DecisionBadge decision={result.decision} size="lg" />
                    <span className="text-xs px-2.5 py-1 rounded-md bg-[var(--surface-elevated)] font-mono font-medium text-[var(--text-primary)] border border-[var(--border)]">
                      Risk: {(result.return_fraud_probability * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="text-xs text-[var(--text-secondary)] mt-1.5 font-mono">
                    Claim ID: <span className="text-[var(--text-primary)] font-semibold">{result.return_id}</span> · Model:{" "}
                    <span className="text-orange-500">{result.model_used}</span>
                  </div>
                </div>

                <div className="text-left sm:text-right">
                  <div className="text-[10px] text-[var(--text-muted)] font-mono">BUYER HISTORICAL RETURN RATE</div>
                  <div className="text-2xl font-bold font-mono text-rose-500 mt-1">
                    {((formData.buyer_return_rate_before ?? 0.8) * 100).toFixed(0)}%
                  </div>
                </div>
              </div>
            </div>

            {/* Warning Codes Breakdown */}
            <div className="p-5 rounded-xl border border-[var(--border)] bg-[var(--surface)] shadow-md space-y-3">
              <div className="text-xs font-bold uppercase tracking-wider text-[var(--text-primary)]">
                Behavioral Reason Codes &amp; Flags
              </div>

              {result.reason_codes.map((code) => (
                <div
                  key={code}
                  className="p-3.5 rounded-lg bg-orange-500/10 border border-orange-500/20 text-xs text-[var(--text-primary)] flex items-start gap-3"
                >
                  <AlertTriangle className="w-4 h-4 text-orange-500 shrink-0 mt-0.5" />
                  <div>
                    <div className="font-bold font-mono text-[var(--text-primary)]">{code}</div>
                    <div className="text-[11px] text-[var(--text-secondary)] mt-0.5">
                      Buyer has returned {formData.buyer_prior_returns} out of {formData.buyer_orders_before_return} past orders within the last 60 days. Exceeds platform 95th percentile threshold.
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
        {/* END GENERATED */}
      </div>
    </div>
  );
}
