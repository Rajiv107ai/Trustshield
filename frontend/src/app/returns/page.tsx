"use client";

import React, { useState } from "react";
import { TrustShieldApi } from "@/lib/api/client";
import { ReturnScoreRequest, ReturnScoreResponse } from "@/lib/types/api";
import {
  RotateCcw,
  Zap,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  HelpCircle,
  ShieldAlert,
  ArrowRight,
  TrendingDown,
} from "lucide-react";

export default function ReturnAbusePage() {
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

  const handleAnalyze = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      const res = await TrustShieldApi.analyzeReturn(formData);
      setResult(res.data);
    } catch (err) {
      console.error("Return analysis failed:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-[#E8EDF3] flex items-center gap-2">
            <RotateCcw className="w-5 h-5 text-orange-400" />
            <span>Specialized Return Abuse & Wardrobing Studio</span>
          </h1>
          <p className="text-xs text-[#8995A3] mt-1">
            Detects serial return abuse, wardrobing, empty-box claims, and coordinated return fraud patterns.
          </p>
        </div>

        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[#111821] border border-[#202A35] text-xs font-mono">
          <span className="text-[#8995A3]">Model:</span>
          <span className="text-orange-400 font-semibold">Phase 3 Return Classifier</span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Form Inputs */}
        <div className="p-5 rounded-xl border border-[#202A35] bg-[#111821] space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold uppercase tracking-wider text-[#E8EDF3]">
              Claim Parameters
            </span>
            <span className="text-[10px] font-mono text-orange-400">POST /return/analyze</span>
          </div>

          <form onSubmit={handleAnalyze} className="space-y-3 text-xs font-mono">
            <div>
              <label className="text-[11px] text-[#8995A3] block mb-1">Return Claim ID</label>
              <input
                type="text"
                value={formData.return_id || ""}
                onChange={(e) => setFormData({ ...formData, return_id: e.target.value })}
                className="w-full bg-[#0E131A] border border-[#202A35] rounded-md px-3 py-1.5 text-[#E8EDF3] focus:border-orange-500 focus:outline-none"
              />
            </div>

            <div>
              <label className="text-[11px] text-[#8995A3] block mb-1">Buyer ID</label>
              <input
                type="text"
                value={formData.buyer_id || ""}
                onChange={(e) => setFormData({ ...formData, buyer_id: e.target.value })}
                className="w-full bg-[#0E131A] border border-[#202A35] rounded-md px-3 py-1.5 text-[#E8EDF3] focus:border-orange-500 focus:outline-none"
              />
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="text-[11px] text-[#8995A3] block mb-1">Order Amount ($)</label>
                <input
                  type="number"
                  value={formData.order_amount || 0}
                  onChange={(e) => setFormData({ ...formData, order_amount: parseFloat(e.target.value) || 0 })}
                  className="w-full bg-[#0E131A] border border-[#202A35] rounded-md px-3 py-1.5 text-[#E8EDF3] focus:border-orange-500 focus:outline-none"
                />
              </div>

              <div>
                <label className="text-[11px] text-[#8995A3] block mb-1">Days to Return</label>
                <input
                  type="number"
                  value={formData.days_to_return || 0}
                  onChange={(e) => setFormData({ ...formData, days_to_return: parseInt(e.target.value) || 0 })}
                  className="w-full bg-[#0E131A] border border-[#202A35] rounded-md px-3 py-1.5 text-[#E8EDF3] focus:border-orange-500 focus:outline-none"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="text-[11px] text-[#8995A3] block mb-1">Prior Returns</label>
                <input
                  type="number"
                  value={formData.buyer_prior_returns || 0}
                  onChange={(e) => setFormData({ ...formData, buyer_prior_returns: parseInt(e.target.value) || 0 })}
                  className="w-full bg-[#0E131A] border border-[#202A35] rounded-md px-3 py-1.5 text-[#E8EDF3] focus:border-orange-500 focus:outline-none"
                />
              </div>

              <div>
                <label className="text-[11px] text-[#8995A3] block mb-1">Total Orders</label>
                <input
                  type="number"
                  value={formData.buyer_orders_before_return || 0}
                  onChange={(e) => setFormData({ ...formData, buyer_orders_before_return: parseInt(e.target.value) || 0 })}
                  className="w-full bg-[#0E131A] border border-[#202A35] rounded-md px-3 py-1.5 text-[#E8EDF3] focus:border-orange-500 focus:outline-none"
                />
              </div>
            </div>

            <div>
              <label className="text-[11px] text-[#8995A3] block mb-1">Claimed Return Reason</label>
              <select
                value={formData.reason}
                onChange={(e) => setFormData({ ...formData, reason: e.target.value })}
                className="w-full bg-[#0E131A] border border-[#202A35] rounded-md px-3 py-1.5 text-[#E8EDF3] focus:border-orange-500 focus:outline-none"
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
              className="w-full mt-2 py-2.5 rounded-lg bg-orange-600 hover:bg-orange-500 text-xs font-bold text-white shadow-lg shadow-orange-600/20 transition-all flex items-center justify-center gap-2"
            >
              <Zap className="w-4 h-4" />
              <span>{loading ? "Analyzing Return Risk..." : "Analyze Return Claim"}</span>
            </button>
          </form>
        </div>

        {/* Results Overview */}
        {result && (
          <div className="lg:col-span-2 space-y-6">
            <div
              className={`p-6 rounded-xl border ${
                result.decision === "BLOCK"
                  ? "bg-rose-500/10 border-rose-500/30 glow-block"
                  : result.decision === "HOLD"
                  ? "bg-orange-500/10 border-orange-500/30 glow-hold"
                  : "bg-emerald-500/10 border-emerald-500/30 glow-allow"
              }`}
            >
              <div className="flex items-center justify-between">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="text-xl font-bold font-mono text-[#E8EDF3]">{result.decision}</span>
                    <span className="text-xs px-2 py-0.5 rounded-full bg-white/10 font-mono text-white">
                      Risk: {(result.return_fraud_probability * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="text-xs text-[#8995A3] mt-1 font-mono">
                    Claim ID: <span className="text-white">{result.return_id}</span> · Model:{" "}
                    <span className="text-orange-400">{result.model_used}</span>
                  </div>
                </div>

                <div className="text-right">
                  <div className="text-[10px] text-[#8995A3] font-mono">BUYER HISTORICAL RETURN RATE</div>
                  <div className="text-2xl font-bold font-mono text-rose-400 mt-1">
                    {((formData.buyer_return_rate_before ?? 0.8) * 100).toFixed(0)}%
                  </div>
                </div>
              </div>
            </div>

            {/* Warning Codes Breakdown */}
            <div className="p-5 rounded-xl border border-[#202A35] bg-[#111821] space-y-3">
              <div className="text-xs font-bold uppercase tracking-wider text-[#E8EDF3]">
                Behavioral Reason Codes & Flags
              </div>

              {result.reason_codes.map((code) => (
                <div
                  key={code}
                  className="p-3.5 rounded-lg bg-orange-500/5 border border-orange-500/20 text-xs text-orange-200 flex items-start gap-3"
                >
                  <AlertTriangle className="w-4 h-4 text-orange-400 shrink-0 mt-0.5" />
                  <div>
                    <div className="font-bold font-mono text-[#E8EDF3]">{code}</div>
                    <div className="text-[11px] text-[#8995A3] mt-0.5">
                      Buyer has returned {formData.buyer_prior_returns} out of {formData.buyer_orders_before_return} past orders within the last 60 days. Exceeds platform 95th percentile threshold.
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
