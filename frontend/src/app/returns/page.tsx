"use client";

import React, { useState } from "react";
import Link from "next/link";
import { TrustShieldApi, ApiError } from "@/lib/api/client";
import { ReturnScoreRequest, ReturnScoreResponse } from "@/lib/types/api";
import { DecisionBadge } from "@/components/ui/DecisionBadge";
import {
  RotateCcw,
  Zap,
  AlertTriangle,
  ShieldAlert,
  FileSearch,
  ArrowRight,
  CheckCircle2,
} from "lucide-react";

interface ReturnPreset {
  id: string;
  label: string;
  decision: "ALLOW" | "REVIEW" | "HOLD" | "BLOCK";
  description: string;
  payload: ReturnScoreRequest;
}

/* BEGIN GENERATED */
const RETURN_PRESETS: ReturnPreset[] = [
  {
    id: "serial_abuser",
    label: "Serial Return Abuser",
    decision: "BLOCK",
    description: "Buyer with 80% return rate and 4 prior returns claiming wrong item.",
    payload: {
      return_id: "RET_CLAIM_9021",
      order_id: "ORD_SIM_RETURN_04",
      buyer_id: "BUYER_REFUND_ABUSER",
      seller_id: "SELLER_ELECTRONICS_09",
      order_amount: 320.0,
      days_to_return: 28,
      buyer_prior_returns: 4,
      buyer_orders_before_return: 5,
      buyer_return_rate_before: 0.8,
      seller_prior_returns: 2,
      seller_orders_before_return: 120,
      seller_return_rate_before: 0.016,
      reason: "wrong_item_received",
    },
  },
  {
    id: "wardrobing",
    label: "Wardrobing Suspect",
    decision: "HOLD",
    description: "High-ticket item returned at day 29 claiming changed mind.",
    payload: {
      return_id: "RET_CLAIM_WARDROBE_02",
      order_id: "ORD_APPAREL_552",
      buyer_id: "BUYER_OCCASIONAL_08",
      seller_id: "SELLER_BOUTIQUE_01",
      order_amount: 480.0,
      days_to_return: 29,
      buyer_prior_returns: 2,
      buyer_orders_before_return: 3,
      buyer_return_rate_before: 0.66,
      seller_prior_returns: 4,
      seller_orders_before_return: 200,
      seller_return_rate_before: 0.02,
      reason: "changed_mind",
    },
  },
  {
    id: "legitimate",
    label: "Legitimate Exchange",
    decision: "ALLOW",
    description: "Long-standing customer returning garment on day 2 due to size mismatch.",
    payload: {
      return_id: "RET_CLAIM_SAFE_10",
      order_id: "ORD_GENUINE_99",
      buyer_id: "BUYER_VERIFIED_77",
      seller_id: "SELLER_REPUTABLE_12",
      order_amount: 54.0,
      days_to_return: 2,
      buyer_prior_returns: 0,
      buyer_orders_before_return: 18,
      buyer_return_rate_before: 0.0,
      seller_prior_returns: 3,
      seller_orders_before_return: 350,
      seller_return_rate_before: 0.008,
      reason: "size_issue",
    },
  },
];

export default function ReturnAbusePage() {
  const [activePresetId, setActivePresetId] = useState<string>("serial_abuser");
  const [formData, setFormData] = useState<ReturnScoreRequest>(RETURN_PRESETS[0].payload);
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

  const handleSelectPreset = (preset: ReturnPreset) => {
    setActivePresetId(preset.id);
    setFormData(preset.payload);
    executeAnalysis(preset.payload);
  };

  const executeAnalysis = async (payload: ReturnScoreRequest) => {
    setLoading(true);
    setErrorState(null);
    try {
      const res = await TrustShieldApi.analyzeReturn(payload);
      setResult(res.data);
    } catch (err: unknown) {
      console.warn("Return analysis notice:", err);
      setResult(null);
      const isAuth = err instanceof ApiError ? err.isAuthError : false;
      const status = err instanceof ApiError ? err.status : undefined;
      const message = err instanceof Error ? err.message : String(err);
      setErrorState({ message, isAuth, status });
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setActivePresetId("custom");
    executeAnalysis(formData);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
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

      {/* 1-Click Interactive Evaluation Presets */}
      <div className="p-4 rounded-xl border border-[var(--border)] bg-[var(--surface)] shadow-md space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Zap className="w-4 h-4 text-orange-500" />
            <span className="text-xs font-bold uppercase tracking-wider text-[var(--text-primary)]">
              Return Risk Scenarios
            </span>
          </div>
          <span className="text-[11px] text-[var(--text-secondary)]">Select a scenario to evaluate model decisions</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {RETURN_PRESETS.map((preset) => {
            const isSelected = activePresetId === preset.id;
            return (
              <button
                key={preset.id}
                type="button"
                onClick={() => handleSelectPreset(preset)}
                className={`p-3 rounded-lg border text-left transition-all cursor-pointer ${
                  isSelected
                    ? "bg-[var(--surface-elevated)] border-orange-500 shadow-sm ring-1 ring-orange-500/30"
                    : "bg-[var(--surface)] border-[var(--border)] hover:bg-[var(--surface-elevated)]"
                }`}
              >
                <div className="flex items-center justify-between gap-1">
                  <span className="text-xs font-semibold text-[var(--text-primary)] truncate">{preset.label}</span>
                  <DecisionBadge decision={preset.decision} size="sm" showIcon={false} />
                </div>
                <p className="text-[11px] text-[var(--text-secondary)] mt-1.5 line-clamp-2 leading-tight">
                  {preset.description}
                </p>
              </button>
            );
          })}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-start">
        {/* Form Inputs */}
        <div className="p-5 rounded-xl border border-[var(--border)] bg-[var(--surface)] shadow-md space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold uppercase tracking-wider text-[var(--text-primary)]">
              Claim Parameters
            </span>
            <span className="text-[10px] font-mono text-orange-500">POST /return/analyze</span>
          </div>

          <form onSubmit={handleSubmit} className="space-y-3 text-xs font-mono">
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
              <label className="text-[11px] text-[var(--text-secondary)] block mb-1">Original Order ID</label>
              <input
                type="text"
                value={formData.order_id || ""}
                onChange={(e) => setFormData({ ...formData, order_id: e.target.value })}
                className="w-full bg-[var(--surface-elevated)] border border-[var(--border)] rounded-md px-3 py-1.5 text-[var(--text-primary)] focus-ring"
              />
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="text-[11px] text-[var(--text-secondary)] block mb-1">Buyer ID</label>
                <input
                  type="text"
                  value={formData.buyer_id || ""}
                  onChange={(e) => setFormData({ ...formData, buyer_id: e.target.value })}
                  className="w-full bg-[var(--surface-elevated)] border border-[var(--border)] rounded-md px-3 py-1.5 text-[var(--text-primary)] focus-ring"
                />
              </div>
              <div>
                <label className="text-[11px] text-[var(--text-secondary)] block mb-1">Seller ID</label>
                <input
                  type="text"
                  value={formData.seller_id || ""}
                  onChange={(e) => setFormData({ ...formData, seller_id: e.target.value })}
                  className="w-full bg-[var(--surface-elevated)] border border-[var(--border)] rounded-md px-3 py-1.5 text-[var(--text-primary)] focus-ring"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="text-[11px] text-[var(--text-secondary)] block mb-1">Order Amount ($)</label>
                <input
                  type="number"
                  step="any"
                  value={formData.order_amount ?? ""}
                  onChange={(e) => setFormData({ ...formData, order_amount: parseFloat(e.target.value) || 0 })}
                  className="w-full bg-[var(--surface-elevated)] border border-[var(--border)] rounded-md px-3 py-1.5 text-[var(--text-primary)] focus-ring"
                />
              </div>
              <div>
                <label className="text-[11px] text-[var(--text-secondary)] block mb-1">Days to Return</label>
                <input
                  type="number"
                  value={formData.days_to_return ?? ""}
                  onChange={(e) => setFormData({ ...formData, days_to_return: parseInt(e.target.value, 10) || 0 })}
                  className="w-full bg-[var(--surface-elevated)] border border-[var(--border)] rounded-md px-3 py-1.5 text-[var(--text-primary)] focus-ring"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="text-[11px] text-[var(--text-secondary)] block mb-1">Buyer Prior Returns</label>
                <input
                  type="number"
                  value={formData.buyer_prior_returns ?? ""}
                  onChange={(e) => {
                    const r = parseInt(e.target.value, 10) || 0;
                    const o = formData.buyer_orders_before_return || 1;
                    setFormData({
                      ...formData,
                      buyer_prior_returns: r,
                      buyer_return_rate_before: Number((r / o).toFixed(2)),
                    });
                  }}
                  className="w-full bg-[var(--surface-elevated)] border border-[var(--border)] rounded-md px-3 py-1.5 text-[var(--text-primary)] focus-ring"
                />
              </div>
              <div>
                <label className="text-[11px] text-[var(--text-secondary)] block mb-1">Buyer Past Orders</label>
                <input
                  type="number"
                  value={formData.buyer_orders_before_return ?? ""}
                  onChange={(e) => {
                    const o = parseInt(e.target.value, 10) || 1;
                    const r = formData.buyer_prior_returns || 0;
                    setFormData({
                      ...formData,
                      buyer_orders_before_return: o,
                      buyer_return_rate_before: Number((r / o).toFixed(2)),
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
              className="w-full mt-2 py-2.5 rounded-lg bg-orange-600 hover:bg-orange-500 text-xs font-bold text-white shadow-md shadow-orange-600/20 transition-all flex items-center justify-center gap-2 focus-ring cursor-pointer"
            >
              <Zap className="w-4 h-4" />
              <span>{loading ? "Analyzing Return Risk..." : "Analyze Return Claim"}</span>
            </button>
          </form>
        </div>

        {/* Results Overview */}
        <div className="lg:col-span-2 space-y-4">
          {errorState && (
            <div className="p-6 rounded-xl border border-rose-500/30 bg-rose-500/10 space-y-3">
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

          {result && (
            <div className="space-y-4">
              <div className="p-6 rounded-xl border border-[var(--border)] bg-[var(--surface)] shadow-md">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                  <div>
                    <div className="flex items-center gap-2">
                      <DecisionBadge decision={result.decision} size="lg" />
                      <span className="text-xs px-2.5 py-1 rounded-md bg-[var(--surface-elevated)] font-mono font-medium text-[var(--text-primary)] border border-[var(--border)]">
                        Calibrated Probability: {(result.return_fraud_probability * 100).toFixed(1)}%
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
                  Behavioral Reason Codes &amp; Evidence Flags
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

              {/* Operator Escalation to Dossier Studio */}
              <div className="p-4 rounded-xl border border-blue-500/30 bg-blue-500/5 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <div>
                  <div className="text-xs font-bold text-[var(--text-primary)] flex items-center gap-1.5">
                    <FileSearch className="w-4 h-4 text-blue-500" />
                    <span>Cross-Workflow Escalation</span>
                  </div>
                  <p className="text-[11px] text-[var(--text-secondary)] mt-0.5">
                    Escalate this return abuse incident to Case Dossier Studio for formal dispute brief generation and HMAC evidence bundling.
                  </p>
                </div>
                <Link
                  href="/investigations"
                  className="px-3.5 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-xs font-semibold text-white shadow-sm flex items-center gap-1.5 shrink-0 self-start sm:self-auto transition-colors cursor-pointer"
                >
                  <span>Open in Investigations</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </Link>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
