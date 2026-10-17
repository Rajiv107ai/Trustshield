"use client";

import React, { useState, useEffect } from "react";
import { useSearchParams } from "next/navigation";
import { useViewMode } from "@/context/ViewModeContext";
import { TrustShieldApi, ApiError } from "@/lib/api/client";
import {
  SCENARIO_PRESETS,
  REASON_CODE_TRANSLATIONS,
  ScenarioPreset,
} from "@/lib/constants/scenarios";
import {
  TransactionScoreRequest,
  TransactionScoreResponse,
} from "@/lib/types/api";
import {
  Zap,
  ShieldAlert,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  HelpCircle,
  Sparkles,
  Microscope,
  RefreshCw,
  Server,
  Layers,
  Check,
} from "lucide-react";
import { Suspense } from "react";

function TransactionAnalyzerContent() {
  const searchParams = useSearchParams();
  const { viewMode } = useViewMode();

  const presetParam = searchParams.get("preset");
  const initialPreset = (presetParam && SCENARIO_PRESETS.find((p) => p.id === presetParam)) || SCENARIO_PRESETS[0];

  // Active form state
  const [formData, setFormData] = useState<TransactionScoreRequest>(
    initialPreset.payload
  );
  const [activePresetId, setActivePresetId] = useState<string>(
    initialPreset.id
  );
  const [loading, setLoading] = useState<boolean>(true);
  const [result, setResult] = useState<TransactionScoreResponse | null>(null);
  const [errorState, setErrorState] = useState<{ message: string; isAuth?: boolean; status?: number } | null>(null);
  const [operatorActionSuccess, setOperatorActionSuccess] = useState<string | null>(null);

  const executeScore = async (payload: TransactionScoreRequest) => {
    setLoading(true);
    setOperatorActionSuccess(null);
    setErrorState(null);
    try {
      const res = await TrustShieldApi.scoreTransaction(payload);
      setResult(res.data);
    } catch (err: unknown) {
      console.error("Scoring failed:", err);
      setResult(null);
      const isAuth = err instanceof ApiError ? err.isAuthError : false;
      const status = err instanceof ApiError ? err.status : undefined;
      const message = err instanceof Error ? err.message : String(err);
      setErrorState({ message, isAuth, status });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const timer = setTimeout(() => {
      executeScore(initialPreset.payload);
    }, 0);
    return () => clearTimeout(timer);
  }, [initialPreset]);

  const handleSelectPreset = (preset: ScenarioPreset) => {
    setActivePresetId(preset.id);
    setFormData(preset.payload);
    executeScore(preset.payload);
  };

  const handleInputChange = (field: keyof TransactionScoreRequest, value: unknown) => {
    const next = { ...formData, [field]: value };
    setFormData(next);
  };

  const handleCustomSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setActivePresetId("custom");
    executeScore(formData);
  };

  const DECISION_CONFIG: Record<string, { bg: string; glow: string; icon: React.ElementType; label: string; sub: string }> = {
    ALLOW: { bg: "bg-emerald-500/10 border-emerald-500/30 text-emerald-400", glow: "glow-allow", icon: CheckCircle2, label: "ALLOW", sub: "Safe for Instant Settlement" },
    REVIEW: { bg: "bg-amber-500/10 border-amber-500/30 text-amber-400", glow: "glow-review", icon: AlertTriangle, label: "REVIEW", sub: "Flagged for Human Verification" },
    HOLD: { bg: "bg-orange-500/10 border-orange-500/30 text-orange-400", glow: "glow-hold", icon: HelpCircle, label: "HOLD", sub: "High-Risk Temporary Freeze" },
    BLOCK: { bg: "bg-rose-500/10 border-rose-500/30 text-rose-400", glow: "glow-block", icon: XCircle, label: "BLOCK", sub: "Confirmed Collusion / Fraud Declared" },
  };
  const badgeConfig = DECISION_CONFIG[result?.decision || "ALLOW"] || DECISION_CONFIG.BLOCK;

  return (
    <div className="space-y-6">
      {/* 1. SCENARIO PRESETS TOOLBAR (Sticky at top) */}
      <div className="p-4 rounded-xl border border-[#202A35] bg-[#111821] shadow-lg space-y-3 sticky top-16 z-20 backdrop-blur-md">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Zap className="w-4 h-4 text-amber-400" />
            <span className="text-xs font-bold uppercase tracking-wider text-[#E8EDF3]">
              1-Click Interactive Evaluation Presets
            </span>
          </div>
          <span className="text-[11px] text-[#8995A3]">Click any scenario to evaluate instantly against the active model</span>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-2.5">
          {SCENARIO_PRESETS.map((preset) => {
            const isSelected = activePresetId === preset.id;
            return (
              <button
                key={preset.id}
                onClick={() => handleSelectPreset(preset)}
                className={`p-3 rounded-lg border text-left transition-all ${
                  isSelected
                    ? "bg-[#151D27] border-blue-500 shadow-md ring-1 ring-blue-500/30"
                    : "bg-[#0E131A] border-[#202A35] hover:bg-[#151D27] hover:border-[#384656]"
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-[#E8EDF3] truncate">{preset.label}</span>
                  <span
                    className={`text-[9px] font-mono px-1.5 py-0.5 rounded font-bold uppercase ${
                      preset.decision === "ALLOW"
                        ? "bg-emerald-500/10 text-emerald-400"
                        : preset.decision === "REVIEW"
                        ? "bg-amber-500/10 text-amber-400"
                        : preset.decision === "HOLD"
                        ? "bg-orange-500/10 text-orange-400"
                        : "bg-rose-500/10 text-rose-400"
                    }`}
                  >
                    {preset.decision}
                  </span>
                </div>
                <p className="text-[11px] text-[#8995A3] mt-1.5 line-clamp-2 leading-tight">
                  {preset.description}
                </p>
              </button>
            );
          })}
        </div>
      </div>

      {errorState && (
        <div className="p-6 rounded-xl border border-rose-500/30 bg-rose-500/10 space-y-3">
          <div className="flex items-center gap-2 text-rose-400 font-bold">
            <ShieldAlert className="w-5 h-5" />
            <span>
              {errorState.isAuth
                ? "Authentication Required (HTTP 401 / 403)"
                : errorState.status
                ? `Transaction Scoring Failed (HTTP ${errorState.status})`
                : "Backend Service Unavailable"}
            </span>
          </div>
          <p className="text-xs text-[#E8EDF3] font-mono leading-relaxed">{errorState.message}</p>
          {errorState.isAuth ? (
            <p className="text-[11px] text-[#8995A3]">
              Authentication credentials were not provided or were rejected. Please configure your API key in{" "}
              <a href="/settings" className="text-blue-400 underline font-semibold">
                Settings
              </a>{" "}
              or switch to Demo Mode.
            </p>
          ) : (
            <p className="text-[11px] text-[#8995A3]">
              The scoring gateway is unreachable or returned an unhandled server error. No fabricated scores will be displayed.
            </p>
          )}
        </div>
      )}

      {/* 2. HERO DECISION BANNER & TRUST SCORE GAUGE */}
      {/* BEGIN GENERATED */}
      {result && (
        <div className={`p-6 rounded-xl border ${badgeConfig.bg} ${badgeConfig.glow} transition-all`}>
          <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6">
            {/* Left: Decision Badge & Summary */}
            <div className="flex items-start gap-4">
              <div className="p-3 rounded-xl bg-black/30 border border-white/10 shrink-0">
                <badgeConfig.icon className="w-8 h-8" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-2xl font-extrabold tracking-tight font-mono">{badgeConfig.label}</span>
                  <span className="text-xs px-2 py-0.5 rounded-full bg-white/10 text-white font-mono uppercase">
                    Risk: {(result.overall_fraud_probability * 100).toFixed(1)}%
                  </span>
                  {result.cold_start && (
                    <span className="text-xs px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-300 font-mono">
                      Cold Start Account
                    </span>
                  )}
                </div>
                <div className="text-sm font-medium mt-1 text-[#E8EDF3]">{badgeConfig.sub}</div>
                <div className="text-xs text-[#8995A3] mt-1 font-mono">
                  Order ID: <span className="text-white font-semibold">{result.order_id}</span> · Model:{" "}
                  <span className="text-blue-400">{result.model_used}</span>
                </div>
              </div>
            </div>

            {/* Right: Dual Risk Metric Gauges */}
            <div className="flex items-center gap-6 self-start lg:self-auto border-t lg:border-t-0 lg:border-l border-white/10 pt-4 lg:pt-0 lg:pl-6">
              {/* Trust Score (0-100) */}
              <div className="text-center">
                <div className="text-[10px] uppercase font-mono text-[#8995A3] tracking-wider">Trust Score</div>
                <div className="text-3xl font-extrabold font-mono mt-1 text-white">
                  {result.trust_score.toFixed(1)}
                  <span className="text-xs font-normal text-[#8995A3]"> / 100</span>
                </div>
                <div className="text-[10px] text-emerald-400 mt-0.5">High Fidelity</div>
              </div>

              {/* Shannon Epistemic Confidence */}
              <div className="text-center">
                <div className="text-[10px] uppercase font-mono text-[#8995A3] tracking-wider">
                  AI Confidence ($1-H_2$)
                </div>
                <div className="text-3xl font-extrabold font-mono mt-1 text-blue-400">
                  {(result.confidence * 100).toFixed(0)}%
                </div>
                <div className="text-[10px] text-[#8995A3] mt-0.5 font-mono">
                  Entropy: {result.predictive_uncertainty ?? 0.08}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 3. DUAL VIEW: EXECUTIVE STORY VIEW vs DEEP AI INSPECTOR */}
      {result && (
        <>
          {viewMode === "executive" ? (
            /* ============================================================ */
            /* EXECUTIVE STORY VIEW: Plain-English explanations & Business actions */
            /* ============================================================ */
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Left 2 Cols: Why AI Flagged This */}
              <div className="lg:col-span-2 p-5 rounded-xl border border-[#202A35] bg-[#111821] space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Sparkles className="w-4 h-4 text-blue-400" />
                    <h3 className="text-xs font-bold uppercase tracking-wider text-[#E8EDF3]">
                      Why The AI Flagged This (Plain-English Story)
                    </h3>
                  </div>
                  <span className="text-[11px] text-[#8995A3] font-mono">
                    {result.reason_codes.length} active risk indicators
                  </span>
                </div>

                {result.reason_codes.length === 0 ? (
                  <div className="p-4 rounded-lg bg-emerald-500/5 border border-emerald-500/20 text-xs text-[#E8EDF3] flex items-center gap-3">
                    <CheckCircle2 className="w-5 h-5 text-emerald-400 shrink-0" />
                    <div>
                      <div className="font-semibold text-emerald-300">Clean Transaction Profile</div>
                      <div className="text-[#8995A3] mt-0.5">
                        No hardware collision, price arbitrage, or graph collusion anomalies detected. Transaction fits historical baseline patterns.
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-2.5">
                    {result.reason_codes.map((code) => {
                      const trans = REASON_CODE_TRANSLATIONS[code] || {
                        title: code,
                        explanation: "Algorithmic risk detector flagged abnormal variance.",
                        severity: "warning",
                      };
                      return (
                        <div
                          key={code}
                          className={`p-3.5 rounded-lg border text-xs ${
                            trans.severity === "critical"
                              ? "bg-rose-500/5 border-rose-500/20 text-rose-200"
                              : "bg-amber-500/5 border-amber-500/20 text-amber-200"
                          }`}
                        >
                          <div className="flex items-center gap-2 font-bold font-mono text-[#E8EDF3]">
                            {trans.severity === "critical" ? (
                              <XCircle className="w-4 h-4 text-rose-400 shrink-0" />
                            ) : (
                              <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
                            )}
                            <span>{trans.title}</span>
                            <span className="text-[10px] font-mono text-[#8995A3]">({code})</span>
                          </div>
                          <div className="text-[11px] text-[#8995A3] mt-1 pl-6 leading-relaxed">
                            {trans.explanation}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}

                {/* Plain English Business Action Recommendation */}
                <div className="pt-2 border-t border-[#202A35]">
                  <div className="text-xs font-semibold text-[#8995A3] uppercase">Ops Recommendation:</div>
                  <div className="text-xs text-[#E8EDF3] mt-1 leading-relaxed">
                    {{
                      ALLOW: "Proceed with automated payment authorization and normal order fulfillment. Expected chargeback probability is under 1%.",
                      REVIEW: "Route transaction to tier-1 fraud queue for manual verification. Call customer or request 3D Secure / OTP authorization.",
                      HOLD: "Do not dispatch items. Place merchant payout on temporary hold pending return policy validation or buyer ID verification.",
                      BLOCK: "Decline payment authorization immediately. Freeze associated device fingerprints and add merchant to collusion monitoring queue.",
                    }[result.decision] || "Algorithmic decision rendered."}
                  </div>
                </div>
              </div>

              {/* Right Col: Operator Action Bar */}
              <div className="p-5 rounded-xl border border-[#202A35] bg-[#111821] flex flex-col justify-between space-y-4">
                <div>
                  <h3 className="text-xs font-bold uppercase tracking-wider text-[#E8EDF3]">
                    Fraud Ops Decision Control
                  </h3>
                  <p className="text-[11px] text-[#8995A3] mt-1">
                    Override or confirm algorithmic recommendations with immediate ledger execution.
                  </p>

                  <div className="space-y-2.5 mt-4">
                    {[
                      { msg: "Transaction Approved & Released for Settlement", bg: "bg-emerald-600/20 hover:bg-emerald-600/30 border-emerald-500/30 text-emerald-300", icon: CheckCircle2, text: "Approve & Release" },
                      { msg: "Transaction Sent to Manual Review Queue", bg: "bg-amber-600/20 hover:bg-amber-600/30 border-amber-500/30 text-amber-300", icon: AlertTriangle, text: "Send for Manual Review" },
                      { msg: "Transaction Confirmed Blocked & Entity Frozen", bg: "bg-rose-600/20 hover:bg-rose-600/30 border-rose-500/30 text-rose-300", icon: XCircle, text: "Confirm Block & Freeze Entity" },
                    ].map((btn, idx) => {
                      const Icon = btn.icon;
                      return (
                        <button key={idx} onClick={() => setOperatorActionSuccess(btn.msg)} className={`w-full py-2.5 px-3 rounded-lg border text-xs font-semibold transition-all flex items-center justify-center gap-2 ${btn.bg}`}>
                          <Icon className="w-4 h-4" />
                          <span>{btn.text}</span>
                        </button>
                      );
                    })}
                  </div>
                </div>

                {operatorActionSuccess && (
                  <div className="p-3 rounded-lg bg-blue-500/10 border border-blue-500/30 text-xs text-blue-300 font-mono flex items-center gap-2">
                    <Check className="w-4 h-4" />
                    <span>{operatorActionSuccess}</span>
                  </div>
                )}
              </div>
            </div>
          ) : (
            /* ============================================================ */
            /* DEEP AI INSPECTOR VIEW: Calibrated math, GNN, Conformal coverage */
            /* ============================================================ */
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* Col 1: Mathematical Risk Decomposition */}
              <div className="p-5 rounded-xl border border-[#202A35] bg-[#111821] space-y-3 font-mono">
                <div className="flex items-center gap-2 text-xs font-bold text-purple-400 uppercase tracking-wider">
                  <Microscope className="w-4 h-4" />
                  <span>Probability Calibration</span>
                </div>
                <div className="space-y-2 text-xs pt-1">
                  {[
                    ["Raw Model Score:", `${((result.raw_fraud_probability ?? result.overall_fraud_probability) * 100).toFixed(2)}%`, "text-[#E8EDF3] font-semibold"],
                    ["Isotonic Calibrated Prob:", `${((result.calibrated_fraud_probability ?? result.overall_fraud_probability) * 100).toFixed(2)}%`, "text-emerald-400 font-semibold"],
                    ["Composite Policy Risk:", `${((result.policy_risk_score ?? result.canonical_trust_engine_risk ?? result.overall_fraud_probability) * 100).toFixed(2)}%`, "text-amber-400 font-semibold"],
                    ["Model Disagreement Spread:", `${(result.model_disagreement * 100).toFixed(2)}%`, "text-amber-400"],
                    ["Policy Thresholds:", "τ = 0.25 / 0.55 / 0.85", "text-blue-400"],
                  ].map(([lbl, val, cls], idx) => (
                    <div key={idx} className={`flex justify-between py-1 ${idx < 4 ? "border-b border-[#202A35]" : ""}`}>
                      <span className="text-[#8995A3]">{lbl}</span>
                      <span className={cls}>{val}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Col 2: Conformal Prediction Set Guarantees */}
              <div className="p-5 rounded-xl border border-[#202A35] bg-[#111821] space-y-3 font-mono">
                <div className="flex items-center gap-2 text-xs font-bold text-blue-400 uppercase tracking-wider">
                  <Layers className="w-4 h-4" />
                  <span>Conformal Coverage Set</span>
                </div>
                <div className="space-y-2 text-xs pt-1">
                  {[
                    ["Prediction Set C(X):", result.conformal_prediction_set && result.conformal_prediction_set.length > 0
                      ? `{ ${result.conformal_prediction_set.join(", ")} }`
                      : (result.calibrated_fraud_probability ?? result.overall_fraud_probability) > 0.65 ? "{ 1 } (High Risk)"
                      : (result.calibrated_fraud_probability ?? result.overall_fraud_probability) < 0.35 ? "{ 0 } (Low Risk)"
                      : "{ 0, 1 } (Ambiguous / Review)", "text-white font-bold bg-white/10 px-2 py-0.5 rounded"],
                    ["Target Miscoverage (α):", "0.05 (95% Coverage)", "text-[#E8EDF3]"],
                    ["Calibration Status:", result.conformal_is_calibrated ? "Calibrated (Val Split)" : "Fallback (Uncalibrated)", result.conformal_is_calibrated ? "text-emerald-400 font-semibold" : "text-amber-400 font-semibold"],
                    ["Conformal Quantile (q̂):", result.conformal_q_hat !== undefined && result.conformal_q_hat !== null ? `q̂ = ${result.conformal_q_hat.toFixed(4)}` : "q̂ = 0.5000 (Default)", "text-[#E8EDF3]"],
                  ].map(([lbl, val, cls], idx) => (
                    <div key={idx} className={`flex justify-between py-1 ${idx < 3 ? "border-b border-[#202A35]" : ""}`}>
                      <span className="text-[#8995A3]">{lbl}</span>
                      <span className={cls}>{val}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Col 3: GNN Node Representation Specs */}
              <div className="p-5 rounded-xl border border-[#202A35] bg-[#111821] space-y-3 font-mono">
                <div className="flex items-center gap-2 text-xs font-bold text-cyan-400 uppercase tracking-wider">
                  <Server className="w-4 h-4" />
                  <span>GNN & Topology State</span>
                </div>
                <div className="space-y-2 text-xs pt-1">
                  {[
                    ["GNN Embedding Space:", "16-Dimensional", "text-[#E8EDF3]"],
                    ["Message Passing:", "2-Hop HeteroConv", "text-[#E8EDF3]"],
                    ["Continuous Time (Time2Vec):", "Δt ≥ 0 asserted", "text-emerald-400"],
                    ["Device Sharing Collision:", `${formData.device_shared_buyer_count ?? 1} devices`, "text-rose-400 font-semibold"],
                  ].map(([lbl, val, cls], idx) => (
                    <div key={idx} className={`flex justify-between py-1 ${idx < 3 ? "border-b border-[#202A35]" : ""}`}>
                      <span className="text-[#8995A3]">{lbl}</span>
                      <span className={cls}>{val}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </>
      )}
      {/* END GENERATED */}

      {/* 4. CUSTOM TRANSACTION INPUT FORM */}
      <div className="p-5 rounded-xl border border-[#202A35] bg-[#111821] space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-xs font-bold uppercase tracking-wider text-[#E8EDF3]">
              Custom Parameter Ingestion Matrix
            </h3>
            <p className="text-[11px] text-[#8995A3]">
              Modify input features to evaluate model response across arbitrary order configurations.
            </p>
          </div>
          <button
            onClick={handleCustomSubmit}
            disabled={loading}
            className="px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-xs font-bold text-white shadow-md transition-all flex items-center gap-2"
          >
            {loading ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Zap className="w-3.5 h-3.5" />}
            <span>Re-score Transaction</span>
          </button>
        </div>

        <form onSubmit={handleCustomSubmit} className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs font-mono">
          {([
            ["order_id", "Order ID", "text"],
            ["buyer_id", "Buyer ID", "text"],
            ["seller_id", "Seller ID", "text"],
            ["amount", "Order Amount ($)", "number", "any"],
            ["buyer_age_days", "Buyer Age (Days)", "number"],
            ["buyer_orders_before", "Prior Orders", "number"],
            ["device_shared_buyer_count", "Shared Devices Count", "number", "0.1"],
            ["share_degree", "Graph Share Degree", "number", "0.1"],
          ] as const).map(([field, label, type, step]) => (
            <div key={field}>
              <label className="text-[11px] text-[#8995A3] block mb-1">{label}</label>
              <input
                type={type}
                step={step}
                value={(formData[field as keyof TransactionScoreRequest] as string | number) ?? ""}
                onChange={(e) => handleInputChange(field as keyof TransactionScoreRequest, type === "number" ? (parseFloat(e.target.value) || 0) : e.target.value)}
                className="w-full bg-[#0E131A] border border-[#202A35] rounded-md px-3 py-1.5 text-[#E8EDF3] focus:border-blue-500 focus:outline-none"
              />
            </div>
          ))}
        </form>
      </div>
    </div>
  );
}

export default function TransactionAnalyzerPage() {
  return (
    <Suspense fallback={<div className="p-12 text-center text-xs font-mono text-[#8995A3]">Loading Risk Analyzer...</div>}>
      <TransactionAnalyzerContent />
    </Suspense>
  );
}
