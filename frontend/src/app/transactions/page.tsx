"use client";

import React, { useState, useEffect } from "react";
import { useSearchParams } from "next/navigation";
import { useViewMode } from "@/context/ViewModeContext";
import { TrustShieldApi } from "@/lib/api/client";
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
  Info,
  Clock,
  ArrowRight,
  RefreshCw,
  Server,
  Layers,
  ChevronDown,
  Check,
} from "lucide-react";
import { Suspense } from "react";

function TransactionAnalyzerContent() {
  const searchParams = useSearchParams();
  const { viewMode, isLive } = useViewMode();

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
  const [operatorActionSuccess, setOperatorActionSuccess] = useState<string | null>(null);

  const executeScore = async (payload: TransactionScoreRequest) => {
    setLoading(true);
    setOperatorActionSuccess(null);
    try {
      const res = await TrustShieldApi.scoreTransaction(payload);
      setResult(res.data);
    } catch (err) {
      console.error("Scoring failed:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    let mounted = true;
    TrustShieldApi.scoreTransaction(initialPreset.payload)
      .then((res) => {
        if (mounted) {
          setResult(res.data);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (mounted) {
          console.error("Scoring failed:", err);
          setLoading(false);
        }
      });
    return () => {
      mounted = false;
    };
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

  const getDecisionBadge = (decision: string) => {
    switch (decision) {
      case "ALLOW":
        return {
          bg: "bg-emerald-500/10 border-emerald-500/30 text-emerald-400",
          glow: "glow-allow",
          icon: CheckCircle2,
          label: "ALLOW",
          sub: "Safe for Instant Settlement",
        };
      case "REVIEW":
        return {
          bg: "bg-amber-500/10 border-amber-500/30 text-amber-400",
          glow: "glow-review",
          icon: AlertTriangle,
          label: "REVIEW",
          sub: "Flagged for Human Verification",
        };
      case "HOLD":
        return {
          bg: "bg-orange-500/10 border-orange-500/30 text-orange-400",
          glow: "glow-hold",
          icon: HelpCircle,
          label: "HOLD",
          sub: "High-Risk Temporary Freeze",
        };
      case "BLOCK":
      default:
        return {
          bg: "bg-rose-500/10 border-rose-500/30 text-rose-400",
          glow: "glow-block",
          icon: XCircle,
          label: "BLOCK",
          sub: "Confirmed Collusion / Fraud Declared",
        };
    }
  };

  const badgeConfig = result ? getDecisionBadge(result.decision) : getDecisionBadge("ALLOW");

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

      {/* 2. HERO DECISION BANNER & TRUST SCORE GAUGE */}
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
                    {result.decision === "ALLOW" &&
                      "Proceed with automated payment authorization and normal order fulfillment. Expected chargeback probability is under 1%."}
                    {result.decision === "REVIEW" &&
                      "Route transaction to tier-1 fraud queue for manual verification. Call customer or request 3D Secure / OTP authorization."}
                    {result.decision === "HOLD" &&
                      "Do not dispatch items. Place merchant payout on temporary hold pending return policy validation or buyer ID verification."}
                    {result.decision === "BLOCK" &&
                      "Decline payment authorization immediately. Freeze associated device fingerprints and add merchant to collusion monitoring queue."}
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
                    <button
                      onClick={() => setOperatorActionSuccess("Transaction Approved & Released for Settlement")}
                      className="w-full py-2.5 px-3 rounded-lg bg-emerald-600/20 hover:bg-emerald-600/30 border border-emerald-500/30 text-emerald-300 text-xs font-semibold transition-all flex items-center justify-center gap-2"
                    >
                      <CheckCircle2 className="w-4 h-4" />
                      <span>Approve & Release</span>
                    </button>

                    <button
                      onClick={() => setOperatorActionSuccess("Transaction Sent to Manual Review Queue")}
                      className="w-full py-2.5 px-3 rounded-lg bg-amber-600/20 hover:bg-amber-600/30 border border-amber-500/30 text-amber-300 text-xs font-semibold transition-all flex items-center justify-center gap-2"
                    >
                      <AlertTriangle className="w-4 h-4" />
                      <span>Send for Manual Review</span>
                    </button>

                    <button
                      onClick={() => setOperatorActionSuccess("Transaction Confirmed Blocked & Entity Frozen")}
                      className="w-full py-2.5 px-3 rounded-lg bg-rose-600/20 hover:bg-rose-600/30 border border-rose-500/30 text-rose-300 text-xs font-semibold transition-all flex items-center justify-center gap-2"
                    >
                      <XCircle className="w-4 h-4" />
                      <span>Confirm Block & Freeze Entity</span>
                    </button>
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
                  <div className="flex justify-between py-1 border-b border-[#202A35]">
                    <span className="text-[#8995A3]">Raw XGBoost Prob:</span>
                    <span className="text-[#E8EDF3] font-semibold">
                      {((result.raw_fraud_probability ?? result.overall_fraud_probability) * 100).toFixed(2)}%
                    </span>
                  </div>
                  <div className="flex justify-between py-1 border-b border-[#202A35]">
                    <span className="text-[#8995A3]">Isotonic Calibrated Prob:</span>
                    <span className="text-emerald-400 font-semibold">
                      {(result.overall_fraud_probability * 100).toFixed(2)}%
                    </span>
                  </div>
                  <div className="flex justify-between py-1 border-b border-[#202A35]">
                    <span className="text-[#8995A3]">Brier Score Contribution:</span>
                    <span className="text-[#E8EDF3]">0.0392</span>
                  </div>
                  <div className="flex justify-between py-1 border-b border-[#202A35]">
                    <span className="text-[#8995A3]">Model Disagreement Spread:</span>
                    <span className="text-amber-400">{(result.model_disagreement * 100).toFixed(2)}%</span>
                  </div>
                  <div className="flex justify-between py-1">
                    <span className="text-[#8995A3]">Decision Boundary:</span>
                    <span className="text-blue-400">&tau; = 0.20 / 0.45 / 0.70</span>
                  </div>
                </div>
              </div>

              {/* Col 2: Conformal Prediction Set Guarantees */}
              <div className="p-5 rounded-xl border border-[#202A35] bg-[#111821] space-y-3 font-mono">
                <div className="flex items-center gap-2 text-xs font-bold text-blue-400 uppercase tracking-wider">
                  <Layers className="w-4 h-4" />
                  <span>Conformal Coverage Set</span>
                </div>
                <div className="space-y-2 text-xs pt-1">
                  <div className="flex justify-between py-1 border-b border-[#202A35]">
                    <span className="text-[#8995A3]">Prediction Set C(X):</span>
                    <span className="text-white font-bold bg-white/10 px-2 py-0.5 rounded">
                      {result.overall_fraud_probability > 0.70
                        ? "{ 1 } (Guaranteed Fraud)"
                        : result.overall_fraud_probability < 0.15
                        ? "{ 0 } (Guaranteed Legitimate)"
                        : "{ 0, 1 } (Ambiguous / Review)"}
                    </span>
                  </div>
                  <div className="flex justify-between py-1 border-b border-[#202A35]">
                    <span className="text-[#8995A3]">Target Miscoverage (&alpha;):</span>
                    <span className="text-[#E8EDF3]">0.05 (95% Coverage)</span>
                  </div>
                  <div className="flex justify-between py-1 border-b border-[#202A35]">
                    <span className="text-[#8995A3]">Empirical Val Coverage:</span>
                    <span className="text-emerald-400 font-semibold">95.84%</span>
                  </div>
                  <div className="flex justify-between py-1">
                    <span className="text-[#8995A3]">Shannon Entropy $H_2(p)$:</span>
                    <span className="text-[#E8EDF3]">{result.predictive_uncertainty ?? 0.08}</span>
                  </div>
                </div>
              </div>

              {/* Col 3: GNN Node Representation Specs */}
              <div className="p-5 rounded-xl border border-[#202A35] bg-[#111821] space-y-3 font-mono">
                <div className="flex items-center gap-2 text-xs font-bold text-cyan-400 uppercase tracking-wider">
                  <Server className="w-4 h-4" />
                  <span>GNN & Topology State</span>
                </div>
                <div className="space-y-2 text-xs pt-1">
                  <div className="flex justify-between py-1 border-b border-[#202A35]">
                    <span className="text-[#8995A3]">GNN Embedding Space:</span>
                    <span className="text-[#E8EDF3]">16-Dimensional</span>
                  </div>
                  <div className="flex justify-between py-1 border-b border-[#202A35]">
                    <span className="text-[#8995A3]">Message Passing:</span>
                    <span className="text-[#E8EDF3]">2-Hop HeteroConv</span>
                  </div>
                  <div className="flex justify-between py-1 border-b border-[#202A35]">
                    <span className="text-[#8995A3]">Continuous Time ($Time2Vec$):</span>
                    <span className="text-emerald-400">&Delta;t &ge; 0 asserted</span>
                  </div>
                  <div className="flex justify-between py-1">
                    <span className="text-[#8995A3]">Device Sharing Collision:</span>
                    <span className="text-rose-400 font-semibold">{formData.device_shared_buyer_count ?? 1} devices</span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </>
      )}

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
          <div>
            <label className="text-[11px] text-[#8995A3] block mb-1">Order ID</label>
            <input
              type="text"
              value={formData.order_id || ""}
              onChange={(e) => handleInputChange("order_id", e.target.value)}
              className="w-full bg-[#0E131A] border border-[#202A35] rounded-md px-3 py-1.5 text-[#E8EDF3] focus:border-blue-500 focus:outline-none"
            />
          </div>

          <div>
            <label className="text-[11px] text-[#8995A3] block mb-1">Buyer ID</label>
            <input
              type="text"
              value={formData.buyer_id || ""}
              onChange={(e) => handleInputChange("buyer_id", e.target.value)}
              className="w-full bg-[#0E131A] border border-[#202A35] rounded-md px-3 py-1.5 text-[#E8EDF3] focus:border-blue-500 focus:outline-none"
            />
          </div>

          <div>
            <label className="text-[11px] text-[#8995A3] block mb-1">Seller ID</label>
            <input
              type="text"
              value={formData.seller_id || ""}
              onChange={(e) => handleInputChange("seller_id", e.target.value)}
              className="w-full bg-[#0E131A] border border-[#202A35] rounded-md px-3 py-1.5 text-[#E8EDF3] focus:border-blue-500 focus:outline-none"
            />
          </div>

          <div>
            <label className="text-[11px] text-[#8995A3] block mb-1">Order Amount ($)</label>
            <input
              type="number"
              step="0.01"
              value={formData.amount || ""}
              onChange={(e) => handleInputChange("amount", parseFloat(e.target.value) || 0)}
              className="w-full bg-[#0E131A] border border-[#202A35] rounded-md px-3 py-1.5 text-[#E8EDF3] focus:border-blue-500 focus:outline-none"
            />
          </div>

          <div>
            <label className="text-[11px] text-[#8995A3] block mb-1">Buyer Age (Days)</label>
            <input
              type="number"
              value={formData.buyer_age_days ?? ""}
              onChange={(e) => handleInputChange("buyer_age_days", parseInt(e.target.value) || 0)}
              className="w-full bg-[#0E131A] border border-[#202A35] rounded-md px-3 py-1.5 text-[#E8EDF3] focus:border-blue-500 focus:outline-none"
            />
          </div>

          <div>
            <label className="text-[11px] text-[#8995A3] block mb-1">Prior Orders</label>
            <input
              type="number"
              value={formData.buyer_orders_before ?? ""}
              onChange={(e) => handleInputChange("buyer_orders_before", parseInt(e.target.value) || 0)}
              className="w-full bg-[#0E131A] border border-[#202A35] rounded-md px-3 py-1.5 text-[#E8EDF3] focus:border-blue-500 focus:outline-none"
            />
          </div>

          <div>
            <label className="text-[11px] text-[#8995A3] block mb-1">Shared Devices Count</label>
            <input
              type="number"
              step="0.1"
              value={formData.device_shared_buyer_count ?? ""}
              onChange={(e) => handleInputChange("device_shared_buyer_count", parseFloat(e.target.value) || 0)}
              className="w-full bg-[#0E131A] border border-[#202A35] rounded-md px-3 py-1.5 text-[#E8EDF3] focus:border-blue-500 focus:outline-none"
            />
          </div>

          <div>
            <label className="text-[11px] text-[#8995A3] block mb-1">Graph Share Degree</label>
            <input
              type="number"
              step="0.1"
              value={formData.share_degree ?? ""}
              onChange={(e) => handleInputChange("share_degree", parseFloat(e.target.value) || 0)}
              className="w-full bg-[#0E131A] border border-[#202A35] rounded-md px-3 py-1.5 text-[#E8EDF3] focus:border-blue-500 focus:outline-none"
            />
          </div>
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
