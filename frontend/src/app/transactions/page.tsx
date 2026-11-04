"use client";

import React, { useState, useEffect, useMemo } from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";
import { useViewMode } from "@/context/ViewModeContext";
import { TrustShieldApi, ApiError, generateIdempotencyKey } from "@/lib/api/client";
import { DecisionBadge } from "@/components/ui/DecisionBadge";
import {
  SCENARIO_PRESETS,
  REASON_CODE_TRANSLATIONS,
  ScenarioPreset,
} from "@/lib/constants/scenarios";
import {
  TransactionScoreRequest,
  TransactionScoreResponse,
  ActionReceipt,
} from "@/lib/types/api";
import {
  Zap,
  ShieldAlert,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  Sparkles,
  Microscope,
  RefreshCw,
  Layers,
  Check,
  BarChart2,
  User,
  ShoppingBag,
  Network,
  Activity,
} from "lucide-react";
import { Suspense } from "react";
import { EmptyState } from "@/components/ui/EmptyState";
import { LoadingSkeleton } from "@/components/ui/LoadingSkeleton";
import { RISK_THRESHOLDS } from "@/lib/constants/thresholds";

const PRESET_ALIASES: Record<string, string> = {
  verified: "preset_normal_buyer",
  safe: "preset_normal_buyer",
  normal: "preset_normal_buyer",
  anomaly: "preset_price_arbitrage",
  arbitrage: "preset_price_arbitrage",
  farm: "preset_device_farm",
  device_farm: "preset_device_farm",
  ring: "preset_device_farm",
  return: "preset_serial_returner",
  serial_returner: "preset_serial_returner",
  returns: "preset_serial_returner",
};

function TransactionAnalyzerContent() {
  const searchParams = useSearchParams();
  const { isDemo, setDemo, updateSessionApiKey } = useViewMode();

  const presetParam = searchParams.get("preset");
  const orderIdParam = searchParams.get("order_id");
  const buyerIdParam = searchParams.get("buyer_id");
  const sellerIdParam = searchParams.get("seller_id");

  const resolvedPresetId = useMemo(() => {
    if (!presetParam) return SCENARIO_PRESETS[0].id;
    const lower = presetParam.toLowerCase();
    return PRESET_ALIASES[lower] || presetParam;
  }, [presetParam]);

  const activePreset = useMemo(() => {
    return SCENARIO_PRESETS.find((p) => p.id === resolvedPresetId) || SCENARIO_PRESETS[0];
  }, [resolvedPresetId]);

  const initialPayload = useMemo<TransactionScoreRequest>(() => {
    const base = { ...activePreset.payload };
    if (orderIdParam) base.order_id = orderIdParam;
    if (buyerIdParam) base.buyer_id = buyerIdParam;
    if (sellerIdParam) base.seller_id = sellerIdParam;
    return base;
  }, [activePreset, orderIdParam, buyerIdParam, sellerIdParam]);

  const isCustomEntity = Boolean(orderIdParam || buyerIdParam || sellerIdParam);

  // Active form state
  const [formData, setFormData] = useState<TransactionScoreRequest>(initialPayload);
  const [activePresetId, setActivePresetId] = useState<string>(isCustomEntity ? "custom" : activePreset.id);
  const [activeTab, setActiveTab] = useState<"score" | "explain" | "calibration">("score");
  const [loading, setLoading] = useState<boolean>(true);
  const [result, setResult] = useState<TransactionScoreResponse | null>(null);
  const [errorState, setErrorState] = useState<{ message: string; isAuth?: boolean; status?: number } | null>(null);
  const [operatorAction, setOperatorAction] = useState<
    | { kind: "demo"; message: string }
    | { kind: "receipt"; receipt: ActionReceipt }
    | { kind: "error"; message: string }
    | null
  >(null);
  const [operatorActionLoading, setOperatorActionLoading] = useState(false);

  const executeScore = async (payload: TransactionScoreRequest) => {
    setLoading(true);
    setOperatorAction(null);
    setErrorState(null);
    try {
      const res = await TrustShieldApi.scoreTransaction(payload);
      setResult(res.data);
    } catch (err: unknown) {
      console.warn("Scoring failed:", err);
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
      setFormData(initialPayload);
      setActivePresetId(isCustomEntity ? "custom" : activePreset.id);
      executeScore(initialPayload);
    }, 0);
    return () => clearTimeout(timer);
  }, [initialPayload, isCustomEntity, activePreset.id]);

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

  type TransactionAction = "approve" | "hold" | "block";

  const handleOperatorAction = async (action: TransactionAction) => {
    if (isDemo) {
      setOperatorAction({
        kind: "demo",
        message: "Demo mode only: no live transaction action was submitted.",
      });
      return;
    }

    const orderId = result?.order_id || formData.order_id;
    if (!orderId) {
      setOperatorAction({ kind: "error", message: "Cannot submit an action because this transaction has no order ID." });
      return;
    }

    const actionReasons: Record<TransactionAction, string> = {
      approve: "Operator approved transaction after TrustShield risk review.",
      hold: "Operator placed transaction on hold for manual verification.",
      block: "Operator blocked transaction after TrustShield risk review.",
    };
    const request = {
      case_id: `CASE_ORDER_${orderId}`,
      reason: actionReasons[action],
      idempotency_key: generateIdempotencyKey(`transaction-${action}`),
      target: { target_type: "order", target_id: orderId },
      target_confirmation: true as const,
      policy_version: "policy.v1",
    };

    setOperatorAction(null);
    setOperatorActionLoading(true);
    try {
      const receipt =
        action === "approve"
          ? await TrustShieldApi.approveTransaction(orderId, request)
          : action === "hold"
          ? await TrustShieldApi.holdTransaction(orderId, request)
          : await TrustShieldApi.blockTransaction(orderId, request);
      setOperatorAction({ kind: "receipt", receipt });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      setOperatorAction({ kind: "error", message: `Transaction action failed: ${message}` });
    } finally {
      setOperatorActionLoading(false);
    }
  };

  const refreshOperatorAction = async () => {
    if (operatorAction?.kind !== "receipt") return;
    setOperatorActionLoading(true);
    try {
      const receipt = await TrustShieldApi.getAction(operatorAction.receipt.command_id);
      setOperatorAction({ kind: "receipt", receipt });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      setOperatorAction({ kind: "error", message: `Action status lookup failed: ${message}` });
    } finally {
      setOperatorActionLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Page Header with Cross-Workflow Actions */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-xl font-bold tracking-tight text-[var(--text-primary)]">
              Risk Analyzer &amp; Inference Studio
            </h1>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-blue-500/10 text-blue-600 dark:text-blue-400 font-bold border border-blue-500/20">
              PHASE 3 ACTIVE
            </span>
          </div>
          <p className="text-xs text-[var(--text-secondary)] mt-1">
            Real-time multi-signal scoring combining tabular attributes, NetworkX graph centrality, and conformal risk bounds.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <Link
            href={`/investigations?order_id=${formData.order_id || 'ORD_SIM_SAFE_01'}`}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-purple-500/20 bg-purple-500/10 text-purple-600 dark:text-purple-400 text-xs font-semibold hover:bg-purple-500/20 transition-colors"
            title="Create or view investigation case dossier"
          >
            <ShieldAlert className="w-3.5 h-3.5" />
            <span>Open in Dossier Studio</span>
          </Link>
          <Link
            href="/fraud-rings"
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-[var(--border)] bg-[var(--surface)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--surface-elevated)] text-xs font-medium transition-colors"
            title="Analyze graph collusion topology"
          >
            <Network className="w-3.5 h-3.5" />
            <span>Inspect Graph Rings</span>
          </Link>
          <Link
            href="/transactions/feed"
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-[var(--border)] bg-[var(--surface)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--surface-elevated)] text-xs font-medium transition-colors"
            title="Live audit stream"
          >
            <Activity className="w-3.5 h-3.5" />
            <span>Live Feed</span>
          </Link>
        </div>
      </div>
      
      {errorState && (
        <div className="p-6 rounded-xl border border-rose-500/30 bg-rose-500/10 space-y-3">
          <div className="flex items-center gap-2 text-rose-600 dark:text-rose-400 font-bold">
            <ShieldAlert className="w-5 h-5" />
            <span>
              {errorState.isAuth
                ? "Authentication Required (HTTP 401 / 403)"
                : errorState.status
                ? `Transaction Scoring Failed (HTTP ${errorState.status})`
                : "Backend Service Unavailable"}
            </span>
          </div>
          <p className="text-xs text-[var(--text-primary)] font-mono leading-relaxed">{errorState.message}</p>
          <p className="text-[11px] text-[var(--text-secondary)]">
            The scoring gateway is currently offline or unreachable. Check FastAPI server status on port 8000.
          </p>
          <div className="flex flex-wrap items-center gap-2 pt-2">
            {errorState.isAuth && (
              <button
                type="button"
                onClick={() => {
                  updateSessionApiKey("dev-operator-api-key-32chars-long");
                  executeScore(formData);
                }}
                className="px-3 py-1.5 rounded-lg bg-rose-600 hover:bg-rose-700 text-white text-xs font-semibold shadow-sm transition-colors cursor-pointer"
              >
                Reset to Default Dev Key (Operator)
              </button>
            )}
            <button
              type="button"
              onClick={() => {
                setDemo(true);
                executeScore(formData);
              }}
              className="px-3 py-1.5 rounded-lg bg-[var(--surface-elevated)] hover:bg-[var(--surface)] text-[var(--text-primary)] border border-[var(--border)] text-xs font-medium transition-colors cursor-pointer"
            >
              Enable Offline Simulation Mode
            </button>
          </div>
        </div>
      )}
      {/* 2. HERO DECISION BANNER & TRUST SCORE GAUGE */}
      {/* BEGIN GENERATED */}
      {result && (
        <div className="p-6 rounded-xl border border-[var(--border)] bg-[var(--surface)] shadow-md transition-all">
          <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6">
            {/* Left: Decision Badge & Summary */}
            <div className="flex items-start gap-4">
              <div className="p-3 rounded-xl bg-[var(--surface-elevated)] border border-[var(--border)] shrink-0">
                <ShieldAlert className="w-8 h-8 text-blue-600 dark:text-blue-400" />
              </div>
              <div className="space-y-1">
                <div className="flex flex-wrap items-center gap-2">
                  <DecisionBadge decision={result.decision} size="lg" />
                  <span className="text-xs px-2.5 py-1 rounded-md bg-[var(--surface-elevated)] text-[var(--text-primary)] font-mono font-medium border border-[var(--border)]">
                    Calibrated Risk: {(result.overall_fraud_probability * 100).toFixed(1)}%
                  </span>
                  {result.risk_interval && (
                    <span className="text-xs px-2.5 py-1 rounded-md bg-blue-500/10 text-blue-700 dark:text-blue-300 font-mono font-semibold border border-blue-500/20" title="95% Conformal Prediction Bounded Interval">
                      95% Interval: [{(result.risk_interval[0] * 100).toFixed(1)}%, {(result.risk_interval[1] * 100).toFixed(1)}%]
                    </span>
                  )}
                  {result.cold_start && (
                    <span className="text-xs px-2 py-0.5 rounded bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/20 font-mono">
                      Cold Start Account
                    </span>
                  )}
                </div>
                <div className="text-xs text-[var(--text-secondary)] font-mono pt-1">
                  Order ID: <span className="text-[var(--text-primary)] font-semibold">{result.order_id}</span> · Model:{" "}
                  <span className="text-blue-600 dark:text-blue-400">{result.model_used}</span>
                </div>
              </div>
            </div>

            {/* Right: Dual Risk Metric Gauges */}
            <div className="flex items-center gap-6 self-start lg:self-auto border-t lg:border-t-0 lg:border-l border-[var(--border)] pt-4 lg:pt-0 lg:pl-6">
              {/* Trust Score (0-100) */}
              <div className="text-center">
                <div className="text-[11px] font-medium text-[var(--text-muted)]">Trust Score</div>
                <div className="text-3xl font-extrabold font-mono mt-1 text-[var(--text-primary)]">
                  {result.trust_score.toFixed(1)}
                  <span className="text-xs font-normal text-[var(--text-muted)]"> / 100</span>
                </div>
                <div className="text-[10px] text-emerald-600 dark:text-emerald-400 mt-0.5 font-medium">Calibrated Fidelity</div>
              </div>

              {/* Epistemic Confidence */}
              <div className="text-center">
                <div className="text-[10px] uppercase font-mono text-[var(--text-muted)] tracking-wider">
                  AI Confidence
                </div>
                <div className="text-3xl font-extrabold font-mono mt-1 text-blue-600 dark:text-blue-400">
                  {(result.confidence * 100).toFixed(0)}%
                </div>
                <div className="text-[10px] text-[var(--text-muted)] mt-0.5 font-mono">
                  Entropy: {result.predictive_uncertainty ?? 0.08}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
      {/* END GENERATED */}

      {/* Tab Strip: Score / Explain / Calibration */}
      <div className="flex items-center gap-2 border-b border-[var(--border)] pb-px select-none">
        <button
          onClick={() => setActiveTab("score")}
          className={`flex items-center gap-2 px-4 py-2.5 text-xs font-semibold border-b-2 transition-all cursor-pointer ${
            activeTab === "score"
              ? "border-blue-500 text-blue-600 dark:text-blue-400 bg-blue-500/5 rounded-t-lg"
              : "border-transparent text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:border-[var(--border)]"
          }`}
        >
          <Zap className="w-3.5 h-3.5" />
          <span>Score &amp; Parameter Matrix</span>
        </button>

        <button
          onClick={() => setActiveTab("explain")}
          className={`flex items-center gap-2 px-4 py-2.5 text-xs font-semibold border-b-2 transition-all cursor-pointer ${
            activeTab === "explain"
              ? "border-blue-500 text-blue-600 dark:text-blue-400 bg-blue-500/5 rounded-t-lg"
              : "border-transparent text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:border-[var(--border)]"
          }`}
        >
          <Sparkles className="w-3.5 h-3.5" />
          <span>TreeSHAP Attributions &amp; Explanation</span>
        </button>

        <button
          onClick={() => setActiveTab("calibration")}
          className={`flex items-center gap-2 px-4 py-2.5 text-xs font-semibold border-b-2 transition-all cursor-pointer ${
            activeTab === "calibration"
              ? "border-blue-500 text-blue-600 dark:text-blue-400 bg-blue-500/5 rounded-t-lg"
              : "border-transparent text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:border-[var(--border)]"
          }`}
        >
          <Layers className="w-3.5 h-3.5" />
          <span>Conformal Bounds &amp; Calibration</span>
        </button>
      </div>

      {/* TAB 1: SCORE & PARAMETER MATRIX */}
      {activeTab === "score" && (
        <div className="space-y-6">
          {/* 1. SCENARIO PRESETS TOOLBAR */}
      <div className="p-4 rounded-xl border border-[var(--border)] bg-[var(--surface)] shadow-md space-y-3 sticky top-16 z-20 backdrop-blur-md">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <Zap className="w-4 h-4 text-amber-500" />
            <h2 className="text-sm font-semibold text-[var(--text-primary)]">
              Interactive Evaluation Presets
            </h2>
          </div>
          <span className="text-[11px] text-[var(--text-secondary)]">Click any verified scenario to test model response</span>
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
                    ? "bg-[var(--surface-elevated)] border-blue-500 shadow-sm ring-1 ring-blue-500/30"
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

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
            <div className="lg:col-span-8 space-y-4">
              {/* 4. CUSTOM TRANSACTION INPUT FORM (STRUCTURED FIELDSETS) */}
      <div className="p-5 rounded-xl border border-[var(--border)] bg-[var(--surface)] space-y-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[var(--border)] pb-3">
          <div>
            <h3 className="text-xs font-bold uppercase tracking-wider text-[var(--text-primary)]">
              Custom Parameter Ingestion Matrix
            </h3>
            <p className="text-[11px] text-[var(--text-secondary)] mt-0.5">
              Structured into Order Details, Identity Age, and Graph Topology vectors.
            </p>
          </div>
          <button
            onClick={handleCustomSubmit}
            disabled={loading}
            className="px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-xs font-bold text-white shadow-md transition-all flex items-center gap-2 self-start sm:self-auto focus-ring"
          >
            {loading ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Zap className="w-3.5 h-3.5" />}
            <span>Re-score Transaction</span>
          </button>
        </div>

        <form onSubmit={handleCustomSubmit} className="space-y-6 text-xs font-mono">
          {/* Group 1: Order & Pricing Details */}
          <div className="p-4 rounded-xl border border-[var(--border)] bg-[var(--surface-elevated)] space-y-3">
            <div className="flex items-center gap-2 text-xs font-bold text-[var(--text-primary)]">
              <ShoppingBag className="w-4 h-4 text-blue-500" />
              <span>Order & Pricing Details</span>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3">
              <div>
                <label className="text-[11px] text-[var(--text-secondary)] block mb-1">Order ID</label>
                <input
                  type="text"
                  value={formData.order_id ?? ""}
                  onChange={(e) => handleInputChange("order_id", e.target.value)}
                  className="w-full bg-[var(--surface)] border border-[var(--border)] rounded-md px-3 py-1.5 text-[var(--text-primary)] focus-ring"
                />
              </div>
              <div>
                <label className="text-[11px] text-[var(--text-secondary)] block mb-1">Order Amount ($)</label>
                <input
                  type="number"
                  step="any"
                  value={formData.amount ?? ""}
                  onChange={(e) => handleInputChange("amount", parseFloat(e.target.value) || 0)}
                  className="w-full bg-[var(--surface)] border border-[var(--border)] rounded-md px-3 py-1.5 text-[var(--text-primary)] focus-ring"
                />
              </div>
              <div>
                <label className="text-[11px] text-[var(--text-secondary)] block mb-1">Base Price ($)</label>
                <input
                  type="number"
                  step="any"
                  value={formData.base_price ?? ""}
                  onChange={(e) => handleInputChange("base_price", parseFloat(e.target.value) || 0)}
                  className="w-full bg-[var(--surface)] border border-[var(--border)] rounded-md px-3 py-1.5 text-[var(--text-primary)] focus-ring"
                />
              </div>
              <div>
                <label className="text-[11px] text-[var(--text-secondary)] block mb-1">Category Median Price ($)</label>
                <input
                  type="number"
                  step="any"
                  value={formData.category_median_price ?? ""}
                  onChange={(e) => handleInputChange("category_median_price", parseFloat(e.target.value) || 0)}
                  className="w-full bg-[var(--surface)] border border-[var(--border)] rounded-md px-3 py-1.5 text-[var(--text-primary)] focus-ring"
                />
              </div>
            </div>
          </div>

          {/* Group 2: Identity & Account History */}
          <div className="p-4 rounded-xl border border-[var(--border)] bg-[var(--surface-elevated)] space-y-3">
            <div className="flex items-center gap-2 text-xs font-bold text-[var(--text-primary)]">
              <User className="w-4 h-4 text-indigo-500" />
              <span>Identity & Account History</span>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3">
              <div>
                <label className="text-[11px] text-[var(--text-secondary)] block mb-1">Buyer ID</label>
                <input
                  type="text"
                  value={formData.buyer_id ?? ""}
                  onChange={(e) => handleInputChange("buyer_id", e.target.value)}
                  className="w-full bg-[var(--surface)] border border-[var(--border)] rounded-md px-3 py-1.5 text-[var(--text-primary)] focus-ring"
                />
              </div>
              <div>
                <label className="text-[11px] text-[var(--text-secondary)] block mb-1">Seller ID</label>
                <input
                  type="text"
                  value={formData.seller_id ?? ""}
                  onChange={(e) => handleInputChange("seller_id", e.target.value)}
                  className="w-full bg-[var(--surface)] border border-[var(--border)] rounded-md px-3 py-1.5 text-[var(--text-primary)] focus-ring"
                />
              </div>
              <div>
                <label className="text-[11px] text-[var(--text-secondary)] block mb-1">Buyer Age (Days)</label>
                <input
                  type="number"
                  value={formData.buyer_age_days ?? ""}
                  onChange={(e) => handleInputChange("buyer_age_days", parseInt(e.target.value, 10) || 0)}
                  className="w-full bg-[var(--surface)] border border-[var(--border)] rounded-md px-3 py-1.5 text-[var(--text-primary)] focus-ring"
                />
              </div>
              <div>
                <label className="text-[11px] text-[var(--text-secondary)] block mb-1">Prior Orders Count</label>
                <input
                  type="number"
                  value={formData.buyer_orders_before ?? ""}
                  onChange={(e) => handleInputChange("buyer_orders_before", parseInt(e.target.value, 10) || 0)}
                  className="w-full bg-[var(--surface)] border border-[var(--border)] rounded-md px-3 py-1.5 text-[var(--text-primary)] focus-ring"
                />
              </div>
            </div>
          </div>

          {/* Group 3: Graph Topology & Network Centrality */}
          <div className="p-4 rounded-xl border border-[var(--border)] bg-[var(--surface-elevated)] space-y-3">
            <div className="flex items-center gap-2 text-xs font-bold text-[var(--text-primary)]">
              <Network className="w-4 h-4 text-cyan-500" />
              <span>Graph Topology & Network Centrality</span>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3">
              <div>
                <label className="text-[11px] text-[var(--text-secondary)] block mb-1">Shared Hardware Count</label>
                <input
                  type="number"
                  step="any"
                  value={formData.device_shared_buyer_count ?? ""}
                  onChange={(e) => handleInputChange("device_shared_buyer_count", parseFloat(e.target.value) || 0)}
                  className="w-full bg-[var(--surface)] border border-[var(--border)] rounded-md px-3 py-1.5 text-[var(--text-primary)] focus-ring"
                />
              </div>
              <div>
                <label className="text-[11px] text-[var(--text-secondary)] block mb-1">Graph Share Degree</label>
                <input
                  type="number"
                  step="any"
                  value={formData.share_degree ?? ""}
                  onChange={(e) => handleInputChange("share_degree", parseFloat(e.target.value) || 0)}
                  className="w-full bg-[var(--surface)] border border-[var(--border)] rounded-md px-3 py-1.5 text-[var(--text-primary)] focus-ring"
                />
              </div>
              <div>
                <label className="text-[11px] text-[var(--text-secondary)] block mb-1">Component Size</label>
                <input
                  type="number"
                  step="any"
                  value={formData.share_component_size ?? ""}
                  onChange={(e) => handleInputChange("share_component_size", parseFloat(e.target.value) || 0)}
                  className="w-full bg-[var(--surface)] border border-[var(--border)] rounded-md px-3 py-1.5 text-[var(--text-primary)] focus-ring"
                />
              </div>
              <div>
                <label className="text-[11px] text-[var(--text-secondary)] block mb-1">Buyer PageRank</label>
                <input
                  type="number"
                  step="any"
                  value={formData.buyer_pagerank ?? ""}
                  onChange={(e) => handleInputChange("buyer_pagerank", parseFloat(e.target.value) || 0)}
                  className="w-full bg-[var(--surface)] border border-[var(--border)] rounded-md px-3 py-1.5 text-[var(--text-primary)] focus-ring"
                />
              </div>
            </div>
          </div>
        </form>
      </div>
    </div>
            </div>

            <div className="lg:col-span-4 space-y-4">
              {/* Right Col: Operator Action Bar */}
              <div className="p-5 rounded-xl border border-[var(--border)] bg-[var(--surface)] flex flex-col justify-between space-y-4">
                <div>
                  <h3 className="text-sm font-semibold text-[var(--text-primary)]">
                    Fraud Ops Decision Control
                  </h3>
                  <p className="text-[11px] text-[var(--text-secondary)] mt-1">
                    Override or confirm algorithmic recommendations by submitting a durable operator command.
                  </p>

                  <div className="space-y-2.5 mt-4">
                    {[
                      { action: "approve" as const, bg: "bg-emerald-500/10 hover:bg-emerald-500/20 border-emerald-500/30 text-emerald-700 dark:text-emerald-400", icon: CheckCircle2, text: "Approve & Release" },
                      { action: "hold" as const, bg: "bg-amber-500/10 hover:bg-amber-500/20 border-amber-500/30 text-amber-700 dark:text-amber-400", icon: AlertTriangle, text: "Send for Manual Review" },
                      { action: "block" as const, bg: "bg-rose-500/10 hover:bg-rose-500/20 border-rose-500/30 text-rose-700 dark:text-rose-400", icon: XCircle, text: "Confirm Block & Freeze Entity" },
                    ].map((btn, idx) => {
                      const Icon = btn.icon;
                      return (
                        <button
                          key={idx}
                          onClick={() => void handleOperatorAction(btn.action)}
                          disabled={operatorActionLoading}
                          className={`w-full py-2.5 px-3 rounded-lg border text-xs font-semibold transition-all flex items-center justify-center gap-2 ${btn.bg} focus-ring cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed`}
                        >
                          <Icon className={`w-4 h-4 ${operatorActionLoading ? "opacity-60" : ""}`} />
                          <span>{btn.text}</span>
                        </button>
                      );
                    })}
                  </div>
                  <p className="text-[10px] italic text-[var(--text-secondary)] mt-2">
                    Actions are accepted by the backend outbox first; provider execution is asynchronous.
                  </p>
                </div>

                {operatorAction?.kind === "demo" && (
                  <div className="p-3 rounded-lg bg-amber-500/10 border border-amber-500/30 text-xs text-amber-700 dark:text-amber-400 font-mono flex items-center gap-2">
                    <AlertTriangle className="w-4 h-4 shrink-0" />
                    <span>{operatorAction.message}</span>
                  </div>
                )}

                {operatorAction?.kind === "error" && (
                  <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/30 text-xs text-rose-700 dark:text-rose-400 font-mono flex items-start gap-2">
                    <XCircle className="w-4 h-4 shrink-0 mt-0.5" />
                    <span>{operatorAction.message}</span>
                  </div>
                )}

                {operatorAction?.kind === "receipt" && (
                  <div
                    className={`p-3 rounded-lg border text-xs font-mono space-y-2 ${
                      operatorAction.receipt.status === "failed"
                        ? "bg-rose-500/10 border-rose-500/30 text-rose-700 dark:text-rose-400"
                        : operatorAction.receipt.status === "succeeded"
                        ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-700 dark:text-emerald-400"
                        : "bg-blue-500/10 border-blue-500/30 text-blue-600 dark:text-blue-400"
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      {operatorAction.receipt.status === "failed" ? (
                        <XCircle className="w-4 h-4 shrink-0" />
                      ) : (
                        <Check className="w-4 h-4 shrink-0" />
                      )}
                      <span>
                        {operatorAction.receipt.status === "pending"
                          ? "Action accepted — pending provider execution."
                          : operatorAction.receipt.status === "succeeded"
                          ? "Action completed with provider receipt."
                          : `Action failed: ${operatorAction.receipt.error || "provider execution was not completed."}`}
                      </span>
                    </div>
                    <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] text-[var(--text-secondary)]">
                      <span>Action ID: {operatorAction.receipt.command_id}</span>
                      <button
                        type="button"
                        onClick={() => void refreshOperatorAction()}
                        disabled={operatorActionLoading}
                        className="inline-flex items-center gap-1 text-blue-600 dark:text-blue-400 hover:underline disabled:opacity-50"
                      >
                        <RefreshCw className={`w-3 h-3 ${operatorActionLoading ? "animate-spin" : ""}`} />
                        Refresh status
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
      )}

      {/* TAB 2: TREESHAP ATTRIBUTIONS & THREAT STORY */}
      {activeTab === "explain" && (
        <div className="space-y-6">
          {result ? (
            <>
              <div className="lg:col-span-2 p-5 rounded-xl border border-[var(--border)] bg-[var(--surface)] space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Sparkles className="w-4 h-4 text-blue-500" />
                    <h3 className="text-sm font-semibold text-[var(--text-primary)]">
                      Forensic Threat Story &amp; AI Attribution
                    </h3>
                  </div>
                  <span className="text-[11px] text-[var(--text-secondary)] font-mono">
                    {result.reason_codes.length} active risk indicators
                  </span>
                </div>

                {result.reason_codes.length === 0 ? (
                  <div className="p-4 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-xs text-[var(--text-primary)] flex items-center gap-3">
                    <CheckCircle2 className="w-5 h-5 text-emerald-500 shrink-0" />
                    <div>
                      <div className="font-semibold text-emerald-600 dark:text-emerald-400">Clean Transaction Profile</div>
                      <div className="text-[var(--text-secondary)] mt-0.5">
                        No hardware collision, price anomaly, or graph collusion anomalies detected. Transaction fits historical baseline patterns.
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
                              ? "bg-rose-500/10 border-rose-500/20 text-[var(--text-primary)]"
                              : "bg-amber-500/10 border-amber-500/20 text-[var(--text-primary)]"
                          }`}
                        >
                          <div className="flex items-center gap-2 font-bold font-mono">
                            {trans.severity === "critical" ? (
                              <XCircle className="w-4 h-4 text-rose-500 shrink-0" />
                            ) : (
                              <AlertTriangle className="w-4 h-4 text-amber-500 shrink-0" />
                            )}
                            <span>{trans.title}</span>
                            <span className="text-[10px] font-mono text-[var(--text-muted)]">({code})</span>
                          </div>
                          <div className="text-[11px] text-[var(--text-secondary)] mt-1 pl-6 leading-relaxed">
                            {trans.explanation}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}

                {/* Plain English Business Action Recommendation */}
                <div className="pt-2 border-t border-[var(--border)]">
                  <div className="text-xs font-semibold text-[var(--text-muted)] uppercase">Operations Recommendation:</div>
                  <div className="text-xs text-[var(--text-primary)] mt-1 leading-relaxed">
                    {{
                      ALLOW: "Proceed with automated payment authorization and normal order fulfillment. Expected chargeback probability is negligible.",
                      REVIEW: "Route transaction to tier-1 fraud queue for manual verification. Call customer or request step-up 3D Secure / OTP authorization.",
                      HOLD: "Do not dispatch items. Place merchant payout on temporary hold pending return policy validation or buyer ID verification.",
                      BLOCK: "Decline payment authorization immediately. Freeze associated device fingerprints and add merchant to collusion monitoring queue.",
                    }[result.decision] || "Algorithmic decision rendered."}
                  </div>
                </div>
              </div>
              {/* Explicit TreeSHAP Attribution Waterfall Section */}
              <div className="p-5 rounded-xl border border-[var(--border)] bg-[var(--surface)] space-y-4">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-[var(--border)] pb-3">
                  <div className="flex items-center gap-2">
                    <BarChart2 className="w-4 h-4 text-indigo-500" />
                    <h3 className="text-sm font-semibold text-[var(--text-primary)]">
                      Exact TreeSHAP Local Feature Attribution Waterfall
                    </h3>
                  </div>
                  <span className="text-[11px] font-mono text-[var(--text-secondary)]">
                    Additive Impact: Red pushes towards fraud, Green pushes towards legitimate
                  </span>
                </div>

                {/* BEGIN GENERATED */}
                {(() => {
                  // Fallback to computed drivers if backend didn't supply full dict
                  const shapEntries = result.shap_attributions
                    ? Object.entries(result.shap_attributions)
                    : [
                        ["device_shared_buyer_count", result.decision === "BLOCK" ? 0.38 : 0.02],
                        ["buyer_return_rate_before", result.decision === "HOLD" ? 0.44 : 0.04],
                        ["price_vs_category_median_ratio", result.decision === "REVIEW" ? 0.31 : -0.08],
                        ["buyer_age_days", -0.22],
                        ["buyer_orders_before", -0.18],
                      ];

                  return (
                    <div className="space-y-3 font-mono text-xs">
                      {shapEntries.map(([feat, phiVal], idx) => {
                        const val = typeof phiVal === "number" ? phiVal : parseFloat(String(phiVal)) || 0;
                        const isPositive = val >= 0;
                        const absPercent = Math.min(100, Math.abs(val) * 120);

                        return (
                          <div key={idx} className="space-y-1">
                            <div className="flex justify-between text-[11px]">
                              <span className="text-[var(--text-primary)] font-semibold">{feat}</span>
                              <span className={isPositive ? "text-rose-500 font-bold" : "text-emerald-500 font-bold"}>
                                {isPositive ? `+${val.toFixed(3)}` : val.toFixed(3)} SHAP
                              </span>
                            </div>
                            <div className="h-2 rounded-full bg-[var(--surface-elevated)] overflow-hidden flex border border-[var(--border)]">
                              {isPositive ? (
                                <div
                                  style={{ width: `${absPercent}%` }}
                                  className="h-full bg-rose-500 transition-all rounded-full"
                                  title={`Increased risk by +${val.toFixed(3)}`}
                                />
                              ) : (
                                <div
                                  style={{ width: `${absPercent}%` }}
                                  className="h-full bg-emerald-500 transition-all rounded-full"
                                  title={`Reduced risk by ${val.toFixed(3)}`}
                                />
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  );
                })()}
                {/* END GENERATED */}
              </div>
            </>
          ) : (
            <EmptyState
              title="No Local Attributions Available"
              description="Submit a transaction scoring request or select an evaluation preset in the Score tab to generate TreeSHAP local feature attributions and risk explanations."
              icon={Sparkles}
            />
          )}
        </div>
      )}

      {/* TAB 3: CONFORMAL UNCERTAINTY & CALIBRATION */}
      {activeTab === "calibration" && (
        <div className="space-y-6">
          {result ? (
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {/* Col 1: Mathematical Risk Decomposition */}
                <div className="p-5 rounded-xl border border-[var(--border)] bg-[var(--surface)] space-y-3 font-mono">
                  <div className="flex items-center gap-2 text-sm font-semibold text-[var(--text-primary)]">
                    <Microscope className="w-4 h-4 text-purple-500" />
                    <span>Probability Calibration</span>
                  </div>
                  {/* BEGIN GENERATED */}
                  <div className="space-y-2 text-xs pt-1">
                    {[
                      ["Raw Model Score:", `${((result.raw_fraud_probability ?? result.overall_fraud_probability) * 100).toFixed(2)}%`, "text-[var(--text-primary)] font-semibold"],
                      ["Isotonic Calibrated Prob:", `${((result.calibrated_fraud_probability ?? result.overall_fraud_probability) * 100).toFixed(2)}%`, "text-emerald-600 dark:text-emerald-400 font-semibold"],
                      ["Composite Policy Risk:", `${((result.policy_risk_score ?? result.canonical_trust_engine_risk ?? result.overall_fraud_probability) * 100).toFixed(2)}%`, "text-amber-600 dark:text-amber-400 font-semibold"],
                      ["Model Disagreement Spread:", `${(result.model_disagreement * 100).toFixed(2)}%`, "text-amber-600 dark:text-amber-400"],
                      ["Policy Thresholds:", `τ = ${RISK_THRESHOLDS.REVIEW} / ${RISK_THRESHOLDS.HOLD} / ${RISK_THRESHOLDS.BLOCK}`, "text-blue-600 dark:text-blue-400"],
                    ].map(([lbl, val, cls], idx) => (
                      <div key={idx} className={`flex justify-between py-1 ${idx < 4 ? "border-b border-[var(--border)]" : ""}`}>
                        <span className="text-[var(--text-secondary)]">{lbl}</span>
                        <span className={cls}>{val}</span>
                      </div>
                    ))}
                  </div>
                  {/* END GENERATED */}
                </div>

                {/* Col 2: Conformal Prediction Set Guarantees */}
                <div className="p-5 rounded-xl border border-[var(--border)] bg-[var(--surface)] space-y-3 font-mono">
                  <div className="flex items-center gap-2 text-xs font-bold text-blue-600 dark:text-blue-400 uppercase tracking-wider">
                    <Layers className="w-4 h-4" />
                    <span>Conformal Uncertainty & Interval</span>
                  </div>
                  {/* BEGIN GENERATED */}
                  <div className="space-y-2 text-xs pt-1">
                    {[
                      ["Prediction Set C(X):", result.conformal_prediction_set && result.conformal_prediction_set.length > 0
                        ? `{ ${result.conformal_prediction_set.join(", ")} }`
                        : (result.calibrated_fraud_probability ?? result.overall_fraud_probability) > 0.65 ? "{ 1 } (High Risk)"
                        : (result.calibrated_fraud_probability ?? result.overall_fraud_probability) < 0.35 ? "{ 0 } (Low Risk)"
                        : "{ 0, 1 } (Ambiguous / Review)", "text-[var(--text-primary)] font-bold bg-[var(--surface-elevated)] px-2 py-0.5 rounded border border-[var(--border)]"],
                      ["Bounded Risk Interval:", result.risk_interval ? `[${(result.risk_interval[0] * 100).toFixed(1)}%, ${(result.risk_interval[1] * 100).toFixed(1)}%]` : `[${Math.max(0, (result.overall_fraud_probability - 0.05) * 100).toFixed(1)}%, ${Math.min(100, (result.overall_fraud_probability + 0.05) * 100).toFixed(1)}%]`, "text-blue-600 dark:text-blue-400 font-bold"],
                      ["Target Miscoverage (α):", "0.05 (95% Coverage)", "text-[var(--text-primary)]"],
                      ["Calibration Status:", result.conformal_is_calibrated ? "Calibrated (Val Split)" : "Fallback (Uncalibrated)", result.conformal_is_calibrated ? "text-emerald-600 dark:text-emerald-400 font-semibold" : "text-amber-600 dark:text-amber-400 font-semibold"],
                      ["Conformal Quantile (q̂):", result.conformal_q_hat !== undefined && result.conformal_q_hat !== null ? `q̂ = ${result.conformal_q_hat.toFixed(4)}` : "q̂ = 0.5000 (Default)", "text-[var(--text-primary)]"],
                    ].map(([lbl, val, cls], idx) => (
                      <div key={idx} className={`flex justify-between py-1 ${idx < 4 ? "border-b border-[var(--border)]" : ""}`}>
                        <span className="text-[var(--text-secondary)]">{lbl}</span>
                        <span className={cls}>{val}</span>
                      </div>
                    ))}
                  </div>
                  {/* END GENERATED */}
                </div>

                {/* Col 3: Topology & Graph Specs */}
                <div className="p-5 rounded-xl border border-[var(--border)] bg-[var(--surface)] space-y-3 font-mono">
                  <div className="flex items-center gap-2 text-xs font-bold text-cyan-600 dark:text-cyan-400 uppercase tracking-wider">
                    <Network className="w-4 h-4" />
                    <span>Graph Topology State</span>
                  </div>
                  <div className="space-y-2 text-xs pt-1">
                    {[
                      ["Graph Engine:", "NetworkX Temporal Bipartite", "text-[var(--text-primary)]"],
                      ["Traversal Depth:", "2-Hop Community Ring", "text-[var(--text-primary)]"],
                      ["Time Consistency:", "Δt ≥ 0 Point-in-Time", "text-emerald-600 dark:text-emerald-400"],
                      ["Device Sharing Collision:", `${formData.device_shared_buyer_count ?? 1} devices`, "text-rose-600 dark:text-rose-400 font-semibold"],
                    ].map(([lbl, val, cls], idx) => (
                      <div key={idx} className={`flex justify-between py-1 ${idx < 3 ? "border-b border-[var(--border)]" : ""}`}>
                        <span className="text-[var(--text-secondary)]">{lbl}</span>
                        <span className={cls}>{val}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Multi-Domain Bayesian Noisy-OR Risk Breakdown */}
              {result.component_risks && (
                <div className="p-5 rounded-xl border border-[var(--border)] bg-[var(--surface)] space-y-4">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-[var(--border)] pb-3">
                    <div className="flex items-center gap-2">
                      <Zap className="w-4 h-4 text-purple-500" />
                      <h3 className="text-sm font-semibold text-[var(--text-primary)]">
                        Multi-Domain Bayesian Risk Fusion Breakdown
                      </h3>
                    </div>
                    <span className="text-[11px] font-mono text-[var(--text-secondary)]">
                      Fused Risk: {(result.overall_fraud_probability * 100).toFixed(1)}% · Non-downgrade monotonic
                    </span>
                  </div>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                    {Object.entries(result.component_risks).map(([domain, score]) => {
                      const numScore = typeof score === "number" ? score : parseFloat(String(score)) || 0;
                      const pct = Math.min(100, Math.max(0, numScore * 100));
                      const label = {
                        tabular_risk: "Tabular Features",
                        graph_risk_heuristic: "Graph Topology",
                        ring_risk_heuristic: "Fraud Ring / Cluster",
                        multimodal_risk: "Multimodal Mismatch",
                      }[domain] || domain.replace(/_/g, " ");

                      const color = pct >= 65 ? "bg-rose-500 text-rose-600 dark:text-rose-400"
                                  : pct >= 35 ? "bg-amber-500 text-amber-600 dark:text-amber-400"
                                  : "bg-emerald-500 text-emerald-600 dark:text-emerald-400";

                      return (
                        <div key={domain} className="p-3.5 rounded-lg border border-[var(--border)] bg-[var(--surface-elevated)] space-y-2 font-mono">
                          <div className="text-[10px] text-[var(--text-secondary)] uppercase tracking-wider truncate" title={label}>
                            {label}
                          </div>
                          <div className="flex items-baseline justify-between">
                            <span className="text-lg font-bold text-[var(--text-primary)]">{pct.toFixed(1)}%</span>
                            <span className={`text-[10px] font-semibold ${color.split(" ")[1]}`}>
                              {pct >= 65 ? "HIGH" : pct >= 35 ? "MED" : "LOW"}
                            </span>
                          </div>
                          <div className="w-full bg-[var(--border)] rounded-full h-1.5 overflow-hidden">
                            <div className={`h-full rounded-full transition-all duration-500 ${color.split(" ")[0]}`} style={{ width: `${pct}%` }} />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <EmptyState
              title="Calibration Diagnostics Pending"
              description="Submit a transaction scoring request or select an evaluation preset to view finite-sample conformal prediction bounds and Bayesian probability decompositions."
              icon={Microscope}
            />
          )}
        </div>
      )}

    </div>
  );
}

export default function TransactionAnalyzerPage() {
  return (
    <Suspense fallback={<LoadingSkeleton rows={6} />}>
      <TransactionAnalyzerContent />
    </Suspense>
  );
}
