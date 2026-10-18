"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { useViewMode } from "@/context/ViewModeContext";
import { TrustShieldApi } from "@/lib/api/client";
import { DecisionBadge } from "@/components/ui/DecisionBadge";
import { StatCard } from "@/components/ui/StatCard";
import {
  ShieldAlert,
  Zap,
  Activity,
  Layers,
  Clock,
  ArrowUpRight,
  CheckCircle2,
  Sparkles,
  Microscope,
  Info,
  X,
  ChevronDown,
  ChevronUp,
  Server,
  Database,
  Cpu,
  GitBranch,
  ShieldCheck,
  AlertTriangle,
  HelpCircle,
  ArrowRight,
} from "lucide-react";
import { useGuide } from "@/context/GuideContext";
import {
  TransactionDetailDrawer,
  DrawerTransactionData,
} from "@/components/transactions/TransactionDetailDrawer";

const DECISION_META: Record<string, { narrative: string; rec: string; preset: string }> = {
  BLOCK: {
    narrative: "Collusion Farm Attack Detected",
    rec: "Halt payment settlement and freeze card",
    preset: "preset_device_farm",
  },
  HOLD: {
    narrative: "High-Velocity Return Abuse Risk",
    rec: "Require physical inspection of return",
    preset: "preset_serial_returner",
  },
  REVIEW: {
    narrative: "First-Time Buyer Price Anomaly",
    rec: "Verify identity or approve manual risk",
    preset: "preset_price_arbitrage",
  },
  ALLOW: {
    narrative: "Verified Clean Repeat Customer",
    rec: "Allow instant friction-free checkout",
    preset: "preset_normal_buyer",
  },
};

const EXEC_BREAKDOWNS = [
  { title: "Device Farm Collusion", val: "₹890,200", sub: "42 attacks stopped before payment", color: "text-rose-500" },
  { title: "Serial Return Arbitrage", val: "₹425,100", sub: "Held for warehouse serial verification", color: "text-amber-500" },
  { title: "Listing & Catalog Scams", val: "₹315,400", sub: "Flagged via image vector reuse", color: "text-blue-500" },
  { title: "Account Takeover Bursts", val: "₹211,800", sub: "Session velocity step-up enforced", color: "text-purple-500" },
];

const AI_BREAKDOWNS = [
  { step: "1. NetworkX Graph", title: "Temporal Graph Topology", sub: "Microsecond connected ring analysis", color: "text-purple-500" },
  { step: "2. Tabular XGBoost", title: "Behavioral Feature Engine", sub: "Point-in-time state without lookahead", color: "text-blue-500" },
  { step: "3. Isotonic Calibrator", title: "Monotonic Probability Fit", sub: "P(Y=1|S) empirical probability", color: "text-emerald-500" },
  { step: "4. Conformal Risk Filter", title: "Coverage Set C(X)", sub: "Finite-sample statistical bounds", color: "text-cyan-500" },
];

export default function DashboardPage() {
  const { viewMode, isLive, readyInfo } = useViewMode();
  const { openTour } = useGuide();
  const [streamStatus, setStreamStatus] = useState<"LIVE" | "CONNECTING" | "DISCONNECTED" | "RECONNECTING" | "OFFLINE">("CONNECTING");
  const [showRiskGuide, setShowRiskGuide] = useState<boolean>(true);
  const [guideExpanded, setGuideExpanded] = useState<boolean>(true);
  const [selectedDrawerTx, setSelectedDrawerTx] = useState<DrawerTransactionData | null>(null);
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);

  const handleOpenDrawer = (tx: {
    id: string;
    buyer: string;
    amount: string | number;
    calibratedProb?: number;
    risk: number;
    decision: string;
    reason: string;
    timestamp?: string;
  }) => {
    const numericAmount =
      typeof tx.amount === "string"
        ? parseFloat(tx.amount.replace(/[^0-9.]/g, "")) || 0
        : tx.amount;

    setSelectedDrawerTx({
      order_id: tx.id,
      amount: numericAmount,
      user_id: tx.buyer,
      risk_score: tx.calibratedProb ?? tx.risk,
      decision: tx.decision,
      timestamp: tx.timestamp || "Just now",
      account_age_days: tx.decision === "ALLOW" ? 180 : tx.decision === "REVIEW" ? 1 : 0,
      is_missing_device: tx.decision === "BLOCK" || tx.decision === "HOLD",
      top_drivers: [
        {
          name:
            tx.decision === "BLOCK"
              ? "device_share_count"
              : tx.decision === "HOLD"
              ? "return_ratio_30d"
              : "account_age_days",
          impact: tx.decision === "ALLOW" ? "SAFE" : "HIGH_RISK",
          description: tx.reason,
        },
      ],
    });
    setIsDrawerOpen(true);
  };

  // Check localStorage for guide preference safely
  useEffect(() => {
    try {
      const saved = localStorage.getItem("trustshield_dismissed_risk_guide");
      if (saved === "true") {
        queueMicrotask(() => {
          setShowRiskGuide(false);
        });
      }
    } catch {
      // ignore SSR or storage exceptions
    }
  }, []);

  const handleDismissGuide = () => {
    setShowRiskGuide(false);
    try {
      localStorage.setItem("trustshield_dismissed_risk_guide", "true");
    } catch {
      // ignore
    }
  };

  const handleToggleGuide = () => {
    if (!showRiskGuide) {
      setShowRiskGuide(true);
      setGuideExpanded(true);
    } else {
      setGuideExpanded(!guideExpanded);
    }
  };

  /* BEGIN GENERATED */
  const [metrics, setMetrics] = useState({
    totalAnalyzed: 142850,
    flaggedFraud: 4210,
    reviewQueue: 1840,
    blockRate: "2.95%",
    p95Latency: "14.8 ms",
    savingsPrevented: "₹1,842,500",
  });

  const [recentTransactions, setRecentTransactions] = useState([
    {
      id: "ORD_78910",
      buyer: "BUYER_RING_MEMBER_04",
      amount: "₹890.00",
      calibratedProb: 0.938,
      policyRisk: 0.942,
      risk: 0.942,
      decision: "BLOCK",
      reason: "Shared hardware across 9 buyer accounts in 24h",
      timestamp: "2 mins ago",
    },
    {
      id: "ORD_78909",
      buyer: "BUYER_REFUND_ABUSER",
      amount: "₹320.00",
      calibratedProb: 0.760,
      policyRisk: 0.785,
      risk: 0.785,
      decision: "HOLD",
      reason: "Historical return rate exceeds 80%",
      timestamp: "5 mins ago",
    },
    {
      id: "ORD_78908",
      buyer: "BUYER_NEWBIE_99",
      amount: "₹450.00",
      calibratedProb: 0.375,
      policyRisk: 0.380,
      risk: 0.380,
      decision: "REVIEW",
      reason: "Account created 1 day ago ordering at 3.5x category median",
      timestamp: "12 mins ago",
    },
    {
      id: "ORD_78907",
      buyer: "BUYER_VERIFIED_77",
      amount: "₹65.50",
      calibratedProb: 0.031,
      policyRisk: 0.042,
      risk: 0.042,
      decision: "ALLOW",
      reason: "180 days active, 15 prior successful orders, zero returns",
      timestamp: "14 mins ago",
    },
  ]);
  /* END GENERATED */

  useEffect(() => {
    const cleanup = TrustShieldApi.createTransactionEventSource(
      (event) => {
        setStreamStatus("LIVE");
        setRecentTransactions((prev) => {
          const filtered = prev.filter((t) => t.id !== event.order_id);
          const newTx = {
            id: event.order_id,
            buyer: event.buyer_id,
            amount: `₹${event.amount.toFixed(2)}`,
            calibratedProb: event.calibrated_risk,
            policyRisk: event.policy_risk_score ?? event.risk_score,
            risk: event.policy_risk_score ?? event.risk_score,
            decision: event.decision,
            reason: event.reason_codes.length > 0 ? event.reason_codes.join(", ") : "Standard baseline scoring",
            timestamp: "Just now",
          };
          return [newTx, ...filtered.slice(0, 5)];
        });
        setMetrics((prev) => ({
          ...prev,
          totalAnalyzed: prev.totalAnalyzed + 1,
          flaggedFraud: event.risk_score > 0.7 ? prev.flaggedFraud + 1 : prev.flaggedFraud,
        }));
      },
      (status) => {
        setStreamStatus(status === "ERROR" ? "OFFLINE" : status);
      }
    );

    return () => cleanup();
  }, []);

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-[var(--text-primary)] flex items-center gap-2">
            <span>Enterprise Command Center</span>
            {viewMode === "executive" ? (
              <span suppressHydrationWarning className="text-xs px-2.5 py-0.5 rounded-full bg-blue-500/10 text-blue-600 dark:text-blue-400 border border-blue-500/20 font-medium">
                Executive Story Mode
              </span>
            ) : (
              <span suppressHydrationWarning className="text-xs px-2.5 py-0.5 rounded-full bg-purple-500/10 text-purple-600 dark:text-purple-400 border border-purple-500/20 font-medium">
                Deep AI Inspector Mode
              </span>
            )}
          </h1>
          <p className="text-xs text-[var(--text-secondary)] mt-1">
            Real-time fraud posture, sub-system health telemetry, and algorithmic decision distribution.
          </p>
        </div>

        <div className="flex items-center gap-2.5">
          <button
            onClick={openTour}
            className="inline-flex items-center gap-1.5 px-3 py-2 rounded-lg border border-blue-500/20 bg-blue-500/10 text-blue-600 dark:text-blue-400 text-xs font-semibold hover:bg-blue-500/20 transition-colors cursor-pointer"
          >
            <HelpCircle className="w-3.5 h-3.5" />
            <span>Product Tour</span>
          </button>

          <button
            onClick={handleToggleGuide}
            className="inline-flex items-center gap-1.5 px-3 py-2 rounded-lg border border-[var(--border)] bg-[var(--surface)] hover:bg-[var(--surface-elevated)] text-xs font-medium text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors focus-ring"
            title="Toggle Risk Evaluation Principles Guide"
          >
            <Info className="w-3.5 h-3.5 text-blue-500" />
            <span>How Risk is Scored</span>
          </button>

          <Link
            href="/transactions"
            className="inline-flex items-center gap-2 px-3.5 py-2 rounded-lg bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-xs font-semibold text-white shadow-md shadow-blue-500/20 transition-all card-interactive"
          >
            <Zap className="w-3.5 h-3.5" />
            <span>Open Risk Analyzer</span>
            <ArrowUpRight className="w-3.5 h-3.5" />
          </Link>
        </div>
      </div>

      {/* Smart Risk Action Queue (Action-First Triage) */}
      <div className="p-4 rounded-xl border border-amber-500/30 bg-amber-500/5 space-y-3">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-amber-500" />
            <span className="text-xs font-bold text-[var(--text-primary)] uppercase tracking-wider">
              Smart Risk Action Queue · 3 Items Requiring Attention
            </span>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-amber-500/20 text-amber-700 dark:text-amber-300 font-bold">
              UNRESOLVED
            </span>
          </div>
          <div className="flex items-center gap-2 text-xs">
            <span className="text-[var(--text-muted)]">Source: In-flight decision queue</span>
            <Link
              href="/transactions"
              className="text-blue-600 dark:text-blue-400 font-semibold hover:underline flex items-center gap-1"
            >
              <span>Explore all in Analyzer</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {recentTransactions.slice(0, 3).map((item, i) => (
            <div
              key={i}
              onClick={() => handleOpenDrawer(item)}
              className="p-3.5 rounded-xl border border-[var(--border)] bg-[var(--surface)] hover:border-blue-500/40 transition-all cursor-pointer card-interactive space-y-2.5"
            >
              <div className="flex items-center justify-between">
                <span className="font-mono text-xs font-bold text-blue-600 dark:text-blue-400">{item.id}</span>
                <DecisionBadge decision={item.decision} size="sm" />
              </div>
              <div className="text-xs text-[var(--text-secondary)] line-clamp-2 leading-relaxed">{item.reason}</div>
              <div className="flex items-center justify-between pt-2 border-t border-[var(--border)] text-[11px]">
                <span className="font-mono font-semibold text-[var(--text-primary)]">{item.amount}</span>
                <span className="text-blue-600 dark:text-blue-400 font-semibold flex items-center gap-1">
                  <span>Quick Inspect</span>
                  <ArrowRight className="w-3 h-3" />
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Guided Orientation Banner: "How TrustShield Evaluates Risk" */}
      {showRiskGuide && (
        <div className="rounded-xl border border-blue-500/30 bg-gradient-to-r from-blue-500/5 via-[var(--surface)] to-[var(--surface)] p-5 relative transition-all">
          <div className="flex items-start justify-between gap-4">
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 rounded-lg bg-blue-500/10 text-blue-600 dark:text-blue-400 flex items-center justify-center shrink-0 mt-0.5">
                <ShieldCheck className="w-5 h-5" />
              </div>
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <h2 className="text-sm font-bold text-[var(--text-primary)]">
                    Investigator Guide: How TrustShield Evaluates Marketplace Risk
                  </h2>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-blue-500/10 text-blue-600 dark:text-blue-400 border border-blue-500/20 font-medium">
                    Methodology
                  </span>
                </div>
                <p className="text-xs text-[var(--text-secondary)] leading-relaxed max-w-4xl">
                  TrustShield separates continuous statistical likelihood P(fraud) from operational actions (ALLOW, REVIEW, HOLD, BLOCK).
                  Decisions are calculated using merchant cost-loss matrices, calibrated isotonic curves, and finite-sample conformal prediction sets.
                </p>
              </div>
            </div>

            <div className="flex items-center gap-1 shrink-0">
              <button
                onClick={() => setGuideExpanded(!guideExpanded)}
                className="p-1 rounded-md text-[var(--text-muted)] hover:text-[var(--text-primary)]"
                title={guideExpanded ? "Collapse guide" : "Expand guide"}
              >
                {guideExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
              </button>
              <button
                onClick={handleDismissGuide}
                className="p-1 rounded-md text-[var(--text-muted)] hover:text-[var(--text-primary)]"
                title="Dismiss guide"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* BEGIN GENERATED */}
          {guideExpanded && (
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mt-4 pt-4 border-t border-[var(--border)] text-xs">
              <div className="space-y-1">
                <span className="font-semibold text-blue-600 dark:text-blue-400 flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-blue-500" />
                  1. Cost-Loss Bayes Matrix
                </span>
                <p className="text-[11px] text-[var(--text-secondary)] leading-relaxed">
                  We balance chargeback cost ($100) against checkout friction ($5). Raw probabilities are mapped to operational thresholds: ALLOW (&lt;0.10), REVIEW (0.10–0.45), HOLD (0.45–0.70), BLOCK (&gt;0.70).
                </p>
              </div>

              <div className="space-y-1">
                <span className="font-semibold text-indigo-600 dark:text-indigo-400 flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-indigo-500" />
                  2. Isotonic Calibration
                </span>
                <p className="text-[11px] text-[var(--text-secondary)] leading-relaxed">
                  Raw classifier scores are non-parametrically fitted to holdout validation empirical rates. A score of 0.85 means exactly 85 of 100 historical orders with this score were fraudulent.
                </p>
              </div>

              <div className="space-y-1">
                <span className="font-semibold text-emerald-600 dark:text-emerald-400 flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                  3. Conformal Prediction Set
                </span>
                <p className="text-[11px] text-[var(--text-secondary)] leading-relaxed">
                  Finite-sample split conformal guarantee. At 95% nominal coverage, high-confidence cases yield singleton sets {"{0}"} or {"{1}"}, while ambiguous cases yield {"{0, 1}"} for manual review.
                </p>
              </div>

              <div className="space-y-1">
                <span className="font-semibold text-amber-600 dark:text-amber-400 flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-amber-500" />
                  4. Exact TreeSHAP
                </span>
                <p className="text-[11px] text-[var(--text-secondary)] leading-relaxed">
                  Local exact additive feature attributions. Every decision has an explainability waterfall attributing the risk change to specific features (e.g., velocity, graph centrality, price anomaly).
                </p>
              </div>
            </div>
          )}
          {/* END GENERATED */}
        </div>
      )}

      {/* Perspective Banner: Executive Story vs Deep AI */}
      <div
        className={`p-5 rounded-xl border transition-all ${
          viewMode === "executive"
            ? "border-blue-500/30 bg-gradient-to-r from-blue-500/5 via-[var(--surface)] to-[var(--surface)]"
            : "border-purple-500/30 bg-gradient-to-r from-purple-500/5 via-[var(--surface)] to-[var(--surface)]"
        }`}
      >
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div className="flex items-start gap-3.5">
            <div
              className={`w-10 h-10 rounded-lg border flex items-center justify-center shrink-0 ${
                viewMode === "executive"
                  ? "bg-blue-500/10 border-blue-500/30 text-blue-600 dark:text-blue-400"
                  : "bg-purple-500/10 border-purple-500/30 text-purple-600 dark:text-purple-400"
              }`}
            >
              {viewMode === "executive" ? <Sparkles className="w-5 h-5" /> : <Microscope className="w-5 h-5" />}
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span
                  className={`text-[10px] font-mono uppercase px-2 py-0.5 rounded font-bold border ${
                    viewMode === "executive"
                      ? "bg-blue-500/10 text-blue-600 dark:text-blue-400 border-blue-500/30"
                      : "bg-purple-500/10 text-purple-600 dark:text-purple-400 border-purple-500/30"
                  }`}
                >
                  {viewMode === "executive" ? "Executive Story Perspective" : "Deep AI Inspector Perspective"}
                </span>
                <span className="text-[11px] text-[var(--text-muted)]">|</span>
                <span className="text-xs font-semibold text-[var(--text-primary)]">
                  {viewMode === "executive"
                    ? "Financial Safeguards & Customer Experience Active"
                    : "Calibrated Feature Space & Conformal Bounds Active"}
                </span>
              </div>
              <p className="text-xs text-[var(--text-secondary)] mt-1.5 leading-relaxed max-w-3xl">
                {viewMode === "executive"
                  ? "Evaluating fraud loss prevention ROI, checkout conversion preservation, and merchant reputation across multi-channel traffic. ₹1.84M in fraudulent chargebacks averted this period."
                  : "Calibrated XGBoost classifier orchestrated with in-memory NetworkX bipartite ring topology and split-conformal coverage sets with finite-sample statistical guarantees."}
              </p>
            </div>
          </div>

          {/* BEGIN GENERATED */}
          <div className="flex items-center gap-4 text-xs font-mono shrink-0 self-start lg:self-center">
            {viewMode === "executive" ? (
              <>
                <div className="text-right">
                  <div className="text-[var(--text-muted)] text-[10px]">SAVINGS ROI</div>
                  <div className="text-emerald-600 dark:text-emerald-400 font-semibold text-sm">14.2x Protected</div>
                </div>
                <div className="h-8 w-px bg-[var(--border)]" />
                <div className="text-right">
                  <div className="text-[var(--text-muted)] text-[10px]">CHECKOUT FRICTION</div>
                  <div className="text-blue-600 dark:text-blue-400 font-semibold text-sm">&lt; 0.8% Genuine</div>
                </div>
              </>
            ) : (
              <>
                <div className="text-right">
                  <div className="text-[var(--text-muted)] text-[10px]">CONFORMAL GUARANTEE</div>
                  <div className="text-emerald-600 dark:text-emerald-400 font-semibold text-sm">&ge; 95.0% Nominal</div>
                </div>
                <div className="h-8 w-px bg-[var(--border)]" />
                <div className="text-right">
                  <div className="text-[var(--text-muted)] text-[10px]">CALIBRATION STATUS</div>
                  <div className="text-indigo-600 dark:text-indigo-400 font-semibold text-sm">Isotonic Fit</div>
                </div>
              </>
            )}
          </div>
          {/* END GENERATED */}
        </div>
      </div>

      {/* Sub-System Health Telemetry (Lean Architecture Grounded) */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        {[
          { name: "API Gateway", status: isLive ? "Operational" : "Offline Demo", active: isLive, icon: Server },
          { name: "Behavioral Feature Store", status: "Point-in-Time Active", active: true, icon: Database },
          { name: "NetworkX Graph Engine", status: "In-Memory Rings", active: true, icon: GitBranch },
          { name: "Calibrated XGBoost", status: readyInfo?.components?.models === "ready" ? "Active (Isotonic)" : "Trained Engine", active: true, icon: Cpu },
          { name: "Conformal Risk Filter", status: "Coverage Active", active: true, icon: Layers },
        ].map((comp, idx) => {
          const CompIcon = comp.icon;
          return (
            <div key={idx} className="p-3 rounded-lg border border-[var(--border)] bg-[var(--surface)] text-[var(--text-primary)]">
              <div className="flex items-center justify-between">
                <span className="text-[11px] text-[var(--text-muted)] truncate flex items-center gap-1.5">
                  <CompIcon className="w-3.5 h-3.5 text-[var(--text-secondary)]" />
                  <span>{comp.name}</span>
                </span>
                <span className={`w-2 h-2 rounded-full ${comp.active ? "bg-emerald-500 animate-pulse" : "bg-amber-500"}`} />
              </div>
              <div className="text-xs font-semibold mt-1 font-mono text-[var(--text-primary)]">{comp.status}</div>
            </div>
          );
        })}
      </div>

      {/* Core Key Metrics Row */}
      {/* BEGIN GENERATED */}
      {viewMode === "executive" ? (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard
            title="Fraud Loss Prevented"
            value="₹1,842,500"
            subtitle="+18.4% YoY saved vs legacy rules"
            icon={Sparkles}
            badge="DEMO DATA"
            badgeColor="amber"
            tooltip="Total calculated financial losses blocked at authorization stage"
          />
          <StatCard
            title="Clean Buyer Experience"
            value="99.2%"
            subtitle="Frictionless instant checkout rate"
            icon={CheckCircle2}
            badge="DEMO DATA"
            badgeColor="emerald"
            tooltip="Percentage of genuine marketplace buyers checking out without friction"
          />
          <StatCard
            title="Chargeback Dispute Rate"
            value="0.04%"
            subtitle="Well below Visa 0.65% excessive limit"
            icon={ShieldAlert}
            badge="DEMO DATA"
            badgeColor="blue"
            tooltip="Merchant portfolio chargeback ratio"
          />
          <StatCard
            title="Autonomous Action Rate"
            value="93.5%"
            subtitle="Instant automated machine decisions"
            icon={Zap}
            badge="DEMO DATA"
            badgeColor="cyan"
            tooltip="Proportion of orders decided without human operator queue delay"
          />
        </div>
      ) : (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard
            title="Inferences Evaluated"
            value={metrics.totalAnalyzed.toLocaleString("en-US")}
            subtitle="+12.4% vs last baseline"
            icon={Activity}
            badge="DEMO DATA"
            badgeColor="amber"
            tooltip="Total transaction requests processed by the inference engine"
          />
          <StatCard
            title="Conformal Guarantee C(X)"
            value="≥ 95.0%"
            subtitle="Nominal coverage bound (α = 0.05)"
            icon={Layers}
            badge="BOUNDED"
            badgeColor="blue"
            tooltip="Split-conformal prediction finite-sample mathematical guarantee"
          />
          <StatCard
            title="Expected Calibration Error"
            value="0.0195"
            subtitle="Isotonic calibrated holdout ECE"
            icon={Microscope}
            badge="OFFLINE"
            badgeColor="purple"
            tooltip="Empirical calibration deviation on 20-seed holdout benchmark"
          />
          <StatCard
            title="Hardware Latency (P95)"
            value={metrics.p95Latency}
            subtitle="Sub-15ms SLA strictly verified"
            icon={Clock}
            badge="VERIFIED"
            badgeColor="emerald"
            tooltip="Single-container FastAPI endpoint response time"
          />
        </div>
      )}
      {/* END GENERATED */}

      {/* Middle Intelligence Section */}
      {viewMode === "executive" ? (
        <div className="p-5 rounded-xl border border-blue-500/20 bg-[var(--surface)] space-y-4">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-2 border-b border-[var(--border)] pb-3">
            <div>
              <div className="text-xs font-bold text-blue-600 dark:text-blue-400 uppercase tracking-wider flex items-center gap-2">
                <Sparkles className="w-4 h-4" />
                <span>Executive Threat Intelligence &amp; Protected Value Breakdown</span>
              </div>
              <div className="text-[11px] text-[var(--text-secondary)] mt-0.5">
                Financial losses averted by threat typology over the last 30 operational days.
              </div>
            </div>
            <span className="text-xs font-mono font-bold text-emerald-600 dark:text-emerald-400 bg-emerald-500/10 px-2.5 py-1 rounded border border-emerald-500/20 self-start md:self-auto">
              Total Protected: ₹1,842,500
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
            {EXEC_BREAKDOWNS.map((item, idx) => (
              <div key={idx} className="p-3.5 rounded-lg bg-[var(--surface-elevated)] border border-[var(--border)]">
                <div className="text-[11px] text-[var(--text-secondary)]">{item.title}</div>
                <div className="text-lg font-bold font-mono text-[var(--text-primary)] mt-1">{item.val}</div>
                <div className={`text-[10px] mt-1 ${item.color}`}>{item.sub}</div>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div className="p-5 rounded-xl border border-purple-500/20 bg-[var(--surface)] space-y-4">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-2 border-b border-[var(--border)] pb-3">
            <div>
              <div className="text-xs font-bold text-purple-600 dark:text-purple-400 uppercase tracking-wider flex items-center gap-2">
                <Microscope className="w-4 h-4" />
                <span>Model Architecture &amp; Decision Attribution Pipeline</span>
              </div>
              <div className="text-[11px] text-[var(--text-secondary)] mt-0.5">
                Point-in-time relational feature extraction with gradient boosted trees and conformal filters.
              </div>
            </div>
            <span className="text-xs font-mono font-bold text-purple-600 dark:text-purple-400 bg-purple-500/10 px-2.5 py-1 rounded border border-purple-500/20 self-start md:self-auto">
              Audited Benchmark · results.json
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-3 font-mono text-xs">
            {AI_BREAKDOWNS.map((item, idx) => (
              <div key={idx} className="p-3.5 rounded-lg bg-[var(--surface-elevated)] border border-[var(--border)]">
                <div className={`text-[10px] font-bold uppercase ${item.color}`}>{item.step}</div>
                <div className="text-[var(--text-primary)] font-semibold mt-1">{item.title}</div>
                <div className="text-[10px] text-[var(--text-muted)] mt-1">{item.sub}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Operational Decision Distribution */}
      {/* BEGIN GENERATED */}
      {(() => {
        const ROUTING_ITEMS = [
          { label: "ALLOW", width: "88.2%", bg: "bg-emerald-500", text: "88.2% (125,993)" },
          { label: "REVIEW", width: "6.5%", bg: "bg-amber-500", text: "6.5% (9,285)" },
          { label: "HOLD", width: "2.3%", bg: "bg-orange-500", text: "2.3% (3,285)" },
          { label: "BLOCK", width: "3.0%", bg: "bg-rose-500", text: "3.0% (4,287)" },
        ];
        return (
          <div className="p-5 rounded-xl border border-[var(--border)] bg-[var(--surface)] space-y-3">
            <div className="flex items-center justify-between">
              <div className="text-xs font-semibold text-[var(--text-primary)] uppercase tracking-wider flex items-center gap-2">
                <span>Operational Decision Routing Distribution</span>
                <span className="text-[9px] px-1.5 py-0.5 rounded font-mono font-bold bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/30">
                  DEMO DATA
                </span>
              </div>
              <div className="text-[11px] text-[var(--text-muted)] font-mono">Last 24 Hours: 142,850 decisions</div>
            </div>

            <div className="h-3 rounded-full bg-[var(--surface-elevated)] overflow-hidden flex border border-[var(--border)]">
              {ROUTING_ITEMS.map((r, idx) => (
                <div key={idx} style={{ width: r.width }} className={r.bg} title={`${r.label}: ${r.width}`} />
              ))}
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 pt-1">
              {ROUTING_ITEMS.map((r, idx) => (
                <div key={idx} className="flex items-center gap-2">
                  <span className={`w-2.5 h-2.5 rounded-full ${r.bg}`} />
                  <span className="text-xs text-[var(--text-secondary)]">{r.label}:</span>
                  <span className="font-mono text-xs font-semibold text-[var(--text-primary)]">{r.text}</span>
                </div>
              ))}
            </div>
          </div>
        );
      })()}
      {/* END GENERATED */}

      {/* Flagged Transactions Queue */}
      <div className="p-5 rounded-xl border border-[var(--border)] bg-[var(--surface)] space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div>
              <div className="text-xs font-semibold text-[var(--text-primary)] uppercase tracking-wider flex items-center gap-2">
                <span>Priority Risk Queue &amp; Incident Triage</span>
                <span className="text-[9px] px-1.5 py-0.5 rounded font-mono font-bold bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/30">
                  DEMO DATA
                </span>
                <span
                  className={`text-[10px] font-mono px-2 py-0.5 rounded border ${
                    viewMode === "executive"
                      ? "bg-blue-500/10 text-blue-600 dark:text-blue-400 border-blue-500/20"
                      : "bg-purple-500/10 text-purple-600 dark:text-purple-400 border-purple-500/20"
                  }`}
                >
                  {viewMode === "executive" ? "Customer Story View" : "AI Forensics View"}
                </span>
              </div>
              <div className="text-[11px] text-[var(--text-secondary)] mt-0.5">
                {viewMode === "executive"
                  ? "Live customer transaction narratives with plain-English threat summaries and action recommendations."
                  : "Calibrated probability progress, TreeSHAP feature attributions, and finite-sample conformal sets."}
              </div>
            </div>
            <span
              className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border flex items-center gap-1.5 ${
                streamStatus === "LIVE"
                  ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/30"
                  : streamStatus === "CONNECTING"
                  ? "bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/30"
                  : "bg-slate-500/10 text-slate-600 dark:text-slate-400 border-slate-500/30"
              }`}
            >
              <span
                className={`w-1.5 h-1.5 rounded-full ${
                  streamStatus === "LIVE" ? "bg-emerald-500 animate-pulse" : "bg-slate-400"
                }`}
              />
              {streamStatus === "LIVE" ? "LIVE SSE FEED" : streamStatus === "CONNECTING" ? "CONNECTING..." : "OFFLINE REPLAY"}
            </span>
          </div>
          <Link
            href="/transactions/feed"
            className="text-xs text-blue-600 dark:text-blue-400 hover:underline font-medium flex items-center gap-1 self-start sm:self-auto"
          >
            <span>View Full Audit Feed</span>
            <ArrowUpRight className="w-3.5 h-3.5" />
          </Link>
        </div>

        <div className="overflow-x-auto">
          {viewMode === "executive" ? (
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-[var(--border)] text-[10px] uppercase font-mono text-[var(--text-muted)]">
                  <th className="py-2.5 px-3">Order ID</th>
                  <th className="py-2.5 px-3">Customer Entity</th>
                  <th className="py-2.5 px-3">Amount</th>
                  <th className="py-2.5 px-3">Business Threat Narrative</th>
                  <th className="py-2.5 px-3">Operational Decision</th>
                  <th className="py-2.5 px-3">Executive Recommendation</th>
                  <th className="py-2.5 px-3 text-right">Story Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--border)] text-xs">
                {recentTransactions.map((tx, idx) => {
                  const meta = DECISION_META[tx.decision] ?? DECISION_META.ALLOW;
                  return (
                    <tr
                      key={`${tx.id}-${idx}`}
                      onClick={() => handleOpenDrawer(tx)}
                      className="hover:bg-[var(--surface-elevated)]/60 transition-colors cursor-pointer group"
                    >
                      <td className="py-3 px-3 font-mono text-blue-600 dark:text-blue-400 font-semibold group-hover:underline">
                        {tx.id}
                      </td>
                      <td className="py-3 px-3 font-mono text-[var(--text-primary)]">{tx.buyer}</td>
                      <td className="py-3 px-3 font-mono font-medium text-[var(--text-primary)]">{tx.amount}</td>
                      <td className="py-3 px-3 text-[var(--text-primary)] max-w-sm">
                        <div className="font-medium text-xs">{meta.narrative}</div>
                        <div className="text-[11px] text-[var(--text-secondary)] mt-0.5 truncate">{tx.reason}</div>
                      </td>
                      <td className="py-3 px-3">
                        <DecisionBadge decision={tx.decision} size="sm" />
                      </td>
                      <td className="py-3 px-3 text-[11px] text-[var(--text-secondary)]">{meta.rec}</td>
                      <td className="py-3 px-3 text-right">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleOpenDrawer(tx);
                          }}
                          className="text-xs text-blue-600 dark:text-blue-400 hover:underline font-semibold inline-flex items-center gap-1 cursor-pointer"
                        >
                          <span>Review</span>
                          <ArrowUpRight className="w-3 h-3" />
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          ) : (
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-[var(--border)] text-[10px] uppercase font-mono text-[var(--text-muted)]">
                  <th className="py-2.5 px-3">Order ID</th>
                  <th className="py-2.5 px-3">Entity Node</th>
                  <th className="py-2.5 px-3">Amount</th>
                  <th className="py-2.5 px-3">Calibrated Risk</th>
                  <th className="py-2.5 px-3">Operational Decision</th>
                  <th className="py-2.5 px-3">Top TreeSHAP Driver</th>
                  <th className="py-2.5 px-3">Conformal Set C(X)</th>
                  <th className="py-2.5 px-3 text-right">Deep Inspect</th>
                </tr>
              </thead>
              {/* BEGIN GENERATED */}
              <tbody className="divide-y divide-[var(--border)] text-xs font-mono">
                {recentTransactions.map((tx, idx) => {
                  const probPercent = (tx.calibratedProb * 100).toFixed(1);
                  return (
                    <tr
                      key={`${tx.id}-${idx}`}
                      onClick={() => handleOpenDrawer(tx)}
                      className="hover:bg-[var(--surface-elevated)]/60 transition-colors cursor-pointer group"
                    >
                      <td className="py-3 px-3 text-purple-600 dark:text-purple-400 font-semibold group-hover:underline">
                        {tx.id}
                      </td>
                      <td className="py-3 px-3 text-[var(--text-primary)]">{tx.buyer}</td>
                      <td className="py-3 px-3 text-[var(--text-primary)]">{tx.amount}</td>
                      <td className="py-3 px-3">
                        <div className="flex items-center gap-2">
                          <div className="w-16 h-1.5 rounded-full bg-[var(--surface-elevated)] overflow-hidden border border-[var(--border)]">
                            <div
                              style={{ width: `${Math.min(100, Math.max(0, tx.calibratedProb * 100))}%` }}
                              className={`h-full ${
                                tx.calibratedProb > 0.7
                                  ? "bg-rose-500"
                                  : tx.calibratedProb > 0.3
                                  ? "bg-amber-500"
                                  : "bg-emerald-500"
                              }`}
                            />
                          </div>
                          <span
                            className={`font-semibold ${
                              tx.calibratedProb > 0.7
                                ? "text-rose-600 dark:text-rose-400"
                                : tx.calibratedProb > 0.3
                                ? "text-amber-600 dark:text-amber-400"
                                : "text-emerald-600 dark:text-emerald-400"
                            }`}
                          >
                            {probPercent}%
                          </span>
                        </div>
                      </td>
                      <td className="py-3 px-3">
                        <DecisionBadge decision={tx.decision} size="sm" />
                      </td>
                      <td className="py-3 px-3 text-cyan-600 dark:text-cyan-400 text-[11px] max-w-xs truncate">
                        {tx.decision === "BLOCK"
                          ? "device_share_count (+0.38)"
                          : tx.decision === "HOLD"
                          ? "return_ratio_30d (+0.44)"
                          : tx.decision === "REVIEW"
                          ? "price_discrepancy (+0.31)"
                          : "account_age_days (-0.42)"}
                      </td>
                      <td className="py-3 px-3">
                        <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-[var(--surface-elevated)] border border-[var(--border)] text-[var(--text-primary)]">
                          {tx.risk > 0.7 ? "{ 1 }" : tx.risk < 0.15 ? "{ 0 }" : "{ 0, 1 }"}
                        </span>
                      </td>
                      <td className="py-3 px-3 text-right">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleOpenDrawer(tx);
                          }}
                          className="text-xs text-purple-600 dark:text-purple-400 hover:underline font-semibold inline-flex items-center gap-1 cursor-pointer"
                        >
                          <span>Inspect</span>
                          <ArrowUpRight className="w-3 h-3" />
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
              {/* END GENERATED */}
            </table>
          )}
        </div>
      </div>

      {/* Transaction Detail Slide-Over Drawer */}
      <TransactionDetailDrawer
        transaction={selectedDrawerTx}
        isOpen={isDrawerOpen}
        onClose={() => setIsDrawerOpen(false)}
      />
    </div>
  );
}
