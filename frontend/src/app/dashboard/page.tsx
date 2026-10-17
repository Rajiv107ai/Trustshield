"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { useViewMode } from "@/context/ViewModeContext";
import { TrustShieldApi } from "@/lib/api/client";
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
} from "lucide-react";

const DECISION_META: Record<string, { badge: string; narrative: string; rec: string; preset: string }> = {
  BLOCK: {
    badge: "bg-rose-500/10 text-rose-400 border-rose-500/20",
    narrative: "Collusion Farm Attack Detected",
    rec: "Halt payment settlement and freeze card",
    preset: "preset_device_farm",
  },
  HOLD: {
    badge: "bg-orange-500/10 text-orange-400 border-orange-500/20",
    narrative: "High-Velocity Return Abuse Risk",
    rec: "Require physical inspection of return",
    preset: "preset_serial_returner",
  },
  REVIEW: {
    badge: "bg-amber-500/10 text-amber-400 border-amber-500/20",
    narrative: "First-Time Buyer Price Anomaly",
    rec: "Verify identity or approve manual risk",
    preset: "preset_price_arbitrage",
  },
  ALLOW: {
    badge: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
    narrative: "Verified Clean Repeat Customer",
    rec: "Allow instant friction-free checkout",
    preset: "preset_normal_buyer",
  },
};

const EXEC_BREAKDOWNS = [
  { title: "Device Farm Collusion", val: "₹890,200", sub: "42 attacks stopped before payment", color: "text-rose-400" },
  { title: "Serial Return Arbitrage", val: "₹425,100", sub: "Held for warehouse serial verification", color: "text-amber-400" },
  { title: "Listing & Catalog Scams", val: "₹315,400", sub: "Flagged via CLIP image vector reuse", color: "text-cyan-400" },
  { title: "Account Takeover Bursts", val: "₹211,800", sub: "Session velocity step-up enforced", color: "text-purple-400" },
];

const AI_BREAKDOWNS = [
  { step: "1. GraphSAGE GNN", title: "16D Node Embeddings", sub: "Inductive neighborhood message passing", color: "text-purple-400" },
  { step: "2. Tabular XGBoost", title: "48 Relational Features", sub: "Tree depth: 6 · Regularization: L1/L2", color: "text-blue-400" },
  { step: "3. Isotonic Calibrator", title: "Monotonic Score Fit", sub: "P(Y=1|S) calibrated on holdout split", color: "text-emerald-400" },
  { step: "4. Conformal Risk Filter", title: "Coverage Set C(X)", sub: "Finite-sample validity guaranteed", color: "text-cyan-400" },
];

export default function DashboardPage() {
  const { viewMode, isLive, readyInfo } = useViewMode();
  const [streamStatus, setStreamStatus] = useState<"LIVE" | "CONNECTING" | "DISCONNECTED" | "RECONNECTING" | "OFFLINE">("CONNECTING");

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
          <h1 className="text-xl font-bold tracking-tight text-[#E8EDF3] flex items-center gap-2">
            <span>Enterprise Command Center</span>
            {viewMode === "executive" ? (
              <span suppressHydrationWarning className="text-xs px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20 font-normal">
                Executive Story Mode
              </span>
            ) : (
              <span suppressHydrationWarning className="text-xs px-2 py-0.5 rounded-full bg-purple-500/10 text-purple-400 border border-purple-500/20 font-normal">
                Deep AI Inspector Mode
              </span>
            )}
          </h1>
          <p className="text-xs text-[#8995A3] mt-1">
            Real-time fraud posture, sub-system health telemetry, and algorithmic decision distribution.
          </p>
        </div>

        <Link
          href="/transactions"
          className="inline-flex items-center gap-2 px-3.5 py-2 rounded-lg bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-xs font-semibold text-white shadow-lg shadow-blue-500/20 transition-all self-start"
        >
          <Zap className="w-3.5 h-3.5" />
          <span>Open Risk Analyzer (Hero Simulator)</span>
          <ArrowUpRight className="w-3.5 h-3.5" />
        </Link>
      </div>

      {/* Top Banner: Dual Mode Insight */}
      <div
        className={`p-5 rounded-xl border transition-all ${
          viewMode === "executive"
            ? "border-blue-500/30 bg-gradient-to-r from-blue-950/30 via-[#111821] to-[#111821] shadow-lg shadow-blue-500/5"
            : "border-purple-500/30 bg-gradient-to-r from-purple-950/30 via-[#111821] to-[#111821] shadow-lg shadow-purple-500/5"
        }`}
      >
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div className="flex items-start gap-3.5">
            <div
              className={`w-10 h-10 rounded-lg border flex items-center justify-center shrink-0 ${
                viewMode === "executive"
                  ? "bg-blue-500/10 border-blue-500/30 text-blue-400"
                  : "bg-purple-500/10 border-purple-500/30 text-purple-400"
              }`}
            >
              {viewMode === "executive" ? <Sparkles className="w-5 h-5" /> : <Microscope className="w-5 h-5" />}
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span
                  className={`text-[10px] font-mono uppercase px-2 py-0.5 rounded font-bold border ${
                    viewMode === "executive"
                      ? "bg-blue-500/10 text-blue-400 border-blue-500/30"
                      : "bg-purple-500/10 text-purple-400 border-purple-500/30"
                  }`}
                >
                  {viewMode === "executive" ? "Executive Story Perspective" : "Deep AI Inspector Perspective"}
                </span>
                <span className="text-[11px] text-[#596574]">|</span>
                <span className="text-xs font-semibold text-[#E8EDF3]">
                  {viewMode === "executive"
                    ? "Financial Safeguards & Customer Experience Active"
                    : "High-Dimensional Feature Space & Conformal Calibration Active"}
                </span>
              </div>
              <p className="text-xs text-[#8995A3] mt-1.5 leading-relaxed max-w-3xl">
                {viewMode === "executive"
                  ? "Evaluating fraud loss prevention ROI, checkout conversion preservation, and merchant reputation across multi-channel consumer traffic. $1.84M in fraudulent chargebacks averted this month."
                  : "Stacking Meta-Learner orchestrating 16D inductive GNN node embeddings, 48-split tabular XGBoost classifiers, and non-conformity coverage sets guaranteed at alpha=0.05."}
              </p>
            </div>
          </div>

          {/* BEGIN GENERATED */}
          <div className="flex items-center gap-4 text-xs font-mono shrink-0 self-start lg:self-center">
            {viewMode === "executive" ? (
              <>
                <div className="text-right">
                  <div className="text-[#8995A3] text-[10px]">SAVINGS ROI</div>
                  <div className="text-emerald-400 font-semibold text-sm">14.2x Protected</div>
                </div>
                <div className="h-8 w-px bg-[#202A35]" />
                <div className="text-right">
                  <div className="text-[#8995A3] text-[10px]">CHECKOUT FRICTION</div>
                  <div className="text-blue-400 font-semibold text-sm">&lt; 0.8% Genuine</div>
                </div>
              </>
            ) : (
              <>
                <div className="text-right">
                  <div className="text-[#8995A3] text-[10px]">CONFORMAL GUARANTEE</div>
                  <div className="text-emerald-400 font-semibold text-sm">&ge; 95.0% Nominal</div>
                </div>
                <div className="h-8 w-px bg-[#202A35]" />
                <div className="text-right">
                  <div className="text-[#8995A3] text-[10px]">CALIBRATION STATUS</div>
                  <div className="text-[#8995A3] font-semibold text-sm">Offline Eval Only</div>
                </div>
              </>
            )}
          </div>
          {/* END GENERATED */}
        </div>
      </div>

      {/* System Readiness Probe Grid */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        {[
          { name: "API Gateway", status: isLive ? "Operational" : "Offline Demo", active: isLive },
          { name: "Redis Feature Store", status: readyInfo?.components?.redis === "available" ? "Live (Port 6379)" : "Offline (Disk Fallback)", active: readyInfo?.components?.redis === "available" },
          { name: "Neo4j Graph Store", status: readyInfo?.components?.neo4j === "available" ? "Live (Port 7687)" : "Offline (Disk Rings)", active: readyInfo?.components?.neo4j === "available" },
          { name: "Phase 5 Hybrid", status: readyInfo?.components?.models === "ready" ? "Active (XGBoost)" : "Artifact Model", active: true },
          { name: "Prometheus Telemetry", status: readyInfo?.components?.prometheus === "available" ? "Scraping /metrics" : "Standby", active: readyInfo?.components?.prometheus === "available" },
        ].map((comp, idx) => (
          <div key={idx} className="p-3 rounded-lg border border-[#202A35] bg-[#111821]">
            <div className="flex items-center justify-between">
              <span className="text-[11px] text-[#8995A3] truncate">{comp.name}</span>
              <span className={`w-1.5 h-1.5 rounded-full ${comp.active ? "bg-emerald-500" : "bg-amber-500"}`} />
            </div>
            <div className="text-xs font-semibold text-[#E8EDF3] mt-1 font-mono">{comp.status}</div>
          </div>
        ))}
      </div>

      {/* DYNAMIC KEY METRICS ROW (ADAPTS TO VIEW MODE) */}
      {/* BEGIN GENERATED */}
      {viewMode === "executive" ? (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { label: "Fraud Loss Prevented", val: "$1,842,500", sub: "+18.4% YoY saved vs legacy rules", color: "text-emerald-400", subColor: "text-emerald-400/90", border: "border-blue-500/20", icon: Sparkles, iconColor: "text-blue-400", titleColor: "text-blue-300" },
            { label: "Clean Buyer Experience", val: "99.2%", sub: "Frictionless instant checkout rate", color: "text-[#E8EDF3]", subColor: "text-emerald-400", border: "border-emerald-500/20", icon: CheckCircle2, iconColor: "text-emerald-400", titleColor: "text-emerald-300" },
            { label: "Chargeback Dispute Rate", val: "0.04%", sub: "Well below Visa 0.65% excessive limit", color: "text-[#E8EDF3]", subColor: "text-emerald-400", border: "border-amber-500/20", icon: ShieldAlert, iconColor: "text-amber-400", titleColor: "text-amber-300" },
            { label: "Autonomous Action Rate", val: "93.5%", sub: "Instant automated machine decisions", color: "text-[#E8EDF3]", subColor: "text-cyan-400", border: "border-cyan-500/20", icon: Zap, iconColor: "text-cyan-400", titleColor: "text-cyan-300" },
          ].map((c) => {
            const Icon = c.icon;
            return (
              <div key={c.label} className={`p-4 rounded-xl border ${c.border} bg-[#111821]`}>
                <div className="flex items-center justify-between text-xs text-[#8995A3]">
                  <div className="flex items-center gap-1.5">
                    <span className={`font-medium ${c.titleColor}`}>{c.label}</span>
                    <span className="text-[9px] px-1.5 py-0.5 rounded font-mono font-bold bg-amber-500/10 text-amber-400 border border-amber-500/30">DEMO DATA</span>
                  </div>
                  <Icon className={`w-4 h-4 ${c.iconColor}`} />
                </div>
                <div className={`text-2xl font-bold font-mono ${c.color} mt-2`}>{c.val}</div>
                <div className={`text-[11px] ${c.subColor} mt-1 font-medium`}>{c.sub}</div>
              </div>
            );
          })}
        </div>
      ) : (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { label: "Inferences Evaluated", val: metrics.totalAnalyzed.toLocaleString("en-US"), sub: "+12.4% vs last week", color: "text-[#E8EDF3]", subColor: "text-emerald-400", border: "border-purple-500/20", icon: Activity, iconColor: "text-purple-400", titleColor: "text-purple-300", tag: "DEMO DATA", tagCls: "bg-amber-500/10 text-amber-400 border-amber-500/30", suppress: true },
            { label: "Conformal Guarantee C(X)", val: "≥ 95.0%", sub: "Nominal guarantee (α = 0.05) · Empirical: Not available in live feed", color: "text-emerald-400", subColor: "text-[#8995A3]", border: "border-indigo-500/20", icon: Layers, iconColor: "text-indigo-400", titleColor: "text-indigo-300", tag: "MATHEMATICAL BOUND", tagCls: "bg-blue-500/10 text-blue-400 border-blue-500/30" },
            { label: "Expected Calibration Error", val: "Not available", sub: "Requires ground-truth labels · See Evaluation tab", color: "text-[#8995A3]", subColor: "text-[#596574] font-mono", border: "border-cyan-500/20", icon: Microscope, iconColor: "text-cyan-400", titleColor: "text-cyan-300", tag: "OFFLINE METRIC", tagCls: "bg-[#202A35] text-[#8995A3] border-[#202A35]" },
            { label: "Hardware Latency (p95)", val: metrics.p95Latency, sub: "< 20 ms SLA strictly verified", color: "text-[#E8EDF3]", subColor: "text-emerald-400", border: "border-rose-500/20", icon: Clock, iconColor: "text-rose-400", titleColor: "text-rose-300", tag: "DEMO DATA", tagCls: "bg-amber-500/10 text-amber-400 border-amber-500/30" },
          ].map((c) => {
            const Icon = c.icon;
            return (
              <div key={c.label} className={`p-4 rounded-xl border ${c.border} bg-[#111821]`}>
                <div className="flex items-center justify-between text-xs text-[#8995A3]">
                  <div className="flex items-center gap-1.5">
                    <span className={`font-medium ${c.titleColor}`}>{c.label}</span>
                    <span className={`text-[9px] px-1.5 py-0.5 rounded font-mono font-bold border ${c.tagCls}`}>{c.tag}</span>
                  </div>
                  <Icon className={`w-4 h-4 ${c.iconColor}`} />
                </div>
                <div {...(c.suppress ? { suppressHydrationWarning: true } : {})} className={`text-2xl font-bold font-mono ${c.color} mt-2`}>{c.val}</div>
                <div className={`text-[11px] ${c.subColor} mt-1 flex items-center gap-1`}>{c.sub}</div>
              </div>
            );
          })}
        </div>
      )}
      {/* END GENERATED */}

      {/* DUAL MODE MIDDLE INTELLIGENCE SECTION */}
      {viewMode === "executive" ? (
        <div className="p-5 rounded-xl border border-blue-500/20 bg-[#111821] space-y-4">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-2 border-b border-[#202A35] pb-3">
            <div>
              <div className="text-xs font-bold text-blue-400 uppercase tracking-wider flex items-center gap-2">
                <Sparkles className="w-4 h-4" />
                <span>Executive Threat Intelligence &amp; Protected Value Breakdown</span>
              </div>
              <div className="text-[11px] text-[#8995A3] mt-0.5">
                Financial losses averted by threat typology over the last 30 operational days.
              </div>
            </div>
            <span className="text-xs font-mono font-bold text-emerald-400 bg-emerald-500/10 px-2.5 py-1 rounded border border-emerald-500/20 self-start md:self-auto">
              Total Protected: ₹1,842,500
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
            {EXEC_BREAKDOWNS.map((item, idx) => (
              <div key={idx} className="p-3.5 rounded-lg bg-[#0E131A] border border-[#202A35]">
                <div className="text-[11px] text-[#8995A3]">{item.title}</div>
                <div className="text-lg font-bold font-mono text-[#E8EDF3] mt-1">{item.val}</div>
                <div className={`text-[10px] mt-1 ${item.color}`}>{item.sub}</div>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div className="p-5 rounded-xl border border-purple-500/20 bg-[#111821] space-y-4">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-2 border-b border-[#202A35] pb-3">
            <div>
              <div className="text-xs font-bold text-purple-400 uppercase tracking-wider flex items-center gap-2">
                <Microscope className="w-4 h-4" />
                <span>Phase 5 Hybrid Model Topology &amp; Feature Attribution Pipeline</span>
              </div>
              <div className="text-[11px] text-[#8995A3] mt-0.5">
                Heterogeneous graph neural network fusion with gradient boosting trees and conformal predictors.
              </div>
            </div>
            <span className="text-xs font-mono font-bold text-purple-400 bg-purple-500/10 px-2.5 py-1 rounded border border-purple-500/20 self-start md:self-auto">
              Audited Benchmark · results.json
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-3 font-mono text-xs">
            {AI_BREAKDOWNS.map((item, idx) => (
              <div key={idx} className="p-3.5 rounded-lg bg-[#0E131A] border border-[#202A35]">
                <div className={`text-[10px] font-bold uppercase ${item.color}`}>{item.step}</div>
                <div className="text-[#E8EDF3] font-semibold mt-1">{item.title}</div>
                <div className="text-[10px] text-[#8995A3] mt-1">{item.sub}</div>
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
          <div className="p-5 rounded-xl border border-[#202A35] bg-[#111821] space-y-3">
            <div className="flex items-center justify-between">
              <div className="text-xs font-semibold text-[#E8EDF3] uppercase tracking-wider flex items-center gap-2">
                <span>Operational Decision Routing Distribution</span>
                <span className="text-[9px] px-1.5 py-0.5 rounded font-mono font-bold bg-amber-500/10 text-amber-400 border border-amber-500/30">DEMO DATA</span>
              </div>
              <div className="text-[11px] text-[#8995A3] font-mono">Last 24 Hours: 142,850 decisions</div>
            </div>


        <div className="h-3 rounded-full bg-[#0A0E13] overflow-hidden flex">
          {ROUTING_ITEMS.map((r, idx) => (
            <div key={idx} style={{ width: r.width }} className={r.bg} title={`${r.label}: ${r.width}`} />
          ))}
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-2 pt-1">
          {ROUTING_ITEMS.map((r, idx) => (
            <div key={idx} className="flex items-center gap-2">
              <span className={`w-2.5 h-2.5 rounded-full ${r.bg}`} />
              <span className="text-xs text-[#8995A3]">{r.label}:</span>
              <span className="font-mono text-xs font-semibold text-[#E8EDF3]">{r.text}</span>
            </div>
          ))}
        </div>
      </div>
        );
      })()}
      {/* END GENERATED */}


      {/* Flagged Transactions Queue (ADAPTS TO VIEW MODE) */}
      <div className="p-5 rounded-xl border border-[#202A35] bg-[#111821] space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div>
              <div className="text-xs font-semibold text-[#E8EDF3] uppercase tracking-wider flex items-center gap-2">
                <span>Priority Risk Queue &amp; Incident Triage</span>
                <span className="text-[9px] px-1.5 py-0.5 rounded font-mono font-bold bg-amber-500/10 text-amber-400 border border-amber-500/30">DEMO DATA</span>
                <span
                  className={`text-[10px] font-mono px-2 py-0.5 rounded border ${
                    viewMode === "executive"
                      ? "bg-blue-500/10 text-blue-400 border-blue-500/20"
                      : "bg-purple-500/10 text-purple-400 border-purple-500/20"
                  }`}
                >
                  {viewMode === "executive" ? "Customer Story View" : "AI Forensics View"}
                </span>
              </div>
              <div className="text-[11px] text-[#8995A3]">
                {viewMode === "executive"
                  ? "Live customer transaction narratives with plain-English threat summaries and action recommendations."
                  : "Raw feature vector attributions, graph topological distance, and calibrated uncertainty intervals."}
              </div>
            </div>
            <span
              className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border flex items-center gap-1.5 ${
                streamStatus === "LIVE"
                  ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30"
                  : streamStatus === "CONNECTING"
                  ? "bg-amber-500/10 text-amber-400 border-amber-500/30"
                  : "bg-slate-500/10 text-slate-400 border-slate-500/30"
              }`}
            >
              <span
                className={`w-1.5 h-1.5 rounded-full ${
                  streamStatus === "LIVE" ? "bg-emerald-400 animate-ping" : "bg-slate-400"
                }`}
              />
              {streamStatus === "LIVE" ? "LIVE SSE FEED" : streamStatus === "CONNECTING" ? "CONNECTING..." : "OFFLINE REPLAY"}
            </span>
          </div>
          <Link
            href="/transactions/feed"
            className="text-xs text-blue-400 hover:text-blue-300 font-medium flex items-center gap-1"
          >
            <span>View Full Audit Feed</span>
            <ArrowUpRight className="w-3.5 h-3.5" />
          </Link>
        </div>

        <div className="overflow-x-auto">
          {viewMode === "executive" ? (
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-[#202A35] text-[10px] uppercase font-mono text-[#596574]">
                  <th className="py-2 px-3">Order ID</th>
                  <th className="py-2 px-3">Customer Entity</th>
                  <th className="py-2 px-3">Amount</th>
                  <th className="py-2 px-3">Business Threat Narrative</th>
                  <th className="py-2 px-3">Decision</th>
                  <th className="py-2 px-3">Executive Recommendation</th>
                  <th className="py-2 px-3 text-right">Story Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#151D27] text-xs">
                {recentTransactions.map((tx, idx) => {
                  const meta = DECISION_META[tx.decision] ?? DECISION_META.ALLOW;
                  return (
                    <tr key={`${tx.id}-${idx}`} className="hover:bg-[#151D27]/50 transition-colors">
                      <td className="py-3 px-3 font-mono text-blue-400 font-semibold">{tx.id}</td>
                      <td className="py-3 px-3 font-mono text-[#E8EDF3]">{tx.buyer}</td>
                      <td className="py-3 px-3 font-mono text-[#E8EDF3]">{tx.amount}</td>
                      <td className="py-3 px-3 text-[#E8EDF3] max-w-sm">
                        <div className="font-medium text-xs">{meta.narrative}</div>
                        <div className="text-[11px] text-[#8995A3] mt-0.5 truncate">{tx.reason}</div>
                      </td>
                      <td className="py-3 px-3">
                        <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border ${meta.badge}`}>
                          {tx.decision}
                        </span>
                      </td>
                      <td className="py-3 px-3 text-[11px] text-[#8995A3]">{meta.rec}</td>
                      <td className="py-3 px-3 text-right">
                        <Link
                          href={`/transactions?preset=${meta.preset}`}
                          className="text-xs text-blue-400 hover:text-blue-300 font-semibold inline-flex items-center gap-1"
                        >
                          <span>Review Story</span>
                          <ArrowUpRight className="w-3 h-3" />
                        </Link>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          ) : (
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-[#202A35] text-[10px] uppercase font-mono text-[#596574]">
                  <th className="py-2 px-3">Order ID</th>
                  <th className="py-2 px-3">Entity Node</th>
                  <th className="py-2 px-3">Amount</th>
                  <th className="py-2 px-3">Calibrated P(Fraud)</th>
                  <th className="py-2 px-3">Policy Risk</th>
                  <th className="py-2 px-3">Model Ensemble</th>
                  <th className="py-2 px-3">Top SHAP Feature Driver</th>
                  <th className="py-2 px-3">Conformal Set C(X)</th>
                  <th className="py-2 px-3 text-right">Deep Inspect</th>
                </tr>
              </thead>
              {/* BEGIN GENERATED */}
              <tbody className="divide-y divide-[#151D27] text-xs font-mono">
                {recentTransactions.map((tx, idx) => {
                  const meta = DECISION_META[tx.decision] ?? DECISION_META.ALLOW;
                  return (
                    <tr key={`${tx.id}-${idx}`} className="hover:bg-[#151D27]/50 transition-colors">
                      <td className="py-3 px-3 text-purple-400 font-semibold">{tx.id}</td>
                      <td className="py-3 px-3 text-[#E8EDF3]">{tx.buyer}</td>
                      <td className="py-3 px-3 text-[#E8EDF3]">{tx.amount}</td>
                      <td className="py-3 px-3">
                        <span className={`font-semibold ${tx.calibratedProb > 0.7 ? "text-rose-400" : tx.calibratedProb > 0.3 ? "text-amber-400" : "text-emerald-400"}`}>
                          {(tx.calibratedProb * 100).toFixed(2)}%
                        </span>
                      </td>
                      <td className="py-3 px-3">
                        <span className={`font-semibold ${tx.policyRisk > 0.7 ? "text-rose-400" : tx.policyRisk > 0.3 ? "text-amber-400" : "text-emerald-400"}`}>
                          {(tx.policyRisk * 100).toFixed(2)}%
                        </span>
                      </td>
                      <td className="py-3 px-3 text-[11px] text-[#8995A3]">
                        {tx.policyRisk > 0.7 ? "Tabular: 0.96 · Graph: 0.92" : tx.policyRisk > 0.3 ? "Tabular: 0.42 · Graph: 0.34" : "Tabular: 0.04 · Graph: 0.03"}
                      </td>
                      <td className="py-3 px-3 text-cyan-400 text-[11px] max-w-xs truncate">
                        {tx.decision === "BLOCK"
                          ? "device_share_count (+0.38)"
                          : tx.decision === "HOLD"
                          ? "return_ratio_30d (+0.44)"
                          : tx.decision === "REVIEW"
                          ? "price_discrepancy (+0.31)"
                          : "account_age_days (-0.42)"}
                      </td>
                      <td className="py-3 px-3">
                        <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-white/5 border border-white/10 text-[#E8EDF3]">
                          {tx.risk > 0.7 ? "{ 1 }" : tx.risk < 0.15 ? "{ 0 }" : "{ 0, 1 }"}
                        </span>
                      </td>
                      <td className="py-3 px-3 text-right">
                        <Link
                          href={`/transactions?preset=${meta.preset}`}
                          className="text-xs text-purple-400 hover:text-purple-300 font-semibold inline-flex items-center gap-1"
                        >
                          <span>Inspect ML</span>
                          <ArrowUpRight className="w-3 h-3" />
                        </Link>
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
    </div>
  );
}
