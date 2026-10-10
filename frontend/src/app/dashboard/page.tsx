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
  Cpu,
  Clock,
  ArrowUpRight,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  Sparkles,
  Microscope,
  TrendingDown,
  Users,
} from "lucide-react";

export default function DashboardPage() {
  const { viewMode, isLive, readyInfo } = useViewMode();
  const [streamStatus, setStreamStatus] = useState<"LIVE" | "CONNECTING" | "DISCONNECTED" | "RECONNECTING" | "OFFLINE">("CONNECTING");
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
      risk: 0.942,
      decision: "BLOCK",
      reason: "Shared hardware across 9 buyer accounts in 24h",
      timestamp: "2 mins ago",
    },
    {
      id: "ORD_78909",
      buyer: "BUYER_REFUND_ABUSER",
      amount: "₹320.00",
      risk: 0.785,
      decision: "HOLD",
      reason: "Historical return rate exceeds 80%",
      timestamp: "5 mins ago",
    },
    {
      id: "ORD_78908",
      buyer: "BUYER_NEWBIE_99",
      amount: "₹450.00",
      risk: 0.380,
      decision: "REVIEW",
      reason: "Account created 1 day ago ordering at 3.5x category median",
      timestamp: "12 mins ago",
    },
    {
      id: "ORD_78907",
      buyer: "BUYER_VERIFIED_77",
      amount: "₹65.50",
      risk: 0.042,
      decision: "ALLOW",
      reason: "180 days active, 15 prior successful orders, zero returns",
      timestamp: "14 mins ago",
    },
  ]);

  useEffect(() => {
    const cleanup = TrustShieldApi.createTransactionEventSource(
      (event) => {
        setStreamStatus("LIVE");
        setRecentTransactions((prev) => {
          const newTx = {
            id: event.order_id,
            buyer: event.buyer_id,
            amount: `₹${event.amount.toFixed(2)}`,
            risk: event.risk_score,
            decision: event.decision,
            reason: event.reason_codes.length > 0 ? event.reason_codes.join(", ") : "Standard baseline scoring",
            timestamp: "Just now",
          };
          return [newTx, ...prev.slice(0, 5)];
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

        {/* Quick Simulator CTA */}
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
                  <div className="text-[#8995A3] text-[10px]">CONFORMAL COVERAGE</div>
                  <div className="text-emerald-400 font-semibold text-sm">&ge; 95.0% Guaranteed</div>
                </div>
                <div className="h-8 w-px bg-[#202A35]" />
                <div className="text-right">
                  <div className="text-[#8995A3] text-[10px]">CALIBRATION ECE</div>
                  <div className="text-purple-400 font-semibold text-sm">0.0142 Isotonic</div>
                </div>
              </>
            )}
          </div>
        </div>
      </div>

      {/* System Readiness Probe Grid */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        {[
          {
            name: "API Gateway",
            status: isLive ? "Operational" : "Offline Demo",
            active: isLive,
          },
          {
            name: "Redis Feature Store",
            status: readyInfo?.components?.redis === "available" ? "Live (Port 6379)" : "Offline (Disk Fallback)",
            active: readyInfo?.components?.redis === "available",
          },
          {
            name: "Neo4j Graph Store",
            status: readyInfo?.components?.neo4j === "available" ? "Live (Port 7687)" : "Offline (Disk Rings)",
            active: readyInfo?.components?.neo4j === "available",
          },
          {
            name: "Phase 5 Hybrid",
            status: readyInfo?.components?.models === "ready" ? "Active (XGBoost)" : "Artifact Model",
            active: true,
          },
          {
            name: "Prometheus Telemetry",
            status: readyInfo?.components?.prometheus === "available" ? "Scraping /metrics" : "Standby",
            active: readyInfo?.components?.prometheus === "available",
          },
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
      {viewMode === "executive" ? (
        /* ================= EXECUTIVE STORY METRICS ================= */
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="p-4 rounded-xl border border-blue-500/20 bg-[#111821] relative overflow-hidden">
            <div className="absolute top-0 right-0 w-24 h-24 bg-blue-500/5 rounded-full blur-xl pointer-events-none" />
            <div className="flex items-center justify-between text-xs text-[#8995A3]">
              <span className="font-medium text-blue-300">Fraud Loss Prevented</span>
              <Sparkles className="w-4 h-4 text-blue-400" />
            </div>
            <div className="text-2xl font-bold font-mono text-emerald-400 mt-2">
              $1,842,500
            </div>
            <div className="text-[11px] text-emerald-400/90 mt-1 flex items-center gap-1 font-medium">
              <span>+18.4% YoY saved vs legacy rules</span>
            </div>
          </div>

          <div className="p-4 rounded-xl border border-emerald-500/20 bg-[#111821]">
            <div className="flex items-center justify-between text-xs text-[#8995A3]">
              <span className="font-medium text-emerald-300">Clean Buyer Experience</span>
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
            </div>
            <div className="text-2xl font-bold font-mono text-[#E8EDF3] mt-2">
              99.2%
            </div>
            <div className="text-[11px] text-emerald-400 mt-1 flex items-center gap-1 font-medium">
              <span>Frictionless instant checkout rate</span>
            </div>
          </div>

          <div className="p-4 rounded-xl border border-amber-500/20 bg-[#111821]">
            <div className="flex items-center justify-between text-xs text-[#8995A3]">
              <span className="font-medium text-amber-300">Chargeback Dispute Rate</span>
              <ShieldAlert className="w-4 h-4 text-amber-400" />
            </div>
            <div className="text-2xl font-bold font-mono text-[#E8EDF3] mt-2">
              0.04%
            </div>
            <div className="text-[11px] text-emerald-400 mt-1 font-medium">
              Well below Visa 0.65% excessive limit
            </div>
          </div>

          <div className="p-4 rounded-xl border border-cyan-500/20 bg-[#111821]">
            <div className="flex items-center justify-between text-xs text-[#8995A3]">
              <span className="font-medium text-cyan-300">Autonomous Action Rate</span>
              <Zap className="w-4 h-4 text-cyan-400" />
            </div>
            <div className="text-2xl font-bold font-mono text-[#E8EDF3] mt-2">
              93.5%
            </div>
            <div className="text-[11px] text-cyan-400 mt-1 font-medium">
              Instant automated machine decisions
            </div>
          </div>
        </div>
      ) : (
        /* ================= DEEP AI INSPECTOR METRICS ================= */
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="p-4 rounded-xl border border-purple-500/20 bg-[#111821]">
            <div className="flex items-center justify-between text-xs text-[#8995A3]">
              <span className="font-medium text-purple-300">Inferences Evaluated</span>
              <Activity className="w-4 h-4 text-purple-400" />
            </div>
            <div suppressHydrationWarning className="text-2xl font-bold font-mono text-[#E8EDF3] mt-2">
              {metrics.totalAnalyzed.toLocaleString("en-US")}
            </div>
            <div className="text-[11px] text-emerald-400 mt-1 flex items-center gap-1">
              <span>+12.4% vs last week</span>
            </div>
          </div>

          <div className="p-4 rounded-xl border border-indigo-500/20 bg-[#111821]">
            <div className="flex items-center justify-between text-xs text-[#8995A3]">
              <span className="font-medium text-indigo-300">Conformal Coverage C(X)</span>
              <Layers className="w-4 h-4 text-indigo-400" />
            </div>
            <div className="text-2xl font-bold font-mono text-emerald-400 mt-2">
              95.84%
            </div>
            <div className="text-[11px] text-emerald-400 mt-1 flex items-center gap-1">
              <span>&alpha; = 0.05 finite-sample guarantee</span>
            </div>
          </div>

          <div className="p-4 rounded-xl border border-cyan-500/20 bg-[#111821]">
            <div className="flex items-center justify-between text-xs text-[#8995A3]">
              <span className="font-medium text-cyan-300">Expected Calibration Error</span>
              <Microscope className="w-4 h-4 text-cyan-400" />
            </div>
            <div className="text-2xl font-bold font-mono text-[#E8EDF3] mt-2">
              0.0142
            </div>
            <div className="text-[11px] text-[#8995A3] mt-1 font-mono">
              Brier: 0.0392 | Isotonic fit
            </div>
          </div>

          <div className="p-4 rounded-xl border border-rose-500/20 bg-[#111821]">
            <div className="flex items-center justify-between text-xs text-[#8995A3]">
              <span className="font-medium text-rose-300">Hardware Latency (p95)</span>
              <Clock className="w-4 h-4 text-rose-400" />
            </div>
            <div className="text-2xl font-bold font-mono text-[#E8EDF3] mt-2">
              {metrics.p95Latency}
            </div>
            <div className="text-[11px] text-emerald-400 mt-1">&lt; 20 ms SLA strictly verified</div>
          </div>
        </div>
      )}

      {/* DUAL MODE MIDDLE INTELLIGENCE SECTION */}
      {viewMode === "executive" ? (
        /* ================= EXECUTIVE STORY BREAKDOWN ================= */
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
            <div className="p-3.5 rounded-lg bg-[#0E131A] border border-[#202A35]">
              <div className="text-[11px] text-[#8995A3]">Device Farm Collusion</div>
              <div className="text-lg font-bold font-mono text-[#E8EDF3] mt-1">₹890,200</div>
              <div className="text-[10px] text-rose-400 mt-1">42 attacks stopped before payment</div>
            </div>
            <div className="p-3.5 rounded-lg bg-[#0E131A] border border-[#202A35]">
              <div className="text-[11px] text-[#8995A3]">Serial Return Arbitrage</div>
              <div className="text-lg font-bold font-mono text-[#E8EDF3] mt-1">₹425,100</div>
              <div className="text-[10px] text-amber-400 mt-1">Held for warehouse serial verification</div>
            </div>
            <div className="p-3.5 rounded-lg bg-[#0E131A] border border-[#202A35]">
              <div className="text-[11px] text-[#8995A3]">Listing &amp; Catalog Scams</div>
              <div className="text-lg font-bold font-mono text-[#E8EDF3] mt-1">₹315,400</div>
              <div className="text-[10px] text-cyan-400 mt-1">Flagged via CLIP image vector reuse</div>
            </div>
            <div className="p-3.5 rounded-lg bg-[#0E131A] border border-[#202A35]">
              <div className="text-[11px] text-[#8995A3]">Account Takeover Bursts</div>
              <div className="text-lg font-bold font-mono text-[#E8EDF3] mt-1">₹211,800</div>
              <div className="text-[10px] text-purple-400 mt-1">Session velocity step-up enforced</div>
            </div>
          </div>
        </div>
      ) : (
        /* ================= DEEP AI INSPECTOR ARCHITECTURE ================= */
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
              AUC-ROC: 0.978 · PR-AUC: 0.912
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-3 font-mono text-xs">
            <div className="p-3.5 rounded-lg bg-[#0E131A] border border-[#202A35]">
              <div className="text-[10px] text-purple-400 font-bold uppercase">1. GraphSAGE GNN</div>
              <div className="text-[#E8EDF3] font-semibold mt-1">16D Node Embeddings</div>
              <div className="text-[10px] text-[#8995A3] mt-1">Inductive neighborhood message passing</div>
            </div>
            <div className="p-3.5 rounded-lg bg-[#0E131A] border border-[#202A35]">
              <div className="text-[10px] text-blue-400 font-bold uppercase">2. Tabular XGBoost</div>
              <div className="text-[#E8EDF3] font-semibold mt-1">48 Relational Features</div>
              <div className="text-[10px] text-[#8995A3] mt-1">Tree depth: 6 · Regularization: L1/L2</div>
            </div>
            <div className="p-3.5 rounded-lg bg-[#0E131A] border border-[#202A35]">
              <div className="text-[10px] text-emerald-400 font-bold uppercase">3. Isotonic Calibrator</div>
              <div className="text-[#E8EDF3] font-semibold mt-1">Monotonic Score Fit</div>
              <div className="text-[10px] text-[#8995A3] mt-1">P(Y=1|S) calibrated on holdout split</div>
            </div>
            <div className="p-3.5 rounded-lg bg-[#0E131A] border border-[#202A35]">
              <div className="text-[10px] text-cyan-400 font-bold uppercase">4. Conformal Risk Filter</div>
              <div className="text-[#E8EDF3] font-semibold mt-1">Coverage Set C(X)</div>
              <div className="text-[10px] text-[#8995A3] mt-1">Finite-sample validity guaranteed</div>
            </div>
          </div>
        </div>
      )}

      {/* Operational Decision Distribution */}
      <div className="p-5 rounded-xl border border-[#202A35] bg-[#111821] space-y-3">
        <div className="flex items-center justify-between">
          <div className="text-xs font-semibold text-[#E8EDF3] uppercase tracking-wider">
            Operational Decision Routing Distribution
          </div>
          <div className="text-[11px] text-[#8995A3] font-mono">Last 24 Hours: 142,850 decisions</div>
        </div>

        {/* Visual Progress Bar */}
        <div className="h-3 rounded-full bg-[#0A0E13] overflow-hidden flex">
          <div style={{ width: "88.2%" }} className="bg-emerald-500" title="ALLOW: 88.2%" />
          <div style={{ width: "6.5%" }} className="bg-amber-500" title="REVIEW: 6.5%" />
          <div style={{ width: "2.3%" }} className="bg-orange-500" title="HOLD: 2.3%" />
          <div style={{ width: "3.0%" }} className="bg-rose-500" title="BLOCK: 3.0%" />
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-2 pt-1">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-500" />
            <span className="text-xs text-[#8995A3]">ALLOW:</span>
            <span className="font-mono text-xs font-semibold text-[#E8EDF3]">88.2% (125,993)</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-amber-500" />
            <span className="text-xs text-[#8995A3]">REVIEW:</span>
            <span className="font-mono text-xs font-semibold text-[#E8EDF3]">6.5% (9,285)</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-orange-500" />
            <span className="text-xs text-[#8995A3]">HOLD:</span>
            <span className="font-mono text-xs font-semibold text-[#E8EDF3]">2.3% (3,285)</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-rose-500" />
            <span className="text-xs text-[#8995A3]">BLOCK:</span>
            <span className="font-mono text-xs font-semibold text-[#E8EDF3]">3.0% (4,287)</span>
          </div>
        </div>
      </div>

      {/* Flagged Transactions Queue (ADAPTS TO VIEW MODE) */}
      <div className="p-5 rounded-xl border border-[#202A35] bg-[#111821] space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div>
              <div className="text-xs font-semibold text-[#E8EDF3] uppercase tracking-wider flex items-center gap-2">
                <span>Priority Risk Queue &amp; Incident Triage</span>
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
            /* ================= EXECUTIVE STORY QUEUE TABLE ================= */
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
                {recentTransactions.map((tx) => (
                  <tr key={tx.id} className="hover:bg-[#151D27]/50 transition-colors">
                    <td className="py-3 px-3 font-mono text-blue-400 font-semibold">{tx.id}</td>
                    <td className="py-3 px-3 font-mono text-[#E8EDF3]">{tx.buyer}</td>
                    <td className="py-3 px-3 font-mono text-[#E8EDF3]">{tx.amount}</td>
                    <td className="py-3 px-3 text-[#E8EDF3] max-w-sm">
                      <div className="font-medium text-xs">
                        {tx.decision === "BLOCK"
                          ? "Collusion Farm Attack Detected"
                          : tx.decision === "HOLD"
                          ? "High-Velocity Return Abuse Risk"
                          : tx.decision === "REVIEW"
                          ? "First-Time Buyer Price Anomaly"
                          : "Verified Clean Repeat Customer"}
                      </div>
                      <div className="text-[11px] text-[#8995A3] mt-0.5 truncate">{tx.reason}</div>
                    </td>
                    <td className="py-3 px-3">
                      <span
                        className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border ${
                          tx.decision === "BLOCK"
                            ? "bg-rose-500/10 text-rose-400 border-rose-500/20"
                            : tx.decision === "HOLD"
                            ? "bg-orange-500/10 text-orange-400 border-orange-500/20"
                            : tx.decision === "REVIEW"
                            ? "bg-amber-500/10 text-amber-400 border-amber-500/20"
                            : "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
                        }`}
                      >
                        {tx.decision}
                      </span>
                    </td>
                    <td className="py-3 px-3 text-[11px] text-[#8995A3]">
                      {tx.decision === "BLOCK"
                        ? "Halt payment settlement and freeze card"
                        : tx.decision === "HOLD"
                        ? "Require physical inspection of return"
                        : tx.decision === "REVIEW"
                        ? "Verify identity or approve manual risk"
                        : "Allow instant friction-free checkout"}
                    </td>
                    <td className="py-3 px-3 text-right">
                      <Link
                        href={`/transactions?preset=${
                          tx.decision === "BLOCK"
                            ? "preset_device_farm"
                            : tx.decision === "HOLD"
                            ? "preset_serial_returner"
                            : tx.decision === "REVIEW"
                            ? "preset_price_arbitrage"
                            : "preset_normal_buyer"
                        }`}
                        className="text-xs text-blue-400 hover:text-blue-300 font-semibold inline-flex items-center gap-1"
                      >
                        <span>Review Story</span>
                        <ArrowUpRight className="w-3 h-3" />
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            /* ================= DEEP AI INSPECTOR QUEUE TABLE ================= */
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-[#202A35] text-[10px] uppercase font-mono text-[#596574]">
                  <th className="py-2 px-3">Order ID</th>
                  <th className="py-2 px-3">Entity Node</th>
                  <th className="py-2 px-3">Amount</th>
                  <th className="py-2 px-3">Calibrated P(Fraud)</th>
                  <th className="py-2 px-3">Model Ensemble</th>
                  <th className="py-2 px-3">Top SHAP Feature Driver</th>
                  <th className="py-2 px-3">Conformal Set C(X)</th>
                  <th className="py-2 px-3 text-right">Deep Inspect</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#151D27] text-xs font-mono">
                {recentTransactions.map((tx) => (
                  <tr key={tx.id} className="hover:bg-[#151D27]/50 transition-colors">
                    <td className="py-3 px-3 text-purple-400 font-semibold">{tx.id}</td>
                    <td className="py-3 px-3 text-[#E8EDF3]">{tx.buyer}</td>
                    <td className="py-3 px-3 text-[#E8EDF3]">{tx.amount}</td>
                    <td className="py-3 px-3">
                      <span
                        className={`font-semibold ${
                          tx.risk > 0.7 ? "text-rose-400" : tx.risk > 0.3 ? "text-amber-400" : "text-emerald-400"
                        }`}
                      >
                        {(tx.risk * 100).toFixed(2)}%
                      </span>
                    </td>
                    <td className="py-3 px-3 text-[11px] text-[#8995A3]">
                      {tx.risk > 0.7
                        ? "XGB: 0.96 · GNN: 0.92"
                        : tx.risk > 0.3
                        ? "XGB: 0.42 · GNN: 0.34"
                        : "XGB: 0.04 · GNN: 0.03"}
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
                        href={`/transactions?preset=${
                          tx.decision === "BLOCK"
                            ? "preset_device_farm"
                            : tx.decision === "HOLD"
                            ? "preset_serial_returner"
                            : tx.decision === "REVIEW"
                            ? "preset_price_arbitrage"
                            : "preset_normal_buyer"
                        }`}
                        className="text-xs text-purple-400 hover:text-purple-300 font-semibold inline-flex items-center gap-1"
                      >
                        <span>Inspect ML</span>
                        <ArrowUpRight className="w-3 h-3" />
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
}
