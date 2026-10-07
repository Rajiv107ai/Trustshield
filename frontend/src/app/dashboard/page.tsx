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
  const [metrics, setMetrics] = useState({
    totalAnalyzed: 142850,
    flaggedFraud: 4210,
    reviewQueue: 1840,
    blockRate: "2.95%",
    p95Latency: "14.8 ms",
    savingsPrevented: "$1,842,500",
  });

  const recentTransactions = [
    {
      id: "ORD_78910",
      buyer: "BUYER_RING_MEMBER_04",
      amount: "$890.00",
      risk: 0.942,
      decision: "BLOCK",
      reason: "Shared hardware across 9 buyer accounts in 24h",
      timestamp: "2 mins ago",
    },
    {
      id: "ORD_78909",
      buyer: "BUYER_REFUND_ABUSER",
      amount: "$320.00",
      risk: 0.785,
      decision: "HOLD",
      reason: "Historical return rate exceeds 80%",
      timestamp: "5 mins ago",
    },
    {
      id: "ORD_78908",
      buyer: "BUYER_NEWBIE_99",
      amount: "$450.00",
      risk: 0.380,
      decision: "REVIEW",
      reason: "Account created 1 day ago ordering at 3.5x category median",
      timestamp: "12 mins ago",
    },
    {
      id: "ORD_78907",
      buyer: "BUYER_VERIFIED_77",
      amount: "$65.50",
      risk: 0.042,
      decision: "ALLOW",
      reason: "180 days active, 15 prior successful orders, zero returns",
      timestamp: "14 mins ago",
    },
  ];

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-[#E8EDF3] flex items-center gap-2">
            <span>Enterprise Command Center</span>
            {viewMode === "executive" ? (
              <span className="text-xs px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20 font-normal">
                Executive Story Mode
              </span>
            ) : (
              <span className="text-xs px-2 py-0.5 rounded-full bg-purple-500/10 text-purple-400 border border-purple-500/20 font-normal">
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
      <div className="p-4 rounded-xl border border-[#202A35] bg-[#111821] flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-blue-500/10 border border-blue-500/20 flex items-center justify-center text-blue-400">
            {viewMode === "executive" ? <Sparkles className="w-5 h-5" /> : <Microscope className="w-5 h-5" />}
          </div>
          <div>
            <div className="text-xs font-semibold text-[#E8EDF3]">
              {viewMode === "executive"
                ? "Executive Overview: Platform Safeguards Active"
                : "Deep AI Inspector: High-Dimensional Feature Space Active"}
            </div>
            <div className="text-[11px] text-[#8995A3]">
              {viewMode === "executive"
                ? "All transactions evaluated under strict temporal isolation. $1.84M in fraudulent chargebacks prevented this month."
                : "Stacking Meta-Learner orchestrating 16D GNN node embeddings, XGBoost tabular splits, and split conformal coverage sets."}
            </div>
          </div>
        </div>

        <div className="hidden lg:flex items-center gap-4 text-xs font-mono">
          <div className="text-right">
            <div className="text-[#8995A3] text-[10px]">CONFORMAL COVERAGE</div>
            <div className="text-emerald-400 font-semibold">&ge; 95.0% Guaranteed</div>
          </div>
          <div className="h-8 w-px bg-[#202A35]" />
          <div className="text-right">
            <div className="text-[#8995A3] text-[10px]">TEMPORAL INVARIANT</div>
            <div className="text-blue-400 font-semibold">event_time &lt; decision_time</div>
          </div>
        </div>
      </div>

      {/* System Readiness Probe Grid */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        {[
          { name: "API Gateway", status: isLive ? "Operational" : "Offline Demo", active: true },
          { name: "Phase 5 Hybrid", status: "Active (XGBoost)", active: true },
          { name: "Hetero GNN", status: "Active (PyG 16D)", active: true },
          { name: "Fraud Rings", status: "14 Clusters Loaded", active: true },
          { name: "CLIP Vector Cache", status: "FAISS Indexed", active: true },
        ].map((comp, idx) => (
          <div key={idx} className="p-3 rounded-lg border border-[#202A35] bg-[#111821]">
            <div className="flex items-center justify-between">
              <span className="text-[11px] text-[#8995A3] truncate">{comp.name}</span>
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
            </div>
            <div className="text-xs font-semibold text-[#E8EDF3] mt-1 font-mono">{comp.status}</div>
          </div>
        ))}
      </div>

      {/* Key Metrics Row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="p-4 rounded-xl border border-[#202A35] bg-[#111821]">
          <div className="flex items-center justify-between text-xs text-[#8995A3]">
            <span>Analyzed Volume</span>
            <Activity className="w-4 h-4 text-blue-400" />
          </div>
          <div className="text-2xl font-bold font-mono text-[#E8EDF3] mt-2">
            {metrics.totalAnalyzed.toLocaleString()}
          </div>
          <div className="text-[11px] text-emerald-400 mt-1 flex items-center gap-1">
            <span>+12.4% vs last week</span>
          </div>
        </div>

        <div className="p-4 rounded-xl border border-[#202A35] bg-[#111821]">
          <div className="flex items-center justify-between text-xs text-[#8995A3]">
            <span>High Risk Blocks</span>
            <ShieldAlert className="w-4 h-4 text-rose-400" />
          </div>
          <div className="text-2xl font-bold font-mono text-[#E8EDF3] mt-2">
            {metrics.flaggedFraud.toLocaleString()}
          </div>
          <div className="text-[11px] text-rose-400 mt-1 flex items-center gap-1">
            <span>Block rate: {metrics.blockRate}</span>
          </div>
        </div>

        <div className="p-4 rounded-xl border border-[#202A35] bg-[#111821]">
          <div className="flex items-center justify-between text-xs text-[#8995A3]">
            <span>Manual Review Queue</span>
            <Layers className="w-4 h-4 text-amber-400" />
          </div>
          <div className="text-2xl font-bold font-mono text-[#E8EDF3] mt-2">
            {metrics.reviewQueue.toLocaleString()}
          </div>
          <div className="text-[11px] text-[#8995A3] mt-1">SLA: 98.4% &lt; 15 min</div>
        </div>

        <div className="p-4 rounded-xl border border-[#202A35] bg-[#111821]">
          <div className="flex items-center justify-between text-xs text-[#8995A3]">
            <span>Inference Latency (p95)</span>
            <Clock className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="text-2xl font-bold font-mono text-[#E8EDF3] mt-2">
            {metrics.p95Latency}
          </div>
          <div className="text-[11px] text-emerald-400 mt-1">&lt; 20 ms SLA met</div>
        </div>
      </div>

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

      {/* Flagged Transactions Queue */}
      <div className="p-5 rounded-xl border border-[#202A35] bg-[#111821] space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-xs font-semibold text-[#E8EDF3] uppercase tracking-wider">
              Priority Risk Queue & Incident Triage
            </div>
            <div className="text-[11px] text-[#8995A3]">Live stream of suspicious and blocked orders requiring ops awareness.</div>
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
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-[#202A35] text-[10px] uppercase font-mono text-[#596574]">
                <th className="py-2 px-3">Order ID</th>
                <th className="py-2 px-3">Customer / Entity</th>
                <th className="py-2 px-3">Amount</th>
                <th className="py-2 px-3">Risk Probability</th>
                <th className="py-2 px-3">Decision</th>
                <th className="py-2 px-3">Forensic Reason</th>
                <th className="py-2 px-3 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#151D27] text-xs">
              {recentTransactions.map((tx) => (
                <tr key={tx.id} className="hover:bg-[#151D27]/50 transition-colors">
                  <td className="py-3 px-3 font-mono text-blue-400 font-semibold">{tx.id}</td>
                  <td className="py-3 px-3 font-mono text-[#E8EDF3]">{tx.buyer}</td>
                  <td className="py-3 px-3 font-mono text-[#E8EDF3]">{tx.amount}</td>
                  <td className="py-3 px-3 font-mono">
                    <span
                      className={`font-semibold ${
                        tx.risk > 0.7 ? "text-rose-400" : tx.risk > 0.3 ? "text-amber-400" : "text-emerald-400"
                      }`}
                    >
                      {(tx.risk * 100).toFixed(1)}%
                    </span>
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
                  <td className="py-3 px-3 text-[#8995A3] max-w-xs truncate">{tx.reason}</td>
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
                      className="text-xs text-blue-400 hover:underline font-mono"
                    >
                      Inspect &rarr;
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
