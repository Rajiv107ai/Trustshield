"use client";

import React, { useState, useEffect } from "react";
import {
  Activity,
  Search,
  ArrowUpRight,
  Wifi,
  WifiOff,
  RefreshCw,
} from "lucide-react";
import { TrustShieldApi } from "@/lib/api/client";
import { StreamTransactionEvent } from "@/lib/types/api";
import { DecisionBadge } from "@/components/ui/DecisionBadge";
import { EmptyState } from "@/components/ui/EmptyState";
import {
  TransactionDetailDrawer,
  DrawerTransactionData,
} from "@/components/transactions/TransactionDetailDrawer";

interface FeedItem {
  id: string;
  timestamp: string;
  buyer: string;
  seller: string;
  amount: number;
  risk: number;
  risk_interval?: [number, number];
  component_risks?: {
    supervised_tabular?: number;
    graph_topology?: number;
    multimodal_listing?: number;
    bayesian_ensemble?: number;
  };
  decision: "ALLOW" | "REVIEW" | "HOLD" | "BLOCK";
  reason: string;
  isNew?: boolean;
}

/* BEGIN GENERATED */
const INITIAL_FEED_DATA: FeedItem[] = [
  { id: "ORD_78921", timestamp: "10:58:12", buyer: "BUYER_RING_MEMBER_04", seller: "SELLER_RING_LEADER_01", amount: 890.00, risk: 0.942, risk_interval: [0.910, 0.974], component_risks: { supervised_tabular: 0.93, graph_topology: 0.97, multimodal_listing: 0.72, bayesian_ensemble: 0.942 }, decision: "BLOCK", reason: "HIGH_DEVICE_COLLISION" },
  { id: "ORD_78920", timestamp: "10:57:44", buyer: "BUYER_REFUND_ABUSER", seller: "SELLER_ELECTRONICS_09", amount: 320.00, risk: 0.785, risk_interval: [0.742, 0.828], component_risks: { supervised_tabular: 0.76, graph_topology: 0.81, multimodal_listing: 0.65, bayesian_ensemble: 0.785 }, decision: "HOLD", reason: "HIGH_RETURN_VELOCITY" },
  { id: "ORD_78919", timestamp: "10:56:30", buyer: "BUYER_NEWBIE_99", seller: "SELLER_UNKNOWN_44", amount: 450.00, risk: 0.380, risk_interval: [0.335, 0.425], component_risks: { supervised_tabular: 0.39, graph_topology: 0.32, multimodal_listing: 0.41, bayesian_ensemble: 0.380 }, decision: "REVIEW", reason: "PRICE_OUTLIER_99TH_PCT" },
  { id: "ORD_78918", timestamp: "10:55:10", buyer: "BUYER_VERIFIED_77", seller: "SELLER_REPUTABLE_12", amount: 65.50, risk: 0.042, risk_interval: [0.018, 0.066], component_risks: { supervised_tabular: 0.04, graph_topology: 0.03, multimodal_listing: 0.05, bayesian_ensemble: 0.042 }, decision: "ALLOW", reason: "CLEAN_BASELINE" },
  { id: "ORD_78917", timestamp: "10:54:02", buyer: "BUYER_MOBILE_USER_11", seller: "SELLER_SHOES_55", amount: 110.00, risk: 0.082, risk_interval: [0.051, 0.113], component_risks: { supervised_tabular: 0.08, graph_topology: 0.07, multimodal_listing: 0.09, bayesian_ensemble: 0.082 }, decision: "ALLOW", reason: "CLEAN_BASELINE" },
  { id: "ORD_78916", timestamp: "10:52:19", buyer: "BUYER_RING_MEMBER_02", seller: "SELLER_RING_LEADER_01", amount: 940.00, risk: 0.915, risk_interval: [0.880, 0.950], component_risks: { supervised_tabular: 0.90, graph_topology: 0.94, multimodal_listing: 0.68, bayesian_ensemble: 0.915 }, decision: "BLOCK", reason: "RING_COLLUSION_SUSPECT" },
  { id: "ORD_78915", timestamp: "10:50:41", buyer: "BUYER_GUEST_104", seller: "SELLER_WATCHES_02", amount: 1200.00, risk: 0.810, risk_interval: [0.768, 0.852], component_risks: { supervised_tabular: 0.82, graph_topology: 0.79, multimodal_listing: 0.84, bayesian_ensemble: 0.810 }, decision: "BLOCK", reason: "COLD_START_BUYER_HIGH_VALUE" },
  { id: "ORD_78914", timestamp: "10:48:33", buyer: "BUYER_ACCOUNT_62", seller: "SELLER_REPUTABLE_12", amount: 44.00, risk: 0.035, risk_interval: [0.012, 0.058], component_risks: { supervised_tabular: 0.03, graph_topology: 0.02, multimodal_listing: 0.04, bayesian_ensemble: 0.035 }, decision: "ALLOW", reason: "CLEAN_BASELINE" },
];

const computeFallbackInterval = (p: number): [number, number] => [
  Math.max(0, Math.round((p - 0.04) * 1000) / 1000),
  Math.min(1, Math.round((p + 0.04) * 1000) / 1000),
];

const computeFallbackComponentRisks = (p: number) => ({
  supervised_tabular: p,
  graph_topology: Math.min(1, Math.round(p * 1.05 * 1000) / 1000),
  multimodal_listing: Math.max(0, Math.round(p * 0.9 * 1000) / 1000),
  bayesian_ensemble: p,
});
/* END GENERATED */

export default function TransactionFeedPage() {
  const [filterDecision, setFilterDecision] = useState<string>("ALL");
  const [search, setSearch] = useState<string>("");
  const [items, setItems] = useState<FeedItem[]>(INITIAL_FEED_DATA);
  const [streamStatus, setStreamStatus] = useState<"LIVE" | "SIMULATION" | "CONNECTING" | "DISCONNECTED" | "RECONNECTING">("CONNECTING");
  const [selectedDrawerTx, setSelectedDrawerTx] = useState<DrawerTransactionData | null>(null);
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);

  const handleOpenDrawer = (tx: FeedItem) => {
    setSelectedDrawerTx({
      order_id: tx.id,
      amount: tx.amount,
      user_id: tx.buyer,
      risk_score: tx.risk,
      risk_interval: tx.risk_interval || computeFallbackInterval(tx.risk),
      component_risks: tx.component_risks || computeFallbackComponentRisks(tx.risk),
      decision: tx.decision,
      timestamp: tx.timestamp,
      account_age_days: tx.decision === "ALLOW" ? 210 : tx.decision === "REVIEW" ? 2 : 0,
      is_missing_device: tx.decision === "BLOCK" || tx.decision === "HOLD",
      top_drivers: [
        {
          name: tx.reason,
          impact: tx.decision === "ALLOW" ? "SAFE" : tx.decision === "REVIEW" ? "MEDIUM_RISK" : "HIGH_RISK",
          description: `Telemetry flagged by ${tx.reason}`,
        },
      ],
    });
    setIsDrawerOpen(true);
  };

  useEffect(() => {
    const cleanup = TrustShieldApi.createTransactionEventSource(
      (event: StreamTransactionEvent) => {
        const newItem: FeedItem = {
          id: event.order_id,
          timestamp: event.timestamp,
          buyer: event.buyer_id,
          seller: event.seller_id,
          amount: event.amount,
          risk: event.calibrated_risk,
          risk_interval: event.risk_interval || computeFallbackInterval(event.calibrated_risk),
          component_risks: event.component_risks || computeFallbackComponentRisks(event.calibrated_risk),
          decision: event.decision as "ALLOW" | "REVIEW" | "HOLD" | "BLOCK",
          reason: event.reason_codes[0] || "CANONICAL_TRUST_ENGINE",
          isNew: true,
        };

        setItems((prev) => [newItem, ...prev.slice(0, 49)]);
      },
      (status) => {
        if (status === "LIVE" || status === "SIMULATION" || status === "CONNECTING" || status === "DISCONNECTED" || status === "RECONNECTING") {
          setStreamStatus(status);
        }
      }
    );

    return () => cleanup();
  }, []);

  const filtered = items.filter((tx) => {
    const matchFilter = filterDecision === "ALL" || tx.decision === filterDecision;
    const matchSearch =
      tx.id.toLowerCase().includes(search.toLowerCase()) ||
      tx.buyer.toLowerCase().includes(search.toLowerCase()) ||
      tx.seller.toLowerCase().includes(search.toLowerCase()) ||
      tx.reason.toLowerCase().includes(search.toLowerCase());
    return matchFilter && matchSearch;
  });

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-xl font-bold tracking-tight text-[var(--text-primary)] flex items-center gap-2">
              <Activity className="w-5 h-5 text-blue-600 dark:text-blue-400" />
              <span>Transaction Live Stream &amp; Audit Feed</span>
            </h1>

            {/* Stream Status Indicator */}
            {streamStatus === "LIVE" && (
              <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-mono font-medium bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/30">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                LIVE STREAM
              </span>
            )}
            {streamStatus === "SIMULATION" && (
              <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-mono font-medium bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/30">
                <span className="w-1.5 h-1.5 rounded-full bg-amber-500 animate-pulse" />
                SIMULATION
              </span>
            )}
            {streamStatus === "CONNECTING" && (
              <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-mono font-medium bg-blue-500/10 text-blue-600 dark:text-blue-400 border border-blue-500/30">
                <RefreshCw className="w-3 h-3 animate-spin" />
                CONNECTING
              </span>
            )}
            {streamStatus === "RECONNECTING" && (
              <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-mono font-medium bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/30">
                <Wifi className="w-3 h-3 animate-pulse" />
                RECONNECTING
              </span>
            )}
            {streamStatus === "DISCONNECTED" && (
              <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-mono font-medium bg-slate-500/10 text-slate-600 dark:text-slate-400 border border-slate-500/30">
                <WifiOff className="w-3 h-3" />
                OFFLINE REPLAY
              </span>
            )}
          </div>
          <p className="text-xs text-[var(--text-secondary)] mt-1">
            Real-time SSE event stream scoring incoming orders dynamically via Canonical Trust Engine.
          </p>
          <p className="text-[10px] text-[var(--text-muted)] italic mt-0.5">
            Note: SSE connection uses URL parameter for compatibility. Use a reverse-proxy with header injection in production.
          </p>
        </div>

        {/* Filter Badges */}
        <div className="flex items-center gap-1.5 flex-wrap">
          {["ALL", "ALLOW", "REVIEW", "HOLD", "BLOCK"].map((dec) => (
            <button
              key={dec}
              onClick={() => setFilterDecision(dec)}
              className={`px-3 py-1 rounded-lg text-xs font-mono font-medium border transition-colors focus-ring ${
                filterDecision === dec
                  ? "bg-blue-600 border-blue-500 text-white shadow-sm"
                  : "bg-[var(--surface)] border-[var(--border)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--surface-elevated)]"
              }`}
            >
              {dec}
            </button>
          ))}
        </div>
      </div>

      {/* Search Bar */}
      <div className="flex items-center px-3.5 py-2.5 rounded-xl bg-[var(--surface)] border border-[var(--border)] shadow-sm">
        <Search className="w-4 h-4 text-[var(--text-muted)] mr-2 shrink-0" />
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Filter by Order ID, Buyer ID, Seller ID, or forensic trigger reason..."
          className="w-full bg-transparent text-xs text-[var(--text-primary)] placeholder-[var(--text-muted)] focus:outline-none font-mono"
        />
        {search && (
          <button
            onClick={() => setSearch("")}
            className="text-xs text-[var(--text-muted)] hover:text-[var(--text-primary)] px-2"
          >
            Clear
          </button>
        )}
      </div>

      {/* Feed Table */}
      <div className="p-4 rounded-xl border border-[var(--border)] bg-[var(--surface)] shadow-md">
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-[var(--border)] text-[10px] uppercase font-mono text-[var(--text-muted)]">
                <th className="py-2.5 px-3">Timestamp</th>
                <th className="py-2.5 px-3">Order ID</th>
                <th className="py-2.5 px-3">Buyer Account</th>
                <th className="py-2.5 px-3">Seller Account</th>
                <th className="py-2.5 px-3">Amount</th>
                <th className="py-2.5 px-3">Calibrated Risk</th>
                <th className="py-2.5 px-3">Operational Decision</th>
                <th className="py-2.5 px-3">Forensic Trigger</th>
                <th className="py-2.5 px-3 text-right">Inspect</th>
              </tr>
            </thead>
            {/* BEGIN GENERATED */}
            <tbody className="divide-y divide-[var(--border)] text-xs font-mono">
              {filtered.map((tx, idx) => (
                <tr
                  key={`${tx.id}-${tx.timestamp}-${idx}`}
                  className={`hover:bg-[var(--surface-elevated)]/60 transition-colors ${
                    tx.isNew ? "bg-blue-500/5 animate-pulse" : ""
                  }`}
                >
                  <td className="py-2.5 px-3 text-[var(--text-muted)] whitespace-nowrap">{tx.timestamp}</td>
                  <td className="py-2.5 px-3 font-semibold text-[var(--text-primary)]">{tx.id}</td>
                  <td className="py-2.5 px-3 text-blue-600 dark:text-blue-400 font-semibold">{tx.buyer}</td>
                  <td className="py-2.5 px-3 text-[var(--text-secondary)]">{tx.seller}</td>
                  <td className="py-2.5 px-3 text-[var(--text-primary)] font-sans font-medium">
                    ${tx.amount.toFixed(2)}
                  </td>
                  <td className="py-2.5 px-3">
                    <div className="flex items-center gap-2">
                      <div className="w-12 h-1.5 rounded-full bg-[var(--surface-elevated)] overflow-hidden border border-[var(--border)]">
                        <div
                          style={{ width: `${Math.min(100, Math.max(0, tx.risk * 100))}%` }}
                          className={`h-full ${
                            tx.risk >= 0.7 ? "bg-rose-500" : tx.risk >= 0.4 ? "bg-amber-500" : "bg-emerald-500"
                          }`}
                        />
                      </div>
                      <span
                        className={`font-bold ${
                          tx.risk >= 0.7
                            ? "text-rose-600 dark:text-rose-400"
                            : tx.risk >= 0.4
                            ? "text-amber-600 dark:text-amber-400"
                            : "text-emerald-600 dark:text-emerald-400"
                        }`}
                      >
                        {(tx.risk * 100).toFixed(1)}%
                      </span>
                    </div>
                  </td>
                  <td className="py-2.5 px-3">
                    <DecisionBadge decision={tx.decision} size="sm" />
                  </td>
                  <td className="py-2.5 px-3 text-[11px] text-[var(--text-secondary)] max-w-[200px] truncate">
                    {tx.reason}
                  </td>
                  <td className="py-2.5 px-3 text-right">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleOpenDrawer(tx);
                      }}
                      className="inline-flex items-center gap-1 text-[11px] text-blue-600 dark:text-blue-400 hover:underline font-sans font-medium cursor-pointer"
                    >
                      <span>Quick View</span>
                      <ArrowUpRight className="w-3 h-3" />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
            {/* END GENERATED */}
          </table>

          {filtered.length === 0 && (
            <div className="py-8">
              <EmptyState
                title="No Transactions Match Filters"
                description={`No events matched the decision filter '${filterDecision}' or query '${search}'.`}
                actionLabel="Reset All Filters"
                onAction={() => {
                  setFilterDecision("ALL");
                  setSearch("");
                }}
              />
            </div>
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
