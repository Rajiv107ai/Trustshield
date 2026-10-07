"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { Activity, Search, Filter, ArrowUpRight, CheckCircle2, AlertTriangle, XCircle, Wifi, WifiOff, RefreshCw } from "lucide-react";
import { TrustShieldApi } from "@/lib/api/client";
import { StreamTransactionEvent } from "@/lib/types/api";

interface FeedItem {
  id: string;
  timestamp: string;
  buyer: string;
  seller: string;
  amount: number;
  risk: number;
  decision: "ALLOW" | "REVIEW" | "HOLD" | "BLOCK";
  reason: string;
  isNew?: boolean;
}

const INITIAL_FEED_DATA: FeedItem[] = [
  { id: "ORD_78921", timestamp: "10:58:12", buyer: "BUYER_RING_MEMBER_04", seller: "SELLER_RING_LEADER_01", amount: 890.00, risk: 0.942, decision: "BLOCK", reason: "HIGH_DEVICE_COLLISION" },
  { id: "ORD_78920", timestamp: "10:57:44", buyer: "BUYER_REFUND_ABUSER", seller: "SELLER_ELECTRONICS_09", amount: 320.00, risk: 0.785, decision: "HOLD", reason: "HIGH_RETURN_VELOCITY" },
  { id: "ORD_78919", timestamp: "10:56:30", buyer: "BUYER_NEWBIE_99", seller: "SELLER_UNKNOWN_44", amount: 450.00, risk: 0.380, decision: "REVIEW", reason: "PRICE_OUTLIER_99TH_PCT" },
  { id: "ORD_78918", timestamp: "10:55:10", buyer: "BUYER_VERIFIED_77", seller: "SELLER_REPUTABLE_12", amount: 65.50, risk: 0.042, decision: "ALLOW", reason: "CLEAN_BASELINE" },
  { id: "ORD_78917", timestamp: "10:54:02", buyer: "BUYER_MOBILE_USER_11", seller: "SELLER_SHOES_55", amount: 110.00, risk: 0.082, decision: "ALLOW", reason: "CLEAN_BASELINE" },
  { id: "ORD_78916", timestamp: "10:52:19", buyer: "BUYER_RING_MEMBER_02", seller: "SELLER_RING_LEADER_01", amount: 940.00, risk: 0.915, decision: "BLOCK", reason: "RING_COLLUSION_SUSPECT" },
  { id: "ORD_78915", timestamp: "10:50:41", buyer: "BUYER_GUEST_104", seller: "SELLER_WATCHES_02", amount: 1200.00, risk: 0.810, decision: "BLOCK", reason: "COLD_START_BUYER_HIGH_VALUE" },
  { id: "ORD_78914", timestamp: "10:48:33", buyer: "BUYER_ACCOUNT_62", seller: "SELLER_REPUTABLE_12", amount: 44.00, risk: 0.035, decision: "ALLOW", reason: "CLEAN_BASELINE" },
];

export default function TransactionFeedPage() {
  const [filterDecision, setFilterDecision] = useState<string>("ALL");
  const [search, setSearch] = useState<string>("");
  const [items, setItems] = useState<FeedItem[]>(INITIAL_FEED_DATA);
  const [streamStatus, setStreamStatus] = useState<"LIVE" | "CONNECTING" | "DISCONNECTED" | "RECONNECTING">("CONNECTING");

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
          decision: event.decision as "ALLOW" | "REVIEW" | "HOLD" | "BLOCK",
          reason: event.reason_codes[0] || "CANONICAL_TRUST_ENGINE",
          isNew: true,
        };

        setItems((prev) => [newItem, ...prev.slice(0, 49)]);
      },
      (status) => {
        if (status === "LIVE" || status === "CONNECTING" || status === "DISCONNECTED" || status === "RECONNECTING") {
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
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-xl font-bold tracking-tight text-[#E8EDF3] flex items-center gap-2">
              <Activity className="w-5 h-5 text-blue-400" />
              <span>Transaction Live Stream & Audit Feed</span>
            </h1>

            {/* Stream Status Indicator */}
            {streamStatus === "LIVE" && (
              <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-mono font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                LIVE STREAM
              </span>
            )}
            {streamStatus === "CONNECTING" && (
              <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-mono font-medium bg-blue-500/10 text-blue-400 border border-blue-500/30">
                <RefreshCw className="w-3 h-3 animate-spin" />
                CONNECTING
              </span>
            )}
            {streamStatus === "RECONNECTING" && (
              <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-mono font-medium bg-amber-500/10 text-amber-400 border border-amber-500/30">
                <Wifi className="w-3 h-3 animate-pulse" />
                RECONNECTING
              </span>
            )}
            {streamStatus === "DISCONNECTED" && (
              <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-mono font-medium bg-zinc-500/10 text-zinc-400 border border-zinc-500/30">
                <WifiOff className="w-3 h-3" />
                OFFLINE REPLAY
              </span>
            )}
          </div>
          <p className="text-xs text-[#8995A3] mt-1">
            Real-time SSE event stream scoring incoming orders dynamically via Canonical Trust Engine.
          </p>
        </div>

        {/* Filter Badges */}
        <div className="flex items-center gap-2">
          {["ALL", "ALLOW", "REVIEW", "HOLD", "BLOCK"].map((dec) => (
            <button
              key={dec}
              onClick={() => setFilterDecision(dec)}
              className={`px-2.5 py-1 rounded text-xs font-mono font-medium border transition-colors ${
                filterDecision === dec
                  ? "bg-blue-600 border-blue-500 text-white"
                  : "bg-[#111821] border-[#202A35] text-[#8995A3] hover:text-[#E8EDF3]"
              }`}
            >
              {dec}
            </button>
          ))}
        </div>
      </div>

      {/* Search Bar */}
      <div className="flex items-center px-3 py-2 rounded-lg bg-[#111821] border border-[#202A35]">
        <Search className="w-4 h-4 text-[#8995A3] mr-2" />
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Filter by Order ID, Buyer ID, Seller ID, or forensic trigger..."
          className="w-full bg-transparent text-xs text-[#E8EDF3] placeholder-[#596574] focus:outline-none font-mono"
        />
      </div>

      {/* Feed Table */}
      <div className="p-4 rounded-xl border border-[#202A35] bg-[#111821]">
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-[#202A35] text-[10px] uppercase font-mono text-[#596574]">
                <th className="py-2 px-3">Timestamp</th>
                <th className="py-2 px-3">Order ID</th>
                <th className="py-2 px-3">Buyer Account</th>
                <th className="py-2 px-3">Seller Account</th>
                <th className="py-2 px-3">Amount</th>
                <th className="py-2 px-3">Calibrated Risk</th>
                <th className="py-2 px-3">Decision</th>
                <th className="py-2 px-3">Forensic Trigger</th>
                <th className="py-2 px-3 text-right">Inspect</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#1B232D] text-xs font-mono">
              {filtered.map((tx) => (
                <tr
                  key={tx.id + tx.timestamp}
                  className={`hover:bg-[#161F2A] transition-colors ${
                    tx.isNew ? "bg-blue-500/5 animate-pulse" : ""
                  }`}
                >
                  <td className="py-2.5 px-3 text-[#8995A3] whitespace-nowrap">{tx.timestamp}</td>
                  <td className="py-2.5 px-3 font-semibold text-[#E8EDF3]">{tx.id}</td>
                  <td className="py-2.5 px-3 text-blue-400">{tx.buyer}</td>
                  <td className="py-2.5 px-3 text-[#A2AEBD]">{tx.seller}</td>
                  <td className="py-2.5 px-3 text-[#E8EDF3] font-sans font-medium">
                    ${tx.amount.toFixed(2)}
                  </td>
                  <td className="py-2.5 px-3">
                    <span
                      className={`font-bold ${
                        tx.risk >= 0.7
                          ? "text-rose-400"
                          : tx.risk >= 0.4
                          ? "text-amber-400"
                          : "text-emerald-400"
                      }`}
                    >
                      {(tx.risk * 100).toFixed(1)}%
                    </span>
                  </td>
                  <td className="py-2.5 px-3">
                    <span
                      className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-bold border ${
                        tx.decision === "BLOCK"
                          ? "bg-rose-500/10 border-rose-500/30 text-rose-400"
                          : tx.decision === "HOLD"
                          ? "bg-amber-500/10 border-amber-500/30 text-amber-400"
                          : tx.decision === "REVIEW"
                          ? "bg-blue-500/10 border-blue-500/30 text-blue-400"
                          : "bg-emerald-500/10 border-emerald-500/30 text-emerald-400"
                      }`}
                    >
                      {tx.decision === "BLOCK" && <XCircle className="w-3 h-3" />}
                      {tx.decision === "HOLD" && <AlertTriangle className="w-3 h-3" />}
                      {tx.decision === "REVIEW" && <Activity className="w-3 h-3" />}
                      {tx.decision === "ALLOW" && <CheckCircle2 className="w-3 h-3" />}
                      {tx.decision}
                    </span>
                  </td>
                  <td className="py-2.5 px-3 text-[11px] text-[#8995A3] max-w-[200px] truncate">
                    {tx.reason}
                  </td>
                  <td className="py-2.5 px-3 text-right">
                    <Link
                      href={`/transactions?preset=${tx.id}`}
                      className="inline-flex items-center gap-1 text-[11px] text-blue-400 hover:text-blue-300 font-sans font-medium"
                    >
                      <span>Analyze</span>
                      <ArrowUpRight className="w-3 h-3" />
                    </Link>
                  </td>
                </tr>
              ))}
              {filtered.length === 0 && (
                <tr>
                  <td colSpan={9} className="py-8 text-center text-[#596574] font-sans text-xs">
                    No transactions match the active search or decision filter.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
