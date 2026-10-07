"use client";

import React, { useState } from "react";
import Link from "next/link";
import { Activity, Search, Filter, ArrowUpRight, CheckCircle2, AlertTriangle, XCircle, HelpCircle } from "lucide-react";

interface FeedItem {
  id: string;
  timestamp: string;
  buyer: string;
  seller: string;
  amount: number;
  risk: number;
  decision: "ALLOW" | "REVIEW" | "HOLD" | "BLOCK";
  reason: string;
}

const FEED_DATA: FeedItem[] = [
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

  const filtered = FEED_DATA.filter((tx) => {
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
          <h1 className="text-xl font-bold tracking-tight text-[#E8EDF3] flex items-center gap-2">
            <Activity className="w-5 h-5 text-blue-400" />
            <span>Transaction Live Stream & Audit Feed</span>
          </h1>
          <p className="text-xs text-[#8995A3] mt-1">
            Real-time chronological ingest log capturing feature states, risk calibration, and decision dispatch.
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
            <tbody className="divide-y divide-[#151D27] text-xs font-mono">
              {filtered.map((tx) => (
                <tr key={tx.id} className="hover:bg-[#151D27]/50 transition-colors">
                  <td className="py-3 px-3 text-[#596574]">{tx.timestamp}</td>
                  <td className="py-3 px-3 text-blue-400 font-semibold">{tx.id}</td>
                  <td className="py-3 px-3 text-[#E8EDF3]">{tx.buyer}</td>
                  <td className="py-3 px-3 text-[#8995A3]">{tx.seller}</td>
                  <td className="py-3 px-3 text-[#E8EDF3]">${tx.amount.toFixed(2)}</td>
                  <td className="py-3 px-3">
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
                      className={`text-[9px] px-2 py-0.5 rounded font-bold border ${
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
                  <td className="py-3 px-3 text-[#8995A3]">{tx.reason}</td>
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
                      className="text-blue-400 hover:underline"
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
