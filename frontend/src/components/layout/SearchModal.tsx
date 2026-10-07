"use client";

import React, { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { useViewMode } from "@/context/ViewModeContext";
import { Search, X, Zap, Network, Tag, RotateCcw, FileSearch, ArrowRight } from "lucide-react";

interface SearchResult {
  id: string;
  type: "TRANSACTION" | "BUYER" | "SELLER" | "DEVICE" | "RING" | "LISTING" | "NAV";
  title: string;
  subtitle: string;
  link: string;
}

const SEARCH_DATABASE: SearchResult[] = [
  // Presets & Transactions
  { id: "ORD_SIM_SAFE_01", type: "TRANSACTION", title: "ORD_SIM_SAFE_01", subtitle: "Verified Repeat Buyer ($65.50) · ALLOW", link: "/transactions?preset=preset_normal_buyer" },
  { id: "ORD_SIM_ANOMALY_02", type: "TRANSACTION", title: "ORD_SIM_ANOMALY_02", subtitle: "New Account Price Anomaly ($450.00) · REVIEW", link: "/transactions?preset=preset_price_arbitrage" },
  { id: "ORD_SIM_FARM_03", type: "TRANSACTION", title: "ORD_SIM_FARM_03", subtitle: "Device Farm Collusion Ring ($890.00) · BLOCK", link: "/transactions?preset=preset_device_farm" },
  { id: "ORD_SIM_RETURN_04", type: "TRANSACTION", title: "ORD_SIM_RETURN_04", subtitle: "Serial Return Abuse ($320.00) · HOLD", link: "/transactions?preset=preset_serial_returner" },
  // Entities
  { id: "BUYER_VERIFIED_77", type: "BUYER", title: "BUYER_VERIFIED_77", subtitle: "180 days active, 15 orders, 0 returns", link: "/trust-graph?search=BUYER_VERIFIED_77" },
  { id: "BUYER_RING_MEMBER_04", type: "BUYER", title: "BUYER_RING_MEMBER_04", subtitle: "Collusion Cluster Member · Risk: 0.95", link: "/trust-graph?search=BUYER_RING_MEMBER_04" },
  { id: "DEV_FARM_99", type: "DEVICE", title: "DEV_FARM_99", subtitle: "Shared across 9 distinct buyer accounts", link: "/trust-graph?search=DEV_FARM_99" },
  { id: "RING_COLLUSION_01", type: "RING", title: "RING_COLLUSION_01", subtitle: "8 members, 42 orders · Avg Risk: 0.912", link: "/fraud-rings" },
  { id: "RING_DEVICE_FARM_04", type: "RING", title: "RING_DEVICE_FARM_04", subtitle: "11 members, 63 orders · Avg Risk: 0.730", link: "/fraud-rings" },
  { id: "LISTING_DEMO_01", type: "LISTING", title: "LISTING_DEMO_01", subtitle: "Flagged Cross-Seller Image Reuse · Similarity: 0.94", link: "/listings" },
  // Nav
  { id: "NAV_CMD", type: "NAV", title: "Command Center Dashboard", subtitle: "System Health & Real-time Metrics", link: "/dashboard" },
  { id: "NAV_ANALYZER", type: "NAV", title: "Transaction Risk Analyzer", subtitle: "Hero scoring engine with scenario simulator", link: "/transactions" },
  { id: "NAV_MODELS", type: "NAV", title: "Model Registry & Calibration", subtitle: "Phase 5 Hybrid, Reliability curve, ECE", link: "/models" },
];

export function SearchModal() {
  const { searchOpen, setSearchOpen } = useViewMode();
  const [query, setQuery] = useState("");
  const router = useRouter();

  useEffect(() => {
    if (!searchOpen) setQuery("");
  }, [searchOpen]);

  if (!searchOpen) return null;

  const filtered = query.trim() === ""
    ? SEARCH_DATABASE.slice(0, 6)
    : SEARCH_DATABASE.filter(
        (item) =>
          item.title.toLowerCase().includes(query.toLowerCase()) ||
          item.subtitle.toLowerCase().includes(query.toLowerCase()) ||
          item.type.toLowerCase().includes(query.toLowerCase())
      );

  const handleSelect = (link: string) => {
    setSearchOpen(false);
    router.push(link);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-20 bg-black/75 backdrop-blur-sm animate-in fade-in duration-150">
      <div className="w-full max-w-xl bg-[#111821] border border-[#202A35] rounded-xl shadow-2xl overflow-hidden">
        {/* Search Input Bar */}
        <div className="flex items-center px-4 border-b border-[#202A35] bg-[#0E131A]">
          <Search className="w-4 h-4 text-[#8995A3] mr-3" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search transactions, buyers, sellers, devices, rings, or pages..."
            className="w-full h-12 bg-transparent text-sm text-[#E8EDF3] placeholder-[#596574] focus:outline-none font-mono"
            autoFocus
          />
          <button onClick={() => setSearchOpen(false)} className="text-[#596574] hover:text-[#E8EDF3]">
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Results List */}
        <div className="max-h-80 overflow-y-auto p-2 divide-y divide-[#151D27]/50">
          {filtered.length === 0 ? (
            <div className="py-8 text-center text-xs text-[#596574]">No matching records found.</div>
          ) : (
            filtered.map((item) => (
              <div
                key={item.id}
                onClick={() => handleSelect(item.link)}
                className="flex items-center justify-between p-2.5 rounded-lg hover:bg-[#151D27] cursor-pointer transition-colors group"
              >
                <div className="flex items-center gap-3">
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-[#0A0E13] text-[#8995A3] border border-[#202A35]">
                    {item.type}
                  </span>
                  <div>
                    <div className="text-xs font-semibold text-[#E8EDF3] group-hover:text-blue-400 font-mono">
                      {item.title}
                    </div>
                    <div className="text-[11px] text-[#8995A3]">{item.subtitle}</div>
                  </div>
                </div>
                <ArrowRight className="w-3.5 h-3.5 text-[#596574] group-hover:text-blue-400 transition-transform group-hover:translate-x-0.5" />
              </div>
            ))
          )}
        </div>

        {/* Footer */}
        <div className="px-4 py-2 bg-[#0A0E13] border-t border-[#202A35] flex items-center justify-between text-[11px] text-[#596574]">
          <span>Type to filter live database</span>
          <span className="font-mono">ESC to close</span>
        </div>
      </div>
    </div>
  );
}
