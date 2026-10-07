"use client";

import React from "react";
import { useViewMode } from "@/context/ViewModeContext";
import { Search, Sparkles, Microscope, CheckCircle2, AlertTriangle, RefreshCw } from "lucide-react";

export function Header() {
  const { viewMode, setViewMode, isLive, apiUrl, refreshHealth } = useViewMode();

  return (
    <header className="h-16 border-b border-[#202A35] bg-[#0E131A] px-6 flex items-center justify-between sticky top-0 z-30 select-none">
      {/* Left Status Indicators */}
      <div className="flex items-center gap-4">
        {/* Backend Connectivity Status */}
        <div className="flex items-center gap-2 px-2.5 py-1 rounded-md bg-[#151D27] border border-[#202A35] text-xs">
          {isLive ? (
            <>
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              <span className="text-emerald-400 font-medium">Gateway Live</span>
            </>
          ) : (
            <>
              <span className="w-2 h-2 rounded-full bg-amber-500" />
              <span className="text-amber-400 font-medium">Offline Simulation</span>
            </>
          )}
          <span className="text-[#596574]">|</span>
          <span className="font-mono text-[11px] text-[#8995A3] truncate max-w-[140px]">{apiUrl}</span>
          <button
            onClick={() => refreshHealth()}
            title="Refresh API Connection"
            className="text-[#596574] hover:text-[#E8EDF3] transition-colors ml-1"
          >
            <RefreshCw className="w-3 h-3" />
          </button>
        </div>

        {/* Model Spec Badge */}
        <div className="hidden md:flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-[#151D27] border border-[#202A35] text-xs">
          <span className="text-[#8995A3]">Engine:</span>
          <span className="font-mono text-blue-400 font-semibold">Phase 5 Hybrid (GNN + XGBoost)</span>
        </div>
      </div>

      {/* Right Controls */}
      <div className="flex items-center gap-3">
        {/* Search Modal Trigger */}
        <button
          onClick={() => {
            const event = new KeyboardEvent("keydown", { key: "k", metaKey: true });
            window.dispatchEvent(event);
          }}
          className="flex items-center gap-2 px-3 py-1.5 rounded-md bg-[#111821] hover:bg-[#151D27] border border-[#202A35] text-xs text-[#8995A3] transition-all"
        >
          <Search className="w-3.5 h-3.5 text-[#596574]" />
          <span>Quick Find...</span>
          <kbd className="font-mono text-[10px] bg-[#0A0E13] px-1.5 py-0.5 rounded border border-[#202A35] text-[#8995A3]">
            ⌘K
          </kbd>
        </button>

        {/* DUAL-AUDIENCE GLOBAL VIEW TOGGLE */}
        <div className="flex items-center p-0.5 bg-[#0A0E13] border border-[#202A35] rounded-lg">
          <button
            onClick={() => setViewMode("executive")}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
              viewMode === "executive"
                ? "bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-sm font-semibold"
                : "text-[#8995A3] hover:text-[#E8EDF3]"
            }`}
          >
            <Sparkles className="w-3.5 h-3.5" />
            <span>Story View</span>
          </button>
          <button
            onClick={() => setViewMode("inspector")}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
              viewMode === "inspector"
                ? "bg-gradient-to-r from-purple-600 to-indigo-600 text-white shadow-sm font-semibold"
                : "text-[#8995A3] hover:text-[#E8EDF3]"
            }`}
          >
            <Microscope className="w-3.5 h-3.5" />
            <span>AI Inspector</span>
          </button>
        </div>
      </div>
    </header>
  );
}
