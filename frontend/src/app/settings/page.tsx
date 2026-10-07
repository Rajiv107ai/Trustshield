"use client";

import React, { useState } from "react";
import { useViewMode } from "@/context/ViewModeContext";
import {
  Settings,
  Server,
  Cpu,
  CheckCircle2,
  Save,
  RefreshCw,
  Sliders,
  Sparkles,
} from "lucide-react";

export default function SettingsPage() {
  const { apiUrl, updateApiUrl, isLive, refreshHealth, viewMode, setViewMode } = useViewMode();
  const [inputUrl, setInputUrl] = useState(apiUrl);
  const [modelVersion, setModelVersion] = useState("phase5-hybrid");
  const [savedSuccess, setSavedSuccess] = useState(false);

  const handleSave = (e: React.FormEvent) => {
    e.preventDefault();
    updateApiUrl(inputUrl);
    setSavedSuccess(true);
    setTimeout(() => setSavedSuccess(false), 3000);
  };

  return (
    <div className="space-y-6 max-w-3xl">
      {/* Header */}
      <div>
        <h1 className="text-xl font-bold tracking-tight text-[#E8EDF3] flex items-center gap-2">
          <Settings className="w-5 h-5 text-[#8995A3]" />
          <span>Platform Settings & Gateway Configuration</span>
        </h1>
        <p className="text-xs text-[#8995A3] mt-1">
          Configure backend API URLs, active machine learning model versions, and console display parameters.
        </p>
      </div>

      {/* API Gateway Configuration */}
      <div className="p-5 rounded-xl border border-[#202A35] bg-[#111821] space-y-4">
        <h3 className="text-xs font-bold uppercase tracking-wider text-[#E8EDF3] flex items-center gap-2">
          <Server className="w-4 h-4 text-blue-400" />
          <span>FastAPI Serving Layer Connectivity</span>
        </h3>

        <form onSubmit={handleSave} className="space-y-3 text-xs font-mono">
          <div>
            <label className="text-[11px] text-[#8995A3] block mb-1">Backend Base API URL</label>
            <div className="flex gap-2">
              <input
                type="text"
                value={inputUrl}
                onChange={(e) => setInputUrl(e.target.value)}
                placeholder="http://localhost:8000"
                className="flex-1 bg-[#0E131A] border border-[#202A35] rounded-md px-3 py-2 text-[#E8EDF3] focus:border-blue-500 focus:outline-none"
              />
              <button
                type="submit"
                className="px-4 py-2 rounded-md bg-blue-600 hover:bg-blue-500 font-bold text-white transition-colors flex items-center gap-2"
              >
                <Save className="w-3.5 h-3.5" />
                <span>Save</span>
              </button>
            </div>
            <p className="text-[11px] text-[#596574] mt-1">
              Current active connection: <span className="text-[#8995A3]">{apiUrl}</span> (
              {isLive ? <span className="text-emerald-400 font-semibold">Live</span> : <span className="text-amber-400 font-semibold">Offline Simulation</span>}
              )
            </p>
          </div>

          {savedSuccess && (
            <div className="p-3 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-xs text-emerald-300 font-mono flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4" />
              <span>Gateway URL saved and connectivity refreshed.</span>
            </div>
          )}
        </form>
      </div>

      {/* Model Deployment Selection */}
      <div className="p-5 rounded-xl border border-[#202A35] bg-[#111821] space-y-4">
        <h3 className="text-xs font-bold uppercase tracking-wider text-[#E8EDF3] flex items-center gap-2">
          <Cpu className="w-4 h-4 text-purple-400" />
          <span>Model Registry Deployment Selection</span>
        </h3>

        <div className="space-y-2 text-xs font-mono">
          {[
            {
              id: "phase5-hybrid",
              title: "Phase 5 Hybrid XGBoost + GNN (Recommended)",
              desc: "Combines 16D PyG HeteroData embeddings with XGBoost tabular features and Isotonic calibration.",
            },
            {
              id: "phase3-rf",
              title: "Phase 3 Tabular + Graph Random Forest",
              desc: "Fallback model utilizing NetworkX degree centrality without deep relational message passing.",
            },
          ].map((m) => (
            <div
              key={m.id}
              onClick={() => setModelVersion(m.id)}
              className={`p-3 rounded-lg border cursor-pointer transition-all ${
                modelVersion === m.id
                  ? "bg-[#151D27] border-purple-500 ring-1 ring-purple-500/30"
                  : "bg-[#0E131A] border-[#202A35] hover:bg-[#151D27]"
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="font-bold text-[#E8EDF3]">{m.title}</span>
                {modelVersion === m.id && <span className="text-emerald-400 text-[10px] font-bold">ACTIVE</span>}
              </div>
              <p className="text-[11px] text-[#8995A3] mt-1 font-sans">{m.desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Default Global View Mode */}
      <div className="p-5 rounded-xl border border-[#202A35] bg-[#111821] space-y-4">
        <h3 className="text-xs font-bold uppercase tracking-wider text-[#E8EDF3] flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-amber-400" />
          <span>Default Console Persona / View Mode</span>
        </h3>

        <div className="grid grid-cols-2 gap-3 text-xs">
          <button
            onClick={() => setViewMode("executive")}
            className={`p-3 rounded-lg border text-left transition-all ${
              viewMode === "executive"
                ? "bg-blue-600/10 border-blue-500 text-blue-300 font-bold"
                : "bg-[#0E131A] border-[#202A35] text-[#8995A3] hover:text-[#E8EDF3]"
            }`}
          >
            <div className="font-semibold text-[#E8EDF3]">Executive Story View</div>
            <div className="text-[11px] text-[#8995A3] mt-1 font-normal">
              Plain English narratives, high-level business impact, and rapid decision buttons.
            </div>
          </button>

          <button
            onClick={() => setViewMode("inspector")}
            className={`p-3 rounded-lg border text-left transition-all ${
              viewMode === "inspector"
                ? "bg-purple-600/10 border-purple-500 text-purple-300 font-bold"
                : "bg-[#0E131A] border-[#202A35] text-[#8995A3] hover:text-[#E8EDF3]"
            }`}
          >
            <div className="font-semibold text-[#E8EDF3]">Deep AI Inspector</div>
            <div className="text-[11px] text-[#8995A3] mt-1 font-normal">
              Raw model probabilities, GNN embeddings, Shannon entropy, and conformal prediction coverage sets.
            </div>
          </button>
        </div>
      </div>
    </div>
  );
}
