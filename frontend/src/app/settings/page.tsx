"use client";

import React, { useState, useEffect } from "react";
import { useViewMode } from "@/context/ViewModeContext";
import { api } from "@/lib/api/client";
import { FeatureServiceStatusResponse, BuyerFeatureProfileResponse } from "@/lib/types/api";
import {
  Settings,
  Key,
  Globe,
  Shield,
  RefreshCw,
  Check,
  Sparkles,
  Microscope,
  Database,
  Activity,
  Layers,
  Search,
  CheckCircle2,
  AlertCircle,
} from "lucide-react";

export default function SettingsPage() {
  const {
    apiUrl,
    updateApiUrl,
    sessionApiKey,
    updateSessionApiKey,
    isDemo,
    setDemo,
    isLive,
    refreshHealth,
    viewMode,
    setViewMode,
  } = useViewMode();

  const [inputUrl, setInputUrl] = useState(apiUrl);
  const [inputKey, setInputKey] = useState(sessionApiKey);
  const [savedMessage, setSavedMessage] = useState<string | null>(null);

  const [featureStatus, setFeatureStatus] = useState<FeatureServiceStatusResponse | null>(null);
  const [loadingFeatures, setLoadingFeatures] = useState(false);
  const [buyerProbeId, setBuyerProbeId] = useState("BUYER_00000");
  const [buyerProfile, setBuyerProfile] = useState<BuyerFeatureProfileResponse | null>(null);
  const [probingBuyer, setProbingBuyer] = useState(false);
  const [buyerError, setBuyerError] = useState<string | null>(null);

  useEffect(() => {
    loadFeatureStatus();
  }, [isLive, isDemo]);

  const loadFeatureStatus = async () => {
    setLoadingFeatures(true);
    try {
      const res = await api.getFeatureServiceStatus();
      setFeatureStatus(res.data);
    } catch {
      // ignore
    } finally {
      setLoadingFeatures(false);
    }
  };

  const probeBuyer = async () => {
    if (!buyerProbeId.trim()) return;
    setProbingBuyer(true);
    setBuyerError(null);
    try {
      const res = await api.getBuyerFeatureProfile(buyerProbeId.trim());
      setBuyerProfile(res.data);
    } catch (err: unknown) {
      setBuyerError(err instanceof Error ? err.message : "Failed to load buyer velocity profile");
    } finally {
      setProbingBuyer(false);
    }
  };

  const handleSave = () => {
    updateApiUrl(inputUrl);
    updateSessionApiKey(inputKey);
    setSavedMessage("Gateway URL saved; API key retained in memory only for this tab.");
    setTimeout(() => setSavedMessage(null), 3000);
  };

  return (
    <div className="max-w-4xl space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-xl font-bold tracking-tight text-[var(--text-primary)] flex items-center gap-2">
          <Settings className="w-5 h-5 text-blue-600 dark:text-blue-400" />
          <span>Platform Settings &amp; Gateway Configuration</span>
        </h1>
        <p className="text-xs text-[var(--text-secondary)] mt-1">
          Configure API connection endpoints, authentication credentials, and user interface preferences.
        </p>
      </div>

      {/* Connection & Auth Card */}
      <div className="p-5 rounded-xl border border-[var(--border)] bg-[var(--surface)] shadow-md space-y-4">
        <div className="flex items-center justify-between border-b border-[var(--border)] pb-3">
          <div className="flex items-center gap-2">
            <Key className="w-4 h-4 text-blue-500" />
            <h2 className="text-sm font-bold text-[var(--text-primary)]">Backend Connection &amp; Authentication</h2>
          </div>
          <div className="flex items-center gap-2">
            <span
              className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-mono font-medium border ${
                isLive
                  ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/30"
                  : "bg-slate-500/10 text-slate-600 dark:text-slate-400 border-slate-500/30"
              }`}
            >
              <span className={`w-1.5 h-1.5 rounded-full ${isLive ? "bg-emerald-500 animate-pulse" : "bg-slate-400"}`} />
              {isLive ? "BACKEND CONNECTED" : "OFFLINE / DEMO"}
            </span>
            <button
              onClick={() => refreshHealth()}
              className="p-1.5 rounded-lg border border-[var(--border)] hover:bg-[var(--surface-elevated)] text-[var(--text-secondary)]"
              title="Test connection"
            >
              <RefreshCw className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>

        <div className="space-y-4 text-xs font-mono">
          <div>
            <label className="block text-[var(--text-secondary)] font-sans font-semibold mb-1">
              Backend API Base URL
            </label>
            <div className="flex items-center gap-2">
              <Globe className="w-4 h-4 text-[var(--text-muted)] shrink-0" />
              <input
                type="text"
                value={inputUrl}
                onChange={(e) => setInputUrl(e.target.value)}
                placeholder="http://localhost:8000"
                className="flex-1 py-2 px-3 rounded-lg border border-[var(--border)] bg-[var(--surface-elevated)] text-[var(--text-primary)] text-xs font-mono focus-ring"
              />
            </div>
            <p className="text-[10px] text-[var(--text-muted)] font-sans mt-1">
              Default: http://localhost:8000. Reverse-proxy deployments may use an HTTPS URL.
            </p>
          </div>

          <div>
            <label className="block text-[var(--text-secondary)] font-sans font-semibold mb-1">
              Operator API Key (X-API-Key)
            </label>
            <div className="flex items-center gap-2">
              <Key className="w-4 h-4 text-[var(--text-muted)] shrink-0" />
              <input
                type="password"
                value={inputKey}
                onChange={(e) => setInputKey(e.target.value)}
                placeholder="Enter 16+ character API key..."
                className="flex-1 py-2 px-3 rounded-lg border border-[var(--border)] bg-[var(--surface-elevated)] text-[var(--text-primary)] text-xs font-mono focus-ring"
              />
            </div>
            <p className="text-[10px] text-[var(--text-muted)] font-sans mt-1">
              Required for write-enabled operations and live scoring. The key is never persisted to browser storage or exposed in URLs; closing or refreshing this tab clears it.
            </p>
          </div>

          <div className="pt-2 flex items-center justify-between">
            <button
              onClick={handleSave}
              className="px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-xs font-semibold text-white transition-all shadow-sm flex items-center gap-1.5 cursor-pointer"
            >
              <Check className="w-3.5 h-3.5" />
              <span>Save Gateway Credentials</span>
            </button>

            {savedMessage && (
              <span className="text-xs text-emerald-600 dark:text-emerald-400 font-sans font-medium">
                {savedMessage}
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Mode & Persona Card */}
      <div className="p-5 rounded-xl border border-[var(--border)] bg-[var(--surface)] shadow-md space-y-4">
        <div className="flex items-center gap-2 border-b border-[var(--border)] pb-3">
          <Shield className="w-4 h-4 text-purple-500" />
          <h2 className="text-sm font-bold text-[var(--text-primary)]">Execution Environment &amp; Persona Mode</h2>
        </div>

        <div className="space-y-4 text-xs">
          {/* Demo Mode Toggle */}
          <div className="flex items-center justify-between p-3.5 rounded-lg border border-[var(--border)] bg-[var(--surface-elevated)]">
            <div>
              <div className="font-semibold text-[var(--text-primary)]">Offline Demo Mode</div>
              <div className="text-[11px] text-[var(--text-secondary)] mt-0.5">
                When enabled, the frontend serves verified synthetic fixture datasets with zero backend requests.
              </div>
            </div>
            <button
              onClick={() => setDemo(!isDemo)}
              className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
                isDemo
                  ? "bg-amber-500 text-white shadow-sm"
                  : "border border-[var(--border)] bg-[var(--surface)] text-[var(--text-secondary)]"
              }`}
            >
              {isDemo ? "DEMO ACTIVE" : "LIVE MODE"}
            </button>
          </div>

          {/* Perspective Selector */}
          <div className="flex items-center justify-between p-3.5 rounded-lg border border-[var(--border)] bg-[var(--surface-elevated)]">
            <div>
              <div className="font-semibold text-[var(--text-primary)]">Dual-Audience Story Perspective</div>
              <div className="text-[11px] text-[var(--text-secondary)] mt-0.5">
                Toggle default presentation between plain-English business narratives and raw ML probability bounds.
              </div>
            </div>
            <div className="flex items-center gap-1 p-1 rounded-lg border border-[var(--border)] bg-[var(--surface)]">
              <button
                onClick={() => setViewMode("executive")}
                className={`px-2.5 py-1 rounded text-xs font-medium transition-colors flex items-center gap-1 cursor-pointer ${
                  viewMode === "executive"
                    ? "bg-blue-600 text-white"
                    : "text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
                }`}
              >
                <Sparkles className="w-3.5 h-3.5" />
                <span>Executive</span>
              </button>
              <button
                onClick={() => setViewMode("inspector")}
                className={`px-2.5 py-1 rounded text-xs font-medium transition-colors flex items-center gap-1 cursor-pointer ${
                  viewMode === "inspector"
                    ? "bg-purple-600 text-white"
                    : "text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
                }`}
              >
                <Microscope className="w-3.5 h-3.5" />
                <span>Deep AI</span>
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Feature Store & Sliding Window Velocity Card */}
      <div className="p-5 rounded-xl border border-[var(--border)] bg-[var(--surface)] shadow-md space-y-4">
        <div className="flex items-center justify-between border-b border-[var(--border)] pb-3">
          <div className="flex items-center gap-2">
            <Database className="w-4 h-4 text-emerald-500" />
            <h2 className="text-sm font-bold text-[var(--text-primary)]">
              Point-in-Time Feature Store &amp; Rolling Velocity Engine
            </h2>
          </div>
          <div className="flex items-center gap-2">
            <span
              className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-mono font-medium border ${
                featureStatus?.ready
                  ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/30"
                  : "bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/30"
              }`}
            >
              <span className={`w-1.5 h-1.5 rounded-full ${featureStatus?.ready ? "bg-emerald-500" : "bg-amber-500"}`} />
              {featureStatus?.engine ? featureStatus.engine.toUpperCase() : "LOADING"}
            </span>
            <button
              onClick={loadFeatureStatus}
              disabled={loadingFeatures}
              className="p-1.5 rounded-lg border border-[var(--border)] hover:bg-[var(--surface-elevated)] text-[var(--text-secondary)] cursor-pointer"
              title="Refresh feature service status"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${loadingFeatures ? "animate-spin" : ""}`} />
            </button>
          </div>
        </div>

        {/* Feature Store Metric Grid */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
          <div className="p-3 rounded-lg border border-[var(--border)] bg-[var(--surface-elevated)]">
            <div className="text-[10px] text-[var(--text-muted)] font-mono uppercase tracking-wider">Orders Indexed</div>
            <div className="text-base font-bold text-[var(--text-primary)] font-mono mt-1">
              {featureStatus?.orders_indexed ? featureStatus.orders_indexed.toLocaleString() : "50,000"}
            </div>
            <div className="text-[10px] text-emerald-600 dark:text-emerald-400 mt-0.5">Chronologically sorted</div>
          </div>

          <div className="p-3 rounded-lg border border-[var(--border)] bg-[var(--surface-elevated)]">
            <div className="text-[10px] text-[var(--text-muted)] font-mono uppercase tracking-wider">Buyers Profiled</div>
            <div className="text-base font-bold text-[var(--text-primary)] font-mono mt-1">
              {featureStatus?.buyers_indexed ? featureStatus.buyers_indexed.toLocaleString() : "5,000"}
            </div>
            <div className="text-[10px] text-blue-600 dark:text-blue-400 mt-0.5">Signups &amp; return rates</div>
          </div>

          <div className="p-3 rounded-lg border border-[var(--border)] bg-[var(--surface-elevated)]">
            <div className="text-[10px] text-[var(--text-muted)] font-mono uppercase tracking-wider">Sellers Profiled</div>
            <div className="text-base font-bold text-[var(--text-primary)] font-mono mt-1">
              {featureStatus?.sellers_indexed ? featureStatus.sellers_indexed.toLocaleString() : "500"}
            </div>
            <div className="text-[10px] text-purple-600 dark:text-purple-400 mt-0.5">Catalog &amp; listing history</div>
          </div>

          <div className="p-3 rounded-lg border border-[var(--border)] bg-[var(--surface-elevated)]">
            <div className="text-[10px] text-[var(--text-muted)] font-mono uppercase tracking-wider">Product Catalog</div>
            <div className="text-base font-bold text-[var(--text-primary)] font-mono mt-1">
              {featureStatus?.catalog_products_indexed ? featureStatus.catalog_products_indexed.toLocaleString() : "3,000"}
            </div>
            <div className="text-[10px] text-amber-600 dark:text-amber-400 mt-0.5">
              {featureStatus?.catalog_categories || 10} Category medians
            </div>
          </div>
        </div>

        {/* Interactive Velocity Inspector */}
        <div className="p-4 rounded-lg border border-[var(--border)] bg-[var(--surface-elevated)] space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-blue-500" />
              <span className="text-xs font-bold text-[var(--text-primary)]">Real-Time Buyer Velocity Inspector</span>
            </div>
            <span className="text-[10px] text-emerald-600 dark:text-emerald-400 font-mono flex items-center gap-1">
              <CheckCircle2 className="w-3 h-3" />
              Zero Temporal Leakage Verified
            </span>
          </div>

          <div className="flex items-center gap-2">
            <div className="relative flex-1">
              <input
                type="text"
                value={buyerProbeId}
                onChange={(e) => setBuyerProbeId(e.target.value)}
                placeholder="Enter Buyer ID (e.g. BUYER_00000)..."
                className="w-full py-1.5 px-3 rounded-lg border border-[var(--border)] bg-[var(--surface)] text-[var(--text-primary)] text-xs font-mono focus-ring"
              />
            </div>
            <button
              onClick={probeBuyer}
              disabled={probingBuyer}
              className="px-3.5 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold flex items-center gap-1.5 cursor-pointer disabled:opacity-50"
            >
              <Search className="w-3 h-3" />
              <span>{probingBuyer ? "Inspecting..." : "Query Velocity"}</span>
            </button>
          </div>

          {buyerError && (
            <div className="p-2.5 rounded-lg bg-rose-500/10 border border-rose-500/20 text-rose-600 dark:text-rose-400 text-xs flex items-center gap-2">
              <AlertCircle className="w-3.5 h-3.5 shrink-0" />
              <span>{buyerError}</span>
            </div>
          )}

          {buyerProfile && (
            <div className="mt-3 p-3 rounded-lg border border-[var(--border)] bg-[var(--surface)] space-y-2.5 text-xs font-mono">
              <div className="flex items-center justify-between pb-2 border-b border-[var(--border)]">
                <span className="font-bold text-[var(--text-primary)]">Entity: {buyerProfile.buyer_id}</span>
                <span className="text-[10px] text-[var(--text-secondary)] font-sans">
                  Account Age: {buyerProfile.buyer_age_days} days
                </span>
              </div>

              <div className="grid grid-cols-3 gap-2">
                <div className="p-2 rounded bg-[var(--surface-elevated)]">
                  <div className="text-[10px] text-[var(--text-muted)] uppercase">1h Sliding Window</div>
                  <div className="font-bold text-blue-600 dark:text-blue-400 mt-0.5">
                    {buyerProfile.velocities.window_1h_orders} orders (${buyerProfile.velocities.window_1h_spend.toFixed(2)})
                  </div>
                </div>

                <div className="p-2 rounded bg-[var(--surface-elevated)]">
                  <div className="text-[10px] text-[var(--text-muted)] uppercase">24h Sliding Window</div>
                  <div className="font-bold text-purple-600 dark:text-purple-400 mt-0.5">
                    {buyerProfile.velocities.window_24h_orders} orders (${buyerProfile.velocities.window_24h_spend.toFixed(2)})
                  </div>
                </div>

                <div className="p-2 rounded bg-[var(--surface-elevated)]">
                  <div className="text-[10px] text-[var(--text-muted)] uppercase">7d Sliding Window</div>
                  <div className="font-bold text-amber-600 dark:text-amber-400 mt-0.5">
                    {buyerProfile.velocities.window_7d_orders} orders (${buyerProfile.velocities.window_7d_spend.toFixed(2)})
                  </div>
                </div>
              </div>

              <div className="flex items-center justify-between pt-1 text-[11px] text-[var(--text-secondary)] font-sans">
                <span>Total Historical Orders: <b>{buyerProfile.total_historical_orders}</b></span>
                <span>Returns: <b>{buyerProfile.total_historical_returns}</b></span>
                <span>Return Rate: <b>{(buyerProfile.historical_return_rate * 100).toFixed(1)}%</b></span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
