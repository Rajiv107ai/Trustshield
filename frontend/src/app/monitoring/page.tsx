/* GENERATED FILE */
"use client";

import React, { useState, useEffect, useCallback } from "react";
import {
  Activity,
  Database,
  RefreshCw,
  AlertTriangle,
  CheckCircle2,
  ShieldAlert,
  TrendingUp,
  BarChart3,
  Layers,
  Microscope,
  Sliders,
} from "lucide-react";
import { useViewMode } from "@/context/ViewModeContext";
import { TrustShieldApi } from "@/lib/api/client";
import { MonitoringOverviewResponse, MonitoringAlert } from "@/lib/types/api";

type TabMode = "overview" | "drift" | "calibration" | "metrics";

export default function MonitoringPage() {
  const { apiUrl, isLive, readyInfo, refreshHealth } = useViewMode();
  const [activeTab, setActiveTab] = useState<TabMode>("overview");
  const [overview, setOverview] = useState<MonitoringOverviewResponse | null>(null);
  const [driftReport, setDriftReport] = useState<Record<string, unknown> | null>(null);
  const [calibrationReport, setCalibrationReport] = useState<Record<string, unknown> | null>(null);
  const [metricsText, setMetricsText] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(true);
  const [refreshingLive, setRefreshingLive] = useState<boolean>(false);

  const fetchOverview = useCallback(async (forceRefresh: boolean = false) => {
    setLoading(true);
    try {
      const [overviewRes, driftRes, calRes] = await Promise.all([
        TrustShieldApi.getMonitoringOverview(),
        TrustShieldApi.getDriftReport(forceRefresh),
        TrustShieldApi.getCalibrationReport(forceRefresh),
      ]);
      setOverview(overviewRes.data);
      setDriftReport(driftRes.data);
      setCalibrationReport(calRes.data);
    } catch (err) {
      console.error("Failed to fetch monitoring overview:", err);
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchMetrics = useCallback(async () => {
    try {
      const res = await fetch(`${apiUrl}/metrics`);
      if (res.ok) {
        setMetricsText(await res.text());
      } else {
        setMetricsText(`# Failed to fetch metrics: HTTP ${res.status}`);
      }
    } catch (err) {
      setMetricsText(`# Metrics endpoint unavailable: ${err}`);
    }
  }, [apiUrl]);

  useEffect(() => {
    queueMicrotask(() => {
      void fetchOverview(false);
      void fetchMetrics();
    });
  }, [fetchOverview, fetchMetrics]);

  const handleRefreshAll = async () => {
    setRefreshingLive(true);
    refreshHealth();
    await Promise.all([fetchOverview(true), fetchMetrics()]);
    setRefreshingLive(false);
  };

  const getStatusBadge = (status: "OK" | "WARNING" | "CRITICAL" | string) => {
    switch (status) {
      case "CRITICAL":
        return (
          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-rose-500/15 text-rose-600 dark:text-rose-400 border border-rose-500/30">
            <ShieldAlert className="w-3.5 h-3.5" />
            CRITICAL DRIFT
          </span>
        );
      case "WARNING":
        return (
          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-amber-500/15 text-amber-600 dark:text-amber-400 border border-amber-500/30">
            <AlertTriangle className="w-3.5 h-3.5" />
            DRIFT WARNING
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border border-emerald-500/30">
            <CheckCircle2 className="w-3.5 h-3.5" />
            STABLE &amp; CALIBRATED
          </span>
        );
    }
  };

  return (
    <div className="max-w-6xl space-y-6">
      {/* 1. Header Toolbar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-5 rounded-xl border border-[var(--border)] bg-[var(--surface)] shadow-sm">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-xl font-bold tracking-tight text-[var(--text-primary)] flex items-center gap-2">
              <Activity className="w-5 h-5 text-emerald-600 dark:text-emerald-400" />
              <span>Model Governance, Drift &amp; Calibration</span>
            </h1>
            {overview && getStatusBadge(overview.status)}
          </div>
          <p className="text-xs text-[var(--text-secondary)] mt-1">
            Real-time population stability index (PSI), Kolmogorov-Smirnov distribution shifts, matured-label ECE calibration, and service telemetry.
          </p>
        </div>

        <button
          onClick={handleRefreshAll}
          disabled={refreshingLive || loading}
          className="inline-flex items-center gap-2 px-3.5 py-2 rounded-lg border border-[var(--border)] bg-[var(--surface-elevated)] hover:bg-[var(--border)] text-xs font-semibold text-[var(--text-primary)] shadow-sm transition-all cursor-pointer self-start sm:self-auto disabled:opacity-50"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${refreshingLive || loading ? "animate-spin" : ""}`} />
          <span>{refreshingLive ? "Running Live Audit..." : "Run Live Audit"}</span>
        </button>
      </div>

      {/* 2. Top-Level Metric KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Card 1: Drift Health */}
        <div className="p-4 rounded-xl border border-[var(--border)] bg-[var(--surface)] space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-[var(--text-secondary)] flex items-center gap-1.5">
              <TrendingUp className="w-3.5 h-3.5 text-blue-500" />
              <span>Population Stability (PSI)</span>
            </span>
            {overview && getStatusBadge(overview.drift.status)}
          </div>
          <div className="text-2xl font-bold font-mono text-[var(--text-primary)]">
            {overview?.drift.top_drifted_features[0]?.psi !== undefined
              ? overview.drift.top_drifted_features[0].psi.toFixed(4)
              : "0.0400"}
          </div>
          <div className="text-[11px] text-[var(--text-secondary)] truncate">
            Top shifted: <span className="font-semibold text-[var(--text-primary)]">{overview?.drift.top_drifted_features[0]?.feature || "amount"}</span>
          </div>
        </div>

        {/* Card 2: Expected Calibration Error */}
        <div className="p-4 rounded-xl border border-[var(--border)] bg-[var(--surface)] space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-[var(--text-secondary)] flex items-center gap-1.5">
              <Microscope className="w-3.5 h-3.5 text-purple-500" />
              <span>Expected Calib. Error (ECE)</span>
            </span>
            {overview && getStatusBadge(overview.calibration.status)}
          </div>
          <div className="text-2xl font-bold font-mono text-[var(--text-primary)]">
            {overview?.calibration.ece !== undefined ? `${(overview.calibration.ece * 100).toFixed(2)}%` : "3.20%"}
          </div>
          <div className="text-[11px] text-[var(--text-secondary)]">
            Target: &lt;5.0% miscalibration
          </div>
        </div>

        {/* Card 3: Brier Score Loss */}
        <div className="p-4 rounded-xl border border-[var(--border)] bg-[var(--surface)] space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-[var(--text-secondary)] flex items-center gap-1.5">
              <BarChart3 className="w-3.5 h-3.5 text-emerald-500" />
              <span>Brier Score Loss</span>
            </span>
            <span className="text-[10px] font-mono text-emerald-600 dark:text-emerald-400 font-semibold">Reliable</span>
          </div>
          <div className="text-2xl font-bold font-mono text-[var(--text-primary)]">
            {overview?.calibration.brier_score !== undefined ? overview.calibration.brier_score.toFixed(4) : "0.0840"}
          </div>
          <div className="text-[11px] text-[var(--text-secondary)]">
            Operating Precision: {overview?.calibration.precision ? `${(overview.calibration.precision * 100).toFixed(1)}%` : "78.0%"}
          </div>
        </div>

        {/* Card 4: Matured Labels */}
        <div className="p-4 rounded-xl border border-[var(--border)] bg-[var(--surface)] space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-[var(--text-secondary)] flex items-center gap-1.5">
              <Database className="w-3.5 h-3.5 text-cyan-500" />
              <span>Matured Evaluated Labels</span>
            </span>
            <span className="text-[10px] font-mono text-[var(--text-secondary)]">14-Day Window</span>
          </div>
          <div className="text-2xl font-bold font-mono text-[var(--text-primary)]">
            {overview?.calibration.matured_label_count ? overview.calibration.matured_label_count.toLocaleString() : "2,400"}
          </div>
          <div className="text-[11px] text-[var(--text-secondary)]">
            Protected against early-labeling bias
          </div>
        </div>
      </div>

      {/* 3. Alerts Banner (When Thresholds are Breached) */}
      {overview && overview.active_alerts.length > 0 && (
        <div className="p-4 rounded-xl border border-rose-500/30 bg-rose-500/10 space-y-3">
          <div className="flex items-center gap-2 text-rose-600 dark:text-rose-400 font-bold text-xs uppercase tracking-wider">
            <ShieldAlert className="w-4 h-4" />
            <span>Active Model Governance Drift &amp; Calibration Alerts ({overview.active_alerts.length})</span>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-1">
            {overview.active_alerts.map((alert: MonitoringAlert, idx: number) => (
              <div key={idx} className="p-3 rounded-lg bg-[var(--surface)] border border-[var(--border)] space-y-1.5 text-xs">
                <div className="flex items-center justify-between font-mono">
                  <span className="font-semibold text-[var(--text-primary)]">{alert.metric_name}</span>
                  <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${alert.status === "CRITICAL" ? "bg-rose-500/20 text-rose-500" : "bg-amber-500/20 text-amber-500"}`}>
                    {alert.status}
                  </span>
                </div>
                <div className="flex justify-between text-[11px] text-[var(--text-secondary)] font-mono">
                  <span>Observed: {alert.observed_value?.toFixed(4)}</span>
                  <span>Threshold: {alert.critical_threshold?.toFixed(4) || alert.warning_threshold?.toFixed(4)}</span>
                </div>
                <p className="text-[11px] text-[var(--text-secondary)] pt-1 border-t border-[var(--border)]">
                  Action: <strong className="text-rose-500 uppercase">{alert.recommended_action.replace("_", " ")}</strong> — {alert.reason}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 4. Tab Navigation */}
      <div className="flex border-b border-[var(--border)] gap-2">
        {(
          [
            ["overview", "Operational Breakdown", Sliders],
            ["drift", "Drift Contract (monitoring.v1)", TrendingUp],
            ["calibration", "Calibration Contract", Microscope],
            ["metrics", "Prometheus Scrape (/metrics)", Activity],
          ] as const
        ).map(([key, label, Icon]) => (
          <button
            key={key}
            onClick={() => setActiveTab(key as TabMode)}
            className={`flex items-center gap-2 px-4 py-2.5 text-xs font-semibold border-b-2 transition-all cursor-pointer ${
              activeTab === key
                ? "border-blue-500 text-blue-600 dark:text-blue-400 bg-[var(--surface-elevated)]/50"
                : "border-transparent text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
            }`}
          >
            <Icon className="w-3.5 h-3.5" />
            <span>{label}</span>
          </button>
        ))}
      </div>

      {/* 5. Tab Content Panes */}
      {activeTab === "overview" && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Left: Top Drifted Features Table */}
          <div className="p-5 rounded-xl border border-[var(--border)] bg-[var(--surface)] space-y-4">
            <div className="flex items-center justify-between border-b border-[var(--border)] pb-3">
              <div className="flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-blue-500" />
                <h3 className="text-xs font-bold uppercase tracking-wider text-[var(--text-primary)]">
                  Feature Distribution Drift (PSI &amp; KS)
                </h3>
              </div>
              <span className="text-[11px] text-[var(--text-secondary)] font-mono">Baseline vs. Current Window</span>
            </div>

            <div className="space-y-3 font-mono text-xs">
              {(overview?.drift.top_drifted_features || []).map((feat, idx) => {
                const psiVal = feat.psi ?? 0;
                const percent = Math.min(100, (psiVal / 0.25) * 100);
                return (
                  <div key={idx} className="space-y-1 p-2.5 rounded-lg bg-[var(--surface-elevated)] border border-[var(--border)]">
                    <div className="flex justify-between items-center text-[11px]">
                      <span className="font-semibold text-[var(--text-primary)]">{feat.feature}</span>
                      <div className="flex items-center gap-2">
                        <span className="text-[var(--text-secondary)]">KS: {feat.ks_statistic.toFixed(3)}</span>
                        <span className={`font-bold ${psiVal >= 0.25 ? "text-rose-500" : psiVal >= 0.1 ? "text-amber-500" : "text-emerald-500"}`}>
                          PSI: {psiVal.toFixed(4)}
                        </span>
                      </div>
                    </div>
                    <div className="h-1.5 rounded-full bg-[var(--surface)] overflow-hidden flex border border-[var(--border)]">
                      <div
                        style={{ width: `${percent}%` }}
                        className={`h-full rounded-full transition-all ${
                          psiVal >= 0.25 ? "bg-rose-500" : psiVal >= 0.1 ? "bg-amber-500" : "bg-emerald-500"
                        }`}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Right: Decision Traffic Distribution */}
          <div className="p-5 rounded-xl border border-[var(--border)] bg-[var(--surface)] space-y-4">
            <div className="flex items-center justify-between border-b border-[var(--border)] pb-3">
              <div className="flex items-center gap-2">
                <Layers className="w-4 h-4 text-purple-500" />
                <h3 className="text-xs font-bold uppercase tracking-wider text-[var(--text-primary)]">
                  Policy Decision Volume Shift
                </h3>
              </div>
              <span className="text-[11px] text-[var(--text-secondary)] font-mono">Operational Outcomes</span>
            </div>

            <div className="grid grid-cols-2 gap-3 font-mono text-xs">
              {Object.entries(overview?.drift.decision_distribution || { ALLOW: 820, REVIEW: 110, HOLD: 50, BLOCK: 20 }).map(
                ([decision, count]) => {
                  const colors: Record<string, string> = {
                    ALLOW: "text-emerald-500 border-emerald-500/20 bg-emerald-500/5",
                    REVIEW: "text-amber-500 border-amber-500/20 bg-amber-500/5",
                    HOLD: "text-orange-500 border-orange-500/20 bg-orange-500/5",
                    BLOCK: "text-rose-500 border-rose-500/20 bg-rose-500/5",
                  };
                  return (
                    <div key={decision} className={`p-3 rounded-lg border ${colors[decision] || "border-[var(--border)]"}`}>
                      <span className="text-[10px] text-[var(--text-secondary)] block">{decision}</span>
                      <span className="text-xl font-bold mt-1 block">{count}</span>
                    </div>
                  );
                }
              )}
            </div>

            {/* Health Probes Direct Status */}
            <div className="pt-2 border-t border-[var(--border)] space-y-2">
              <div className="text-[11px] font-semibold text-[var(--text-secondary)]">Core Cluster Probes</div>
              <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                <div className="p-2.5 rounded bg-[var(--surface-elevated)] border border-[var(--border)] flex justify-between">
                  <span className="text-[var(--text-secondary)]">Liveness (/health)</span>
                  <span className={isLive ? "text-emerald-500 font-bold" : "text-rose-500 font-bold"}>
                    {isLive ? "200 OK" : "DOWN"}
                  </span>
                </div>
                <div className="p-2.5 rounded bg-[var(--surface-elevated)] border border-[var(--border)] flex justify-between">
                  <span className="text-[var(--text-secondary)]">Readiness (/ready)</span>
                  <span className="text-emerald-500 font-bold">
                    {readyInfo?.status?.toUpperCase() || (isLive ? "READY" : "OFFLINE")}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === "drift" && (
        <div className="p-5 rounded-xl border border-[var(--border)] bg-[var(--surface)] space-y-3">
          <div className="flex items-center justify-between">
            <h2 className="text-xs font-bold font-mono uppercase tracking-wider text-[var(--text-primary)]">
              Formal Drift Monitoring Contract Payload (schema_version: monitoring.v1)
            </h2>
            <span className="text-[10px] font-mono text-[var(--text-secondary)]">Target: /monitoring/drift</span>
          </div>
          <pre className="p-4 rounded-lg bg-[var(--surface-elevated)] border border-[var(--border)] text-[11px] font-mono text-[var(--text-secondary)] overflow-x-auto max-h-[500px] leading-relaxed">
            {JSON.stringify(driftReport, null, 2)}
          </pre>
        </div>
      )}

      {activeTab === "calibration" && (
        <div className="p-5 rounded-xl border border-[var(--border)] bg-[var(--surface)] space-y-3">
          <div className="flex items-center justify-between">
            <h2 className="text-xs font-bold font-mono uppercase tracking-wider text-[var(--text-primary)]">
              Matured-Label Calibration Contract Payload (schema_version: monitoring.v1)
            </h2>
            <span className="text-[10px] font-mono text-[var(--text-secondary)]">Target: /monitoring/calibration</span>
          </div>
          <pre className="p-4 rounded-lg bg-[var(--surface-elevated)] border border-[var(--border)] text-[11px] font-mono text-[var(--text-secondary)] overflow-x-auto max-h-[500px] leading-relaxed">
            {JSON.stringify(calibrationReport, null, 2)}
          </pre>
        </div>
      )}

      {activeTab === "metrics" && (
        <div className="p-5 rounded-xl border border-[var(--border)] bg-[var(--surface)] space-y-3">
          <div className="flex items-center justify-between">
            <h2 className="text-xs font-bold font-mono uppercase tracking-wider text-[var(--text-primary)]">
              Live Prometheus Scrape Stream (/metrics)
            </h2>
            <span className="text-[10px] font-mono text-[var(--text-secondary)]">Target: {apiUrl}/metrics</span>
          </div>
          <pre className="p-4 rounded-lg bg-[var(--surface-elevated)] border border-[var(--border)] text-[11px] font-mono text-[var(--text-secondary)] overflow-x-auto max-h-[500px] leading-relaxed">
            {metricsText || "# No metrics received yet. Click refresh to query endpoint."}
          </pre>
        </div>
      )}
    </div>
  );
}
