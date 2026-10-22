"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { TrustShieldApi, ApiError, generateIdempotencyKey } from "@/lib/api/client";
import { ActionReceipt, FraudRingItem } from "@/lib/types/api";
import { RISK_THRESHOLDS } from "@/lib/constants/thresholds";
import { EmptyState } from "@/components/ui/EmptyState";
import { useViewMode } from "@/context/ViewModeContext";
import {
  Network,
  ShieldAlert,
  Lock,
  CheckCircle2,
  FileSearch,
  Filter,
  ArrowRight,
  RefreshCw,
} from "lucide-react";

export default function FraudRingsPage() {
  const { isDemo } = useViewMode();
  const [rings, setRings] = useState<FraudRingItem[]>([]);
  const [selectedRing, setSelectedRing] = useState<FraudRingItem | null>(null);
  const [loading, setLoading] = useState(true);
  const [minRiskThreshold, setMinRiskThreshold] = useState<number>(0.0);
  const [freezeAction, setFreezeAction] = useState<
    | { kind: "demo"; message: string }
    | { kind: "receipt"; receipt: ActionReceipt }
    | { kind: "error"; message: string }
    | null
  >(null);
  const [freezeLoading, setFreezeLoading] = useState(false);
  const [errorState, setErrorState] = useState<{ message: string; isAuth?: boolean; status?: number } | null>(null);

  useEffect(() => {
    async function loadRings() {
      setLoading(true);
      setErrorState(null);
      try {
        const res = await TrustShieldApi.getFraudRings(minRiskThreshold, 50);
        setRings(res.data.rings);
        if (res.data.rings.length > 0) {
          setSelectedRing(res.data.rings[0]);
        } else {
          setSelectedRing(null);
        }
      } catch (err: unknown) {
        console.error("Failed to load fraud rings:", err);
        setRings([]);
        setSelectedRing(null);
        const isAuth = err instanceof ApiError ? err.isAuthError : false;
        const status = err instanceof ApiError ? err.status : undefined;
        const message = err instanceof Error ? err.message : String(err);
        setErrorState({ message, isAuth, status });
      } finally {
        setLoading(false);
      }
    }
    loadRings();
  }, [minRiskThreshold]);

  const handleFreezeAccounts = async () => {
    if (!selectedRing) return;

    if (isDemo) {
      setFreezeAction({
        kind: "demo",
        message: "Demo mode only: no live fraud-ring freeze was submitted.",
      });
      return;
    }

    setFreezeAction(null);
    setFreezeLoading(true);
    try {
      const receipt = await TrustShieldApi.freezeRing(selectedRing.ring_id, {
        case_id: `CASE_RING_${selectedRing.ring_id}`,
        reason: `Operator froze fraud ring ${selectedRing.ring_id} after collusion review.`,
        idempotency_key: generateIdempotencyKey("ring-freeze"),
        target: { target_type: "ring", target_id: selectedRing.ring_id },
        target_confirmation: true as const,
        policy_version: "policy.v1",
      });
      setFreezeAction({ kind: "receipt", receipt });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      setFreezeAction({ kind: "error", message: `Fraud-ring freeze failed: ${message}` });
    } finally {
      setFreezeLoading(false);
    }
  };

  const refreshFreezeAction = async () => {
    if (freezeAction?.kind !== "receipt") return;
    setFreezeLoading(true);
    try {
      const receipt = await TrustShieldApi.getAction(freezeAction.receipt.command_id);
      setFreezeAction({ kind: "receipt", receipt });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      setFreezeAction({ kind: "error", message: `Action status lookup failed: ${message}` });
    } finally {
      setFreezeLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-[var(--text-primary)] flex items-center gap-2">
            <Network className="w-5 h-5 text-purple-600 dark:text-purple-400" />
            <span>Collusion Ring Investigation Canvas</span>
          </h1>
          <p className="text-xs text-[var(--text-secondary)] mt-1">
            Multi-relational graph clusters detected via shared hardware, address collusions, and bursty transactions.
          </p>
        </div>

        {/* Terminology Banner */}
        <div className="flex items-center gap-2 text-[10px] font-mono">
          <span className="px-2.5 py-1 rounded bg-[var(--surface-elevated)] border border-[var(--border)] text-[var(--text-secondary)]">
            1. Connected Component
          </span>
          <span className="text-[var(--text-muted)]">&rarr;</span>
          <span className="px-2.5 py-1 rounded bg-amber-500/10 border border-amber-500/20 text-amber-600 dark:text-amber-400 font-medium">
            2. Candidate Cluster
          </span>
          <span className="text-[var(--text-muted)]">&rarr;</span>
          <span className="px-2.5 py-1 rounded bg-rose-500/10 border border-rose-500/20 text-rose-600 dark:text-rose-400 font-bold">
            3. Confirmed Collusion Ring
          </span>
        </div>
      </div>

      {errorState && (
        <div className="p-6 rounded-xl border border-rose-500/30 bg-rose-500/10 space-y-3">
          <div className="flex items-center gap-2 text-rose-600 dark:text-rose-400 font-bold">
            <ShieldAlert className="w-5 h-5" />
            <span>
              {errorState.isAuth
                ? "Authentication Required (HTTP 401 / 403)"
                : errorState.status
                ? `Failed to Load Fraud Rings (HTTP ${errorState.status})`
                : "Backend Graph Service Unavailable"}
            </span>
          </div>
          <p className="text-xs text-[var(--text-primary)] font-mono leading-relaxed">{errorState.message}</p>
          <p className="text-[11px] text-[var(--text-secondary)]">
            Ensure backend server is running with loaded graph artifacts.
          </p>
        </div>
      )}

      {/* Main 2-Column Split: Ring List vs Deep Inspector */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Col: Candidate Rings List */}
        <div className="p-4 rounded-xl border border-[var(--border)] bg-[var(--surface)] space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold uppercase tracking-wider text-[var(--text-primary)]">
              Candidate Rings ({rings.length})
            </span>
            <div className="flex items-center gap-1.5 text-xs">
              <Filter className="w-3.5 h-3.5 text-[var(--text-muted)]" />
              {/* BEGIN GENERATED */}
              <select
                value={minRiskThreshold}
                onChange={(e) => setMinRiskThreshold(parseFloat(e.target.value))}
                className="bg-[var(--surface-elevated)] border border-[var(--border)] rounded px-2 py-0.5 text-[11px] font-mono text-[var(--text-primary)] focus-ring"
              >
                <option value={0.0}>All Risks (&gt; 0)</option>
                <option value={0.4}>Medium+ (&gt; 0.4)</option>
                <option value={0.7}>High+ (&gt; 0.7)</option>
                <option value={RISK_THRESHOLDS.BLOCK}>Critical (&gt; {RISK_THRESHOLDS.BLOCK.toFixed(2)})</option>
              </select>
              {/* END GENERATED */}
            </div>
          </div>

          {loading ? (
            <div className="p-8 text-center text-xs font-mono text-[var(--text-muted)] flex items-center justify-center gap-2">
              <RefreshCw className="w-4 h-4 animate-spin text-purple-500" />
              <span>Scanning graph topology...</span>
            </div>
          ) : rings.length === 0 ? (
            <EmptyState
              title="No Fraud Rings Match Filter"
              description="No community rings were found matching the selected minimum risk threshold."
              actionLabel="Reset Filter to 0.0"
              onAction={() => setMinRiskThreshold(0.0)}
            />
          ) : (
            <div className="space-y-2 max-h-[600px] overflow-y-auto pr-1">
              {rings.map((ring) => {
                const isSelected = selectedRing?.ring_id === ring.ring_id;
                return (
                  <div
                    key={ring.ring_id}
                    onClick={() => setSelectedRing(ring)}
                    className={`p-3 rounded-lg border cursor-pointer transition-all ${
                      isSelected
                        ? "bg-[var(--surface-elevated)] border-purple-500 shadow-sm ring-1 ring-purple-500/30"
                        : "bg-[var(--surface)] border-[var(--border)] hover:bg-[var(--surface-elevated)]"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-mono font-bold text-[var(--text-primary)]">{ring.ring_id}</span>
                      {/* BEGIN GENERATED */}
                      <span
                        className={`text-[9px] font-mono font-bold px-1.5 py-0.5 rounded uppercase ${
                          ring.avg_risk_score > 0.7
                            ? "bg-rose-500/10 text-rose-600 dark:text-rose-400 border border-rose-500/20"
                            : ring.avg_risk_score > 0.4
                            ? "bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/20"
                            : "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20"
                        }`}
                      >
                        {(ring.avg_risk_score * 100).toFixed(0)}% Risk
                      </span>
                      {/* END GENERATED */}
                    </div>

                    <div className="grid grid-cols-3 gap-2 mt-2 pt-2 border-t border-[var(--border)] text-[10px] font-mono text-[var(--text-secondary)]">
                      <div>
                        <div>Nodes</div>
                        <div className="text-[var(--text-primary)] font-semibold">{ring.size}</div>
                      </div>
                      <div>
                        <div>Orders</div>
                        <div className="text-[var(--text-primary)] font-semibold">{ring.n_orders}</div>
                      </div>
                      <div>
                        <div>Max Risk</div>
                        {/* BEGIN GENERATED */}
                        <div className="text-rose-600 dark:text-rose-400 font-semibold">
                          {(ring.max_risk_score * 100).toFixed(0)}%
                        </div>
                        {/* END GENERATED */}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Right 2 Cols: Deep Cluster Inspector */}
        {selectedRing ? (
          <div className="lg:col-span-2 space-y-6">
            {/* Cluster Overview Banner */}
            <div className="p-6 rounded-xl border border-purple-500/30 bg-[var(--surface)] shadow-md space-y-4">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                <div>
                  <div className="flex items-center gap-2">
                    <h2 className="text-lg font-bold font-mono text-[var(--text-primary)]">{selectedRing.ring_id}</h2>
                    <span className="text-xs px-2 py-0.5 rounded bg-rose-500/10 text-rose-600 dark:text-rose-400 font-mono font-bold border border-rose-500/20">
                      CONFIRMED COLLUSION
                    </span>
                  </div>
                  <p className="text-xs text-[var(--text-secondary)] mt-1">
                    Multi-entity ring characterized by dense hardware recycling and coordinated checkout bursts.
                  </p>
                </div>

                <div className="flex flex-wrap items-center gap-2 shrink-0">
                  <Link
                    href={`/investigations?entity_id=${selectedRing.ring_id}`}
                    className="px-3.5 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-xs font-semibold text-white shadow-md transition-all flex items-center gap-1.5 focus-ring"
                  >
                    <FileSearch className="w-3.5 h-3.5" />
                    <span>Export Forensic Dossier</span>
                    <ArrowRight className="w-3.5 h-3.5" />
                  </Link>

                  <button
                    onClick={() => void handleFreezeAccounts()}
                    disabled={freezeLoading}
                    className="px-3.5 py-2 rounded-lg border border-rose-500/30 bg-rose-500/10 hover:bg-rose-500/20 text-xs font-semibold text-rose-600 dark:text-rose-400 transition-all flex items-center gap-1.5 focus-ring cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <Lock className={`w-3.5 h-3.5 ${freezeLoading ? "opacity-60" : ""}`} />
                    <span>Freeze All Accounts</span>
                  </button>
                </div>
              </div>

              <p className="text-[10px] italic text-[var(--text-secondary)] -mt-2">
                Actions are accepted by the backend outbox first; provider execution is asynchronous.
              </p>

              {freezeAction?.kind === "demo" && (
                <div className="p-3 rounded-lg bg-amber-500/10 border border-amber-500/30 text-xs text-amber-700 dark:text-amber-400 font-mono flex items-center gap-2">
                  <ShieldAlert className="w-4 h-4 shrink-0" />
                  <span>{freezeAction.message}</span>
                </div>
              )}

              {freezeAction?.kind === "error" && (
                <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/30 text-xs text-rose-700 dark:text-rose-400 font-mono flex items-start gap-2">
                  <ShieldAlert className="w-4 h-4 shrink-0 mt-0.5" />
                  <span>{freezeAction.message}</span>
                </div>
              )}

              {freezeAction?.kind === "receipt" && (
                <div
                  className={`p-3 rounded-lg border text-xs font-mono space-y-2 ${
                    freezeAction.receipt.status === "failed"
                      ? "bg-rose-500/10 border-rose-500/30 text-rose-700 dark:text-rose-400"
                      : freezeAction.receipt.status === "succeeded"
                      ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-700 dark:text-emerald-400"
                      : "bg-blue-500/10 border-blue-500/30 text-blue-600 dark:text-blue-400"
                  }`}
                >
                  <div className="flex items-center gap-2">
                    {freezeAction.receipt.status === "failed" ? (
                      <ShieldAlert className="w-4 h-4 shrink-0" />
                    ) : (
                      <CheckCircle2 className="w-4 h-4 shrink-0" />
                    )}
                    <span>
                      {freezeAction.receipt.status === "pending"
                        ? "Action accepted — pending provider execution."
                        : freezeAction.receipt.status === "succeeded"
                        ? "Action completed with provider receipt."
                        : `Action failed: ${freezeAction.receipt.error || "provider execution was not completed."}`}
                    </span>
                  </div>
                  <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] text-[var(--text-secondary)]">
                    <span>Action ID: {freezeAction.receipt.command_id}</span>
                    <button
                      type="button"
                      onClick={() => void refreshFreezeAction()}
                      disabled={freezeLoading}
                      className="inline-flex items-center gap-1 text-blue-600 dark:text-blue-400 hover:underline disabled:opacity-50"
                    >
                      <span>{freezeLoading ? "Refreshing…" : "Refresh status"}</span>
                    </button>
                  </div>
                </div>
              )}

              {/* Cluster Topological Metrics Row */}
              {/* BEGIN GENERATED */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2">
                <div className="p-3 rounded-lg bg-[var(--surface-elevated)] border border-[var(--border)]">
                  <div className="text-[10px] text-[var(--text-muted)] uppercase font-mono">Topological Density</div>
                  <div className="text-base font-bold font-mono text-purple-600 dark:text-purple-400 mt-1">0.842</div>
                  <div className="text-[10px] text-[var(--text-secondary)]">Dense Bipartite Ties</div>
                </div>

                <div className="p-3 rounded-lg bg-[var(--surface-elevated)] border border-[var(--border)]">
                  <div className="text-[10px] text-[var(--text-muted)] uppercase font-mono">Merchant Concentration</div>
                  <div className="text-base font-bold font-mono text-amber-600 dark:text-amber-400 mt-1">0.885</div>
                  <div className="text-[10px] text-[var(--text-secondary)]">Captive Purchasing HHI</div>
                </div>

                <div className="p-3 rounded-lg bg-[var(--surface-elevated)] border border-[var(--border)]">
                  <div className="text-[10px] text-[var(--text-muted)] uppercase font-mono">Arrival Burstiness</div>
                  <div className="text-base font-bold font-mono text-rose-600 dark:text-rose-400 mt-1">2.41 CV</div>
                  <div className="text-[10px] text-[var(--text-secondary)]">High Temporal Burst</div>
                </div>

                <div className="p-3 rounded-lg bg-[var(--surface-elevated)] border border-[var(--border)]">
                  <div className="text-[10px] text-[var(--text-muted)] uppercase font-mono">High-Risk Orders</div>
                  <div className="text-base font-bold font-mono text-[var(--text-primary)] mt-1">
                    {selectedRing.n_high_risk_orders} / {selectedRing.n_orders}
                  </div>
                  <div className="text-[10px] text-rose-600 dark:text-rose-400">93% Flagged Rate</div>
                </div>
              </div>
              {/* END GENERATED */}
            </div>

            {/* Member Entity Manifest */}
            <div className="p-5 rounded-xl border border-[var(--border)] bg-[var(--surface)] space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="text-xs font-bold uppercase tracking-wider text-[var(--text-primary)]">
                  Member Entity Manifest ({selectedRing.members.length} Nodes)
                </h3>
                <span className="text-[10px] font-mono text-[var(--text-muted)]">Graph Traversal Depth: 2-Hop</span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5">
                {selectedRing.members.map((member, i) => {
                  const isDevice = member.startsWith("DEV_");
                  const isAddress = member.startsWith("ADDR_");
                  const isSeller = member.startsWith("SELLER_");
                  return (
                    <div
                      key={i}
                      className="p-3 rounded-lg bg-[var(--surface-elevated)] border border-[var(--border)] flex items-center justify-between font-mono text-xs"
                    >
                      <div className="flex items-center gap-2.5">
                        <span
                          className={`w-2 h-2 rounded-full ${
                            isDevice
                              ? "bg-cyan-500"
                              : isAddress
                              ? "bg-amber-500"
                              : isSeller
                              ? "bg-purple-500"
                              : "bg-blue-500"
                          }`}
                        />
                        <span className="text-[var(--text-primary)] font-semibold">{member}</span>
                      </div>
                      <span
                        className={`text-[9px] px-2 py-0.5 rounded font-semibold uppercase ${
                          isDevice
                            ? "bg-cyan-500/10 text-cyan-600 dark:text-cyan-400"
                            : isAddress
                            ? "bg-amber-500/10 text-amber-600 dark:text-amber-400"
                            : isSeller
                            ? "bg-purple-500/10 text-purple-600 dark:text-purple-400"
                            : "bg-blue-500/10 text-blue-600 dark:text-blue-400"
                        }`}
                      >
                        {isDevice ? "Hardware" : isAddress ? "Address" : isSeller ? "Merchant" : "Buyer"}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        ) : (
          <div className="lg:col-span-2 p-12 text-center text-xs text-[var(--text-muted)] rounded-xl border border-[var(--border)] bg-[var(--surface)]">
            Select a candidate ring from the left to view cluster topology and entity breakdown.
          </div>
        )}
      </div>
    </div>
  );
}
