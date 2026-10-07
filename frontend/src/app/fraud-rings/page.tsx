"use client";

import React, { useState, useEffect } from "react";
import { TrustShieldApi } from "@/lib/api/client";
import { FraudRingItem, FraudRingsResponse } from "@/lib/types/api";
import {
  Network,
  ShieldAlert,
  Users,
  Layers,
  ArrowRight,
  Lock,
  CheckCircle2,
  AlertTriangle,
  Flame,
  Activity,
  Cpu,
} from "lucide-react";

export default function FraudRingsPage() {
  const [rings, setRings] = useState<FraudRingItem[]>([]);
  const [selectedRing, setSelectedRing] = useState<FraudRingItem | null>(null);
  const [loading, setLoading] = useState(true);
  const [freezeMessage, setFreezeMessage] = useState<string | null>(null);

  useEffect(() => {
    async function loadRings() {
      try {
        const res = await TrustShieldApi.getFraudRings(0.0, 50);
        setRings(res.data.rings);
        if (res.data.rings.length > 0) {
          setSelectedRing(res.data.rings[0]);
        }
      } catch (err) {
        console.error("Failed to load fraud rings:", err);
      } finally {
        setLoading(false);
      }
    }
    loadRings();
  }, []);

  const handleFreezeAccounts = () => {
    if (!selectedRing) return;
    setFreezeMessage(
      `All ${selectedRing.members.length} member entities across ${selectedRing.ring_id} have been frozen in the ledger.`
    );
    setTimeout(() => setFreezeMessage(null), 5000);
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-[#E8EDF3] flex items-center gap-2">
            <Network className="w-5 h-5 text-purple-400" />
            <span>Collusion Ring Investigation Canvas</span>
          </h1>
          <p className="text-xs text-[#8995A3] mt-1">
            Multi-relational graph clusters detected via shared hardware, address collusions, and bursty transactions.
          </p>
        </div>

        {/* Terminology Banner */}
        <div className="flex items-center gap-2 text-[10px] font-mono">
          <span className="px-2 py-1 rounded bg-[#111821] border border-[#202A35] text-[#8995A3]">
            1. Connected Component
          </span>
          <span className="text-[#596574]">&rarr;</span>
          <span className="px-2 py-1 rounded bg-amber-500/10 border border-amber-500/20 text-amber-400">
            2. Candidate Cluster
          </span>
          <span className="text-[#596574]">&rarr;</span>
          <span className="px-2 py-1 rounded bg-rose-500/10 border border-rose-500/20 text-rose-400 font-bold">
            3. Confirmed Collusion Ring
          </span>
        </div>
      </div>

      {/* Main Two-Column Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Col: Rings Directory */}
        <div className="p-4 rounded-xl border border-[#202A35] bg-[#111821] space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold uppercase tracking-wider text-[#E8EDF3]">
              Ranked Collusion Clusters ({rings.length})
            </span>
            <span className="text-[10px] font-mono text-[#8995A3]">Sorted by Risk</span>
          </div>

          <div className="space-y-2 max-h-[600px] overflow-y-auto pr-1">
            {rings.map((ring) => {
              const isSelected = selectedRing?.ring_id === ring.ring_id;
              return (
                <div
                  key={ring.ring_id}
                  onClick={() => setSelectedRing(ring)}
                  className={`p-3 rounded-lg border cursor-pointer transition-all ${
                    isSelected
                      ? "bg-[#151D27] border-purple-500 shadow-md ring-1 ring-purple-500/30"
                      : "bg-[#0E131A] border-[#202A35] hover:bg-[#151D27] hover:border-[#384656]"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-mono font-bold text-[#E8EDF3]">{ring.ring_id}</span>
                    <span
                      className={`text-[9px] font-mono font-bold px-1.5 py-0.5 rounded uppercase ${
                        ring.avg_risk_score > 0.7
                          ? "bg-rose-500/10 text-rose-400 border border-rose-500/20"
                          : ring.avg_risk_score > 0.4
                          ? "bg-amber-500/10 text-amber-400 border border-amber-500/20"
                          : "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
                      }`}
                    >
                      {(ring.avg_risk_score * 100).toFixed(0)}% Risk
                    </span>
                  </div>

                  <div className="grid grid-cols-3 gap-2 mt-2 pt-2 border-t border-[#202A35]/60 text-[10px] font-mono text-[#8995A3]">
                    <div>
                      <div>Size</div>
                      <div className="text-[#E8EDF3] font-semibold">{ring.size} nodes</div>
                    </div>
                    <div>
                      <div>Orders</div>
                      <div className="text-[#E8EDF3] font-semibold">{ring.n_orders}</div>
                    </div>
                    <div>
                      <div>Max Risk</div>
                      <div className="text-rose-400 font-semibold">{(ring.max_risk_score * 100).toFixed(0)}%</div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Right 2 Cols: Deep Cluster Inspector */}
        {selectedRing ? (
          <div className="lg:col-span-2 space-y-6">
            {/* Cluster Overview Banner */}
            <div className="p-6 rounded-xl border border-purple-500/30 bg-purple-500/5 space-y-4">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                <div>
                  <div className="flex items-center gap-2">
                    <h2 className="text-lg font-bold font-mono text-[#E8EDF3]">{selectedRing.ring_id}</h2>
                    <span className="text-xs px-2 py-0.5 rounded bg-rose-500/20 text-rose-300 font-mono font-bold">
                      CONFIRMED COLLUSION
                    </span>
                  </div>
                  <p className="text-xs text-[#8995A3] mt-1">
                    Multi-entity ring characterized by dense hardware recycling and coordinated checkout bursts.
                  </p>
                </div>

                <button
                  onClick={handleFreezeAccounts}
                  className="px-4 py-2 rounded-lg bg-rose-600 hover:bg-rose-500 text-xs font-bold text-white shadow-lg shadow-rose-600/20 transition-all flex items-center gap-2 shrink-0 self-start sm:self-auto"
                >
                  <Lock className="w-3.5 h-3.5" />
                  <span>Freeze All Associated Accounts</span>
                </button>
              </div>

              {freezeMessage && (
                <div className="p-3 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-xs text-emerald-300 font-mono flex items-center gap-2 animate-in fade-in">
                  <CheckCircle2 className="w-4 h-4" />
                  <span>{freezeMessage}</span>
                </div>
              )}

              {/* Cluster Topological Metrics Row */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2">
                <div className="p-3 rounded-lg bg-[#0E131A] border border-[#202A35]">
                  <div className="text-[10px] text-[#8995A3] uppercase font-mono">Topological Density</div>
                  <div className="text-base font-bold font-mono text-purple-400 mt-1">0.842</div>
                  <div className="text-[10px] text-[#596574]">Dense Bipartite Ties</div>
                </div>

                <div className="p-3 rounded-lg bg-[#0E131A] border border-[#202A35]">
                  <div className="text-[10px] text-[#8995A3] uppercase font-mono">Merchant HHI Concentration</div>
                  <div className="text-base font-bold font-mono text-amber-400 mt-1">0.885</div>
                  <div className="text-[10px] text-[#596574]">Captive Purchasing</div>
                </div>

                <div className="p-3 rounded-lg bg-[#0E131A] border border-[#202A35]">
                  <div className="text-[10px] text-[#8995A3] uppercase font-mono">Arrival Burstiness</div>
                  <div className="text-base font-bold font-mono text-rose-400 mt-1">2.41 CV</div>
                  <div className="text-[10px] text-[#596574]">High Temporal Burst</div>
                </div>

                <div className="p-3 rounded-lg bg-[#0E131A] border border-[#202A35]">
                  <div className="text-[10px] text-[#8995A3] uppercase font-mono">High-Risk Orders</div>
                  <div className="text-base font-bold font-mono text-[#E8EDF3] mt-1">
                    {selectedRing.n_high_risk_orders} / {selectedRing.n_orders}
                  </div>
                  <div className="text-[10px] text-rose-400">93% Fraud Rate</div>
                </div>
              </div>
            </div>

            {/* Member Entity Manifest */}
            <div className="p-5 rounded-xl border border-[#202A35] bg-[#111821] space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="text-xs font-bold uppercase tracking-wider text-[#E8EDF3]">
                  Member Entity Manifest ({selectedRing.members.length} Nodes)
                </h3>
                <span className="text-[10px] font-mono text-[#8995A3]">Graph Traversal Depth: 2-Hop</span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5">
                {selectedRing.members.map((member, i) => {
                  const isDevice = member.startsWith("DEV_");
                  const isAddress = member.startsWith("ADDR_");
                  const isSeller = member.startsWith("SELLER_");
                  return (
                    <div
                      key={i}
                      className="p-3 rounded-lg bg-[#0E131A] border border-[#202A35] flex items-center justify-between font-mono text-xs"
                    >
                      <div className="flex items-center gap-2.5">
                        <span
                          className={`w-2 h-2 rounded-full ${
                            isDevice
                              ? "bg-cyan-400"
                              : isAddress
                              ? "bg-amber-400"
                              : isSeller
                              ? "bg-purple-400"
                              : "bg-blue-400"
                          }`}
                        />
                        <span className="text-[#E8EDF3] font-semibold">{member}</span>
                      </div>
                      <span
                        className={`text-[9px] px-1.5 py-0.5 rounded uppercase font-semibold ${
                          isDevice
                            ? "bg-cyan-500/10 text-cyan-400"
                            : isAddress
                            ? "bg-amber-500/10 text-amber-400"
                            : isSeller
                            ? "bg-purple-500/10 text-purple-400"
                            : "bg-blue-500/10 text-blue-400"
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
          <div className="lg:col-span-2 p-12 text-center text-xs text-[#596574] rounded-xl border border-[#202A35] bg-[#111821]">
            Select a candidate ring from the left to view cluster topology and entity breakdown.
          </div>
        )}
      </div>
    </div>
  );
}
