"use client";

import React, { useState } from "react";
import {
  FileSearch,
  CheckCircle2,
  AlertTriangle,
  Clock,
  ShieldAlert,
  ArrowRight,
  Layers,
  Network,
  Tag,
  RotateCcw,
  User,
  Smartphone,
  Lock,
} from "lucide-react";

interface CaseItem {
  id: string;
  title: string;
  severity: "CRITICAL" | "HIGH" | "MEDIUM";
  date: string;
  primaryEntity: string;
  assignedTo: string;
  summary: string;
  timeline: { time: string; event: string; type: "ORDER" | "DEVICE" | "RETURN" | "GRAPH" }[];
  evidence: {
    transaction: string;
    graph: string;
    behavioral: string;
    multimodal: string;
    returns: string;
  };
}

const CASES_DATA: CaseItem[] = [
  {
    id: "CASE-001",
    title: "Device Farm Collusion Cluster #1",
    severity: "CRITICAL",
    date: "2026-10-07 22:45",
    primaryEntity: "RING_COLLUSION_01",
    assignedTo: "Senior Fraud Investigator",
    summary:
      "Nine distinct buyer identities sharing identical mobile hardware signatures placing burst orders with a common single-merchant store within a 4-hour window.",
    timeline: [
      { time: "18:12:04", event: "BUYER_RING_MEMBER_01 placed ORD_78901 ($890.00)", type: "ORDER" },
      { time: "18:44:19", event: "DEV_FARM_99 fingerprint detected on BUYER_RING_MEMBER_02 login", type: "DEVICE" },
      { time: "19:15:33", event: "BUYER_RING_MEMBER_02 placed ORD_78904 ($910.00)", type: "ORDER" },
      { time: "20:02:11", event: "Graph edge weight exceeded threshold (HHI: 0.88)", type: "GRAPH" },
      { time: "22:45:00", event: "Automated Trust Engine flagged cluster with 94.2% fraud risk", type: "GRAPH" },
    ],
    evidence: {
      transaction: "Cumulative cluster order volume: $14,240.00. 100% of transactions exceed 90th percentile basket size.",
      graph: "Graph density: 0.842. 8-node bipartite clique with shared hardware DEV_FARM_99.",
      behavioral: "Mean time between account registration and high-value purchase: 4.2 hours.",
      multimodal: "N/A — tabular and topological fraud vector.",
      returns: "1 coordinated refund claim submitted for previous transaction batch.",
    },
  },
  {
    id: "CASE-002",
    title: "Cross-Seller Catalog Photo Theft",
    severity: "HIGH",
    date: "2026-10-07 19:20",
    primaryEntity: "SELLER_SUSPECT_44",
    assignedTo: "Catalog Integrity Team",
    summary:
      "New merchant listing high-end electronics using imagery stolen from authorized merchant with 94.5% CLIP cosine similarity match.",
    timeline: [
      { time: "14:10:00", event: "Merchant SELLER_SUSPECT_44 registered account", type: "ORDER" },
      { time: "16:22:15", event: "Uploaded LISTING_CANON_501 ($340.00)", type: "ORDER" },
      { time: "16:22:18", event: "FAISS vector nearest neighbor query triggered", type: "GRAPH" },
      { time: "19:20:00", event: "Self-match exclusion confirmed catalog theft from PROD_B07XYZ99", type: "GRAPH" },
    ],
    evidence: {
      transaction: "Priced at $340.00, slightly undercutting median price ($350.00) to maximize quick sales.",
      graph: "Zero historical sales or seller trust degree in graph ledger.",
      behavioral: "Seller account age: 3 days.",
      multimodal: "CLIP cosine visual similarity: 0.945 against catalog hero image.",
      returns: "Zero historical returns.",
    },
  },
  {
    id: "CASE-003",
    title: "Serial Wardrobing & Empty Box Refund Abuse",
    severity: "MEDIUM",
    date: "2026-10-07 14:10",
    primaryEntity: "BUYER_REFUND_ABUSER",
    assignedTo: "Chargeback Resolution Unit",
    summary:
      "Customer with historical 80% return rate submitting another high-value 'wrong item received' chargeback claim.",
    timeline: [
      { time: "Sep 12", event: "Claimed refund for ORD_5521 ($210.00) — Wrong Item", type: "RETURN" },
      { time: "Sep 28", event: "Claimed refund for ORD_6104 ($180.00) — Defective", type: "RETURN" },
      { time: "Oct 05", event: "Placed ORD_SIM_RETURN_04 ($320.00)", type: "ORDER" },
      { time: "Oct 07", event: "Submitted RET_CLAIM_9021 within 48h of delivery", type: "RETURN" },
    ],
    evidence: {
      transaction: "Order amount: $320.00.",
      graph: "Single buyer node connected to 2 distinct merchant targets.",
      behavioral: "Account return frequency: 4 refunds out of 5 lifetime purchases.",
      multimodal: "N/A.",
      returns: "Return rate of 80% vs platform average of 4.2%.",
    },
  },
];

export default function InvestigationsPage() {
  const [selectedCase, setSelectedCase] = useState<CaseItem>(CASES_DATA[0]);

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-[#E8EDF3] flex items-center gap-2">
            <FileSearch className="w-5 h-5 text-blue-400" />
            <span>Case Management & Grounded Evidence Dossiers</span>
          </h1>
          <p className="text-xs text-[#8995A3] mt-1">
            Forensic case files with verifiable multi-channel evidence timelines and deterministic anti-hallucination bounds.
          </p>
        </div>

        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[#111821] border border-[#202A35] text-xs font-mono">
          <span className="text-[#8995A3]">Guard:</span>
          <span className="text-emerald-400 font-semibold">100% Grounded in Ledger</span>
        </div>
      </div>

      {/* Two Column Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Col: Cases List */}
        <div className="p-4 rounded-xl border border-[#202A35] bg-[#111821] space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold uppercase tracking-wider text-[#E8EDF3]">
              Active Incident Dossiers ({CASES_DATA.length})
            </span>
            <span className="text-[10px] font-mono text-[#8995A3]">Priority Sorted</span>
          </div>

          <div className="space-y-2">
            {CASES_DATA.map((c) => {
              const isSelected = selectedCase.id === c.id;
              return (
                <div
                  key={c.id}
                  onClick={() => setSelectedCase(c)}
                  className={`p-3 rounded-lg border cursor-pointer transition-all ${
                    isSelected
                      ? "bg-[#151D27] border-blue-500 shadow-md ring-1 ring-blue-500/30"
                      : "bg-[#0E131A] border-[#202A35] hover:bg-[#151D27] hover:border-[#384656]"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-mono font-bold text-[#E8EDF3]">{c.id}</span>
                    <span
                      className={`text-[9px] font-mono font-bold px-1.5 py-0.5 rounded uppercase ${
                        c.severity === "CRITICAL"
                          ? "bg-rose-500/10 text-rose-400 border border-rose-500/20"
                          : c.severity === "HIGH"
                          ? "bg-orange-500/10 text-orange-400 border border-orange-500/20"
                          : "bg-amber-500/10 text-amber-400 border border-amber-500/20"
                      }`}
                    >
                      {c.severity}
                    </span>
                  </div>

                  <div className="text-xs font-semibold text-[#E8EDF3] mt-1 line-clamp-1">{c.title}</div>
                  <div className="text-[11px] text-[#8995A3] mt-1 flex items-center justify-between font-mono">
                    <span>{c.primaryEntity}</span>
                    <span>{c.date}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Right 2 Cols: Deep Dossier View */}
        <div className="lg:col-span-2 space-y-6">
          {/* Dossier Header Card */}
          <div className="p-6 rounded-xl border border-[#202A35] bg-[#111821] space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-lg font-bold font-mono text-[#E8EDF3]">{selectedCase.id}: {selectedCase.title}</span>
                  <span
                    className={`text-xs px-2 py-0.5 rounded font-mono font-bold uppercase ${
                      selectedCase.severity === "CRITICAL"
                        ? "bg-rose-500/20 text-rose-300"
                        : "bg-amber-500/20 text-amber-300"
                    }`}
                  >
                    {selectedCase.severity}
                  </span>
                </div>
                <div className="text-xs text-[#8995A3] mt-1 font-mono">
                  Primary Target: <span className="text-white font-semibold">{selectedCase.primaryEntity}</span> · Opened: {selectedCase.date}
                </div>
              </div>

              <div className="flex items-center gap-2">
                <button className="px-3 py-1.5 rounded-lg bg-blue-600/20 hover:bg-blue-600/30 border border-blue-500/30 text-blue-300 text-xs font-mono font-semibold transition-colors">
                  Export Dossier PDF
                </button>
              </div>
            </div>

            <p className="text-xs text-[#E8EDF3] leading-relaxed bg-[#0E131A] p-3.5 rounded-lg border border-[#202A35]">
              {selectedCase.summary}
            </p>
          </div>

          {/* Chronological Evidence Timeline */}
          <div className="p-5 rounded-xl border border-[#202A35] bg-[#111821] space-y-4">
            <div className="text-xs font-bold uppercase tracking-wider text-[#E8EDF3]">
              Chronological Evidence Sequence
            </div>

            <div className="relative pl-6 space-y-4 before:absolute before:left-2 before:top-2 before:bottom-2 before:w-0.5 before:bg-[#202A35]">
              {selectedCase.timeline.map((step, idx) => (
                <div key={idx} className="relative flex items-start gap-3">
                  <span className="absolute -left-6 top-1 w-2.5 h-2.5 rounded-full bg-blue-500 ring-4 ring-[#111821]" />
                  <div className="text-xs font-mono">
                    <span className="text-[#596574] mr-2">[{step.time}]</span>
                    <span className="text-[#E8EDF3]">{step.event}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Evidence Channel Breakdown Cards */}
          <div className="p-5 rounded-xl border border-[#202A35] bg-[#111821] space-y-4">
            <div className="text-xs font-bold uppercase tracking-wider text-[#E8EDF3]">
              Channel-Specific Evidence Dossier
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
              <div className="p-3.5 rounded-lg bg-[#0E131A] border border-[#202A35]">
                <div className="text-[10px] font-mono text-blue-400 font-bold uppercase">1. Transaction Evidence</div>
                <div className="text-[#8995A3] mt-1 leading-relaxed">{selectedCase.evidence.transaction}</div>
              </div>

              <div className="p-3.5 rounded-lg bg-[#0E131A] border border-[#202A35]">
                <div className="text-[10px] font-mono text-purple-400 font-bold uppercase">2. Graph Topology Evidence</div>
                <div className="text-[#8995A3] mt-1 leading-relaxed">{selectedCase.evidence.graph}</div>
              </div>

              <div className="p-3.5 rounded-lg bg-[#0E131A] border border-[#202A35]">
                <div className="text-[10px] font-mono text-amber-400 font-bold uppercase">3. Behavioral Velocity</div>
                <div className="text-[#8995A3] mt-1 leading-relaxed">{selectedCase.evidence.behavioral}</div>
              </div>

              <div className="p-3.5 rounded-lg bg-[#0E131A] border border-[#202A35]">
                <div className="text-[10px] font-mono text-cyan-400 font-bold uppercase">4. Multimodal / CLIP Evidence</div>
                <div className="text-[#8995A3] mt-1 leading-relaxed">{selectedCase.evidence.multimodal}</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
