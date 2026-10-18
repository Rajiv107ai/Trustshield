"use client";

import React, { useState, useEffect, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import {
  FileSearch,
  CheckCircle2,
  ShieldAlert,
  Sparkles,
  RefreshCw,
  BookOpen,
  Printer,
  Download,
  MessageSquare,
  Tag,
  Plus,
} from "lucide-react";
import { TrustShieldApi, ApiError } from "@/lib/api/client";
import { DossierResponse } from "@/lib/types/api";

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

const cItem = (
  id: string,
  title: string,
  severity: CaseItem["severity"],
  date: string,
  primaryEntity: string,
  assignedTo: string,
  summary: string,
  timeline: CaseItem["timeline"],
  evidence: CaseItem["evidence"]
): CaseItem => ({ id, title, severity, date, primaryEntity, assignedTo, summary, timeline, evidence });

/* BEGIN GENERATED */
const CASES_DATA: CaseItem[] = [
  cItem("CASE-001", "Device Farm Collusion Cluster #1", "CRITICAL", "2026-10-07 22:45", "ORD_78901", "Senior Fraud Investigator",
    "Nine distinct buyer identities sharing identical mobile hardware signatures placing burst orders with a common single-merchant store within a 4-hour window.",
    [
      { time: "18:12:04", event: "BUYER_RING_MEMBER_01 placed ORD_78901 (₹890.00)", type: "ORDER" },
      { time: "18:44:19", event: "DEV_FARM_99 fingerprint detected on BUYER_RING_MEMBER_02 login", type: "DEVICE" },
      { time: "19:15:33", event: "BUYER_RING_MEMBER_02 placed ORD_78904 (₹910.00)", type: "ORDER" },
      { time: "20:02:11", event: "Graph edge weight exceeded threshold (HHI: 0.88)", type: "GRAPH" },
      { time: "22:45:00", event: "Automated Trust Engine flagged cluster with high fraud risk", type: "GRAPH" },
    ],
    {
      transaction: "Cumulative cluster order volume: ₹14,240.00. 100% of transactions exceed 90th percentile basket size.",
      graph: "Graph density: 0.842. 8-node bipartite clique with shared hardware DEV_FARM_99.",
      behavioral: "Mean time between account registration and high-value purchase: 4.2 hours.",
      multimodal: "N/A — tabular and topological fraud vector.",
      returns: "1 coordinated refund claim submitted for previous transaction batch.",
    }),
  cItem("CASE-002", "Cross-Seller Catalog Photo Theft", "HIGH", "2026-10-07 19:20", "ORD_78920", "Catalog Integrity Team",
    "New merchant listing high-end electronics using imagery stolen from authorized merchant with 94.5% CLIP cosine similarity match.",
    [
      { time: "14:10:00", event: "Merchant SELLER_SUSPECT_44 registered account", type: "ORDER" },
      { time: "16:22:15", event: "Uploaded LISTING_CANON_501 (₹340.00)", type: "ORDER" },
      { time: "16:22:18", event: "FAISS vector nearest neighbor query triggered", type: "GRAPH" },
      { time: "19:20:00", event: "Self-match exclusion confirmed catalog theft from PROD_B07XYZ99", type: "GRAPH" },
    ],
    {
      transaction: "Priced at ₹340.00, slightly undercutting median price (₹350.00) to maximize quick sales.",
      graph: "Zero historical sales or seller trust degree in graph ledger.",
      behavioral: "Seller account age: 3 days.",
      multimodal: "CLIP cosine visual similarity: 0.945 against catalog hero image.",
      returns: "Zero historical returns.",
    }),
  cItem("CASE-003", "Serial Wardrobing & Empty Box Refund Abuse", "MEDIUM", "2026-10-07 14:10", "ORD_78907", "Chargeback Resolution Unit",
    "Customer with historical 80% return rate submitting another high-value 'wrong item received' chargeback claim.",
    [
      { time: "Sep 12", event: "Claimed refund for ORD_5521 (₹210.00) — Wrong Item", type: "RETURN" },
      { time: "Sep 28", event: "Claimed refund for ORD_6104 (₹180.00) — Defective", type: "RETURN" },
      { time: "Oct 05", event: "Placed ORD_SIM_RETURN_04 (₹320.00)", type: "ORDER" },
      { time: "Oct 07", event: "Submitted RET_CLAIM_9021 within 48h of delivery", type: "RETURN" },
    ],
    {
      transaction: "Order amount: ₹320.00.",
      graph: "Single buyer node connected to 2 distinct merchant targets.",
      behavioral: "Account return frequency: 4 refunds out of 5 lifetime purchases.",
      multimodal: "N/A.",
      returns: "Return rate of 80% vs platform average of 4.2%.",
    }),
];
/* END GENERATED */

function InvestigationsContent() {
  const searchParams = useSearchParams();
  const targetParam = searchParams.get("order_id") || searchParams.get("entity_id");

  const [selectedCaseId, setSelectedCaseId] = useState<string>(() => {
    if (targetParam) {
      const match = CASES_DATA.find((c) => c.primaryEntity === targetParam || c.id === targetParam);
      if (match) return match.id;
    }
    return CASES_DATA[0].id;
  });

  const selectedCase =
    CASES_DATA.find((c) => c.id === selectedCaseId) ??
    (targetParam ? CASES_DATA.find((c) => c.primaryEntity === targetParam) : undefined) ??
    CASES_DATA[0];

  const [liveDossier, setLiveDossier] = useState<DossierResponse | null>(null);
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [errorState, setErrorState] = useState<{ message: string; isAuth?: boolean; status?: number } | null>(null);

  // Investigator Local Notes (Honest Browser Storage)
  const [notes, setNotes] = useState<{ text: string; author: string; timestamp: string; tag: string }[]>([]);
  const [newNoteText, setNewNoteText] = useState("");
  const [selectedTag, setSelectedTag] = useState("Under Review");

  useEffect(() => {
    setLiveDossier(null);
    setErrorState(null);
    try {
      const stored = localStorage.getItem(`trustshield_notes_${selectedCase.id}`);
      if (stored) {
        const parsed = JSON.parse(stored);
        if (Array.isArray(parsed)) {
          queueMicrotask(() => setNotes(parsed));
          return;
        }
      }
      queueMicrotask(() => setNotes([]));
    } catch {
      queueMicrotask(() => setNotes([]));
    }
  }, [selectedCase.id]);

  const handleAddNote = (e: React.FormEvent) => {
    e.preventDefault();
    if (!newNoteText.trim()) return;
    const newNote = {
      text: newNoteText.trim(),
      author: "Operator (Local Session)",
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      tag: selectedTag,
    };
    const updated = [newNote, ...notes];
    setNotes(updated);
    setNewNoteText("");
    try {
      localStorage.setItem(`trustshield_notes_${selectedCase.id}`, JSON.stringify(updated));
    } catch {
      // Storage unavailable
    }
  };

  const handleSynthesizeDossier = async () => {
    setIsGenerating(true);
    setErrorState(null);
    try {
      const res = await TrustShieldApi.generateDossier({
        entity_type: "transaction",
        entity_id: selectedCase.primaryEntity,
        include_graph_evidence: true,
      });
      setLiveDossier(res.data);
    } catch (err: unknown) {
      console.error("Dossier generation error:", err);
      setLiveDossier(null);
      const isAuth = err instanceof ApiError ? err.isAuthError : false;
      const status = err instanceof ApiError ? err.status : undefined;
      const message = err instanceof Error ? err.message : String(err);
      setErrorState({ message, isAuth, status });
    } finally {
      setIsGenerating(false);
    }
  };

  const handlePrintDossier = () => {
    if (typeof window !== "undefined") {
      window.print();
    }
  };

  const handleExportJson = () => {
    const exportData = {
      platform: "TrustShield AI Lean Investigation Console",
      exported_at: new Date().toISOString(),
      case_id: selectedCase.id,
      title: selectedCase.title,
      severity: selectedCase.severity,
      target_entity: selectedCase.primaryEntity,
      summary: liveDossier ? liveDossier.executive_summary : selectedCase.summary,
      evidence_channels: selectedCase.evidence,
      timeline: selectedCase.timeline,
      analyst_notes_local: notes,
      status: "STRICTLY_GROUNDED_AUDIT_TRAIL",
    };
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `trustshield_dossier_${selectedCase.id}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-[var(--text-primary)] flex items-center gap-2">
            <FileSearch className="w-5 h-5 text-blue-600 dark:text-blue-400" />
            <span>Case Management &amp; Grounded Evidence Dossiers</span>
          </h1>
          <p className="text-xs text-[var(--text-secondary)] mt-1">
            Forensic case files with verifiable multi-channel evidence timelines, in-memory graph topology findings, and deterministic telemetry bounds.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={handleExportJson}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-[var(--border)] bg-[var(--surface)] hover:bg-[var(--surface-elevated)] text-xs font-medium text-[var(--text-primary)] transition-colors focus-ring cursor-pointer"
            title="Download full JSON evidence report"
          >
            <Download className="w-3.5 h-3.5 text-blue-500" />
            <span>Export JSON</span>
          </button>

          <button
            onClick={handlePrintDossier}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-[var(--border)] bg-[var(--surface)] hover:bg-[var(--surface-elevated)] text-xs font-medium text-[var(--text-primary)] transition-colors focus-ring cursor-pointer"
            title="Print printable evidence dossier"
          >
            <Printer className="w-3.5 h-3.5 text-[var(--text-secondary)]" />
            <span>Print Dossier</span>
          </button>

          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[var(--surface)] border border-[var(--border)] text-xs font-mono">
            <span className="text-[var(--text-muted)]">Guard:</span>
            <span className="text-emerald-600 dark:text-emerald-400 font-semibold">Strictly Grounded</span>
          </div>
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
                ? `Dossier Synthesis Failed (HTTP ${errorState.status})`
                : "Backend Service Unavailable"}
            </span>
          </div>
          <p className="text-xs text-[var(--text-primary)] font-mono leading-relaxed">{errorState.message}</p>
          <p className="text-[11px] text-[var(--text-secondary)]">
            Verify that the FastAPI backend is running on port 8000.
          </p>
        </div>
      )}

      {/* Two Column Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Col: Cases List */}
        <div className="p-4 rounded-xl border border-[var(--border)] bg-[var(--surface)] space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold uppercase tracking-wider text-[var(--text-primary)]">
              Active Incident Dossiers ({CASES_DATA.length})
            </span>
            <span className="text-[10px] font-mono text-[var(--text-muted)]">Priority Sorted</span>
          </div>

          <div className="space-y-2">
            {CASES_DATA.map((c) => {
              const isSelected = selectedCase.id === c.id;
              return (
                <div
                  key={c.id}
                  onClick={() => {
                    setSelectedCaseId(c.id);
                    setLiveDossier(null);
                  }}
                  className={`p-3 rounded-lg border cursor-pointer transition-all ${
                    isSelected
                      ? "bg-[var(--surface-elevated)] border-blue-500 shadow-sm ring-1 ring-blue-500/30"
                      : "bg-[var(--surface)] border-[var(--border)] hover:bg-[var(--surface-elevated)]"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-mono font-bold text-[var(--text-primary)]">{c.id}</span>
                    <span
                      className={`text-[9px] font-mono font-bold px-1.5 py-0.5 rounded uppercase ${
                        c.severity === "CRITICAL"
                          ? "bg-rose-500/10 text-rose-600 dark:text-rose-400 border border-rose-500/20"
                          : c.severity === "HIGH"
                          ? "bg-orange-500/10 text-orange-600 dark:text-orange-400 border border-orange-500/20"
                          : "bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/20"
                      }`}
                    >
                      {c.severity}
                    </span>
                  </div>

                  <div className="text-xs font-semibold text-[var(--text-primary)] mt-1 line-clamp-1">{c.title}</div>
                  <div className="text-[11px] text-[var(--text-secondary)] mt-1 flex items-center justify-between font-mono">
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
          <div className="p-6 rounded-xl border border-[var(--border)] bg-[var(--surface)] shadow-md space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <div>
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-lg font-bold font-mono text-[var(--text-primary)]">
                    {selectedCase.id}: {selectedCase.title}
                  </span>
                  <span
                    className={`text-xs px-2 py-0.5 rounded font-mono font-bold uppercase border ${
                      selectedCase.severity === "CRITICAL"
                        ? "bg-rose-500/10 text-rose-600 dark:text-rose-400 border-rose-500/20"
                        : "bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/20"
                    }`}
                  >
                    {selectedCase.severity}
                  </span>
                </div>
                <div className="text-xs text-[var(--text-secondary)] mt-1 font-mono flex items-center gap-2">
                  <span>Target: <strong className="text-[var(--text-primary)]">{selectedCase.primaryEntity}</strong></span>
                  <span>·</span>
                  <span>Opened: {selectedCase.date}</span>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <button
                  onClick={handleSynthesizeDossier}
                  disabled={isGenerating}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white text-xs font-mono font-semibold transition-all shadow-md shadow-blue-500/20 disabled:opacity-50 focus-ring"
                >
                  {isGenerating ? (
                    <>
                      <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                      <span>Synthesizing...</span>
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-3.5 h-3.5 text-blue-200" />
                      <span>Synthesize Live Dossier</span>
                    </>
                  )}
                </button>
              </div>
            </div>

            <p className="text-xs text-[var(--text-primary)] leading-relaxed bg-[var(--surface-elevated)] p-3.5 rounded-lg border border-[var(--border)]">
              {liveDossier ? liveDossier.executive_summary : selectedCase.summary}
            </p>
          </div>

          {/* Live Synthesized Findings Section */}
          {liveDossier && (
            <div className="p-6 rounded-xl border border-blue-500/30 bg-[var(--surface)] space-y-4">
              <div className="flex items-center justify-between border-b border-[var(--border)] pb-3">
                <div className="flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-blue-500" />
                  <span className="text-xs font-bold uppercase tracking-wider text-[var(--text-primary)]">
                    Live Telemetry Synthesis (Deterministic Audit Pack)
                  </span>
                </div>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-blue-500/10 text-blue-600 dark:text-blue-400 border border-blue-500/20 font-bold">
                  Decision: {liveDossier.decision}
                </span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div className="p-3.5 rounded-lg bg-[var(--surface-elevated)] border border-[var(--border)] space-y-1">
                  <span className="text-[11px] font-mono font-bold uppercase text-blue-600 dark:text-blue-400">
                    1. Observed Facts
                  </span>
                  <ul className="text-xs text-[var(--text-primary)] space-y-1 list-disc pl-4 pt-1">
                    {liveDossier.observed_evidence.verified_facts.map((fact: string, idx: number) => (
                      <li key={idx}>{fact}</li>
                    ))}
                  </ul>
                </div>

                <div className="p-3.5 rounded-lg bg-[var(--surface-elevated)] border border-[var(--border)] space-y-1">
                  <span className="text-[11px] font-mono font-bold uppercase text-indigo-600 dark:text-indigo-400">
                    2. Triggered Reason Codes
                  </span>
                  <ul className="text-xs text-[var(--text-primary)] space-y-1 list-disc pl-4 pt-1">
                    {liveDossier.model_inference.triggered_reason_codes.map((rf: string, idx: number) => (
                      <li key={idx}>{rf}</li>
                    ))}
                  </ul>
                </div>

                <div className="p-3.5 rounded-lg bg-[var(--surface-elevated)] border border-[var(--border)] space-y-1">
                  <span className="text-[11px] font-mono font-bold uppercase text-cyan-600 dark:text-cyan-400">
                    3. Network Topology Findings
                  </span>
                  <div className="text-xs text-[var(--text-primary)] font-mono pt-1">
                    Risk Category: <strong className="text-[var(--text-primary)]">{liveDossier.risk_level}</strong>
                  </div>
                  <div className="text-[11px] text-[var(--text-secondary)]">
                    {liveDossier.graph_findings.topology_summary}
                  </div>
                </div>

                <div className="p-3.5 rounded-lg bg-[var(--surface-elevated)] border border-[var(--border)] space-y-1">
                  <span className="text-[11px] font-mono font-bold uppercase text-emerald-600 dark:text-emerald-400 flex items-center gap-1.5">
                    <CheckCircle2 className="w-3.5 h-3.5" />
                    4. Recommended Action Protocol
                  </span>
                  <div className="text-xs font-semibold text-[var(--text-primary)]">
                    {liveDossier.recommendation.action}
                  </div>
                  <div className="text-[11px] text-[var(--text-secondary)]">
                    Clear Protocol: {liveDossier.recommendation.required_evidence_to_clear}
                  </div>
                </div>
              </div>

              {/* RAG Policy Guidelines */}
              {liveDossier.retrieved_policy_guidelines.length > 0 && (
                <div className="p-3.5 rounded-lg bg-[var(--surface-elevated)] border border-[var(--border)] space-y-1.5">
                  <span className="text-[10px] font-mono font-bold uppercase text-[var(--text-secondary)] flex items-center gap-1">
                    <BookOpen className="w-3 h-3 text-blue-500" />
                    Retrieved Forensic RAG Policy Guidelines
                  </span>
                  <div className="space-y-1 text-[11px] text-[var(--text-primary)]">
                    {liveDossier.retrieved_policy_guidelines.map((pol, idx) => (
                      <div key={idx} className="bg-[var(--surface)] p-2 rounded border border-[var(--border)]">
                        {pol}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Chronological Evidence Timeline */}
          <div className="p-5 rounded-xl border border-[var(--border)] bg-[var(--surface)] space-y-4">
            <div className="text-xs font-bold uppercase tracking-wider text-[var(--text-primary)]">
              Chronological Evidence Sequence
            </div>

            <div className="relative pl-6 space-y-4 before:absolute before:left-2 before:top-2 before:bottom-2 before:w-0.5 before:bg-[var(--border)]">
              {selectedCase.timeline.map((step, idx) => (
                <div key={idx} className="relative flex items-start gap-3">
                  <span className="absolute -left-6 top-1 w-2.5 h-2.5 rounded-full bg-blue-500 ring-4 ring-[var(--surface)]" />
                  <div className="text-xs font-mono">
                    <span className="text-[var(--text-muted)] mr-2">[{step.time}]</span>
                    <span className="text-[var(--text-primary)]">{step.event}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Evidence Channel Breakdown Cards */}
          <div className="p-5 rounded-xl border border-[var(--border)] bg-[var(--surface)] space-y-4">
            <div className="text-xs font-bold uppercase tracking-wider text-[var(--text-primary)]">
              Channel-Specific Evidence Dossier
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
              {[
                { title: "1. Transaction Evidence", color: "text-blue-600 dark:text-blue-400", text: selectedCase.evidence.transaction },
                { title: "2. Graph Topology Evidence", color: "text-purple-600 dark:text-purple-400", text: selectedCase.evidence.graph },
                { title: "3. Behavioral Velocity", color: "text-amber-600 dark:text-amber-400", text: selectedCase.evidence.behavioral },
                { title: "4. Multimodal / CLIP Evidence", color: "text-cyan-600 dark:text-cyan-400", text: selectedCase.evidence.multimodal },
              ].map((ch) => (
                <div key={ch.title} className="p-3.5 rounded-lg bg-[var(--surface-elevated)] border border-[var(--border)]">
                  <div className={`text-[10px] font-mono font-bold uppercase ${ch.color}`}>{ch.title}</div>
                  <div className="text-[var(--text-secondary)] mt-1 leading-relaxed">{ch.text}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Section B: Human Investigator Notes & Case Tagging */}
          <div className="p-5 rounded-xl border border-[var(--border)] bg-[var(--surface)] space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-[var(--border)] pb-3">
              <div className="flex items-center gap-2">
                <MessageSquare className="w-4 h-4 text-blue-500" />
                <span className="text-xs font-bold uppercase tracking-wider text-[var(--text-primary)]">
                  Investigator Case Notes &amp; Audit Conclusions
                </span>
              </div>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-blue-500/10 text-blue-600 dark:text-blue-400 border border-blue-500/20">
                Saved locally in this browser
              </span>
            </div>

            {/* Note Entry Form */}
            <form onSubmit={handleAddNote} className="space-y-3">
              <div className="flex flex-col sm:flex-row gap-3">
                <div className="sm:w-1/3 space-y-1">
                  <label className="text-[11px] font-medium text-[var(--text-secondary)] flex items-center gap-1">
                    <Tag className="w-3 h-3" />
                    <span>Case Status Tag</span>
                  </label>
                  <select
                    value={selectedTag}
                    onChange={(e) => setSelectedTag(e.target.value)}
                    className="w-full text-xs p-2 rounded-lg border border-[var(--border)] bg-[var(--surface-elevated)] text-[var(--text-primary)] focus:outline-none"
                  >
                    <option value="Under Review">Under Review</option>
                    <option value="Confirmed Fraud Ring">Confirmed Fraud Ring</option>
                    <option value="False Positive (Cleared)">False Positive (Cleared)</option>
                    <option value="Awaiting ID Verification">Awaiting ID Verification</option>
                    <option value="Chargeback Disputed">Chargeback Disputed</option>
                  </select>
                </div>

                <div className="flex-1 space-y-1">
                  <label className="text-[11px] font-medium text-[var(--text-secondary)]">
                    Analyst Findings / Action Note
                  </label>
                  <input
                    type="text"
                    value={newNoteText}
                    onChange={(e) => setNewNoteText(e.target.value)}
                    placeholder="Enter case observations, telephone verification results, or dispute notes..."
                    className="w-full text-xs p-2 rounded-lg border border-[var(--border)] bg-[var(--surface-elevated)] text-[var(--text-primary)] placeholder-[var(--text-muted)] focus:outline-none"
                  />
                </div>
              </div>

              <div className="flex justify-end">
                <button
                  type="submit"
                  disabled={!newNoteText.trim()}
                  className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white font-semibold text-xs transition-colors cursor-pointer"
                >
                  <Plus className="w-3.5 h-3.5" />
                  <span>Add Case Note</span>
                </button>
              </div>
            </form>

            {/* Existing Notes List */}
            <div className="space-y-2 pt-2">
              {notes.length === 0 ? (
                <div className="p-4 text-center rounded-lg bg-[var(--surface-elevated)] border border-[var(--border)] text-xs text-[var(--text-muted)]">
                  No analyst notes added for {selectedCase.id} yet. Notes typed above will be saved locally in this browser.
                </div>
              ) : (
                notes.map((note, idx) => (
                  <div
                    key={idx}
                    className="p-3 rounded-lg bg-[var(--surface-elevated)] border border-[var(--border)] space-y-1 text-xs"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="font-semibold text-[var(--text-primary)]">{note.author}</span>
                        <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-blue-500/10 text-blue-500 border border-blue-500/20">
                          {note.tag}
                        </span>
                      </div>
                      <span className="text-[10px] font-mono text-[var(--text-muted)]">{note.timestamp}</span>
                    </div>
                    <p className="text-[var(--text-secondary)] leading-relaxed pl-1">{note.text}</p>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function InvestigationsPage() {
  return (
    <Suspense fallback={<div className="p-12 text-center text-xs font-mono text-[var(--text-muted)]">Loading Case Management...</div>}>
      <InvestigationsContent />
    </Suspense>
  );
}
