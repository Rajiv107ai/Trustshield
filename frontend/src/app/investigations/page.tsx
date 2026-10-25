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
  Bot,
  ShieldCheck,
  Key,
  Copy,
  Check,
  Activity,
  Scale,
  FileText,
  AlertTriangle,
  Cpu,
  Terminal,
  ArrowRight,
} from "lucide-react";
import { TrustShieldApi, ApiError } from "@/lib/api/client";
import {
  DossierResponse,
  CopilotInvestigationResponse,
  VerifyBriefResponse,
} from "@/lib/types/api";

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

  // Tab State: "copilot" | "timeline" | "notes"
  const [activeTab, setActiveTab] = useState<"copilot" | "timeline" | "notes">("copilot");

  // Autonomous Multi-Agent Copilot State
  const [targetEntityId, setTargetEntityId] = useState<string>(selectedCase.primaryEntity);
  const [targetEntityType, setTargetEntityType] = useState<string>("transaction");
  const [copilotPrompt, setCopilotPrompt] = useState<string>("");
  const [includeDisputeBrief, setIncludeDisputeBrief] = useState<boolean>(true);
  const [isCopilotRunning, setIsCopilotRunning] = useState<boolean>(false);
  const [copilotResponse, setCopilotResponse] = useState<CopilotInvestigationResponse | null>(null);
  const [copilotError, setCopilotError] = useState<string | null>(null);

  // Cryptographic Dispute Brief Verification State
  const [isVerifying, setIsVerifying] = useState<boolean>(false);
  const [verifyResult, setVerifyResult] = useState<VerifyBriefResponse | null>(null);
  const [verifyError, setVerifyError] = useState<string | null>(null);
  const [copiedBrief, setCopiedBrief] = useState<boolean>(false);
  const [copiedDigest, setCopiedDigest] = useState<boolean>(false);
  const [copiedSig, setCopiedSig] = useState<boolean>(false);
  const [showFullBriefMarkdown, setShowFullBriefMarkdown] = useState<boolean>(false);

  // Legacy Dossier state (for deterministic audit pack)
  const [liveDossier, setLiveDossier] = useState<DossierResponse | null>(null);
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [errorState, setErrorState] = useState<{ message: string; isAuth?: boolean; status?: number } | null>(null);

  // Investigator Local Notes (Honest Browser Storage)
  const [notes, setNotes] = useState<{ text: string; author: string; timestamp: string; tag: string }[]>([]);
  const [newNoteText, setNewNoteText] = useState("");
  const [selectedTag, setSelectedTag] = useState("Under Review");

  // Sync targetEntityId when selected case changes
  useEffect(() => {
    queueMicrotask(() => {
      setTargetEntityId(selectedCase.primaryEntity);
      setCopilotResponse(null);
      setCopilotError(null);
      setVerifyResult(null);
      setVerifyError(null);
      setShowFullBriefMarkdown(false);
    });
  }, [selectedCase.id, selectedCase.primaryEntity]);

  useEffect(() => {
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

  const handleLaunchCopilot = async () => {
    setIsCopilotRunning(true);
    setCopilotError(null);
    setVerifyResult(null);
    setVerifyError(null);

    try {
      const res = await TrustShieldApi.investigateWithCopilot({
        entity_type: targetEntityType,
        entity_id: targetEntityId.trim() || selectedCase.primaryEntity,
        prompt: copilotPrompt.trim() || undefined,
        include_dispute_brief: includeDisputeBrief,
      });
      setCopilotResponse(res.data);
    } catch (err: unknown) {
      console.error("Copilot investigation failed:", err);
      const message = err instanceof Error ? err.message : String(err);
      setCopilotError(message);
    } finally {
      setIsCopilotRunning(false);
    }
  };

  const handleVerifyBrief = async () => {
    if (!copilotResponse?.dispute_brief) return;
    setIsVerifying(true);
    setVerifyError(null);
    try {
      const res = await TrustShieldApi.verifyDisputeBrief({
        brief_id: copilotResponse.dispute_brief.brief_id,
        sha256_evidence_digest: copilotResponse.dispute_brief.sha256_evidence_digest,
        hmac_signature: copilotResponse.dispute_brief.hmac_signature,
      });
      setVerifyResult(res.data);
    } catch (err: unknown) {
      console.error("Brief verification failed:", err);
      const msg = err instanceof Error ? err.message : String(err);
      setVerifyError(msg);
    } finally {
      setIsVerifying(false);
    }
  };

  const handleCopyMarkdown = () => {
    if (!copilotResponse?.dispute_brief?.markdown_content) return;
    navigator.clipboard.writeText(copilotResponse.dispute_brief.markdown_content);
    setCopiedBrief(true);
    setTimeout(() => setCopiedBrief(false), 2500);
  };

  const handleCopyDigest = () => {
    if (!copilotResponse?.dispute_brief?.sha256_evidence_digest) return;
    navigator.clipboard.writeText(copilotResponse.dispute_brief.sha256_evidence_digest);
    setCopiedDigest(true);
    setTimeout(() => setCopiedDigest(false), 2000);
  };

  const handleCopySig = () => {
    if (!copilotResponse?.dispute_brief?.hmac_signature) return;
    navigator.clipboard.writeText(copilotResponse.dispute_brief.hmac_signature);
    setCopiedSig(true);
    setTimeout(() => setCopiedSig(false), 2000);
  };

  const handleDownloadBriefJson = () => {
    if (!copilotResponse?.dispute_brief) return;
    const blob = new Blob([JSON.stringify(copilotResponse.dispute_brief, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `trustshield_signed_brief_${copilotResponse.dispute_brief.brief_id}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
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
      platform: "TrustShield AI Autonomous Investigation Console",
      exported_at: new Date().toISOString(),
      case_id: selectedCase.id,
      title: selectedCase.title,
      severity: selectedCase.severity,
      target_entity: selectedCase.primaryEntity,
      copilot_response: copilotResponse,
      live_dossier: liveDossier,
      analyst_notes_local: notes,
      status: "AUTHENTICATED_FORENSIC_TRAIL",
    };
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `trustshield_case_${selectedCase.id}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      {/* Top Header Bar */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-[var(--text-primary)] flex items-center gap-2">
            <FileSearch className="w-5 h-5 text-blue-600 dark:text-blue-400" />
            <span>Autonomous Fraud Investigation &amp; Case Forensics</span>
          </h1>
          <p className="text-xs text-[var(--text-secondary)] mt-1">
            Autonomous multi-agent fraud investigation with real-time graph neighbor traversal, sliding-window velocity profiling, and cryptographically signed dispute briefs.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={handleExportJson}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-[var(--border)] bg-[var(--surface)] hover:bg-[var(--surface-elevated)] text-xs font-medium text-[var(--text-primary)] transition-colors focus-ring cursor-pointer"
            title="Download full JSON evidence report"
          >
            <Download className="w-3.5 h-3.5 text-blue-500" />
            <span>Export Case JSON</span>
          </button>

          <button
            onClick={handlePrintDossier}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-[var(--border)] bg-[var(--surface)] hover:bg-[var(--surface-elevated)] text-xs font-medium text-[var(--text-primary)] transition-colors focus-ring cursor-pointer"
            title="Print printable evidence dossier"
          >
            <Printer className="w-3.5 h-3.5 text-[var(--text-secondary)]" />
            <span>Print Report</span>
          </button>

          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[var(--surface)] border border-[var(--border)] text-xs font-mono">
            <Bot className="w-3.5 h-3.5 text-indigo-500" />
            <span className="text-[var(--text-muted)]">Copilot:</span>
            <span className="text-emerald-600 dark:text-emerald-400 font-semibold">Autonomous v2.4</span>
          </div>
        </div>
      </div>

      {/* Main Grid: Left Cases List, Right Deep Copilot / Dossier View */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Case Selector */}
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
                    setErrorState(null);
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

          {/* Quick Info Box */}
          <div className="mt-4 p-3 rounded-lg bg-[var(--surface-elevated)] border border-[var(--border)] text-[11px] text-[var(--text-secondary)] space-y-1">
            <div className="font-semibold text-[var(--text-primary)] flex items-center gap-1.5">
              <ShieldCheck className="w-3.5 h-3.5 text-blue-500" />
              <span>Multi-Agent Capabilities</span>
            </div>
            <p className="leading-relaxed">
              Equipped with live tools: <code>inspect_graph_neighbors</code>, <code>fetch_realtime_feature_profile</code>, <code>query_action_history</code>, <code>retrieve_fraud_policy</code>, and <code>generate_cryptographic_dispute_brief</code>.
            </p>
          </div>
        </div>

        {/* Right 2 Columns: Multi-Agent Copilot & Evidence Console */}
        <div className="lg:col-span-2 space-y-6">
          {/* Case Header Card */}
          <div className="p-5 rounded-xl border border-[var(--border)] bg-[var(--surface)] shadow-sm space-y-3">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
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
                  <span>·</span>
                  <span>Assigned: {selectedCase.assignedTo}</span>
                </div>
              </div>

              {/* View Switcher Tabs */}
              <div className="flex items-center gap-1 p-1 bg-[var(--surface-elevated)] border border-[var(--border)] rounded-lg">
                <button
                  onClick={() => setActiveTab("copilot")}
                  className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all flex items-center gap-1.5 cursor-pointer ${
                    activeTab === "copilot"
                      ? "bg-blue-600 text-white shadow-sm font-semibold"
                      : "text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
                  }`}
                >
                  <Bot className="w-3.5 h-3.5" />
                  <span>Agent Copilot</span>
                </button>

                <button
                  onClick={() => setActiveTab("timeline")}
                  className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all flex items-center gap-1.5 cursor-pointer ${
                    activeTab === "timeline"
                      ? "bg-blue-600 text-white shadow-sm font-semibold"
                      : "text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
                  }`}
                >
                  <Activity className="w-3.5 h-3.5" />
                  <span>Evidence Channels</span>
                </button>

                <button
                  onClick={() => setActiveTab("notes")}
                  className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all flex items-center gap-1.5 cursor-pointer ${
                    activeTab === "notes"
                      ? "bg-blue-600 text-white shadow-sm font-semibold"
                      : "text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
                  }`}
                >
                  <MessageSquare className="w-3.5 h-3.5" />
                  <span>Notes ({notes.length})</span>
                </button>
              </div>
            </div>

            <p className="text-xs text-[var(--text-primary)] leading-relaxed bg-[var(--surface-elevated)] p-3 rounded-lg border border-[var(--border)]">
              {selectedCase.summary}
            </p>
          </div>

          {/* TAB 1: AUTONOMOUS MULTI-AGENT COPILOT VIEW */}
          {activeTab === "copilot" && (
            <div className="space-y-6">
              {/* Copilot Launcher Control Bar */}
              <div className="p-5 rounded-xl border border-blue-500/30 bg-gradient-to-br from-[var(--surface)] to-blue-500/5 shadow-sm space-y-4">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-[var(--border)] pb-3">
                  <div className="flex items-center gap-2">
                    <Sparkles className="w-4 h-4 text-blue-500" />
                    <span className="text-xs font-bold uppercase tracking-wider text-[var(--text-primary)]">
                      Autonomous Fraud Copilot Multi-Agent Orchestrator
                    </span>
                  </div>
                  <div className="flex items-center gap-2 text-[10px] font-mono">
                    <span className="px-2 py-0.5 rounded bg-blue-500/10 text-blue-600 dark:text-blue-400 border border-blue-500/20 font-semibold">
                      5 Live Specialist Tools
                    </span>
                    <span className="px-2 py-0.5 rounded bg-indigo-500/10 text-indigo-600 dark:text-indigo-400 border border-indigo-500/20 font-semibold">
                      SHA-256 Signed Briefs
                    </span>
                  </div>
                </div>

                {/* Form Controls */}
                <div className="grid grid-cols-1 sm:grid-cols-4 gap-3">
                  <div className="space-y-1">
                    <label className="text-[10px] font-mono uppercase text-[var(--text-secondary)] font-semibold">
                      Target Entity Type
                    </label>
                    <select
                      value={targetEntityType}
                      onChange={(e) => setTargetEntityType(e.target.value)}
                      className="w-full text-xs p-2 rounded-lg border border-[var(--border)] bg-[var(--surface-elevated)] text-[var(--text-primary)] focus:outline-none focus:ring-1 focus:ring-blue-500"
                    >
                      <option value="transaction">transaction</option>
                      <option value="buyer">buyer</option>
                      <option value="seller">seller</option>
                      <option value="ring">ring</option>
                    </select>
                  </div>

                  <div className="sm:col-span-3 space-y-1">
                    <label className="text-[10px] font-mono uppercase text-[var(--text-secondary)] font-semibold">
                      Target ID (Case Primary Entity)
                    </label>
                    <input
                      type="text"
                      value={targetEntityId}
                      onChange={(e) => setTargetEntityId(e.target.value)}
                      placeholder="e.g. ORD_78901, BUYER_00000, DEV_FARM_99"
                      className="w-full text-xs p-2 rounded-lg border border-[var(--border)] bg-[var(--surface-elevated)] text-[var(--text-primary)] font-mono focus:outline-none focus:ring-1 focus:ring-blue-500"
                    />
                  </div>
                </div>

                <div className="space-y-1">
                  <label className="text-[10px] font-mono uppercase text-[var(--text-secondary)] font-semibold flex items-center justify-between">
                    <span>Investigator Directive / Optional Prompt</span>
                    <span className="text-[10px] text-[var(--text-muted)] font-normal">Leave blank for full autonomous multi-channel analysis</span>
                  </label>
                  <input
                    type="text"
                    value={copilotPrompt}
                    onChange={(e) => setCopilotPrompt(e.target.value)}
                    placeholder="e.g. Inspect graph neighbors for ring collusion, calculate feature velocity, and generate chargeback defense brief..."
                    className="w-full text-xs p-2.5 rounded-lg border border-[var(--border)] bg-[var(--surface-elevated)] text-[var(--text-primary)] placeholder-[var(--text-muted)] focus:outline-none focus:ring-1 focus:ring-blue-500"
                  />
                </div>

                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pt-1">
                  <label className="flex items-center gap-2 cursor-pointer text-xs text-[var(--text-primary)]">
                    <input
                      type="checkbox"
                      checked={includeDisputeBrief}
                      onChange={(e) => setIncludeDisputeBrief(e.target.checked)}
                      className="rounded border-[var(--border)] text-blue-600 focus:ring-blue-500 cursor-pointer"
                    />
                    <span className="font-medium">Generate Cryptographic Legal Dispute Brief (HMAC-SHA256 Signed)</span>
                  </label>

                  <button
                    onClick={handleLaunchCopilot}
                    disabled={isCopilotRunning || !targetEntityId.trim()}
                    className="inline-flex items-center justify-center gap-2 px-5 py-2.5 rounded-lg bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 text-white font-semibold text-xs transition-all shadow-md shadow-blue-500/20 disabled:opacity-50 cursor-pointer focus-ring"
                  >
                    {isCopilotRunning ? (
                      <>
                        <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                        <span>Orchestrating Autonomous Agents...</span>
                      </>
                    ) : (
                      <>
                        <Sparkles className="w-3.5 h-3.5 text-blue-200" />
                        <span>Launch Autonomous Investigation</span>
                      </>
                    )}
                  </button>
                </div>
              </div>

              {/* Loading State Spinner & Agent Status */}
              {isCopilotRunning && (
                <div className="p-8 rounded-xl border border-blue-500/30 bg-[var(--surface)] text-center space-y-4">
                  <div className="inline-flex p-3 rounded-full bg-blue-500/10 text-blue-600 dark:text-blue-400 animate-pulse">
                    <Cpu className="w-8 h-8 animate-spin" />
                  </div>
                  <div className="space-y-1">
                    <h3 className="text-sm font-bold text-[var(--text-primary)]">
                      Autonomous Multi-Agent Copilot in Execution
                    </h3>
                    <p className="text-xs text-[var(--text-secondary)] max-w-md mx-auto">
                      Executing specialized forensic tools across graph topology, point-in-time velocity feature service, outbox records, policy guidelines, and legal cryptographic synthesizers...
                    </p>
                  </div>
                  <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 max-w-2xl mx-auto pt-2 text-[10px] font-mono">
                    <span className="p-1.5 rounded bg-[var(--surface-elevated)] border border-[var(--border)] text-blue-500 animate-pulse">
                      1. GraphForensics
                    </span>
                    <span className="p-1.5 rounded bg-[var(--surface-elevated)] border border-[var(--border)] text-indigo-500 animate-pulse">
                      2. FeatureVelocity
                    </span>
                    <span className="p-1.5 rounded bg-[var(--surface-elevated)] border border-[var(--border)] text-amber-500 animate-pulse">
                      3. OutboxAuditor
                    </span>
                    <span className="p-1.5 rounded bg-[var(--surface-elevated)] border border-[var(--border)] text-purple-500 animate-pulse">
                      4. PolicyRAG
                    </span>
                    <span className="p-1.5 rounded bg-[var(--surface-elevated)] border border-[var(--border)] text-emerald-500 animate-pulse">
                      5. DisputeSigner
                    </span>
                  </div>
                </div>
              )}

              {/* Error State */}
              {copilotError && (
                <div className="p-4 rounded-xl border border-rose-500/30 bg-rose-500/10 space-y-2">
                  <div className="flex items-center gap-2 text-rose-600 dark:text-rose-400 font-bold text-xs">
                    <AlertTriangle className="w-4 h-4" />
                    <span>Copilot Analysis Failed</span>
                  </div>
                  <p className="text-xs text-[var(--text-primary)] font-mono">{copilotError}</p>
                </div>
              )}

              {/* COPILOT RESULTS PRESENTATION */}
              {copilotResponse && !isCopilotRunning && (
                <div className="space-y-6">
                  {/* 1. Primary Fraud Hypothesis & Executive Narrative */}
                  <div className="p-6 rounded-xl border border-blue-500/40 bg-[var(--surface)] shadow-md space-y-4">
                    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[var(--border)] pb-3">
                      <div className="flex items-center gap-2">
                        <Bot className="w-5 h-5 text-blue-500" />
                        <div>
                          <div className="text-xs font-mono font-bold uppercase tracking-wider text-[var(--text-muted)]">
                            Primary Autonomous Hypothesis
                          </div>
                          <div className="text-sm sm:text-base font-bold text-[var(--text-primary)]">
                            {copilotResponse.primary_fraud_hypothesis}
                          </div>
                        </div>
                      </div>

                      <div className="flex items-center gap-2">
                        <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-blue-500/10 text-blue-600 dark:text-blue-400 border border-blue-500/20 font-bold">
                          Status: {copilotResponse.status}
                        </span>
                        <span
                          className={`text-xs font-mono font-bold px-2.5 py-1 rounded-full border ${
                            copilotResponse.confidence_score >= 0.8
                              ? "bg-rose-500/10 text-rose-600 dark:text-rose-400 border-rose-500/20"
                              : copilotResponse.confidence_score >= 0.5
                              ? "bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/20"
                              : "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20"
                          }`}
                        >
                          {(copilotResponse.confidence_score * 100).toFixed(0)}% Confidence
                        </span>
                      </div>
                    </div>

                    {/* Copilot Narrative */}
                    <div className="space-y-2">
                      <div className="text-xs font-semibold text-[var(--text-primary)] flex items-center gap-1.5">
                        <FileText className="w-3.5 h-3.5 text-blue-500" />
                        <span>Executive Forensic Narrative</span>
                      </div>
                      <p className="text-xs text-[var(--text-primary)] leading-relaxed bg-[var(--surface-elevated)] p-4 rounded-lg border border-[var(--border)] font-normal">
                        {copilotResponse.copilot_narrative}
                      </p>
                    </div>

                    {/* Counterfactual Defense Argument */}
                    <div className="p-4 rounded-lg bg-indigo-500/5 border border-indigo-500/20 space-y-1.5">
                      <div className="text-xs font-semibold text-indigo-600 dark:text-indigo-400 flex items-center gap-1.5">
                        <Scale className="w-3.5 h-3.5" />
                        <span>Counterfactual Defense &amp; Benign Alternative Rejection</span>
                      </div>
                      <p className="text-xs text-[var(--text-secondary)] leading-relaxed">
                        {copilotResponse.counterfactual_defense}
                      </p>
                    </div>

                    {/* Pre-populated Recommended Action */}
                    {copilotResponse.recommended_action && (
                      <div className="p-3.5 rounded-lg bg-[var(--surface-elevated)] border border-[var(--border)] flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                        <div className="space-y-0.5">
                          <div className="text-[10px] font-mono uppercase text-[var(--text-muted)] font-bold">
                            Recommended Immediate Outbox Action
                          </div>
                          <div className="text-xs font-semibold text-[var(--text-primary)] flex items-center gap-2">
                            <span className="font-mono text-blue-600 dark:text-blue-400">
                              {copilotResponse.recommended_action.action_type}
                            </span>
                            <ArrowRight className="w-3 h-3 text-[var(--text-muted)]" />
                            <span className="font-mono">
                              {copilotResponse.recommended_action.target_type}: {copilotResponse.recommended_action.target_id}
                            </span>
                          </div>
                          <div className="text-[11px] text-[var(--text-secondary)]">
                            {copilotResponse.recommended_action.reason}
                          </div>
                        </div>

                        <span className="inline-flex items-center gap-1 px-3 py-1.5 rounded bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20 text-xs font-mono font-semibold self-start sm:self-auto">
                          <CheckCircle2 className="w-3 h-3" />
                          <span>Action Ready</span>
                        </span>
                      </div>
                    )}
                  </div>

                  {/* 2. Multi-Agent Tool Execution Trace */}
                  <div className="p-5 rounded-xl border border-[var(--border)] bg-[var(--surface)] space-y-3">
                    <div className="flex items-center justify-between border-b border-[var(--border)] pb-2.5">
                      <div className="flex items-center gap-2">
                        <Terminal className="w-4 h-4 text-indigo-500" />
                        <span className="text-xs font-bold uppercase tracking-wider text-[var(--text-primary)]">
                          Specialist Tool Execution Trace ({copilotResponse.tool_executions.length} Invocations)
                        </span>
                      </div>
                      <span className="text-[10px] font-mono text-[var(--text-muted)]">
                        Real-Time Causal Data Boundary
                      </span>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      {copilotResponse.tool_executions.map((exec, idx) => (
                        <div
                          key={idx}
                          className="p-3.5 rounded-lg bg-[var(--surface-elevated)] border border-[var(--border)] space-y-2 text-xs"
                        >
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-1.5">
                              <span className="w-2 h-2 rounded-full bg-emerald-500" />
                              <span className="font-bold text-[var(--text-primary)]">{exec.agent}</span>
                            </div>
                            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-[var(--surface)] border border-[var(--border)] text-[var(--text-muted)]">
                              {exec.latency_ms.toFixed(1)} ms
                            </span>
                          </div>

                          <div className="font-mono text-[11px] text-blue-600 dark:text-blue-400 font-semibold">
                            {exec.tool}()
                          </div>

                          <p className="text-[11px] text-[var(--text-secondary)] leading-relaxed">
                            {exec.summary}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* 3. Cryptographically Signed Dispute & Chargeback Defense Brief */}
                  {copilotResponse.dispute_brief && (
                    <div className="p-6 rounded-xl border border-emerald-500/40 bg-gradient-to-b from-[var(--surface)] to-emerald-500/5 shadow-md space-y-5">
                      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[var(--border)] pb-3">
                        <div className="flex items-center gap-2.5">
                          <div className="p-2 rounded-lg bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20">
                            <ShieldCheck className="w-5 h-5" />
                          </div>
                          <div>
                            <div className="text-xs font-mono font-bold uppercase tracking-wider text-[var(--text-muted)]">
                              Tamper-Evident Dispute Defense
                            </div>
                            <div className="text-sm font-bold text-[var(--text-primary)]">
                              Cryptographic Chargeback Defense Brief ({copilotResponse.dispute_brief.brief_id})
                            </div>
                          </div>
                        </div>

                        <div className="flex items-center gap-2">
                          <span
                            className={`text-xs font-mono font-bold px-2.5 py-1 rounded-full uppercase border ${
                              copilotResponse.dispute_brief.chargeback_defense_viability === "STRONG"
                                ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20"
                                : copilotResponse.dispute_brief.chargeback_defense_viability === "MODERATE"
                                ? "bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/20"
                                : "bg-rose-500/10 text-rose-600 dark:text-rose-400 border-rose-500/20"
                            }`}
                          >
                            Viability: {copilotResponse.dispute_brief.chargeback_defense_viability}
                          </span>
                        </div>
                      </div>

                      {/* Cryptographic Provenance Bar */}
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs font-mono">
                        <div className="p-3 rounded-lg bg-[var(--surface-elevated)] border border-[var(--border)] space-y-1">
                          <div className="flex items-center justify-between text-[10px] text-[var(--text-muted)] font-bold">
                            <span className="flex items-center gap-1">
                              <Key className="w-3 h-3 text-blue-500" />
                              SHA-256 Evidence Digest
                            </span>
                            <button
                              onClick={handleCopyDigest}
                              className="text-blue-500 hover:underline flex items-center gap-0.5 cursor-pointer"
                            >
                              {copiedDigest ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
                              <span>{copiedDigest ? "Copied" : "Copy"}</span>
                            </button>
                          </div>
                          <div className="truncate text-[11px] text-[var(--text-primary)] font-semibold select-all" title={copilotResponse.dispute_brief.sha256_evidence_digest}>
                            {copilotResponse.dispute_brief.sha256_evidence_digest}
                          </div>
                        </div>

                        <div className="p-3 rounded-lg bg-[var(--surface-elevated)] border border-[var(--border)] space-y-1">
                          <div className="flex items-center justify-between text-[10px] text-[var(--text-muted)] font-bold">
                            <span className="flex items-center gap-1">
                              <ShieldCheck className="w-3 h-3 text-emerald-500" />
                              HMAC-SHA256 Signature
                            </span>
                            <button
                              onClick={handleCopySig}
                              className="text-blue-500 hover:underline flex items-center gap-0.5 cursor-pointer"
                            >
                              {copiedSig ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
                              <span>{copiedSig ? "Copied" : "Copy"}</span>
                            </button>
                          </div>
                          <div className="truncate text-[11px] text-[var(--text-primary)] font-semibold select-all" title={copilotResponse.dispute_brief.hmac_signature}>
                            {copilotResponse.dispute_brief.hmac_signature}
                          </div>
                        </div>
                      </div>

                      {/* Compelling Evidence Points List */}
                      <div className="space-y-2">
                        <div className="text-xs font-bold text-[var(--text-primary)] uppercase tracking-wider flex items-center gap-1.5">
                          <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500" />
                          <span>Compelling Evidence Points (Visa CE 3.0 / Mastercard Defense)</span>
                        </div>
                        <ul className="text-xs text-[var(--text-primary)] space-y-1.5 list-disc pl-5 bg-[var(--surface-elevated)] p-4 rounded-lg border border-[var(--border)]">
                          {copilotResponse.dispute_brief.compelling_evidence_points.map((point, idx) => (
                            <li key={idx} className="leading-relaxed">
                              {point}
                            </li>
                          ))}
                        </ul>
                      </div>

                      {/* 1-Click Verification Section */}
                      <div className="p-4 rounded-lg bg-[var(--surface-elevated)] border border-[var(--border)] space-y-3">
                        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                          <div className="space-y-0.5">
                            <div className="text-xs font-bold text-[var(--text-primary)] flex items-center gap-1.5">
                              <ShieldCheck className="w-4 h-4 text-emerald-500" />
                              <span>Cryptographic Integrity Verification</span>
                            </div>
                            <p className="text-[11px] text-[var(--text-secondary)]">
                              Validate the evidence digest and HMAC signature against the server key using constant-time verification.
                            </p>
                          </div>

                          <button
                            onClick={handleVerifyBrief}
                            disabled={isVerifying}
                            className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-mono font-semibold text-xs transition-all shadow-sm disabled:opacity-50 cursor-pointer self-start sm:self-auto focus-ring"
                          >
                            {isVerifying ? (
                              <>
                                <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                                <span>Verifying...</span>
                              </>
                            ) : (
                              <>
                                <ShieldCheck className="w-3.5 h-3.5" />
                                <span>Verify Authenticity</span>
                              </>
                            )}
                          </button>
                        </div>

                        {verifyResult && (
                          <div
                            className={`p-3 rounded-lg border text-xs font-mono space-y-1 ${
                              verifyResult.is_valid
                                ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-700 dark:text-emerald-300"
                                : "bg-rose-500/10 border-rose-500/30 text-rose-700 dark:text-rose-300"
                            }`}
                          >
                            <div className="flex items-center gap-2 font-bold">
                              {verifyResult.is_valid ? (
                                <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                              ) : (
                                <AlertTriangle className="w-4 h-4 text-rose-500" />
                              )}
                              <span>
                                {verifyResult.is_valid
                                  ? "VALIDATION PASSED: Tamper-Evident Authenticity Confirmed"
                                  : "VALIDATION FAILED: Tampering Detected"}
                              </span>
                            </div>
                            <div className="text-[11px] pl-6 space-y-0.5">
                              <div>Digest Match: {verifyResult.digest_matches ? "TRUE" : "FALSE"}</div>
                              <div>Signature Match: {verifyResult.signature_matches ? "TRUE" : "FALSE"}</div>
                              <div>Verified At: {verifyResult.verification_timestamp}</div>
                              <div>Notes: {verifyResult.notes}</div>
                            </div>
                          </div>
                        )}

                        {verifyError && (
                          <div className="p-3 rounded-lg border border-rose-500/30 bg-rose-500/10 text-xs font-mono text-rose-600">
                            Verification error: {verifyError}
                          </div>
                        )}
                      </div>

                      {/* Action Buttons for Brief */}
                      <div className="flex flex-wrap items-center gap-2 pt-1">
                        <button
                          onClick={handleCopyMarkdown}
                          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-[var(--border)] bg-[var(--surface-elevated)] hover:bg-[var(--surface)] text-xs font-medium text-[var(--text-primary)] transition-colors cursor-pointer"
                        >
                          {copiedBrief ? <Check className="w-3.5 h-3.5 text-emerald-500" /> : <Copy className="w-3.5 h-3.5 text-blue-500" />}
                          <span>{copiedBrief ? "Copied Legal Markdown!" : "Copy Legal Brief (Markdown)"}</span>
                        </button>

                        <button
                          onClick={handleDownloadBriefJson}
                          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-[var(--border)] bg-[var(--surface-elevated)] hover:bg-[var(--surface)] text-xs font-medium text-[var(--text-primary)] transition-colors cursor-pointer"
                        >
                          <Download className="w-3.5 h-3.5 text-emerald-500" />
                          <span>Download Signed Brief JSON</span>
                        </button>

                        <button
                          onClick={() => setShowFullBriefMarkdown(!showFullBriefMarkdown)}
                          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-[var(--border)] bg-[var(--surface-elevated)] hover:bg-[var(--surface)] text-xs font-medium text-[var(--text-primary)] transition-colors cursor-pointer"
                        >
                          <FileText className="w-3.5 h-3.5 text-indigo-500" />
                          <span>{showFullBriefMarkdown ? "Hide Legal Brief Preview" : "Preview Full Legal Brief"}</span>
                        </button>
                      </div>

                      {/* Full Markdown Preview Collapsible */}
                      {showFullBriefMarkdown && (
                        <div className="p-4 rounded-lg bg-[var(--surface)] border border-[var(--border)] text-xs font-mono leading-relaxed space-y-2 whitespace-pre-wrap select-all max-h-96 overflow-y-auto">
                          {copilotResponse.dispute_brief.markdown_content}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}

              {/* Default Welcome / Instructions banner if no investigation run yet */}
              {!copilotResponse && !isCopilotRunning && !copilotError && (
                <div className="p-8 rounded-xl border border-[var(--border)] bg-[var(--surface)] text-center space-y-3">
                  <div className="inline-flex p-3 rounded-full bg-blue-500/10 text-blue-600 dark:text-blue-400">
                    <Bot className="w-8 h-8" />
                  </div>
                  <h3 className="text-sm font-bold text-[var(--text-primary)]">
                    Autonomous Multi-Agent Copilot Ready
                  </h3>
                  <p className="text-xs text-[var(--text-secondary)] max-w-md mx-auto">
                    Click <strong>Launch Autonomous Investigation</strong> above to deploy the 5 specialist agents across graph neighbors, causal velocity profiles, action outbox, and dispute brief synthesis.
                  </p>
                </div>
              )}
            </div>
          )}

          {/* TAB 2: CHRONOLOGICAL EVIDENCE & TIMELINE VIEW */}
          {activeTab === "timeline" && (
            <div className="space-y-6">
              {/* Telemetry Synthesis Header */}
              <div className="p-4 rounded-xl border border-[var(--border)] bg-[var(--surface)] flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <div className="space-y-0.5">
                  <div className="text-xs font-bold text-[var(--text-primary)] uppercase tracking-wider">
                    Deterministic Audit Dossier
                  </div>
                  <p className="text-xs text-[var(--text-secondary)]">
                    Synthesize deterministic telemetry across observed facts and triggered model reason codes.
                  </p>
                </div>

                <button
                  onClick={handleSynthesizeDossier}
                  disabled={isGenerating}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white text-xs font-mono font-semibold transition-all shadow-md shadow-blue-500/20 disabled:opacity-50 cursor-pointer self-start sm:self-auto focus-ring"
                >
                  {isGenerating ? (
                    <>
                      <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                      <span>Synthesizing...</span>
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-3.5 h-3.5 text-blue-200" />
                      <span>Synthesize Deterministic Pack</span>
                    </>
                  )}
                </button>
              </div>

              {errorState && (
                <div className="p-4 rounded-xl border border-rose-500/30 bg-rose-500/10 space-y-2">
                  <div className="flex items-center gap-2 text-rose-600 dark:text-rose-400 font-bold text-xs">
                    <ShieldAlert className="w-4 h-4" />
                    <span>Dossier Synthesis Failed</span>
                  </div>
                  <p className="text-xs text-[var(--text-primary)] font-mono">{errorState.message}</p>
                </div>
              )}

              {/* Live Synthesized Findings Section */}
              {liveDossier && (
                <div className="p-5 rounded-xl border border-blue-500/30 bg-[var(--surface)] space-y-4">
                  <div className="flex items-center justify-between border-b border-[var(--border)] pb-3">
                    <div className="flex items-center gap-2">
                      <Sparkles className="w-4 h-4 text-blue-500" />
                      <span className="text-xs font-bold uppercase tracking-wider text-[var(--text-primary)]">
                        Live Telemetry Synthesis (Deterministic Audit Pack)
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      {/* BEGIN GENERATED */}
                      {liveDossier.model_inference?.risk_interval && (
                        <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-blue-500/10 text-blue-700 dark:text-blue-300 border border-blue-500/20 font-semibold" title="95% Conformal Prediction Bounded Interval">
                          95% Interval: [{(liveDossier.model_inference.risk_interval[0] * 100).toFixed(1)}%, {(liveDossier.model_inference.risk_interval[1] * 100).toFixed(1)}%]
                        </span>
                      )}
                      {/* END GENERATED */}
                      <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-blue-500/10 text-blue-600 dark:text-blue-400 border border-blue-500/20 font-bold">
                        Decision: {liveDossier.decision}
                      </span>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {liveDossier.model_inference?.component_risks && (
                      <div className="col-span-1 md:col-span-2 p-3.5 rounded-lg bg-[var(--surface-elevated)] border border-[var(--border)] space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-[11px] font-mono font-bold uppercase text-purple-600 dark:text-purple-400 flex items-center gap-1.5">
                            <Activity className="w-3.5 h-3.5" />
                            Multi-Domain Bayesian Fusion Breakdown
                          </span>
                          {liveDossier.model_inference.conformal_prediction_set && (
                            <span className="text-[10px] font-mono text-[var(--text-secondary)]">
                              Conformal Set: {liveDossier.model_inference.conformal_prediction_set}
                            </span>
                          )}
                        </div>
                        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-1 font-mono text-xs">
                          {Object.entries(liveDossier.model_inference.component_risks).map(([domain, score]) => {
                            const val = typeof score === "number" ? score : parseFloat(String(score)) || 0;
                            const pct = Math.min(100, Math.max(0, val * 100));
                            const label = {
                              tabular_risk: "Tabular",
                              graph_risk_heuristic: "Graph Topology",
                              ring_risk_heuristic: "Collusion Ring",
                              multimodal_risk: "Multimodal",
                            }[domain] || domain.replace(/_/g, " ");

                            return (
                              <div key={domain} className="bg-[var(--surface)] p-2 rounded border border-[var(--border)]">
                                <div className="text-[10px] text-[var(--text-secondary)] uppercase truncate">{label}</div>
                                <div className="text-sm font-bold text-[var(--text-primary)] mt-0.5">{pct.toFixed(1)}%</div>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    )}

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

              {/* Chronological Evidence Sequence */}
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
            </div>
          )}

          {/* TAB 3: HUMAN INVESTIGATOR NOTES & CASE TAGGING */}
          {activeTab === "notes" && (
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
          )}
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
