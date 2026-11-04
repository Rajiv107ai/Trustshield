"use client";

import React, { useState } from "react";
import Link from "next/link";
import {
  ShieldAlert,
  Network,
  Scale,
  FileText,
  ArrowRight,
  Layers,
  HelpCircle,
  Sparkles,
} from "lucide-react";
import { useGuide } from "@/context/GuideContext";
import { DecisionBadge } from "@/components/ui/DecisionBadge";

interface InteractiveScenario {
  id: string;
  name: string;
  category: string;
  badge: "ALLOW" | "REVIEW" | "HOLD" | "BLOCK";
  riskScore: number;
  customerSummary: string;
  plainExplanation: string;
  keyDrivers: string[];
  actionLabel: string;
  targetUrl: string;
}

/* BEGIN GENERATED */
const DEMO_SCENARIOS: InteractiveScenario[] = [
  {
    id: "scen-safe",
    name: "Trusted Repeat Shopper",
    category: "Verified Good Customer",
    badge: "ALLOW",
    riskScore: 0.02,
    customerSummary: "Account active 420 days · 14 completed past orders · Verified home address",
    plainExplanation: "This customer has an established history with verified shipping and payment methods. The order is automatically approved in under 15ms without adding friction.",
    keyDrivers: [
      "Established account history (> 1 year)",
      "Cardholder name matches shipping profile",
      "Clean residential IP with no past disputes",
    ],
    actionLabel: "Friction-Free Approval",
    targetUrl: "/transactions?preset=preset_normal_buyer",
  },
  {
    id: "scen-carding",
    name: "Stolen Card Rapid Burst",
    category: "Payment Carding Attack",
    badge: "BLOCK",
    riskScore: 0.94,
    customerSummary: "Account age: 18 minutes · 5 failed checkout attempts · Discrepant billing country",
    plainExplanation: "The customer is attempting rapid purchases using stolen payment credentials. TrustShield instantly blocks the transaction to prevent chargeback fees.",
    keyDrivers: [
      "Rapid checkout burst (5 orders in 10 minutes)",
      "Brand-new account created under 30 minutes ago",
      "Shipping address does not match card issuing country",
    ],
    actionLabel: "Immediate Auto-Block",
    targetUrl: "/transactions?preset=preset_device_farm",
  },
  {
    id: "scen-ring",
    name: "Collusive Merchant Ring",
    category: "Marketplace Syndicate",
    badge: "HOLD",
    riskScore: 0.81,
    customerSummary: "Device hash shared with 3 previously banned seller accounts",
    plainExplanation: "This merchant account shares hardware fingerprints with a known fraud ring. Funds are placed on temporary hold for forensic review before payout.",
    keyDrivers: [
      "Shared device fingerprint with banned cluster",
      "High return velocity on previous listings",
      "Connected component in fraud graph topology",
    ],
    actionLabel: "Temporary Fulfillment Hold",
    targetUrl: "/fraud-rings",
  },
];
/* END GENERATED */

export default function PlatformOverviewPage() {
  const { openTour } = useGuide();
  const [activeScenarioId, setActiveScenarioId] = useState<string>("scen-safe");

  const activeScenario = DEMO_SCENARIOS.find((s) => s.id === activeScenarioId) || DEMO_SCENARIOS[0];

  return (
    <div className="space-y-8">
      {/* Platform Architecture Hero Banner */}
      <div className="relative overflow-hidden rounded-2xl border border-[var(--border)] bg-gradient-to-r from-[var(--surface-elevated)]/90 via-[var(--surface)] to-[var(--surface-elevated)]/90 p-8 shadow-sm space-y-6">
        <div className="absolute top-0 left-0 right-0 h-[2px] bg-gradient-to-r from-blue-500 via-purple-500 to-emerald-500" />
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-5">
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-500/10 border border-blue-500/25 text-blue-600 dark:text-blue-400 text-xs font-semibold">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75" />
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-blue-500" />
                </span>
                <span>Production Architecture · Phase 3 Active</span>
              </span>
            </div>
            <h1 className="text-2xl sm:text-3xl font-extrabold text-[var(--text-primary)] tracking-tight">
              Platform Architecture &amp; Engine Overview
            </h1>
            <p className="text-xs sm:text-sm text-[var(--text-secondary)] max-w-2xl leading-relaxed">
              Sub-15ms calibrated inference, temporal bipartite graph topology, multimodal catalog integrity, and finite-sample conformal prediction sets.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3 shrink-0">
            <Link
              href="/dashboard"
              className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-bold text-xs shadow-md shadow-blue-500/20 transition-all card-interactive focus-ring"
            >
              <ShieldAlert className="w-4 h-4" />
              <span>Open Command Center</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </Link>

            <button
              onClick={openTour}
              className="inline-flex items-center gap-1.5 px-3.5 py-2.5 rounded-xl border border-[var(--border)] bg-[var(--surface-elevated)] hover:bg-[var(--surface)] text-xs font-semibold text-[var(--text-primary)] transition-all focus-ring cursor-pointer shadow-xs"
            >
              <HelpCircle className="w-3.5 h-3.5 text-blue-500" />
              <span>Interactive Tour</span>
            </button>
          </div>
        </div>

        {/* 4 Quick Stat Badges */}
        <div className="pt-2 grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs border-t border-[var(--border)]">
          <div className="p-3.5 rounded-xl bg-[var(--surface-elevated)]/70 border border-[var(--border)] hover:border-emerald-500/30 transition-all">
            <span className="text-[11px] text-[var(--text-muted)] font-medium block">Inference Speed</span>
            <span className="text-sm font-bold text-emerald-600 dark:text-emerald-400 mt-0.5 block">{"< 15ms P95"}</span>
          </div>
          <div className="p-3.5 rounded-xl bg-[var(--surface-elevated)]/70 border border-[var(--border)] hover:border-blue-500/30 transition-all">
            <span className="text-[11px] text-[var(--text-muted)] font-medium block">Decision Rigor</span>
            <span className="text-sm font-bold text-blue-600 dark:text-blue-400 mt-0.5 block">Cost-Loss Bayes</span>
          </div>
          <div className="p-3.5 rounded-xl bg-[var(--surface-elevated)]/70 border border-[var(--border)] hover:border-purple-500/30 transition-all">
            <span className="text-[11px] text-[var(--text-muted)] font-medium block">Certainty Bound</span>
            <span className="text-sm font-bold text-purple-600 dark:text-purple-400 mt-0.5 block">{/* BEGIN GENERATED */} 95.0% Guaranteed {/* END GENERATED */}</span>
          </div>
          <div className="p-3.5 rounded-xl bg-[var(--surface-elevated)]/70 border border-[var(--border)] hover:border-cyan-500/30 transition-all">
            <span className="text-[11px] text-[var(--text-muted)] font-medium block">Deployment</span>
            <span className="text-sm font-bold text-cyan-600 dark:text-cyan-400 mt-0.5 block">Single Container</span>
          </div>
        </div>
      </div>

      {/* 4 Core Functional Engine Modules */}
      <div className="space-y-4">
        <div>
          <h2 className="text-base font-bold text-[var(--text-primary)]">
            Core Specialized Engine Modules
          </h2>
          <p className="text-xs text-[var(--text-secondary)]">
            Navigate directly to any specialized fraud defense capability.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          {/* Module 1: Calibrated Cost-Loss Decision Engine */}
          <div className="p-6 rounded-2xl border border-[var(--border)] bg-gradient-to-b from-[var(--surface-elevated)]/60 to-[var(--surface)] hover:border-blue-500/40 hover:shadow-lg transition-all space-y-4 flex flex-col justify-between card-interactive">
            <div className="space-y-2.5">
              <div className="flex items-center justify-between">
                <div className="w-10 h-10 rounded-xl bg-blue-500/10 border border-blue-500/20 text-blue-600 dark:text-blue-400 flex items-center justify-center shadow-xs">
                  <Scale className="w-5 h-5" />
                </div>
                <span className="text-[10px] font-mono px-2.5 py-1 rounded-full bg-blue-500/10 text-blue-600 dark:text-blue-400 border border-blue-500/25 font-semibold">
                  Risk Operations
                </span>
              </div>
              <h3 className="text-base font-bold text-[var(--text-primary)]">
                Calibrated Cost-Loss Decision Engine
              </h3>
              <p className="text-xs text-[var(--text-secondary)] leading-relaxed">
                Raw probabilities from XGBoost are calibrated using Isotonic Regression to eliminate overconfidence.
                Operational decisions (ALLOW, REVIEW, HOLD, BLOCK) are computed via Bayes cost-loss matrices that weigh
                chargeback losses against customer checkout friction.
              </p>
            </div>

            <div className="pt-3 border-t border-[var(--border)] flex items-center justify-between">
              <span className="text-[11px] font-mono text-[var(--text-muted)]">Sub-15ms Real-Time SLA</span>
              <Link
                href="/transactions"
                className="inline-flex items-center gap-1.5 text-xs font-bold text-blue-600 dark:text-blue-400 hover:underline"
              >
                <span>Open Risk Analyzer</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </Link>
            </div>
          </div>

          {/* Module 2: Temporal Graph Topology */}
          <div className="p-6 rounded-2xl border border-[var(--border)] bg-gradient-to-b from-[var(--surface-elevated)]/60 to-[var(--surface)] hover:border-purple-500/40 hover:shadow-lg transition-all space-y-4 flex flex-col justify-between card-interactive">
            <div className="space-y-2.5">
              <div className="flex items-center justify-between">
                <div className="w-10 h-10 rounded-xl bg-purple-500/10 border border-purple-500/20 text-purple-600 dark:text-purple-400 flex items-center justify-center shadow-xs">
                  <Network className="w-5 h-5" />
                </div>
                <span className="text-[10px] font-mono px-2.5 py-1 rounded-full bg-purple-500/10 text-purple-600 dark:text-purple-400 border border-purple-500/25 font-semibold">
                  Forensics &amp; Rings
                </span>
              </div>
              <h3 className="text-base font-bold text-[var(--text-primary)]">
                Temporal Graph Topology &amp; Collusion Rings
              </h3>
              <p className="text-xs text-[var(--text-secondary)] leading-relaxed">
                Detects multi-entity fraud rings, synthetic identity farms, and seller collusion clusters using in-memory
                temporal bipartite graphs. Eliminates heavyweight graph databases while providing instant community
                detection and PageRank risk propagation.
              </p>
            </div>

            <div className="pt-3 border-t border-[var(--border)] flex items-center justify-between">
              <span className="text-[11px] font-mono text-[var(--text-muted)]">708 Detected Rings</span>
              <Link
                href="/fraud-rings"
                className="inline-flex items-center gap-1.5 text-xs font-bold text-purple-600 dark:text-purple-400 hover:underline"
              >
                <span>Explore Fraud Rings</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </Link>
            </div>
          </div>

          {/* Module 3: Multimodal Catalog & Return Abuse */}
          <div className="p-6 rounded-2xl border border-[var(--border)] bg-gradient-to-b from-[var(--surface-elevated)]/60 to-[var(--surface)] hover:border-emerald-500/40 hover:shadow-lg transition-all space-y-4 flex flex-col justify-between card-interactive">
            <div className="space-y-2.5">
              <div className="flex items-center justify-between">
                <div className="w-10 h-10 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-600 dark:text-emerald-400 flex items-center justify-center shadow-xs">
                  <Layers className="w-5 h-5" />
                </div>
                <span className="text-[10px] font-mono px-2.5 py-1 rounded-full bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/25 font-semibold">
                  Catalog &amp; Returns
                </span>
              </div>
              <h3 className="text-base font-bold text-[var(--text-primary)]">
                Multimodal Catalog &amp; Return Abuse Defense
              </h3>
              <p className="text-xs text-[var(--text-secondary)] leading-relaxed">
                Identifies deceptive marketplace listings through image-text embedding cosine discrepancies (CLIP embeddings). Analyzes
                return velocity and refund anomalies to block repeat wardrobing, empty box claims, and claim velocity abuse.
              </p>
            </div>

            <div className="pt-3 border-t border-[var(--border)] flex items-center gap-3">
              <Link
                href="/listings"
                className="inline-flex items-center gap-1.5 text-xs font-bold text-emerald-600 dark:text-emerald-400 hover:underline"
              >
                <span>Listing Intelligence</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </Link>
              <span className="text-[var(--text-muted)]">·</span>
              <Link
                href="/returns"
                className="inline-flex items-center gap-1.5 text-xs font-bold text-emerald-600 dark:text-emerald-400 hover:underline"
              >
                <span>Return Abuse Studio</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </Link>
            </div>
          </div>

          {/* Module 4: TreeSHAP & Conformal Sets */}
          <div className="p-6 rounded-2xl border border-[var(--border)] bg-gradient-to-b from-[var(--surface-elevated)]/60 to-[var(--surface)] hover:border-amber-500/40 hover:shadow-lg transition-all space-y-4 flex flex-col justify-between card-interactive">
            <div className="space-y-2.5">
              <div className="flex items-center justify-between">
                <div className="w-10 h-10 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-600 dark:text-amber-400 flex items-center justify-center shadow-xs">
                  <FileText className="w-5 h-5" />
                </div>
                <span className="text-[10px] font-mono px-2.5 py-1 rounded-full bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/25 font-semibold">
                  Case Forensics
                </span>
              </div>
              <h3 className="text-base font-bold text-[var(--text-primary)]">
                Autonomous Multi-Agent Case Dossiers
              </h3>
              <p className="text-xs text-[var(--text-secondary)] leading-relaxed">
                Investigators never receive black-box predictions. Every decision includes an exact TreeSHAP attribution
                waterfall explaining key risk drivers, alongside a cryptographic HMAC-SHA256 signed legal dispute brief.
              </p>
            </div>

            <div className="pt-3 border-t border-[var(--border)] flex items-center justify-between">
              <span className="text-[11px] font-mono text-[var(--text-muted)]">Signed Dispute Briefs</span>
              <Link
                href="/investigations"
                className="inline-flex items-center gap-1.5 text-xs font-bold text-amber-600 dark:text-amber-400 hover:underline"
              >
                <span>Case Dossier Studio</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </Link>
            </div>
          </div>
        </div>
      </div>

      {/* Interactive Scenario Preview Card */}
      <div className="rounded-2xl border border-[var(--border)] bg-[radial-gradient(ellipse_at_top_right,rgba(56,117,246,0.1),transparent_50%)] bg-[var(--surface)] p-6 shadow-sm space-y-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-[var(--border)] pb-3">
          <div>
            <h2 className="text-sm font-bold text-[var(--text-primary)] flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-blue-500" />
              <span>Interactive Scenario Simulation</span>
            </h2>
            <p className="text-xs text-[var(--text-secondary)] mt-0.5">
              Select a benchmark profile to inspect how TrustShield evaluates risk and provides clear operational guidance.
            </p>
          </div>
        </div>

        {/* 3 Scenario Selector Buttons */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {DEMO_SCENARIOS.map((scen) => {
            const isSelected = scen.id === activeScenarioId;
            return (
              <button
                key={scen.id}
                onClick={() => setActiveScenarioId(scen.id)}
                className={`p-3.5 rounded-xl border text-left transition-all cursor-pointer ${
                  isSelected
                    ? "border-blue-500 bg-[var(--surface-elevated)] shadow-sm ring-1 ring-blue-500/30"
                    : "border-[var(--border)] bg-[var(--surface)] hover:bg-[var(--surface-elevated)]"
                }`}
              >
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-[10px] font-mono text-[var(--text-muted)]">{scen.category}</span>
                  <DecisionBadge decision={scen.badge} size="sm" />
                </div>
                <div className="font-semibold text-xs text-[var(--text-primary)]">{scen.name}</div>
                <div className="text-[11px] text-[var(--text-secondary)] mt-1 truncate">{scen.actionLabel}</div>
              </button>
            );
          })}
        </div>

        {/* Selected Scenario Preview */}
        <div className="p-5 rounded-xl border border-[var(--border)] bg-[var(--surface-elevated)] space-y-3">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-[var(--border)]">
            <div>
              <div className="text-[11px] font-medium text-[var(--text-muted)]">Profile Details</div>
              <div className="text-sm font-bold text-[var(--text-primary)]">{activeScenario.name}</div>
              <div className="text-xs text-[var(--text-secondary)] mt-0.5">{activeScenario.customerSummary}</div>
            </div>

            <div className="flex items-center gap-3">
              <div className="text-right font-mono">
                <div className="text-[10px] text-[var(--text-muted)]">Calibrated Risk</div>
                <div className="text-lg font-bold text-[var(--text-primary)]">
                  {(activeScenario.riskScore * 100).toFixed(1)}%
                </div>
              </div>
              <DecisionBadge decision={activeScenario.badge} />
            </div>
          </div>

          <div className="space-y-1 text-xs">
            <span className="font-semibold text-[var(--text-primary)]">Plain-English Decision Rationale:</span>
            <p className="text-[var(--text-secondary)] leading-relaxed bg-[var(--surface)] p-3 rounded-lg border border-[var(--border)]">
              {activeScenario.plainExplanation}
            </p>
          </div>

          <div className="pt-2 flex items-center justify-end">
            <Link
              href={activeScenario.targetUrl}
              className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-white font-semibold text-xs transition-colors shadow-sm focus-ring"
            >
              <span>Test Scenario in Live Console</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
