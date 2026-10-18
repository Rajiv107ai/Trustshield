"use client";

import React, { useState } from "react";
import Link from "next/link";
import {
  ShieldAlert,
  Zap,
  Network,
  Scale,
  FileText,
  ArrowRight,
  Activity,
  CheckCircle2,
  Layers,
  ChevronRight,
  Terminal,
  HelpCircle,
  Sparkles,
  Users,
  Eye,
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
    customerSummary: "Account age: 420 days · 14 completed past orders · Verified home address",
    plainExplanation: "This customer has an established history with verified shipping and payment methods. The order is automatically approved in under 15ms without adding friction.",
    keyDrivers: [
      "Established account history (> 1 year)",
      "Cardholder name matches shipping profile",
      "Clean residential IP with no past disputes",
    ],
    actionLabel: "Friction-Free Approval",
    targetUrl: "/transactions",
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
    targetUrl: "/transactions",
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

export default function LandingPage() {
  const { openTour } = useGuide();
  const [activeScenarioId, setActiveScenarioId] = useState<string>("scen-safe");

  const activeScenario = DEMO_SCENARIOS.find((s) => s.id === activeScenarioId) || DEMO_SCENARIOS[0];

  return (
    <div className="relative overflow-hidden">
      {/* Background Subtle Gradient Glow */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[1000px] h-[500px] bg-gradient-to-b from-blue-500/10 via-indigo-500/5 to-transparent blur-3xl -z-10 pointer-events-none" />

      {/* Hero Section */}
      <section className="pt-16 pb-16 md:pt-24 md:pb-24 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
        <div className="text-center max-w-3xl mx-auto space-y-6">
          {/* Release Badge */}
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full border border-blue-500/20 bg-blue-500/10 text-blue-600 dark:text-blue-400 text-xs font-mono font-medium">
            <span className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
            <span>TrustShield AI · Enterprise Marketplace Protection</span>
            <span className="text-[var(--text-muted)]">|</span>
            <span className="text-[var(--text-secondary)]">Single-Container Serving</span>
          </div>

          {/* Main Headline */}
          <h1 className="text-3xl sm:text-5xl lg:text-6xl font-extrabold tracking-tight text-[var(--text-primary)] leading-tight">
            Stop Marketplace Fraud{" "}
            <span className="bg-gradient-to-r from-blue-600 via-indigo-500 to-cyan-500 bg-clip-text text-transparent">
              Before It Costs You Money
            </span>
          </h1>

          {/* Subtitle */}
          <p className="text-base sm:text-lg text-[var(--text-secondary)] leading-relaxed max-w-2xl mx-auto">
            Sub-15ms calibrated risk decisions, connected fraud ring detection, and plain-English explainability.
            Built for fraud teams, risk managers, and business operators without requiring a degree in machine learning.
          </p>

          {/* CTAs */}
          <div className="pt-4 flex flex-col sm:flex-row items-center justify-center gap-4">
            <Link
              href="/dashboard"
              className="w-full sm:w-auto inline-flex items-center justify-center gap-2.5 px-6 py-3.5 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-sm font-semibold text-white shadow-lg shadow-blue-500/25 transition-all transform hover:-translate-y-0.5 card-interactive"
            >
              <ShieldAlert className="w-4 h-4" />
              <span>Launch Investigator Console</span>
              <ArrowRight className="w-4 h-4 ml-0.5" />
            </Link>

            <button
              onClick={openTour}
              className="w-full sm:w-auto inline-flex items-center justify-center gap-2 px-5 py-3.5 rounded-xl border border-blue-500/30 bg-blue-500/10 hover:bg-blue-500/20 text-sm font-semibold text-blue-600 dark:text-blue-400 transition-all card-interactive cursor-pointer"
            >
              <HelpCircle className="w-4 h-4" />
              <span>How It Works (2-Min Tour)</span>
            </button>

            <Link
              href="/transactions"
              className="w-full sm:w-auto inline-flex items-center justify-center gap-2 px-5 py-3.5 rounded-xl border border-[var(--border)] bg-[var(--surface-elevated)] hover:bg-[var(--surface)] text-sm font-semibold text-[var(--text-primary)] transition-all card-interactive"
            >
              <Terminal className="w-4 h-4 text-[var(--text-secondary)]" />
              <span>Explore Live Analyzer</span>
            </Link>
          </div>

          {/* Key Metric Strip */}
          <div className="pt-8 grid grid-cols-2 md:grid-cols-4 gap-4 max-w-4xl mx-auto text-left">
            <div className="p-4 rounded-xl border border-[var(--border)] bg-[var(--surface)]/70 backdrop-blur-sm">
              <div className="text-xs text-[var(--text-muted)] font-medium">Inference Speed</div>
              {/* BEGIN GENERATED */}
              <div className="text-xl font-bold font-mono text-[var(--text-primary)] mt-1">{"< 15ms"}</div>
              {/* END GENERATED */}
              <div className="text-[11px] text-[var(--text-secondary)] mt-0.5">Real-time checkout response</div>
            </div>

            <div className="p-4 rounded-xl border border-[var(--border)] bg-[var(--surface)]/70 backdrop-blur-sm">
              <div className="text-xs text-[var(--text-muted)] font-medium">Decision Rigor</div>
              <div className="text-xl font-bold font-mono text-emerald-500 mt-1">Cost-Loss</div>
              <div className="text-[11px] text-[var(--text-secondary)] mt-0.5">Weighs loss vs customer friction</div>
            </div>

            <div className="p-4 rounded-xl border border-[var(--border)] bg-[var(--surface)]/70 backdrop-blur-sm">
              <div className="text-xs text-[var(--text-muted)] font-medium">Statistical Certainty</div>
              {/* BEGIN GENERATED */}
              <div className="text-xl font-bold font-mono text-indigo-500 mt-1">95.0% Guaranteed</div>
              {/* END GENERATED */}
              <div className="text-[11px] text-[var(--text-secondary)] mt-0.5">Finite-sample conformal bounds</div>
            </div>

            <div className="p-4 rounded-xl border border-[var(--border)] bg-[var(--surface)]/70 backdrop-blur-sm">
              <div className="text-xs text-[var(--text-muted)] font-medium">Architecture</div>
              <div className="text-xl font-bold font-mono text-blue-500 mt-1">Single-Container</div>
              <div className="text-[11px] text-[var(--text-secondary)] mt-0.5">Zero Redis / Neo4j / Kafka debt</div>
            </div>
          </div>
        </div>
      </section>

      {/* Interactive Scenario Simulator */}
      <section className="py-12 border-t border-[var(--border)] bg-[var(--surface-elevated)]/40">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center max-w-2xl mx-auto mb-8 space-y-2">
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-blue-500/10 text-blue-600 dark:text-blue-400 text-xs font-semibold uppercase tracking-wider">
              <Sparkles className="w-3.5 h-3.5" />
              <span>Interactive Live Preview</span>
            </span>
            <h2 className="text-2xl font-bold text-[var(--text-primary)]">
              See How TrustShield Evaluates Marketplace Risk
            </h2>
            <p className="text-xs text-[var(--text-secondary)]">
              Click a scenario below to see how our engine analyzes signals and provides clear operational guidance.
            </p>
          </div>

          {/* Scenario Tabs */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-6">
            {DEMO_SCENARIOS.map((scen) => {
              const isSelected = scen.id === activeScenarioId;
              return (
                <button
                  key={scen.id}
                  onClick={() => setActiveScenarioId(scen.id)}
                  className={`p-4 rounded-xl border text-left transition-all cursor-pointer ${
                    isSelected
                      ? "border-blue-500 bg-[var(--surface)] shadow-md"
                      : "border-[var(--border)] bg-[var(--surface-elevated)] hover:bg-[var(--surface)]"
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-[11px] font-mono text-[var(--text-muted)]">{scen.category}</span>
                    <DecisionBadge decision={scen.badge} />
                  </div>
                  <div className="font-semibold text-sm text-[var(--text-primary)]">{scen.name}</div>
                  <div className="text-[11px] text-[var(--text-muted)] mt-1 truncate">{scen.actionLabel}</div>
                </button>
              );
            })}
          </div>

          {/* Active Scenario Card */}
          <div className="p-6 sm:p-8 rounded-2xl border border-[var(--border)] bg-[var(--surface)] shadow-xl space-y-6">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-6 border-b border-[var(--border)]">
              <div className="space-y-1">
                <div className="text-xs font-mono text-[var(--text-muted)]">SIMULATED ORDER PROFILE</div>
                <h3 className="text-xl font-bold text-[var(--text-primary)]">{activeScenario.name}</h3>
                <p className="text-xs text-[var(--text-secondary)]">{activeScenario.customerSummary}</p>
              </div>

              <div className="flex items-center gap-4">
                <div className="text-right">
                  <div className="text-[11px] text-[var(--text-muted)]">Calibrated Risk</div>
                  {/* BEGIN GENERATED */}
                  <div className="text-2xl font-extrabold font-mono text-[var(--text-primary)]">
                    {(activeScenario.riskScore * 100).toFixed(1)}%
                  </div>
                  {/* END GENERATED */}
                </div>
                <DecisionBadge decision={activeScenario.badge} />
              </div>
            </div>

            {/* Plain-Language Explanation */}
            <div className="space-y-3">
              <div className="text-xs font-semibold text-[var(--text-primary)] flex items-center gap-1.5">
                <HelpCircle className="w-3.5 h-3.5 text-blue-500" />
                <span>Plain-English Decision Rationale:</span>
              </div>
              <p className="text-sm text-[var(--text-secondary)] leading-relaxed p-4 rounded-xl bg-[var(--surface-elevated)] border border-[var(--border)]">
                {activeScenario.plainExplanation}
              </p>
            </div>

            {/* Key Contributing Signals */}
            <div className="space-y-3">
              <div className="text-xs font-semibold text-[var(--text-primary)]">Key Signals Detected:</div>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                {activeScenario.keyDrivers.map((driver, idx) => (
                  <div
                    key={idx}
                    className="p-3 rounded-lg bg-[var(--surface-elevated)] border border-[var(--border)] text-xs text-[var(--text-secondary)] flex items-start gap-2"
                  >
                    <CheckCircle2 className="w-4 h-4 text-blue-500 shrink-0 mt-0.5" />
                    <span>{driver}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* CTA to investigate */}
            <div className="pt-2 flex items-center justify-end">
              <Link
                href={activeScenario.targetUrl}
                className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-semibold text-xs transition-colors shadow-sm"
              >
                <span>Inspect in Live Console</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Built for Everyone on Your Team */}
      <section className="py-20 border-t border-[var(--border)]">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center max-w-2xl mx-auto mb-16 space-y-3">
            <h2 className="text-xs font-mono font-bold tracking-wider uppercase text-blue-600 dark:text-blue-400">
              Role-Based Usability
            </h2>
            <p className="text-2xl sm:text-3xl font-bold text-[var(--text-primary)] tracking-tight">
              Designed for Everyone on Your Team
            </p>
            <p className="text-sm text-[var(--text-secondary)]">
              Whether you are an investigator, operations leader, or engineer, TrustShield provides the right level of depth.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="p-6 rounded-2xl border border-[var(--border)] bg-[var(--surface)] space-y-4">
              <div className="w-10 h-10 rounded-xl bg-blue-500/10 text-blue-600 dark:text-blue-400 flex items-center justify-center">
                <Users className="w-5 h-5" />
              </div>
              <h3 className="text-base font-bold text-[var(--text-primary)]">For Risk & Operations Managers</h3>
              <p className="text-xs text-[var(--text-secondary)] leading-relaxed">
                Clear action-oriented queues (ALLOW, REVIEW, HOLD, BLOCK). Protect merchant revenue while reducing
                human review overhead by over {/* BEGIN GENERATED */}70%{/* END GENERATED */}.
              </p>
              <ul className="text-xs text-[var(--text-muted)] space-y-1.5 pl-4 list-disc">
                <li>Automated triage for pending reviews</li>
                <li>Grounded metrics with clear definitions</li>
                <li>Zero false-positive disruption for VIP buyers</li>
              </ul>
            </div>

            <div className="p-6 rounded-2xl border border-[var(--border)] bg-[var(--surface)] space-y-4">
              <div className="w-10 h-10 rounded-xl bg-purple-500/10 text-purple-600 dark:text-purple-400 flex items-center justify-center">
                <Eye className="w-5 h-5" />
              </div>
              <h3 className="text-base font-bold text-[var(--text-primary)]">For Fraud & Trust Investigators</h3>
              <p className="text-xs text-[var(--text-secondary)] leading-relaxed">
                Complete forensic investigation workspace. Understand exactly why an order was flagged, explore connected
                fraud rings, and export printable dossiers.
              </p>
              <ul className="text-xs text-[var(--text-muted)] space-y-1.5 pl-4 list-disc">
                <li>TreeSHAP risk driver breakdowns</li>
                <li>Connected device & IP graph networks</li>
                <li>Printable court-admissible audit reports</li>
              </ul>
            </div>

            <div className="p-6 rounded-2xl border border-[var(--border)] bg-[var(--surface)] space-y-4">
              <div className="w-10 h-10 rounded-xl bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 flex items-center justify-center">
                <CheckCircle2 className="w-5 h-5" />
              </div>
              <h3 className="text-base font-bold text-[var(--text-primary)]">For Marketplace Founders & CTOs</h3>
              <p className="text-xs text-[var(--text-secondary)] leading-relaxed">
                Sub-15ms inference in a single container. Zero distributed infrastructure maintenance (no Redis, Neo4j,
                or Kafka cluster bills to worry about).
              </p>
              <ul className="text-xs text-[var(--text-muted)] space-y-1.5 pl-4 list-disc">
                <li>Single-container FastAPI deployment</li>
                <li>Audited 20-seed pre-registered benchmarks</li>
                <li>{/* BEGIN GENERATED */}95%{/* END GENERATED */} guaranteed finite-sample conformal sets</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* 4 Core Capability Pillars */}
      <section id="capabilities" className="py-20 border-t border-[var(--border)] bg-[var(--surface-elevated)]/40 scroll-mt-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center max-w-2xl mx-auto mb-16 space-y-3">
            <h2 className="text-xs font-mono font-bold tracking-wider uppercase text-blue-600 dark:text-blue-400">
              Core Capabilities
            </h2>
            <p className="text-2xl sm:text-3xl font-bold text-[var(--text-primary)] tracking-tight">
              Enterprise Fraud Defense Without Compromise
            </p>
            <p className="text-sm text-[var(--text-secondary)]">
              Every engine module is mathematically calibrated, leak-free, and directly integrated into investigator workflows.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {/* Pillar 1: Calibrated Transaction Scoring */}
            <div className="p-8 rounded-2xl border border-[var(--border)] bg-[var(--surface)] hover:border-blue-500/40 transition-all card-interactive group">
              <div className="w-12 h-12 rounded-xl bg-blue-500/10 text-blue-600 dark:text-blue-400 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                <Scale className="w-6 h-6" />
              </div>
              <h3 className="text-lg font-bold text-[var(--text-primary)] mb-2">
                1. Calibrated Cost-Loss Decision Engine
              </h3>
              <p className="text-sm text-[var(--text-secondary)] leading-relaxed mb-4">
                Raw probabilities from XGBoost are calibrated using Isotonic Regression to eliminate overconfidence.
                Operational decisions (ALLOW, REVIEW, HOLD, BLOCK) are computed via Bayes cost-loss matrices that weigh
                chargeback losses against customer checkout friction.
              </p>
              <div className="flex flex-wrap gap-2 text-xs font-mono text-[var(--text-muted)]">
                <span className="px-2.5 py-1 rounded-md bg-[var(--surface-elevated)] border border-[var(--border)]">
                  Isotonic Calibration
                </span>
                <span className="px-2.5 py-1 rounded-md bg-[var(--surface-elevated)] border border-[var(--border)]">
                  Cost Matrix ($5 vs $100)
                </span>
                <span className="px-2.5 py-1 rounded-md bg-[var(--surface-elevated)] border border-[var(--border)]">
                  Sub-15ms Scoring
                </span>
              </div>
            </div>

            {/* Pillar 2: Temporal Graph Topology */}
            <div className="p-8 rounded-2xl border border-[var(--border)] bg-[var(--surface)] hover:border-indigo-500/40 transition-all card-interactive group">
              <div className="w-12 h-12 rounded-xl bg-indigo-500/10 text-indigo-600 dark:text-indigo-400 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                <Network className="w-6 h-6" />
              </div>
              <h3 className="text-lg font-bold text-[var(--text-primary)] mb-2">
                2. Temporal Graph Topology & Fraud Rings
              </h3>
              <p className="text-sm text-[var(--text-secondary)] leading-relaxed mb-4">
                Detects multi-entity fraud rings, synthetic identity farms, and seller collusion clusters using in-memory
                temporal bipartite graphs. Eliminates heavyweight graph databases while providing instant community
                detection and PageRank risk propagation.
              </p>
              <div className="flex flex-wrap gap-2 text-xs font-mono text-[var(--text-muted)]">
                <span className="px-2.5 py-1 rounded-md bg-[var(--surface-elevated)] border border-[var(--border)]">
                  NetworkX Graph Engine
                </span>
                <span className="px-2.5 py-1 rounded-md bg-[var(--surface-elevated)] border border-[var(--border)]">
                  Connected Components
                </span>
                <span className="px-2.5 py-1 rounded-md bg-[var(--surface-elevated)] border border-[var(--border)]">
                  Forensic Dossiers
                </span>
              </div>
            </div>

            {/* Pillar 3: Multimodal Catalog & Return Abuse */}
            <div className="p-8 rounded-2xl border border-[var(--border)] bg-[var(--surface)] hover:border-emerald-500/40 transition-all card-interactive group">
              <div className="w-12 h-12 rounded-xl bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                <Layers className="w-6 h-6" />
              </div>
              <h3 className="text-lg font-bold text-[var(--text-primary)] mb-2">
                3. Multimodal Catalog & Return Abuse Defense
              </h3>
              <p className="text-sm text-[var(--text-secondary)] leading-relaxed mb-4">
                Identifies deceptive marketplace listings through image-text embedding cosine discrepancies. Analyzes
                return velocity and refund anomalies to block repeat wardrobing, empty box claims, and claim velocity abuse.
              </p>
              <div className="flex flex-wrap gap-2 text-xs font-mono text-[var(--text-muted)]">
                <span className="px-2.5 py-1 rounded-md bg-[var(--surface-elevated)] border border-[var(--border)]">
                  Cosine Discrepancy
                </span>
                <span className="px-2.5 py-1 rounded-md bg-[var(--surface-elevated)] border border-[var(--border)]">
                  Wardrobing Risk
                </span>
                <span className="px-2.5 py-1 rounded-md bg-[var(--surface-elevated)] border border-[var(--border)]">
                  Refund Velocity Scoring
                </span>
              </div>
            </div>

            {/* Pillar 4: TreeSHAP & Conformal Sets */}
            <div className="p-8 rounded-2xl border border-[var(--border)] bg-[var(--surface)] hover:border-amber-500/40 transition-all card-interactive group">
              <div className="w-12 h-12 rounded-xl bg-amber-500/10 text-amber-600 dark:text-amber-400 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                <FileText className="w-6 h-6" />
              </div>
              <h3 className="text-lg font-bold text-[var(--text-primary)] mb-2">
                4. Exact TreeSHAP & Conformal Prediction Sets
              </h3>
              <p className="text-sm text-[var(--text-secondary)] leading-relaxed mb-4">
                Investigators never receive black-box predictions. Every decision includes an exact TreeSHAP attribution
                waterfall explaining key risk drivers, alongside a statistical confidence set that bounds model uncertainty.
              </p>
              <div className="flex flex-wrap gap-2 text-xs font-mono text-[var(--text-muted)]">
                <span className="px-2.5 py-1 rounded-md bg-[var(--surface-elevated)] border border-[var(--border)]">
                  Exact TreeSHAP
                </span>
                <span className="px-2.5 py-1 rounded-md bg-[var(--surface-elevated)] border border-[var(--border)]">
                  Finite-Sample Conformal
                </span>
                <span className="px-2.5 py-1 rounded-md bg-[var(--surface-elevated)] border border-[var(--border)]">
                  Court-Admissible Logs
                </span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Investigator Workflow Timeline */}
      <section id="workflow" className="py-20 border-t border-[var(--border)] scroll-mt-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center max-w-2xl mx-auto mb-16 space-y-3">
            <h2 className="text-xs font-mono font-bold tracking-wider uppercase text-blue-600 dark:text-blue-400">
              Investigation Lifecycle
            </h2>
            <p className="text-2xl sm:text-3xl font-bold text-[var(--text-primary)] tracking-tight">
              From Real-Time Ingestion to Forensic Dossier
            </p>
            <p className="text-sm text-[var(--text-secondary)]">
              An end-to-end autonomous pipeline designed for high-throughput marketplace operations.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 relative">
            {/* Step 1 */}
            <div className="p-6 rounded-xl border border-[var(--border)] bg-[var(--surface)] relative">
              <div className="flex items-center justify-between mb-4">
                <span className="w-8 h-8 rounded-lg bg-blue-500/10 text-blue-600 dark:text-blue-400 flex items-center justify-center font-mono font-bold text-sm">
                  01
                </span>
                <Zap className="w-4 h-4 text-blue-500" />
              </div>
              <h4 className="text-base font-bold text-[var(--text-primary)] mb-2">Ingestion & State</h4>
              <p className="text-xs text-[var(--text-secondary)] leading-relaxed">
                Order details, device fingerprints, and merchant signals arrive via REST API. Point-in-time features
                are extracted without temporal lookahead leakage.
              </p>
            </div>

            {/* Step 2 */}
            <div className="p-6 rounded-xl border border-[var(--border)] bg-[var(--surface)] relative">
              <div className="flex items-center justify-between mb-4">
                <span className="w-8 h-8 rounded-lg bg-indigo-500/10 text-indigo-600 dark:text-indigo-400 flex items-center justify-center font-mono font-bold text-sm">
                  02
                </span>
                <Scale className="w-4 h-4 text-indigo-500" />
              </div>
              <h4 className="text-base font-bold text-[var(--text-primary)] mb-2">Calibrated Inference</h4>
              <p className="text-xs text-[var(--text-secondary)] leading-relaxed">
                Dual XGBoost models score behavioral velocity and graph topology. Probabilities are calibrated via isotonic
                curves and evaluated against cost-loss thresholds.
              </p>
            </div>

            {/* Step 3 */}
            <div className="p-6 rounded-xl border border-[var(--border)] bg-[var(--surface)] relative">
              <div className="flex items-center justify-between mb-4">
                <span className="w-8 h-8 rounded-lg bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 flex items-center justify-center font-mono font-bold text-sm">
                  03
                </span>
                <Activity className="w-4 h-4 text-emerald-500" />
              </div>
              <h4 className="text-base font-bold text-[var(--text-primary)] mb-2">Attribution & Bands</h4>
              <p className="text-xs text-[var(--text-secondary)] leading-relaxed">
                TreeSHAP calculates exact per-feature contributions. Conformal prediction sets provide rigorous coverage
                guarantees to distinguish borderline cases.
              </p>
            </div>

            {/* Step 4 */}
            <div className="p-6 rounded-xl border border-[var(--border)] bg-[var(--surface)] relative">
              <div className="flex items-center justify-between mb-4">
                <span className="w-8 h-8 rounded-lg bg-amber-500/10 text-amber-600 dark:text-amber-400 flex items-center justify-center font-mono font-bold text-sm">
                  04
                </span>
                <FileText className="w-4 h-4 text-amber-500" />
              </div>
              <h4 className="text-base font-bold text-[var(--text-primary)] mb-2">Dossier Assembly</h4>
              <p className="text-xs text-[var(--text-secondary)] leading-relaxed">
                High-risk rings and repeat offenders are bundled into structured investigation dossiers with full audit
                trails, entity link graphs, and printable exports.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Zero-Daemon Architecture Section */}
      <section id="architecture" className="py-20 border-t border-[var(--border)] bg-[var(--surface-elevated)]/40 scroll-mt-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center max-w-3xl mx-auto mb-16 space-y-3">
            <h2 className="text-xs font-mono font-bold tracking-wider uppercase text-blue-600 dark:text-blue-400">
              Architecture Advantage
            </h2>
            <p className="text-2xl sm:text-3xl font-bold text-[var(--text-primary)] tracking-tight">
              Zero-Daemon Architecture: High Performance, Zero Bloat
            </p>
            <p className="text-sm text-[var(--text-secondary)]">
              Traditional fraud stacks force engineering teams to deploy and maintain 5 to 7 distributed background
              daemons (Redis, Neo4j, Kafka, PyG). TrustShield AI collapsed this entire pipeline into a resilient,
              single-container FastAPI service.
            </p>
          </div>

          {/* Architecture Comparison: Legacy Distributed vs TrustShield AI Single-Container */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
            {/* Legacy Stack Card */}
            <div className="p-6 sm:p-8 rounded-2xl border border-red-500/20 bg-[var(--surface)] relative overflow-hidden space-y-6">
              <div className="flex items-center justify-between pb-4 border-b border-[var(--border)]">
                <div>
                  <span className="text-[10px] font-mono font-bold px-2 py-0.5 rounded bg-red-500/10 text-red-500 border border-red-500/20 uppercase">
                    Legacy Distributed Architecture
                  </span>
                  <h3 className="text-lg font-bold text-[var(--text-primary)] mt-1.5">
                    Distributed Research Monolith
                  </h3>
                </div>
                <div className="text-right">
                  <span className="text-xs font-mono text-red-500 font-bold">5+ Daemons</span>
                  <div className="text-[10px] text-[var(--text-muted)]">High DevOps Debt</div>
                </div>
              </div>

              <div className="space-y-3">
                <div className="p-3 rounded-xl bg-[var(--surface-elevated)] border border-[var(--border)] flex items-start gap-3">
                  <div className="w-2 h-2 rounded-full bg-red-500 mt-1.5 shrink-0" />
                  <div>
                    <div className="text-xs font-semibold text-[var(--text-primary)]">Redis Cache &amp; Session Cluster</div>
                    <div className="text-[11px] text-[var(--text-muted)] mt-0.5">Connection pool limits, cache eviction bugs, and memory leakage.</div>
                  </div>
                </div>

                <div className="p-3 rounded-xl bg-[var(--surface-elevated)] border border-[var(--border)] flex items-start gap-3">
                  <div className="w-2 h-2 rounded-full bg-red-500 mt-1.5 shrink-0" />
                  <div>
                    <div className="text-xs font-semibold text-[var(--text-primary)]">Neo4j / TigerGraph Graph DB</div>
                    <div className="text-[11px] text-[var(--text-muted)] mt-0.5">External query latency (40-80ms hops) and graph database license overhead.</div>
                  </div>
                </div>

                <div className="p-3 rounded-xl bg-[var(--surface-elevated)] border border-[var(--border)] flex items-start gap-3">
                  <div className="w-2 h-2 rounded-full bg-red-500 mt-1.5 shrink-0" />
                  <div>
                    <div className="text-xs font-semibold text-[var(--text-primary)]">Apache Kafka / Celery Workers</div>
                    <div className="text-[11px] text-[var(--text-muted)] mt-0.5">Consumer lag, message re-ordering, and asynchronous state synchronization failures.</div>
                  </div>
                </div>

                <div className="p-3 rounded-xl bg-[var(--surface-elevated)] border border-[var(--border)] flex items-start gap-3">
                  <div className="w-2 h-2 rounded-full bg-red-500 mt-1.5 shrink-0" />
                  <div>
                    <div className="text-xs font-semibold text-[var(--text-primary)]">PyTorch / PyG Deep Learning Daemon</div>
                    <div className="text-[11px] text-[var(--text-muted)] mt-0.5">Heavy GPU hosting cost, cuda initialization delays, and negative transfer degradation.</div>
                  </div>
                </div>
              </div>

              <div className="p-4 rounded-xl bg-red-500/5 border border-red-500/10 flex items-center justify-between text-xs">
                <span className="text-[var(--text-secondary)]">Observed P99 Scoring Latency:</span>
                <span className="font-mono font-bold text-red-500">120ms - 250ms</span>
              </div>
            </div>

            {/* TrustShield Lean Single-Container Card */}
            <div className="p-6 sm:p-8 rounded-2xl border border-emerald-500/30 bg-[var(--surface)] relative overflow-hidden space-y-6 shadow-xl shadow-emerald-500/5">
              <div className="flex items-center justify-between pb-4 border-b border-[var(--border)]">
                <div>
                  <span className="text-[10px] font-mono font-bold px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-500 border border-emerald-500/20 uppercase">
                    TrustShield AI Architecture
                  </span>
                  <h3 className="text-lg font-bold text-[var(--text-primary)] mt-1.5">
                    Zero-Daemon FastAPI Engine
                  </h3>
                </div>
                <div className="text-right">
                  <span className="text-xs font-mono text-emerald-500 font-bold">Zero Daemons</span>
                  <div className="text-[10px] text-[var(--text-muted)]">Single Container</div>
                </div>
              </div>

              <div className="space-y-3">
                <div className="p-3 rounded-xl bg-[var(--surface-elevated)] border border-[var(--border)] flex items-start gap-3">
                  <CheckCircle2 className="w-4 h-4 text-emerald-500 mt-0.5 shrink-0" />
                  <div>
                    <div className="text-xs font-semibold text-[var(--text-primary)]">In-Memory Temporal Graph (NetworkX)</div>
                    <div className="text-[11px] text-[var(--text-muted)] mt-0.5">Bipartite projections and connected component clusters computed in-process.</div>
                  </div>
                </div>

                <div className="p-3 rounded-xl bg-[var(--surface-elevated)] border border-[var(--border)] flex items-start gap-3">
                  <CheckCircle2 className="w-4 h-4 text-emerald-500 mt-0.5 shrink-0" />
                  <div>
                    <div className="text-xs font-semibold text-[var(--text-primary)]">Native XGBoost C++ Runtime + Isotonic Calibration</div>
                    <div className="text-[11px] text-[var(--text-muted)] mt-0.5">Zero inter-process network overhead; probabilities calibrated directly at scoring time.</div>
                  </div>
                </div>

                <div className="p-3 rounded-xl bg-[var(--surface-elevated)] border border-[var(--border)] flex items-start gap-3">
                  <CheckCircle2 className="w-4 h-4 text-emerald-500 mt-0.5 shrink-0" />
                  <div>
                    <div className="text-xs font-semibold text-[var(--text-primary)]">Point-in-Time Temporal Feature Isolation</div>
                    <div className="text-[11px] text-[var(--text-muted)] mt-0.5">Guaranteed zero lookahead leakage between chronological training and live inference.</div>
                  </div>
                </div>

                <div className="p-3 rounded-xl bg-[var(--surface-elevated)] border border-[var(--border)] flex items-start gap-3">
                  <CheckCircle2 className="w-4 h-4 text-emerald-500 mt-0.5 shrink-0" />
                  <div>
                    <div className="text-xs font-semibold text-[var(--text-primary)]">Pure CPU Single-Container Serving</div>
                    <div className="text-[11px] text-[var(--text-muted)] mt-0.5">No GPU cluster bills; runs on standard virtual machines or serverless containers.</div>
                  </div>
                </div>
              </div>

              <div className="p-4 rounded-xl bg-emerald-500/5 border border-emerald-500/10 flex items-center justify-between text-xs">
                <span className="text-[var(--text-secondary)]">Observed P99 Scoring Latency:</span>
                {/* BEGIN GENERATED */}
                <span className="font-mono font-bold text-emerald-500">{"< 15ms (10x Faster)"}</span>
                {/* END GENERATED */}
              </div>
            </div>
          </div>

          {/* Architecture Benefits Strip */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
            <div className="p-5 rounded-xl border border-[var(--border)] bg-[var(--surface)]">
              <div className="text-xs font-mono text-blue-500 font-bold uppercase mb-1">01 · Zero DevOps Overhead</div>
              <h4 className="text-sm font-bold text-[var(--text-primary)] mb-1">Single Docker Container</h4>
              <p className="text-xs text-[var(--text-secondary)] leading-relaxed">
                No cluster configuration, zookeeper coordination, or database schema migrations. Starts in seconds.
              </p>
            </div>

            <div className="p-5 rounded-xl border border-[var(--border)] bg-[var(--surface)]">
              <div className="text-xs font-mono text-emerald-500 font-bold uppercase mb-1">02 · Zero Network Hops</div>
              <h4 className="text-sm font-bold text-[var(--text-primary)] mb-1">In-Process Calculations</h4>
              <p className="text-xs text-[var(--text-secondary)] leading-relaxed">
                Graph features, tabular models, and TreeSHAP waterfall values execute in the same memory space.
              </p>
            </div>

            <div className="p-5 rounded-xl border border-[var(--border)] bg-[var(--surface)]">
              <div className="text-xs font-mono text-purple-500 font-bold uppercase mb-1">03 · Verified Rigor</div>
              <h4 className="text-sm font-bold text-[var(--text-primary)] mb-1">Point-in-Time Integrity</h4>
              <p className="text-xs text-[var(--text-secondary)] leading-relaxed">
                Mathematically isolated historical states ensure training signals match production behavior.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Audited Benchmark Summary Section */}
      <section id="audited-results" className="py-20 border-t border-[var(--border)] scroll-mt-20">
        <span id="benchmarks" className="-top-20 relative block" />
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col md:flex-row md:items-end justify-between mb-12 gap-4">
            <div>
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-blue-500/20 bg-blue-500/10 text-blue-600 dark:text-blue-400 text-xs font-mono font-medium mb-3">
                <span className="w-1.5 h-1.5 rounded-full bg-blue-500" />
                <span>Audited Results &amp; Statistical Rigor</span>
              </div>
              <h2 className="text-2xl sm:text-3xl font-bold text-[var(--text-primary)] tracking-tight">
                20-Seed Pre-Registered Benchmark Results
              </h2>
              <p className="text-sm text-[var(--text-secondary)] mt-1 max-w-2xl">
                Evaluated across 20 randomized seeds with Holm-Bonferroni correction and strict chronological train/test splits.
                Empirical audit comparing TrustShield AI against legacy distributed baselines.
              </p>
            </div>
            <Link
              href="/evaluation"
              className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold transition-colors shadow-sm"
            >
              <span>View Full Statistical Dossier</span>
              <ChevronRight className="w-4 h-4" />
            </Link>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            <div className="p-6 rounded-xl border border-[var(--border)] bg-[var(--surface)]">
              <div className="text-xs text-[var(--text-muted)] font-mono">PRIMARY METRIC</div>
              <div className="text-sm font-semibold text-[var(--text-primary)] mt-1">ROC-AUC (20 Seeds)</div>
              {/* BEGIN GENERATED */}
              <div className="text-3xl font-extrabold font-mono text-blue-600 dark:text-blue-400 mt-2">0.9620</div>
              <div className="text-xs text-[var(--text-secondary)] mt-1 font-mono">Std Dev: ±0.0084</div>
              {/* END GENERATED */}
              <p className="text-xs text-[var(--text-muted)] mt-3">
                Superior class discrimination on heavily imbalanced marketplace fraud distributions.
              </p>
            </div>

            <div className="p-6 rounded-xl border border-[var(--border)] bg-[var(--surface)]">
              <div className="text-xs text-[var(--text-muted)] font-mono">PRECISION-RECALL</div>
              <div className="text-sm font-semibold text-[var(--text-primary)] mt-1">PR-AUC (20 Seeds)</div>
              {/* BEGIN GENERATED */}
              <div className="text-3xl font-extrabold font-mono text-indigo-600 dark:text-indigo-400 mt-2">0.7842</div>
              <div className="text-xs text-[var(--text-secondary)] mt-1 font-mono">Std Dev: ±0.0152</div>
              {/* END GENERATED */}
              <p className="text-xs text-[var(--text-muted)] mt-3">
                Minimizes false positive alerts while maintaining high recall on high-value fraud vectors.
              </p>
            </div>

            <div className="p-6 rounded-xl border border-[var(--border)] bg-[var(--surface)]">
              <div className="text-xs text-[var(--text-muted)] font-mono">RELIABILITY</div>
              <div className="text-sm font-semibold text-[var(--text-primary)] mt-1">Expected Calibration Error</div>
              {/* BEGIN GENERATED */}
              <div className="text-3xl font-extrabold font-mono text-emerald-600 dark:text-emerald-400 mt-2">0.0195</div>
              <div className="text-xs text-[var(--text-secondary)] mt-1 font-mono">Isotonic Calibrated</div>
              {/* END GENERATED */}
              <p className="text-xs text-[var(--text-muted)] mt-3">
                Output probabilities reflect empirical ground-truth likelihood for accurate Bayes thresholding.
              </p>
            </div>
          </div>

          {/* Audit Verification Callout Box */}
          <div className="p-6 rounded-2xl border border-[var(--border)] bg-[var(--surface-elevated)]/60 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
            <div className="space-y-1">
              <div className="text-xs font-mono font-bold text-emerald-600 dark:text-emerald-400">
                AUDIT VERDICT: [SOUND WITH SPECIFIC LIMITATIONS]
              </div>
              <div className="text-sm font-semibold text-[var(--text-primary)]">
                GraphSAGE GNN Pruning Was Empirically Justified
              </div>
              <p className="text-xs text-[var(--text-secondary)]">
                Over-parameterized GraphSAGE GNNs suffered an empirical degradation of {/* BEGIN GENERATED */}-0.0263{/* END GENERATED */} ROC-AUC ({/* BEGIN GENERATED */}p = 0.0083{/* END GENERATED */}) compared to point-in-time XGBoost.
              </p>
            </div>
            <Link
              href="/evaluation"
              className="inline-flex items-center gap-1.5 text-xs font-semibold text-blue-600 dark:text-blue-400 hover:underline shrink-0"
            >
              <span>Read Auditor Verdict &amp; Ablations</span>
              <ChevronRight className="w-3.5 h-3.5" />
            </Link>
          </div>
        </div>
      </section>

      {/* Bottom CTA Banner */}
      <section className="py-16 border-t border-[var(--border)] bg-gradient-to-b from-[var(--surface)] to-[var(--surface-elevated)]">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center space-y-6">
          <h2 className="text-2xl sm:text-3xl font-bold text-[var(--text-primary)]">
            Ready to Protect Your Marketplace Revenue?
          </h2>
          <p className="text-sm text-[var(--text-secondary)] max-w-xl mx-auto">
            Launch the TrustShield AI console to explore live transaction telemetry, simulate high-risk scenarios,
            and inspect forensic fraud rings.
          </p>
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 pt-2">
            <Link
              href="/dashboard"
              className="w-full sm:w-auto inline-flex items-center justify-center gap-2 px-6 py-3.5 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-sm font-semibold text-white shadow-lg shadow-blue-500/25 transition-all card-interactive"
            >
              <ShieldAlert className="w-4 h-4" />
              <span>Launch Investigator Console</span>
              <ArrowRight className="w-4 h-4 ml-1" />
            </Link>
            <button
              onClick={openTour}
              className="w-full sm:w-auto inline-flex items-center justify-center gap-2 px-6 py-3.5 rounded-xl border border-[var(--border)] bg-[var(--surface)] hover:bg-[var(--surface-elevated)] text-sm font-semibold text-[var(--text-primary)] transition-all card-interactive cursor-pointer"
            >
              <HelpCircle className="w-4 h-4 text-blue-500" />
              <span>Start Product Tour</span>
            </button>
          </div>
        </div>
      </section>
    </div>
  );
}
