# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

- **Primary Persona:** Trust & Safety Analysts and Fraud Operations Specialists monitoring live transaction streams, conducting case investigations, and issuing enforcement actions (ALLOW, REVIEW, HOLD, BLOCK).
- **Secondary Persona:** Machine Learning Engineers and Risk Governance Officers inspecting model calibration drift, TreeSHAP feature attributions, and 20-seed benchmark provenance.

## Product Purpose

TrustShield AI is an enterprise-grade e-commerce fraud intelligence platform designed to eliminate temporal feature leakage, detect coordinated collusion rings, and provide auditable, explainable risk decisioning in sub-50ms latency.

Success is measured by:
1. Fast, unambiguous decision triage for high-volume transactions.
2. Forensic clarity into multi-entity fraud rings (shared devices, shared addresses, buyer-seller collusion).
3. Zero temporal leakage and calibrated certainty in risk scoring.

## Positioning

Unlike black-box fraud APIs or disconnected analytics dashboards, TrustShield unifies strict point-in-time causal feature stores, graph topology intelligence, multimodal CLIP product analysis, and Conformal Risk Control into a single, cohesive operator console.

## Operating Context

- **Environment:** Web-based operational console (`Operate` mode), optimized for continuous desktop usage under varied ambient lighting (deep dark theme default with high-contrast light theme).
- **Core Rituals:** Real-time transaction alert triage, deep-dive fraud ring forensics, automated dossier export for merchant disputes, and model health verification.
- **Dual Mode:** Live Gateway mode (connecting to live FastAPI backend at `/transaction/score`) and self-contained Simulation Demo mode.

## Capabilities and Constraints

- **Real-time Scoring:** Instant evaluation across 29 tabular, graph, and historical signals.
- **TreeSHAP Attributions:** Top positive and negative feature contributions rendered in real-time.
- **Graph Forensics:** Interactive connected component, PageRank, and collusion cluster exploration.
- **Multimodal CLIP:** Image-text catalog consistency scoring for counterfeit or miscategorized items.
- **Strict Latency & Accessibility:** Fast rendering, keyboard command palette (`Cmd/Ctrl + K`), full keyboard navigation, and WCAG AA contrast compliance.

## Brand Commitments

- **Tone & Voice:** Authoritative, disciplined, forensic, and precise. No marketing fluff or speculative terminology.
- **Visual Stance:** Dense, clean, and utilitarian. Form follows function; information density without visual chaos.
- **Color Semantics:** Strict traffic-light decision mappings (`#22C55E` Allow, `#F59E0B` Review, `#F97316` Hold, `#EF4444` Block).

## Evidence on Hand

- Canonical synthetic marketplace dataset (`synthetic_v2_1`): 20,000 transactions across genuine orders, fake listings, return abuse, and collusion rings.
- Pre-computed 20-seed empirical benchmark metrics and calibration artifacts in `results/results.json`.
- 187 passing unit, regression, and operational invariant tests.

## Product Principles

1. **Evidence Precedes Decision:** Every high-risk score must be anchored in traceable reason codes, point-in-time signals, and SHAP feature attributions.
2. **Speed Without Sacrificing Precision:** High information density and zero-latency layout transitions allow operators to evaluate cases in seconds.
3. **Fail-Closed and Auditable:** Operational actions must generate immutable cryptographic dispute briefs and audit event records.
4. **Restraint and Focus:** The interface recedes to let data, risk distributions, and network graphs command attention.

