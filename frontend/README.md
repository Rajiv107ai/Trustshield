# TrustShield AI — Enterprise Frontend Console

[![Next.js](https://img.shields.io/badge/Next.js-16.4+-black.svg)](https://nextjs.org)
[![React](https://img.shields.io/badge/React-19.3+-blue.svg)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-Strict-3178C6.svg)](https://www.typescriptlang.org)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind-v4-38B2AC.svg)](https://tailwindcss.com)

Production-grade enterprise user interface for the **TrustShield AI** fraud intelligence platform. Built with Next.js 16 (App Router), React 19, strict TypeScript, and a high-density dark console design system.

---

## 🚀 Quickstart

```bash
# 1. Install dependencies
npm install

# 2. Start the development server
npm run dev

# 3. Open browser
http://localhost:3000
```

The frontend connects by default to the FastAPI backend running at `http://localhost:8000`. If the backend is not yet started, the interface automatically runs in deterministic fallback mode using pre-verified scenario datasets.

---

## 🏛️ Views & Navigation Reference

| Path | Screen Name | Description |
| :--- | :--- | :--- |
| **`/dashboard`** | **Command Center** | System readiness probe, key volume metrics, decision distribution bar, and live flagged queue. |
| **`/transactions`** | **Risk Analyzer (Hero)** | 1-Click scenario simulator, decision banner, Trust score gauge, TreeSHAP attributions, and conformal uncertainty. |
| **`/transactions/feed`** | **Audit Stream** | High-density real-time transaction feed with Server-Sent Events (SSE) live streaming and risk filtering. |
| **`/fraud-rings`** | **Collusion Canvas** | Ranked candidate clusters, topological density, merchant concentration (HHI), and account freeze actions. |
| **`/trust-graph`** | **Trust Graph** | Multi-entity relational graph canvas (Buyers, Sellers, Devices, Addresses) with historical timeline controls. |
| **`/listings`** | **Listing Intelligence** | CLIP visual-semantic embeddings and FAISS nearest-neighbor indexing detecting stolen catalog photos. |
| **`/returns`** | **Return Abuse** | Serial return abuse detector evaluating return velocity and wardrobing risk. |
| **`/investigations`** | **Evidence Dossiers** | Formal investigation incident dossiers with TreeSHAP risk breakdowns, graph topology, and anti-hallucination indicators. |
| **`/models`** | **Model Registry** | Model provenance, Isotonic probability calibration ECE reliability curve, and Brier metrics. |
| **`/evaluation`** | **Evaluation Studio** | Algorithmic ablation ladders, test ROC/PR benchmarks, and dynamic decision threshold matrix. |
| **`/monitoring`** | **Mesh Telemetry** | Kubernetes Liveness (`/health`) vs. Readiness (`/ready`), live Prometheus metrics, latency percentiles, and Docker mesh status. |
| **`/settings`** | **Settings** | Base API URL configuration, active model version switcher, and default persona settings. |

---

## 👥 Dual-Audience Mode Toggle

Toggle between views instantly using the header switch:
- **`✨ Executive Story View`**: Translates model outputs into plain-English narratives, risk meters, and clear operational recommendations.
- **`🔬 Deep AI Inspector`**: Exposes raw calibrated probabilities, GNN 16-dimensional node embeddings, continuous-time edge assertions, and split conformal coverage sets ($\{0\}$, $\{1\}$, $\{0, 1\}$).

---

## ⌨️ Global Search (⌘K / Ctrl+K)

Press `⌘K` (Mac) or `Ctrl+K` (Windows/Linux) anywhere in the application to instantly search transactions (`ORD_...`), buyers, merchants, hardware devices, fraud rings, and navigation pages.

---

## 🏗️ Production Build

```bash
npm run build
```
Generates an optimized static production bundle.
