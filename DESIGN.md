---
name: TrustShield AI Design System
description: High-density forensic intelligence and operational risk console
colors:
  background: "#0B0F14"
  foreground: "#E8EDF3"
  surface: "#111821"
  surface-elevated: "#151D27"
  surface-hover: "#1A2432"
  border: "#202A35"
  border-subtle: "rgba(255, 255, 255, 0.08)"
  text-primary: "#E8EDF3"
  text-secondary: "#8995A3"
  text-muted: "#596574"
  allow: "#22C55E"
  review: "#F59E0B"
  hold: "#F97316"
  block: "#EF4444"
  intelligence-blue: "#3B82F6"
  graph-purple: "#8B5CF6"
  device-cyan: "#06B6D4"
typography:
  display:
    fontFamily: "var(--font-sans), -apple-system, BlinkMacSystemFont, sans-serif"
    fontSize: "2rem"
    fontWeight: 600
    lineHeight: 1.2
    letterSpacing: "-0.02em"
  headline:
    fontFamily: "var(--font-sans), -apple-system, BlinkMacSystemFont, sans-serif"
    fontSize: "1.25rem"
    fontWeight: 600
    lineHeight: 1.3
    letterSpacing: "-0.01em"
  body:
    fontFamily: "var(--font-sans), -apple-system, BlinkMacSystemFont, sans-serif"
    fontSize: "0.875rem"
    fontWeight: 400
    lineHeight: 1.5
    letterSpacing: "0em"
  code:
    fontFamily: "var(--font-mono), 'JetBrains Mono', monospace"
    fontSize: "0.75rem"
    fontWeight: 500
    lineHeight: 1.4
    letterSpacing: "0.02em"
rounded:
  sm: "4px"
  md: "6px"
  lg: "8px"
  full: "9999px"
spacing:
  xs: "4px"
  sm: "8px"
  md: "16px"
  lg: "24px"
  xl: "32px"
components:
  button-primary:
    backgroundColor: "{colors.intelligence-blue}"
    textColor: "{colors.foreground}"
    rounded: "{rounded.md}"
    padding: "8px 16px"
  card-base:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.text-primary}"
    rounded: "{rounded.lg}"
    padding: "16px 20px"
---

# Design System

## Overview

TrustShield AI is a high-density, forensic operations console designed for Trust & Safety analysts and risk officers. Its visual language prioritizes rapid legibility, strict status color-coding, low visual fatigue during extended shifts, and clean information hierarchy.

The design mode is **Operate**: the interface recedes into an obsidian surface layer so that live transaction data, TreeSHAP feature attributions, and collusion network graphs command operator focus.

## Colors

The palette is engineered around dark-mode operational ergonomics (`#0B0F14` base background) with an adapted, high-contrast light mode.

### Base Tonal Structure
- **Canvas / Background (`#0B0F14`):** Deep obsidian backdrop minimizing eye strain.
- **Surface (`#111821`):** Base container card background for panels, feeds, and tables.
- **Surface Elevated (`#151D27`):** Elevated cards, active filters, and interactive popovers.
- **Surface Hover (`#1A2432`):** Micro-interaction response for table rows, menu items, and buttons.
- **Border (`#202A35`):** Subtle separator maintaining clean definition without harsh division.

### Semantic Decision Tokens
Decision colors are standardized and reserved exclusively for risk states:
- **Allow (`#22C55E`):** Low risk, approved transactions.
- **Review (`#F59E0B`):** Borderline anomalies requiring human analyst inspection.
- **Hold (`#F97316`):** High risk, payment or order deferred pending identity verification.
- **Block (`#EF4444`):** Confirmed fraud or critical policy violations.

### Domain Accents
- **Intelligence Blue (`#3B82F6`):** Primary action buttons, active navigation markers, and TreeSHAP positive SHAP values.
- **Graph Purple (`#8B5CF6`):** Fraud-ring networks, shared-address links, and community clusters.
- **Device Cyan (`#06B6D4`):** Device fingerprints, browser hashes, and hardware collision signals.

## Typography

TrustShield uses a crisp typographic hierarchy pairing modern sans-serif body text with a dedicated monospaced face for numerical analysis.

- **Interface Font:** `Geist Sans`, `-apple-system`, `BlinkMacSystemFont`, `Segoe UI`, `Roboto`.
- **Data & Measurement Font:** `Geist Mono`, `JetBrains Mono`, `monospace`.
- **Tabular Numerals:** All amounts, percentages, timestamps, latency metrics, and fraud probabilities enforce `font-variant-numeric: tabular-nums` (`tabular-nums`) to prevent jitter across updating streams.
- **Heading Restraint:** Headings are balanced and compact. Never use uppercase tracking as an artificial eyebrow above a heading.

## Layout

- **Frame Architecture:** Two-column AppShell layout:
  - Sticky Left Sidebar: 256px (`w-64`) fixed navigation with domain categories.
  - Sticky Top Header: 64px (`h-16`) containing gateway health, simulation toggle, model spec badge, quick search, and theme controls.
  - Main Workspace: Flex container with fluid grid and standard 24px (`px-6 py-6`) padding.
- **Grid Rhythm:** 4px base increment, with standard gaps of 8px (`gap-2`), 16px (`gap-4`), and 24px (`gap-6`).
- **Dense Data Tables:** High data density (row height ~40–44px) allowing analysts to view 15–20 transactions above the fold.

## Elevation & Depth

- **Tonal Layering:** Depth is communicated primarily through subtle background luminance shifts (`--background` $\to$ `--surface` $\to$ `--surface-elevated`) rather than heavy dropshadows.
- **Directional Shadows:** When interactive elements lift on hover (`card-interactive`), shadows use soft blur and subtle vertical offset: `box-shadow: 0 8px 24px -6px rgba(0, 0, 0, 0.25)`.
- **Glow Accents:** Low-opacity, diffuse semantic halos (`glow-allow`, `glow-block`) accentuate critical risk badges without blinding the operator.

## Shapes

- **Borders & Radii:** Consistent geometry across all modules:
  - Badges & Pills: `rounded-full` or `rounded-md` (4px–6px).
  - Cards & Panels: `rounded-lg` (8px).
  - Modals & Drawers: `rounded-xl` (12px).
- **Border Discipline:** Standard 1px solid borders (`border border-[var(--border)]`). Avoid thick 2px+ borders or heavy colored side strips (`border-l-4`).

## Components

### 1. Header
- Displays live connectivity state (Gateway Live / Backend Offline / Demo Simulation).
- Model artifact metadata badge (`Phase 3 | XGBoost + CLIP`).
- Global search / command palette trigger (`Cmd + K`).
- Quick theme toggle with instant CSS variable swap.

### 2. Sidebar Navigation
- Structured by operational hierarchy: *Overview*, *Risk Operations*, *Forensics & Intelligence*, *Science & Governance*.
- Active route highlighted with subtle surface tint and high-contrast icon.

### 3. StatCard
- Single-concept risk metrics with tabular numeral values and comparison deltas.
- Clean typography without redundant decorative containers.

### 4. DecisionBadge
- Compact, high-visibility status pill rendering `ALLOW`, `REVIEW`, `HOLD`, or `BLOCK` with matching dot indicator.

### 5. Transaction Detail Drawer
- Right-hand slide-out drawer providing 360-degree forensic context: buyer/seller metadata, device graph, and TreeSHAP attribution waterfall.

## Do's and Don'ts

### Do:
- Always use `tabular-nums` for timestamps, transaction amounts, and probabilities.
- Anchor every high-risk score to an explicit reason code and TreeSHAP feature attribution.
- Provide clear loading skeletons (`LoadingSkeleton`) and meaningful empty states (`EmptyState`) for all feeds.
- Ensure all interactive buttons, rows, and toggles have visible `:focus-visible` focus rings (`.focus-ring`).

### Don't:
- Never use gradient text for headings or titles.
- Never place small uppercase "eyebrow" kickers above headings.
- Never use raw monospace fonts for ordinary UI labels or prose.
- Never use saturated colored side-borders (`border-l-4`) on cards or alerts.
- Never rely solely on color to communicate risk; always pair colors with text labels (`ALLOW`, `BLOCK`).

