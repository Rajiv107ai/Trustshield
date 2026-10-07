# TrustShield AI — Documentation & Governance Hub

This directory serves as the centralized repository for all engineering audits, scientific research reports, regulatory model cards, architectural specifications, and bug tracking inventories across the **TrustShield AI** platform.

---

## 🧭 Master Documentation Index

### 🛠️ 1. Technical Repair & Verification (Current State)
- [**Master Technical Repair Report (`FINAL_REPAIR_REPORT.md`)**](FINAL_REPAIR_REPORT.md)  
  *Authoritative record of the engineering repair campaign.* Documents the resolution of all 22 cataloged defects (`TS-001` to `TS-022`), temporal leakage elimination, production probability calibration, and empirical verification (192 passing tests across unit, regression, serving, and mesh seeder suites).
- [**Repair Baseline State (`REPAIR_BASELINE.md`)**](REPAIR_BASELINE.md)  
  Pre-repair vulnerability baseline, checklist, and architectural baseline.
- [**Tracked Bug Registry (`BUG_INVENTORY.json`)**](BUG_INVENTORY.json)  
  Machine-readable JSON registry of tracked defects, files modified, and verification evidence.

---

### 🔬 2. Scientific Audits & Empirical Benchmarks
- [**49-Point Scientific Re-Audit (`49_POINT_REAUDIT.md`)**](49_POINT_REAUDIT.md)  
  Exhaustive line-by-line audit examining statistical validity, temporal leakage, graph isolation, and mathematical proofs.
- [**Final Before/After Benchmark Report (`FINAL_BEFORE_AFTER.md`)**](FINAL_BEFORE_AFTER.md)  
  Empirical comparison of legacy vs. repaired pipelines under strict chronological test set evaluation.
- [**Phase 0: Baseline Audit (`BASELINE_AUDIT.md`)**](BASELINE_AUDIT.md)  
  Initial codebase inspection, identifying early architectural shortcomings and data leakage risks.
- [**Full Project Audit Baseline (`FULL_PROJECT_AUDIT_BASELINE.md`)**](FULL_PROJECT_AUDIT_BASELINE.md)  
  Holistic whole-project inventory across all modules and scripts.
- [**Full Project Audit Report (`FULL_PROJECT_AUDIT_REPORT.md`)**](FULL_PROJECT_AUDIT_REPORT.md)  
  Detailed findings from full-system static and dynamic verification.

---

### 🏛️ 3. Advanced Research & Architecture
- [**Phase 2: Final Advanced Research Report (`PHASE2_FINAL_REPORT.md`)**](PHASE2_FINAL_REPORT.md)  
  Comprehensive mathematical synthesis of Heterogeneous GNNs (`HeteroData`), $Time2Vec$ continuous-time edge learning, multimodal CLIP/FAISS vector retrieval, and Conformal Prediction uncertainty quantification.
- [**Phase 2: Pre-Implementation Baseline (`PHASE2_BASELINE.md`)**](PHASE2_BASELINE.md)  
  Baseline state before deploying Phase 2 advanced research algorithms.
- [**Phase 2: Production Roadmap (`PHASE2_ROADMAP.md`)**](PHASE2_ROADMAP.md)  
  Strategic engineering design for high-throughput streaming (Apache Kafka, Apache Flink) and enterprise graph databases (Neo4j).
- [**Architecture Blueprint (`architecture.md`)**](architecture.md) & [**Design Specifications (`design.md`)**](design.md)  
  System topology, component interaction patterns, and data flow specifications.

---

### 📋 4. Governance, Compliance & Safety
- [**System Model Card (`MODEL_CARD.md`)**](MODEL_CARD.md)  
  Standardized AI governance card detailing intended use, prohibited applications, training data distributions, fairness evaluations, and ethical considerations.
- [**Technical Limitations & Disclosure (`LIMITATIONS.md`)**](LIMITATIONS.md)  
  Explicit boundary conditions, cold-start limitations, throughput boundaries, and operational assumptions.
- [**Testing Protocols (`testing.md`)**](testing.md)  
  Guidelines for running unit, integration, temporal leakage, and regression test suites.
- [**Project Rules & Conventions (`rules.md`)**](rules.md)  
  Coding standards, temporal safety invariants, and commit conventions.

---

### 🎨 5. Enterprise Frontend & UX Specification
- [**Frontend Master Prompts Package (`../frontend_master_prompts/README.md`)**](../frontend_master_prompts/README.md)  
  Production-grade frontend specifications for autonomous UI creation:
  - `00_MASTER_FRONTEND_PROMPT.md`: 12 enterprise console views with strict Zero-Fake-Data backend grounding.
  - `01_TECH_STACK_AND_TOKENS.md`: Dark console palette, decision color tokens, and JetBrains Mono typography.
  - `02_API_SCHEMAS_TYPESCRIPT.md`: TypeScript contracts mirroring 100% of Pydantic v2 schemas.
  - `03_SCENARIOS_AND_DUAL_MODE.md`: 4 interactive demo scenarios and plain-English reason translations.

