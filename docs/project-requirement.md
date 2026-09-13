# TrustShield AI — Project Requirements

## What this is
An end-to-end e-commerce fraud intelligence platform detecting coordinated
fraud across sellers, buyers, listings, orders, returns, and connected
entities using behavioral ML, multimodal AI, and graph-based fraud
detection. Built primarily as a **resume-worthy AI/ML/Data Science
portfolio project** — not primarily a frontend project. A frontend (simple
React+Tailwind visualization layer) is a low-priority stretch goal only.

## Core idea
Unify a fake-listing detector and a return-fraud detector via a **Trust
Graph / Fraud Graph** connecting Seller, Buyer, Product, Listing, Order,
Return, Device, and Address nodes — rather than building them as isolated
models. The differentiator is combining behavioral + relational (graph) +
(stretch) multimodal signal, not any single model type.

## Primary research question
Does combining behavioral, multimodal, and graph-based relational
information improve detection of coordinated e-commerce fraud vs. isolated
transaction-level models?

Secondary research questions:
- Does the graph add value over tabular features alone?
- Does multimodal signal (image/text) add value?
- Can fraud RINGS be detected, not just individual fraudulent transactions?
- Does a GNN outperform simpler hand-engineered graph features?
- What's the recall / false-positive trade-off, and how does it change
  under different cost assumptions?

## Preferred technical stack
Python / Pandas / Scikit-learn / XGBoost / PyTorch / Transformers for ML;
NetworkX + PyTorch Geometric for graph (Neo4j only if justified at scale);
FastAPI / Pydantic / SQLAlchemy / PostgreSQL backend; SHAP for
explainability; MLflow / Docker / GitHub Actions for MLOps. Used
**selectively** — not adding technology just because it sounds advanced.

## Priorities
No immediate time crunch. Quality over speed, experiments over UI polish,
evidence over claims. Estimated ~6-9 weeks part-time across 7 phases.

## How Claude should behave on this project
Act as a senior AI/ML architect / researcher / critical technical advisor:
- Challenge weak assumptions.
- Identify technical risks proactively.
- Check for data leakage, class imbalance, and train/test methodology
  issues as a matter of course, not only when asked.
- Use only real, measured metrics — never fabricate numbers.
- For important decisions, discuss: what might be missing, the strongest
  counterargument, the biggest technical risk, and a recommended next
  action — with confidence labels (Certain / Likely / Possible).
- When information is missing, ask the single most important
  clarification rather than making many silent assumptions.
