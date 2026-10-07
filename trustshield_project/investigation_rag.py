"""Forensic Knowledge Base & Retrieval-Augmented Generation (RAG) for TrustShield Fraud Ops.

Maintains indexed operational guidelines, policy rules, and feature definitions.
Enables the Investigation Agent to retrieve factual policy context to support its forensic analysis.
CRITICAL CONSTRAINT: RAG provides context for human explanation only; it never predicts fraud risk.
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass




@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    title: str
    category: str      # 'policy' | 'typology' | 'feature_definition'
    content: str
    keywords: List[str]


# Pre-indexed knowledge corpus for e-commerce fraud investigations
CORPUS: List[KnowledgeChunk] = [
    KnowledgeChunk(
        chunk_id="POL-01",
        title="Hardware Identifier & Device Collision Policy",
        category="policy",
        content=(
            "When 2 or more buyer accounts transact using the identical hardware fingerprint "
            "within 72 hours, transactions must be subjected to step-up multi-factor verification. "
            "If accounts share a physical delivery destination, freeze payouts and initiate manual review."
        ),
        keywords=["device", "hardware", "collision", "sharing", "fingerprint", "ring"],
    ),
    KnowledgeChunk(
        chunk_id="POL-02",
        title="Excessive Return Abuse & Speed-to-Return Thresholds",
        category="policy",
        content=(
            "A buyer return velocity exceeding 50% across >=3 orders, specifically with returns filed "
            "within 1 to 3 days of delivery, indicates wardrobing or refund manipulation. "
            "Tag account as HIGH_RETURN_VELOCITY and require receipt of physical merchandise before refunds."
        ),
        keywords=["return", "abuse", "velocity", "refund", "buyer_returns_before"],
    ),
    KnowledgeChunk(
        chunk_id="POL-03",
        title="Multimodal Listing Integrity & Brand Protection",
        category="policy",
        content=(
            "Listings exhibiting CLIP image-text cosine similarity < 0.45 or cross-seller visual "
            "reuse with a price undercut > 40% below category median are flagged as counterfeit risk. "
            "Suspend listing from public search pending merchant invoice validation."
        ),
        keywords=["listing", "clip", "counterfeit", "image", "mismatch", "fake", "similarity"],
    ),
    KnowledgeChunk(
        chunk_id="POL-04",
        title="Cold-Start New Account Review Guidelines",
        category="policy",
        content=(
            "Entities with fewer than 3 historical interactions exhibit high epistemic uncertainty. "
            "If transaction amount exceeds $150.00 for a cold-start buyer, hold transaction for automated "
            "address verification. Do not auto-block without negative behavioral evidence."
        ),
        keywords=["cold_start", "history", "new", "unvetted", "insufficient"],
    ),
    KnowledgeChunk(
        chunk_id="TYP-01",
        title="Merchant-Buyer Collusion & Cashback Extraction",
        category="typology",
        content=(
            "Collusion rings establish a shell seller account, funnel orders from synthetic buyer accounts, "
            "and immediately issue refunds or capture promotional marketplace coupons. "
            "Key indicator is a high seller-buyer concentration HHI (> 0.60) with zero customer reviews."
        ),
        keywords=["collusion", "merchant", "hhi", "concentration", "shell"],
    ),
    KnowledgeChunk(
        chunk_id="DEF-01",
        title="Conformal Uncertainty & Shannon Entropy Definitions",
        category="feature_definition",
        content=(
            "Shannon entropy measures predictive uncertainty in [0, 1]. A conformal prediction set "
            "of {0, 1} indicates that neither legitimate nor fraud classes can be excluded at the "
            "95% confidence level, requiring human investigation."
        ),
        keywords=["conformal", "entropy", "uncertainty", "confidence", "disagreement"],
    ),
]


class ForensicRAGIndex:
    """Keyword & semantic retrieval index over fraud policy documentation."""

    def __init__(self, corpus: Optional[List[KnowledgeChunk]] = None):
        self.corpus = corpus or CORPUS

    def search(self, query: str, top_k: int = 3) -> List[KnowledgeChunk]:
        """Retrieve top-k relevant knowledge chunks based on query terms."""
        q_tokens = set(query.lower().replace("_", " ").split())
        scored: List[Tuple[float, KnowledgeChunk]] = []

        for chunk in self.corpus:
            score = 0.0
            # Keyword matches
            for kw in chunk.keywords:
                if kw in q_tokens or any(kw in t for t in q_tokens):
                    score += 2.0
            # Content matches
            c_text = chunk.content.lower()
            for token in q_tokens:
                if len(token) > 3 and token in c_text:
                    score += 1.0

            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:top_k]]
