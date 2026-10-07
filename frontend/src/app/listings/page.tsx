"use client";

import React, { useState } from "react";
import { TrustShieldApi } from "@/lib/api/client";
import { ListingScoreRequest, ListingScoreResponse } from "@/lib/types/api";
import {
  Tag,
  Search,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Cpu,
  Layers,
  Image as ImageIcon,
  ArrowRight,
  ShieldAlert,
  Zap,
} from "lucide-react";

export default function ListingIntelligencePage() {
  const [formData, setFormData] = useState<ListingScoreRequest>({
    listing_id: "LISTING_CANON_501",
    product_id: "PROD_B07XYZ99",
    displayed_product_id: "PROD_B07XYZ99_GENERIC",
    price: 340.00,
    base_price: 350.00,
    category_median_price: 330.00,
    seller_age_days_at_listing: 3,
    seller_listings_before: 1,
    multimodal_similarity_score: 0.945,
  });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ListingScoreResponse | null>({
    listing_id: "LISTING_CANON_501",
    fake_listing_probability: 0.785,
    risk_label: "high",
    model_used: "CLIP ViT-B/32 + FAISS Index (Phase 4)",
    clip_scored: true,
    multimodal_similarity_score: 0.945,
    investigator_narrative: {
      visual_collision_detected: true,
      catalog_match_id: "PROD_B07XYZ99",
      flagged_reason: "Cross-seller product photo reuse without licensing authorization",
      faiss_cosine_similarity: 0.945,
    },
  });

  const handleAnalyze = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      const res = await TrustShieldApi.analyzeListing(formData);
      setResult(res.data);
    } catch (err) {
      console.error("Listing analysis failed:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-[#E8EDF3] flex items-center gap-2">
            <Tag className="w-5 h-5 text-cyan-400" />
            <span>Multimodal Listing & Counterfeit Intelligence</span>
          </h1>
          <p className="text-xs text-[#8995A3] mt-1">
            Vision-language CLIP representations and sub-millisecond FAISS vector similarity detecting stolen imagery.
          </p>
        </div>

        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[#111821] border border-[#202A35] text-xs font-mono">
          <span className="text-[#8995A3]">Index:</span>
          <span className="text-cyan-400 font-semibold">FAISS Inner Product (Normalized Cosine)</span>
        </div>
      </div>

      {/* Main Analysis Form & Visual Results */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Col: Listing Ingestion Form */}
        <div className="p-5 rounded-xl border border-[#202A35] bg-[#111821] space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold uppercase tracking-wider text-[#E8EDF3]">
              Listing Inspector Inputs
            </span>
            <span className="text-[10px] font-mono text-cyan-400">POST /listing/analyze</span>
          </div>

          <form onSubmit={handleAnalyze} className="space-y-3 text-xs font-mono">
            <div>
              <label className="text-[11px] text-[#8995A3] block mb-1">Listing ID</label>
              <input
                type="text"
                value={formData.listing_id || ""}
                onChange={(e) => setFormData({ ...formData, listing_id: e.target.value })}
                className="w-full bg-[#0E131A] border border-[#202A35] rounded-md px-3 py-1.5 text-[#E8EDF3] focus:border-cyan-500 focus:outline-none"
              />
            </div>

            <div>
              <label className="text-[11px] text-[#8995A3] block mb-1">Product ID (Claimed)</label>
              <input
                type="text"
                value={formData.product_id || ""}
                onChange={(e) => setFormData({ ...formData, product_id: e.target.value })}
                className="w-full bg-[#0E131A] border border-[#202A35] rounded-md px-3 py-1.5 text-[#E8EDF3] focus:border-cyan-500 focus:outline-none"
              />
            </div>

            <div>
              <label className="text-[11px] text-[#8995A3] block mb-1">Displayed Product ID</label>
              <input
                type="text"
                value={formData.displayed_product_id || ""}
                onChange={(e) => setFormData({ ...formData, displayed_product_id: e.target.value })}
                className="w-full bg-[#0E131A] border border-[#202A35] rounded-md px-3 py-1.5 text-[#E8EDF3] focus:border-cyan-500 focus:outline-none"
              />
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="text-[11px] text-[#8995A3] block mb-1">Listing Price ($)</label>
                <input
                  type="number"
                  value={formData.price || 0}
                  onChange={(e) => setFormData({ ...formData, price: parseFloat(e.target.value) || 0 })}
                  className="w-full bg-[#0E131A] border border-[#202A35] rounded-md px-3 py-1.5 text-[#E8EDF3] focus:border-cyan-500 focus:outline-none"
                />
              </div>

              <div>
                <label className="text-[11px] text-[#8995A3] block mb-1">Seller Age (Days)</label>
                <input
                  type="number"
                  value={formData.seller_age_days_at_listing || 0}
                  onChange={(e) => setFormData({ ...formData, seller_age_days_at_listing: parseInt(e.target.value) || 0 })}
                  className="w-full bg-[#0E131A] border border-[#202A35] rounded-md px-3 py-1.5 text-[#E8EDF3] focus:border-cyan-500 focus:outline-none"
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full mt-2 py-2.5 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-xs font-bold text-white shadow-lg shadow-cyan-600/20 transition-all flex items-center justify-center gap-2"
            >
              <Zap className="w-4 h-4" />
              <span>{loading ? "Computing CLIP Embeddings..." : "Execute Multimodal Audit"}</span>
            </button>
          </form>
        </div>

        {/* Right 2 Cols: Multimodal Visual Evidence */}
        {result && (
          <div className="lg:col-span-2 space-y-6">
            {/* Verdict Card */}
            <div
              className={`p-6 rounded-xl border ${
                result.fake_listing_probability > 0.5
                  ? "bg-rose-500/10 border-rose-500/30 glow-block"
                  : "bg-emerald-500/10 border-emerald-500/30 glow-allow"
              }`}
            >
              <div className="flex items-center justify-between">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="text-xl font-bold font-mono text-[#E8EDF3]">
                      {result.fake_listing_probability > 0.5 ? "SUSPICIOUS LISTING" : "AUTHENTIC LISTING"}
                    </span>
                    <span className="text-xs px-2 py-0.5 rounded-full bg-white/10 font-mono text-white">
                      Risk: {(result.fake_listing_probability * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="text-xs text-[#8995A3] mt-1 font-mono">
                    Model: <span className="text-cyan-400">{result.model_used}</span> · Scored via Real CLIP:{" "}
                    <span className="text-emerald-400 font-bold">{result.clip_scored ? "YES" : "FALLBACK"}</span>
                  </div>
                </div>

                <div className="text-right">
                  <div className="text-[10px] text-[#8995A3] font-mono">FAISS COSINE SIMILARITY</div>
                  <div className="text-3xl font-bold font-mono text-cyan-400 mt-1">
                    {(result.multimodal_similarity_score * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
            </div>

            {/* Side-by-Side Visual Comparison Cards */}
            <div className="p-5 rounded-xl border border-[#202A35] bg-[#111821] space-y-4">
              <div className="text-xs font-bold uppercase tracking-wider text-[#E8EDF3]">
                Visual Similarity & Catalog Evidence
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Submitted Listing */}
                <div className="p-4 rounded-lg bg-[#0E131A] border border-[#202A35] space-y-3">
                  <div className="flex items-center justify-between text-xs font-mono">
                    <span className="text-[#8995A3]">SUBMITTED LISTING IMAGE</span>
                    <span className="text-rose-400 font-semibold">Under Inspection</span>
                  </div>
                  <div className="h-40 rounded-lg bg-[#151D27] border border-[#202A35] flex flex-col items-center justify-center text-[#596574]">
                    <ImageIcon className="w-10 h-10 text-cyan-500/40" />
                    <span className="text-xs mt-2 font-mono text-[#8995A3]">Seller Upload Ref #4029</span>
                  </div>
                  <div className="text-[11px] font-mono text-[#8995A3]">
                    Title: <span className="text-[#E8EDF3]">Brand New High-End Electronics (Sealed)</span>
                  </div>
                </div>

                {/* Stolen Catalog Item Match */}
                <div className="p-4 rounded-lg bg-[#0E131A] border border-[#202A35] space-y-3">
                  <div className="flex items-center justify-between text-xs font-mono">
                    <span className="text-[#8995A3]">CATALOG REFERENCE MATCH</span>
                    <span className="text-cyan-400 font-semibold">94.5% Visual Match</span>
                  </div>
                  <div className="h-40 rounded-lg bg-[#151D27] border border-[#202A35] flex flex-col items-center justify-center text-[#596574]">
                    <ImageIcon className="w-10 h-10 text-blue-500/40" />
                    <span className="text-xs mt-2 font-mono text-[#8995A3]">Official Catalog Item #B07XYZ99</span>
                  </div>
                  <div className="text-[11px] font-mono text-[#8995A3]">
                    Original Merchant: <span className="text-blue-400">SELLER_AUTHORISED_OFFICIAL</span>
                  </div>
                </div>
              </div>

              {/* Forensic Narrative */}
              <div className="p-3.5 rounded-lg bg-[#0E131A] border border-[#202A35] text-xs">
                <div className="font-semibold text-rose-300 font-mono flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 text-rose-400" />
                  <span>Investigator Narrative (Deterministic Anti-Hallucination Guard)</span>
                </div>
                <p className="text-[11px] text-[#8995A3] mt-1.5 leading-relaxed">
                  The listing imagery submitted by this 3-day-old merchant matches an existing official catalog image with 94.5% cosine similarity. Self-match exclusion was asserted; merchant is not authorized for catalog reuse. Recommendation: quarantine listing from search results.
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
