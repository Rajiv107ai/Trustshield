"use client";

import React, { useState, useEffect } from "react";
import { TrustShieldApi } from "@/lib/api/client";
import { ListingScoreRequest, ListingScoreResponse } from "@/lib/types/api";
import { useViewMode } from "@/context/ViewModeContext";
import {
  Tag,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Layers,
  Zap,
  Sparkles,
  Microscope,
  Check,
  ShieldAlert,
  ArrowRight,
  ExternalLink,
} from "lucide-react";

interface ListingScenarioPreset {
  id: string;
  name: string;
  badge: string;
  badgeColor: "rose" | "amber" | "emerald";
  description: string;
  payload: ListingScoreRequest;
  sellerImage: string;
  sellerImageTitle: string;
  catalogImage: string;
  catalogImageTitle: string;
  originalMerchant: string;
}

const LISTING_PRESETS: ListingScenarioPreset[] = [
  {
    id: "preset_stolen_catalog_dslr",
    name: "Stolen Catalog Asset (Counterfeit DSLR)",
    badge: "IMAGE THEFT COLLISION",
    badgeColor: "rose",
    description: "New 3-day-old seller re-uploaded official manufacturer studio photo at 75% discount. Self-match unauthorized.",
    payload: {
      listing_id: "LISTING_CANON_501",
      product_id: "PROD_B07XYZ99",
      displayed_product_id: "PROD_B07XYZ99_STOLEN",
      price: 450.0,
      base_price: 1899.0,
      category_median_price: 1850.0,
      seller_age_days_at_listing: 3,
      seller_listings_before: 1,
      multimodal_similarity_score: 0.948,
    },
    sellerImage: "/images/camera_seller_upload.jpg",
    sellerImageTitle: "Seller Upload: Unlicensed Manufacturer Studio Photo",
    catalogImage: "/images/camera_catalog_match.jpg",
    catalogImageTitle: "Official Catalog Master Asset #B07XYZ99",
    originalMerchant: "CANON_OFFICIAL_GLOBAL",
  },
  {
    id: "preset_bait_switch_accessory",
    name: "Bait & Switch Scam (Tech -> Bulk Cable)",
    badge: "TEXT-IMAGE MISMATCH",
    badgeColor: "amber",
    description: "Listing title claims $1,200 hardware, but uploaded photo is a cheap bulk USB wire in plastic sleeve.",
    payload: {
      listing_id: "LISTING_BAIT_SWITCH_102",
      product_id: "PROD_LAPTOP_991",
      displayed_product_id: "PROD_GENERIC_WIRE_01",
      price: 299.0,
      base_price: 1200.0,
      category_median_price: 1150.0,
      seller_age_days_at_listing: 5,
      seller_listings_before: 2,
      multimodal_similarity_score: 0.182,
    },
    sellerImage: "/images/fake_accessory.jpg",
    sellerImageTitle: "Uploaded Photo: Generic Bulk Wire in Baggy",
    catalogImage: "/images/camera_catalog_match.jpg",
    catalogImageTitle: "Claimed Catalog Item: Pro Studio Equipment",
    originalMerchant: "PREMIUM_ELECTRONICS_FLAGSHIP",
  },
  {
    id: "preset_authentic_authorized_merchant",
    name: "Authorized Reseller (Authentic Verified)",
    badge: "VERIFIED AUTHENTIC",
    badgeColor: "emerald",
    description: "Established 420-day merchant with 150+ successful listings, valid pricing, and authorized catalog licensing.",
    payload: {
      listing_id: "LISTING_AUTHENTIC_204",
      product_id: "PROD_B07XYZ99",
      displayed_product_id: "PROD_B07XYZ99",
      price: 1820.0,
      base_price: 1899.0,
      category_median_price: 1850.0,
      seller_age_days_at_listing: 420,
      seller_listings_before: 154,
      multimodal_similarity_score: 0.892,
    },
    sellerImage: "/images/camera_seller_upload.jpg",
    sellerImageTitle: "Merchant Product Photograph (Authorized Stockist)",
    catalogImage: "/images/camera_catalog_match.jpg",
    catalogImageTitle: "Official Catalog Master Asset #B07XYZ99",
    originalMerchant: "CANON_OFFICIAL_GLOBAL",
  },
];

export default function ListingIntelligencePage() {
  const { viewMode } = useViewMode();
  const [activePreset, setActivePreset] = useState<ListingScenarioPreset>(LISTING_PRESETS[0]);
  const [formData, setFormData] = useState<ListingScoreRequest>(LISTING_PRESETS[0].payload);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ListingScoreResponse | null>(null);
  const [actionSuccess, setActionSuccess] = useState<string | null>(null);

  const executeAudit = async (payload: ListingScoreRequest) => {
    setLoading(true);
    setActionSuccess(null);
    try {
      const res = await TrustShieldApi.analyzeListing(payload);
      setResult(res.data);
    } catch (err) {
      console.error("Listing analysis failed:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    executeAudit(LISTING_PRESETS[0].payload);
  }, []);

  const selectPreset = (preset: ListingScenarioPreset) => {
    setActivePreset(preset);
    setFormData(preset.payload);
    executeAudit(preset.payload);
  };

  const handleAnalyze = (e: React.FormEvent) => {
    e.preventDefault();
    executeAudit(formData);
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-[#E8EDF3] flex items-center gap-2">
            <Tag className="w-5 h-5 text-cyan-400" />
            <span>Multimodal Listing &amp; Counterfeit Intelligence</span>
            {viewMode === "executive" ? (
              <span className="text-xs px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20 font-normal">
                Brand Protection View
              </span>
            ) : (
              <span className="text-xs px-2 py-0.5 rounded-full bg-purple-500/10 text-purple-400 border border-purple-500/20 font-normal">
                CLIP ViT-B/32 Vector Inspector
              </span>
            )}
          </h1>
          <p className="text-xs text-[#8995A3] mt-1">
            Vision-language CLIP representations and sub-millisecond FAISS vector similarity detecting stolen imagery and counterfeit listings.
          </p>
        </div>

        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[#111821] border border-[#202A35] text-xs font-mono">
          <span className="text-[#8995A3]">Index:</span>
          <span className="text-cyan-400 font-semibold">FAISS Inner Product (Normalized Cosine)</span>
        </div>
      </div>

      {/* Preset Scenario Selector */}
      <div className="space-y-2">
        <div className="text-xs font-semibold text-[#8995A3] uppercase tracking-wider flex items-center justify-between">
          <span>Test Scenarios (Pre-configured Multimodal Audits)</span>
          <span className="text-[10px] font-mono text-[#596574]">Click to execute instant audit</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {LISTING_PRESETS.map((p) => {
            const isSelected = activePreset.id === p.id;
            return (
              <button
                key={p.id}
                onClick={() => selectPreset(p)}
                className={`p-3.5 rounded-xl border text-left transition-all ${
                  isSelected
                    ? "bg-[#151D27] border-cyan-500/50 ring-1 ring-cyan-500/30 shadow-md shadow-cyan-500/10"
                    : "bg-[#111821] border-[#202A35] hover:border-[#2E3C4D] hover:bg-[#131B24]"
                }`}
              >
                <div className="flex items-center justify-between">
                  <span
                    className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border ${
                      p.badgeColor === "rose"
                        ? "bg-rose-500/10 text-rose-400 border-rose-500/20"
                        : p.badgeColor === "amber"
                        ? "bg-amber-500/10 text-amber-400 border-amber-500/20"
                        : "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
                    }`}
                  >
                    {p.badge}
                  </span>
                  {isSelected && <span className="text-xs font-mono font-bold text-cyan-400">ACTIVE</span>}
                </div>
                <div className="text-xs font-semibold text-[#E8EDF3] mt-2">{p.name}</div>
                <div className="text-[11px] text-[#8995A3] mt-1 line-clamp-2 leading-relaxed">{p.description}</div>
              </button>
            );
          })}
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

            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="text-[11px] text-[#8995A3] block mb-1">Catalog Base Price ($)</label>
                <input
                  type="number"
                  value={formData.base_price || 0}
                  onChange={(e) => setFormData({ ...formData, base_price: parseFloat(e.target.value) || 0 })}
                  className="w-full bg-[#0E131A] border border-[#202A35] rounded-md px-3 py-1.5 text-[#E8EDF3] focus:border-cyan-500 focus:outline-none"
                />
              </div>

              <div>
                <label className="text-[11px] text-[#8995A3] block mb-1">Seller Prior Listings</label>
                <input
                  type="number"
                  value={formData.seller_listings_before ?? 1}
                  onChange={(e) => setFormData({ ...formData, seller_listings_before: parseInt(e.target.value) || 0 })}
                  className="w-full bg-[#0E131A] border border-[#202A35] rounded-md px-3 py-1.5 text-[#E8EDF3] focus:border-cyan-500 focus:outline-none"
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full mt-2 py-2.5 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-xs font-bold text-white shadow-lg shadow-cyan-600/20 transition-all flex items-center justify-center gap-2 cursor-pointer"
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
              className={`p-6 rounded-xl border transition-all ${
                result.fake_listing_probability > 0.5
                  ? "bg-rose-500/10 border-rose-500/30 glow-block"
                  : "bg-emerald-500/10 border-emerald-500/30 glow-allow"
              }`}
            >
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                <div>
                  <div className="flex items-center gap-2.5">
                    <span className="text-xl font-bold font-mono text-[#E8EDF3]">
                      {result.fake_listing_probability > 0.5 ? "SUSPICIOUS / COUNTERFEIT LISTING" : "VERIFIED AUTHENTIC LISTING"}
                    </span>
                    <span
                      className={`text-xs px-2.5 py-0.5 rounded-full font-mono font-bold border ${
                        result.fake_listing_probability > 0.5
                          ? "bg-rose-500/20 text-rose-300 border-rose-500/30"
                          : "bg-emerald-500/20 text-emerald-300 border-emerald-500/30"
                      }`}
                    >
                      Risk: {(result.fake_listing_probability * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="text-xs text-[#8995A3] mt-1.5 font-mono">
                    Model: <span className="text-cyan-400">{result.model_used}</span> · CLIP Engine:{" "}
                    <span className="text-emerald-400 font-bold">{result.clip_scored ? "ViT-B/32 Active" : "TF-IDF Fallback"}</span>
                  </div>
                </div>

                <div className="text-left sm:text-right">
                  <div className="text-[10px] text-[#8995A3] font-mono uppercase tracking-wider">
                    FAISS Cosine Similarity
                  </div>
                  <div
                    className={`text-3xl font-bold font-mono mt-1 ${
                      result.multimodal_similarity_score > 0.90 && result.fake_listing_probability > 0.5
                        ? "text-rose-400"
                        : result.multimodal_similarity_score < 0.35
                        ? "text-amber-400"
                        : "text-emerald-400"
                    }`}
                  >
                    {(result.multimodal_similarity_score * 100).toFixed(1)}%
                  </div>
                  <div className="text-[10px] text-[#8995A3] font-mono mt-0.5">
                    {result.multimodal_similarity_score > 0.90 && result.fake_listing_probability > 0.5
                      ? "Stolen Image Collision"
                      : result.multimodal_similarity_score < 0.35
                      ? "Visual-Text Mismatch"
                      : "Authorized Photo Alignment"}
                  </div>
                </div>
              </div>
            </div>

            {/* Side-by-Side Visual Comparison Cards with Real Images */}
            <div className="p-5 rounded-xl border border-[#202A35] bg-[#111821] space-y-4">
              <div className="flex items-center justify-between">
                <div className="text-xs font-bold uppercase tracking-wider text-[#E8EDF3] flex items-center gap-2">
                  <Layers className="w-4 h-4 text-cyan-400" />
                  <span>Visual Similarity &amp; Catalog Evidence</span>
                </div>
                <div className="text-[11px] font-mono text-[#8995A3]">
                  512-Dimensional CLIP Embedding Projection
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Submitted Listing */}
                <div className="p-4 rounded-lg bg-[#0E131A] border border-[#202A35] space-y-3">
                  <div className="flex items-center justify-between text-xs font-mono">
                    <span className="text-[#8995A3]">SUBMITTED SELLER UPLOAD</span>
                    <span
                      className={`font-semibold ${
                        result.fake_listing_probability > 0.5 ? "text-rose-400" : "text-emerald-400"
                      }`}
                    >
                      {result.fake_listing_probability > 0.5 ? "Flagged Asset" : "Clean Asset"}
                    </span>
                  </div>
                  <div className="relative h-48 rounded-lg overflow-hidden border border-[#202A35] bg-black flex items-center justify-center">
                    <img
                      src={activePreset.sellerImage}
                      alt="Submitted Seller Asset"
                      className="w-full h-full object-contain"
                    />
                    <div className="absolute top-2 right-2 px-2 py-0.5 rounded bg-black/80 backdrop-blur text-[10px] font-mono text-cyan-300 border border-cyan-500/30">
                      Seller Upload
                    </div>
                  </div>
                  <div className="text-[11px] font-mono text-[#8995A3]">
                    Caption: <span className="text-[#E8EDF3]">{activePreset.sellerImageTitle}</span>
                  </div>
                </div>

                {/* Catalog Item Match */}
                <div className="p-4 rounded-lg bg-[#0E131A] border border-[#202A35] space-y-3">
                  <div className="flex items-center justify-between text-xs font-mono">
                    <span className="text-[#8995A3]">CATALOG REFERENCE MATCH</span>
                    <span className="text-cyan-400 font-semibold">
                      {(result.multimodal_similarity_score * 100).toFixed(1)}% Visual Match
                    </span>
                  </div>
                  <div className="relative h-48 rounded-lg overflow-hidden border border-[#202A35] bg-black flex items-center justify-center">
                    <img
                      src={activePreset.catalogImage}
                      alt="Official Catalog Asset"
                      className="w-full h-full object-contain"
                    />
                    <div className="absolute top-2 right-2 px-2 py-0.5 rounded bg-black/80 backdrop-blur text-[10px] font-mono text-emerald-300 border border-emerald-500/30">
                      Official Catalog
                    </div>
                  </div>
                  <div className="text-[11px] font-mono text-[#8995A3]">
                    Original Merchant: <span className="text-blue-400">{activePreset.originalMerchant}</span>
                  </div>
                </div>
              </div>

              {/* Dynamic Visual Alignment Gauge */}
              <div className="p-3.5 rounded-lg bg-[#0E131A] border border-[#202A35] space-y-2">
                <div className="flex items-center justify-between text-xs font-mono">
                  <span className="text-[#8995A3]">Visual Hash Alignment Spectrum</span>
                  <span className="text-[#E8EDF3] font-semibold">
                    Similarity: {(result.multimodal_similarity_score * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="h-2 rounded-full bg-[#151D27] overflow-hidden flex">
                  <div
                    style={{ width: `${Math.min(result.multimodal_similarity_score * 100, 100)}%` }}
                    className={`h-full transition-all duration-500 ${
                      result.fake_listing_probability > 0.5 && result.multimodal_similarity_score > 0.9
                        ? "bg-rose-500"
                        : result.multimodal_similarity_score < 0.4
                        ? "bg-amber-500"
                        : "bg-emerald-500"
                    }`}
                  />
                </div>
                <div className="flex justify-between text-[10px] font-mono text-[#596574]">
                  <span>0.0 (Unrelated)</span>
                  <span>0.50 (Threshold)</span>
                  <span>1.0 (Exact Image Duplicate)</span>
                </div>
              </div>

              {/* Dynamic Forensic Narrative & Key Evidence */}
              <div className="p-4 rounded-lg bg-[#0E131A] border border-[#202A35] space-y-3">
                <div className="flex items-center justify-between">
                  <div className="font-semibold font-mono text-xs flex items-center gap-2 text-[#E8EDF3]">
                    {result.fake_listing_probability > 0.5 ? (
                      <AlertTriangle className="w-4 h-4 text-rose-400" />
                    ) : (
                      <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                    )}
                    <span>Dynamic Investigator Narrative (Evidence Grounded)</span>
                  </div>
                  <span
                    className={`text-[10px] font-mono px-2 py-0.5 rounded border font-semibold ${
                      result.fake_listing_probability > 0.5
                        ? "bg-rose-500/10 text-rose-400 border-rose-500/20"
                        : "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
                    }`}
                  >
                    {result.fake_listing_probability > 0.5 ? "ACTION: QUARANTINE" : "ACTION: VERIFIED SAFE"}
                  </span>
                </div>

                {/* Evidence bullets if provided by API */}
                {Array.isArray(result.investigator_narrative?.key_evidence) &&
                result.investigator_narrative.key_evidence.length > 0 ? (
                  <ul className="space-y-1.5 text-xs text-[#8995A3] pl-2">
                    {result.investigator_narrative.key_evidence.map((ev: unknown, i: number) => (
                      <li key={i} className="flex items-start gap-2">
                        <span className="text-cyan-400 mt-1">•</span>
                        <span>{String(ev)}</span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-xs text-[#8995A3] leading-relaxed">
                    {typeof result.investigator_narrative?.summary === "string"
                      ? result.investigator_narrative.summary
                      : result.fake_listing_probability > 0.5
                      ? `The listing imagery submitted by this ${formData.seller_age_days_at_listing}-day-old merchant matches an existing official catalog image with ${(result.multimodal_similarity_score * 100).toFixed(1)}% cosine similarity. Self-match exclusion was asserted: seller is not authorized for catalog asset reuse.`
                      : `Product image aligns with legitimate merchant inventory baseline (${(result.multimodal_similarity_score * 100).toFixed(1)}% similarity). Seller historical tenure (${formData.seller_age_days_at_listing} days) confirms authentic distributor relationship.`}
                  </p>
                )}

                {/* Recommended action */}
                <div className="pt-2 border-t border-[#202A35] flex items-center justify-between text-xs">
                  <span className="text-[#8995A3] font-mono">Recommended Policy Action:</span>
                  <span className="text-[#E8EDF3] font-semibold">
                    {typeof result.investigator_narrative?.recommended_action === "string"
                      ? result.investigator_narrative.recommended_action
                      : result.fake_listing_probability > 0.5
                      ? "Quarantine listing from marketplace and request distributor invoices."
                      : "Approve listing with regular monitoring."}
                  </span>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="pt-2 flex flex-col sm:flex-row gap-3">
                {result.fake_listing_probability > 0.5 ? (
                  <>
                    <button
                      onClick={() => setActionSuccess("Listing quarantined from search results. Merchant payout held.")}
                      className="flex-1 py-2 px-3 rounded-lg bg-rose-600 hover:bg-rose-500 text-xs font-semibold text-white transition-all flex items-center justify-center gap-2 cursor-pointer shadow-lg shadow-rose-600/20"
                    >
                      <XCircle className="w-4 h-4" />
                      <span>Quarantine Listing &amp; Hold Payout</span>
                    </button>
                    <button
                      onClick={() => setActionSuccess("Automated RFQ sent to merchant requesting authorization invoices.")}
                      className="py-2 px-3 rounded-lg bg-[#151D27] hover:bg-[#1C2633] border border-[#202A35] text-xs font-semibold text-[#E8EDF3] transition-all flex items-center justify-center gap-2 cursor-pointer"
                    >
                      <AlertTriangle className="w-4 h-4 text-amber-400" />
                      <span>Request Inventory Proof</span>
                    </button>
                  </>
                ) : (
                  <button
                    onClick={() => setActionSuccess("Listing approved and elevated in global marketplace search.")}
                    className="flex-1 py-2 px-3 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-xs font-semibold text-white transition-all flex items-center justify-center gap-2 cursor-pointer shadow-lg shadow-emerald-600/20"
                  >
                    <CheckCircle2 className="w-4 h-4" />
                    <span>Approve &amp; Whitelist Listing</span>
                  </button>
                )}
              </div>

              {actionSuccess && (
                <div className="p-3 rounded-lg bg-blue-500/10 border border-blue-500/30 text-xs text-blue-300 font-mono flex items-center gap-2">
                  <Check className="w-4 h-4 shrink-0 text-blue-400" />
                  <span>{actionSuccess}</span>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
