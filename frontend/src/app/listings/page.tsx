"use client";

import React, { useState, useEffect } from "react";
import { TrustShieldApi, ApiError, generateIdempotencyKey } from "@/lib/api/client";
import { ActionReceipt, ListingScoreRequest, ListingScoreResponse } from "@/lib/types/api";
import { RISK_THRESHOLDS } from "@/lib/constants/thresholds";
import { useViewMode } from "@/context/ViewModeContext";
import {
  Tag,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Layers,
  Zap,
  Check,
  ShieldAlert,
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

const lp = (
  id: string, name: string, badge: string, badgeColor: "rose" | "amber" | "emerald",
  description: string, payload: ListingScoreRequest,
  sellerImage: string, sellerImageTitle: string,
  catalogImage: string, catalogImageTitle: string, originalMerchant: string
): ListingScenarioPreset => ({
  id, name, badge, badgeColor, description, payload,
  sellerImage, sellerImageTitle, catalogImage, catalogImageTitle, originalMerchant
});

/* BEGIN GENERATED */
const LISTING_PRESETS: ListingScenarioPreset[] = [
  lp("preset_stolen_catalog_dslr", "Stolen Catalog Asset (Counterfeit DSLR)", "IMAGE THEFT COLLISION", "rose",
    "New 3-day-old seller re-uploaded official manufacturer studio photo at 75% discount. Self-match unauthorized.",
    { listing_id: "LISTING_CANON_501", product_id: "PROD_B07XYZ99", displayed_product_id: "PROD_B07XYZ99_STOLEN", price: 450.0, base_price: 1899.0, category_median_price: 1850.0, seller_age_days_at_listing: 3, seller_listings_before: 1, multimodal_similarity_score: 0.948 },
    "/images/camera_seller_upload.jpg", "Seller Upload: Unlicensed Manufacturer Studio Photo", "/images/camera_catalog_match.jpg", "Official Catalog Master Asset #B07XYZ99", "CANON_OFFICIAL_GLOBAL"
  ),
  lp("preset_bait_switch_accessory", "Bait & Switch Scam (Tech -> Bulk Cable)", "TEXT-IMAGE MISMATCH", "amber",
    "Listing title claims $1,200 hardware, but uploaded photo is a cheap bulk USB wire in plastic sleeve.",
    { listing_id: "LISTING_BAIT_SWITCH_102", product_id: "PROD_LAPTOP_991", displayed_product_id: "PROD_GENERIC_WIRE_01", price: 299.0, base_price: 1200.0, category_median_price: 1150.0, seller_age_days_at_listing: 5, seller_listings_before: 2, multimodal_similarity_score: 0.182 },
    "/images/fake_accessory.jpg", "Uploaded Photo: Generic Bulk Wire in Baggy", "/images/camera_catalog_match.jpg", "Claimed Catalog Item: Pro Studio Equipment", "PREMIUM_ELECTRONICS_FLAGSHIP"
  ),
  lp("preset_authentic_authorized_merchant", "Authorized Reseller (Authentic Verified)", "VERIFIED AUTHENTIC", "emerald",
    "Established 420-day merchant with 150+ successful listings, valid pricing, and authorized catalog licensing.",
    { listing_id: "LISTING_AUTHENTIC_204", product_id: "PROD_B07XYZ99", displayed_product_id: "PROD_B07XYZ99", price: 1820.0, base_price: 1899.0, category_median_price: 1850.0, seller_age_days_at_listing: 420, seller_listings_before: 154, multimodal_similarity_score: 0.892 },
    "/images/camera_seller_upload.jpg", "Merchant Product Photograph (Authorized Stockist)", "/images/camera_catalog_match.jpg", "Official Catalog Master Asset #B07XYZ99", "CANON_OFFICIAL_GLOBAL"
  ),
];
/* END GENERATED */

export default function ListingIntelligencePage() {
  const { viewMode, isLive, isDemo } = useViewMode();
  const [activePreset, setActivePreset] = useState<ListingScenarioPreset>(LISTING_PRESETS[0]);
  const [formData, setFormData] = useState<ListingScoreRequest>(LISTING_PRESETS[0].payload);
  const [loading, setLoading] = useState(true);
  const [result, setResult] = useState<ListingScoreResponse | null>(null);
  const [errorState, setErrorState] = useState<{ message: string; isAuth?: boolean; status?: number } | null>(null);
  const [listingAction, setListingAction] = useState<
    | { kind: "demo"; message: string }
    | { kind: "receipt"; receipt: ActionReceipt }
    | { kind: "error"; message: string }
    | null
  >(null);
  const [listingActionLoading, setListingActionLoading] = useState(false);

  const executeAudit = async (payload: ListingScoreRequest) => {
    setLoading(true);
    setListingAction(null);
    setErrorState(null);
    try {
      const res = await TrustShieldApi.analyzeListing(payload);
      setResult(res.data);
    } catch (err: unknown) {
      console.error("Listing analysis failed:", err);
      setResult(null);
      const isAuth = err instanceof ApiError ? err.isAuthError : false;
      const status = err instanceof ApiError ? err.status : undefined;
      const message = err instanceof Error ? err.message : String(err);
      setErrorState({ message, isAuth, status });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const timer = setTimeout(() => {
      executeAudit(LISTING_PRESETS[0].payload);
    }, 0);
    return () => clearTimeout(timer);
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

  const handleQuarantineListing = async () => {
    if (isDemo) {
      setListingAction({
        kind: "demo",
        message: "Demo mode only: no live listing quarantine was submitted.",
      });
      return;
    }

    const listingId = result?.listing_id || formData.listing_id;
    if (!listingId) {
      setListingAction({ kind: "error", message: "Cannot quarantine this listing because it has no listing ID." });
      return;
    }

    setListingAction(null);
    setListingActionLoading(true);
    try {
      const receipt = await TrustShieldApi.quarantineListing(listingId, {
        case_id: `CASE_LISTING_${listingId}`,
        reason: "Operator quarantined listing after multimodal counterfeit review.",
        idempotency_key: generateIdempotencyKey("listing-quarantine"),
        target: { target_type: "listing", target_id: listingId },
        target_confirmation: true,
        policy_version: "policy.v1",
      });
      setListingAction({ kind: "receipt", receipt });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      setListingAction({ kind: "error", message: `Listing quarantine failed: ${message}` });
    } finally {
      setListingActionLoading(false);
    }
  };

  const refreshListingAction = async () => {
    if (listingAction?.kind !== "receipt") return;
    setListingActionLoading(true);
    try {
      const receipt = await TrustShieldApi.getAction(listingAction.receipt.command_id);
      setListingAction({ kind: "receipt", receipt });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      setListingAction({ kind: "error", message: `Action status lookup failed: ${message}` });
    } finally {
      setListingActionLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-[var(--text-primary)] flex items-center gap-2">
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
          <p className="text-xs text-[var(--text-secondary)] mt-1">
            Vision-language CLIP representations and sub-millisecond FAISS vector similarity detecting stolen imagery and counterfeit listings.
          </p>
        </div>

        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[var(--surface)] border border-[var(--border)] text-xs font-mono">
          <span className="text-[var(--text-secondary)]">Index:</span>
          <span className="text-cyan-400 font-semibold">FAISS Inner Product (Normalized Cosine)</span>
        </div>
      </div>

      {/* Preset Scenario Selector */}
      <div className="space-y-2">
        <div className="text-xs font-semibold text-[var(--text-secondary)] uppercase tracking-wider flex items-center justify-between">
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
                    : "bg-[var(--surface)] border-[var(--border)] hover:border-[#2E3C4D] hover:bg-[#131B24]"
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
                <div className="text-xs font-semibold text-[var(--text-primary)] mt-2">{p.name}</div>
                <div className="text-[11px] text-[var(--text-secondary)] mt-1 line-clamp-2 leading-relaxed">{p.description}</div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Main Analysis Form & Visual Results */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Col: Listing Ingestion Form */}
        <div className="p-5 rounded-xl border border-[var(--border)] bg-[var(--surface)] space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold uppercase tracking-wider text-[var(--text-primary)]">
              Listing Inspector Inputs
            </span>
            <span className="text-[10px] font-mono text-cyan-400">POST /listing/analyze</span>
          </div>

          <form onSubmit={handleAnalyze} className="space-y-3 text-xs font-mono">
            {([
              { key: "listing_id", label: "Listing ID" },
              { key: "product_id", label: "Product ID (Claimed)" },
              { key: "displayed_product_id", label: "Displayed Product ID" },
            ] as const).map(({ key, label }) => (
              <div key={key}>
                <label className="text-[11px] text-[var(--text-secondary)] block mb-1">{label}</label>
                <input
                  type="text"
                  value={(formData[key] as string) || ""}
                  onChange={(e) => setFormData({ ...formData, [key]: e.target.value })}
                  className="w-full bg-[var(--surface-elevated)] border border-[var(--border)] rounded-md px-3 py-1.5 text-[var(--text-primary)] focus:border-cyan-500 focus:outline-none"
                />
              </div>
            ))}

            <div className="grid grid-cols-2 gap-2">
              {([
                { key: "price", label: "Listing Price ($)", parse: parseFloat },
                { key: "seller_age_days_at_listing", label: "Seller Age (Days)", parse: parseInt },
                { key: "base_price", label: "Catalog Base Price ($)", parse: parseFloat },
                { key: "seller_listings_before", label: "Seller Prior Listings", parse: parseInt },
              ] as const).map(({ key, label, parse }) => (
                <div key={key}>
                  <label className="text-[11px] text-[var(--text-secondary)] block mb-1">{label}</label>
                  <input
                    type="number"
                    value={formData[key] ?? 0}
                    onChange={(e) => setFormData({ ...formData, [key]: parse(e.target.value) || 0 })}
                    className="w-full bg-[var(--surface-elevated)] border border-[var(--border)] rounded-md px-3 py-1.5 text-[var(--text-primary)] focus:border-cyan-500 focus:outline-none"
                  />
                </div>
              ))}
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
        {errorState && (
          <div className="lg:col-span-2 p-6 rounded-xl border border-rose-500/30 bg-rose-500/10 space-y-3">
            <div className="flex items-center gap-2 text-rose-400 font-bold">
              <ShieldAlert className="w-5 h-5" />
              <span>
                {errorState.isAuth
                  ? "Authentication Required (HTTP 401 / 403)"
                  : errorState.status
                  ? `Listing Analysis Failed (HTTP ${errorState.status})`
                  : "Backend Service Unavailable"}
              </span>
            </div>
            <p className="text-xs text-[var(--text-primary)] font-mono leading-relaxed">{errorState.message}</p>
            {errorState.isAuth ? (
              <p className="text-[11px] text-[var(--text-secondary)]">
                Authentication credentials were not provided or were rejected. Please configure your API key in{" "}
                <a href="/settings" className="text-cyan-400 underline font-semibold">
                  Settings
                </a>{" "}
                or switch to Demo Mode.
              </p>
            ) : (
              <p className="text-[11px] text-[var(--text-secondary)]">
                The multimodal analysis endpoint is unreachable or experienced a server error. No fabricated scores will be displayed.
              </p>
            )}
          </div>
        )}

        {/* BEGIN GENERATED */}
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
                    <span className="text-xl font-bold font-mono text-[var(--text-primary)]">
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
                  <div className="text-xs text-[var(--text-secondary)] mt-1.5 font-mono">
                    Model: <span className="text-cyan-400">{result.model_used}</span> · CLIP Engine:{" "}
                    <span className={result.clip_scored && isLive ? "text-emerald-400 font-bold" : "text-amber-400 font-bold"}>
                      {result.clip_scored && isLive ? "ViT-B/32 Active" : "TF-IDF / Heuristic Fallback"}
                    </span>
                  </div>
                </div>

                <div className="text-left sm:text-right">
                  <div className="text-[10px] text-[var(--text-secondary)] font-mono uppercase tracking-wider">
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
                  <div className="text-[10px] text-[var(--text-secondary)] font-mono mt-0.5">
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
            <div className="p-5 rounded-xl border border-[var(--border)] bg-[var(--surface)] space-y-4">
              <div className="flex items-center justify-between">
                <div className="text-xs font-bold uppercase tracking-wider text-[var(--text-primary)] flex items-center gap-2">
                  <Layers className="w-4 h-4 text-cyan-400" />
                  <span>Visual Similarity &amp; Catalog Evidence</span>
                </div>
                <div className="text-[11px] font-mono text-[var(--text-secondary)]">
                  512-Dimensional CLIP Embedding Projection
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {[
                  {
                    header: "SUBMITTED SELLER UPLOAD",
                    statusText: result.fake_listing_probability > 0.5 ? "Flagged Asset" : "Clean Asset",
                    statusColor: result.fake_listing_probability > 0.5 ? "text-rose-400" : "text-emerald-400",
                    img: activePreset.sellerImage,
                    tag: "Seller Upload",
                    tagColor: "text-cyan-300 border-cyan-500/30",
                    captionLabel: "Caption:",
                    captionVal: activePreset.sellerImageTitle,
                    captionColor: "text-[var(--text-primary)]",
                  },
                  {
                    header: "CATALOG REFERENCE MATCH",
                    statusText: `${(result.multimodal_similarity_score * 100).toFixed(1)}% Visual Match`,
                    statusColor: "text-cyan-400 font-semibold",
                    img: activePreset.catalogImage,
                    tag: "Official Catalog",
                    tagColor: "text-emerald-300 border-emerald-500/30",
                    captionLabel: "Original Merchant:",
                    captionVal: activePreset.originalMerchant,
                    captionColor: "text-blue-400",
                  },
                ].map((card, i) => (
                  <div key={i} className="p-4 rounded-lg bg-[var(--surface-elevated)] border border-[var(--border)] space-y-3">
                    <div className="flex items-center justify-between text-xs font-mono">
                      <span className="text-[var(--text-secondary)]">{card.header}</span>
                      <span className={card.statusColor}>{card.statusText}</span>
                    </div>
                    <div className="relative h-48 rounded-lg overflow-hidden border border-[var(--border)] bg-black flex items-center justify-center">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={card.img} alt={card.header} className="w-full h-full object-contain" />
                      <div className={`absolute top-2 right-2 px-2 py-0.5 rounded bg-black/80 backdrop-blur text-[10px] font-mono border ${card.tagColor}`}>
                        {card.tag}
                      </div>
                    </div>
                    <div className="text-[11px] font-mono text-[var(--text-secondary)]">
                      {card.captionLabel} <span className={card.captionColor}>{card.captionVal}</span>
                    </div>
                  </div>
                ))}
              </div>

              {/* Dynamic Visual Alignment Gauge */}
              <div className="p-3.5 rounded-lg bg-[var(--surface-elevated)] border border-[var(--border)] space-y-2">
                <div className="flex items-center justify-between text-xs font-mono">
                  <span className="text-[var(--text-secondary)]">Visual Hash Alignment Spectrum</span>
                  <span className="text-[var(--text-primary)] font-semibold">
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
                   <span>{RISK_THRESHOLDS.LISTING_QUARANTINE.toFixed(2)} (Threshold)</span>
                  <span>1.0 (Exact Image Duplicate)</span>
                </div>
              </div>

              {/* Dynamic Forensic Narrative & Key Evidence */}
              <div className="p-4 rounded-lg bg-[var(--surface-elevated)] border border-[var(--border)] space-y-3">
                <div className="flex items-center justify-between">
                  <div className="font-semibold font-mono text-xs flex items-center gap-2 text-[var(--text-primary)]">
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
                  <ul className="space-y-1.5 text-xs text-[var(--text-secondary)] pl-2">
                    {result.investigator_narrative.key_evidence.map((ev: unknown, i: number) => (
                      <li key={i} className="flex items-start gap-2">
                        <span className="text-cyan-400 mt-1">•</span>
                        <span>{String(ev)}</span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-xs text-[var(--text-secondary)] leading-relaxed">
                    {typeof result.investigator_narrative?.summary === "string"
                      ? result.investigator_narrative.summary
                      : result.fake_listing_probability > 0.5
                      ? `The listing imagery submitted by this ${formData.seller_age_days_at_listing}-day-old merchant matches an existing official catalog image with ${(result.multimodal_similarity_score * 100).toFixed(1)}% cosine similarity. Self-match exclusion was asserted: seller is not authorized for catalog asset reuse.`
                      : `Product image aligns with legitimate merchant inventory baseline (${(result.multimodal_similarity_score * 100).toFixed(1)}% similarity). Seller historical tenure (${formData.seller_age_days_at_listing} days) confirms authentic distributor relationship.`}
                  </p>
                )}

                {/* Recommended action */}
                <div className="pt-2 border-t border-[var(--border)] flex items-center justify-between text-xs">
                  <span className="text-[var(--text-secondary)] font-mono">Recommended Policy Action:</span>
                  <span className="text-[var(--text-primary)] font-semibold">
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
                      onClick={() => void handleQuarantineListing()}
                      disabled={listingActionLoading}
                      className="flex-1 py-2 px-3 rounded-lg bg-rose-600 hover:bg-rose-500 text-xs font-semibold text-white transition-all flex items-center justify-center gap-2 cursor-pointer shadow-lg shadow-rose-600/20 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      <XCircle className={`w-4 h-4 ${listingActionLoading ? "opacity-60" : ""}`} />
                      <span>Quarantine Listing &amp; Hold Payout</span>
                    </button>
                    <button
                      onClick={() =>
                        setListingAction({
                          kind: "demo",
                          message: "Demo-only control: inventory proof requests are not connected to a live action route.",
                        })
                      }
                      disabled={!isDemo || listingActionLoading}
                      title={isDemo ? "Demo-only control" : "No live inventory-proof action route is configured"}
                      className="py-2 px-3 rounded-lg bg-[#151D27] hover:bg-[#1C2633] border border-[var(--border)] text-xs font-semibold text-[var(--text-primary)] transition-all flex items-center justify-center gap-2 cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      <AlertTriangle className="w-4 h-4 text-amber-400" />
                      <span>Request Inventory Proof</span>
                    </button>
                  </>
                ) : (
                  <div className="flex-1 py-2 px-3 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-xs text-emerald-700 dark:text-emerald-400 font-mono flex items-center justify-center gap-2">
                    <CheckCircle2 className="w-4 h-4" />
                    <span>Listing is below quarantine threshold; no operator action required.</span>
                  </div>
                )}
              </div>

              <p className="text-[10px] italic text-[var(--text-secondary)] mt-1">
                Actions are accepted by the backend outbox first; provider execution is asynchronous.
              </p>

              {listingAction?.kind === "demo" && (
                <div className="p-3 rounded-lg bg-amber-500/10 border border-amber-500/30 text-xs text-amber-700 dark:text-amber-400 font-mono flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 shrink-0" />
                  <span>{listingAction.message}</span>
                </div>
              )}

              {listingAction?.kind === "error" && (
                <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/30 text-xs text-rose-700 dark:text-rose-400 font-mono flex items-start gap-2">
                  <XCircle className="w-4 h-4 shrink-0 mt-0.5" />
                  <span>{listingAction.message}</span>
                </div>
              )}

              {listingAction?.kind === "receipt" && (
                <div
                  className={`p-3 rounded-lg border text-xs font-mono space-y-2 ${
                    listingAction.receipt.status === "failed"
                      ? "bg-rose-500/10 border-rose-500/30 text-rose-700 dark:text-rose-400"
                      : listingAction.receipt.status === "succeeded"
                      ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-700 dark:text-emerald-400"
                      : "bg-blue-500/10 border-blue-500/30 text-blue-600 dark:text-blue-400"
                  }`}
                >
                  <div className="flex items-center gap-2">
                    {listingAction.receipt.status === "failed" ? (
                      <XCircle className="w-4 h-4 shrink-0" />
                    ) : (
                      <Check className="w-4 h-4 shrink-0" />
                    )}
                    <span>
                      {listingAction.receipt.status === "pending"
                        ? "Action accepted — pending provider execution."
                        : listingAction.receipt.status === "succeeded"
                        ? "Action completed with provider receipt."
                        : `Action failed: ${listingAction.receipt.error || "provider execution was not completed."}`}
                    </span>
                  </div>
                  <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] text-[var(--text-secondary)]">
                    <span>Action ID: {listingAction.receipt.command_id}</span>
                    <button
                      type="button"
                      onClick={() => void refreshListingAction()}
                      disabled={listingActionLoading}
                      className="inline-flex items-center gap-1 text-blue-600 dark:text-blue-400 hover:underline disabled:opacity-50"
                    >
                      <span>{listingActionLoading ? "Refreshing…" : "Refresh status"}</span>
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
        {/* END GENERATED */}
      </div>
    </div>
  );
}
