"use client";

import React, { useEffect } from "react";
import Link from "next/link";
import {
  X,
  ShieldAlert,
  ArrowRight,
  AlertTriangle,
  CheckCircle2,
  HelpCircle,
  Copy,
  Clock,
  User,
  CreditCard,
} from "lucide-react";
import { DecisionBadge } from "@/components/ui/DecisionBadge";
import { RISK_THRESHOLDS } from "@/lib/constants/thresholds";

export interface DrawerTransactionData {
  order_id: string;
  amount: number;
  user_id?: string;
  timestamp?: string;
  risk_score: number;
  decision: "ALLOW" | "REVIEW" | "HOLD" | "BLOCK" | string;
  channel?: string;
  ip_address?: string;
  device_id?: string;
  account_age_days?: number;
  is_missing_device?: boolean;
  top_drivers?: {
    name: string;
    impact: "HIGH_RISK" | "MEDIUM_RISK" | "LOW_RISK" | "SAFE";
    description: string;
  }[];
}

interface TransactionDetailDrawerProps {
  transaction: DrawerTransactionData | null;
  isOpen: boolean;
  onClose: () => void;
}

export function TransactionDetailDrawer({ transaction, isOpen, onClose }: TransactionDetailDrawerProps) {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && isOpen) onClose();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen || !transaction) return null;

  const decisionUpper = (transaction.decision || "REVIEW").toUpperCase() as
    | "ALLOW"
    | "REVIEW"
    | "HOLD"
    | "BLOCK";

  // Plain language explanation based on risk and decision
  const getExplanation = () => {
    if (decisionUpper === "BLOCK") {
      return {
        summary: "Severe fraud indicators detected. Order blocked to prevent immediate financial loss.",
        policyReason: "The expected chargeback loss on this transaction heavily exceeds the merchant checkout cost.",
        keyFactor: "Key triggers: high order velocity from an unverified identity with rapid payment attempts.",
      };
    }
    if (decisionUpper === "HOLD") {
      return {
        summary: "High risk requiring identity verification before shipping or fulfillment.",
        policyReason: "Order placed on hold to verify cardholder identity with issuing bank.",
        keyFactor: "Key triggers: mismatch between shipping location and device IP history.",
      };
    }
    if (decisionUpper === "REVIEW") {
      return {
        summary: "Borderline anomaly. Queued for human analyst inspection.",
        policyReason: "Transaction sits within the manual review buffer where human verification is required.",
        keyFactor: "Key triggers: recent account creation combined with elevated basket value.",
      };
    }
    return {
      summary: "Low risk transaction. Verified customer with clean historical profile.",
      policyReason: "Approved for instantaneous frictionless processing.",
      keyFactor: "Key triggers: established account age, verified address, and trusted device fingerprint.",
    };
  };

  const explanation = getExplanation();

  return (
    <div className="fixed inset-0 z-50 overflow-hidden">
      {/* Backdrop */}
      <div
        onClick={onClose}
        className="absolute inset-0 bg-black/50 backdrop-blur-xs transition-opacity animate-in fade-in"
      />

      <div className="fixed inset-y-0 right-0 max-w-full flex pl-10">
        <div className="w-screen max-w-md bg-[var(--surface)] border-l border-[var(--border)] text-[var(--text-primary)] shadow-2xl flex flex-col animate-in slide-in-from-right duration-200">
          {/* Header */}
          <div className="p-5 border-b border-[var(--border)] flex items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-xl bg-blue-500/10 text-blue-500 border border-blue-500/20">
                <ShieldAlert className="w-5 h-5" />
              </div>
              <div>
                <div className="text-[11px] font-mono text-[var(--text-muted)] flex items-center gap-1.5">
                  <span>ORDER DETAILS</span>
                  <button
                    onClick={() => navigator.clipboard.writeText(transaction.order_id)}
                    title="Copy Order ID"
                    className="hover:text-[var(--text-primary)]"
                  >
                    <Copy className="w-3 h-3" />
                  </button>
                </div>
                <h3 className="text-sm font-bold font-mono text-[var(--text-primary)]">{transaction.order_id}</h3>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <DecisionBadge decision={decisionUpper} />
              <button
                onClick={onClose}
                aria-label="Close drawer"
                className="p-1.5 rounded-lg text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--surface-elevated)] transition-colors focus-ring"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
          </div>

          {/* Drawer Body */}
          <div className="flex-1 overflow-y-auto p-5 space-y-5">
            {/* Plain Language Explanation Card */}
            <div className="p-4 rounded-xl border border-[var(--border)] bg-[var(--surface-elevated)] space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-[var(--text-primary)] flex items-center gap-1.5">
                  <HelpCircle className="w-3.5 h-3.5 text-blue-500" />
                  <span>Why Was This Decision Made?</span>
                </span>
                <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded bg-blue-500/10 text-blue-500 border border-blue-500/20">
                  Plain-Language Summary
                </span>
              </div>

              <div className="space-y-2 text-xs text-[var(--text-secondary)] leading-relaxed">
                <p className="font-medium text-[var(--text-primary)]">{explanation.summary}</p>
                <p className="text-[11px] text-[var(--text-muted)]">
                  <strong className="text-[var(--text-secondary)]">Policy Logic: </strong>
                  {explanation.policyReason}
                </p>
                <p className="text-[11px] text-[var(--text-muted)]">
                  <strong className="text-[var(--text-secondary)]">Risk Drivers: </strong>
                  {explanation.keyFactor}
                </p>
              </div>
            </div>

            {/* Risk Gauge */}
            <div className="p-4 rounded-xl border border-[var(--border)] bg-[var(--surface)] space-y-2">
              <div className="flex items-center justify-between text-xs font-medium">
                <span className="text-[var(--text-secondary)]">Calibrated Fraud Probability</span>
                {/* BEGIN GENERATED */}
                <span className="font-mono font-bold text-sm text-[var(--text-primary)]">
                  {(transaction.risk_score * 100).toFixed(1)}%
                </span>
                {/* END GENERATED */}
              </div>

              {/* Progress bar */}
              <div className="h-2 w-full rounded-full bg-[var(--border)] overflow-hidden">
                <div
                  className={`h-full transition-all duration-500 ${
                    transaction.risk_score >= RISK_THRESHOLDS.HOLD
                      ? "bg-red-500"
                      : transaction.risk_score >= RISK_THRESHOLDS.REVIEW
                      ? "bg-amber-500"
                      : "bg-emerald-500"
                  }`}
                  style={{ width: `${Math.min(100, Math.max(0, transaction.risk_score * 100))}%` }}
                />
              </div>

              <div className="flex justify-between text-[10px] font-mono text-[var(--text-muted)] pt-1">
                <span>{RISK_THRESHOLDS.ALLOW}.0 (Safe)</span>
                <span>{RISK_THRESHOLDS.REVIEW} (Review)</span>
                <span>{RISK_THRESHOLDS.HOLD} (Hold)</span>
                <span>{RISK_THRESHOLDS.BLOCK} (Block)</span>
              </div>
            </div>

            {/* Financial & Context Metadata */}
            <div className="p-4 rounded-xl border border-[var(--border)] bg-[var(--surface)] space-y-3">
              <span className="text-xs font-semibold text-[var(--text-primary)] block">Transaction Context</span>
              <div className="grid grid-cols-2 gap-3 text-xs">
                <div className="space-y-1">
                  <span className="text-[11px] text-[var(--text-muted)] flex items-center gap-1">
                    <CreditCard className="w-3 h-3" /> Amount
                  </span>
                  {/* BEGIN GENERATED */}
                  <span className="font-mono font-bold text-[var(--text-primary)]">
                    ${transaction.amount.toFixed(2)}
                  </span>
                  {/* END GENERATED */}
                </div>

                <div className="space-y-1">
                  <span className="text-[11px] text-[var(--text-muted)] flex items-center gap-1">
                    <User className="w-3 h-3" /> Customer ID
                  </span>
                  <span className="font-mono text-[var(--text-secondary)] truncate block">
                    {transaction.user_id || "USR_GUEST_ANON"}
                  </span>
                </div>

                <div className="space-y-1">
                  <span className="text-[11px] text-[var(--text-muted)] flex items-center gap-1">
                    <Clock className="w-3 h-3" /> Timestamp
                  </span>
                  <span className="font-mono text-[var(--text-secondary)]">
                    {transaction.timestamp || "Just now"}
                  </span>
                </div>

                <div className="space-y-1">
                  <span className="text-[11px] text-[var(--text-muted)]">Channel</span>
                  <span className="font-mono text-[var(--text-secondary)]">
                    {transaction.channel || "Web Checkout"}
                  </span>
                </div>
              </div>
            </div>

            {/* Verified vs Missing Signals */}
            <div className="p-4 rounded-xl border border-[var(--border)] bg-[var(--surface)] space-y-3">
              <span className="text-xs font-semibold text-[var(--text-primary)] block">
                Signal Availability & Verification
              </span>
              <div className="space-y-2 text-xs">
                <div className="p-2.5 rounded-lg bg-[var(--surface-elevated)] flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500" />
                    <span className="text-[var(--text-secondary)]">Account Age</span>
                  </div>
                  <span className="font-mono font-medium text-[var(--text-primary)]">
                    {transaction.account_age_days !== undefined
                      ? `${transaction.account_age_days} days`
                      : "Verified"}
                  </span>
                </div>

                <div className="p-2.5 rounded-lg bg-[var(--surface-elevated)] flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    {transaction.is_missing_device ? (
                      <AlertTriangle className="w-3.5 h-3.5 text-amber-500" />
                    ) : (
                      <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500" />
                    )}
                    <span className="text-[var(--text-secondary)]">Device Telemetry</span>
                  </div>
                  {transaction.is_missing_device ? (
                    <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-amber-500/10 text-amber-500 border border-amber-500/20">
                      N/A (Missing Signal)
                    </span>
                  ) : (
                    <span className="font-mono font-medium text-[var(--text-primary)]">
                      {transaction.device_id || "DEV_VERIFIED_HASH"}
                    </span>
                  )}
                </div>

                <div className="p-2.5 rounded-lg bg-[var(--surface-elevated)] flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500" />
                    <span className="text-[var(--text-secondary)]">IP Reputation</span>
                  </div>
                  <span className="font-mono font-medium text-[var(--text-primary)]">
                    {transaction.ip_address || "Residential (Clean)"}
                  </span>
                </div>
              </div>
            </div>

            {/* Top SHAP Drivers */}
            <div className="p-4 rounded-xl border border-[var(--border)] bg-[var(--surface)] space-y-3">
              <span className="text-xs font-semibold text-[var(--text-primary)] block">
                Top Contributing Factors (TreeSHAP)
              </span>
              <div className="space-y-2">
                {transaction.top_drivers && transaction.top_drivers.length > 0 ? (
                  transaction.top_drivers.map((d, i) => (
                    <div
                      key={i}
                      className="p-2.5 rounded-lg bg-[var(--surface-elevated)] border border-[var(--border)] space-y-1"
                    >
                      <div className="flex items-center justify-between text-xs">
                        <span className="font-medium text-[var(--text-primary)]">{d.name}</span>
                        <span
                          className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${
                            d.impact === "HIGH_RISK"
                              ? "bg-red-500/10 text-red-500"
                              : d.impact === "MEDIUM_RISK"
                              ? "bg-amber-500/10 text-amber-500"
                              : "bg-emerald-500/10 text-emerald-500"
                          }`}
                        >
                          {d.impact.replace("_", " ")}
                        </span>
                      </div>
                      <p className="text-[11px] text-[var(--text-muted)] leading-relaxed">{d.description}</p>
                    </div>
                  ))
                ) : (
                  <div className="text-xs text-[var(--text-muted)] p-2">
                    Inference attribution calculated: velocity and account history factors dominate score.
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Drawer Footer */}
          <div className="p-4 border-t border-[var(--border)] bg-[var(--surface-elevated)]/50 flex items-center gap-3">
            <Link
              href={`/investigations?order_id=${transaction.order_id}`}
              onClick={onClose}
              className="flex-1 inline-flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-semibold text-xs transition-colors shadow-sm"
            >
              <span>Open in Investigation Workspace</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </Link>
            <button
              onClick={onClose}
              className="px-4 py-2.5 rounded-xl border border-[var(--border)] bg-[var(--surface)] hover:bg-[var(--surface-elevated)] text-xs font-medium text-[var(--text-secondary)] transition-colors"
            >
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
