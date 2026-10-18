"use client";

import React, { useState, useEffect } from "react";
import {
  ShieldAlert,
  X,
  ChevronRight,
  ChevronLeft,
  CheckCircle2,
  Sparkles,
  Scale,
  Network,
  HelpCircle,
  ArrowRight,
} from "lucide-react";
import Link from "next/link";

interface OnboardingGuideModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export function OnboardingGuideModal({ isOpen, onClose }: OnboardingGuideModalProps) {
  const [currentStep, setCurrentStep] = useState(0);

  const steps = [
    {
      badge: "Step 1 of 5 · Introduction",
      title: "Welcome to TrustShield AI",
      subtitle: "Enterprise fraud prevention designed for everyday operations.",
      icon: <ShieldAlert className="w-6 h-6 text-blue-500" />,
      content: (
        <div className="space-y-4 text-xs text-[var(--text-secondary)] leading-relaxed">
          <p>
            TrustShield monitors marketplace transactions in real time to prevent financial losses like stolen credit
            cards, account takeovers, and fraudulent merchant rings.
          </p>
          <div className="p-3.5 rounded-xl border border-[var(--border)] bg-[var(--surface-elevated)] space-y-2">
            <div className="font-semibold text-[var(--text-primary)] text-xs flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-blue-500" />
              <span>How it works in 3 simple steps:</span>
            </div>
            <ul className="list-disc pl-4 space-y-1 text-[var(--text-muted)]">
              <li>An order is submitted through your marketplace checkout.</li>
              <li>The engine evaluates customer velocity, device trust, and network links in under 15ms.</li>
              <li>You receive a clear operational recommendation: approve, review, or block.</li>
            </ul>
          </div>
        </div>
      ),
    },
    {
      badge: "Step 2 of 5 · Decision Logic",
      title: "Risk Score vs. Policy Decision",
      subtitle: "Why an estimated probability is different from an operational action.",
      icon: <Scale className="w-6 h-6 text-amber-500" />,
      content: (
        <div className="space-y-4 text-xs text-[var(--text-secondary)] leading-relaxed">
          <p>
            Many fraud tools show a raw probability number without explaining what to do. TrustShield translates
            mathematical risk into 4 clear business actions based on financial cost-loss balance:
          </p>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="p-2.5 rounded-lg border border-emerald-500/20 bg-emerald-500/5">
              <div className="font-bold text-emerald-600 dark:text-emerald-400">ALLOW</div>
              <div className="text-[11px] text-[var(--text-muted)] mt-0.5">
                Low risk. Seamless friction-free checkout for legitimate customers.
              </div>
            </div>
            <div className="p-2.5 rounded-lg border border-amber-500/20 bg-amber-500/5">
              <div className="font-bold text-amber-600 dark:text-amber-400">REVIEW</div>
              <div className="text-[11px] text-[var(--text-muted)] mt-0.5">
                Borderline risk. Sent to human investigator queue or 3DS verification.
              </div>
            </div>
            <div className="p-2.5 rounded-lg border border-orange-500/20 bg-orange-500/5">
              <div className="font-bold text-orange-600 dark:text-orange-400">HOLD</div>
              <div className="text-[11px] text-[var(--text-muted)] mt-0.5">
                Elevated risk. Temporary fulfillment freeze pending identity verification.
              </div>
            </div>
            <div className="p-2.5 rounded-lg border border-red-500/20 bg-red-500/5">
              <div className="font-bold text-red-600 dark:text-red-400">BLOCK</div>
              <div className="text-[11px] text-[var(--text-muted)] mt-0.5">
                Severe risk. Immediate auto-cancellation to protect against chargebacks.
              </div>
            </div>
          </div>
        </div>
      ),
    },
    {
      badge: "Step 3 of 5 · Explainability",
      title: "Understanding the 'Why' (Risk Drivers)",
      subtitle: "No black-box guesses — see the exact signals behind every score.",
      icon: <HelpCircle className="w-6 h-6 text-indigo-500" />,
      content: (
        <div className="space-y-4 text-xs text-[var(--text-secondary)] leading-relaxed">
          <p>
            Whenever a transaction is evaluated, TrustShield breaks down the top factors that influenced the decision
            using TreeSHAP explainability:
          </p>
          <div className="space-y-2">
            <div className="p-3 rounded-lg border border-[var(--border)] bg-[var(--surface-elevated)] flex items-start gap-3">
              <div className="w-2.5 h-2.5 rounded-full bg-red-500 shrink-0 mt-1" />
              <div>
                <span className="font-semibold text-[var(--text-primary)]">Red Factors (Risk Increasers): </span>
                <span className="text-[var(--text-muted)]">
                  Signals that made the order look suspicious (e.g. brand-new account age, rapid order burst, multiple
                  failed payment attempts).
                </span>
              </div>
            </div>
            <div className="p-3 rounded-lg border border-[var(--border)] bg-[var(--surface-elevated)] flex items-start gap-3">
              <div className="w-2.5 h-2.5 rounded-full bg-emerald-500 shrink-0 mt-1" />
              <div>
                <span className="font-semibold text-[var(--text-primary)]">Green Factors (Trust Protectors): </span>
                <span className="text-[var(--text-muted)]">
                  Signals that proved legitimacy (e.g. established account history, verified billing zip code match).
                </span>
              </div>
            </div>
          </div>
        </div>
      ),
    },
    {
      badge: "Step 4 of 5 · Forensic Networks",
      title: "Catching Connected Fraud Rings",
      subtitle: "Detect organized syndicates sharing devices, IPs, or bank accounts.",
      icon: <Network className="w-6 h-6 text-purple-500" />,
      content: (
        <div className="space-y-4 text-xs text-[var(--text-secondary)] leading-relaxed">
          <p>
            Sophisticated fraudsters rarely use a single account. They create dozens of fake buyer and seller accounts
            to coordinate fake reviews or wash stolen funds.
          </p>
          <div className="p-3.5 rounded-xl border border-[var(--border)] bg-[var(--surface-elevated)] space-y-2">
            <div className="font-semibold text-[var(--text-primary)]">How the Network Graph protects you:</div>
            <p className="text-[var(--text-muted)]">
              Our in-memory graph connects entities based on shared fingerprints (device IDs, credit cards, IP subnets).
              If an account links to a known fraud syndicate, its risk score updates instantly.
            </p>
          </div>
        </div>
      ),
    },
    {
      badge: "Step 5 of 5 · Ready to Explore",
      title: "You Are Ready to Investigate!",
      subtitle: "Take your first step into the TrustShield operations console.",
      icon: <CheckCircle2 className="w-6 h-6 text-emerald-500" />,
      content: (
        <div className="space-y-4 text-xs text-[var(--text-secondary)] leading-relaxed">
          <p>
            You can access this tour anytime by clicking the <strong>Guide</strong> icon in the top header or pressing{" "}
            <kbd className="px-1.5 py-0.5 rounded bg-[var(--surface-elevated)] border border-[var(--border)] font-mono text-[10px]">
              Ctrl+K
            </kbd>
            .
          </p>
          <div className="pt-2 flex flex-col sm:flex-row gap-3">
            <Link
              href="/dashboard"
              onClick={onClose}
              className="flex-1 inline-flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-semibold text-xs transition-colors"
            >
              <span>Go to Mission Dashboard</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </Link>
            <Link
              href="/transactions"
              onClick={onClose}
              className="flex-1 inline-flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl border border-[var(--border)] bg-[var(--surface-elevated)] hover:bg-[var(--surface)] text-[var(--text-primary)] font-semibold text-xs transition-colors"
            >
              <span>Try Transaction Analyzer</span>
            </Link>
          </div>
        </div>
      ),
    },
  ];

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!isOpen) return;
      if (e.key === "Escape") onClose();
      if (e.key === "ArrowRight" && currentStep < steps.length - 1) setCurrentStep((prev) => prev + 1);
      if (e.key === "ArrowLeft" && currentStep > 0) setCurrentStep((prev) => prev - 1);
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, currentStep, steps.length, onClose]);

  if (!isOpen) return null;

  const current = steps[currentStep];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-in fade-in duration-200">
      <div
        role="dialog"
        aria-modal="true"
        aria-label="TrustShield Onboarding Guide"
        className="w-full max-w-lg rounded-2xl border border-[var(--border)] bg-[var(--surface)] text-[var(--text-primary)] shadow-2xl overflow-hidden flex flex-col transition-all"
      >
        {/* Modal Header */}
        <div className="p-6 border-b border-[var(--border)] flex items-start justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-[var(--surface-elevated)] border border-[var(--border)]">
              {current.icon}
            </div>
            <div>
              <div className="text-[11px] font-mono font-medium text-blue-600 dark:text-blue-400 uppercase tracking-wider">
                {current.badge}
              </div>
              <h3 className="text-base font-bold text-[var(--text-primary)] mt-0.5">{current.title}</h3>
            </div>
          </div>
          <button
            onClick={onClose}
            aria-label="Close tour"
            className="p-1.5 rounded-lg text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--surface-elevated)] transition-colors focus-ring"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-6 flex-1 space-y-3">{current.content}</div>

        {/* Modal Footer */}
        <div className="p-4 border-t border-[var(--border)] bg-[var(--surface-elevated)]/50 flex items-center justify-between gap-4">
          {/* Progress dots */}
          <div className="flex items-center gap-1.5">
            {steps.map((_, idx) => (
              <button
                key={idx}
                onClick={() => setCurrentStep(idx)}
                aria-label={`Go to step ${idx + 1}`}
                className={`w-2 h-2 rounded-full transition-all ${
                  idx === currentStep
                    ? "w-6 bg-blue-600 dark:bg-blue-400"
                    : "bg-[var(--border)] hover:bg-[var(--text-muted)]"
                }`}
              />
            ))}
          </div>

          <div className="flex items-center gap-2">
            {currentStep > 0 && (
              <button
                onClick={() => setCurrentStep((prev) => prev - 1)}
                className="inline-flex items-center gap-1 px-3 py-1.5 rounded-lg border border-[var(--border)] bg-[var(--surface)] hover:bg-[var(--surface-elevated)] text-xs font-medium text-[var(--text-secondary)] transition-colors"
              >
                <ChevronLeft className="w-3.5 h-3.5" />
                <span>Back</span>
              </button>
            )}

            {currentStep < steps.length - 1 ? (
              <button
                onClick={() => setCurrentStep((prev) => prev + 1)}
                className="inline-flex items-center gap-1 px-4 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold shadow-sm transition-colors"
              >
                <span>Next</span>
                <ChevronRight className="w-3.5 h-3.5" />
              </button>
            ) : (
              <button
                onClick={onClose}
                className="inline-flex items-center gap-1 px-4 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold shadow-sm transition-colors"
              >
                <span>Finish Guide</span>
                <CheckCircle2 className="w-3.5 h-3.5" />
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
