"use client";

import React from "react";
import { CheckCircle2, Activity, AlertTriangle, XCircle } from "lucide-react";

export type DecisionType = "ALLOW" | "REVIEW" | "HOLD" | "BLOCK" | string;

interface DecisionBadgeProps {
  decision: DecisionType;
  size?: "sm" | "md" | "lg";
  showIcon?: boolean;
  className?: string;
}

export function DecisionBadge({
  decision,
  size = "md",
  showIcon = true,
  className = "",
}: DecisionBadgeProps) {
  const norm = (decision || "").toUpperCase();

  const sizeClasses = {
    sm: "px-2 py-0.5 text-[10px] gap-1",
    md: "px-2.5 py-1 text-xs gap-1.5",
    lg: "px-3.5 py-1.5 text-sm gap-2",
  }[size];

  const iconSizes = {
    sm: "w-3 h-3",
    md: "w-3.5 h-3.5",
    lg: "w-4 h-4",
  }[size];

  let config = {
    label: norm || "UNKNOWN",
    classes: "bg-slate-500/10 border-slate-500/30 text-slate-400 dark:text-slate-300",
    icon: Activity,
    title: "System Evaluation",
  };

  if (norm === "ALLOW") {
    config = {
      label: "ALLOW",
      classes: "bg-emerald-500/10 border-emerald-500/30 text-emerald-700 dark:text-emerald-400 font-semibold",
      icon: CheckCircle2,
      title: "Automated Checkout Allowed (Low Fraud Risk)",
    };
  } else if (norm === "REVIEW") {
    config = {
      label: "REVIEW",
      classes: "bg-amber-500/10 border-amber-500/30 text-amber-700 dark:text-amber-400 font-semibold",
      icon: Activity,
      title: "Cost-Loss Review Queued (Human Forensic Verification Required)",
    };
  } else if (norm === "HOLD") {
    config = {
      label: "HOLD",
      classes: "bg-orange-500/10 border-orange-500/30 text-orange-700 dark:text-orange-400 font-semibold",
      icon: AlertTriangle,
      title: "Payment Authorization Hold (Step-Up Authentication Triggered)",
    };
  } else if (norm === "BLOCK") {
    config = {
      label: "BLOCK",
      classes: "bg-rose-500/10 border-rose-500/30 text-rose-700 dark:text-rose-400 font-semibold",
      icon: XCircle,
      title: "Transaction Blocked (Critical Fraud Likelihood)",
    };
  }

  const Icon = config.icon;

  return (
    <span
      title={config.title}
      aria-label={`Risk Decision: ${config.label}. ${config.title}`}
      className={`inline-flex items-center rounded-md border font-mono transition-colors ${sizeClasses} ${config.classes} ${className}`}
    >
      {showIcon && <Icon className={`${iconSizes} shrink-0`} />}
      <span>{config.label}</span>
    </span>
  );
}
