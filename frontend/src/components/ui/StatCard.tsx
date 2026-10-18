"use client";

import React from "react";
import { LucideIcon, Info } from "lucide-react";
import { PlainTooltip } from "@/components/common/PlainTooltip";

interface StatCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon?: LucideIcon;
  badge?: string;
  badgeColor?: "blue" | "emerald" | "amber" | "purple" | "cyan" | "rose";
  tooltip?: string;
  className?: string;
}

export function StatCard({
  title,
  value,
  subtitle,
  icon: Icon,
  badge,
  badgeColor = "blue",
  tooltip,
  className = "",
}: StatCardProps) {
  const badgeStyles = {
    blue: "bg-blue-500/10 text-blue-600 dark:text-blue-400 border-blue-500/20",
    emerald: "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20",
    amber: "bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/20",
    purple: "bg-purple-500/10 text-purple-600 dark:text-purple-400 border-purple-500/20",
    cyan: "bg-cyan-500/10 text-cyan-600 dark:text-cyan-400 border-cyan-500/20",
    rose: "bg-rose-500/10 text-rose-600 dark:text-rose-400 border-rose-500/20",
  }[badgeColor];

  return (
    <div
      className={`card-interactive p-4 rounded-xl border border-[var(--border)] bg-[var(--surface)] text-[var(--text-primary)] shadow-sm relative ${className}`}
    >
      <div className="flex items-center justify-between gap-2 mb-2">
        <span className="text-xs font-medium text-[var(--text-secondary)] flex items-center gap-1.5">
          <span>{title}</span>
          {tooltip && (
            <PlainTooltip term={title} explanation={tooltip} position="bottom">
              <span className="cursor-help text-[var(--text-muted)] hover:text-[var(--text-secondary)] inline-flex items-center">
                <Info className="w-3.5 h-3.5" />
              </span>
            </PlainTooltip>
          )}
        </span>
        {Icon && (
          <div className="w-7 h-7 rounded-lg bg-[var(--surface-elevated)] border border-[var(--border)] flex items-center justify-center text-[var(--intelligence-blue)]">
            <Icon className="w-4 h-4" />
          </div>
        )}
      </div>

      <div className="flex items-baseline gap-2">
        <span className="text-2xl font-bold tracking-tight text-[var(--text-primary)] font-mono tabular-nums">
          {value}
        </span>
        {badge && (
          <span className={`text-[10px] font-mono uppercase px-1.5 py-0.5 rounded border font-semibold ${badgeStyles}`}>
            {badge}
          </span>
        )}
      </div>

      {subtitle && (
        <p className="text-[11px] text-[var(--text-muted)] mt-1.5 leading-relaxed truncate">
          {subtitle}
        </p>
      )}
    </div>
  );
}
