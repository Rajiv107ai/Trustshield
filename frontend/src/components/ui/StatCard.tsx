"use client";

import React from "react";
import { LucideIcon, Info, TrendingUp, TrendingDown } from "lucide-react";
import { PlainTooltip } from "@/components/common/PlainTooltip";

interface StatCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon?: LucideIcon;
  badge?: string;
  badgeColor?: "blue" | "emerald" | "amber" | "purple" | "cyan" | "rose";
  trend?: { value: string; positive?: boolean };
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
  trend,
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

  const iconGlowStyles = {
    blue: "text-blue-500 bg-blue-500/10 border-blue-500/20",
    emerald: "text-emerald-500 bg-emerald-500/10 border-emerald-500/20",
    amber: "text-amber-500 bg-amber-500/10 border-amber-500/20",
    purple: "text-purple-500 bg-purple-500/10 border-purple-500/20",
    cyan: "text-cyan-500 bg-cyan-500/10 border-cyan-500/20",
    rose: "text-rose-500 bg-rose-500/10 border-rose-500/20",
  }[badgeColor];

  return (
    <div
      className={`group relative overflow-hidden p-5 rounded-2xl border border-[var(--border)] bg-gradient-to-b from-[var(--surface-elevated)]/50 via-[var(--surface)] to-[var(--surface)] text-[var(--text-primary)] shadow-sm hover:border-[var(--intelligence-blue)]/40 hover:shadow-[0_12px_32px_-8px_rgba(0,0,0,0.3)] transition-all duration-300 ${className}`}
    >
      {/* Subtle top ambient sheen */}
      <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-white/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none" />

      <div className="flex items-center justify-between gap-3 mb-3">
        <span className="text-xs font-semibold tracking-wide text-[var(--text-secondary)] flex items-center gap-1.5">
          <span>{title}</span>
          {tooltip && (
            <PlainTooltip term={title} explanation={tooltip} position="bottom">
              <span className="cursor-help text-[var(--text-muted)] hover:text-[var(--text-secondary)] transition-colors inline-flex items-center">
                <Info className="w-3.5 h-3.5" />
              </span>
            </PlainTooltip>
          )}
        </span>
        {Icon && (
          <div
            className={`w-8 h-8 rounded-xl border flex items-center justify-center transition-transform duration-300 group-hover:scale-105 shadow-xs ${iconGlowStyles}`}
          >
            <Icon className="w-4 h-4" />
          </div>
        )}
      </div>

      <div className="flex items-baseline justify-between gap-2">
        <div className="flex items-baseline gap-2">
          <span className="text-2xl lg:text-3xl font-bold tracking-tight text-[var(--text-primary)] font-mono tabular-nums">
            {value}
          </span>
          {badge && (
            <span
              className={`text-[10px] font-mono uppercase px-2 py-0.5 rounded-full border font-semibold tracking-wide ${badgeStyles}`}
            >
              {badge}
            </span>
          )}
        </div>

        {trend && (
          <span
            className={`inline-flex items-center gap-1 text-[11px] font-medium font-mono ${
              trend.positive !== false
                ? "text-emerald-500 dark:text-emerald-400"
                : "text-rose-500 dark:text-rose-400"
            }`}
          >
            {trend.positive !== false ? (
              <TrendingUp className="w-3 h-3" />
            ) : (
              <TrendingDown className="w-3 h-3" />
            )}
            {trend.value}
          </span>
        )}
      </div>

      {subtitle && (
        <p className="text-[11px] text-[var(--text-muted)] mt-2 leading-relaxed truncate">
          {subtitle}
        </p>
      )}
    </div>
  );
}
