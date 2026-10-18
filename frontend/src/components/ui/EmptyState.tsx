"use client";

import React from "react";
import { LucideIcon, SearchX } from "lucide-react";

interface EmptyStateProps {
  title: string;
  description: string;
  icon?: LucideIcon;
  actionLabel?: string;
  onAction?: () => void;
  className?: string;
}

export function EmptyState({
  title,
  description,
  icon: Icon = SearchX,
  actionLabel,
  onAction,
  className = "",
}: EmptyStateProps) {
  return (
    <div
      className={`flex flex-col items-center justify-center p-8 text-center rounded-xl border border-dashed border-[var(--border)] bg-[var(--surface)] text-[var(--text-primary)] my-4 ${className}`}
    >
      <div className="w-12 h-12 rounded-xl bg-[var(--surface-elevated)] border border-[var(--border)] flex items-center justify-center text-[var(--text-muted)] mb-3">
        <Icon className="w-6 h-6" />
      </div>
      <h3 className="text-sm font-semibold text-[var(--text-primary)] mb-1">{title}</h3>
      <p className="text-xs text-[var(--text-secondary)] max-w-sm mb-4 leading-relaxed">
        {description}
      </p>
      {actionLabel && onAction && (
        <button
          onClick={onAction}
          className="px-3.5 py-1.5 rounded-lg bg-[var(--surface-elevated)] hover:bg-[var(--border)] border border-[var(--border)] text-xs font-semibold text-[var(--text-primary)] transition-all cursor-pointer focus-ring"
        >
          {actionLabel}
        </button>
      )}
    </div>
  );
}
