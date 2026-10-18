"use client";

import React, { useState } from "react";
import { HelpCircle } from "lucide-react";

interface PlainTooltipProps {
  term?: string;
  explanation: string;
  source?: string;
  position?: "top" | "bottom";
  children?: React.ReactNode;
}

export function PlainTooltip({ term, explanation, source, position = "top", children }: PlainTooltipProps) {
  const [isVisible, setIsVisible] = useState(false);

  return (
    <span className="relative inline-flex items-center">
      {children ? (
        <span
          onMouseEnter={() => setIsVisible(true)}
          onMouseLeave={() => setIsVisible(false)}
          onFocus={() => setIsVisible(true)}
          onBlur={() => setIsVisible(false)}
          tabIndex={0}
          className="cursor-help underline decoration-dotted decoration-[var(--text-muted)] underline-offset-4 focus:outline-none"
        >
          {children}
        </span>
      ) : (
        <button
          type="button"
          aria-label={term ? `Learn more about ${term}` : "Learn more"}
          onMouseEnter={() => setIsVisible(true)}
          onMouseLeave={() => setIsVisible(false)}
          onFocus={() => setIsVisible(true)}
          onBlur={() => setIsVisible(false)}
          className="inline-flex p-0.5 text-[var(--text-muted)] hover:text-[var(--intelligence-blue)] focus:outline-none transition-colors"
        >
          <HelpCircle className="w-3.5 h-3.5" />
        </button>
      )}

      {isVisible && (
        <span
          role="tooltip"
          className={`absolute z-50 ${position === "bottom" ? "top-full mt-2" : "bottom-full mb-2"} left-1/2 -translate-x-1/2 w-64 p-3 rounded-lg border border-[var(--border)] bg-[var(--surface-elevated)] text-[var(--text-primary)] shadow-xl text-xs font-normal leading-relaxed pointer-events-none`}
        >
          {term && <span className="block font-semibold text-[var(--text-primary)] mb-1">{term}</span>}
          <span className="block text-[var(--text-secondary)]">{explanation}</span>
          {source && (
            <span className="block mt-2 pt-1.5 border-t border-[var(--border)] text-[10px] text-[var(--text-muted)] font-mono">
              Source: {source}
            </span>
          )}
        </span>
      )}
    </span>
  );
}
