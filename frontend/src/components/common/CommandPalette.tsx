"use client";

import React, { useState, useEffect, useMemo, useRef } from "react";
import { useRouter } from "next/navigation";
import {
  Search,
  LayoutDashboard,
  ShieldAlert,
  Radio,
  Network,
  FolderGit2,
  Image as ImageIcon,
  RotateCcw,
  Cpu,
  BarChart2,
  HelpCircle,
  Sun,
  Moon,
  Printer,
  X,
  Sparkles,
} from "lucide-react";
import { useTheme } from "@/context/ThemeContext";

interface CommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
  onOpenTour?: () => void;
}

interface CommandItem {
  id: string;
  category: "Navigation" | "Actions" | "Scenarios";
  title: string;
  subtitle: string;
  icon: React.ReactNode;
  action: () => void;
}

export function CommandPalette({ isOpen, onClose, onOpenTour }: CommandPaletteProps) {
  const router = useRouter();
  const { theme, toggleTheme } = useTheme();
  const [query, setQuery] = useState("");
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const commands: CommandItem[] = useMemo(
    () => [
      // Navigation
      {
        id: "nav-dashboard",
        category: "Navigation",
        title: "Go to Mission Dashboard",
        subtitle: "Overview of high-risk items, triage queue & system health",
        icon: <LayoutDashboard className="w-4 h-4 text-blue-500" />,
        action: () => {
          router.push("/dashboard");
          onClose();
        },
      },
      {
        id: "nav-transactions",
        category: "Navigation",
        title: "Go to Transaction Risk Analyzer",
        subtitle: "Interactive form, live TreeSHAP waterfall & presets",
        icon: <ShieldAlert className="w-4 h-4 text-amber-500" />,
        action: () => {
          router.push("/transactions");
          onClose();
        },
      },
      {
        id: "nav-feed",
        category: "Navigation",
        title: "Go to Real-Time Audit Feed",
        subtitle: "Live simulated stream of incoming marketplace orders",
        icon: <Radio className="w-4 h-4 text-emerald-500" />,
        action: () => {
          router.push("/transactions/feed");
          onClose();
        },
      },
      {
        id: "nav-rings",
        category: "Navigation",
        title: "Go to Forensic Fraud Rings",
        subtitle: "Graph community clusters and shared fingerprint detection",
        icon: <Network className="w-4 h-4 text-purple-500" />,
        action: () => {
          router.push("/fraud-rings");
          onClose();
        },
      },
      {
        id: "nav-investigations",
        category: "Navigation",
        title: "Go to Investigation Dossiers",
        subtitle: "Unified evidence workspace, notes & printable dossiers",
        icon: <FolderGit2 className="w-4 h-4 text-indigo-500" />,
        action: () => {
          router.push("/investigations");
          onClose();
        },
      },
      {
        id: "nav-listings",
        category: "Navigation",
        title: "Go to Multimodal Listing Analyzer",
        subtitle: "Inspect image vs text catalog discrepancies",
        icon: <ImageIcon className="w-4 h-4 text-cyan-500" />,
        action: () => {
          router.push("/listings");
          onClose();
        },
      },
      {
        id: "nav-returns",
        category: "Navigation",
        title: "Go to Return Abuse Scoring",
        subtitle: "Evaluate buyer return velocity and refund risk",
        icon: <RotateCcw className="w-4 h-4 text-rose-500" />,
        action: () => {
          router.push("/returns");
          onClose();
        },
      },
      {
        id: "nav-models",
        category: "Navigation",
        title: "Go to Model Registry & Governance",
        subtitle: "Inspect active weights, feature schemas & conformal bounds",
        icon: <Cpu className="w-4 h-4 text-blue-500" />,
        action: () => {
          router.push("/models");
          onClose();
        },
      },
      {
        id: "nav-evaluation",
        category: "Navigation",
        title: "Go to Benchmark Evaluation",
        subtitle: "Audited 20-seed statistical benchmarks and ROC-AUC metrics",
        icon: <BarChart2 className="w-4 h-4 text-emerald-500" />,
        action: () => {
          router.push("/evaluation");
          onClose();
        },
      },

      // Actions
      {
        id: "act-tour",
        category: "Actions",
        title: "Launch Interactive Guided Tour",
        subtitle: "Step-by-step walkthrough explaining TrustShield for non-tech users",
        icon: <HelpCircle className="w-4 h-4 text-blue-500" />,
        action: () => {
          onClose();
          onOpenTour?.();
        },
      },
      {
        id: "act-theme",
        category: "Actions",
        title: `Switch to ${theme === "dark" ? "Light" : "Dark"} Mode`,
        subtitle: "Toggle color theme with high-contrast accessibility",
        icon: theme === "dark" ? <Sun className="w-4 h-4 text-amber-400" /> : <Moon className="w-4 h-4 text-blue-600" />,
        action: () => {
          toggleTheme();
          onClose();
        },
      },
      {
        id: "act-print",
        category: "Actions",
        title: "Print or Export Evidence Dossier",
        subtitle: "Generate a court-admissible audit printout of current findings",
        icon: <Printer className="w-4 h-4 text-[var(--text-secondary)]" />,
        action: () => {
          onClose();
          setTimeout(() => window.print(), 200);
        },
      },

      // Scenarios
      {
        id: "scen-carding",
        category: "Scenarios",
        title: "Preset: High-Risk Carding Attack",
        subtitle: "Simulate rapid checkout burst from a brand-new untrusted device",
        icon: <Sparkles className="w-4 h-4 text-red-500" />,
        action: () => {
          router.push("/transactions");
          onClose();
        },
      },
      {
        id: "scen-verified",
        category: "Scenarios",
        title: "Preset: Verified Low-Risk Buyer",
        subtitle: "Simulate a trusted account with established 400-day history",
        icon: <Sparkles className="w-4 h-4 text-emerald-500" />,
        action: () => {
          router.push("/transactions");
          onClose();
        },
      },
    ],
    [router, onClose, onOpenTour, theme, toggleTheme]
  );

  const filtered = useMemo(() => {
    if (!query.trim()) return commands;
    const q = query.toLowerCase();
    return commands.filter(
      (c) =>
        c.title.toLowerCase().includes(q) ||
        c.subtitle.toLowerCase().includes(q) ||
        c.category.toLowerCase().includes(q)
    );
  }, [commands, query]);

  useEffect(() => {
    if (isOpen) {
      queueMicrotask(() => {
        setQuery("");
        setSelectedIndex(0);
      });
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }, [isOpen]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!isOpen) return;

      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
      } else if (e.key === "ArrowDown") {
        e.preventDefault();
        setSelectedIndex((prev) => (prev < filtered.length - 1 ? prev + 1 : 0));
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setSelectedIndex((prev) => (prev > 0 ? prev - 1 : filtered.length - 1));
      } else if (e.key === "Enter" && filtered[selectedIndex]) {
        e.preventDefault();
        filtered[selectedIndex].action();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, filtered, selectedIndex, onClose]);

  if (!isOpen) return null;

  return (
    <div
      onClick={onClose}
      className="fixed inset-0 z-50 flex items-start justify-center pt-20 p-4 bg-black/60 backdrop-blur-sm animate-in fade-in duration-150"
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Command Navigation Palette"
        onClick={(e) => e.stopPropagation()}
        className="w-full max-w-xl rounded-2xl border border-[var(--border)] bg-[var(--surface)] text-[var(--text-primary)] shadow-2xl overflow-hidden flex flex-col"
      >
        {/* Search Input Bar */}
        <div className="p-4 border-b border-[var(--border)] flex items-center gap-3">
          <Search className="w-5 h-5 text-[var(--text-muted)] shrink-0" />
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setSelectedIndex(0);
            }}
            onKeyDown={(e) => {
              if (e.key === "Escape") {
                e.preventDefault();
                e.stopPropagation();
                onClose();
              } else if (e.key === "ArrowDown") {
                e.preventDefault();
                setSelectedIndex((prev) => (prev < filtered.length - 1 ? prev + 1 : 0));
              } else if (e.key === "ArrowUp") {
                e.preventDefault();
                setSelectedIndex((prev) => (prev > 0 ? prev - 1 : filtered.length - 1));
              } else if (e.key === "Enter" && filtered[selectedIndex]) {
                e.preventDefault();
                filtered[selectedIndex].action();
              }
            }}
            placeholder="Type a page, action, or preset... (e.g. 'dossier', 'theme', 'carding')"
            className="flex-1 bg-transparent text-sm text-[var(--text-primary)] placeholder-[var(--text-muted)] focus:outline-none"
          />
          {query && (
            <button
              type="button"
              onClick={() => {
                setQuery("");
                setSelectedIndex(0);
              }}
              title="Clear search query"
              className="p-1 text-[var(--text-muted)] hover:text-[var(--text-primary)] cursor-pointer"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          )}
          <button
            type="button"
            onClick={onClose}
            title="Close search modal (Escape)"
            className="flex items-center gap-1 px-2 py-1 rounded-lg bg-[var(--surface-elevated)] hover:bg-[var(--surface-hover)] border border-[var(--border)] hover:border-blue-500/40 text-[10px] font-mono font-medium text-[var(--text-muted)] hover:text-[var(--text-primary)] cursor-pointer transition-colors"
          >
            <span>ESC</span>
            <X className="w-3 h-3 text-[var(--text-muted)]" />
          </button>
        </div>

        {/* Results List */}
        <div className="max-h-80 overflow-y-auto p-2 divide-y divide-[var(--border)]/30">
          {filtered.length === 0 ? (
            <div className="p-6 text-center text-xs text-[var(--text-muted)]">
              No commands or pages found matching &ldquo;{query}&rdquo;.
            </div>
          ) : (
            filtered.map((item, idx) => {
              const isSelected = idx === selectedIndex;
              return (
                <button
                  key={item.id}
                  onClick={item.action}
                  onMouseEnter={() => setSelectedIndex(idx)}
                  className={`w-full text-left p-3 rounded-xl flex items-center gap-3 transition-colors ${
                    isSelected ? "bg-[var(--surface-elevated)] border border-blue-500/30" : "hover:bg-[var(--surface-elevated)]/50"
                  }`}
                >
                  <div className="p-2 rounded-lg bg-[var(--surface)] border border-[var(--border)] shrink-0">
                    {item.icon}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-semibold text-[var(--text-primary)] truncate">{item.title}</span>
                      <span className="text-[10px] font-mono uppercase px-1.5 py-0.5 rounded bg-[var(--border)]/40 text-[var(--text-muted)]">
                        {item.category}
                      </span>
                    </div>
                    <div className="text-[11px] text-[var(--text-secondary)] truncate mt-0.5">{item.subtitle}</div>
                  </div>
                </button>
              );
            })
          )}
        </div>

        {/* Footer shortcuts */}
        <div className="p-3 border-t border-[var(--border)] bg-[var(--surface-elevated)]/40 flex items-center justify-between text-[11px] text-[var(--text-muted)] font-mono">
          <div className="flex items-center gap-3">
            <span>↑↓ to navigate</span>
            <span>↵ to select</span>
            <button
              type="button"
              onClick={onClose}
              className="hover:text-[var(--text-primary)] hover:underline cursor-pointer transition-colors"
            >
              ESC to close
            </button>
          </div>
          <span>TrustShield AI</span>
        </div>
      </div>
    </div>
  );
}
