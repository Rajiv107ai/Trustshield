"use client";

import React from "react";
import { useViewMode } from "@/context/ViewModeContext";
import { useTheme } from "@/context/ThemeContext";
import { useGuide } from "@/context/GuideContext";
import { Sparkles, Microscope, RefreshCw, Sun, Moon, Search, HelpCircle } from "lucide-react";

export function Header() {
  const { viewMode, setViewMode, isLive, isDemo, setDemo, apiUrl, refreshHealth } = useViewMode();
  const { theme, toggleTheme } = useTheme();
  const { openTour, openPalette } = useGuide();

  return (
    <header className="h-16 border-b border-[var(--border)] bg-[var(--surface)] px-6 flex items-center justify-between sticky top-0 z-30 select-none transition-colors">
      {/* Left Status Indicators */}
      <div className="flex items-center gap-4">
        {/* Backend Connectivity Status */}
        <div className="flex items-center gap-2 px-2.5 py-1 rounded-md bg-[var(--surface-elevated)] border border-[var(--border)] text-xs">
          {isDemo ? (
            <>
              <span className="w-2 h-2 rounded-full bg-amber-500" />
              <span className="text-amber-600 dark:text-amber-400 font-medium">Demo Mode (Simulation)</span>
              <button
                onClick={() => setDemo(false)}
                className="text-[10px] text-blue-600 dark:text-cyan-400 hover:underline ml-1 font-mono cursor-pointer"
              >
                [Live]
              </button>
            </>
          ) : isLive ? (
            <>
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              <span className="text-emerald-600 dark:text-emerald-400 font-medium">Gateway Live</span>
            </>
          ) : (
            <>
              <span className="w-2 h-2 rounded-full bg-rose-500" />
              <span className="text-rose-600 dark:text-rose-400 font-medium">Backend Offline</span>
              <button
                onClick={() => setDemo(true)}
                className="text-[10px] text-amber-600 dark:text-amber-400 hover:underline ml-1 font-mono cursor-pointer"
              >
                [Demo]
              </button>
            </>
          )}
          <span className="text-[var(--text-muted)]">|</span>
          <span className="font-mono text-[11px] text-[var(--text-secondary)] truncate max-w-[140px]">{apiUrl}</span>
          <button
            onClick={() => refreshHealth()}
            title="Refresh API Connection"
            className="text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors ml-1 cursor-pointer focus-ring"
            aria-label="Refresh connection health"
          >
            <RefreshCw className="w-3 h-3" />
          </button>
        </div>

        {/* Model Spec Badge */}
        <div className="hidden lg:flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-[var(--surface-elevated)] border border-[var(--border)] text-xs">
          <span className="text-[var(--text-secondary)]">Engine:</span>
          <span className="font-mono text-blue-600 dark:text-cyan-400 font-semibold">Phase 3 Combined (Graph + XGBoost)</span>
        </div>
      </div>

      {/* Right Controls */}
      <div className="flex items-center gap-2.5">
        {/* Command Palette Button */}
        <button
          onClick={openPalette}
          title="Open Command Palette (Ctrl+K)"
          className="flex items-center gap-2 px-2.5 py-1.5 rounded-lg border border-[var(--border)] bg-[var(--surface-elevated)] hover:bg-[var(--surface-hover)] text-xs text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors cursor-pointer focus-ring"
        >
          <Search className="w-3.5 h-3.5 text-[var(--text-muted)]" />
          <span className="hidden sm:inline">Commands</span>
          <kbd className="hidden sm:inline text-[10px] font-mono px-1 py-0.5 rounded bg-[var(--surface)] border border-[var(--border)] text-[var(--text-muted)]">
            Ctrl K
          </kbd>
        </button>

        {/* Guided Tour Button */}
        <button
          onClick={openTour}
          title="Launch Guided Product Tour"
          className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border border-blue-500/20 bg-blue-500/10 text-blue-600 dark:text-blue-400 text-xs font-semibold hover:bg-blue-500/20 transition-colors cursor-pointer focus-ring"
        >
          <HelpCircle className="w-3.5 h-3.5" />
          <span className="hidden md:inline">Tour & Guide</span>
        </button>

        {/* Theme Toggle (Light/Dark) */}
        <button
          onClick={toggleTheme}
          aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
          title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
          className="w-9 h-9 rounded-lg border border-[var(--border)] bg-[var(--surface-elevated)] flex items-center justify-center text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors cursor-pointer focus-ring"
        >
          {theme === "dark" ? (
            <Sun className="w-4 h-4 text-amber-400 hover:text-amber-300" />
          ) : (
            <Moon className="w-4 h-4 text-blue-600 hover:text-blue-700" />
          )}
        </button>

        {/* DUAL-AUDIENCE GLOBAL VIEW TOGGLE */}
        <div className="flex items-center gap-2">
          <div className="hidden xl:flex items-center gap-1 text-[11px] font-mono text-[var(--text-secondary)]">
            <span className="text-[var(--text-muted)]">Perspective:</span>
            <span
              suppressHydrationWarning
              className={`font-semibold ${viewMode === "executive" ? "text-blue-600 dark:text-blue-400" : "text-purple-600 dark:text-purple-400"}`}
            >
              {viewMode === "executive" ? "Executive ROI" : "Deep ML Forensics"}
            </span>
          </div>

          <div className="flex items-center p-0.5 bg-[var(--background)] border border-[var(--border)] rounded-lg">
            <button
              suppressHydrationWarning
              onClick={() => setViewMode("executive")}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all cursor-pointer ${
                viewMode === "executive"
                  ? "bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-sm font-semibold"
                  : "text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
              }`}
            >
              <Sparkles className="w-3.5 h-3.5" />
              <span>Story View</span>
            </button>
            <button
              suppressHydrationWarning
              onClick={() => setViewMode("inspector")}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all cursor-pointer ${
                viewMode === "inspector"
                  ? "bg-gradient-to-r from-purple-600 to-indigo-600 text-white shadow-sm font-semibold"
                  : "text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
              }`}
            >
              <Microscope className="w-3.5 h-3.5" />
              <span>AI Inspector</span>
            </button>
          </div>
        </div>
      </div>
    </header>
  );
}
