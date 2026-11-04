"use client";

import React from "react";
import Link from "next/link";
import { useViewMode } from "@/context/ViewModeContext";
import { useTheme } from "@/context/ThemeContext";
import { useGuide } from "@/context/GuideContext";
import {
  Sparkles,
  Microscope,
  RefreshCw,
  Sun,
  Moon,
  Search,
  Settings as SettingsIcon,
  Menu,
  ShieldCheck,
  Zap,
} from "lucide-react";

export interface HeaderProps {
  onOpenMobileNav?: () => void;
}

export function Header({ onOpenMobileNav }: HeaderProps = {}) {
  const { viewMode, setViewMode, isLive, isDemo, setDemo, apiUrl, refreshHealth } = useViewMode();
  const { theme, toggleTheme } = useTheme();
  const { openTour, openPalette } = useGuide();

  return (
    <header className="h-16 border-b border-[var(--border)] bg-[var(--surface)]/85 backdrop-blur-md px-4 sm:px-6 flex items-center justify-between sticky top-0 z-30 select-none transition-colors">
      {/* Left Status Indicators */}
      <div className="flex items-center gap-2 sm:gap-3.5">
        {onOpenMobileNav && (
          <button
            onClick={onOpenMobileNav}
            aria-label="Open navigation sidebar"
            className="lg:hidden p-2 rounded-xl border border-[var(--border)] bg-[var(--surface-elevated)] hover:bg-[var(--surface-hover)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-all focus-ring cursor-pointer"
          >
            <Menu className="w-4 h-4" />
          </button>
        )}

        {/* Backend Connectivity Status Pill with live radar pulse */}
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-[var(--surface-elevated)] border border-[var(--border)] text-xs shadow-xs">
          {isDemo ? (
            <div className="flex items-center gap-2">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-amber-400 opacity-75" />
                <span className="relative inline-flex rounded-full h-2 w-2 bg-amber-500" />
              </span>
              <span className="text-amber-500 dark:text-amber-400 font-semibold tracking-wide text-[11px]">
                Simulation Demo
              </span>
              <button
                onClick={() => setDemo(false)}
                title="Switch to Live Gateway connection"
                className="text-[10px] px-1.5 py-0.2 rounded bg-amber-500/10 text-amber-500 hover:bg-amber-500/20 font-mono transition-colors cursor-pointer"
              >
                Go Live
              </button>
            </div>
          ) : isLive ? (
            <div className="flex items-center gap-2">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
              </span>
              <span className="text-emerald-500 dark:text-emerald-400 font-semibold tracking-wide text-[11px] flex items-center gap-1">
                <span>Gateway Active</span>
                <span className="text-[10px] font-mono text-[var(--text-muted)] font-normal hidden sm:inline">(&lt;15ms)</span>
              </span>
            </div>
          ) : (
            <div className="flex items-center gap-2">
              <span className="relative flex h-2 w-2">
                <span className="relative inline-flex rounded-full h-2 w-2 bg-rose-500" />
              </span>
              <span className="text-rose-500 dark:text-rose-400 font-semibold tracking-wide text-[11px]">
                Gateway Offline
              </span>
              <button
                onClick={() => setDemo(true)}
                title="Enable Simulation Mode to browse with demo data"
                className="text-[10px] px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/20 hover:bg-amber-500/20 font-mono font-semibold cursor-pointer"
              >
                Enable Demo
              </button>
            </div>
          )}

          <span className="text-[var(--text-muted)]/40 hidden sm:inline">•</span>

          <Link
            href="/settings"
            title="Configure Gateway & API Key in Settings"
            className="hidden sm:inline font-mono text-[11px] text-[var(--text-secondary)] hover:text-blue-500 dark:hover:text-blue-400 truncate max-w-[120px] transition-colors"
          >
            {apiUrl}
          </Link>

          <button
            onClick={() => refreshHealth()}
            title="Refresh connection status"
            className="text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors p-0.5 rounded hover:bg-[var(--surface)] cursor-pointer focus-ring"
            aria-label="Refresh connection health"
          >
            <RefreshCw className="w-3 h-3" />
          </button>
        </div>

        {/* Model Spec Badge */}
        <div className="hidden xl:flex items-center gap-2 px-3 py-1.5 rounded-full bg-blue-500/5 border border-blue-500/20 text-xs">
          <ShieldCheck className="w-3.5 h-3.5 text-blue-500" />
          <span className="text-[11px] text-[var(--text-secondary)]">Production Engine:</span>
          <span className="font-mono text-[11px] text-blue-600 dark:text-blue-400 font-semibold">
            Phase 3 Combined (GNN + XGBoost)
          </span>
        </div>
      </div>

      {/* Right Controls */}
      <div className="flex items-center gap-2 sm:gap-2.5">
        {/* Command Palette Button */}
        <button
          onClick={openPalette}
          title="Open Command Palette (Ctrl+K)"
          className="flex items-center gap-2 px-3 py-1.5 rounded-xl border border-[var(--border)] bg-[var(--surface-elevated)] hover:bg-[var(--surface-hover)] text-xs text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-all cursor-pointer focus-ring shadow-xs group"
        >
          <Search className="w-3.5 h-3.5 text-[var(--text-muted)] group-hover:text-blue-500 transition-colors" />
          <span className="hidden sm:inline font-medium">Search</span>
          <kbd className="hidden sm:inline text-[10px] font-mono px-1.5 py-0.5 rounded-md bg-[var(--surface)] border border-[var(--border)] text-[var(--text-muted)] shadow-2xs">
            Ctrl K
          </kbd>
        </button>

        {/* Guided Tour Button */}
        <button
          onClick={openTour}
          title="Launch Guided Product Tour"
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl border border-blue-500/20 bg-blue-500/10 text-blue-600 dark:text-blue-400 text-xs font-semibold hover:bg-blue-500/20 hover:border-blue-500/30 transition-all cursor-pointer focus-ring shadow-xs"
        >
          <Sparkles className="w-3.5 h-3.5" />
          <span className="hidden md:inline">Tour</span>
        </button>

        {/* Settings Shortcut Button */}
        <Link
          href="/settings"
          title="Settings & API Keys"
          className="w-9 h-9 rounded-xl border border-[var(--border)] bg-[var(--surface-elevated)] hover:bg-[var(--surface-hover)] flex items-center justify-center text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-all cursor-pointer focus-ring shadow-xs"
        >
          <SettingsIcon className="w-4 h-4" />
        </Link>

        {/* Theme Toggle (Light/Dark) */}
        <button
          onClick={toggleTheme}
          aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
          title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
          className="w-9 h-9 rounded-xl border border-[var(--border)] bg-[var(--surface-elevated)] hover:bg-[var(--surface-hover)] flex items-center justify-center text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-all cursor-pointer focus-ring shadow-xs"
        >
          {theme === "dark" ? (
            <Sun className="w-4 h-4 text-amber-400 hover:text-amber-300 transition-colors" />
          ) : (
            <Moon className="w-4 h-4 text-blue-600 hover:text-blue-700 transition-colors" />
          )}
        </button>

        {/* DUAL-AUDIENCE GLOBAL VIEW TOGGLE */}
        <div className="flex items-center p-0.5 bg-[var(--surface-elevated)] border border-[var(--border)] rounded-xl shadow-xs">
          <button
            suppressHydrationWarning
            onClick={() => setViewMode("executive")}
            className={`flex items-center gap-1.5 px-2.5 sm:px-3 py-1.5 rounded-lg text-xs transition-all cursor-pointer ${
              viewMode === "executive"
                ? "bg-[var(--surface)] text-[var(--text-primary)] font-semibold shadow-xs border border-[var(--border)] text-blue-600 dark:text-blue-400"
                : "text-[var(--text-secondary)] hover:text-[var(--text-primary)] font-medium"
            }`}
          >
            <Zap className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Story</span>
          </button>
          <button
            suppressHydrationWarning
            onClick={() => setViewMode("inspector")}
            className={`flex items-center gap-1.5 px-2.5 sm:px-3 py-1.5 rounded-lg text-xs transition-all cursor-pointer ${
              viewMode === "inspector"
                ? "bg-[var(--surface)] text-[var(--text-primary)] font-semibold shadow-xs border border-[var(--border)] text-purple-600 dark:text-purple-400"
                : "text-[var(--text-secondary)] hover:text-[var(--text-primary)] font-medium"
            }`}
          >
            <Microscope className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Inspector</span>
          </button>
        </div>
      </div>
    </header>
  );
}
