"use client";

import React, { useState } from "react";
import { usePathname } from "next/navigation";
import Link from "next/link";
import { Sidebar } from "@/components/layout/Sidebar";
import { Header } from "@/components/layout/Header";
import { useTheme } from "@/context/ThemeContext";
import { useGuide } from "@/context/GuideContext";
import { ShieldAlert, Sun, Moon, ArrowRight, Menu, X, HelpCircle, Search } from "lucide-react";

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const isLandingPage = pathname === "/";
  const { theme, toggleTheme } = useTheme();
  const { openTour, openPalette } = useGuide();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  if (isLandingPage) {
    return (
      <div className="min-h-screen flex flex-col bg-[var(--background)] text-[var(--text-primary)]">
        {/* Sleek Public Landing Navigation Header */}
        <header className="border-b border-[var(--border)] bg-[var(--surface)] sticky top-0 z-50 backdrop-blur-md">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
            <Link href="/" className="flex items-center gap-2.5 group">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center text-white shadow-md shadow-blue-500/20 group-hover:scale-105 transition-transform">
                <ShieldAlert className="w-5 h-5 text-white" />
              </div>
              <div className="flex items-center gap-1.5">
                <span className="font-bold text-base tracking-tight text-[var(--text-primary)]">TrustShield</span>
                <span className="text-[10px] uppercase font-mono font-bold px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-600 dark:text-blue-400 border border-blue-500/20">
                  AI
                </span>
              </div>
            </Link>

            {/* Desktop Navigation Links */}
            <nav className="hidden md:flex items-center gap-6 text-xs font-medium text-[var(--text-secondary)]">
              <a href="#capabilities" className="hover:text-[var(--text-primary)] transition-colors">
                Core Capabilities
              </a>
              <a href="#workflow" className="hover:text-[var(--text-primary)] transition-colors">
                Decision Pipeline
              </a>
              <a href="#architecture" className="hover:text-[var(--text-primary)] transition-colors">
                Zero-Daemon Architecture
              </a>
              <a href="#audited-results" className="hover:text-[var(--text-primary)] transition-colors">
                Audited Results
              </a>
            </nav>

            <div className="flex items-center gap-2.5">
              {/* Commands Button */}
              <button
                onClick={openPalette}
                title="Search and Commands (Ctrl+K)"
                className="hidden sm:flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border border-[var(--border)] bg-[var(--surface-elevated)] hover:bg-[var(--surface-hover)] text-xs text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors"
              >
                <Search className="w-3.5 h-3.5 text-[var(--text-muted)]" />
                <span className="hidden lg:inline text-[11px]">Commands</span>
                <kbd className="hidden lg:inline text-[10px] font-mono px-1 py-0.5 rounded bg-[var(--surface)] border border-[var(--border)] text-[var(--text-muted)]">
                  Ctrl K
                </kbd>
              </button>

              {/* Product Tour Trigger */}
              <button
                onClick={openTour}
                className="hidden sm:flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border border-blue-500/20 bg-blue-500/10 text-blue-600 dark:text-blue-400 text-xs font-semibold hover:bg-blue-500/20 transition-colors"
              >
                <HelpCircle className="w-3.5 h-3.5" />
                <span>Product Tour</span>
              </button>

              {/* Light/Dark Theme Switch */}
              <button
                onClick={toggleTheme}
                aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
                className="w-9 h-9 rounded-lg border border-[var(--border)] bg-[var(--surface-elevated)] flex items-center justify-center text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors focus-ring"
              >
                {theme === "dark" ? <Sun className="w-4 h-4 text-amber-400" /> : <Moon className="w-4 h-4 text-blue-600" />}
              </button>

              {/* Primary Console CTA */}
              <Link
                href="/dashboard"
                className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-xs font-semibold text-white shadow-md shadow-blue-500/20 transition-all card-interactive"
              >
                <span>Launch Console</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </Link>

              {/* Mobile Menu Toggle */}
              <button
                onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                className="md:hidden p-2 rounded-lg border border-[var(--border)] text-[var(--text-secondary)]"
                aria-label="Toggle navigation menu"
              >
                {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
              </button>
            </div>
          </div>

          {/* Mobile Navigation Drawer */}
          {mobileMenuOpen && (
            <div className="md:hidden border-t border-[var(--border)] bg-[var(--surface)] p-4 space-y-3">
              <a
                href="#capabilities"
                onClick={() => setMobileMenuOpen(false)}
                className="block text-xs font-medium text-[var(--text-secondary)] hover:text-[var(--text-primary)] py-1"
              >
                Core Capabilities
              </a>
              <a
                href="#workflow"
                onClick={() => setMobileMenuOpen(false)}
                className="block text-xs font-medium text-[var(--text-secondary)] hover:text-[var(--text-primary)] py-1"
              >
                Decision Pipeline
              </a>
              <a
                href="#architecture"
                onClick={() => setMobileMenuOpen(false)}
                className="block text-xs font-medium text-[var(--text-secondary)] hover:text-[var(--text-primary)] py-1"
              >
                Zero-Daemon Architecture
              </a>
              <a
                href="#audited-results"
                onClick={() => setMobileMenuOpen(false)}
                className="block text-xs font-medium text-[var(--text-secondary)] hover:text-[var(--text-primary)] py-1"
              >
                Audited Results
              </a>
            </div>
          )}
        </header>

        {/* Public Landing Content */}
        <main className="flex-1">{children}</main>

        {/* Public Landing Footer */}
        <footer className="border-t border-[var(--border)] bg-[var(--surface)] py-8 px-4 sm:px-6 lg:px-8 text-xs text-[var(--text-muted)]">
          <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-2 font-mono">
              <span className="font-semibold text-[var(--text-primary)]">TrustShield AI</span>
              <span>·</span>
              <span>v2.1.0-lean</span>
              <span>·</span>
              <span className="text-emerald-500 font-medium">Single-Container Serving</span>
            </div>
            <div className="flex items-center gap-4 text-xs font-medium text-[var(--text-secondary)]">
              <Link href="/dashboard" className="hover:text-[var(--text-primary)] transition-colors">
                Command Center
              </Link>
              <Link href="/transactions" className="hover:text-[var(--text-primary)] transition-colors">
                Risk Analyzer
              </Link>
              <Link href="/evaluation" className="hover:text-[var(--text-primary)] transition-colors">
                20-Seed Benchmarks
              </Link>
            </div>
          </div>
        </footer>
      </div>
    );
  }

  // Operator Console Layout (for /dashboard, /transactions, /fraud-rings, etc.)
  return (
    <div className="flex w-full min-h-screen bg-[var(--background)] text-[var(--text-primary)]">
      <Sidebar />
      <div className="flex-1 flex flex-col min-w-0 overflow-y-auto">
        <Header />
        <main className="flex-1 p-6 max-w-7xl w-full mx-auto">{children}</main>
      </div>
    </div>
  );
}
