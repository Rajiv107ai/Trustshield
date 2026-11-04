"use client";

import React, { useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  ShieldAlert,
  LayoutDashboard,
  Zap,
  Activity,
  Network,
  GitBranch,
  Tag,
  RotateCcw,
  FileSearch,
  Cpu,
  BarChart3,
  ExternalLink,
  Settings,
  Home,
  X,
} from "lucide-react";

const NAV_ITEMS = [
  {
    category: "OPERATIONAL HUB",
    items: [
      { name: "Home / Overview", href: "/overview", icon: Home },
      { name: "Command Center", href: "/dashboard", icon: LayoutDashboard },
    ],
  },
  {
    category: "RISK OPERATIONS",
    items: [
      { name: "Risk Analyzer", href: "/transactions", icon: Zap, badge: "Phase 3" },
      { name: "Transaction Feed", href: "/transactions/feed", icon: Activity, badge: "Live" },
      { name: "Listing Intelligence", href: "/listings", icon: Tag, badge: "CLIP" },
      { name: "Return Abuse", href: "/returns", icon: RotateCcw },
    ],
  },
  {
    category: "FORENSICS & INTELLIGENCE",
    items: [
      { name: "Fraud Rings", href: "/fraud-rings", icon: Network, badge: "Graph" },
      { name: "Trust Graph", href: "/trust-graph", icon: GitBranch, badge: "Topology" },
      { name: "Investigation Cases", href: "/investigations", icon: FileSearch },
    ],
  },
  {
    category: "SCIENCE & GOVERNANCE",
    items: [
      { name: "Model Registry", href: "/models", icon: Cpu, badge: "v2.1.0" },
      { name: "Evaluation Studio", href: "/evaluation", icon: BarChart3, badge: "20 Seeds" },
      { name: "Model Monitoring", href: "/monitoring", icon: Activity, badge: "Drift" },
    ],
  },
  {
    category: "SYSTEM & GATEWAY",
    items: [
      { name: "Settings & API Keys", href: "/settings", icon: Settings },
    ],
  },
];

function NavItemsList({
  pathname,
  onNavigate,
}: {
  pathname: string;
  onNavigate?: () => void;
}) {
  return (
    <nav aria-label="Main Navigation" className="flex-1 overflow-y-auto py-4 px-3 space-y-6">
      {NAV_ITEMS.map((section) => (
        <div key={section.category} className="space-y-1">
          <div className="px-3 text-[10px] font-semibold tracking-wider text-[var(--text-muted)] uppercase">
            {section.category}
          </div>
          {section.items.map((item) => {
            const Icon = item.icon;
            const isActive =
              pathname === item.href ||
              (item.href !== "/dashboard" &&
                pathname.startsWith(item.href) &&
                item.href !== "/transactions" &&
                pathname !== "/transactions/feed");
            const isHeroActive = item.href === "/transactions" && pathname === "/transactions";

            return (
              <Link
                key={item.href}
                href={item.href}
                onClick={onNavigate}
                className={`flex items-center justify-between px-3 py-2 rounded-xl text-xs font-medium transition-all focus-ring ${
                  isActive || isHeroActive
                    ? "bg-gradient-to-r from-blue-500/15 via-blue-500/5 to-transparent text-[var(--text-primary)] border-l-2 border-blue-500 font-semibold shadow-xs"
                    : "text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--surface-elevated)]/60 hover:translate-x-0.5"
                }`}
              >
                <div className="flex items-center gap-2.5">
                  <Icon
                    className={`w-4 h-4 transition-colors ${
                      isActive || isHeroActive ? "text-blue-500 dark:text-blue-400" : "text-[var(--text-muted)]"
                    }`}
                  />
                  <span>{item.name}</span>
                </div>
                {item.badge && (
                  <span
                    className={`text-[9px] px-2 py-0.5 rounded-full font-mono uppercase font-semibold border ${
                      item.badge === "Live"
                        ? "bg-emerald-500/10 text-emerald-500 dark:text-emerald-400 border-emerald-500/20"
                        : item.badge === "Phase 3"
                        ? "bg-blue-500/10 text-blue-500 dark:text-blue-400 border-blue-500/20"
                        : item.badge === "Graph"
                        ? "bg-purple-500/10 text-purple-500 dark:text-purple-400 border-purple-500/20"
                        : "bg-cyan-500/10 text-cyan-500 dark:text-cyan-400 border-cyan-500/20"
                    }`}
                  >
                    {item.badge}
                  </span>
                )}
              </Link>
            );
          })}
        </div>
      ))}
    </nav>
  );
}

function SidebarFooter() {
  return (
    <div className="p-3.5 border-t border-[var(--border)] bg-[var(--surface-elevated)]/40 text-[11px] text-[var(--text-muted)] flex items-center justify-between shrink-0">
      <div className="flex items-center gap-2">
        <span className="relative flex h-2 w-2">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
          <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
        </span>
        <span className="font-mono text-[10px] text-[var(--text-secondary)]">Engine Active</span>
      </div>
      <div className="font-mono text-[10px] px-2 py-0.5 rounded-md bg-[var(--surface-elevated)] border border-[var(--border)] text-emerald-600 dark:text-emerald-400 font-medium">
        P95 &lt;15ms
      </div>
    </div>
  );
}

export interface SidebarProps {
  mobileOpen?: boolean;
  onCloseMobile?: () => void;
}

export function Sidebar({ mobileOpen = false, onCloseMobile }: SidebarProps = {}) {
  const pathname = usePathname();

  useEffect(() => {
    if (!mobileOpen) return;

    // Lock background body scroll while mobile drawer is open
    const originalOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onCloseMobile?.();
      }
    };

    // Close mobile drawer if viewport expands to desktop (>= 1024px)
    const handleResize = () => {
      if (window.innerWidth >= 1024) {
        onCloseMobile?.();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("resize", handleResize);

    return () => {
      document.body.style.overflow = originalOverflow;
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("resize", handleResize);
    };
  }, [mobileOpen, onCloseMobile]);

  return (
    <>
      {/* Desktop Sticky Sidebar (Visible lg and above) */}
      <aside className="hidden lg:flex w-64 border-r border-[var(--border)] bg-[var(--surface)] flex-col shrink-0 select-none transition-colors h-screen sticky top-0">
        {/* Brand Header */}
        <div className="h-16 px-5 flex items-center justify-between border-b border-[var(--border)] shrink-0">
          <Link href="/dashboard" className="flex items-center gap-3 group">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-blue-600 to-indigo-600 flex items-center justify-center text-white shadow-md shadow-blue-500/25 group-hover:scale-105 transition-all duration-300">
              <ShieldAlert className="w-5 h-5 text-white" />
            </div>
            <div>
              <div className="font-bold tracking-tight text-sm text-[var(--text-primary)] flex items-center gap-1.5">
                <span>TrustShield</span>
                <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded-full bg-blue-500/10 text-blue-600 dark:text-blue-400 border border-blue-500/20 font-semibold">
                  AI
                </span>
              </div>
              <div className="text-[11px] text-[var(--text-secondary)]">Risk &amp; Trust Platform</div>
            </div>
          </Link>

          <Link
            href="/overview"
            title="View Public Overview Landing Page"
            className="text-[var(--text-muted)] hover:text-[var(--text-primary)] p-1.5 rounded-lg hover:bg-[var(--surface-elevated)] transition-colors focus-ring"
          >
            <ExternalLink className="w-3.5 h-3.5" />
          </Link>
        </div>

        {/* Navigation Items */}
        <NavItemsList pathname={pathname} />

        {/* Footer Info */}
        <SidebarFooter />
      </aside>

      {/* Mobile Drawer (Visible < lg when opened) */}
      {mobileOpen && (
        <div className="fixed inset-0 z-50 lg:hidden flex">
          {/* Backdrop overlay */}
          <div
            onClick={onCloseMobile}
            aria-hidden="true"
            className="fixed inset-0 bg-black/60 backdrop-blur-xs transition-opacity duration-200"
          />

          {/* Off-canvas drawer panel */}
          <aside
            aria-label="Mobile Navigation"
            className="relative w-72 max-w-[85vw] bg-[var(--surface)] border-r border-[var(--border)] flex flex-col h-full shadow-2xl z-10 select-none animate-in slide-in-from-left duration-200"
          >
            {/* Mobile Header with Close Button */}
            <div className="h-16 px-5 flex items-center justify-between border-b border-[var(--border)] shrink-0">
              <Link
                href="/dashboard"
                onClick={onCloseMobile}
                className="flex items-center gap-3 group"
              >
                <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-blue-600 to-indigo-600 flex items-center justify-center text-white shadow-md shadow-blue-500/25">
                  <ShieldAlert className="w-5 h-5 text-white" />
                </div>
                <div>
                  <div className="font-bold tracking-tight text-sm text-[var(--text-primary)] flex items-center gap-1.5">
                    <span>TrustShield</span>
                    <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded-full bg-blue-500/10 text-blue-600 dark:text-blue-400 border border-blue-500/20 font-semibold">
                      AI
                    </span>
                  </div>
                  <div className="text-[11px] text-[var(--text-secondary)]">Risk &amp; Trust Platform</div>
                </div>
              </Link>

              <button
                onClick={onCloseMobile}
                aria-label="Close navigation sidebar"
                className="p-1.5 rounded-lg border border-[var(--border)] bg-[var(--surface-elevated)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors focus-ring cursor-pointer"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Navigation Items */}
            <NavItemsList pathname={pathname} onNavigate={onCloseMobile} />

            {/* Footer Info */}
            <SidebarFooter />
          </aside>
        </div>
      )}
    </>
  );
}
