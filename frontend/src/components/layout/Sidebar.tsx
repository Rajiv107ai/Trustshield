"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  ShieldAlert,
  LayoutDashboard,
  Zap,
  Activity,
  Network,
  Tag,
  RotateCcw,
  FileSearch,
  Cpu,
  BarChart3,
  CheckCircle2,
  ExternalLink,
} from "lucide-react";

const NAV_ITEMS = [
  {
    category: "OVERVIEW",
    items: [
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
      { name: "Investigation Cases", href: "/investigations", icon: FileSearch },
    ],
  },
  {
    category: "SCIENCE & GOVERNANCE",
    items: [
      { name: "Model Registry", href: "/models", icon: Cpu, badge: "v2.1.0" },
      { name: "Evaluation Studio", href: "/evaluation", icon: BarChart3, badge: "20 Seeds" },
    ],
  },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="w-64 border-r border-[var(--border)] bg-[var(--surface)] flex flex-col shrink-0 select-none transition-colors">
      {/* Brand Header */}
      <div className="h-16 px-5 flex items-center justify-between border-b border-[var(--border)]">
        <Link href="/dashboard" className="flex items-center gap-2.5 group">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center text-white shadow-md shadow-blue-500/20 group-hover:scale-105 transition-transform">
            <ShieldAlert className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="font-bold tracking-tight text-sm text-[var(--text-primary)] flex items-center gap-1.5">
              <span>TrustShield</span>
              <span className="text-[10px] uppercase font-mono px-1 py-0.5 rounded bg-blue-500/10 text-blue-600 dark:text-blue-400 border border-blue-500/20">
                AI
              </span>
            </div>
            <div className="text-[11px] text-[var(--text-secondary)]">Enterprise Console</div>
          </div>
        </Link>

        <Link
          href="/"
          title="View Public Overview Landing Page"
          className="text-[var(--text-muted)] hover:text-[var(--text-primary)] p-1.5 rounded-md hover:bg-[var(--surface-elevated)] transition-colors"
        >
          <ExternalLink className="w-3.5 h-3.5" />
        </Link>
      </div>

      {/* Nav List */}
      <nav aria-label="Main Navigation" className="flex-1 overflow-y-auto py-4 px-3 space-y-6">
        {NAV_ITEMS.map((section) => (
          <div key={section.category} className="space-y-1">
            <div className="px-3 text-[10px] font-semibold tracking-wider text-[var(--text-muted)] uppercase font-mono">
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
                  className={`flex items-center justify-between px-3 py-2 rounded-lg text-xs font-medium transition-colors ${
                    isActive || isHeroActive
                      ? "bg-[var(--surface-elevated)] text-[var(--text-primary)] border-l-2 border-blue-500 font-semibold shadow-sm"
                      : "text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--surface-elevated)]"
                  }`}
                >
                  <div className="flex items-center gap-2.5">
                    <Icon
                      className={`w-4 h-4 ${
                        isActive || isHeroActive ? "text-blue-600 dark:text-blue-400" : "text-[var(--text-muted)]"
                      }`}
                    />
                    <span>{item.name}</span>
                  </div>
                  {item.badge && (
                    <span
                      className={`text-[9px] px-1.5 py-0.5 rounded font-mono uppercase font-semibold border ${
                        item.badge === "Live"
                          ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20"
                          : item.badge === "Phase 3"
                          ? "bg-blue-500/10 text-blue-600 dark:text-blue-400 border-blue-500/20"
                          : item.badge === "Graph"
                          ? "bg-purple-500/10 text-purple-600 dark:text-purple-400 border-purple-500/20"
                          : "bg-cyan-500/10 text-cyan-600 dark:text-cyan-400 border-cyan-500/20"
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

      {/* Footer Info */}
      <div className="p-3 border-t border-[var(--border)] bg-[var(--surface-elevated)] text-[11px] text-[var(--text-muted)] flex items-center justify-between">
        <div className="flex items-center gap-1.5">
          <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500" />
          <span className="font-mono text-[10px] text-[var(--text-secondary)]">v2.1.0-lean</span>
        </div>
        <div className="font-mono text-[10px] text-emerald-600 dark:text-emerald-400 font-medium">
          328 Tests Passed
        </div>
      </div>
    </aside>
  );
}
