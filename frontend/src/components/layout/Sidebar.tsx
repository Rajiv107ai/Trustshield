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
  Share2,
  Tag,
  RotateCcw,
  FileSearch,
  Cpu,
  BarChart3,
  Server,
  Settings,
  Flame,
} from "lucide-react";

const NAV_ITEMS = [
  {
    category: "DETECTION & SCORING",
    items: [
      { name: "Command Center", href: "/dashboard", icon: LayoutDashboard },
      { name: "Risk Analyzer", href: "/transactions", icon: Zap, badge: "Hero" },
      { name: "Transaction Stream", href: "/transactions/feed", icon: Activity },
    ],
  },
  {
    category: "GRAPH & FORENSICS",
    items: [
      { name: "Fraud Rings", href: "/fraud-rings", icon: Network, badge: "GNN" },
      { name: "Trust Graph", href: "/trust-graph", icon: Share2 },
      { name: "Listing Intelligence", href: "/listings", icon: Tag, badge: "CLIP" },
      { name: "Return Abuse", href: "/returns", icon: RotateCcw },
      { name: "Investigation Cases", href: "/investigations", icon: FileSearch },
    ],
  },
  {
    category: "OBSERVABILITY & MODELS",
    items: [
      { name: "Model Registry", href: "/models", icon: Cpu },
      { name: "Evaluation Studio", href: "/evaluation", icon: BarChart3 },
      { name: "Telemetry & Mesh", href: "/monitoring", icon: Server },
      { name: "Settings", href: "/settings", icon: Settings },
    ],
  },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="w-64 border-r border-[#202A35] bg-[#0E131A] flex flex-col shrink-0 select-none">
      {/* Brand Header */}
      <div className="h-16 px-5 flex items-center justify-between border-b border-[#202A35]">
        <Link href="/dashboard" className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center text-white shadow-lg shadow-blue-500/20">
            <ShieldAlert className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="font-bold tracking-tight text-sm text-[#E8EDF3] flex items-center gap-1.5">
              <span>TrustShield</span>
              <span className="text-[10px] uppercase font-mono px-1 py-0.5 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20">
                AI
              </span>
            </div>
            <div className="text-[11px] text-[#8995A3]">Enterprise Console</div>
          </div>
        </Link>
      </div>

      {/* Nav List */}
      <div className="flex-1 overflow-y-auto py-4 px-3 space-y-6">
        {NAV_ITEMS.map((section) => (
          <div key={section.category} className="space-y-1">
            <div className="px-3 text-[10px] font-semibold tracking-wider text-[#596574] uppercase">
              {section.category}
            </div>
            {section.items.map((item) => {
              const Icon = item.icon;
              const isActive = pathname === item.href || (item.href !== "/dashboard" && pathname.startsWith(item.href) && item.href !== "/transactions" && pathname !== "/transactions/feed");
              const isHeroActive = item.href === "/transactions" && pathname === "/transactions";

              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`flex items-center justify-between px-3 py-2 rounded-md text-xs font-medium transition-colors ${
                    isActive || isHeroActive
                      ? "bg-[#151D27] text-white border-l-2 border-blue-500 font-semibold shadow-sm"
                      : "text-[#8995A3] hover:text-[#E8EDF3] hover:bg-[#111821]"
                  }`}
                >
                  <div className="flex items-center gap-2.5">
                    <Icon className={`w-4 h-4 ${isActive || isHeroActive ? "text-blue-400" : "text-[#596574]"}`} />
                    <span>{item.name}</span>
                  </div>
                  {item.badge && (
                    <span
                      className={`text-[9px] px-1.5 py-0.2 rounded font-mono uppercase font-semibold ${
                        item.badge === "Hero"
                          ? "bg-amber-500/10 text-amber-400 border border-amber-500/20"
                          : item.badge === "GNN"
                          ? "bg-purple-500/10 text-purple-400 border border-purple-500/20"
                          : "bg-cyan-500/10 text-cyan-400 border border-cyan-500/20"
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
      </div>

      {/* Footer Info */}
      <div className="p-3 border-t border-[#202A35] bg-[#0A0E13] text-[11px] text-[#596574] flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Flame className="w-3.5 h-3.5 text-amber-500" />
          <span>v2.0.0-phase5</span>
        </div>
        <div className="font-mono text-[10px] text-[#8995A3]">192 Tests Passed</div>
      </div>
    </aside>
  );
}
