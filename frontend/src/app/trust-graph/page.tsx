"use client";

import React from "react";
import Link from "next/link";
import { GitBranch, Network, ArrowRight } from "lucide-react";

export default function TrustGraphPage() {
  return (
    <div className="max-w-4xl space-y-6">
      <div>
        <h1 className="text-xl font-bold tracking-tight text-[var(--text-primary)] flex items-center gap-2">
          <GitBranch className="w-5 h-5 text-purple-600 dark:text-purple-400" />
          <span>Multi-Relational Trust Graph Canvas</span>
        </h1>
        <p className="text-xs text-[var(--text-secondary)] mt-1">
          Interactive full-topology multi-entity relational canvas (Buyers, Sellers, Devices, Bank Cards, Physical Addresses).
        </p>
      </div>

      <div className="p-8 rounded-xl border border-[var(--border)] bg-[var(--surface)] text-center space-y-4">
        <div className="w-12 h-12 rounded-full bg-purple-500/10 text-purple-600 dark:text-purple-400 mx-auto flex items-center justify-center">
          <Network className="w-6 h-6" />
        </div>
        <div>
          <h2 className="text-sm font-bold text-[var(--text-primary)]">Interactive Canvas Under Development</h2>
          <p className="text-xs text-[var(--text-secondary)] mt-1 max-w-md mx-auto">
            The full-mesh 500k-node interactive WebGL graph explorer is scheduled for the next release.
            In this build, precomputed bipartite collusion clusters and component metrics are available via the Collusion Ring Canvas.
          </p>
        </div>
        <div>
          <Link
            href="/fraud-rings"
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-xs font-semibold text-white transition-all shadow-sm"
          >
            <span>Explore Detected Collusion Rings</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </Link>
        </div>
      </div>
    </div>
  );
}
