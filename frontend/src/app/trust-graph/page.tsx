"use client";

import React, { useState } from "react";
import { useSearchParams } from "next/navigation";
import {
  Share2,
  Search,
  Filter,
  Maximize2,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Sliders,
  ShieldAlert,
  Smartphone,
  MapPin,
  Store,
  User,
  ShoppingBag,
  Info,
  Clock,
  X,
} from "lucide-react";

interface GraphNode {
  id: string;
  type: "buyer" | "seller" | "device" | "address" | "order";
  label: string;
  x: number;
  y: number;
  risk: number;
  details: {
    ageDays?: number;
    connectedOrders?: number;
    connectedDevices?: number;
    ipHash?: string;
    addressLine?: string;
    fraudFlagged?: boolean;
  };
}

interface GraphEdge {
  from: string;
  to: string;
  label: string;
  suspicious?: boolean;
}

const INITIAL_NODES: GraphNode[] = [
  { id: "BUYER_RING_MEMBER_04", type: "buyer", label: "BUYER_RING_MEMBER_04", x: 280, y: 160, risk: 0.94, details: { ageDays: 4, connectedOrders: 9, connectedDevices: 3, fraudFlagged: true } },
  { id: "BUYER_RING_MEMBER_02", type: "buyer", label: "BUYER_RING_MEMBER_02", x: 220, y: 320, risk: 0.91, details: { ageDays: 2, connectedOrders: 6, connectedDevices: 2, fraudFlagged: true } },
  { id: "BUYER_VERIFIED_77", type: "buyer", label: "BUYER_VERIFIED_77", x: 680, y: 220, risk: 0.04, details: { ageDays: 180, connectedOrders: 15, connectedDevices: 1, fraudFlagged: false } },
  { id: "DEV_FARM_99", type: "device", label: "DEV_FARM_99", x: 380, y: 240, risk: 0.98, details: { connectedDevices: 9, ipHash: "192.168.1.104", fraudFlagged: true } },
  { id: "ADDR_WH_01", type: "address", label: "ADDR_WH_01 (Drop Ship)", x: 290, y: 420, risk: 0.89, details: { addressLine: "Suite 4B, Industrial Blvd", fraudFlagged: true } },
  { id: "SELLER_RING_LEADER_01", type: "seller", label: "SELLER_RING_LEADER_01", x: 490, y: 360, risk: 0.92, details: { ageDays: 12, connectedOrders: 63, fraudFlagged: true } },
  { id: "SELLER_REPUTABLE_12", type: "seller", label: "SELLER_REPUTABLE_12", x: 740, y: 380, risk: 0.03, details: { ageDays: 420, connectedOrders: 412, fraudFlagged: false } },
  { id: "ORD_SIM_FARM_03", type: "order", label: "ORD_SIM_FARM_03 ($890)", x: 420, y: 110, risk: 0.95, details: { connectedOrders: 1, fraudFlagged: true } },
];

const INITIAL_EDGES: GraphEdge[] = [
  { from: "BUYER_RING_MEMBER_04", to: "DEV_FARM_99", label: "USES_DEVICE", suspicious: true },
  { from: "BUYER_RING_MEMBER_02", to: "DEV_FARM_99", label: "USES_DEVICE", suspicious: true },
  { from: "BUYER_RING_MEMBER_04", to: "ADDR_WH_01", label: "SHIPS_TO", suspicious: true },
  { from: "BUYER_RING_MEMBER_02", to: "ADDR_WH_01", label: "SHIPS_TO", suspicious: true },
  { from: "BUYER_RING_MEMBER_04", to: "SELLER_RING_LEADER_01", label: "TRANSACTS_WITH", suspicious: true },
  { from: "BUYER_RING_MEMBER_04", to: "ORD_SIM_FARM_03", label: "PLACES_ORDER", suspicious: true },
  { from: "BUYER_VERIFIED_77", to: "SELLER_REPUTABLE_12", label: "TRANSACTS_WITH", suspicious: false },
];
import { Suspense } from "react";

function TrustGraphContent() {
  const searchParams = useSearchParams();
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(INITIAL_NODES[0]);
  const [filterSuspicious, setFilterSuspicious] = useState<boolean>(false);
  const [timelineStep, setTimelineStep] = useState<number>(30); // days
  const [searchQuery, setSearchQuery] = useState<string>(searchParams.get("search") || "");

  const filteredNodes = INITIAL_NODES.filter((n) => {
    if (filterSuspicious && n.risk < 0.5) return false;
    if (searchQuery && !n.label.toLowerCase().includes(searchQuery.toLowerCase())) return false;
    return true;
  });

  const getNodeColor = (type: string) => {
    switch (type) {
      case "buyer": return { bg: "#3B82F6", border: "#60A5FA", icon: User };
      case "seller": return { bg: "#8B5CF6", border: "#A78BFA", icon: Store };
      case "device": return { bg: "#06B6D4", border: "#22D3EE", icon: Smartphone };
      case "address": return { bg: "#F59E0B", border: "#FBBF24", icon: MapPin };
      case "order": return { bg: "#10B981", border: "#34D399", icon: ShoppingBag };
      default: return { bg: "#64748B", border: "#94A3B8", icon: Info };
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-[#E8EDF3] flex items-center gap-2">
            <Share2 className="w-5 h-5 text-blue-400" />
            <span>Trust Graph Explorer & Multi-Relational Canvas</span>
          </h1>
          <p className="text-xs text-[#8995A3] mt-1">
            2-hop heterogeneous graph traversal mapping bipartite buyers, merchants, shared devices, and delivery destinations.
          </p>
        </div>

        {/* Legend */}
        <div className="flex items-center gap-3 text-[11px] font-mono">
          <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-blue-500" /> Buyer</span>
          <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-purple-500" /> Seller</span>
          <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-cyan-500" /> Device</span>
          <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-amber-500" /> Address</span>
          <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-emerald-500" /> Order</span>
        </div>
      </div>

      {/* Main Canvas + Inspector Drawer Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Left 3 Cols: Interactive Visual Graph */}
        <div className="lg:col-span-3 rounded-xl border border-[#202A35] bg-[#0E131A] overflow-hidden flex flex-col h-[650px] relative">
          {/* Controls Bar */}
          <div className="p-3 border-b border-[#202A35] bg-[#111821] flex flex-wrap items-center justify-between gap-3 text-xs">
            <div className="flex items-center gap-2">
              <div className="flex items-center px-2 py-1 rounded bg-[#0A0E13] border border-[#202A35]">
                <Search className="w-3.5 h-3.5 text-[#596574] mr-1.5" />
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Find entity..."
                  className="bg-transparent text-[11px] text-[#E8EDF3] placeholder-[#596574] focus:outline-none w-28 font-mono"
                />
              </div>

              <button
                onClick={() => setFilterSuspicious(!filterSuspicious)}
                className={`px-2.5 py-1 rounded text-xs font-mono font-medium border transition-colors ${
                  filterSuspicious
                    ? "bg-rose-500/20 text-rose-300 border-rose-500/40"
                    : "bg-[#0A0E13] text-[#8995A3] border-[#202A35] hover:text-[#E8EDF3]"
                }`}
              >
                {filterSuspicious ? "Suspicious Only: ON" : "Filter Suspicious"}
              </button>
            </div>

            {/* Temporal Timeline Slider */}
            <div className="flex items-center gap-2 font-mono text-[11px] text-[#8995A3]">
              <Clock className="w-3.5 h-3.5 text-blue-400" />
              <span>Cutoff: T - {timelineStep}d</span>
              <input
                type="range"
                min="1"
                max="90"
                value={timelineStep}
                onChange={(e) => setTimelineStep(parseInt(e.target.value))}
                className="w-24 accent-blue-500"
              />
            </div>
          </div>

          {/* SVG Canvas */}
          <div className="flex-1 w-full h-full relative cursor-crosshair overflow-hidden">
            <svg className="w-full h-full">
              {/* Grid Lines Pattern */}
              <defs>
                <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
                  <path d="M 40 0 L 0 0 0 40" fill="none" stroke="rgba(255,255,255,0.03)" strokeWidth="1" />
                </pattern>
              </defs>
              <rect width="100%" height="100%" fill="url(#grid)" />

              {/* Render Edges */}
              {INITIAL_EDGES.map((edge, idx) => {
                const source = INITIAL_NODES.find((n) => n.id === edge.from);
                const target = INITIAL_NODES.find((n) => n.id === edge.to);
                if (!source || !target) return null;
                const isSelected = selectedNode?.id === source.id || selectedNode?.id === target.id;
                return (
                  <g key={idx}>
                    <line
                      x1={source.x}
                      y1={source.y}
                      x2={target.x}
                      y2={target.y}
                      stroke={edge.suspicious ? (isSelected ? "#EF4444" : "#F87171") : (isSelected ? "#3B82F6" : "#475569")}
                      strokeWidth={isSelected ? 3 : 1.5}
                      strokeDasharray={edge.suspicious ? "4,4" : undefined}
                      opacity={isSelected ? 1 : 0.6}
                    />
                  </g>
                );
              })}

              {/* Render Nodes */}
              {filteredNodes.map((node) => {
                const config = getNodeColor(node.type);
                const isSelected = selectedNode?.id === node.id;
                return (
                  <g
                    key={node.id}
                    transform={`translate(${node.x}, ${node.y})`}
                    onClick={() => setSelectedNode(node)}
                    className="cursor-pointer transition-transform hover:scale-110"
                  >
                    <circle
                      r={isSelected ? 26 : 20}
                      fill={config.bg}
                      stroke={isSelected ? "#FFFFFF" : config.border}
                      strokeWidth={isSelected ? 3 : 1.5}
                      className={node.risk > 0.7 ? "animate-pulse" : ""}
                    />
                    {node.risk > 0.7 && (
                      <circle
                        r={isSelected ? 34 : 28}
                        fill="none"
                        stroke="#EF4444"
                        strokeWidth="1.5"
                        strokeDasharray="3,3"
                        opacity="0.8"
                      />
                    )}
                    <text
                      y={isSelected ? 42 : 36}
                      textAnchor="middle"
                      fill="#E8EDF3"
                      fontSize="10"
                      fontFamily="monospace"
                      fontWeight="600"
                    >
                      {node.label.length > 18 ? node.label.slice(0, 16) + "..." : node.label}
                    </text>
                  </g>
                );
              })}
            </svg>

            {/* Bottom Invariant Banner */}
            <div className="absolute bottom-3 left-3 px-3 py-1.5 rounded-lg bg-[#0A0E13]/90 border border-[#202A35] text-[10px] font-mono text-[#8995A3] backdrop-blur-sm">
              &Delta;t &ge; 0 temporal directionality enforced · Zero retrospective snooping
            </div>
          </div>
        </div>

        {/* Right Col: Entity Inspector Drawer */}
        <div className="rounded-xl border border-[#202A35] bg-[#111821] p-5 space-y-4">
          <div className="flex items-center justify-between border-b border-[#202A35] pb-3">
            <span className="text-xs font-bold uppercase tracking-wider text-[#E8EDF3]">
              Entity Inspector
            </span>
            {selectedNode && (
              <span className="text-[10px] font-mono px-2 py-0.5 rounded uppercase font-semibold bg-blue-500/10 text-blue-400">
                {selectedNode.type}
              </span>
            )}
          </div>

          {selectedNode ? (
            <div className="space-y-4 text-xs font-mono">
              <div>
                <div className="text-[10px] text-[#8995A3]">ENTITY IDENTIFIER</div>
                <div className="text-sm font-bold text-[#E8EDF3] mt-0.5 break-all">{selectedNode.id}</div>
              </div>

              {/* Risk Level Gauge */}
              <div className="p-3 rounded-lg bg-[#0E131A] border border-[#202A35]">
                <div className="flex items-center justify-between text-[10px]">
                  <span className="text-[#8995A3]">GNN ESTIMATED RISK</span>
                  <span
                    className={`font-bold ${
                      selectedNode.risk > 0.7 ? "text-rose-400" : selectedNode.risk > 0.3 ? "text-amber-400" : "text-emerald-400"
                    }`}
                  >
                    {(selectedNode.risk * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="h-1.5 rounded-full bg-[#151D27] mt-2 overflow-hidden">
                  <div
                    style={{ width: `${selectedNode.risk * 100}%` }}
                    className={selectedNode.risk > 0.7 ? "bg-rose-500 h-full" : "bg-emerald-500 h-full"}
                  />
                </div>
              </div>

              {/* Entity Attributes */}
              <div className="space-y-2 text-[11px] pt-1">
                {selectedNode.details.ageDays !== undefined && (
                  <div className="flex justify-between py-1 border-b border-[#202A35]/60">
                    <span className="text-[#8995A3]">Account Age:</span>
                    <span className="text-[#E8EDF3]">{selectedNode.details.ageDays} days</span>
                  </div>
                )}
                {selectedNode.details.connectedOrders !== undefined && (
                  <div className="flex justify-between py-1 border-b border-[#202A35]/60">
                    <span className="text-[#8995A3]">Connected Orders:</span>
                    <span className="text-[#E8EDF3]">{selectedNode.details.connectedOrders}</span>
                  </div>
                )}
                {selectedNode.details.connectedDevices !== undefined && (
                  <div className="flex justify-between py-1 border-b border-[#202A35]/60">
                    <span className="text-[#8995A3]">Device Collisions:</span>
                    <span className="text-rose-400 font-semibold">{selectedNode.details.connectedDevices}</span>
                  </div>
                )}
                {selectedNode.details.ipHash && (
                  <div className="flex justify-between py-1 border-b border-[#202A35]/60">
                    <span className="text-[#8995A3]">Hardware IP:</span>
                    <span className="text-[#E8EDF3]">{selectedNode.details.ipHash}</span>
                  </div>
                )}
                {selectedNode.details.addressLine && (
                  <div className="flex justify-between py-1 border-b border-[#202A35]/60">
                    <span className="text-[#8995A3]">Shipping Dest:</span>
                    <span className="text-[#E8EDF3]">{selectedNode.details.addressLine}</span>
                  </div>
                )}
                <div className="flex justify-between py-1">
                  <span className="text-[#8995A3]">Collusion Ring:</span>
                  <span className={selectedNode.details.fraudFlagged ? "text-rose-400 font-bold" : "text-emerald-400"}>
                    {selectedNode.details.fraudFlagged ? "RING_COLLUSION_01" : "None"}
                  </span>
                </div>
              </div>

              {/* Action Button */}
              {selectedNode.details.fraudFlagged && (
                <button className="w-full py-2 px-3 rounded-lg bg-rose-600/20 hover:bg-rose-600/30 border border-rose-500/30 text-rose-300 font-semibold text-xs transition-colors mt-2">
                  Freeze Node in Graph Ledger
                </button>
              )}
            </div>
          ) : (
            <div className="py-12 text-center text-[#596574] text-xs">
              Click any node in the canvas to inspect its relational attributes.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default function TrustGraphPage() {
  return (
    <Suspense fallback={<div className="p-12 text-center text-xs font-mono text-[#8995A3]">Loading Trust Graph...</div>}>
      <TrustGraphContent />
    </Suspense>
  );
}
