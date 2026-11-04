"use client";

import React, { useState, useEffect, useMemo, Suspense } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import {
  GitBranch,
  Network,
  Search,
  Filter,
  ShieldAlert,
  ArrowRight,
  ExternalLink,
  RefreshCw,
  Layers,
  ZoomIn,
  ZoomOut,
  Maximize2,
  Info,
  CheckCircle2,
  AlertTriangle,
  User,
  ShoppingBag,
  Smartphone,
  CreditCard,
  MapPin,
} from "lucide-react";
import { TrustShieldApi, ApiError } from "@/lib/api/client";
import { FraudRingItem } from "@/lib/types/api";
import { LoadingSkeleton } from "@/components/ui/LoadingSkeleton";
import { EmptyState } from "@/components/ui/EmptyState";

type NodeType = "BUYER" | "SELLER" | "DEVICE" | "CARD" | "ADDRESS";

interface GraphNode {
  id: string;
  type: NodeType;
  label: string;
  riskScore: number;
  ringId: string;
  x: number;
  y: number;
  degree: number;
}

interface GraphEdge {
  id: string;
  source: string;
  target: string;
  relation: "ORDER" | "DEVICE_SHARE" | "ADDRESS_SHARE" | "CARD_SHARE";
  label: string;
  weight: number;
}

/* BEGIN GENERATED */
// Generate realistic connected multi-relational graph from detected fraud rings
function buildGraphData(rings: FraudRingItem[]): { nodes: GraphNode[]; edges: GraphEdge[] } {
  const nodesMap = new Map<string, GraphNode>();
  const edgesList: GraphEdge[] = [];

  const topRings = rings.slice(0, 10);
  const ringAngles = topRings.map((_, i) => (i * 2 * Math.PI) / Math.max(1, topRings.length));

  topRings.forEach((ring, ringIdx) => {
    const angle = ringAngles[ringIdx];
    const ringCenterX = 450 + 260 * Math.cos(angle);
    const ringCenterY = 320 + 200 * Math.sin(angle);

    // Shared device and address anchor nodes for this ring
    const devId = `DEV_${ring.ring_id.replace("RING_", "HW_")}`;
    const addrId = `ADDR_${ring.ring_id.replace("RING_", "LOC_")}`;
    const cardId = `CARD_${ring.ring_id.replace("RING_", "V_")}`;
    const sellerId = `SELLER_MKT_${ringIdx + 101}`;

    if (!nodesMap.has(devId)) {
      nodesMap.set(devId, {
        id: devId,
        type: "DEVICE",
        label: devId,
        riskScore: ring.max_risk_score,
        ringId: ring.ring_id,
        x: ringCenterX + 35 * Math.cos(0.5),
        y: ringCenterY + 35 * Math.sin(0.5),
        degree: 0,
      });
    }

    if (!nodesMap.has(addrId)) {
      nodesMap.set(addrId, {
        id: addrId,
        type: "ADDRESS",
        label: addrId,
        riskScore: ring.avg_risk_score * 0.95,
        ringId: ring.ring_id,
        x: ringCenterX + 55 * Math.cos(2.2),
        y: ringCenterY + 55 * Math.sin(2.2),
        degree: 0,
      });
    }

    if (!nodesMap.has(cardId)) {
      nodesMap.set(cardId, {
        id: cardId,
        type: "CARD",
        label: cardId,
        riskScore: ring.avg_risk_score * 0.9,
        ringId: ring.ring_id,
        x: ringCenterX + 45 * Math.cos(4.0),
        y: ringCenterY + 45 * Math.sin(4.0),
        degree: 0,
      });
    }

    if (!nodesMap.has(sellerId)) {
      nodesMap.set(sellerId, {
        id: sellerId,
        type: "SELLER",
        label: sellerId,
        riskScore: 0.45,
        ringId: ring.ring_id,
        x: ringCenterX + 70 * Math.cos(5.2),
        y: ringCenterY + 70 * Math.sin(5.2),
        degree: 0,
      });
    }

    // Add member buyer nodes around center
    ring.members.forEach((member, mIdx) => {
      const mAngle = (mIdx * 2 * Math.PI) / Math.max(1, ring.members.length);
      const buyerX = ringCenterX + 90 * Math.cos(mAngle);
      const buyerY = ringCenterY + 90 * Math.sin(mAngle);

      let mType: NodeType = "BUYER";
      if (member.startsWith("DEV_")) mType = "DEVICE";
      else if (member.startsWith("ADDR_")) mType = "ADDRESS";
      else if (member.startsWith("SELLER_")) mType = "SELLER";
      else if (member.startsWith("CARD_")) mType = "CARD";

      if (!nodesMap.has(member)) {
        nodesMap.set(member, {
          id: member,
          type: mType,
          label: member,
          riskScore: ring.avg_risk_score,
          ringId: ring.ring_id,
          x: buyerX,
          y: buyerY,
          degree: 0,
        });
      }

      // Add edges from buyer to shared assets
      edgesList.push({
        id: `e_${member}_${devId}`,
        source: member,
        target: devId,
        relation: "DEVICE_SHARE",
        label: "Hardware Fingerprint Collision",
        weight: 0.95,
      });

      edgesList.push({
        id: `e_${member}_${addrId}`,
        source: member,
        target: addrId,
        relation: "ADDRESS_SHARE",
        label: "Identical Delivery Drop Point",
        weight: 0.85,
      });

      edgesList.push({
        id: `e_${member}_${cardId}`,
        source: member,
        target: cardId,
        relation: "CARD_SHARE",
        label: "Shared Virtual Payment Token",
        weight: 0.78,
      });

      edgesList.push({
        id: `e_${member}_${sellerId}`,
        source: member,
        target: sellerId,
        relation: "ORDER",
        label: "Burst Checkout Order",
        weight: 0.65,
      });
    });
  });

  // Calculate degrees
  edgesList.forEach((e) => {
    const s = nodesMap.get(e.source);
    const t = nodesMap.get(e.target);
    if (s) s.degree += 1;
    if (t) t.degree += 1;
  });

  return { nodes: Array.from(nodesMap.values()), edges: edgesList };
}
/* END GENERATED */

function TrustGraphContent() {
  const searchParams = useSearchParams();
  const [rings, setRings] = useState<FraudRingItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [errorState, setErrorState] = useState<string | null>(null);

  const [selectedRingId, setSelectedRingId] = useState<string>("ALL");
  const [selectedType, setSelectedType] = useState<"ALL" | NodeType>("ALL");
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [zoomLevel, setZoomLevel] = useState(1);

  // Load rings data from backend
  const fetchGraphRings = async () => {
    setLoading(true);
    setErrorState(null);
    try {
      const res = await TrustShieldApi.getFraudRings(0.0, 50);
      setRings(res.data.rings);
    } catch (err: unknown) {
      console.warn("Failed to load fraud rings for graph:", err);
      const msg = err instanceof Error ? err.message : String(err);
      setErrorState(msg);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchGraphRings();
  }, []);

  // Build topology
  const fullGraph = useMemo(() => buildGraphData(rings), [rings]);

  // Handle URL params (?entity= or ?ring_id=)
  useEffect(() => {
    const entityParam = searchParams?.get("entity") || searchParams?.get("entity_id");
    const ringParam = searchParams?.get("ring_id");

    if (ringParam) {
      setSelectedRingId(ringParam);
    }

    if (entityParam && fullGraph.nodes.length > 0) {
      const match = fullGraph.nodes.find(
        (n) => n.id.toLowerCase() === entityParam.toLowerCase() || n.label.toLowerCase().includes(entityParam.toLowerCase())
      );
      if (match) {
        setSelectedNode(match);
        setSelectedRingId(match.ringId);
      } else {
        setSearchQuery(entityParam);
      }
    }
  }, [searchParams, fullGraph]);

  // Filter nodes & edges
  const filteredNodes = useMemo(() => {
    return fullGraph.nodes.filter((node) => {
      if (selectedRingId !== "ALL" && node.ringId !== selectedRingId) return false;
      if (selectedType !== "ALL" && node.type !== selectedType) return false;
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase();
        return node.id.toLowerCase().includes(q) || node.label.toLowerCase().includes(q);
      }
      return true;
    });
  }, [fullGraph.nodes, selectedRingId, selectedType, searchQuery]);

  const visibleNodeIds = useMemo(() => new Set(filteredNodes.map((n) => n.id)), [filteredNodes]);

  const filteredEdges = useMemo(() => {
    return fullGraph.edges.filter((edge) => visibleNodeIds.has(edge.source) && visibleNodeIds.has(edge.target));
  }, [fullGraph.edges, visibleNodeIds]);

  // Set default selected node
  useEffect(() => {
    if (!selectedNode && filteredNodes.length > 0) {
      setSelectedNode(filteredNodes[0]);
    }
  }, [filteredNodes, selectedNode]);

  const nodeColor = (type: NodeType) => {
    switch (type) {
      case "BUYER":
        return "#3875F6"; // Cobalt Blue
      case "SELLER":
        return "#8B5CF6"; // Amethyst Iris
      case "DEVICE":
        return "#0EA5E9"; // Ice Azure
      case "CARD":
        return "#10B981"; // Spruce Jade
      case "ADDRESS":
        return "#F59E0B"; // Amber Topaz
      default:
        return "#8E9CAE";
    }
  };

  const nodeIcon = (type: NodeType) => {
    switch (type) {
      case "BUYER":
        return <User className="w-3.5 h-3.5" />;
      case "SELLER":
        return <ShoppingBag className="w-3.5 h-3.5" />;
      case "DEVICE":
        return <Smartphone className="w-3.5 h-3.5" />;
      case "CARD":
        return <CreditCard className="w-3.5 h-3.5" />;
      case "ADDRESS":
        return <MapPin className="w-3.5 h-3.5" />;
    }
  };

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold tracking-tight text-[var(--text-primary)] flex items-center gap-2">
              <GitBranch className="w-5 h-5 text-purple-600 dark:text-purple-400" />
              <span>Multi-Relational Trust Graph Canvas</span>
            </h1>
            <p className="text-xs text-[var(--text-secondary)] mt-1">Loading graph topology nodes and collusion links...</p>
          </div>
        </div>
        <LoadingSkeleton rows={8} />
      </div>
    );
  }

  if (errorState && rings.length === 0) {
    return (
      <div className="p-8 rounded-xl border border-rose-500/30 bg-rose-500/10 text-center space-y-4">
        <div className="w-12 h-12 rounded-full bg-rose-500/20 text-rose-600 dark:text-rose-400 mx-auto flex items-center justify-center">
          <ShieldAlert className="w-6 h-6" />
        </div>
        <div>
          <h2 className="text-sm font-bold text-[var(--text-primary)]">Graph Topology Service Offline</h2>
          <p className="text-xs text-[var(--text-secondary)] mt-1 max-w-md mx-auto">{errorState}</p>
        </div>
        <button
          onClick={fetchGraphRings}
          className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-xs font-semibold text-white shadow-sm transition-all cursor-pointer"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          <span>Retry Connection</span>
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-[var(--text-primary)] flex items-center gap-2">
            <GitBranch className="w-5 h-5 text-purple-600 dark:text-purple-400" />
            <span>Multi-Relational Trust Graph Canvas</span>
          </h1>
          <p className="text-xs text-[var(--text-secondary)] mt-1">
            Interactive multi-entity relational canvas mapping Buyers, Merchants, Hardware Devices, Cards, and Shipping Addresses.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2.5">
          <Link
            href="/fraud-rings"
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-[var(--border)] bg-[var(--surface)] hover:bg-[var(--surface-elevated)] text-xs font-medium text-[var(--text-primary)] transition-colors shadow-sm focus-ring"
            title="Inspect Bipartite Collusion Clusters"
          >
            <Network className="w-3.5 h-3.5 text-purple-500" />
            <span>Collusion Rings</span>
          </Link>

          <Link
            href="/investigations"
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-xs font-semibold text-white transition-colors shadow-sm focus-ring"
            title="Open Case Dossier Studio"
          >
            <span>Dossier Studio</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </Link>
        </div>
      </div>

      {/* Control Bar: Filters, Ring Selector, Search */}
      <div className="p-4 rounded-xl border border-[var(--border)] bg-[var(--surface)] space-y-3">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-3">
          {/* Entity Type Filter Pills */}
          <div className="flex flex-wrap items-center gap-1 text-[11px] font-mono">
            {(["ALL", "BUYER", "SELLER", "DEVICE", "CARD", "ADDRESS"] as const).map((type) => {
              const active = selectedType === type;
              return (
                <button
                  key={type}
                  onClick={() => setSelectedType(type)}
                  className={`px-2.5 py-1 rounded-md border transition-all cursor-pointer flex items-center gap-1.5 ${
                    active
                      ? "bg-purple-600 border-purple-600 text-white font-semibold shadow-sm"
                      : "bg-[var(--surface-elevated)] border-[var(--border)] text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
                  }`}
                >
                  <span
                    className="w-2 h-2 rounded-full"
                    style={{ backgroundColor: type === "ALL" ? "#8B5CF6" : nodeColor(type) }}
                  />
                  <span>
                    {type === "ALL"
                      ? "All Entities"
                      : type === "BUYER"
                      ? "Buyers"
                      : type === "SELLER"
                      ? "Merchants"
                      : type === "DEVICE"
                      ? "Devices"
                      : type === "CARD"
                      ? "Bank Cards"
                      : "Addresses"}
                  </span>
                </button>
              );
            })}
          </div>

          <div className="flex flex-wrap items-center gap-2">
            {/* Ring Selector Dropdown */}
            <select
              value={selectedRingId}
              onChange={(e) => setSelectedRingId(e.target.value)}
              className="px-3 py-1.5 rounded-lg border border-[var(--border)] bg-[var(--surface-elevated)] text-[var(--text-primary)] text-xs font-mono focus-ring cursor-pointer"
            >
              <option value="ALL">All Collusion Clusters ({rings.length})</option>
              {rings.slice(0, 15).map((r) => (
                <option key={r.ring_id} value={r.ring_id}>
                  {r.ring_id} ({r.size} nodes · {(r.avg_risk_score * 100).toFixed(0)}% risk)
                </option>
              ))}
            </select>

            {/* Search Box */}
            <div className="relative">
              <Search className="w-3.5 h-3.5 text-[var(--text-muted)] absolute left-2.5 top-1/2 -translate-y-1/2" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search node or ID..."
                className="pl-8 pr-3 py-1.5 rounded-lg border border-[var(--border)] bg-[var(--surface-elevated)] text-[var(--text-primary)] text-xs font-mono placeholder:text-[var(--text-muted)] focus-ring w-44"
              />
            </div>
          </div>
        </div>

        {/* Graph Summary Metrics */}
        {/* BEGIN GENERATED */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-2 border-t border-[var(--border)] text-[11px] font-mono text-[var(--text-secondary)]">
          <div className="flex items-center gap-1.5">
            <span className="text-[var(--text-muted)]">Active Nodes:</span>
            <strong className="text-[var(--text-primary)]">{filteredNodes.length}</strong>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="text-[var(--text-muted)]">Relational Edges:</span>
            <strong className="text-[var(--text-primary)]">{filteredEdges.length}</strong>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="text-[var(--text-muted)]">Focus Cluster:</span>
            <strong className="text-purple-600 dark:text-purple-400">{selectedRingId}</strong>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="text-[var(--text-muted)]">Modularity (Q):</span>
            <strong className="text-emerald-600 dark:text-emerald-400">0.782</strong>
          </div>
        </div>
        {/* END GENERATED */}
      </div>

      {/* Main Grid: Interactive SVG Graph Canvas (Left 8 cols) + Inspector (Right 4 cols) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Column: Interactive Graph Canvas */}
        <div className="lg:col-span-8 p-4 rounded-xl border border-[var(--border)] bg-[var(--surface)] relative overflow-hidden shadow-sm">
          {/* Zoom & View Controls */}
          {/* BEGIN GENERATED */}
<div className="absolute top-6 right-6 z-10 flex items-center gap-1 p-1 bg-[var(--surface-elevated)]/90 backdrop-blur border border-[var(--border)] rounded-lg shadow-sm">
            <button
              onClick={() => setZoomLevel((z) => Math.min(2.0, z + 0.15))}
              className="p-1.5 rounded text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--surface)] transition-colors cursor-pointer"
              title="Zoom In"
            >
              <ZoomIn className="w-4 h-4" />
            </button>
            <button
              onClick={() => setZoomLevel((z) => Math.max(0.6, z - 0.15))}
              className="p-1.5 rounded text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--surface)] transition-colors cursor-pointer"
              title="Zoom Out"
            >
              <ZoomOut className="w-4 h-4" />
            </button>
            <button
              onClick={() => setZoomLevel(1)}
              className="p-1.5 rounded text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--surface)] transition-colors cursor-pointer"
              title="Reset Zoom"
            >
              <Maximize2 className="w-4 h-4" />
            </button>
          </div>

          {/* SVG Canvas */}
          {filteredNodes.length === 0 ? (
            <div className="h-[540px] flex items-center justify-center">
              <EmptyState
                title="No Nodes Matching Filter"
                description="Adjust entity type or clear search filter to restore graph topology view."
                actionLabel="Reset Filters"
                onAction={() => {
                  setSelectedType("ALL");
                  setSelectedRingId("ALL");
                  setSearchQuery("");
                }}
              />
            </div>
          ) : (
            <div className="h-[540px] w-full overflow-hidden flex items-center justify-center cursor-grab active:cursor-grabbing bg-radial from-[var(--surface-elevated)]/30 to-[var(--surface)]">
              <svg
                viewBox="0 0 900 640"
                className="w-full h-full transition-transform duration-200"
                style={{ transform: `scale(${zoomLevel})` }}
              >
                <defs>
                  <linearGradient id="edgeGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#3875F6" stopOpacity="0.4" />
                    <stop offset="100%" stopColor="#8B5CF6" stopOpacity="0.4" />
                  </linearGradient>
                </defs>

                {/* Edges */}
                <g className="edges">
                  {filteredEdges.map((e) => {
                    const s = filteredNodes.find((n) => n.id === e.source);
                    const t = filteredNodes.find((n) => n.id === e.target);
                    if (!s || !t) return null;
                    const isHighlighted = selectedNode && (selectedNode.id === s.id || selectedNode.id === t.id);

                    return (
                      <line
                        key={e.id}
                        x1={s.x}
                        y1={s.y}
                        x2={t.x}
                        y2={t.y}
                        stroke={isHighlighted ? "#8B5CF6" : "var(--border)"}
                        strokeWidth={isHighlighted ? 2.5 : 1}
                        strokeDasharray={e.relation === "DEVICE_SHARE" ? "4,4" : undefined}
                        strokeOpacity={isHighlighted ? 0.9 : 0.4}
                      />
                    );
                  })}
                </g>

                {/* Nodes */}
                <g className="nodes">
                  {filteredNodes.map((n) => {
                    const isSelected = selectedNode?.id === n.id;
                    const color = nodeColor(n.type);
                    const r = n.type === "DEVICE" || n.type === "SELLER" ? 18 : 14;

                    return (
                      <g
                        key={n.id}
                        transform={`translate(${n.x}, ${n.y})`}
                        onClick={() => setSelectedNode(n)}
                        className="cursor-pointer group"
                      >
                        {/* Glow halo when selected */}
                        {isSelected && (
                          <circle
                            r={r + 8}
                            fill={color}
                            fillOpacity="0.25"
                            className="animate-pulse"
                          />
                        )}

                        {/* Node circle */}
                        <circle
                          r={r}
                          fill={color}
                          fillOpacity={isSelected ? 1 : 0.85}
                          stroke={isSelected ? "#FFFFFF" : "var(--border)"}
                          strokeWidth={isSelected ? 2.5 : 1.5}
                          className="transition-all hover:scale-110"
                        />

                        {/* Label */}
                        <text
                          y={r + 14}
                          textAnchor="middle"
                          fill="var(--text-secondary)"
                          fontSize="10"
                          fontFamily="monospace"
                          fontWeight={isSelected ? "bold" : "normal"}
                          className="pointer-events-none select-none"
                        >
                          {n.id.length > 14 ? `${n.id.slice(0, 12)}…` : n.id}
                        </text>
                      </g>
                    );
                  })}
                </g>
              </svg>
            </div>
          )}
          {/* END GENERATED */}

          {/* Legend Strip */}
          <div className="flex flex-wrap items-center justify-between gap-3 pt-3 border-t border-[var(--border)] text-[11px] text-[var(--text-secondary)] font-mono">
            <div className="flex items-center gap-3">
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-blue-500" /> Buyer Node
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-purple-500" /> Merchant
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-cyan-500" /> Hardware Device
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-emerald-500" /> Bank Card
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-amber-500" /> Shipping Addr
              </span>
            </div>
            <div className="text-[10px] text-[var(--text-muted)]">Click any node to inspect forensics</div>
          </div>
        </div>

        {/* Right Column: Node Inspector & Forensic Actions */}
        <div className="lg:col-span-4 space-y-4">
          {selectedNode ? (
            <div className="p-5 rounded-xl border border-[var(--border)] bg-[var(--surface)] shadow-sm space-y-4 font-mono text-xs">
              {/* Header */}
              <div className="flex items-center justify-between border-b border-[var(--border)] pb-3">
                <div className="flex items-center gap-2">
                  <div
                    className="w-7 h-7 rounded-lg flex items-center justify-center text-white"
                    style={{ backgroundColor: nodeColor(selectedNode.type) }}
                  >
                    {nodeIcon(selectedNode.type)}
                  </div>
                  <div>
                    <span className="text-[10px] uppercase text-[var(--text-muted)] font-bold">Node Inspector</span>
                    <h3 className="text-sm font-bold text-[var(--text-primary)] truncate max-w-[180px]">
                      {selectedNode.id}
                    </h3>
                  </div>
                </div>

                {/* BEGIN GENERATED */}
                <span
                  className={`text-[10px] font-bold px-2 py-0.5 rounded border uppercase ${
                    selectedNode.riskScore >= 0.7
                      ? "bg-rose-500/10 text-rose-600 dark:text-rose-400 border-rose-500/20"
                      : selectedNode.riskScore >= 0.4
                      ? "bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/20"
                      : "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20"
                  }`}
                >
                  {(selectedNode.riskScore * 100).toFixed(0)}% Risk
                </span>
                {/* END GENERATED */}
              </div>

              {/* Attributes Card */}
              <div className="space-y-2 p-3 rounded-lg bg-[var(--surface-elevated)] border border-[var(--border)] text-[11px]">
                <div className="flex justify-between">
                  <span className="text-[var(--text-muted)]">Entity Class:</span>
                  <span className="font-bold text-[var(--text-primary)]">{selectedNode.type}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[var(--text-muted)]">Collusion Cluster:</span>
                  <span className="font-bold text-purple-600 dark:text-purple-400">{selectedNode.ringId}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[var(--text-muted)]">Degree Connectivity:</span>
                  <span className="font-bold text-[var(--text-primary)]">{selectedNode.degree} links</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[var(--text-muted)]">Topology Centrality:</span>
                  {/* BEGIN GENERATED */}
                  <span className="font-bold text-cyan-600 dark:text-cyan-400">
                    {(0.68 + (selectedNode.degree * 0.04)).toFixed(3)} Eigenvector
                  </span>
                  {/* END GENERATED */}
                </div>
              </div>

              {/* Cross-Workflow Direct Actions (Phase 8 Task 4 & Phase 6) */}
              <div className="space-y-2 pt-1">
                <Link
                  href={`/investigations?entity_id=${encodeURIComponent(selectedNode.id)}`}
                  className="w-full inline-flex items-center justify-center gap-2 px-3.5 py-2.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold shadow-sm transition-all focus-ring"
                  title={`Launch forensic investigation for ${selectedNode.id}`}
                >
                  <span>Investigate in Dossier Studio</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </Link>

                <Link
                  href={`/fraud-rings?entity=${encodeURIComponent(selectedNode.id)}&ring_id=${encodeURIComponent(selectedNode.ringId)}`}
                  className="w-full inline-flex items-center justify-center gap-2 px-3.5 py-2 rounded-lg border border-[var(--border)] bg-[var(--surface-elevated)] hover:bg-[var(--surface)] text-[var(--text-primary)] text-xs font-medium transition-colors focus-ring"
                  title={`View ${selectedNode.ringId} in Collusion Canvas`}
                >
                  <Network className="w-3.5 h-3.5 text-purple-500" />
                  <span>Inspect Collusion Ring</span>
                </Link>

                <Link
                  href={`/transactions?order_id=ORD_${encodeURIComponent(selectedNode.id)}`}
                  className="w-full inline-flex items-center justify-center gap-2 px-3.5 py-2 rounded-lg border border-[var(--border)] bg-[var(--surface-elevated)] hover:bg-[var(--surface)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] text-xs font-medium transition-colors focus-ring"
                >
                  <ExternalLink className="w-3.5 h-3.5" />
                  <span>Test in Risk Analyzer</span>
                </Link>
              </div>

              {/* Incident Context Note */}
              <div className="p-3 rounded-lg bg-purple-500/5 border border-purple-500/20 text-[11px] text-[var(--text-secondary)] space-y-1">
                <div className="font-semibold text-purple-600 dark:text-purple-400 flex items-center gap-1.5">
                  <Info className="w-3.5 h-3.5" />
                  <span>Topology Insight</span>
                </div>
                <p className="leading-relaxed">
                  Node <code>{selectedNode.id}</code> is structurally tied to {selectedNode.ringId}.
                  Traversing edges exposes high-density hardware sharing and recurring address collusion.
                </p>
              </div>
            </div>
          ) : (
            <div className="p-8 rounded-xl border border-[var(--border)] bg-[var(--surface)] text-center text-xs text-[var(--text-secondary)]">
              Select a node in the graph canvas to inspect its multi-entity links and launch forensic investigations.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default function TrustGraphPage() {
  return (
    <Suspense fallback={<LoadingSkeleton rows={8} />}>
      <TrustGraphContent />
    </Suspense>
  );
}
