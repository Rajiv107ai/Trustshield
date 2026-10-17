"use client";

import React, { useState, useEffect, useMemo, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { useViewMode } from "@/context/ViewModeContext";
import {
  Share2,
  Search,
  Clock,
  User,
  Store,
  Smartphone,
  MapPin,
  ShoppingBag,
  Info,
  Lock,
  Unlock,
  Radio,
  Layers,
  CheckCircle2,
  PlusCircle,
  type LucideIcon,
} from "lucide-react";

// ============================================================================
// Types
// ============================================================================

interface GraphNode {
  id: string;
  type: "buyer" | "seller" | "device" | "address" | "order";
  label: string;
  x: number;
  y: number;
  risk: number;
  createdAtDaysAgo: number; // For temporal cutoff filtering
  frozen?: boolean;
  expanded?: boolean;
  details: {
    ageDays?: number;
    connectedOrders?: number;
    connectedDevices?: number;
    ipHash?: string;
    addressLine?: string;
    fraudFlagged?: boolean;
    clusterId?: string;
    financialExposure?: number;
    gnnEmbeddingSample?: number[];
    pageRank?: number;
    degree?: number;
  };
}

interface GraphEdge {
  id: string;
  from: string;
  to: string;
  label: string;
  suspicious?: boolean;
  createdAtDaysAgo: number;
}

interface ClusterDefinition {
  id: string;
  name: string;
  tag: string;
  riskScore: number;
  riskLevel: "high" | "medium" | "low";
  description: string;
  executiveSummary: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
}

// ============================================================================
// Preset Cluster Topologies (Multi-Relational Datasets)
// ============================================================================

/* BEGIN GENERATED */
let _eid = 0;
const n = (id: string, type: GraphNode["type"], x: number, y: number, risk: number, daysAgo: number, details: GraphNode["details"] = {}, label?: string): GraphNode =>
  ({ id, type, label: label || id, x, y, risk, createdAtDaysAgo: daysAgo, details });
const e = (from: string, to: string, label: string, suspicious: boolean = true, daysAgo: number = 1): GraphEdge =>
  ({ id: `e${++_eid}`, from, to, label, suspicious, createdAtDaysAgo: daysAgo });

const cDef = (
  id: string, name: string, tag: string, riskScore: number, riskLevel: "high" | "medium" | "low",
  description: string, executiveSummary: string, nodes: GraphNode[], edges: GraphEdge[]
): ClusterDefinition => ({ id, name, tag, riskScore, riskLevel, description, executiveSummary, nodes, edges });

const CLUSTERS: ClusterDefinition[] = [
  cDef(
    "RING_COLLUSION_01", "Ring #01: Device Farm & Collusion", "Device Farm", 0.94, "high",
    "2-hop bipartite graph mapping shared device collision and synthetic buyer accounts.",
    "Active botnet farm recycling Android hardware IDs to orchestrate 42 coordinated checkout attempts totaling $38,400 in potential chargeback exposure.",
    [
      n("BUYER_RING_MEMBER_04", "buyer", 280, 160, 0.94, 4, { ageDays: 4, connectedOrders: 9, connectedDevices: 3, fraudFlagged: true, clusterId: "RING_COLLUSION_01", financialExposure: 8450, gnnEmbeddingSample: [0.891, -0.421, 0.763, 0.129], pageRank: 0.0412, degree: 4 }),
      n("BUYER_RING_MEMBER_02", "buyer", 220, 320, 0.91, 2, { ageDays: 2, connectedOrders: 6, connectedDevices: 2, fraudFlagged: true, clusterId: "RING_COLLUSION_01", financialExposure: 5200, gnnEmbeddingSample: [0.884, -0.395, 0.741, 0.118], pageRank: 0.0354, degree: 3 }),
      n("BUYER_VERIFIED_77", "buyer", 680, 220, 0.04, 180, { ageDays: 180, connectedOrders: 15, connectedDevices: 1, fraudFlagged: false, clusterId: "BENIGN_NETWORK", financialExposure: 0, gnnEmbeddingSample: [-0.621, 0.142, -0.312, 0.041], pageRank: 0.0098, degree: 1 }),
      n("DEV_FARM_99", "device", 380, 240, 0.98, 6, { connectedDevices: 9, ipHash: "192.168.1.104", fraudFlagged: true, clusterId: "RING_COLLUSION_01", financialExposure: 18900, gnnEmbeddingSample: [0.952, -0.812, 0.915, 0.442], pageRank: 0.0891, degree: 5 }, "DEV_FARM_99 (Emulated)"),
      n("ADDR_WH_01", "address", 290, 420, 0.89, 25, { addressLine: "Suite 4B, Industrial Blvd", fraudFlagged: true, clusterId: "RING_COLLUSION_01", financialExposure: 14200, gnnEmbeddingSample: [0.712, -0.219, 0.589, 0.301], pageRank: 0.0385, degree: 2 }, "ADDR_WH_01 (Drop Ship Hub)"),
      n("SELLER_RING_LEADER_01", "seller", 490, 360, 0.92, 12, { ageDays: 12, connectedOrders: 63, fraudFlagged: true, clusterId: "RING_COLLUSION_01", financialExposure: 38400, gnnEmbeddingSample: [0.901, -0.582, 0.812, 0.399], pageRank: 0.0712, degree: 4 }),
      n("SELLER_REPUTABLE_12", "seller", 740, 380, 0.03, 420, { ageDays: 420, connectedOrders: 412, fraudFlagged: false, clusterId: "BENIGN_NETWORK", financialExposure: 0, gnnEmbeddingSample: [-0.781, 0.312, -0.512, -0.108], pageRank: 0.0152, degree: 1 }),
      n("ORD_SIM_FARM_03", "order", 420, 110, 0.95, 1, { connectedOrders: 1, fraudFlagged: true, clusterId: "RING_COLLUSION_01", financialExposure: 890, gnnEmbeddingSample: [0.841, -0.612, 0.771, 0.291], pageRank: 0.0125, degree: 1 }, "ORD_SIM_FARM_03 ($890)"),
    ],
    [
      e("BUYER_RING_MEMBER_04", "DEV_FARM_99", "USES_DEVICE", true, 4),
      e("BUYER_RING_MEMBER_02", "DEV_FARM_99", "USES_DEVICE", true, 2),
      e("BUYER_RING_MEMBER_04", "ADDR_WH_01", "SHIPS_TO", true, 4),
      e("BUYER_RING_MEMBER_02", "ADDR_WH_01", "SHIPS_TO", true, 2),
      e("BUYER_RING_MEMBER_04", "SELLER_RING_LEADER_01", "TRANSACTS_WITH", true, 3),
      e("BUYER_RING_MEMBER_04", "ORD_SIM_FARM_03", "PLACES_ORDER", true, 1),
      e("BUYER_VERIFIED_77", "SELLER_REPUTABLE_12", "TRANSACTS_WITH", false, 30),
    ]
  ),
  cDef(
    "RING_PRICE_ARBITRAGE_02", "Ring #02: Price Arbitrage & Freight Hub", "Price Arbitrage", 0.85, "high",
    "Syndicate exploiting discounted promotional codes and shipping to commercial freight forwarders.",
    "Coordinated buyer cluster purchasing high-demand luxury electronics using unauthorized coupons and shipping to an unvetted freight forwarder terminal.",
    [
      n("BUYER_ARBITRAGE_112", "buyer", 250, 180, 0.88, 8, { ageDays: 8, connectedOrders: 14, connectedDevices: 2, fraudFlagged: true, financialExposure: 12400 }),
      n("BUYER_ARBITRAGE_114", "buyer", 320, 340, 0.82, 5, { ageDays: 5, connectedOrders: 8, connectedDevices: 2, fraudFlagged: true, financialExposure: 7800 }),
      n("DEV_ROTATING_PROXY_11", "device", 420, 220, 0.93, 4, { connectedDevices: 8, ipHash: "104.28.19.44", fraudFlagged: true, financialExposure: 20200 }),
      n("ADDR_PORT_TERMINAL_7", "address", 520, 380, 0.91, 45, { addressLine: "Dock 14, Bay Forwarding Corp", fraudFlagged: true, financialExposure: 24500 }),
      n("SELLER_MERCHANT_44", "seller", 620, 190, 0.86, 18, { ageDays: 18, connectedOrders: 34, fraudFlagged: true, financialExposure: 18500 }),
      n("ORD_LUXURY_BULK_09", "order", 350, 100, 0.89, 2, { connectedOrders: 1, fraudFlagged: true, financialExposure: 4200 }, "ORD_LUXURY_BULK_09 ($4,200)"),
    ],
    [
      e("BUYER_ARBITRAGE_112", "DEV_ROTATING_PROXY_11", "USES_DEVICE", true, 8),
      e("BUYER_ARBITRAGE_114", "DEV_ROTATING_PROXY_11", "USES_DEVICE", true, 5),
      e("BUYER_ARBITRAGE_112", "ADDR_PORT_TERMINAL_7", "SHIPS_TO", true, 8),
      e("BUYER_ARBITRAGE_114", "ADDR_PORT_TERMINAL_7", "SHIPS_TO", true, 5),
      e("BUYER_ARBITRAGE_112", "SELLER_MERCHANT_44", "TRANSACTS_WITH", true, 6),
      e("BUYER_ARBITRAGE_112", "ORD_LUXURY_BULK_09", "PLACES_ORDER", true, 2),
    ]
  ),
  cDef(
    "RING_RETURN_ABUSE_03", "Ring #03: Rapid Refund & Wardrobing Syndicate", "Return Abuse", 0.78, "high",
    "Repeated empty-box claims and high-velocity wardrobing returns against luxury fashion apparel.",
    "Syndicate exploiting return guarantees with rapid chargebacks within 24 hours of delivery, sharing burner phone emulators.",
    [
      n("BUYER_REFUND_01", "buyer", 260, 200, 0.89, 35, { ageDays: 35, connectedOrders: 9, fraudFlagged: true, financialExposure: 9400 }, "BUYER_REFUND_01 (7 Returns)"),
      n("BUYER_REFUND_02", "buyer", 320, 360, 0.84, 28, { ageDays: 28, connectedOrders: 6, fraudFlagged: true, financialExposure: 6100 }, "BUYER_REFUND_02 (5 Returns)"),
      n("DEV_MOBILE_EMULATOR_04", "device", 450, 260, 0.94, 14, { connectedDevices: 4, ipHash: "172.56.21.90", fraudFlagged: true, financialExposure: 15500 }),
      n("SELLER_FASHION_BOUTIQUE_88", "seller", 620, 260, 0.12, 320, { ageDays: 320, connectedOrders: 890, fraudFlagged: false, financialExposure: 0 }),
      n("ORD_DESIGNER_RETURN_42", "order", 380, 120, 0.88, 4, { connectedOrders: 1, fraudFlagged: true, financialExposure: 1450 }, "ORD_DESIGNER_RETURN_42 ($1,450)"),
    ],
    [
      e("BUYER_REFUND_01", "DEV_MOBILE_EMULATOR_04", "USES_DEVICE", true, 35),
      e("BUYER_REFUND_02", "DEV_MOBILE_EMULATOR_04", "USES_DEVICE", true, 28),
      e("BUYER_REFUND_01", "SELLER_FASHION_BOUTIQUE_88", "TRANSACTS_WITH", true, 10),
      e("BUYER_REFUND_02", "SELLER_FASHION_BOUTIQUE_88", "TRANSACTS_WITH", true, 8),
      e("BUYER_REFUND_01", "ORD_DESIGNER_RETURN_42", "PLACES_ORDER", true, 4),
    ]
  ),
  cDef(
    "RING_BENIGN_CLUSTER_06", "Baseline: Verified Legitimate Traffic", "Benign Network", 0.04, "low",
    "Established consumers with years of verified transaction history, distinct devices, and low return rates.",
    "Clean, organic e-commerce purchases from high-tenure accounts with zero hardware collisions and trusted payment methods.",
    [
      n("BUYER_VERIFIED_ALICE", "buyer", 280, 220, 0.02, 450, { ageDays: 450, connectedOrders: 38, connectedDevices: 1, fraudFlagged: false, financialExposure: 0 }),
      n("DEV_ALICE_IPHONE_15", "device", 420, 160, 0.01, 450, { connectedDevices: 1, ipHash: "73.189.44.12", fraudFlagged: false, financialExposure: 0 }),
      n("ADDR_HOME_VERIFIED", "address", 420, 320, 0.02, 450, { addressLine: "742 Evergreen Terrace", fraudFlagged: false, financialExposure: 0 }),
      n("SELLER_APPLE_AUTHORIZED", "seller", 620, 240, 0.01, 1200, { ageDays: 1200, connectedOrders: 5400, fraudFlagged: false, financialExposure: 0 }),
      n("ORD_AIRPODS_CLEAN", "order", 350, 100, 0.02, 10, { connectedOrders: 1, fraudFlagged: false, financialExposure: 0 }, "ORD_AIRPODS_CLEAN ($249)"),
    ],
    [
      e("BUYER_VERIFIED_ALICE", "DEV_ALICE_IPHONE_15", "USES_DEVICE", false, 450),
      e("BUYER_VERIFIED_ALICE", "ADDR_HOME_VERIFIED", "SHIPS_TO", false, 450),
      e("BUYER_VERIFIED_ALICE", "SELLER_APPLE_AUTHORIZED", "TRANSACTS_WITH", false, 10),
      e("BUYER_VERIFIED_ALICE", "ORD_AIRPODS_CLEAN", "PLACES_ORDER", false, 10),
    ]
  ),
];

/* END GENERATED */

// ============================================================================
// Main Component
// ============================================================================

function TrustGraphContent() {
  const searchParams = useSearchParams();
  const { viewMode } = useViewMode();

  // Active Cluster & Dynamic Graph State
  const [activeClusterId, setActiveClusterId] = useState<string>("RING_COLLUSION_01");
  const [nodes, setNodes] = useState<GraphNode[]>(CLUSTERS[0].nodes);
  const [edges, setEdges] = useState<GraphEdge[]>(CLUSTERS[0].edges);

  // Inspector & Interactions
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(CLUSTERS[0].nodes[0]);
  const [filterSuspicious, setFilterSuspicious] = useState<boolean>(false);
  const [timelineCutoffDays, setTimelineCutoffDays] = useState<number>(30); // T - X days
  const [searchQuery, setSearchQuery] = useState<string>(searchParams.get("search") || "");
  const [isLiveStreaming, setIsLiveStreaming] = useState<boolean>(false);
  const [streamCount, setStreamCount] = useState<number>(0);
  const [actionFeedback, setActionFeedback] = useState<string | null>(null);

  // Switch cluster
  const handleSelectCluster = (clusterId: string) => {
    const cluster = CLUSTERS.find((c) => c.id === clusterId);
    if (!cluster) return;
    setActiveClusterId(clusterId);
    setNodes(cluster.nodes);
    setEdges(cluster.edges);
    setSelectedNode(cluster.nodes[0] || null);
    setActionFeedback(`Loaded ${cluster.name}`);
    setTimeout(() => setActionFeedback(null), 3500);
  };

  /* BEGIN GENERATED */
  // Real-time live ingestion stream simulation
  useEffect(() => {
    if (!isLiveStreaming) return;

    const interval = setInterval(() => {
      const orderNum = Math.floor(1000 + Math.random() * 9000);
      const isFraud = Math.random() > 0.35;
      const amount = Math.floor(120 + Math.random() * 850);
      const newOrderId = `ORD_STREAM_${orderNum}`;

      // Pick a buyer node
      const buyers = nodes.filter((n) => n.type === "buyer");
      const targetBuyer = buyers.length > 0 ? buyers[Math.floor(Math.random() * buyers.length)] : null;
      if (!targetBuyer) return;

      const newOrderNode: GraphNode = {
        id: newOrderId,
        type: "order",
        label: `${newOrderId} ($${amount})`,
        x: Math.max(120, Math.min(680, targetBuyer.x + (Math.random() * 140 - 70))),
        y: Math.max(80, Math.min(540, targetBuyer.y + (Math.random() * 120 - 60))),
        risk: isFraud ? 0.92 : 0.08,
        createdAtDaysAgo: 0,
        details: {
          connectedOrders: 1,
          fraudFlagged: isFraud,
          financialExposure: amount,
          clusterId: activeClusterId,
        },
      };

      const newEdge: GraphEdge = {
        id: `e_stream_${Date.now()}`,
        from: targetBuyer.id,
        to: newOrderId,
        label: "PLACES_ORDER",
        suspicious: isFraud,
        createdAtDaysAgo: 0,
      };

      setNodes((prev) => [...prev, newOrderNode]);
      setEdges((prev) => [...prev, newEdge]);
      setStreamCount((prev) => prev + 1);
      setSelectedNode(newOrderNode);
    }, 3800);

    return () => clearInterval(interval);
  }, [isLiveStreaming, nodes, activeClusterId]);

  // 2-Hop Dynamic Neighborhood Expansion
  const handleExpandNeighborhood = () => {
    if (!selectedNode) return;
    const expandId = `${selectedNode.id}_PROXY_${Math.floor(10 + Math.random() * 89)}`;
    const isDevice = selectedNode.type !== "device";

    const newNode: GraphNode = {
      id: expandId,
      type: isDevice ? "device" : "buyer",
      label: `${expandId}`,
      x: Math.max(100, Math.min(720, selectedNode.x + (Math.random() * 160 - 80))),
      y: Math.max(80, Math.min(520, selectedNode.y + (Math.random() * 140 - 70))),
      risk: Math.max(0.75, selectedNode.risk * 0.95),
      createdAtDaysAgo: 3,
      expanded: true,
      details: {
        connectedDevices: 3,
        ipHash: `10.244.${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}`,
        fraudFlagged: true,
        clusterId: activeClusterId,
      },
    };

    const newEdge: GraphEdge = {
      id: `e_expand_${Date.now()}`,
      from: selectedNode.id,
      to: expandId,
      label: isDevice ? "USES_DEVICE" : "CO_OCCURS_WITH",
      suspicious: true,
      createdAtDaysAgo: 3,
    };

    setNodes((prev) => [...prev, newNode]);
    setEdges((prev) => [...prev, newEdge]);
    setSelectedNode(newNode);
    setActionFeedback(`Discovered 2-Hop Entity: ${expandId}`);
    setTimeout(() => setActionFeedback(null), 4000);
  };
  /* END GENERATED */

  // Toggle Freeze in Graph Ledger
  const handleToggleFreeze = () => {
    if (!selectedNode) return;
    const newFrozenState = !selectedNode.frozen;

    setNodes((prev) =>
      prev.map((n) => (n.id === selectedNode.id ? { ...n, frozen: newFrozenState } : n))
    );

    setSelectedNode((prev) => (prev ? { ...prev, frozen: newFrozenState } : null));

    setActionFeedback(
      newFrozenState
        ? `Node ${selectedNode.id} FROZEN in Graph Ledger. Payouts and sessions revoked.`
        : `Node ${selectedNode.id} unfrozen and restored to active routing.`
    );
    setTimeout(() => setActionFeedback(null), 4500);
  };

  // Filtered nodes based on Temporal Window, Risk Threshold, and Search Query
  const filteredNodes = useMemo(() => {
    return nodes.filter((n) => {
      // Temporal Cutoff: if entity was created more recently than the cutoff lookback window
      if (timelineCutoffDays > 0 && n.createdAtDaysAgo > timelineCutoffDays) {
        return false;
      }
      if (filterSuspicious && n.risk < 0.5) return false;
      if (searchQuery && !n.label.toLowerCase().includes(searchQuery.toLowerCase())) return false;
      return true;
    });
  }, [nodes, timelineCutoffDays, filterSuspicious, searchQuery]);

  // Edges connecting visible nodes
  const visibleEdges = useMemo(() => {
    const visibleIds = new Set(filteredNodes.map((n) => n.id));
    return edges.filter((e) => {
      if (!visibleIds.has(e.from) || !visibleIds.has(e.to)) return false;
      if (timelineCutoffDays > 0 && e.createdAtDaysAgo > timelineCutoffDays) return false;
      return true;
    });
  }, [edges, filteredNodes, timelineCutoffDays]);

  const activeCluster = useMemo(
    () => CLUSTERS.find((c) => c.id === activeClusterId) || CLUSTERS[0],
    [activeClusterId]
  );

  const NODE_CONFIG: Record<string, { bg: string; border: string; icon: LucideIcon }> = {
    buyer: { bg: "#3B82F6", border: "#60A5FA", icon: User },
    seller: { bg: "#8B5CF6", border: "#A78BFA", icon: Store },
    device: { bg: "#06B6D4", border: "#22D3EE", icon: Smartphone },
    address: { bg: "#F59E0B", border: "#FBBF24", icon: MapPin },
    order: { bg: "#10B981", border: "#34D399", icon: ShoppingBag },
  };
  const getNodeColor = (type: string) => NODE_CONFIG[type] || { bg: "#64748B", border: "#94A3B8", icon: Info };

  return (
    <div className="space-y-6">
      {/* BEGIN GENERATED */}
      {/* Header & Topology Controls */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2.5">
            <h1 className="text-xl font-bold tracking-tight text-[#E8EDF3] flex items-center gap-2">
              <Share2 className="w-5 h-5 text-blue-400" />
              <span>Trust Graph Explorer & Multi-Relational Canvas</span>
            </h1>
            <span
              className={`text-[10px] font-mono px-2 py-0.5 rounded uppercase font-semibold ${
                activeCluster.riskLevel === "high"
                  ? "bg-rose-500/10 text-rose-400 border border-rose-500/20"
                  : "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
              }`}
            >
              {activeCluster.tag}
            </span>
          </div>
          <p className="text-xs text-[#8995A3] mt-1">
            2-hop heterogeneous graph traversal mapping bipartite buyers, merchants, shared devices, and delivery destinations.
          </p>
        </div>

        {/* Legend */}
        <div className="flex items-center gap-3 text-[11px] font-mono bg-[#111821] px-3 py-1.5 rounded-lg border border-[#202A35]">
          {(["Buyer:bg-blue-500", "Seller:bg-purple-500", "Device:bg-cyan-500", "Address:bg-amber-500", "Order:bg-emerald-500"] as const).map((item) => {
            const [lbl, bg] = item.split(":");
            return <span key={lbl} className="flex items-center gap-1.5"><span className={`w-2.5 h-2.5 rounded-full ${bg}`} /> {lbl}</span>;
          })}
        </div>
      </div>

      {/* Cluster Switcher & Live Stream Controls */}
      <div className="p-3 rounded-xl border border-[#202A35] bg-[#111821] flex flex-wrap items-center justify-between gap-3">
        {/* Cluster Tabs */}
        <div className="flex items-center gap-1.5 overflow-x-auto pb-1 sm:pb-0">
          <span className="text-xs font-mono text-[#8995A3] mr-1 hidden sm:inline">Cluster:</span>
          {CLUSTERS.map((cluster) => {
            const isSelected = cluster.id === activeClusterId;
            return (
              <button
                key={cluster.id}
                onClick={() => handleSelectCluster(cluster.id)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all shrink-0 flex items-center gap-1.5 ${
                  isSelected
                    ? "bg-blue-600/20 border border-blue-500/50 text-blue-300 font-semibold shadow-sm shadow-blue-500/20"
                    : "bg-[#0A0E13] border border-[#202A35] text-[#8995A3] hover:text-[#E8EDF3] hover:bg-[#151D27]"
                }`}
              >
                <Layers className="w-3.5 h-3.5" />
                <span>{cluster.name.split(":")[0]}</span>
                <span
                  className={`text-[9px] px-1 py-0.2 rounded font-mono ${
                    cluster.riskScore > 0.5 ? "bg-rose-500/20 text-rose-300" : "bg-emerald-500/20 text-emerald-300"
                  }`}
                >
                  {(cluster.riskScore * 100).toFixed(0)}%
                </span>
              </button>
            );
          })}
        </div>

        {/* Live Ingestion Stream Toggle */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setIsLiveStreaming(!isLiveStreaming)}
            className={`px-3 py-1.5 rounded-lg text-xs font-mono font-medium border flex items-center gap-2 transition-all ${
              isLiveStreaming
                ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/50 shadow-sm shadow-emerald-500/20 animate-pulse"
                : "bg-[#0A0E13] text-[#8995A3] border-[#202A35] hover:text-[#E8EDF3]"
            }`}
          >
            <Radio className={`w-3.5 h-3.5 ${isLiveStreaming ? "text-emerald-400 animate-spin" : "text-[#596574]"}`} />
            <span>{isLiveStreaming ? `Live Stream Active (${streamCount} added)` : "Live Ingestion Stream"}</span>
          </button>
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

            {/* Dynamic Temporal Slider */}
            <div className="flex items-center gap-2 font-mono text-[11px] text-[#8995A3]">
              <Clock className="w-3.5 h-3.5 text-blue-400" />
              <span>Cutoff: T - {timelineCutoffDays}d</span>
              <input
                type="range"
                min="1"
                max="90"
                value={timelineCutoffDays}
                onChange={(e) => setTimelineCutoffDays(parseInt(e.target.value))}
                className="w-24 accent-blue-500"
              />
              <span className="text-[10px] text-[#596574]">
                ({filteredNodes.length}/{nodes.length} nodes)
              </span>
            </div>
          </div>

          {/* Action Feedback Banner */}
          {actionFeedback && (
            <div className="absolute top-14 left-4 z-20 px-3 py-1.5 rounded-md bg-blue-500/20 border border-blue-500/40 text-blue-300 text-xs font-mono flex items-center gap-2 shadow-lg backdrop-blur-md animate-in fade-in">
              <CheckCircle2 className="w-3.5 h-3.5" />
              <span>{actionFeedback}</span>
            </div>
          )}

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
              {visibleEdges.map((edge) => {
                const source = filteredNodes.find((n) => n.id === edge.from);
                const target = filteredNodes.find((n) => n.id === edge.to);
                if (!source || !target) return null;
                const isSelected = selectedNode?.id === source.id || selectedNode?.id === target.id;
                return (
                  <g key={edge.id}>
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
                const isFrozen = !!node.frozen;

                return (
                  <g
                    key={node.id}
                    transform={`translate(${node.x}, ${node.y})`}
                    onClick={() => setSelectedNode(node)}
                    className="cursor-pointer transition-transform hover:scale-110"
                  >
                    {/* Pulsing Alert Ring for High Risk */}
                    {node.risk > 0.7 && !isFrozen && (
                      <circle
                        r={isSelected ? 34 : 28}
                        fill="none"
                        stroke="#EF4444"
                        strokeWidth="1.5"
                        strokeDasharray="3,3"
                        opacity="0.8"
                        className="animate-spin"
                        style={{ animationDuration: "12s" }}
                      />
                    )}

                    {/* Frozen Icy Halo */}
                    {isFrozen && (
                      <circle
                        r={isSelected ? 34 : 28}
                        fill="none"
                        stroke="#38BDF8"
                        strokeWidth="2"
                        strokeDasharray="2,2"
                        opacity="0.9"
                      />
                    )}

                    {/* Base Node Circle */}
                    <circle
                      r={isSelected ? 26 : 20}
                      fill={isFrozen ? "#0369A1" : config.bg}
                      stroke={isSelected ? "#FFFFFF" : isFrozen ? "#38BDF8" : config.border}
                      strokeWidth={isSelected ? 3 : 1.5}
                      className={node.risk > 0.7 && !isFrozen ? "animate-pulse" : ""}
                    />

                    {/* Node Text Label */}
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

                    {/* Frozen Badge Icon */}
                    {isFrozen && (
                      <text
                        y="4"
                        textAnchor="middle"
                        fill="#FFFFFF"
                        fontSize="12"
                        fontFamily="monospace"
                        fontWeight="bold"
                      >
                        ❄
                      </text>
                    )}
                  </g>
                );
              })}
            </svg>

            {/* Bottom Invariant Banner */}
            <div className="absolute bottom-3 left-3 px-3 py-1.5 rounded-lg bg-[#0A0E13]/90 border border-[#202A35] text-[10px] font-mono text-[#8995A3] backdrop-blur-sm flex items-center gap-3">
              <span>&Delta;t &ge; 0 temporal directionality enforced · Zero retrospective snooping</span>
              <span className="text-emerald-400 font-bold">● Active Cluster: {activeCluster.tag}</span>
            </div>
          </div>
        </div>

        {/* Right Col: Entity Inspector Drawer */}
        <div className="rounded-xl border border-[#202A35] bg-[#111821] p-5 space-y-4">
          <div className="flex items-center justify-between border-b border-[#202A35] pb-3">
            <div className="flex items-center gap-2">
              <span className="text-xs font-bold uppercase tracking-wider text-[#E8EDF3]">
                Entity Inspector
              </span>
              {viewMode === "executive" ? (
                <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-400">Story View</span>
              ) : (
                <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-purple-500/10 text-purple-400">ML Forensic</span>
              )}
            </div>
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
                {selectedNode.frozen && (
                  <span className="inline-flex items-center gap-1 mt-1 px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 text-[10px] font-bold">
                    <Lock className="w-3 h-3" />
                    FROZEN IN GRAPH LEDGER
                  </span>
                )}
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

              {/* Story View (Executive ROI Mode) */}
              {viewMode === "executive" ? (
                <div className="p-3 rounded-lg bg-[#0E131A] border border-[#202A35] space-y-2 text-xs">
                  <div className="text-[10px] text-[#8995A3] uppercase font-bold">Business Exposure & Narrative</div>
                  <div className="text-[#E8EDF3] leading-relaxed text-[11px] font-sans">
                    {selectedNode.risk > 0.7
                      ? `High risk entity associated with ${activeCluster.name}. Immediate risk of payment chargebacks and synthetic account abuse.`
                      : "Verified baseline consumer entity with established clean purchase history and zero suspicious ties."}
                  </div>
                  {selectedNode.details.financialExposure !== undefined && (
                    <div className="flex justify-between pt-1 border-t border-[#202A35]/60 text-[11px]">
                      <span className="text-[#8995A3]">Exposure Value:</span>
                      <span className="text-rose-400 font-bold">${selectedNode.details.financialExposure.toLocaleString("en-US")}</span>
                    </div>
                  )}
                </div>
              ) : (
                /* AI Inspector (Deep ML Mode) */
                <div className="p-3 rounded-lg bg-[#0E131A] border border-[#202A35] space-y-2 text-[10px]">
                  <div className="text-[#8995A3] uppercase font-bold">GraphSAGE Embedding Vector (16-D)</div>
                  <div className="font-mono text-purple-400 bg-[#090D12] p-1.5 rounded border border-[#202A35]/60">
                    {selectedNode.details.gnnEmbeddingSample
                      ? `[${selectedNode.details.gnnEmbeddingSample.join(", ")}, ...]`
                      : "[0.812, -0.342, 0.491, 0.119, ...]"}
                  </div>
                  <div className="flex justify-between pt-1 text-[11px]">
                    <span className="text-[#8995A3]">Centrality / Degree:</span>
                    <span className="text-emerald-400">{selectedNode.details.degree ?? 3} edges</span>
                  </div>
                </div>
              )}

              {/* Entity Attributes */}
              <div className="space-y-2 text-[11px] pt-1">
                {(
                  [
                    selectedNode.details.ageDays !== undefined ? { l: "Account Age:", v: `${selectedNode.details.ageDays} days` } : null,
                    selectedNode.details.connectedOrders !== undefined ? { l: "Connected Orders:", v: selectedNode.details.connectedOrders } : null,
                    selectedNode.details.connectedDevices !== undefined ? { l: "Device Collisions:", v: selectedNode.details.connectedDevices, c: "text-rose-400 font-semibold" } : null,
                    selectedNode.details.ipHash ? { l: "Hardware IP:", v: selectedNode.details.ipHash } : null,
                    selectedNode.details.addressLine ? { l: "Shipping Dest:", v: selectedNode.details.addressLine } : null,
                    { l: "Collusion Ring:", v: selectedNode.details.fraudFlagged ? activeCluster.id : "None", c: selectedNode.details.fraudFlagged ? "text-rose-400 font-bold" : "text-emerald-400" },
                  ] as ({ l: string; v: string | number; c?: string } | null)[]
                ).filter((item): item is { l: string; v: string | number; c?: string } => item !== null).map((item, idx, arr) => (
                  <div key={item.l} className={`flex justify-between py-1 ${idx < arr.length - 1 ? "border-b border-[#202A35]/60" : ""}`}>
                    <span className="text-[#8995A3]">{item.l}</span>
                    <span className={item.c || "text-[#E8EDF3]"}>{item.v}</span>
                  </div>
                ))}
              </div>


              {/* Interactive Operations Buttons */}
              <div className="space-y-2 pt-2">
                {/* 2-Hop Expand Neighborhood */}
                <button
                  onClick={handleExpandNeighborhood}
                  className="w-full py-2 px-3 rounded-lg bg-blue-600/20 hover:bg-blue-600/30 border border-blue-500/40 text-blue-300 font-semibold text-xs transition-colors flex items-center justify-center gap-1.5"
                >
                  <PlusCircle className="w-3.5 h-3.5" />
                  <span>Expand 2-Hop Neighborhood</span>
                </button>

                {/* Freeze / Unfreeze Node Button */}
                <button
                  onClick={handleToggleFreeze}
                  className={`w-full py-2 px-3 rounded-lg font-semibold text-xs transition-colors flex items-center justify-center gap-1.5 border ${
                    selectedNode.frozen
                      ? "bg-cyan-600/20 hover:bg-cyan-600/30 border-cyan-500/40 text-cyan-300"
                      : "bg-rose-600/20 hover:bg-rose-600/30 border-rose-500/30 text-rose-300"
                  }`}
                >
                  {selectedNode.frozen ? (
                    <>
                      <Unlock className="w-3.5 h-3.5" />
                      <span>Unfreeze Node in Ledger</span>
                    </>
                  ) : (
                    <>
                      <Lock className="w-3.5 h-3.5" />
                      <span>Freeze Node in Graph Ledger</span>
                    </>
                  )}
                </button>
              </div>
            </div>
          ) : (
            <div className="py-12 text-center text-[#596574] text-xs">
              Click any node in the canvas to inspect its relational attributes.
            </div>
          )}
        </div>
      </div>
      {/* END GENERATED */}
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
