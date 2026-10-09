"use client";

import React, { useState, useEffect, useMemo, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { useViewMode } from "@/context/ViewModeContext";
import {
  Share2,
  Search,
  Filter,
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
  Sparkles,
  Microscope,
  CheckCircle2,
  AlertTriangle,
  RotateCcw,
  PlusCircle,
  Eye,
  Sliders,
  ShieldCheck,
  ShieldAlert,
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

const CLUSTERS: ClusterDefinition[] = [
  {
    id: "RING_COLLUSION_01",
    name: "Ring #01: Device Farm & Collusion",
    tag: "Device Farm",
    riskScore: 0.94,
    riskLevel: "high",
    description: "2-hop bipartite graph mapping shared device collision and synthetic buyer accounts.",
    executiveSummary: "Active botnet farm recycling Android hardware IDs to orchestrate 42 coordinated checkout attempts totaling $38,400 in potential chargeback exposure.",
    nodes: [
      {
        id: "BUYER_RING_MEMBER_04",
        type: "buyer",
        label: "BUYER_RING_MEMBER_04",
        x: 280,
        y: 160,
        risk: 0.94,
        createdAtDaysAgo: 4,
        details: {
          ageDays: 4,
          connectedOrders: 9,
          connectedDevices: 3,
          fraudFlagged: true,
          clusterId: "RING_COLLUSION_01",
          financialExposure: 8450,
          gnnEmbeddingSample: [0.891, -0.421, 0.763, 0.129],
          pageRank: 0.0412,
          degree: 4,
        },
      },
      {
        id: "BUYER_RING_MEMBER_02",
        type: "buyer",
        label: "BUYER_RING_MEMBER_02",
        x: 220,
        y: 320,
        risk: 0.91,
        createdAtDaysAgo: 2,
        details: {
          ageDays: 2,
          connectedOrders: 6,
          connectedDevices: 2,
          fraudFlagged: true,
          clusterId: "RING_COLLUSION_01",
          financialExposure: 5200,
          gnnEmbeddingSample: [0.884, -0.395, 0.741, 0.118],
          pageRank: 0.0354,
          degree: 3,
        },
      },
      {
        id: "BUYER_VERIFIED_77",
        type: "buyer",
        label: "BUYER_VERIFIED_77",
        x: 680,
        y: 220,
        risk: 0.04,
        createdAtDaysAgo: 180,
        details: {
          ageDays: 180,
          connectedOrders: 15,
          connectedDevices: 1,
          fraudFlagged: false,
          clusterId: "BENIGN_NETWORK",
          financialExposure: 0,
          gnnEmbeddingSample: [-0.621, 0.142, -0.312, 0.041],
          pageRank: 0.0098,
          degree: 1,
        },
      },
      {
        id: "DEV_FARM_99",
        type: "device",
        label: "DEV_FARM_99 (Emulated)",
        x: 380,
        y: 240,
        risk: 0.98,
        createdAtDaysAgo: 6,
        details: {
          connectedDevices: 9,
          ipHash: "192.168.1.104",
          fraudFlagged: true,
          clusterId: "RING_COLLUSION_01",
          financialExposure: 18900,
          gnnEmbeddingSample: [0.952, -0.812, 0.915, 0.442],
          pageRank: 0.0891,
          degree: 5,
        },
      },
      {
        id: "ADDR_WH_01",
        type: "address",
        label: "ADDR_WH_01 (Drop Ship Hub)",
        x: 290,
        y: 420,
        risk: 0.89,
        createdAtDaysAgo: 25,
        details: {
          addressLine: "Suite 4B, Industrial Blvd",
          fraudFlagged: true,
          clusterId: "RING_COLLUSION_01",
          financialExposure: 14200,
          gnnEmbeddingSample: [0.712, -0.219, 0.589, 0.301],
          pageRank: 0.0385,
          degree: 2,
        },
      },
      {
        id: "SELLER_RING_LEADER_01",
        type: "seller",
        label: "SELLER_RING_LEADER_01",
        x: 490,
        y: 360,
        risk: 0.92,
        createdAtDaysAgo: 12,
        details: {
          ageDays: 12,
          connectedOrders: 63,
          fraudFlagged: true,
          clusterId: "RING_COLLUSION_01",
          financialExposure: 38400,
          gnnEmbeddingSample: [0.901, -0.582, 0.812, 0.399],
          pageRank: 0.0712,
          degree: 4,
        },
      },
      {
        id: "SELLER_REPUTABLE_12",
        type: "seller",
        label: "SELLER_REPUTABLE_12",
        x: 740,
        y: 380,
        risk: 0.03,
        createdAtDaysAgo: 420,
        details: {
          ageDays: 420,
          connectedOrders: 412,
          fraudFlagged: false,
          clusterId: "BENIGN_NETWORK",
          financialExposure: 0,
          gnnEmbeddingSample: [-0.781, 0.312, -0.512, -0.108],
          pageRank: 0.0152,
          degree: 1,
        },
      },
      {
        id: "ORD_SIM_FARM_03",
        type: "order",
        label: "ORD_SIM_FARM_03 ($890)",
        x: 420,
        y: 110,
        risk: 0.95,
        createdAtDaysAgo: 1,
        details: {
          connectedOrders: 1,
          fraudFlagged: true,
          clusterId: "RING_COLLUSION_01",
          financialExposure: 890,
          gnnEmbeddingSample: [0.841, -0.612, 0.771, 0.291],
          pageRank: 0.0125,
          degree: 1,
        },
      },
    ],
    edges: [
      { id: "e1", from: "BUYER_RING_MEMBER_04", to: "DEV_FARM_99", label: "USES_DEVICE", suspicious: true, createdAtDaysAgo: 4 },
      { id: "e2", from: "BUYER_RING_MEMBER_02", to: "DEV_FARM_99", label: "USES_DEVICE", suspicious: true, createdAtDaysAgo: 2 },
      { id: "e3", from: "BUYER_RING_MEMBER_04", to: "ADDR_WH_01", label: "SHIPS_TO", suspicious: true, createdAtDaysAgo: 4 },
      { id: "e4", from: "BUYER_RING_MEMBER_02", to: "ADDR_WH_01", label: "SHIPS_TO", suspicious: true, createdAtDaysAgo: 2 },
      { id: "e5", from: "BUYER_RING_MEMBER_04", to: "SELLER_RING_LEADER_01", label: "TRANSACTS_WITH", suspicious: true, createdAtDaysAgo: 3 },
      { id: "e6", from: "BUYER_RING_MEMBER_04", to: "ORD_SIM_FARM_03", label: "PLACES_ORDER", suspicious: true, createdAtDaysAgo: 1 },
      { id: "e7", from: "BUYER_VERIFIED_77", to: "SELLER_REPUTABLE_12", label: "TRANSACTS_WITH", suspicious: false, createdAtDaysAgo: 30 },
    ],
  },
  {
    id: "RING_PRICE_ARBITRAGE_02",
    name: "Ring #02: Price Arbitrage & Freight Hub",
    tag: "Price Arbitrage",
    riskScore: 0.85,
    riskLevel: "high",
    description: "Syndicate exploiting discounted promotional codes and shipping to commercial freight forwarders.",
    executiveSummary: "Coordinated buyer cluster purchasing high-demand luxury electronics using unauthorized coupons and shipping to an unvetted freight forwarder terminal.",
    nodes: [
      {
        id: "BUYER_ARBITRAGE_112",
        type: "buyer",
        label: "BUYER_ARBITRAGE_112",
        x: 250,
        y: 180,
        risk: 0.88,
        createdAtDaysAgo: 8,
        details: { ageDays: 8, connectedOrders: 14, connectedDevices: 2, fraudFlagged: true, financialExposure: 12400 },
      },
      {
        id: "BUYER_ARBITRAGE_114",
        type: "buyer",
        label: "BUYER_ARBITRAGE_114",
        x: 320,
        y: 340,
        risk: 0.82,
        createdAtDaysAgo: 5,
        details: { ageDays: 5, connectedOrders: 8, connectedDevices: 2, fraudFlagged: true, financialExposure: 7800 },
      },
      {
        id: "DEV_ROTATING_PROXY_11",
        type: "device",
        label: "DEV_ROTATING_PROXY_11",
        x: 420,
        y: 220,
        risk: 0.93,
        createdAtDaysAgo: 4,
        details: { connectedDevices: 8, ipHash: "104.28.19.44", fraudFlagged: true, financialExposure: 20200 },
      },
      {
        id: "ADDR_PORT_TERMINAL_7",
        type: "address",
        label: "ADDR_PORT_TERMINAL_7",
        x: 520,
        y: 380,
        risk: 0.91,
        createdAtDaysAgo: 45,
        details: { addressLine: "Dock 14, Bay Forwarding Corp", fraudFlagged: true, financialExposure: 24500 },
      },
      {
        id: "SELLER_MERCHANT_44",
        type: "seller",
        label: "SELLER_MERCHANT_44",
        x: 620,
        y: 190,
        risk: 0.86,
        createdAtDaysAgo: 18,
        details: { ageDays: 18, connectedOrders: 34, fraudFlagged: true, financialExposure: 18500 },
      },
      {
        id: "ORD_LUXURY_BULK_09",
        type: "order",
        label: "ORD_LUXURY_BULK_09 ($4,200)",
        x: 350,
        y: 100,
        risk: 0.89,
        createdAtDaysAgo: 2,
        details: { connectedOrders: 1, fraudFlagged: true, financialExposure: 4200 },
      },
    ],
    edges: [
      { id: "e201", from: "BUYER_ARBITRAGE_112", to: "DEV_ROTATING_PROXY_11", label: "USES_DEVICE", suspicious: true, createdAtDaysAgo: 8 },
      { id: "e202", from: "BUYER_ARBITRAGE_114", to: "DEV_ROTATING_PROXY_11", label: "USES_DEVICE", suspicious: true, createdAtDaysAgo: 5 },
      { id: "e203", from: "BUYER_ARBITRAGE_112", to: "ADDR_PORT_TERMINAL_7", label: "SHIPS_TO", suspicious: true, createdAtDaysAgo: 8 },
      { id: "e204", from: "BUYER_ARBITRAGE_114", to: "ADDR_PORT_TERMINAL_7", label: "SHIPS_TO", suspicious: true, createdAtDaysAgo: 5 },
      { id: "e205", from: "BUYER_ARBITRAGE_112", to: "SELLER_MERCHANT_44", label: "TRANSACTS_WITH", suspicious: true, createdAtDaysAgo: 6 },
      { id: "e206", from: "BUYER_ARBITRAGE_112", to: "ORD_LUXURY_BULK_09", label: "PLACES_ORDER", suspicious: true, createdAtDaysAgo: 2 },
    ],
  },
  {
    id: "RING_RETURN_ABUSE_03",
    name: "Ring #03: Rapid Refund & Wardrobing Syndicate",
    tag: "Return Abuse",
    riskScore: 0.78,
    riskLevel: "high",
    description: "Repeated empty-box claims and high-velocity wardrobing returns against luxury fashion apparel.",
    executiveSummary: "Syndicate exploiting return guarantees with rapid chargebacks within 24 hours of delivery, sharing burner phone emulators.",
    nodes: [
      {
        id: "BUYER_REFUND_01",
        type: "buyer",
        label: "BUYER_REFUND_01 (7 Returns)",
        x: 260,
        y: 200,
        risk: 0.89,
        createdAtDaysAgo: 35,
        details: { ageDays: 35, connectedOrders: 9, fraudFlagged: true, financialExposure: 9400 },
      },
      {
        id: "BUYER_REFUND_02",
        type: "buyer",
        label: "BUYER_REFUND_02 (5 Returns)",
        x: 320,
        y: 360,
        risk: 0.84,
        createdAtDaysAgo: 28,
        details: { ageDays: 28, connectedOrders: 6, fraudFlagged: true, financialExposure: 6100 },
      },
      {
        id: "DEV_MOBILE_EMULATOR_04",
        type: "device",
        label: "DEV_MOBILE_EMULATOR_04",
        x: 450,
        y: 260,
        risk: 0.94,
        createdAtDaysAgo: 14,
        details: { connectedDevices: 4, ipHash: "172.56.21.90", fraudFlagged: true, financialExposure: 15500 },
      },
      {
        id: "SELLER_FASHION_BOUTIQUE_88",
        type: "seller",
        label: "SELLER_FASHION_BOUTIQUE_88",
        x: 620,
        y: 260,
        risk: 0.12,
        createdAtDaysAgo: 320,
        details: { ageDays: 320, connectedOrders: 890, fraudFlagged: false, financialExposure: 0 },
      },
      {
        id: "ORD_DESIGNER_RETURN_42",
        type: "order",
        label: "ORD_DESIGNER_RETURN_42 ($1,450)",
        x: 380,
        y: 120,
        risk: 0.88,
        createdAtDaysAgo: 4,
        details: { connectedOrders: 1, fraudFlagged: true, financialExposure: 1450 },
      },
    ],
    edges: [
      { id: "e301", from: "BUYER_REFUND_01", to: "DEV_MOBILE_EMULATOR_04", label: "USES_DEVICE", suspicious: true, createdAtDaysAgo: 35 },
      { id: "e302", from: "BUYER_REFUND_02", to: "DEV_MOBILE_EMULATOR_04", label: "USES_DEVICE", suspicious: true, createdAtDaysAgo: 28 },
      { id: "e303", from: "BUYER_REFUND_01", to: "SELLER_FASHION_BOUTIQUE_88", label: "TRANSACTS_WITH", suspicious: true, createdAtDaysAgo: 10 },
      { id: "e304", from: "BUYER_REFUND_02", to: "SELLER_FASHION_BOUTIQUE_88", label: "TRANSACTS_WITH", suspicious: true, createdAtDaysAgo: 8 },
      { id: "e305", from: "BUYER_REFUND_01", to: "ORD_DESIGNER_RETURN_42", label: "PLACES_ORDER", suspicious: true, createdAtDaysAgo: 4 },
    ],
  },
  {
    id: "RING_BENIGN_CLUSTER_06",
    name: "Baseline: Verified Legitimate Traffic",
    tag: "Benign Network",
    riskScore: 0.04,
    riskLevel: "low",
    description: "Established consumers with years of verified transaction history, distinct devices, and low return rates.",
    executiveSummary: "Clean, organic e-commerce purchases from high-tenure accounts with zero hardware collisions and trusted payment methods.",
    nodes: [
      {
        id: "BUYER_VERIFIED_ALICE",
        type: "buyer",
        label: "BUYER_VERIFIED_ALICE",
        x: 280,
        y: 220,
        risk: 0.02,
        createdAtDaysAgo: 450,
        details: { ageDays: 450, connectedOrders: 38, connectedDevices: 1, fraudFlagged: false, financialExposure: 0 },
      },
      {
        id: "DEV_ALICE_IPHONE_15",
        type: "device",
        label: "DEV_ALICE_IPHONE_15",
        x: 420,
        y: 160,
        risk: 0.01,
        createdAtDaysAgo: 450,
        details: { connectedDevices: 1, ipHash: "73.189.44.12", fraudFlagged: false, financialExposure: 0 },
      },
      {
        id: "ADDR_HOME_VERIFIED",
        type: "address",
        label: "ADDR_HOME_VERIFIED",
        x: 420,
        y: 320,
        risk: 0.02,
        createdAtDaysAgo: 450,
        details: { addressLine: "742 Evergreen Terrace", fraudFlagged: false, financialExposure: 0 },
      },
      {
        id: "SELLER_APPLE_AUTHORIZED",
        type: "seller",
        label: "SELLER_APPLE_AUTHORIZED",
        x: 620,
        y: 240,
        risk: 0.01,
        createdAtDaysAgo: 1200,
        details: { ageDays: 1200, connectedOrders: 5400, fraudFlagged: false, financialExposure: 0 },
      },
      {
        id: "ORD_AIRPODS_CLEAN",
        type: "order",
        label: "ORD_AIRPODS_CLEAN ($249)",
        x: 350,
        y: 100,
        risk: 0.02,
        createdAtDaysAgo: 10,
        details: { connectedOrders: 1, fraudFlagged: false, financialExposure: 0 },
      },
    ],
    edges: [
      { id: "e601", from: "BUYER_VERIFIED_ALICE", to: "DEV_ALICE_IPHONE_15", label: "USES_DEVICE", suspicious: false, createdAtDaysAgo: 450 },
      { id: "e602", from: "BUYER_VERIFIED_ALICE", to: "ADDR_HOME_VERIFIED", label: "SHIPS_TO", suspicious: false, createdAtDaysAgo: 450 },
      { id: "e603", from: "BUYER_VERIFIED_ALICE", to: "SELLER_APPLE_AUTHORIZED", label: "TRANSACTS_WITH", suspicious: false, createdAtDaysAgo: 10 },
      { id: "e604", from: "BUYER_VERIFIED_ALICE", to: "ORD_AIRPODS_CLEAN", label: "PLACES_ORDER", suspicious: false, createdAtDaysAgo: 10 },
    ],
  },
];

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
          <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-blue-500" /> Buyer</span>
          <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-purple-500" /> Seller</span>
          <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-cyan-500" /> Device</span>
          <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-amber-500" /> Address</span>
          <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-emerald-500" /> Order</span>
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
                    {selectedNode.details.fraudFlagged ? activeCluster.id : "None"}
                  </span>
                </div>
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
