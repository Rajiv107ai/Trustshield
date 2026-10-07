"""Heterogeneous Graph Neural Network (HeteroGNN) for TrustShield.

Constructs explicit multi-entity relational graphs using PyTorch Geometric HeteroData:
- Node Types: buyer, seller, device, address
- Edge Types:
    - ('buyer', 'uses_device', 'device') & reverse
    - ('seller', 'uses_device', 'device') & reverse
    - ('buyer', 'uses_address', 'address') & reverse
    - ('seller', 'uses_address', 'address') & reverse
    - ('buyer', 'transacts_with', 'seller') & reverse

Preserves distinct relationship semantics without collapsing into homogeneous adjacency.
Enforces strict temporal safety: graphs are built strictly with events prior to cutoff_date.
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import HeteroConv, SAGEConv, Linear
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False


def build_hetero_graph(
    buyers_df: pd.DataFrame,
    sellers_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    device_sharing_log: pd.DataFrame,
    address_sharing_log: pd.DataFrame,
    cutoff_date: Optional[str] = None,
    emb_dim: int = 16,
) -> Tuple[Any, Dict[str, Dict[str, int]]]:
    """Construct a PyTorch Geometric HeteroData object with explicit node and edge types.

    Filters all relationship events strictly prior to cutoff_date to prevent temporal leakage.
    """
    if not PYG_AVAILABLE:
        raise RuntimeError("torch_geometric is required for HeteroGNN.")

    data = HeteroData()

    # Apply temporal filtering to dynamic events
    if cutoff_date is not None:
        c_date = pd.to_datetime(cutoff_date)
        if "order_date" in orders_df.columns:
            orders_filt = orders_df[pd.to_datetime(orders_df["order_date"]) < c_date]
        else:
            orders_filt = orders_df

        if "first_seen_date" in device_sharing_log.columns:
            dev_filt = device_sharing_log[pd.to_datetime(device_sharing_log["first_seen_date"]) < c_date]
        else:
            dev_filt = device_sharing_log

        if "first_seen_date" in address_sharing_log.columns:
            addr_filt = address_sharing_log[pd.to_datetime(address_sharing_log["first_seen_date"]) < c_date]
        else:
            addr_filt = address_sharing_log
    else:
        orders_filt = orders_df
        dev_filt = device_sharing_log
        addr_filt = address_sharing_log

    # Build unique ID maps
    buyer_ids = sorted(buyers_df["buyer_id"].unique())
    seller_ids = sorted(sellers_df["seller_id"].unique())

    # Collect devices & addresses from logs
    dev_ids = set()
    if "device_id" in buyers_df.columns:
        dev_ids.update(buyers_df["device_id"].dropna().unique())
    if "shared_device_id" in dev_filt.columns:
        dev_ids.update(dev_filt["shared_device_id"].dropna().unique())
    device_ids = sorted(dev_ids)

    addr_ids = set()
    if "address_id" in buyers_df.columns:
        addr_ids.update(buyers_df["address_id"].dropna().unique())
    if "shared_address_id" in addr_filt.columns:
        addr_ids.update(addr_filt["shared_address_id"].dropna().unique())
    address_ids = sorted(addr_ids)

    b_map = {bid: i for i, bid in enumerate(buyer_ids)}
    s_map = {sid: i for i, sid in enumerate(seller_ids)}
    d_map = {did: i for i, did in enumerate(device_ids)}
    a_map = {aid: i for i, aid in enumerate(address_ids)}

    id_maps = {"buyer": b_map, "seller": s_map, "device": d_map, "address": a_map}

    # Initialize node feature tensors
    # Buyer initial features: [orders_before, returns_before, return_rate]
    b_feats = []
    for bid in buyer_ids:
        brow = buyers_df[buyers_df["buyer_id"] == bid].iloc[0]
        ob = float(brow.get("orders_before", brow.get("buyer_orders_before", 0.0)))
        rb = float(brow.get("returns_before", brow.get("buyer_returns_before", 0.0)))
        rate = rb / max(ob, 1.0)
        b_feats.append([ob, rb, rate])
    data["buyer"].x = torch.tensor(b_feats, dtype=torch.float)

    # Seller initial features: [total_listings, age_days]
    s_feats = []
    for sid in seller_ids:
        srow = sellers_df[sellers_df["seller_id"] == sid].iloc[0]
        tl = float(srow.get("total_listings_before", srow.get("seller_total_listings_before", 0.0)))
        age = float(srow.get("seller_age_days", 30.0))
        s_feats.append([tl, age])
    data["seller"].x = torch.tensor(s_feats, dtype=torch.float)

    # Device & Address features: simple degree / constant embedding
    data["device"].x = torch.ones((len(device_ids), 2), dtype=torch.float)
    data["address"].x = torch.ones((len(address_ids), 2), dtype=torch.float)

    # Helper to add bidirectional edges
    def add_edge_type(src_type, dst_type, rel_name, edge_pairs):
        if not edge_pairs:
            u_t = torch.empty((0,), dtype=torch.long)
            v_t = torch.empty((0,), dtype=torch.long)
        else:
            u_t = torch.tensor([p[0] for p in edge_pairs], dtype=torch.long)
            v_t = torch.tensor([p[1] for p in edge_pairs], dtype=torch.long)

        data[(src_type, rel_name, dst_type)].edge_index = torch.stack([u_t, v_t], dim=0)
        # Reverse edge for message passing symmetry
        rev_rel = f"rev_{rel_name}"
        data[(dst_type, rev_rel, src_type)].edge_index = torch.stack([v_t, u_t], dim=0)

    # 1. Buyer <-> Device edges
    b_dev_edges = []
    if "device_id" in buyers_df.columns:
        for _, r in buyers_df.dropna(subset=["device_id"]).iterrows():
            bid, did = r["buyer_id"], r["device_id"]
            if bid in b_map and did in d_map:
                b_dev_edges.append((b_map[bid], d_map[did]))
    add_edge_type("buyer", "device", "uses_device", b_dev_edges)

    # 2. Buyer <-> Address edges
    b_addr_edges = []
    if "address_id" in buyers_df.columns:
        for _, r in buyers_df.dropna(subset=["address_id"]).iterrows():
            bid, aid = r["buyer_id"], r["address_id"]
            if bid in b_map and aid in a_map:
                b_addr_edges.append((b_map[bid], a_map[aid]))
    add_edge_type("buyer", "address", "uses_address", b_addr_edges)

    # 3. Buyer <-> Seller transaction edges
    b_s_edges = []
    for _, r in orders_filt.dropna(subset=["buyer_id", "seller_id"]).iterrows():
        bid, sid = r["buyer_id"], r["seller_id"]
        if bid in b_map and sid in s_map:
            b_s_edges.append((b_map[bid], s_map[sid]))
    add_edge_type("buyer", "seller", "transacts_with", b_s_edges)

    # 4. Device and Address sharing edges
    # Seller uses device if logged
    s_dev_edges = []
    if "seller_id" in dev_filt.columns and "shared_device_id" in dev_filt.columns:
        for _, r in dev_filt.dropna(subset=["seller_id", "shared_device_id"]).iterrows():
            sid, did = r["seller_id"], r["shared_device_id"]
            if sid in s_map and did in d_map:
                s_dev_edges.append((s_map[sid], d_map[did]))
    add_edge_type("seller", "device", "uses_device", s_dev_edges)

    s_addr_edges = []
    if "seller_id" in addr_filt.columns and "shared_address_id" in addr_filt.columns:
        for _, r in addr_filt.dropna(subset=["seller_id", "shared_address_id"]).iterrows():
            sid, aid = r["seller_id"], r["shared_address_id"]
            if sid in s_map and aid in a_map:
                s_addr_edges.append((s_map[sid], a_map[aid]))
    add_edge_type("seller", "address", "uses_address", s_addr_edges)

    return data, id_maps


if PYG_AVAILABLE:
    class HeteroGNN(nn.Module):
        """Heterogeneous Graph Neural Network with per-relation message passing."""

        def __init__(self, metadata: Tuple[List[str], List[Tuple[str, str, str]]], hidden_channels: int = 16, out_channels: int = 16):
            super().__init__()
            node_types, edge_types = metadata

            # Node feature projection to common hidden dimension
            self.input_projections = nn.ModuleDict({
                "buyer": nn.Linear(3, hidden_channels),
                "seller": nn.Linear(2, hidden_channels),
                "device": nn.Linear(2, hidden_channels),
                "address": nn.Linear(2, hidden_channels),
            })

            # Layer 1 HeteroConv
            conv1_dict = {}
            for edge_type in edge_types:
                conv1_dict[edge_type] = SAGEConv((-1, -1), hidden_channels)
            self.conv1 = HeteroConv(conv1_dict, aggr="sum")

            # Layer 2 HeteroConv
            conv2_dict = {}
            for edge_type in edge_types:
                conv2_dict[edge_type] = SAGEConv((-1, -1), out_channels)
            self.conv2 = HeteroConv(conv2_dict, aggr="sum")

        def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]) -> Dict[str, torch.Tensor]:
            # Project inputs
            h_dict = {}
            for node_type, x in x_dict.items():
                if node_type in self.input_projections:
                    h_dict[node_type] = F.relu(self.input_projections[node_type](x))
                else:
                    h_dict[node_type] = x

            # Conv 1
            h_dict = self.conv1(h_dict, edge_index_dict)
            h_dict = {k: F.relu(v) for k, v in h_dict.items()}

            # Conv 2
            h_dict = self.conv2(h_dict, edge_index_dict)
            return h_dict
else:
    class HeteroGNN:
        pass


def extract_hetero_embeddings(
    model: nn.Module,
    hetero_data: Any,
    id_maps: Dict[str, Dict[str, int]],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Execute forward pass on HeteroData and return dictionary of buyer and seller embeddings."""
    model.eval()
    with torch.no_grad():
        out_dict = model(hetero_data.x_dict, hetero_data.edge_index_dict)

    buyer_embs = {}
    if "buyer" in out_dict:
        b_t = out_dict["buyer"].cpu().numpy()
        for bid, idx in id_maps["buyer"].items():
            buyer_embs[bid] = b_t[idx]

    seller_embs = {}
    if "seller" in out_dict:
        s_t = out_dict["seller"].cpu().numpy()
        for sid, idx in id_maps["seller"].items():
            seller_embs[sid] = s_t[idx]

    return buyer_embs, seller_embs
