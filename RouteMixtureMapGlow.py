"""Query-centric LaneGraph route-mixture normalizing flow.

The module deliberately keeps the proven invertible Glow core and places a
history/map-only route proposal model in front of it.  For route component k,

    p_k(x | c) = p_flow(x - tau_k(c) | c, k),

so the proposal translation has unit Jacobian.  The returned scene likelihood
is the exact log-sum-exp over the normalized categorical route distribution.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from MapGlow11_27_original import Glow as InvertibleGlow
from trajectory_representation import (
    _last_history_state_and_motion_torch,
    _wrap_periodic_torch,
)


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weight = mask.to(dtype=x.dtype).unsqueeze(-1)
    return (x * weight).sum(dim=dim) / weight.sum(dim=dim).clamp_min(1.0)


def _last_valid_tokens(
    tokens: torch.Tensor, mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return last valid token and index for [B,V,T,D]/[B,T,V]."""
    b, v, t, d = tokens.shape
    agent_time_mask = mask.permute(0, 2, 1)
    time = torch.arange(t, device=tokens.device).view(1, 1, t)
    index = torch.where(agent_time_mask, time, torch.full_like(time, -1)).amax(-1)
    valid = index >= 0
    gather = index.clamp_min(0).view(b, v, 1, 1).expand(-1, -1, 1, d)
    result = torch.gather(tokens, 2, gather).squeeze(2)
    return result * valid.unsqueeze(-1), index.clamp_min(0)


class FourierFeatures(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, bands: int = 8):
        super().__init__()
        frequencies = 2.0 ** torch.arange(bands, dtype=torch.float32)
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.proj = nn.Sequential(
            nn.Linear(input_dim * bands * 2, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        phase = value.unsqueeze(-1) * self.frequencies.to(value) * math.pi
        encoded = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
        return self.proj(encoded.flatten(-2))


class RelativeGraphAttention(nn.Module):
    """Dense typed graph attention with query-centric metric edge geometry."""

    def __init__(self, dim: int = 256, heads: int = 8, edge_types: int = 4):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.edge_type_bias = nn.Embedding(edge_types + 1, heads)
        self.relative_bias = FourierFeatures(4, heads, bands=8)
        self.ln1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
        )
        self.ln2 = nn.LayerNorm(dim)

    @staticmethod
    def _relative_descriptor(xy: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        delta = xy.unsqueeze(1) - xy.unsqueeze(2)  # target j minus query i
        distance = torch.linalg.vector_norm(delta, dim=-1)
        bearing = torch.atan2(delta[..., 1], delta[..., 0]) - yaw.unsqueeze(-1)
        orientation = yaw.unsqueeze(1) - yaw.unsqueeze(2)
        return torch.stack(
            [
                torch.log1p(distance),
                torch.sin(bearing),
                torch.cos(bearing),
                torch.sin(orientation),
            ],
            dim=-1,
        )

    def forward(
        self,
        x: torch.Tensor,
        xy_m: torch.Tensor,
        yaw: torch.Tensor,
        node_mask: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        b, n, _ = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        logits = torch.einsum("bihd,bjhd->bhij", q, k) / math.sqrt(self.head_dim)
        logits = logits + self.relative_bias(
            self._relative_descriptor(xy_m, yaw)
        ).permute(0, 3, 1, 2)

        allowed = torch.eye(n, dtype=torch.bool, device=x.device).unsqueeze(0).expand(b, -1, -1).clone()
        relation = torch.full((b, n, n), 4, dtype=torch.long, device=x.device)
        for batch_index in range(b):
            valid_edges = edge_mask[batch_index]
            if not valid_edges.any():
                continue
            edges = edge_index[batch_index, valid_edges].long()
            types = edge_type[batch_index, valid_edges].long().clamp(0, 3)
            source, target = edges[:, 0], edges[:, 1]
            in_range = (source >= 0) & (source < n) & (target >= 0) & (target < n)
            source, target, types = source[in_range], target[in_range], types[in_range]
            allowed[batch_index, target, source] = True
            relation[batch_index, target, source] = types
        logits = logits + self.edge_type_bias(relation).permute(0, 3, 1, 2)
        valid_pair = allowed & node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        logits = logits.masked_fill(~valid_pair.unsqueeze(1), -1e4)
        weights = torch.softmax(logits, dim=-1) * valid_pair.unsqueeze(1).to(logits.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        attended = torch.einsum("bhij,bjhd->bihd", weights, v).reshape(b, n, self.dim)
        x = self.ln1(x + self.out(attended))
        x = self.ln2(x + self.ffn(x))
        return x * node_mask.unsqueeze(-1)


class RelativeCrossAttention(nn.Module):
    def __init__(self, dim: int = 256, heads: int = 8):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.relative_bias = FourierFeatures(3, heads, bands=8)
        self.ln1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
        )
        self.ln2 = nn.LayerNorm(dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        query_xy: torch.Tensor,
        query_yaw: torch.Tensor,
        key_xy: torch.Tensor,
        query_mask: torch.Tensor,
        key_mask: torch.Tensor,
    ) -> torch.Tensor:
        b, nq, _ = query.shape
        nk = key.shape[1]
        q = self.q(query).view(b, nq, self.heads, self.head_dim)
        k = self.k(key).view(b, nk, self.heads, self.head_dim)
        v = self.v(key).view(b, nk, self.heads, self.head_dim)
        delta = key_xy.unsqueeze(1) - query_xy.unsqueeze(2)
        bearing = torch.atan2(delta[..., 1], delta[..., 0]) - query_yaw.unsqueeze(-1)
        descriptor = torch.stack(
            [torch.log1p(torch.linalg.vector_norm(delta, dim=-1)), torch.sin(bearing), torch.cos(bearing)],
            dim=-1,
        )
        logits = torch.einsum("bihd,bjhd->bhij", q, k) / math.sqrt(self.head_dim)
        logits = logits + self.relative_bias(descriptor).permute(0, 3, 1, 2)
        pair_mask = query_mask.unsqueeze(-1) & key_mask.unsqueeze(1)
        logits = logits.masked_fill(~pair_mask.unsqueeze(1), -1e4)
        weight = torch.softmax(logits, -1) * pair_mask.unsqueeze(1).to(logits.dtype)
        weight = weight / weight.sum(-1, keepdim=True).clamp_min(1e-8)
        attended = torch.einsum("bhij,bjhd->bihd", weight, v).reshape(b, nq, self.dim)
        x = self.ln1(query + self.out(attended))
        x = self.ln2(x + self.ffn(x))
        return x * query_mask.unsqueeze(-1)


class QueryCentricSceneEncoder(nn.Module):
    def __init__(self, history_input_dim: int = 5, dim: int = 256, heads: int = 8):
        super().__init__()
        self.dim = dim
        self.point_mlp = nn.Sequential(
            nn.Linear(9, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.speed_mlp = nn.Linear(1, dim)
        self.subtype_embed = nn.Embedding(8, dim, padding_idx=0)
        self.boundary_embed = nn.Embedding(8, dim, padding_idx=0)
        self.map_ln = nn.LayerNorm(dim)
        self.graph_layers = nn.ModuleList(
            [RelativeGraphAttention(dim, heads) for _ in range(3)]
        )
        if history_input_dim != 5:
            raise ValueError("query-centric history encoder requires five dynamic channels")
        self.history_mlp = nn.Sequential(
            nn.Linear(7, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        temporal_layer = nn.TransformerEncoderLayer(
            dim, heads, 4 * dim, dropout=0.0, activation="gelu", batch_first=True, norm_first=True
        )
        self.temporal = nn.TransformerEncoder(temporal_layer, num_layers=2)
        self.agent_type = nn.Embedding(10, dim, padding_idx=0)
        self.agent_interaction = RelativeGraphAttention(dim, heads, edge_types=4)
        self.agent_to_map = RelativeCrossAttention(dim, heads)

    def forward(
        self,
        *,
        history_data: torch.Tensor,
        history_timestep_mask: torch.Tensor,
        context_agent_mask: torch.Tensor,
        agent_types: Optional[torch.Tensor],
        map_data: torch.Tensor,
        map_mask: torch.Tensor,
        map_speed_limit: torch.Tensor,
        map_lane_subtype: torch.Tensor,
        map_left_boundary_type: torch.Tensor,
        map_right_boundary_type: torch.Tensor,
        lane_edge_index: torch.Tensor,
        lane_edge_type: torch.Tensor,
        lane_edge_mask: torch.Tensor,
        state_scales: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        b, _, th, v = history_data.shape
        _, lanes, points, _ = map_data.shape
        dtype = history_data.dtype
        position_scale = state_scales[:, 0].to(dtype)
        velocity_scale = state_scales[:, 1].to(dtype)
        yaw_scale = state_scales[:, 2].to(dtype)

        map_valid = map_mask.any(-1)
        xy_m = map_data[..., :2] * position_scale.view(b, 1, 1, 1)
        point_valid = map_mask
        delta = torch.zeros_like(xy_m)
        delta[:, :, 1:] = xy_m[:, :, 1:] - xy_m[:, :, :-1]
        if points > 1:
            delta[:, :, 0] = delta[:, :, 1]
        lane_delta = xy_m[:, :, -1] - xy_m[:, :, 0]
        lane_yaw = torch.atan2(lane_delta[..., 1], lane_delta[..., 0])
        cos_lane = torch.cos(lane_yaw).unsqueeze(-1)
        sin_lane = torch.sin(lane_yaw).unsqueeze(-1)
        centered = xy_m - xy_m[:, :, :1]
        local_centered = torch.stack(
            [
                cos_lane * centered[..., 0] + sin_lane * centered[..., 1],
                -sin_lane * centered[..., 0] + cos_lane * centered[..., 1],
            ], dim=-1,
        )
        local_delta = torch.stack(
            [
                cos_lane * delta[..., 0] + sin_lane * delta[..., 1],
                -sin_lane * delta[..., 0] + cos_lane * delta[..., 1],
            ], dim=-1,
        )
        yaw = torch.atan2(delta[..., 1], delta[..., 0]) - lane_yaw.unsqueeze(-1)
        type_features = map_data[..., 3:6]
        point_input = torch.cat(
            [
                local_centered / 10.0,
                local_delta / 10.0,
                torch.sin(yaw).unsqueeze(-1),
                torch.cos(yaw).unsqueeze(-1),
                type_features,
            ],
            dim=-1,
        )
        point_tokens = self.point_mlp(point_input) * point_valid.unsqueeze(-1)
        lane_tokens = _masked_mean(point_tokens, point_valid, dim=2)
        lane_tokens = lane_tokens + self.speed_mlp((map_speed_limit / 30.0).unsqueeze(-1))
        lane_tokens = lane_tokens + self.subtype_embed(map_lane_subtype.long().clamp(0, 7))
        lane_tokens = lane_tokens + self.boundary_embed(map_left_boundary_type.long().clamp(0, 7))
        lane_tokens = lane_tokens + self.boundary_embed(map_right_boundary_type.long().clamp(0, 7))
        lane_tokens = self.map_ln(lane_tokens) * map_valid.unsqueeze(-1)
        lane_xy = _masked_mean(xy_m, point_valid, dim=2)
        for layer in self.graph_layers:
            lane_tokens = layer(
                lane_tokens, lane_xy, lane_yaw, map_valid,
                lane_edge_index, lane_edge_type, lane_edge_mask,
            )

        history_mask = history_timestep_mask.bool()
        history_xy_m = history_data[:, :2].permute(0, 3, 2, 1) * position_scale.view(b, 1, 1, 1)
        history_v_m = history_data[:, 2:4].permute(0, 3, 2, 1) * velocity_scale.view(b, 1, 1, 1)
        history_yaw = history_data[:, 4].permute(0, 2, 1) * yaw_scale.view(b, 1, 1)
        anchor, _, _ = _last_history_state_and_motion_torch(history_data, history_mask)
        anchor_xy_m = anchor[:, :2].permute(0, 2, 1) * position_scale.view(b, 1, 1)
        anchor_yaw = anchor[:, 4] * yaw_scale.view(b, 1)
        cos_anchor = torch.cos(anchor_yaw).view(b, v, 1)
        sin_anchor = torch.sin(anchor_yaw).view(b, v, 1)
        relative_xy = history_xy_m - anchor_xy_m.unsqueeze(2)
        local_history_xy = torch.stack(
            [
                cos_anchor * relative_xy[..., 0] + sin_anchor * relative_xy[..., 1],
                -sin_anchor * relative_xy[..., 0] + cos_anchor * relative_xy[..., 1],
            ], dim=-1,
        )
        local_history_v = torch.stack(
            [
                cos_anchor * history_v_m[..., 0] + sin_anchor * history_v_m[..., 1],
                -sin_anchor * history_v_m[..., 0] + cos_anchor * history_v_m[..., 1],
            ], dim=-1,
        )
        relative_history_yaw = history_yaw - anchor_yaw.unsqueeze(-1)
        time = torch.arange(th, device=history_data.device, dtype=dtype).view(1, 1, th, 1) / max(th - 1, 1)
        history_input = torch.cat(
            [
                local_history_xy / 100.0,
                local_history_v / 30.0,
                torch.sin(relative_history_yaw).unsqueeze(-1),
                torch.cos(relative_history_yaw).unsqueeze(-1),
                time.expand(b, v, -1, -1),
            ],
            dim=-1,
        )
        tokens = self.history_mlp(history_input).reshape(b * v, th, self.dim)
        token_mask = ~history_mask.permute(0, 2, 1).reshape(b * v, th)
        causal = torch.triu(torch.ones(th, th, dtype=torch.bool, device=tokens.device), diagonal=1)
        all_pad = token_mask.all(-1)
        safe_mask = token_mask.clone()
        safe_mask[all_pad, 0] = False
        tokens = self.temporal(tokens, mask=causal, src_key_padding_mask=safe_mask)
        tokens = tokens.view(b, v, th, self.dim)
        agent_tokens, last_index = _last_valid_tokens(tokens, history_mask)
        gather_xy = last_index.view(b, v, 1, 1).expand(-1, -1, 1, 2)
        agent_xy = torch.gather(history_xy_m, 2, gather_xy).squeeze(2)
        gather_yaw = last_index.view(b, v, 1)
        agent_yaw = torch.gather(history_yaw, 2, gather_yaw).squeeze(2)
        agent_valid = context_agent_mask.bool() & history_mask.any(1)
        if agent_types is not None:
            agent_tokens = agent_tokens + self.agent_type(agent_types.long().clamp(0, 9))

        # Fully connected agent graph; relation geometry remains query-centric.
        agent_edges = []
        for source in range(v):
            for target in range(v):
                if source != target:
                    agent_edges.append((source, target))
        agent_edge_index = torch.tensor(agent_edges, device=tokens.device, dtype=torch.long)
        agent_edge_index = agent_edge_index.unsqueeze(0).expand(b, -1, -1)
        agent_edge_mask = torch.ones(
            b, len(agent_edges), dtype=torch.bool, device=tokens.device
        )
        agent_edge_type = torch.zeros_like(agent_edge_mask, dtype=torch.long)
        agent_tokens = self.agent_interaction(
            agent_tokens, agent_xy, agent_yaw, agent_valid,
            agent_edge_index, agent_edge_type, agent_edge_mask,
        )
        agent_tokens = self.agent_to_map(
            agent_tokens, lane_tokens, agent_xy, agent_yaw,
            lane_xy, agent_valid, map_valid,
        )
        return {
            "agent_tokens": agent_tokens,
            "lane_tokens": lane_tokens,
            "agent_xy_m": agent_xy,
            "agent_yaw": agent_yaw,
            "agent_valid": agent_valid,
            "lane_valid": map_valid,
        }


class RouteProposalDecoder(nn.Module):
    def __init__(self, dim: int = 256, modes: int = 6, future_steps: int = 30):
        super().__init__()
        self.modes = modes
        self.future_steps = future_steps
        self.mode_embed = nn.Parameter(torch.randn(modes, dim) * 0.02)
        self.mode_to_scene = nn.MultiheadAttention(dim, 8, batch_first=True, dropout=0.0)
        self.mode_self = nn.MultiheadAttention(dim, 8, batch_first=True, dropout=0.0)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.position_head = nn.Sequential(
            nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, future_steps * 2)
        )
        self.logit_head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(
        self, agent_tokens: torch.Tensor, lane_tokens: torch.Tensor,
        agent_valid: torch.Tensor, lane_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, v, d = agent_tokens.shape
        k = self.modes
        query = agent_tokens.unsqueeze(2) + self.mode_embed.view(1, 1, k, d)
        query_flat = query.reshape(b * v, k, d)
        lanes = lane_tokens.unsqueeze(1).expand(b, v, -1, -1).reshape(b * v, lane_tokens.shape[1], d)
        lane_pad = ~lane_valid.unsqueeze(1).expand(b, v, -1).reshape(b * v, -1)
        all_pad = lane_pad.all(-1)
        lane_pad = lane_pad.clone()
        lane_pad[all_pad, 0] = False
        mapped, _ = self.mode_to_scene(query_flat, lanes, lanes, key_padding_mask=lane_pad)
        query_flat = self.ln1(query_flat + mapped)
        interacted, _ = self.mode_self(query_flat, query_flat, query_flat)
        query_flat = self.ln2(query_flat + interacted)
        route_tokens = query_flat.view(b, v, k, d) * agent_valid.unsqueeze(-1).unsqueeze(-1)
        local_position = self.position_head(route_tokens).view(b, v, k, self.future_steps, 2)
        local_position = local_position.permute(0, 2, 3, 1, 4).contiguous()  # B,K,T,V,2
        pooled = (
            route_tokens * agent_valid.unsqueeze(-1).unsqueeze(-1)
        ).sum(1) / agent_valid.sum(1).clamp_min(1).view(b, 1, 1)
        logits = self.logit_head(pooled).squeeze(-1)
        return local_position, logits, route_tokens


class Glow(nn.Module):
    """Drop-in training interface for the route-mixture model."""

    model_arch = "route_mixture_glow"

    def __init__(
        self,
        in_channel: int,
        condition_dim: int,
        n_flow: int,
        n_block: int,
        affine: bool = True,
        conv_lu: bool = True,
        history_input_dim: Optional[int] = None,
        route_modes: int = 6,
        future_steps: int = 30,
        hidden_dim: int = 256,
        mixture_chunk_size: int = 2,
    ):
        super().__init__()
        if in_channel != 5:
            raise ValueError("RouteMixtureMapGlow currently requires five dynamic channels")
        self.in_channel = in_channel
        self.route_modes = int(route_modes)
        self.future_steps = int(future_steps)
        self.mixture_chunk_size = max(1, int(mixture_chunk_size))
        self.scene_encoder = QueryCentricSceneEncoder(
            history_input_dim=history_input_dim or in_channel, dim=hidden_dim
        )
        self.route_decoder = RouteProposalDecoder(hidden_dim, self.route_modes, self.future_steps)
        self.flow = InvertibleGlow(
            in_channel, condition_dim, n_flow, n_block,
            affine=affine, conv_lu=conv_lu, history_input_dim=history_input_dim,
        )

    @property
    def blocks(self):
        return self.flow.blocks

    @staticmethod
    def _required_topology(kwargs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        required = (
            "map_speed_limit", "map_lane_subtype", "map_left_boundary_type",
            "map_right_boundary_type", "lane_edge_index", "lane_edge_type", "lane_edge_mask",
        )
        missing = [name for name in required if kwargs.get(name) is None]
        if missing:
            raise ValueError(f"route_mixture_glow requires v5 topology tensors: {missing}")
        return {name: kwargs[name] for name in required}

    def _route_outputs(self, **kwargs) -> Dict[str, torch.Tensor]:
        topology = self._required_topology(kwargs)
        encoded = self.scene_encoder(
            history_data=kwargs["history_data"],
            history_timestep_mask=kwargs["history_timestep_mask"],
            context_agent_mask=kwargs["context_agent_mask"],
            agent_types=kwargs.get("agent_types"),
            map_data=kwargs["map_data"], map_mask=kwargs["map_mask"],
            state_scales=kwargs["state_scales"], **topology,
        )
        local_position, logits, route_tokens = self.route_decoder(
            encoded["agent_tokens"], encoded["lane_tokens"],
            encoded["agent_valid"], encoded["lane_valid"],
        )
        proposal = self._physically_consistent_proposal(local_position, **kwargs)
        return {
            **encoded,
            "proposal": proposal,
            "route_logits": logits,
            "route_probabilities": torch.softmax(logits, dim=-1),
            "route_tokens": route_tokens,
        }

    def _physically_consistent_proposal(self, local_position: torch.Tensor, **kwargs) -> torch.Tensor:
        history = kwargs["history_data"]
        history_mask = kwargs["history_timestep_mask"].bool()
        scales = kwargs["state_scales"].to(history)
        b, k, t, v, _ = local_position.shape
        if t != self.future_steps:
            raise ValueError(f"proposal steps {t} != configured future_steps {self.future_steps}")
        anchor, step_motion, has_anchor = _last_history_state_and_motion_torch(history, history_mask)
        yaw_scale = scales[:, 2].view(b, 1)
        anchor_yaw_rad = anchor[:, 4] * yaw_scale
        cos_yaw = torch.cos(anchor_yaw_rad).view(b, 1, 1, v)
        sin_yaw = torch.sin(anchor_yaw_rad).view(b, 1, 1, v)

        # Network output is bounded metric displacement around the CV baseline.
        local_xy_normalized = torch.tanh(local_position) * (20.0 / scales[:, 0].view(b, 1, 1, 1, 1))
        local_x, local_y = local_xy_normalized[..., 0], local_xy_normalized[..., 1]
        world_x = cos_yaw * local_x - sin_yaw * local_y
        world_y = sin_yaw * local_x + cos_yaw * local_y
        baseline = anchor[:, :2].permute(0, 2, 1).view(b, 1, 1, v, 2) + (
            torch.arange(1, t + 1, device=history.device, dtype=history.dtype).view(1, 1, t, 1, 1)
            * step_motion.permute(0, 2, 1).view(b, 1, 1, v, 2)
        )
        absolute_xy = baseline + torch.stack([world_x, world_y], dim=-1)
        anchor_xy = anchor[:, :2].permute(0, 2, 1).view(b, 1, 1, v, 2).expand(-1, k, -1, -1, -1)
        previous_xy = torch.cat([anchor_xy, absolute_xy[:, :, :-1]], dim=2)
        velocity_world = (absolute_xy - previous_xy) * (
            scales[:, 0] / (0.1 * scales[:, 1])
        ).view(b, 1, 1, 1, 1)
        anchor_velocity = anchor[:, 2:4].permute(0, 2, 1).view(b, 1, 1, v, 2)
        velocity_residual = velocity_world - anchor_velocity
        local_vx = cos_yaw * velocity_residual[..., 0] + sin_yaw * velocity_residual[..., 1]
        local_vy = -sin_yaw * velocity_residual[..., 0] + cos_yaw * velocity_residual[..., 1]
        speed = torch.linalg.vector_norm(velocity_world, dim=-1)
        yaw_rad = torch.atan2(velocity_world[..., 1], velocity_world[..., 0])
        yaw_rad = torch.where(speed > 1e-3, yaw_rad, anchor_yaw_rad.view(b, 1, 1, v))
        yaw_residual = _wrap_periodic_torch(
            yaw_rad / yaw_scale.view(b, 1, 1, 1) - anchor[:, 4].view(b, 1, 1, v),
            2.0 * math.pi / yaw_scale.view(b, 1, 1, 1),
        )
        proposal = torch.stack([local_x, local_y, local_vx, local_vy, yaw_residual], dim=2)
        valid = kwargs["context_agent_mask"].view(b, 1, 1, 1, v)
        return proposal * valid.to(proposal.dtype)

    @staticmethod
    def _repeat_batch(value, repeats: int):
        if not torch.is_tensor(value):
            return value
        return value.repeat_interleave(repeats, dim=0)

    def _flow_kwargs(self, kwargs: Dict[str, torch.Tensor], repeats: int) -> Dict[str, torch.Tensor]:
        accepted = (
            "map_data", "map_mask", "agent_types", "history_data", "target_vehicle_mask",
            "history_vehicle_mask", "timestep_mask", "context_agent_mask",
            "history_timestep_mask", "scene_stats", "state_scales", "static_dimensions",
        )
        return {
            name: self._repeat_batch(kwargs[name], repeats)
            for name in accepted if kwargs.get(name) is not None
        }

    def _auxiliary_losses(
        self, input: torch.Tensor, proposal: torch.Tensor,
        logits: torch.Tensor, timestep_mask: torch.Tensor, state_scales: torch.Tensor,
        history_data: torch.Tensor, history_timestep_mask: torch.Tensor,
        map_data: torch.Tensor, map_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        mask = timestep_mask.bool().unsqueeze(1)  # B,1,T,V
        scale = state_scales[:, 0].view(-1, 1, 1, 1)
        displacement = torch.linalg.vector_norm(
            (proposal[:, :, :2] - input[:, None, :2]) * scale.unsqueeze(2), dim=2
        )
        yaw_change = input[:, 4].abs() * state_scales[:, 2].view(-1, 1, 1)
        turning = (yaw_change.amax(1) > math.radians(15.0)).to(input.dtype)
        agent_weight = 1.0 + turning
        weight = mask.to(input.dtype) * agent_weight[:, None, None, :]
        component_ade = (displacement * weight).sum((2, 3)) / weight.sum((2, 3)).clamp_min(1.0)
        best_mode = component_ade.detach().argmin(-1)
        proposal_loss = component_ade.gather(1, best_mode[:, None]).mean()
        classification_loss = F.cross_entropy(logits, best_mode)
        position_m = proposal[:, :, :2] * scale.unsqueeze(2)
        acceleration = position_m[:, :, :, 2:] - 2.0 * position_m[:, :, :, 1:-1] + position_m[:, :, :, :-2]
        smooth_mask = mask[:, :, 2:] & mask[:, :, 1:-1] & mask[:, :, :-2]
        smoothness_loss = (
            torch.linalg.vector_norm(acceleration, dim=2) * smooth_mask.to(input.dtype)
        ).sum() / smooth_mask.sum().clamp_min(1)

        # WTA alone leaves non-winning route queries almost unconstrained and
        # can collapse all proposals to the same path. Enforce a modest metric
        # endpoint margin, then keep those endpoints close to actual map
        # centerline points. The two terms work together: diversity cannot be
        # satisfied by arbitrary off-road offsets.
        endpoint = proposal[:, :, :2, -1].permute(0, 1, 3, 2) * scale.squeeze(2).unsqueeze(1)
        endpoint_by_agent = endpoint.permute(0, 2, 1, 3)  # [B,V,K,2]
        endpoint_distance = torch.cdist(endpoint_by_agent, endpoint_by_agent)
        pair_mask = torch.triu(
            torch.ones(
                self.route_modes, self.route_modes,
                dtype=torch.bool, device=input.device,
            ), diagonal=1,
        ).view(1, 1, self.route_modes, self.route_modes)
        valid_agent = timestep_mask.bool().any(1).unsqueeze(-1).unsqueeze(-1)
        diversity_mask = pair_mask & valid_agent
        diversity_loss = (
            F.relu(2.0 - endpoint_distance).square()
            * diversity_mask.to(input.dtype)
        ).sum() / diversity_mask.sum().clamp_min(1)

        anchor, step_motion, _ = _last_history_state_and_motion_torch(
            history_data, history_timestep_mask.bool()
        )
        yaw0 = anchor[:, 4] * state_scales[:, 2].view(-1, 1)
        cos_yaw = torch.cos(yaw0).view(input.shape[0], 1, 1, -1)
        sin_yaw = torch.sin(yaw0).view(input.shape[0], 1, 1, -1)
        selected_steps = torch.arange(
            min(4, proposal.shape[3] - 1), proposal.shape[3], 5,
            device=input.device,
        )
        local_xy = proposal[:, :, :2, selected_steps].permute(0, 1, 3, 4, 2)
        world_residual = torch.stack(
            [
                cos_yaw * local_xy[..., 0] - sin_yaw * local_xy[..., 1],
                sin_yaw * local_xy[..., 0] + cos_yaw * local_xy[..., 1],
            ], dim=-1,
        )
        step_number = (selected_steps + 1).to(input.dtype).view(1, 1, -1, 1, 1)
        baseline = (
            anchor[:, :2].permute(0, 2, 1).view(-1, 1, 1, proposal.shape[-1], 2)
            + step_number * step_motion.permute(0, 2, 1).view(-1, 1, 1, proposal.shape[-1], 2)
        )
        proposal_xy_m = (baseline + world_residual) * state_scales[:, 0].view(-1, 1, 1, 1, 1)
        map_xy_m = map_data[..., :2] * state_scales[:, 0].view(-1, 1, 1, 1)
        centerline_mask = map_mask.bool() & (map_data[..., 3] > 0.5)
        compliance_sum = input.new_zeros(())
        compliance_count = input.new_zeros(())
        proposal_valid = timestep_mask[:, selected_steps].bool()
        for batch_index in range(input.shape[0]):
            map_points = map_xy_m[batch_index][centerline_mask[batch_index]]
            if map_points.numel() == 0:
                continue
            query = proposal_xy_m[batch_index].reshape(-1, 2)
            nearest = torch.cdist(query, map_points).amin(-1).view(
                self.route_modes, len(selected_steps), proposal.shape[-1]
            )
            valid = proposal_valid[batch_index].unsqueeze(0).expand_as(nearest)
            compliance_sum = compliance_sum + (nearest * valid.to(nearest.dtype)).sum()
            compliance_count = compliance_count + valid.sum()
        map_compliance_loss = compliance_sum / compliance_count.clamp_min(1)
        return {
            "proposal_loss": proposal_loss,
            "route_classification_loss": classification_loss,
            "proposal_smoothness_loss": smoothness_loss,
            "proposal_diversity_loss": diversity_loss,
            "proposal_map_compliance_loss": map_compliance_loss,
            "best_route_mode": best_mode,
            "component_proposal_ade": component_ade,
        }

    def forward(self, input, condition=None, **kwargs):
        del condition  # target-derived labels are intentionally not conditioner inputs
        b, _, t, v = input.shape
        if t != self.future_steps:
            raise ValueError(f"input future steps {t} != configured {self.future_steps}")
        routes = self._route_outputs(**kwargs)
        proposal = routes["proposal"]
        component_log_likelihood = []
        component_log_p = []
        component_logdet = []
        mask = kwargs["timestep_mask"].bool().unsqueeze(1).unsqueeze(2)
        for start in range(0, self.route_modes, self.mixture_chunk_size):
            stop = min(start + self.mixture_chunk_size, self.route_modes)
            count = stop - start
            residual = input[:, None] - proposal[:, start:stop]
            residual = torch.where(mask, residual, torch.full_like(residual, -1.0))
            residual = residual.reshape(b * count, self.in_channel, t, v)
            route_ids = torch.arange(start + 1, stop + 1, device=input.device).view(1, count, 1)
            route_ids = route_ids.expand(b, -1, v).reshape(b * count, v)
            flow_kwargs = self._flow_kwargs(kwargs, count)
            log_p, logdet, _ = self.flow(residual, condition=route_ids, **flow_kwargs)
            component_log_p.append(log_p.view(b, count))
            component_logdet.append(logdet.view(b, count))
            component_log_likelihood.append((log_p + logdet).view(b, count))
        component_log_p = torch.cat(component_log_p, dim=1)
        component_logdet = torch.cat(component_logdet, dim=1)
        component_log_likelihood = torch.cat(component_log_likelihood, dim=1)
        log_weight = torch.log_softmax(routes["route_logits"], dim=-1)
        mixture_log_likelihood = torch.logsumexp(log_weight + component_log_likelihood, dim=-1)
        aux = {
            **routes,
            **self._auxiliary_losses(
                input, proposal, routes["route_logits"],
                kwargs["timestep_mask"], kwargs["state_scales"],
                kwargs["history_data"], kwargs["history_timestep_mask"],
                kwargs["map_data"], kwargs["map_mask"],
            ),
            "component_log_p": component_log_p,
            "component_logdet": component_logdet,
            "component_log_likelihood": component_log_likelihood,
        }
        # The route-specific Jacobians live inside the log-sum-exp and cannot
        # be separated into a single scalar logdet. Return the exact likelihood
        # as log_p and zero as the compatibility logdet.
        return mixture_log_likelihood, torch.zeros_like(mixture_log_likelihood), aux

    def reverse(
        self, z_list, condition=None, mode_index=None, return_aux: bool = False,
        guidance_scale: float = 1.0, **kwargs,
    ):
        del condition
        routes = self._route_outputs(**kwargs)
        b = routes["proposal"].shape[0]
        if mode_index is None:
            mode_index = routes["route_probabilities"].argmax(-1)
        elif not torch.is_tensor(mode_index):
            mode_index = torch.full((b,), int(mode_index), dtype=torch.long, device=routes["proposal"].device)
        mode_index = mode_index.to(device=routes["proposal"].device, dtype=torch.long)
        if tuple(mode_index.shape) != (b,) or (mode_index < 0).any() or (mode_index >= self.route_modes).any():
            raise ValueError(f"mode_index must be [B] with values in [0,{self.route_modes - 1}]")
        route_ids = (mode_index + 1).unsqueeze(-1).expand(-1, routes["agent_valid"].shape[1])
        flow_kwargs = self._flow_kwargs(kwargs, 1)
        residual = self.flow.reverse(
            z_list, condition=route_ids, guidance_scale=guidance_scale, **flow_kwargs
        )
        gather = mode_index.view(b, 1, 1, 1, 1).expand(
            -1, 1, self.in_channel, self.future_steps, residual.shape[-1]
        )
        proposal = torch.gather(routes["proposal"], 1, gather).squeeze(1)
        valid = kwargs["timestep_mask"].bool().unsqueeze(1)
        sample = torch.where(valid, residual + proposal, residual)
        if return_aux:
            return sample, routes
        return sample
