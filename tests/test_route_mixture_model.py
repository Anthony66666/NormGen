import math
import sys
import types
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from RouteMixtureMapGlow import Glow, QueryCentricSceneEncoder  # noqa: E402
from data_preprocess import (  # noqa: E402
    LANE_EDGE_TYPE_MAPPING,
    _infer_lanelet_edges,
)
from trajectory_representation import decode_constant_velocity_residual_torch  # noqa: E402


class _ToyFlow(nn.Module):
    def forward(self, x, condition=None, **kwargs):
        del kwargs
        log_p = -x.square().flatten(1).sum(1) + condition[:, 0].to(x.dtype) * 0.1
        return log_p, torch.zeros_like(log_p), [x]

    def reverse(self, z_list, condition=None, **kwargs):
        del condition, kwargs
        return z_list[0]


class RouteMixtureMathTest(unittest.TestCase):
    def test_exact_mixture_uses_component_jacobians_inside_logsumexp(self):
        model = Glow(
            5, 32, 1, 1, history_input_dim=5, route_modes=2,
            future_steps=2, hidden_dim=32, mixture_chunk_size=1,
        )
        model.flow = _ToyFlow()
        proposal = torch.zeros(1, 2, 5, 2, 1)
        proposal[:, 1] = 0.25
        logits = torch.tensor([[0.3, -0.2]])

        def fake_routes(_self, **kwargs):
            del kwargs
            return {
                "proposal": proposal,
                "route_logits": logits,
                "route_probabilities": torch.softmax(logits, -1),
            }

        model._route_outputs = types.MethodType(fake_routes, model)
        target = torch.full((1, 5, 2, 1), 0.1)
        mask = torch.ones(1, 2, 1, dtype=torch.bool)
        log_p, logdet, aux = model(
            target,
            timestep_mask=mask,
            state_scales=torch.tensor([[50.0, 15.0, math.pi]]),
            history_data=torch.zeros(1, 5, 2, 1),
            history_timestep_mask=torch.ones(1, 2, 1, dtype=torch.bool),
            map_data=torch.tensor([[[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                      [0.1, 0.0, 0.0, 1.0, 0.0, 0.0]]]]),
            map_mask=torch.ones(1, 1, 2, dtype=torch.bool),
        )
        component0 = -target.square().sum() + 0.1
        component1 = -(target - 0.25).square().sum() + 0.2
        expected = torch.logsumexp(
            torch.log_softmax(logits[0], -1) + torch.stack([component0, component1]), dim=0
        )
        torch.testing.assert_close(log_p[0], expected)
        torch.testing.assert_close(logdet, torch.zeros_like(logdet))
        torch.testing.assert_close(
            aux["component_log_likelihood"][0], torch.stack([component0, component1])
        )

    def test_proposal_translation_has_unit_jacobian(self):
        x = torch.randn(10, dtype=torch.double, requires_grad=True)
        tau = torch.randn(10, dtype=torch.double)
        jacobian = torch.autograd.functional.jacobian(lambda value: value - tau, x)
        self.assertAlmostEqual(torch.linalg.slogdet(jacobian).logabsdet.item(), 0.0, places=12)

    def test_position_derived_proposal_is_kinematically_consistent(self):
        model = Glow(
            5, 32, 1, 1, history_input_dim=5, route_modes=2,
            future_steps=3, hidden_dim=32,
        )
        history = torch.zeros(1, 5, 2, 1)
        history[:, 0, 0, 0] = 0.00
        history[:, 0, 1, 0] = 0.02
        history[:, 2, :, 0] = 10.0 / 15.0
        mask = torch.ones(1, 2, 1, dtype=torch.bool)
        future_mask = torch.ones(1, 3, 1, dtype=torch.bool)
        kwargs = {
            "history_data": history,
            "history_timestep_mask": mask,
            "state_scales": torch.tensor([[50.0, 15.0, math.pi]]),
            "context_agent_mask": torch.ones(1, 1, dtype=torch.bool),
        }
        proposal = model._physically_consistent_proposal(
            torch.randn(1, 2, 3, 1, 2) * 0.1, **kwargs
        )
        for mode in range(2):
            absolute = decode_constant_velocity_residual_torch(
                proposal[:, mode], history, future_mask, mask, yaw_scale=torch.tensor([math.pi])
            )
            implied_velocity = (absolute[:, :2, 1:] - absolute[:, :2, :-1]) * 50.0 / 0.1
            saved_velocity = absolute[:, 2:4, 1:] * 15.0
            torch.testing.assert_close(implied_velocity, saved_velocity, rtol=1e-5, atol=1e-5)


class LaneGraphContractTest(unittest.TestCase):
    def test_inferred_edges_have_reciprocals_and_typed_neighbors(self):
        lanes = [
            {
                "lanelet_id": 1,
                "coords": np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
                "left_way_id": "a", "right_way_id": "shared",
            },
            {
                "lanelet_id": 2,
                "coords": np.asarray([[1.0, 0.0], [2.0, 0.0]], dtype=np.float32),
                "left_way_id": "b", "right_way_id": "c",
            },
            {
                "lanelet_id": 3,
                "coords": np.asarray([[0.0, -1.0], [1.0, -1.0]], dtype=np.float32),
                "left_way_id": "shared", "right_way_id": "d",
            },
        ]
        edges = set(_infer_lanelet_edges(lanes))
        self.assertIn((1, 2, LANE_EDGE_TYPE_MAPPING["successor"]), edges)
        self.assertIn((2, 1, LANE_EDGE_TYPE_MAPPING["predecessor"]), edges)
        self.assertIn((1, 3, LANE_EDGE_TYPE_MAPPING["right"]), edges)
        self.assertIn((3, 1, LANE_EDGE_TYPE_MAPPING["left"]), edges)

    def test_query_encoder_is_translation_and_rotation_invariant(self):
        torch.manual_seed(7)
        encoder = QueryCentricSceneEncoder(history_input_dim=5, dim=32, heads=8).eval()
        b, v, t, lanes, points = 1, 2, 3, 2, 3
        history = torch.zeros(b, 5, t, v)
        history[:, 0, :, 0] = torch.tensor([0.0, 0.1, 0.2])
        history[:, 1, :, 1] = torch.tensor([0.0, 0.1, 0.2])
        history[:, 2, :, 0] = 0.5
        history[:, 3, :, 1] = 0.5
        history[:, 4, :, 1] = 0.5
        map_data = torch.zeros(b, lanes, points, 6)
        map_data[0, 0, :, 0] = torch.tensor([0.0, 0.1, 0.2])
        map_data[0, 1, :, 1] = torch.tensor([0.0, 0.1, 0.2])
        map_data[..., 3] = 1.0
        common = dict(
            history_timestep_mask=torch.ones(b, t, v, dtype=torch.bool),
            context_agent_mask=torch.ones(b, v, dtype=torch.bool),
            agent_types=torch.ones(b, v, dtype=torch.long),
            map_mask=torch.ones(b, lanes, points, dtype=torch.bool),
            map_speed_limit=torch.zeros(b, lanes),
            map_lane_subtype=torch.ones(b, lanes, dtype=torch.long),
            map_left_boundary_type=torch.zeros(b, lanes, dtype=torch.long),
            map_right_boundary_type=torch.zeros(b, lanes, dtype=torch.long),
            lane_edge_index=torch.tensor([[[0, 1], [1, 0]]]),
            lane_edge_type=torch.tensor([[0, 1]]),
            lane_edge_mask=torch.ones(b, 2, dtype=torch.bool),
            state_scales=torch.tensor([[50.0, 15.0, math.pi]]),
        )
        original = encoder(history_data=history, map_data=map_data, **common)

        angle = 0.7
        rotation = torch.tensor(
            [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
        )
        transformed_history = history.clone()
        transformed_history[:, :2] = torch.einsum("ij,bjtv->bitv", rotation, history[:, :2])
        transformed_history[:, :2] += torch.tensor([0.3, -0.2]).view(1, 2, 1, 1)
        transformed_history[:, 2:4] = torch.einsum("ij,bjtv->bitv", rotation, history[:, 2:4])
        transformed_history[:, 4] += angle / math.pi
        transformed_map = map_data.clone()
        transformed_map[..., :2] = torch.einsum("ij,blpj->blpi", rotation, map_data[..., :2])
        transformed_map[..., :2] += torch.tensor([0.3, -0.2]).view(1, 1, 1, 2)
        transformed = encoder(
            history_data=transformed_history, map_data=transformed_map, **common
        )
        torch.testing.assert_close(
            original["agent_tokens"], transformed["agent_tokens"], rtol=2e-4, atol=2e-4
        )
        torch.testing.assert_close(
            original["lane_tokens"], transformed["lane_tokens"], rtol=2e-4, atol=2e-4
        )


if __name__ == "__main__":
    unittest.main()
