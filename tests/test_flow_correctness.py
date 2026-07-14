import math
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from MapGlow11_27_original import (  # noqa: E402
    ActNorm,
    AffineCoupling,
    AgentInteractionModule,
    Block,
    ContextEncoder,
    Flow,
    Glow,
    InvConv2d,
    ZeroLinear,
    gaussian_log_p,
)


class ProbabilityAndAngleTest(unittest.TestCase):
    def test_gaussian_has_unclipped_tails_and_safe_masking(self):
        mean = torch.zeros(2, dtype=torch.float64)
        log_sd = torch.zeros_like(mean)
        at_1000 = gaussian_log_p(torch.tensor([1000.0, 0.0], dtype=torch.float64), mean, log_sd)
        at_2000 = gaussian_log_p(torch.tensor([2000.0, 0.0], dtype=torch.float64), mean, log_sd)

        self.assertLess(at_2000[0], at_1000[0] - 1_000_000.0)
        self.assertAlmostEqual(
            at_1000[0].item(),
            -0.5 * math.log(2.0 * math.pi) - 0.5 * 1000.0**2,
            places=7,
        )

        masked = gaussian_log_p(
            torch.tensor([1.0, float("nan")]),
            torch.zeros(2),
            torch.zeros(2),
            mask=torch.tensor([True, False]),
        )
        self.assertTrue(torch.isfinite(masked).all())
        self.assertEqual(masked[1].item(), 0.0)

    def test_relative_yaw_uses_radians_for_pi_normalized_input(self):
        module = AgentInteractionModule(filter_size=32, num_heads=8, num_layers=1)
        trajectory = torch.zeros(1, 5, 1, 2)
        trajectory[0, 4, 0, 1] = 0.5  # pi/2 radians after denormalisation
        relative = module.compute_relative_features(
            trajectory, torch.ones(1, 2, dtype=torch.bool)
        )
        self.assertAlmostEqual(relative[0, 0, 0, 1, 4].item(), -math.pi / 2, places=6)
        self.assertAlmostEqual(relative[0, 0, 1, 0, 4].item(), math.pi / 2, places=6)

        scaled = module.compute_relative_features(
            trajectory,
            torch.ones(1, 2, dtype=torch.bool),
            yaw_scale=torch.tensor([2.0]),
        )
        self.assertAlmostEqual(scaled[0, 0, 0, 1, 4].item(), -1.0, places=6)


class InvertiblePrimitiveTest(unittest.TestCase):
    def test_actnorm_legacy_scale_migration_preserves_sign_and_inverse(self):
        module = ActNorm(3).double()
        loc = torch.tensor([[[[0.2]], [[-0.1]], [[0.4]]]], dtype=torch.float64)
        legacy_scale = torch.tensor([[[[-2.0]], [[0.5]], [[3.0]]]], dtype=torch.float64)
        module.load_state_dict(
            {
                "loc": loc,
                "scale": legacy_scale,
                "initialized": torch.tensor(True),
            },
            strict=True,
        )

        x = torch.randn(2, 3, 2, 2, dtype=torch.float64)
        y, _ = module(x)
        torch.testing.assert_close(y, legacy_scale * (x + loc), rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(module.reverse(y), x, rtol=1e-12, atol=1e-12)
        self.assertTrue((module._effective_scale().abs() > 0).all())

    def test_legacy_dense_conv_weight_loads_exactly_in_both_backends(self):
        raw = torch.randn(4, 4)
        legacy_weight, _ = torch.linalg.qr(raw)
        for use_lu in (True, False):
            with self.subTest(lu=use_lu):
                module = InvConv2d(4, lu=use_lu)
                module.load_state_dict(
                    {"weight": legacy_weight.unsqueeze(-1).unsqueeze(-1)}, strict=True
                )
                torch.testing.assert_close(
                    module.weight.squeeze(-1).squeeze(-1),
                    legacy_weight,
                    rtol=1e-5,
                    atol=1e-6,
                )

    def test_context_metadata_encoders_are_legacy_checkpoint_compatible(self):
        source = ContextEncoder(filter_size=32, num_heads=8, history_input_dim=5)
        legacy_state = {
            key: value
            for key, value in source.state_dict().items()
            if not key.startswith(("scene_stats_encoder.", "static_dimensions_encoder."))
        }
        restored = ContextEncoder(filter_size=32, num_heads=8, history_input_dim=5)
        restored.load_state_dict(legacy_state, strict=True)

    def test_lu_and_dense_logdet_match_autograd_jacobian(self):
        torch.manual_seed(12)
        for use_lu in (True, False):
            with self.subTest(lu=use_lu):
                module = InvConv2d(3, lu=use_lu).double()
                flat = torch.randn(3, dtype=torch.float64, requires_grad=True)

                def transform(vector):
                    output, _ = module(vector.reshape(1, 3, 1, 1))
                    return output.reshape(3)

                jacobian = torch.autograd.functional.jacobian(transform, flat)
                _, analytic = module(flat.reshape(1, 3, 1, 1))
                jacobian_logdet = torch.linalg.slogdet(jacobian)[1]
                torch.testing.assert_close(
                    analytic[0], jacobian_logdet, rtol=1e-9, atol=1e-9
                )

                output, _ = module(flat.detach().reshape(1, 3, 1, 1))
                reconstructed = module.reverse(output)
                torch.testing.assert_close(
                    reconstructed.reshape(3), flat.detach(), rtol=1e-10, atol=1e-10
                )

    def test_partial_channel_flow_logdet_matches_valid_subspace_jacobian(self):
        torch.manual_seed(21)
        module = Flow(4, condition_dim=8, affine=True, conv_lu=True).double().train()
        mask = torch.tensor([True, False, True, False]).reshape(1, 4, 1, 1)
        context = {
            "kv": None,
            "kv_mask": None,
            "agent_anchor_xy": torch.zeros(1, 1, 2, dtype=torch.float64),
            "agent_valid_mask": torch.ones(1, 1, dtype=torch.bool),
        }

        # Initialise ActNorm once, then exercise a genuinely non-identity coupling.
        module(torch.randn(1, 4, 1, 1, dtype=torch.float64), context=context, timestep_mask=mask)
        with torch.no_grad():
            for child in module.coupling.modules():
                if isinstance(child, ZeroLinear):
                    child.weight.normal_(0.0, 0.02)
                    child.bias.normal_(0.0, 0.02)

        base = torch.randn(4, dtype=torch.float64)
        valid_index = torch.tensor([0, 2])
        valid_input = base[valid_index].clone().requires_grad_(True)

        def valid_transform(values):
            full = base.scatter(0, valid_index, values).reshape(1, 4, 1, 1)
            output, _ = module(full, context=context, timestep_mask=mask)
            return output.reshape(-1)[valid_index]

        jacobian = torch.autograd.functional.jacobian(valid_transform, valid_input)
        full_input = base.scatter(0, valid_index, valid_input).reshape(1, 4, 1, 1)
        output, analytic = module(full_input, context=context, timestep_mask=mask)
        torch.testing.assert_close(
            analytic[0], torch.linalg.slogdet(jacobian)[1], rtol=1e-8, atol=1e-8
        )
        reconstructed = module.reverse(output, context=context, timestep_mask=mask)
        torch.testing.assert_close(reconstructed, full_input, rtol=1e-9, atol=1e-9)


class ConditionerCorrectnessTest(unittest.TestCase):
    def test_context_converts_normalized_anchor_to_metric_rope_coordinates(self):
        normalized = torch.tensor(
            [[[0.25, -0.10], [-0.50, 0.20]]], dtype=torch.float32
        )
        scene_stats = torch.tensor([[120.0, -35.0, 40.0]], dtype=torch.float32)

        metric = ContextEncoder._metric_rope_anchor(normalized, scene_stats)

        torch.testing.assert_close(metric, normalized * 40.0)
        # The global centre is deliberately not added: RoPE centres the agent
        # coordinates before applying rotations, so only the metric scale is
        # physically relevant.
        self.assertFalse(torch.allclose(metric, normalized + scene_stats[:, None, :2]))

        with self.assertRaisesRegex(ValueError, "finite and positive"):
            ContextEncoder._metric_rope_anchor(
                normalized,
                torch.tensor([[0.0, 0.0, 0.0]]),
            )

    def test_coupling_rope_receives_exogenous_anchor_not_latent_xy(self):
        coupling = AffineCoupling(4, condition_dim=8, filter_size=32).train()
        self.assertFalse(coupling.spatiotemporal_extractor.use_input_coordinates)

        captured = {}
        rope = coupling.spatiotemporal_extractor.spatial_attn_rope
        original_forward = rope.forward

        def capture_forward(features, coordinates, key_padding_mask=None):
            captured["coordinates"] = coordinates.detach().clone()
            return original_forward(features, coordinates, key_padding_mask=key_padding_mask)

        rope.forward = capture_forward
        in_a = torch.randn(1, 2, 3, 2)
        anchor = torch.tensor([[[10.0, -3.0], [4.0, 7.0]]])
        context = {
            "kv": None,
            "kv_mask": None,
            "agent_anchor_xy": anchor,
            "agent_rope_xy": anchor,
            "agent_valid_mask": torch.ones(1, 2, dtype=torch.bool),
        }
        coupling._compute_affine_params(
            in_a,
            torch.ones(1, 2, dtype=torch.bool),
            context,
        )

        expected = anchor.unsqueeze(1).expand(1, 3, 2, 2).reshape(3, 2, 2)
        torch.testing.assert_close(captured["coordinates"], expected)
        self.assertFalse(torch.allclose(captured["coordinates"], in_a[:, :2].permute(0, 2, 3, 1).reshape(3, 2, 2)))

    def test_coupling_rope_projects_each_future_token_in_metric_space(self):
        coupling = AffineCoupling(4, condition_dim=8, filter_size=32).eval()
        captured = {}
        rope = coupling.spatiotemporal_extractor.spatial_attn_rope
        original_forward = rope.forward

        def capture_forward(features, coordinates, key_padding_mask=None):
            captured["coordinates"] = coordinates.detach().clone()
            return original_forward(
                features, coordinates, key_padding_mask=key_padding_mask
            )

        rope.forward = capture_forward
        anchor = torch.tensor([[[10.0, -3.0], [4.0, 7.0]]])
        step = torch.tensor([[[2.0, 0.5], [-1.0, 1.0]]])
        coupling._compute_affine_params(
            torch.randn(1, 2, 3, 2),
            torch.ones(1, 2, dtype=torch.bool),
            {
                "kv": None,
                "kv_mask": None,
                "agent_rope_xy": anchor,
                "agent_rope_step_xy": step,
                "agent_valid_mask": torch.ones(1, 2, dtype=torch.bool),
            },
        )

        offsets = torch.tensor([1.5, 3.5, 5.5]).view(3, 1, 1)
        expected = anchor[0].unsqueeze(0) + offsets * step[0].unsqueeze(0)
        torch.testing.assert_close(captured["coordinates"], expected)
        self.assertAlmostEqual(
            coupling.spatiotemporal_extractor.spatial_attn_rope.m2idx_x.item(),
            2.0 * math.pi / 100.0,
            places=6,
        )

    def test_lane_selection_keeps_history_reachable_distant_lane(self):
        encoder = ContextEncoder(
            filter_size=32, num_heads=8, history_input_dim=5, topk_lanes=1
        )
        lane_features = torch.randn(1, 2, 32)
        map_data = torch.zeros(1, 2, 1, 6)
        map_data[0, 0, 0, :2] = torch.tensor([0.0, 0.2])
        map_data[0, 1, 0, :2] = torch.tensor([3.0, 0.0])
        map_mask = torch.ones(1, 2, 1, dtype=torch.bool)

        _, lane_pad, selected_data, selected_mask = encoder._select_topk_lanes(
            lane_features,
            map_data,
            map_mask,
            agent_xy=torch.zeros(1, 1, 2),
            agent_step_xy=torch.tensor([[[0.1, 0.0]]]),
            agent_valid_mask=torch.ones(1, 1, dtype=torch.bool),
            anchor_valid_mask=torch.ones(1, 1, dtype=torch.bool),
        )

        torch.testing.assert_close(
            selected_data[0, 0, 0, 0, :2], torch.tensor([3.0, 0.0])
        )
        self.assertFalse(lane_pad.any())
        self.assertTrue(selected_mask.all())

    def test_conditional_base_prior_changes_with_context(self):
        block = Block(
            2, condition_dim=8, n_flow=1, split=True,
            affine=True, conv_lu=True, history_input_dim=5,
        ).eval()
        first = block.prior_context_head[0]
        last = block.prior_context_head[-1]
        with torch.no_grad():
            first.weight.zero_()
            first.bias.zero_()
            first.weight.copy_(torch.eye(first.weight.shape[0]))
            last.weight.fill_(0.01)
            last.bias.zero_()

        base_context = {
            "B": 1,
            "V": 1,
            "kv_mask": torch.zeros(1, 2, dtype=torch.bool),
        }
        zero_context = dict(base_context, kv=torch.zeros(1, 2, first.weight.shape[0]))
        one_context = dict(base_context, kv=torch.ones(1, 2, first.weight.shape[0]))
        zero_stats = block._conditional_prior_stats(
            zero_context, time_steps=2, dtype=torch.float32, device=torch.device("cpu")
        )
        one_stats = block._conditional_prior_stats(
            one_context, time_steps=2, dtype=torch.float32, device=torch.device("cpu")
        )

        self.assertEqual(tuple(one_stats.shape), (1, 4, 2, 1))
        self.assertFalse(torch.allclose(zero_stats, one_stats))

    def test_train_mode_full_conditioner_is_deterministic_and_reconstructs(self):
        torch.manual_seed(7)
        model = Glow(
            2,
            condition_dim=8,
            n_flow=1,
            n_block=1,
            affine=True,
            conv_lu=True,
            history_input_dim=5,
        ).train()

        # Exercise a genuinely non-identity affine coupling.
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, ZeroLinear):
                    module.weight.normal_(0.0, 0.01)
                    module.bias.normal_(0.0, 0.01)

        # Every attention path used by context/coupling must be deterministic
        # even while the model is in train mode.
        for module in model.modules():
            if isinstance(module, torch.nn.MultiheadAttention):
                self.assertEqual(float(module.dropout), 0.0)
            if isinstance(module, torch.nn.Dropout):
                self.assertEqual(float(module.p), 0.0)
            if hasattr(module, "dropout_p"):
                self.assertEqual(float(module.dropout_p), 0.0)

        target = torch.randn(1, 2, 4, 2)
        history = torch.randn(1, 5, 3, 2)
        target_mask = torch.ones(1, 4, 2, dtype=torch.bool)
        history_mask = torch.ones(1, 3, 2, dtype=torch.bool)
        agent_mask = torch.ones(1, 2, dtype=torch.bool)
        map_data = torch.zeros(1, 2, 3, 6)
        map_data[..., :2] = torch.randn(1, 2, 3, 2)
        map_mask = torch.ones(1, 2, 3, dtype=torch.bool)

        kwargs = {
            "map_data": map_data,
            "map_mask": map_mask,
            "agent_types": torch.ones(1, 2, dtype=torch.long),
            "history_data": history,
            "target_vehicle_mask": agent_mask,
            "history_vehicle_mask": agent_mask,
            "context_agent_mask": agent_mask,
            "history_timestep_mask": history_mask,
            "timestep_mask": target_mask,
            "scene_stats": torch.tensor([[100.0, -50.0, 20.0]]),
            "static_dimensions": torch.ones(1, 2, 2),
        }

        with torch.no_grad():
            _, _, latents = model(target, **kwargs)
            reconstructed = model.reverse(latents, reconstruct=True, **kwargs)

        torch.testing.assert_close(reconstructed, target, rtol=1e-5, atol=2e-6)

    def test_glow_last_block_respects_non_lu_option(self):
        model = Glow(2, condition_dim=8, n_flow=1, n_block=1, conv_lu=False)
        self.assertFalse(model.blocks[-1].flows[0].invconv.lu)


if __name__ == "__main__":
    unittest.main()
