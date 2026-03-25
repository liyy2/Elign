import unittest

import torch

from verl_diffusion.model.molfm_model import MolFMModel


class _DummyDequantizer:
    def reverse(self, tensor):
        return tensor


class _DummyMolFMCore(torch.nn.Module):
    def __init__(self, drift_value: float = 0.0):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.n_dims = 3
        self.in_node_nf = 6
        self.include_charges = True
        self.num_classes = 5
        self.norm_values = (1.0, 1.0, 1.0)
        self.norm_biases = (None, 0.0, 0.0)
        self.discrete_path = "OT_path"
        self.cat_loss_step = -1
        self.drift_value = float(drift_value)

    def flow_drift(self, t, z, node_mask, edge_mask, context):
        del t, node_mask, edge_mask, context
        return torch.full_like(z, self.drift_value)

    def sample_combined_position_feature_noise(self, n_samples, n_nodes, node_mask):
        del node_mask
        return torch.zeros(n_samples, n_nodes, self.n_dims + self.in_node_nf)

    def sample_p_xh_given_z0(self, dequantizer, z0, node_mask):
        del dequantizer
        x = z0[:, :, : self.n_dims]
        h = {
            "categorical": z0[:, :, self.n_dims : self.n_dims + self.num_classes] * node_mask,
            "integer": z0[:, :, -1:] * node_mask,
        }
        return x, h


class TestMolFMModel(unittest.TestCase):
    def test_prefix_steps_stay_deterministic_and_suffix_steps_use_sde_sigma(self):
        core = _DummyMolFMCore()
        model = MolFMModel(
            core,
            _DummyDequantizer(),
            config=type("Cfg", (), {"diffusion_steps": 4})(),
            policy_config={
                "time_step": 4,
                "sde_mode": "constant",
                "policy_start_idx": 2,
                "sde_window_size": 1,
                "sde_noise_scale": 0.5,
                "sde_coordinate_noise_scale": 0.5,
                "sde_feature_noise_scale": 0.25,
            },
        )

        zt = torch.zeros(1, 2, 9)
        node_mask = torch.ones(1, 2, 1)
        edge_mask = torch.ones(1, 4, 1)

        # Step index 0 (s = 3/4) is before the stochastic suffix.
        prefix_s = torch.tensor([[0.75]])
        prefix_t = torch.tensor([[1.0]])
        zs, log_p, mu, sigma, _ = model.sample_p_zs_given_zt(
            prefix_s,
            prefix_t,
            zt,
            node_mask,
            edge_mask,
        )
        self.assertTrue(torch.allclose(zs, mu))
        self.assertTrue(torch.allclose(log_p, torch.zeros_like(log_p)))
        self.assertTrue(torch.allclose(sigma, torch.zeros_like(sigma)))

        # Step index 2 (s = 1/4) is inside the stochastic suffix.
        active_s = torch.tensor([[0.25]])
        active_t = torch.tensor([[0.50]])
        prev_sample = torch.zeros_like(zt)
        _, active_log_p, _, active_sigma, _ = model.sample_p_zs_given_zt(
            active_s,
            active_t,
            zt,
            node_mask,
            edge_mask,
            prev_sample=prev_sample,
        )
        self.assertTrue(torch.isfinite(active_log_p).all())
        self.assertTrue(torch.allclose(active_sigma[0, 0, :3], torch.full((3,), 0.25)))
        self.assertTrue(torch.allclose(active_sigma[0, 0, 3:], torch.full((6,), 0.125)))

        # The next step should be deterministic again because the SDE window size is 1.
        inactive_s = torch.tensor([[0.0]])
        inactive_t = torch.tensor([[0.25]])
        _, inactive_log_p, inactive_mu, inactive_sigma, _ = model.sample_p_zs_given_zt(
            inactive_s,
            inactive_t,
            zt,
            node_mask,
            edge_mask,
        )
        self.assertTrue(torch.allclose(inactive_log_p, torch.zeros_like(inactive_log_p)))
        self.assertTrue(torch.allclose(inactive_sigma, torch.zeros_like(inactive_sigma)))
        self.assertTrue(torch.allclose(inactive_mu, zt))

    def test_sigma_corrected_coord_only_uses_corrected_coordinate_mean(self):
        core = _DummyMolFMCore(drift_value=1.0)
        model = MolFMModel(
            core,
            _DummyDequantizer(),
            config=type("Cfg", (), {"diffusion_steps": 4})(),
            policy_config={
                "time_step": 4,
                "sde_mode": "sigma_corrected_coord_only",
                "policy_start_idx": 2,
                "sde_window_size": 1,
                "sde_coordinate_noise_scale": 0.2,
                "sde_feature_noise_scale": 0.3,
                "sde_min_sigma": 1e-6,
            },
        )

        zt = torch.zeros(1, 2, 9)
        node_mask = torch.ones(1, 2, 1)
        edge_mask = torch.ones(1, 4, 1)

        active_s = torch.tensor([[0.25]])
        active_t = torch.tensor([[0.50]])
        prev_sample = torch.zeros_like(zt)
        _, active_log_p, active_mu, active_sigma, z0_pred = model.sample_p_zs_given_zt(
            active_s,
            active_t,
            zt,
            node_mask,
            edge_mask,
            prev_sample=prev_sample,
        )

        # Coordinates use the sigma-corrected mean; features remain deterministic ODE updates.
        self.assertTrue(torch.allclose(active_mu[0, 0, :3], torch.full((3,), -0.255), atol=1e-6))
        self.assertTrue(torch.allclose(active_mu[0, 0, 3:], torch.full((6,), -0.25), atol=1e-6))
        self.assertTrue(torch.allclose(active_sigma[0, 0, :3], torch.full((3,), 0.1), atol=1e-6))
        self.assertTrue(torch.allclose(active_sigma[0, 0, 3:], torch.zeros(6), atol=1e-6))

        # z0_pred exposes the x0 estimate on coordinates for the corrected path.
        self.assertTrue(torch.allclose(z0_pred[0, 0, :3], torch.full((3,), -0.5), atol=1e-6))
        self.assertTrue(torch.allclose(z0_pred[0, 0, 3:], torch.full((6,), -0.25), atol=1e-6))

        # Log-prob remains finite even though feature dimensions are deterministic.
        self.assertTrue(torch.isfinite(active_log_p).all())
        self.assertTrue(torch.all(active_log_p < 0))

    def test_sigma_corrected_hb_corrects_feature_mean_and_sigma(self):
        core = _DummyMolFMCore(drift_value=1.0)
        model = MolFMModel(
            core,
            _DummyDequantizer(),
            config=type("Cfg", (), {"diffusion_steps": 4})(),
            policy_config={
                "time_step": 4,
                "sde_mode": "sigma_corrected_hb",
                "policy_start_idx": 2,
                "sde_window_size": 1,
                "sde_coordinate_noise_scale": 0.2,
                "sde_feature_noise_scale": 0.4,
                "sde_min_sigma": 1e-6,
            },
        )

        zt = torch.zeros(1, 2, 9)
        node_mask = torch.ones(1, 2, 1)
        edge_mask = torch.ones(1, 4, 1)

        active_s = torch.tensor([[0.25]])
        active_t = torch.tensor([[0.50]])
        prev_sample = torch.zeros_like(zt)
        _, active_log_p, active_mu, active_sigma, z0_pred = model.sample_p_zs_given_zt(
            active_s,
            active_t,
            zt,
            node_mask,
            edge_mask,
            prev_sample=prev_sample,
        )

        self.assertTrue(torch.allclose(active_mu[0, 0, :3], torch.full((3,), -0.255), atol=1e-6))
        self.assertTrue(torch.allclose(active_mu[0, 0, 3:], torch.full((6,), -0.27), atol=1e-6))
        self.assertTrue(torch.allclose(active_sigma[0, 0, :3], torch.full((3,), 0.1), atol=1e-6))
        self.assertTrue(torch.allclose(active_sigma[0, 0, 3:], torch.full((6,), 0.2), atol=1e-6))

        self.assertTrue(torch.allclose(z0_pred[0, 0, :3], torch.full((3,), -0.5), atol=1e-6))
        self.assertTrue(torch.allclose(z0_pred[0, 0, 3:], torch.full((6,), -0.5), atol=1e-6))
        self.assertTrue(torch.isfinite(active_log_p).all())
        self.assertTrue(torch.all(active_log_p < 0))


if __name__ == "__main__":
    unittest.main()
