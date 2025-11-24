import torch
from tqdm import tqdm as tq
from .base import BaseModel
from edm_source.qm9.models import get_model
from edm_source.equivariant_diffusion.en_diffusion import EnVariationalDiffusion
from torch.nn import functional as F
from verl_diffusion.utils.math import policy_step_logprob

class EDMModel(BaseModel, EnVariationalDiffusion):
    def __init__(self, model, config):
        # Initialize EnVariationalDiffusion first since it's a nn.Module
        # Get configuration from the existing model
        model_config = {
            'dynamics': model.dynamics,
            'in_node_nf': model.in_node_nf,
            'n_dims': 3,
            'timesteps': config.diffusion_steps,
            'noise_schedule': config.diffusion_noise_schedule,  # Default value from the example
            'noise_precision': config.diffusion_noise_precision,     # Default value
            'loss_type': config.diffusion_loss_type,          # Default value
            'norm_values': config.normalize_factors,           # Default value
            'include_charges': config.include_charges      # Default value
        }
        
        # Initialize EnVariationalDiffusion with the configuration
        EnVariationalDiffusion.__init__(
            self,
            dynamics=model_config['dynamics'],
            in_node_nf=model_config['in_node_nf'],
            n_dims=model_config['n_dims'],
            timesteps=model_config['timesteps'],
            noise_schedule=model_config['noise_schedule'],
            noise_precision=model_config['noise_precision'],
            loss_type=model_config['loss_type'],
            norm_values=model_config['norm_values'],
            include_charges=model_config['include_charges']
        )

        # Initialize BaseModel
        BaseModel.__init__(self)
        # Store the model after parent classes are initialized
        self.model = model
        self.config = config
        
    def get_mask(self, nodesxsample, batch_size, max_n_nodes):
        """
        Generate node and edge masks based on the number of nodes
        
        Args:
            nodesxsample: Number of nodes per sample
            batch_size: Batch size
            max_n_nodes: Maximum number of nodes
            
        Returns:
            tuple: (node_mask, edge_mask)
        """
        # Create node mask
        node_mask = torch.zeros(batch_size, max_n_nodes)
        for i in range(batch_size):
            node_mask[i, 0:nodesxsample[i]] = 1
        
        # Create edge mask
        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        edge_mask *= diag_mask
        edge_mask = edge_mask.view(batch_size * max_n_nodes * max_n_nodes, 1)
        node_mask = node_mask.unsqueeze(2)
        
        return node_mask, edge_mask
    
    def _assert_mean_zero_with_mask(self, x, node_mask, eps=1e-10):
        self._assert_correctly_masked(x, node_mask)
        largest_value = x.abs().max().item()
        error = torch.sum(x, dim=1, keepdim=True).abs().max().item()
        rel_error = error / (largest_value + eps)
        assert rel_error < 1e-2, f'Mean is not zero, relative_error {rel_error}'
        
    def _assert_correctly_masked(self, variable, node_mask):
        assert (variable * (1 - node_mask)).abs().max().item() < 1e-4, \
            'Variables not masked properly.'
            
    def _remove_mean_with_mask(self, x, node_mask):
        masked_max_abs_value = (x * (1 - node_mask)).abs().sum().item()
        assert masked_max_abs_value < 1e-5, f'Error {masked_max_abs_value} too high'
        N = node_mask.sum(1, keepdims=True)

        mean = torch.sum(x, dim=1, keepdim=True) / N
        x = x - mean * node_mask
        return x
    
    def load(self, model_path):
        flow_state_dict = torch.load(model_path, map_location= self.config.device )
        self.model.load_state_dict(flow_state_dict)
        
    @torch.no_grad()
    def sample(
        self,
        n_samples,
        n_nodes,
        node_mask,
        edge_mask,
        context=None,
        fix_noise=False,
        timestep=1000,
        group_index=None,
        share_initial_noise=False,
        skip_prefix=0,
    ):
        """Draw samples from the generative model.

        When ``share_initial_noise`` and ``skip_prefix`` are both provided, the
        method first rolls out a shared trajectory of length ``skip_prefix`` per
        group and only samples fresh noise beyond that boundary. This mirrors the
        grouped-sampling strategy described in DanceGRPO.
        """
        self.T = timestep
        if share_initial_noise and group_index is None:
            raise ValueError("group_index must be provided when share_initial_noise is enabled.")

        if fix_noise:
            # Noise is broadcasted over the batch axis, useful for visualizations.
            z = self.sample_combined_position_feature_noise(1, n_nodes, node_mask)
        else:
            z = self._sample_initial_latents(
                n_samples=n_samples,
                n_nodes=n_nodes,
                node_mask=node_mask,
                group_index=group_index,
                share_initial_noise=share_initial_noise,
            )

        self._assert_mean_zero_with_mask(z[:, :, :self.n_dims], node_mask)

        share_prefix = bool(share_initial_noise and skip_prefix > 0)
        if not share_prefix:
            return self._sample_full_batch(
                z,
                node_mask,
                edge_mask,
                context,
                fix_noise,
                n_samples,
            )

        schedule = list(reversed(range(0, self.T)))
        prefix_steps = min(int(skip_prefix), len(schedule))
        if prefix_steps <= 0:
            return self._sample_full_batch(
                z,
                node_mask,
                edge_mask,
                context,
                fix_noise,
                n_samples,
            )

        if edge_mask.dim() == 2:
            # Restore per-sample edge mask layout: [batch, n_nodes, n_nodes, 1]
            edge_mask = edge_mask.view(n_samples, n_nodes, n_nodes, -1)

        group_index = group_index.view(-1)
        # Each entry in ``group_index`` identifies which prompts share a prefix.
        unique_groups = torch.unique(group_index)

        context_slices = None
        if context is not None:
            # Pre-slice the conditioning payload so each group member receives its
            # own prompt features without repeatedly indexing deep structures.
            context_slices = [self._slice_context(context, idx) for idx in range(n_samples)]

        # Reshape the batch into (num_groups, group_size, ...) so the references and
        # their branched continuations can be driven in parallel.
        group_ids, group_counts = torch.unique(group_index, sorted=True, return_counts=True)
        if group_counts.numel() == 0:
            raise ValueError("No group indices provided for shared-prefix sampling.")
        if not torch.all(group_counts == group_counts[0]):
            raise ValueError("All shared-prefix groups must have the same number of members.")

        group_size = int(group_counts[0].item())
        group_members = []
        for group_id in group_ids:
            member_indices = (group_index == group_id).nonzero(as_tuple=True)[0].sort().values
            if member_indices.numel() != group_size:
                raise ValueError("Group membership mismatch while stacking shared-prefix batches.")
            group_members.append(member_indices)
        group_members = torch.stack(group_members, dim=0).to(z.device)

        num_groups = group_members.shape[0]
        ref_indices = group_members[:, 0]
        branch_indices = group_members[:, 1:] if group_size > 1 else None

        grouped_z = z[group_members]
        grouped_node_mask = node_mask[group_members]
        grouped_edge_mask = edge_mask[group_members]

        ref_z = grouped_z[:, 0]
        ref_node_mask = grouped_node_mask[:, 0]
        ref_edge_mask = grouped_edge_mask[:, 0]

        if group_size > 1:
            expected_mask = ref_node_mask.unsqueeze(1).expand_as(grouped_node_mask[:, 1:])
            if not torch.equal(grouped_node_mask[:, 1:], expected_mask):
                raise ValueError("All members of a shared prefix group must share the same node mask.")

        ref_context = None
        if context_slices is not None:
            ref_context_slices = [context_slices[idx] for idx in ref_indices.tolist()]
            ref_context = self._collate_context(ref_context_slices)

        ref_result = self._rollout_member(
            initial_z=ref_z,
            node_mask=ref_node_mask,
            edge_mask=ref_edge_mask,
            context=ref_context,
            fix_noise=fix_noise,
            schedule=schedule,
            start_idx=0,
            prefix_cache=None,
        )

        results = [None for _ in range(n_samples)]
        ref_split = self._split_member_results(ref_result)
        for idx_tensor, member_result in zip(ref_indices.tolist(), ref_split):
            results[idx_tensor] = member_result

        if group_size > 1:
            # Prepare the cached prefix once per group, then tile it across members.
            prefix_cache = self._extract_batched_prefix_cache(ref_result, prefix_steps)
            branch_prefix_cache = self._repeat_prefix_cache(prefix_cache, repeats=group_size - 1)

            # Flatten the remaining members so the tail of every group rolls out together.
            branch_z = grouped_z[:, 1:].reshape(-1, *grouped_z.shape[2:])
            branch_node_mask = grouped_node_mask[:, 1:].reshape(-1, *grouped_node_mask.shape[2:])
            branch_edge_mask = grouped_edge_mask[:, 1:].reshape(-1, *grouped_edge_mask.shape[2:])

            branch_context = None
            if context_slices is not None:
                branch_index_list = branch_indices.reshape(-1).tolist()
                branch_context_slices = [context_slices[idx] for idx in branch_index_list]
                branch_context = self._collate_context(branch_context_slices)

            branch_result = self._rollout_member(
                initial_z=branch_z,
                node_mask=branch_node_mask,
                edge_mask=branch_edge_mask,
                context=branch_context,
                fix_noise=fix_noise,
                schedule=schedule,
                start_idx=prefix_steps,
                prefix_cache=branch_prefix_cache,
            )

            branch_split = self._split_member_results(branch_result)
            for idx_tensor, member_result in zip(branch_indices.reshape(-1).tolist(), branch_split):
                results[idx_tensor] = member_result

        latents, logps, timesteps, mus, sigmas, x, h, z0_preds = self._stack_member_results(results)

        self._assert_mean_zero_with_mask(x, node_mask)

        max_cog = torch.sum(x, dim=1, keepdim=True).abs().max().item()
        if max_cog > 5e-2:
            print(f'Warning cog drift with error {max_cog:.3f}. Projecting '
                  f'the positions down.')
            x = self._remove_mean_with_mask(x, node_mask)

        return x, h, latents, logps, timesteps, mus, sigmas, z0_preds
    
    def sample_p_xh_given_z0(self, z0, node_mask, edge_mask, context = None, fix_noise=False, prev_sample=None):
        """Samples x ~ p(x|z0)."""
        
        zeros = torch.zeros(size=(z0.size(0), 1), device=z0.device)
        gamma_0 = self.gamma(zeros)
        # Computes sqrt(sigma_0^2 / alpha_0^2)

        sigma_x = self.SNR(-0.5 * gamma_0).unsqueeze(1)
        net_out = self.phi(z0, zeros, node_mask, edge_mask, context)

        # Compute mu for p(zs | zt).
        mu_x = self.compute_x_pred(net_out, z0, gamma_0)

        xh = self.sample_normal(mu=mu_x, sigma=sigma_x, node_mask=node_mask, fix_noise=fix_noise)
        if prev_sample != None :
            # import pdb; pdb.set_trace()
            log_p = self.compute_log_p_zs_given_zt(prev_sample, mu_x, sigma_x, node_mask = node_mask)
        else:
            log_p = self.compute_log_p_zs_given_zt(xh, mu_x, sigma_x, node_mask = node_mask)
        x = xh[:, :, :self.n_dims]

        h_int = z0[:, :, -1:] if self.include_charges else torch.zeros(0).to(z0.device)
        x, h_cat, h_int = self.unnormalize(x, z0[:, :, self.n_dims:-1], h_int, node_mask)

        h_cat = F.one_hot(torch.argmax(h_cat, dim=2), self.num_classes) * node_mask
        h_int = torch.round(h_int).long() * node_mask
        h = {'integer': h_int, 'categorical': h_cat}

        return x, h, mu_x, sigma_x.squeeze(-1), log_p, zeros, xh

    def compute_log_p_zs_given_zt(self, x, mu, sigma, node_mask=None):
        """Compute per-sample log-probability of zs conditioned on zt."""
        if node_mask is None:
            mask = torch.ones_like(x[..., :1], dtype=x.dtype, device=x.device)
        else:
            mask = node_mask.to(dtype=x.dtype, device=x.device)
        return policy_step_logprob(x, mu, sigma, mask)

    def sample_p_zs_given_zt(self, s, t, zt, node_mask, edge_mask, context=None, fix_noise=False, prev_sample=None):
        """Samples from zs ~ p(zs | zt). """
    
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = \
            self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt)

        sigma_s = self.sigma(gamma_s, target_tensor=zt)
        sigma_t = self.sigma(gamma_t, target_tensor=zt)
        # Neural net prediction 
        # zt shape number of prompts \times num nodes \times 9
        # z number of prompts \times 1
        # node_mask number of prompts \times num nodes \times 9
        # edge mask is flattened across batch and 
        eps_t = self.phi(zt, t, node_mask, edge_mask, context)

        # Compute z0 prediction
        gamma_t = self.gamma(t)
        z0_pred = self.compute_x_pred(eps_t, zt, gamma_t)

        # Compute mu for p(zs | zt).
        self._assert_mean_zero_with_mask(zt[:, :, :self.n_dims], node_mask)
        self._assert_mean_zero_with_mask(eps_t[:, :, :self.n_dims], node_mask)
        mu = zt / alpha_t_given_s - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_t

        # Compute sigma for p(zs | zt).
        sigma = sigma_t_given_s * sigma_s / sigma_t

        # Sample zs given the paramters derived from zt.
        zs = self.sample_normal(mu, sigma, node_mask, fix_noise)
        
        # Project down to avoid numerical runaway of the center of gravity.
        zs = torch.cat(
            [self._remove_mean_with_mask(zs[:, :, :self.n_dims],
                                                   node_mask),
             zs[:, :, self.n_dims:]], dim=2
        )
        
        # compute logp
        if prev_sample is not None:
            log_p = self.compute_log_p_zs_given_zt(prev_sample, mu, sigma, node_mask=node_mask)
        else:
            log_p = self.compute_log_p_zs_given_zt(zs, mu, sigma,  node_mask=node_mask)
            
        return zs, log_p, mu, sigma, z0_pred

    def _sample_initial_latents(self, n_samples, n_nodes, node_mask, group_index, share_initial_noise):
        """Draw the starting latent noise, optionally synchronizing by group.

        When ``share_initial_noise`` is true, grouped prompts reuse the same base
        noise for their shared prefix. The returned tensor already contains the
        broadcasted latents so the sampling loop can branch immediately.
        """
        if share_initial_noise:
            group_index = group_index.to(node_mask.device).view(-1)
            if group_index.shape[0] != n_samples:
                raise ValueError(
                    f"group_index length {group_index.shape[0]} does not match n_samples {n_samples}."
                )
            noise = torch.zeros(
                (n_samples, n_nodes, self.n_dims + self.in_node_nf),
                device=node_mask.device,
                dtype=node_mask.dtype,
            )
            unique_groups = torch.unique(group_index)
            for group_id in unique_groups:
                member_indices = (group_index == group_id).nonzero(as_tuple=True)[0]
                group_node_mask = node_mask[member_indices]
                base_noise = self.sample_combined_position_feature_noise(1, n_nodes, group_node_mask[:1])
                noise[member_indices] = base_noise.expand(member_indices.numel(), -1, -1)
            return noise

        return self.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask)

    def _slice_context(self, context, index):
        """Extract the ``index``-th prompt context while preserving structure.

        The sampler receives per-sample conditioning tensors, dictionaries, or
        sequences. This helper mirrors ``torch.utils.data.default_collate`` in
        reverse so that each branched member sees only its own prompt features.
        """
        if context is None:
            return None
        if isinstance(context, torch.Tensor):
            if context.size(0) == 1:
                return context
            return context[index : index + 1]
        if isinstance(context, dict):
            sliced = {}
            for key, value in context.items():
                if isinstance(value, (torch.Tensor, dict, list, tuple)):
                    sliced[key] = self._slice_context(value, index)
                else:
                    sliced[key] = value
            return sliced
        if isinstance(context, (list, tuple)):
            sliced_elems = []
            for value in context:
                if isinstance(value, (torch.Tensor, dict, list, tuple)):
                    sliced_elems.append(self._slice_context(value, index))
                else:
                    sliced_elems.append(value)
            if isinstance(context, tuple):
                return tuple(sliced_elems)
            return sliced_elems
        return context

    def _collate_context(self, contexts):
        """Stack a list of per-member contexts into a batched structure."""
        if contexts is None or len(contexts) == 0:
            return None

        example = contexts[0]
        if example is None:
            if any(ctx is not None for ctx in contexts):
                raise ValueError("Mixed None/non-None contexts cannot be collated.")
            return None
        if isinstance(example, torch.Tensor):
            return torch.cat(contexts, dim=0)
        if isinstance(example, dict):
            return {key: self._collate_context([ctx[key] for ctx in contexts]) for key in example}
        if isinstance(example, (list, tuple)):
            collated = [self._collate_context([ctx[i] for ctx in contexts]) for i in range(len(example))]
            return tuple(collated) if isinstance(example, tuple) else collated
        if isinstance(example, (int, float, str, bool)):
            if any(ctx != example for ctx in contexts):
                raise ValueError("Scalar contexts must be identical across branched members.")
            return example
        return example

    def _sample_full_batch(self, z, node_mask, edge_mask, context, fix_noise, n_samples):
        latents = []
        logps = []
        timesteps = []
        mus = []
        sigmas = []
        z0_preds = []
        latents.append(z)
        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s in tq(reversed(range(0, self.T)), desc="sampling", leave=False, unit="step"):
            s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T
            z, logp, mu, sigma, z0_pred = self.sample_p_zs_given_zt(
                s_array,
                t_array,
                z,
                node_mask,
                edge_mask,
                context,
                fix_noise=fix_noise,
            )
            latents.append(z)
            logps.append(logp)
            timesteps.append(s)
            mus.append(mu)
            sigmas.append(sigma)
            z0_preds.append(z0_pred)

        # Finally sample p(x, h | z_0).
        x, h, mu, sigma, logp, s, z = self.sample_p_xh_given_z0(
            z, node_mask, edge_mask, context, fix_noise=fix_noise
        )

        latents.append(z)
        logps.append(logp)
        timesteps.append(0)
        mus.append(mu)
        sigmas.append(sigma.unsqueeze(-1))
        z0_preds.append(z)

        self._assert_mean_zero_with_mask(x, node_mask)

        max_cog = torch.sum(x, dim=1, keepdim=True).abs().max().item()
        if max_cog > 5e-2:
            print(
                f'Warning cog drift with error {max_cog:.3f}. Projecting '
                f'the positions down.'
            )
            x = self._remove_mean_with_mask(x, node_mask)

        return x, h, latents, logps, timesteps, mus, sigmas, z0_preds

    def _rollout_member(
        self,
        initial_z,
        node_mask,
        edge_mask,
        context,
        fix_noise,
        schedule,
        start_idx,
        prefix_cache=None,
    ):
        """Simulate one or more members' diffusion trajectories in parallel.

        Args:
            initial_z: Starting latents with a leading batch dimension.
            start_idx: Where to resume in ``schedule`` (0 for reference members,
                ``prefix_steps`` for branched members).
            prefix_cache: Optional cached history from the reference trajectory.
        """
        latents = []
        logps = []
        mus = []
        sigmas = []
        timesteps = []
        z0_preds = []

        batch_size = initial_z.shape[0]

        if prefix_cache is None:
            z = initial_z
            latents.append(z)
        else:
            # Broadcast the cached prefix so every branched member resumes from
            # the exact same latent history.
            latents = [
                frame.expand(batch_size, *frame.shape[1:]).clone()
                for frame in prefix_cache["latents"]
            ]
            logps = [
                frame.expand(batch_size, *frame.shape[1:]).clone()
                for frame in prefix_cache["logps"]
            ]
            mus = [
                frame.expand(batch_size, *frame.shape[1:]).clone()
                for frame in prefix_cache["mus"]
            ]
            sigmas = [
                frame.expand(batch_size, *frame.shape[1:]).clone()
                for frame in prefix_cache["sigmas"]
            ]
            z0_preds = [
                frame.expand(batch_size, *frame.shape[1:]).clone()
                for frame in prefix_cache["z0_preds"]
            ]
            timesteps = list(prefix_cache["timesteps"])
            z = latents[-1]

        # ``sample_p_zs_given_zt`` expects the flattened edge layout used during
        # training, so reshape once outside the timestep loop for efficiency.
        flat_edge_mask = edge_mask.view(edge_mask.shape[0], -1, edge_mask.shape[-1])
        flat_edge_mask = flat_edge_mask.reshape(-1, flat_edge_mask.shape[-1])

        for schedule_idx in range(start_idx, len(schedule)):
            s = schedule[schedule_idx]
            s_array = torch.full((batch_size, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T
            z, logp, mu, sigma, z0_pred = self.sample_p_zs_given_zt(
                s_array,
                t_array,
                z,
                node_mask,
                flat_edge_mask,
                context,
                fix_noise=fix_noise,
            )
            latents.append(z)
            logps.append(logp)
            timesteps.append(s)
            mus.append(mu)
            sigmas.append(sigma)
            z0_preds.append(z0_pred)

        x, h, mu, sigma, logp, s, z = self.sample_p_xh_given_z0(
            z, node_mask, flat_edge_mask, context, fix_noise=fix_noise
        )

        latents.append(z)
        logps.append(logp)
        timesteps.append(0)
        mus.append(mu)
        sigmas.append(sigma.unsqueeze(-1))
        z0_preds.append(z)

        return {
            "latents": latents,
            "logps": logps,
            "mus": mus,
            "sigmas": sigmas,
            "timesteps": timesteps,
            "x": x,
            "h": h,
            "z0_preds": z0_preds,
        }

    def _split_member_results(self, batched_result):
        """Split a batched rollout dictionary back into per-member records."""
        batch_size = batched_result["x"].shape[0]
        member_results = []

        for idx in range(batch_size):
            member_result = {
                "latents": [frame[idx : idx + 1].clone() for frame in batched_result["latents"]],
                "logps": [frame[idx : idx + 1].clone() for frame in batched_result["logps"]],
                "mus": [frame[idx : idx + 1].clone() for frame in batched_result["mus"]],
                "sigmas": [frame[idx : idx + 1].clone() for frame in batched_result["sigmas"]],
                "z0_preds": [frame[idx : idx + 1].clone() for frame in batched_result["z0_preds"]],
                "timesteps": list(batched_result["timesteps"]),
                "x": batched_result["x"][idx : idx + 1].clone(),
                "h": {
                    "categorical": batched_result["h"]["categorical"][idx : idx + 1].clone(),
                    "integer": batched_result["h"]["integer"][idx : idx + 1].clone(),
                },
            }
            member_results.append(member_result)

        return member_results

    def _extract_batched_prefix_cache(self, member_result, prefix_steps):
        """Clone the shared prefix from a batched reference rollout."""
        if prefix_steps <= 0:
            return None
        cache = {
            "latents": [
                frame.clone()
                for frame in member_result["latents"][: prefix_steps + 1]
            ],
            "logps": [
                frame.clone()
                for frame in member_result["logps"][:prefix_steps]
            ],
            "mus": [
                frame.clone()
                for frame in member_result["mus"][:prefix_steps]
            ],
            "sigmas": [
                frame.clone()
                for frame in member_result["sigmas"][:prefix_steps]
            ],
            "z0_preds": [
                frame.clone()
                for frame in member_result["z0_preds"][:prefix_steps]
            ],
            "timesteps": list(member_result["timesteps"][:prefix_steps]),
        }
        return cache

    def _repeat_prefix_cache(self, prefix_cache, repeats):
        """Tile a batched prefix cache across additional group members."""
        if prefix_cache is None or repeats <= 0:
            return prefix_cache

        repeated_cache = {
            "latents": [
                frame.repeat_interleave(repeats, dim=0)
                for frame in prefix_cache["latents"]
            ],
            "logps": [
                frame.repeat_interleave(repeats, dim=0)
                for frame in prefix_cache["logps"]
            ],
            "mus": [
                frame.repeat_interleave(repeats, dim=0)
                for frame in prefix_cache["mus"]
            ],
            "sigmas": [
                frame.repeat_interleave(repeats, dim=0)
                for frame in prefix_cache["sigmas"]
            ],
            "z0_preds": [
                frame.repeat_interleave(repeats, dim=0)
                for frame in prefix_cache["z0_preds"]
            ],
            "timesteps": list(prefix_cache["timesteps"]),
        }
        return repeated_cache

    def _stack_member_results(self, member_results):
        """Stack the per-member dictionaries into batched tensors.

        The return layout mirrors ``_sample_full_batch`` so downstream consumers
        can treat shared-prefix and independent sampling paths identically.
        """
        if any(result is None for result in member_results):
            missing = [idx for idx, result in enumerate(member_results) if result is None]
            raise ValueError(f"Missing diffusion trajectories for sample indices: {missing}")

        num_members = len(member_results)
        num_latent_frames = len(member_results[0]["latents"])
        num_logp_frames = len(member_results[0]["logps"])
        num_z0_frames = len(member_results[0]["z0_preds"])

        latents = []
        logps = []
        mus = []
        sigmas = []
        z0_preds = []

        for frame_idx in range(num_latent_frames):
            frame = torch.cat(
                [member_results[i]["latents"][frame_idx] for i in range(num_members)], dim=0
            )
            latents.append(frame)

        for frame_idx in range(num_logp_frames):
            logp_frame = torch.cat(
                [member_results[i]["logps"][frame_idx] for i in range(num_members)], dim=0
            )
            mu_frame = torch.cat(
                [member_results[i]["mus"][frame_idx] for i in range(num_members)], dim=0
            )
            sigma_frame = torch.cat(
                [member_results[i]["sigmas"][frame_idx] for i in range(num_members)], dim=0
            )
            logps.append(logp_frame)
            mus.append(mu_frame)
            sigmas.append(sigma_frame)
        
        for frame_idx in range(num_z0_frames):
            z0_frame = torch.cat(
                [member_results[i]["z0_preds"][frame_idx] for i in range(num_members)], dim=0
            )
            z0_preds.append(z0_frame)

        timesteps = list(member_results[0]["timesteps"])
        x = torch.cat([member_results[i]["x"] for i in range(num_members)], dim=0)
        h_cat = torch.cat(
            [member_results[i]["h"]["categorical"] for i in range(num_members)], dim=0
        )
        h_int = torch.cat([member_results[i]["h"]["integer"] for i in range(num_members)], dim=0)
        h = {"categorical": h_cat, "integer": h_int}

        return latents, logps, timesteps, mus, sigmas, x, h, z0_preds
