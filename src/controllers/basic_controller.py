from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VQVAE(nn.Module):


    def __init__(self, obs_shape, n_actions, codebook_size, embedding_dim, beta_commit=0.25, ema_decay=0.95,
                 decorr_weight=0.01):
        super(VQVAE, self).__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.beta_commit = beta_commit
        self.decorr_weight = decorr_weight
        self.n_actions = n_actions


        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )


        self.conditional_encoder = nn.Sequential(
            nn.Linear(32 + n_actions, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )

        # Codebook for action semantics
        self.codebook = nn.Embedding(codebook_size, embedding_dim)
        self.codebook.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)


        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        self._forward_count = 0
        self._reset_interval = 100

    def _compute_decorrelation_loss(self):
        prototypes = self.codebook.weight
        p_norm = F.normalize(prototypes, p=2, dim=1)
        cosine_sim = th.mm(p_norm, p_norm.t())
        identity = th.eye(self.codebook_size, device=prototypes.device)
        diff = cosine_sim - identity
        decorr_loss = diff.pow(2).sum() / (self.codebook_size * (self.codebook_size - 1))
        return decorr_loss

    def encode(self, obs, actions):

        obs_encoded = self.obs_encoder(obs)

        if actions.dim() == 1 or actions.shape[-1] != self.n_actions:
            actions_onehot = F.one_hot(actions.long(), num_classes=self.n_actions).float()
        else:
            actions_onehot = actions

        conditional_input = th.cat([obs_encoded, actions_onehot], dim=-1)
        z_e = self.conditional_encoder(conditional_input)

        distances = (z_e.unsqueeze(1) - self.codebook.weight.unsqueeze(0)).pow(2).sum(-1)
        codes = th.argmin(distances, dim=1)

        return codes

    def encode_all_actions(self, obs):

        batch_size = obs.shape[0]

        obs_encoded = self.obs_encoder(obs)

        obs_encoded_expanded = obs_encoded.unsqueeze(1).expand(
            batch_size, self.n_actions, -1
        )

        all_actions_onehot = th.eye(self.n_actions, device=obs.device).unsqueeze(0).expand(
            batch_size, -1, -1
        )

        conditional_input = th.cat([obs_encoded_expanded, all_actions_onehot], dim=-1)
        conditional_input_flat = conditional_input.reshape(batch_size * self.n_actions, -1)

        z_e = self.conditional_encoder(conditional_input_flat)

        distances = (z_e.unsqueeze(1) - self.codebook.weight.unsqueeze(0)).pow(2).sum(-1)
        codes = th.argmin(distances, dim=1)

        codes = codes.reshape(batch_size, self.n_actions)
        return codes

    def forward(self, obs, actions):
        obs_encoded = self.obs_encoder(obs)

        if actions.dim() == 1 or actions.shape[-1] != self.n_actions:
            actions_onehot = F.one_hot(actions.long(), num_classes=self.n_actions).float()
        else:
            actions_onehot = actions

        conditional_input = th.cat([obs_encoded, actions_onehot], dim=-1)
        z_e = self.conditional_encoder(conditional_input)

        distances = (z_e.unsqueeze(1) - self.codebook.weight.unsqueeze(0)).pow(2).sum(-1)
        codes = th.argmin(distances, dim=1)
        z_q = self.codebook(codes)

        obs_recon = self.decoder(z_q)


        recon_loss = F.mse_loss(obs_recon, obs_encoded.detach())
        vq_loss = F.mse_loss(z_q.detach(), z_e)
        commit_loss = F.mse_loss(z_q, z_e.detach())


        decorr_loss = self._compute_decorrelation_loss()

        loss = recon_loss + vq_loss + self.beta_commit * commit_loss + self.decorr_weight * decorr_loss


        if self.training:
            self._forward_count += 1
            if self._forward_count % self._reset_interval == 0:
                self._reset_dead_codes(codes)

        return z_q, codes, loss

    def _reset_dead_codes(self, codes):
        with th.no_grad():
            used = th.unique(codes)
            all_codes = th.arange(self.codebook_size, device=codes.device)
            dead_mask = ~th.isin(all_codes, used)
            dead_indices = th.where(dead_mask)[0]
            alive_indices = th.where(~dead_mask)[0]

            if len(dead_indices) == 0 or len(alive_indices) == 0:
                return

            for idx in dead_indices:
                donor = alive_indices[th.randint(len(alive_indices), (1,))]
                self.codebook.weight.data[idx] = (
                    self.codebook.weight.data[donor]
                    + th.randn(self.embedding_dim, device=codes.device) * 0.01
                )


# This multi-agent controller shares parameters between agents
class BasicMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None

        # Conditional VQ-VAE for action semantic encoding
        self.use_vqvae = getattr(args, "use_vqvae", False)
        if self.use_vqvae:
            obs_shape = scheme["obs"]["vshape"]
            n_actions = args.n_actions
            codebook_size = getattr(args, "vqvae_codebook_size", 32)
            embedding_dim = getattr(args, "vqvae_embedding_dim", 32)
            beta_commit = getattr(args, "vqvae_beta_commit", 0.25)
            ema_decay = getattr(args, "vqvae_ema_decay", 0.95)
            decorr_weight = getattr(args, "vqvae_decorr_weight", 0.01)

            self.vqvae = VQVAE(obs_shape, n_actions, codebook_size, embedding_dim, beta_commit, ema_decay,
                               decorr_weight)

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)

        semantic_codes = None
        if self.use_vqvae and hasattr(self.action_selector, 'select_action'):
            obs = ep_batch["obs"][:, t_ep]
            batch_size = obs.shape[0]
            obs_flat = obs.reshape(batch_size * self.n_agents, -1)

            with th.no_grad():
                self.vqvae.eval()
                action_semantic_codes = self.vqvae.encode_all_actions(obs_flat)
                self.vqvae.train()

            semantic_codes = action_semantic_codes.reshape(batch_size, self.n_agents, self.args.n_actions)

        if hasattr(self.action_selector,
                   'select_action') and 'semantic_codes' in self.action_selector.select_action.__code__.co_varnames:
            chosen_actions = self.action_selector.select_action(
                agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode,
                semantic_codes=semantic_codes[bs] if semantic_codes is not None else None
            )
        else:
            chosen_actions = self.action_selector.select_action(
                agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode
            )

        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                              + th.ones_like(agent_outs) * self.action_selector.epsilon / epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)

    def parameters(self):
        return self.agent.parameters()

    def vqvae_parameters(self):
        if self.use_vqvae:
            return self.vqvae.parameters()
        return []

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())
        if self.use_vqvae and hasattr(other_mac, 'vqvae'):
            self.vqvae.load_state_dict(other_mac.vqvae.state_dict())

    def cuda(self):
        self.agent.cuda()
        if self.use_vqvae:
            self.vqvae.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))
        if self.use_vqvae:
            th.save(self.vqvae.state_dict(), "{}/vqvae.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        if self.use_vqvae:
            self.vqvae.load_state_dict(th.load("{}/vqvae.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape