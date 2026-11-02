import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
from torch.optim import RMSprop, Adam


class QLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = Adam(params=self.params, lr=args.lr, eps=args.optim_eps)

        # VQ-VAE optimizer (separate from RL optimizer)
        self.use_vqvae = getattr(args, "use_vqvae", False)
        self.vqvae_train_freq = getattr(args, "vqvae_train_freq", 5)
        self.vqvae_sample_ratio = getattr(args, "vqvae_sample_ratio", 0.5)
        self.vqvae_train_count = 0

        if self.use_vqvae and hasattr(mac, 'vqvae'):
            vqvae_lr = getattr(args, "vqvae_lr", 0.0003)
            self.vqvae_optimiser = Adam(mac.vqvae_parameters(), lr=vqvae_lr)
        else:
            self.vqvae_optimiser = None

        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        vqvae_loss = None
        if self.use_vqvae and self.vqvae_optimiser is not None:
            self.vqvae_train_count += 1
            if self.vqvae_train_count % self.vqvae_train_freq == 0:
                vqvae_loss = self._train_vqvae(batch)

        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean",
                                 (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                                 t_env)

            if vqvae_loss is not None:
                self.logger.log_stat("vqvae_loss", vqvae_loss.item(), t_env)

            self.log_stats_t = t_env

    def _train_vqvae(self, batch: EpisodeBatch):

        obs = batch["obs"][:, :-1]  # [batch, seq_len, n_agents, obs_dim]
        actions = batch["actions"][:, :-1]  # [batch, seq_len, n_agents, 1]
        mask = batch["filled"][:, :-1].float()  # [batch, seq_len]

        batch_size = obs.shape[0]
        seq_length = obs.shape[1]
        n_agents = obs.shape[2]

        total_samples = batch_size * seq_length * n_agents
        n_samples = max(1, int(total_samples * self.vqvae_sample_ratio))

        obs_flat = obs.reshape(batch_size * seq_length * n_agents, -1)
        actions_flat = actions.reshape(batch_size * seq_length * n_agents)
        mask_flat = mask.reshape(batch_size * seq_length, 1).expand(-1, n_agents).reshape(-1)

        valid_indices = th.where(mask_flat > 0)[0]
        if len(valid_indices) == 0:
            return th.tensor(0.0)

        sample_size = min(n_samples, len(valid_indices))
        sampled_indices = valid_indices[th.randperm(len(valid_indices))[:sample_size]]

        obs_sampled = obs_flat[sampled_indices]
        actions_sampled = actions_flat[sampled_indices]

        _, _, vqvae_loss = self.mac.vqvae(obs_sampled, actions_sampled)

        self.vqvae_optimiser.zero_grad()
        vqvae_loss.backward()
        th.nn.utils.clip_grad_norm_(self.mac.vqvae_parameters(), self.args.grad_norm_clip)
        self.vqvae_optimiser.step()

        return vqvae_loss.detach()

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))
        if self.vqvae_optimiser is not None:
            th.save(self.vqvae_optimiser.state_dict(), "{}/vqvae_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
        if self.vqvae_optimiser is not None:
            try:
                self.vqvae_optimiser.load_state_dict(
                    th.load("{}/vqvae_opt.th".format(path), map_location=lambda storage, loc: storage))
            except:
                pass