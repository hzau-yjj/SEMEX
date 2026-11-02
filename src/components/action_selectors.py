import torch as th
from torch.distributions import Categorical
from .epsilon_schedules import DecayThenFlatSchedule
import numpy as np

REGISTRY = {}


class MultinomialActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0.0] = 0.0

        self.epsilon = self.schedule.eval(t_env)

        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            picked_actions = Categorical(masked_policies).sample().long()

        return picked_actions


REGISTRY["multinomial"] = MultinomialActionSelector


class EpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = 0.0

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()

        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        return picked_actions


REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector


class SQUABSWActionSelector():

    def __init__(self, args):
        self.args = args
        self.window_size = getattr(args, "squab_window_size", 1000)
        self.beta = getattr(args, "squab_beta", 1.0)
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.codebook_size = getattr(args, "vqvae_codebook_size", 64)

        self.count_table = np.zeros((self.n_agents, self.n_actions, self.codebook_size), dtype=np.int32)
        self.window_ptr = 0
        self.window_full = False

        self.history_buffer = np.zeros((self.window_size, self.n_agents, 2), dtype=np.int32)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False, semantic_codes=None):
        batch_size = agent_inputs.shape[0]

        # mask unavailable actions
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")

        if test_mode or semantic_codes is None:
            picked_actions = masked_q_values.max(dim=2)[1]
        else:
            exploration_bonus = self._compute_bonus_batch(semantic_codes)

            ucb_values = masked_q_values + exploration_bonus
            ucb_values[avail_actions == 0.0] = -float("inf")
            picked_actions = ucb_values.max(dim=2)[1]

            self._update_window_batch(picked_actions.cpu().numpy(), semantic_codes.cpu().numpy())

        return picked_actions

    def _compute_bonus_batch(self, semantic_codes):
        batch_size = semantic_codes.shape[0]
        device = semantic_codes.device

        codes_np = semantic_codes.cpu().numpy()  # [batch_size, n_agents, n_actions]

        counts = np.zeros((batch_size, self.n_agents, self.n_actions), dtype=np.float32)

        for b in range(batch_size):
            for i in range(self.n_agents):
                for a in range(self.n_actions):
                    action_semantic_code = codes_np[b, i, a]
                    counts[b, i, a] = self.count_table[i, a, action_semantic_code]

        bonus = self.beta / np.sqrt(counts + 1)

        return th.from_numpy(bonus).to(device)

    def _update_window_batch(self, actions_batch, semantic_codes_batch):

        batch_size = actions_batch.shape[0]

        for b in range(batch_size):
            if self.window_full:
                old_actions = self.history_buffer[self.window_ptr, :, 0]
                old_codes = self.history_buffer[self.window_ptr, :, 1]
                for i in range(self.n_agents):
                    self.count_table[i, old_actions[i], old_codes[i]] -= 1


            current_actions = actions_batch[b]  # [n_agents]
            current_codes = np.zeros(self.n_agents, dtype=np.int32)

            for i in range(self.n_agents):
                action = current_actions[i]
                current_codes[i] = semantic_codes_batch[b, i, action]

            self.history_buffer[self.window_ptr, :, 0] = current_actions
            self.history_buffer[self.window_ptr, :, 1] = current_codes

            for i in range(self.n_agents):
                action = current_actions[i]
                code = current_codes[i]
                self.count_table[i, action, code] += 1

            # 更新指针
            self.window_ptr = (self.window_ptr + 1) % self.window_size
            if self.window_ptr == 0:
                self.window_full = True

    def reset(self):
        self.count_table.fill(0)
        self.history_buffer.fill(0)
        self.window_ptr = 0
        self.window_full = False


REGISTRY["squab_sw"] = SQUABSWActionSelector