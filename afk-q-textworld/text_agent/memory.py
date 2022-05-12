import random
import copy
from collections import namedtuple
import numpy as np
import torch
from generic import to_pt


# a snapshot of state to be stored in replay memory
game_transition = namedtuple('transition', ('observation', 'quest', 'verb_indices',
                             'adj_indices', 'noun_indices', 'nodes', 'vocab_mask_ids', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity=100000, priority_fraction=0.0, discount_gamma=1.0, accumulate_reward_from_final=False):
        # prioritized replay memory
        self.priority_fraction = priority_fraction
        self.discount_gamma = discount_gamma
        self.alpha_capacity = int(capacity * priority_fraction)
        self.beta_capacity = capacity - self.alpha_capacity
        self.alpha_memory, self.beta_memory = [], []
        self.accumulate_reward_from_final = accumulate_reward_from_final

    def push(self, is_prior, t):
        """Saves a transition."""
        if self.priority_fraction == 0.0:
            is_prior = False
        trajectory = []
        for i in range(len(t)):
            trajectory.append(game_transition(
                t[i][0], t[i][1], t[i][2], t[i][3], t[i][4], t[i][5], t[i][6], t[i][7]))
        if is_prior:
            self.alpha_memory.append(trajectory)
            if len(self.alpha_memory) > self.alpha_capacity:
                remove_id = np.random.randint(self.alpha_capacity)
                self.alpha_memory = self.alpha_memory[:remove_id] + \
                    self.alpha_memory[remove_id + 1:]
        else:
            self.beta_memory.append(trajectory)
            if len(self.beta_memory) > self.beta_capacity:
                remove_id = np.random.randint(self.beta_capacity)
                self.beta_memory = self.beta_memory[:remove_id] + \
                    self.beta_memory[remove_id + 1:]

    def _get_single_transition(self, n, which_memory):
        if len(which_memory) == 0:
            return None
        assert n > 0
        trajectory_id = np.random.randint(len(which_memory))
        trajectory = which_memory[trajectory_id]

        if len(trajectory) <= n:
            return None
        head = np.random.randint(0, len(trajectory) - n)
        final = len(trajectory) - 1

        # all good
        obs = trajectory[head].observation
        quest = trajectory[head].quest
        verb_indices = trajectory[head].verb_indices
        adj_indices = trajectory[head].adj_indices
        noun_indices = trajectory[head].noun_indices
        nodes = trajectory[head].nodes
        vocab_mask_ids = trajectory[head].vocab_mask_ids
        next_obs = trajectory[head + n].observation
        next_nodes = trajectory[head + n].nodes
        next_vocab_mask_ids = trajectory[head + n].vocab_mask_ids

        # 1 2 [3] 4 5 (6) 7 8 9f
        how_long = final - head + 1 if self.accumulate_reward_from_final else n + 1
        accumulated_rewards = [self.discount_gamma ** i *
                               trajectory[head + i].reward for i in range(how_long)]
        accumulated_rewards = accumulated_rewards[:n + 1]
        reward = torch.sum(torch.stack(accumulated_rewards))

        return (obs, quest, verb_indices, adj_indices, noun_indices, nodes, vocab_mask_ids, reward, next_obs, next_nodes, next_vocab_mask_ids, n)

    def _get_batch(self, n_list, which_memory):
        res = []
        for i in range(len(n_list)):
            output = self._get_single_transition(n_list[i], which_memory)
            if output is None:
                continue
            res.append(output)
        if len(res) == 0:
            return None
        return res

    def get_batch(self, batch_size, multi_step=1):
        from_alpha = min(int(self.priority_fraction *
                         batch_size), len(self.alpha_memory))
        from_beta = min(batch_size - from_alpha, len(self.beta_memory))
        res = []
        if from_alpha == 0:
            res_alpha = None
        else:
            res_alpha = self._get_batch(np.random.randint(
                1, multi_step + 1, size=from_alpha), self.alpha_memory)
        if from_beta == 0:
            res_beta = None
        else:
            res_beta = self._get_batch(np.random.randint(
                1, multi_step + 1, size=from_beta), self.beta_memory)
        if res_alpha is None and res_beta is None:
            return None
        if res_alpha is not None:
            res += res_alpha
        if res_beta is not None:
            res += res_beta
        random.shuffle(res)

        obs_list, quest_list, verb_indices_list, adj_indices_list, noun_indices_list, nodes_list, reward_list, actual_n_list = [
        ], [], [], [], [], [], [], []
        next_obs_list, next_nodes_list = [], []
        verb_mask_ids, adj_mask_ids, noun_mask_ids = [], [], []
        next_verb_mask_ids, next_adj_mask_ids, next_noun_mask_ids = [], [], []

        for item in res:
            obs, quest, verb_indices, adj_indices, noun_indices, nodes, vocab_mask_ids, reward, next_obs, next_nodes, next_vocab_mask_ids, n = item
            obs_list.append(obs)
            quest_list.append(quest)
            verb_indices_list.append(verb_indices)
            adj_indices_list.append(adj_indices)
            noun_indices_list.append(noun_indices)
            nodes_list.append(nodes)
            verb_mask_ids.append(vocab_mask_ids[0])
            adj_mask_ids.append(vocab_mask_ids[1])
            noun_mask_ids.append(vocab_mask_ids[2])
            reward_list.append(reward)
            next_obs_list.append(next_obs)
            next_nodes_list.append(next_nodes)
            next_verb_mask_ids.append(next_vocab_mask_ids[0])
            next_adj_mask_ids.append(next_vocab_mask_ids[1])
            next_noun_mask_ids.append(next_vocab_mask_ids[2])
            actual_n_list.append(n)

        vocab_mask_ids_list = [verb_mask_ids, adj_mask_ids, noun_mask_ids]
        next_vocab_mask_ids_list = [next_verb_mask_ids, next_adj_mask_ids, next_noun_mask_ids]
        verb_indices_list = torch.stack(verb_indices_list, 0)  # batch x 1
        adj_indices_list = torch.stack(adj_indices_list, 0)  # batch x 1
        noun_indices_list = torch.stack(noun_indices_list, 0)  # batch x 1
        reward_list = torch.stack(reward_list, 0)  # batch
        actual_n_list = np.array(actual_n_list)

        return [obs_list, quest_list, verb_indices_list, adj_indices_list, noun_indices_list, nodes_list, vocab_mask_ids_list, reward_list, next_obs_list, next_nodes_list, next_vocab_mask_ids_list, actual_n_list]

    def _get_single_sequence_transition(self, which_memory, sample_history_length, contains_first_step):
        if len(which_memory) == 0:
            return None
        assert sample_history_length > 1
        trajectory_id = np.random.randint(len(which_memory))
        trajectory = which_memory[trajectory_id]

        _padded_trajectory = copy.deepcopy(trajectory)
        trajectory_mask = [1.0 for _ in range(len(_padded_trajectory))]
        if contains_first_step:
            while len(_padded_trajectory) <= sample_history_length:
                _padded_trajectory = _padded_trajectory + \
                    [copy.copy(_padded_trajectory[-1])]
                trajectory_mask.append(0.0)
            head = 0
        else:
            if len(_padded_trajectory) - sample_history_length <= 1:
                return None
            head = np.random.randint(
                1, len(_padded_trajectory) - sample_history_length)
        # tail = head + sample_history_length - 1
        final = len(_padded_trajectory) - 1

        seq_obs, seq_verb_indices, seq_adj_indices, seq_noun_indices, seq_nodes, seq_vocab_mask_ids, seq_reward = [
        ], [], [], [], [], [], []
        seq_next_obs, seq_next_nodes, seq_next_vocab_mask_ids = [], [], []
        quest = _padded_trajectory[head].quest
        for j in range(sample_history_length):
            seq_obs.append(_padded_trajectory[head + j].observation)
            seq_verb_indices.append(_padded_trajectory[head + j].verb_indices)
            seq_adj_indices.append(_padded_trajectory[head + j].adj_indices)
            seq_noun_indices.append(_padded_trajectory[head + j].noun_indices)
            seq_nodes.append(_padded_trajectory[head + j].nodes)
            seq_vocab_mask_ids.append(_padded_trajectory[head + j].vocab_mask_ids)
            seq_next_obs.append(_padded_trajectory[head + j + 1].observation)
            seq_next_nodes.append(_padded_trajectory[head + j + 1].nodes)
            seq_next_vocab_mask_ids.append(
                _padded_trajectory[head + j + 1].vocab_mask_ids)

            how_long = final - (head + j) + \
                1 if self.accumulate_reward_from_final else 1
            accumulated_rewards = [self.discount_gamma ** i *
                                   _padded_trajectory[head + j + i].reward for i in range(how_long)]
            accumulated_rewards = accumulated_rewards[:1]
            reward = torch.sum(torch.stack(accumulated_rewards))
            seq_reward.append(reward)

        trajectory_mask = trajectory_mask[:sample_history_length]

        return [seq_obs, seq_verb_indices, seq_adj_indices, seq_noun_indices, seq_nodes, seq_vocab_mask_ids, seq_reward,
                seq_next_obs, seq_next_nodes, seq_next_vocab_mask_ids, quest, trajectory_mask]

    def _get_batch_of_sequences(self, which_memory, batch_size, sample_history_length, contains_first_step):
        assert sample_history_length > 1

        obs, quest, verb_indices, adj_indices, noun_indices, nodes, vocab_mask_ids, reward = [
        ], [], [], [], [], [], [], []
        next_obs, next_nodes, next_vocab_mask_ids = [], [], []
        trajectory_mask = []
        for _ in range(sample_history_length):
            obs.append([])
            verb_indices.append([])
            adj_indices.append([])
            noun_indices.append([])
            nodes.append([])
            vocab_mask_ids.append([])
            reward.append([])
            next_obs.append([])
            next_nodes.append([])
            next_vocab_mask_ids.append([])
            trajectory_mask.append([])

        for _ in range(batch_size):
            t = self._get_single_sequence_transition(
                which_memory, sample_history_length, contains_first_step)
            if t is None:
                continue
            quest.append(t[10])
            for step in range(sample_history_length):
                obs[step].append(t[0][step])
                verb_indices[step].append(t[1][step])
                adj_indices[step].append(t[2][step])
                noun_indices[step].append(t[3][step])
                nodes[step].append(t[4][step])
                vocab_mask_ids[step].append(t[5][step])
                reward[step].append(t[6][step])
                next_obs[step].append(t[7][step])
                next_nodes[step].append(t[8][step])
                next_vocab_mask_ids[step].append(t[9][step])
                trajectory_mask[step].append(t[11][step])

        if len(quest) == 0:
            return None
        return [obs, quest, verb_indices, adj_indices, noun_indices, nodes, vocab_mask_ids, reward,
                next_obs, next_nodes, next_vocab_mask_ids, trajectory_mask]

    def get_batch_of_sequences(self, batch_size, sample_history_length):

        from_alpha = min(int(self.priority_fraction *
                         batch_size), len(self.alpha_memory))
        from_beta = min(batch_size - from_alpha, len(self.beta_memory))

        random_number = np.random.uniform(low=0.0, high=1.0, size=(1,))
        # hard coded here. So 5% of the sampled batches will have first step
        contains_first_step = random_number[0] < 0.2

        if from_alpha == 0:
            res_alpha = None
        else:
            res_alpha = self._get_batch_of_sequences(
                self.alpha_memory, from_alpha, sample_history_length, contains_first_step)
        if from_beta == 0:
            res_beta = None
        else:
            res_beta = self._get_batch_of_sequences(
                self.beta_memory, from_beta, sample_history_length, contains_first_step)
        if res_alpha is None and res_beta is None:
            return None, None

        obs, quest, verb_indices, adj_indices, noun_indices, nodes, vocab_mask_ids, reward, = [
        ], [], [], [], [], [], [], []
        next_obs, next_nodes, next_vocab_mask_ids, trajectory_mask = [], [], [], []
        for _ in range(sample_history_length):
            obs.append([])
            verb_indices.append([])
            adj_indices.append([])
            noun_indices.append([])
            nodes.append([])
            vocab_mask_ids.append([])
            reward.append([])
            next_obs.append([])
            next_nodes.append([])
            next_vocab_mask_ids.append([])
            trajectory_mask.append([])

        if res_alpha is not None:
            __obs, __quest, __verb_indices, __adj_indices, __noun_indices, __nodes, __vocab_mask_ids, __reward, __next_obs, __next_nodes, __next_vocab_mask_ids, __trajectory_mask = res_alpha
            quest += __quest
            for i in range(sample_history_length):
                obs[i] += __obs[i]
                verb_indices[i] += __verb_indices[i]
                adj_indices[i] += __adj_indices[i]
                noun_indices[i] += __noun_indices[i]
                nodes[i] += __nodes[i]
                vocab_mask_ids[i] += __vocab_mask_ids[i]
                reward[i] += __reward[i]
                next_obs[i] += __next_obs[i]
                next_nodes[i] += __next_nodes[i]
                next_vocab_mask_ids[i] += __next_vocab_mask_ids[i]
                trajectory_mask[i] += __trajectory_mask[i]

        if res_beta is not None:
            __obs, __quest, __verb_indices, __adj_indices, __noun_indices, __nodes, __vocab_mask_ids, __reward, __next_obs, __next_nodes, __next_vocab_mask_ids, __trajectory_mask = res_beta
            quest += __quest
            for i in range(sample_history_length):
                obs[i] += __obs[i]
                verb_indices[i] += __verb_indices[i]
                adj_indices[i] += __adj_indices[i]
                noun_indices[i] += __noun_indices[i]
                nodes[i] += __nodes[i]
                vocab_mask_ids[i] += __vocab_mask_ids[i]
                reward[i] += __reward[i]
                next_obs[i] += __next_obs[i]
                next_nodes[i] += __next_nodes[i]
                next_vocab_mask_ids[i] += __next_vocab_mask_ids[i]
                trajectory_mask[i] += __trajectory_mask[i]

        reordered_vocab_mask_ids, reordered_next_vocab_mask_ids = [], []
        for i in range(sample_history_length):

            verb_mask_ids, adj_mask_ids, noun_mask_ids = [], [], []
            next_verb_mask_ids, next_adj_mask_ids, next_noun_mask_ids = [], [], []
            for b in range(len(verb_indices[i])):  # batch size
                # vocab_mask_ids[i]: batch x 3
                verb_mask_ids.append(vocab_mask_ids[i][b][0])
                adj_mask_ids.append(vocab_mask_ids[i][b][1])
                noun_mask_ids.append(vocab_mask_ids[i][b][2])
                next_verb_mask_ids.append(next_vocab_mask_ids[i][b][0])
                next_adj_mask_ids.append(next_vocab_mask_ids[i][b][1])
                next_noun_mask_ids.append(next_vocab_mask_ids[i][b][2])
            reordered_vocab_mask_ids.append([verb_mask_ids, adj_mask_ids, noun_mask_ids])
            reordered_next_vocab_mask_ids.append([next_verb_mask_ids, next_adj_mask_ids, next_noun_mask_ids])

            verb_indices[i] = torch.stack(verb_indices[i], 0)  # batch x 1
            adj_indices[i] = torch.stack(adj_indices[i], 0)  # batch x 1
            noun_indices[i] = torch.stack(noun_indices[i], 0)  # batch x 1
            reward[i] = torch.stack(reward[i], 0)  # batch
            trajectory_mask[i] = to_pt(
                np.array(trajectory_mask[i]), enable_cuda=False, type="float")  # batch

        return [obs, quest, verb_indices, adj_indices, noun_indices, nodes, reordered_vocab_mask_ids, reward,
                next_obs, next_nodes, reordered_next_vocab_mask_ids, trajectory_mask], contains_first_step

    def __len__(self):
        return len(self.alpha_memory) + len(self.beta_memory)
