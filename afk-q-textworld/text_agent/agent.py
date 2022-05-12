import random
import copy
import math
import string
import codecs
from collections import namedtuple
from os.path import join as pjoin

import numpy as np

import torch
import torch.nn.functional as F

import memory
from model import LSTM_DQN
from generic import to_np, to_pt, _words_to_ids, pad_sequences
from generic import max_len, ez_gather_dim_1, tokenize
from layers import masked_softmax, masked_log_softmax
from knowledge_graph import SimpleGraph


class CustomAgent:
    def __init__(self, config):
        self.mode = "train"
        self.config = config
        print(self.config)
        self.load_config()

        self.online_net = LSTM_DQN(
            config=self.config, word_vocab=self.word_vocab)
        self.target_net = LSTM_DQN(
            config=self.config, word_vocab=self.word_vocab)
        self.online_net.train()
        self.target_net.train()
        self.update_target_net()
        for param in self.target_net.parameters():
            param.requires_grad = False

        if self.use_cuda:
            self.online_net.cuda()
            self.target_net.cuda()

        self.kg = SimpleGraph(self.word2id, self.stopwords,
                              self.setting, self.adj_vocab, self.noun_vocab)

        # optimizer
        self.optimizer = torch.optim.Adam(self.online_net.parameters(
        ), lr=self.config['training']['optimizer']['learning_rate'])
        self.clip_grad_norm = self.config['training']['optimizer']['clip_grad_norm']
        # graph input cache
        self.relation_representation_cache = {}

    def load_config(self):
        # word vocab
        self.word_vocab = []
        with codecs.open("./vocabularies/word_vocab.txt", mode='r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                self.word_vocab.append(line.strip())
        self.word2id = {}
        for i, w in enumerate(self.word_vocab):
            self.word2id[w] = i
        # stopwords
        self.stopwords = set()
        with codecs.open("./vocabularies/corenlp_stopwords.txt", mode='r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                self.stopwords.add(line)
        self.stopwords = self.stopwords | set(string.punctuation)

        # adjectives
        self.adj_vocab = []
        with codecs.open("./vocabularies/adjectives.txt", mode='r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                self.adj_vocab.append(line.strip())
        # nouns
        self.noun_vocab = []
        with codecs.open("./vocabularies/entities.txt", mode='r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                self.noun_vocab.append(line.strip())

        self.setting = self.config['general']['setting']
        self.node_capacity = self.config['general']['node_capacity']
        self.kg_expansion_reward = self.config['general']['kg_expansion_reward']
        self.use_negative_reward = self.config['general']['use_negative_reward']

        self.batch_size = self.config['training']['batch_size']
        self.max_nb_steps_per_episode = self.config['training']['max_nb_steps_per_episode']
        self.max_episode = self.config['training']['max_episode']
        self.target_net_update_frequency = self.config['training']['target_net_update_frequency']
        self.learn_start_from_this_episode = self.config['training']['learn_start_from_this_episode']
        self.run_eval = self.config['evaluate']['run_eval']
        self.eval_batch_size = self.config['evaluate']['batch_size']
        self.learning_rate = self.config['training']['optimizer']['learning_rate']
        self.learning_rate_warmup_until = self.config['training']['optimizer']['learning_rate_warmup_until']
        self.verb_entropy_coefficient = self.config['training']['verb_entropy_coefficient']

        # Set the random seed manually for reproducibility.
        self.random_seed = self.config['general']['random_seed']
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            if not self.config['general']['use_cuda']:
                print(
                    "WARNING: CUDA device detected but 'use_cuda: false' found in config.yaml")
                self.use_cuda = False
            else:
                torch.backends.cudnn.deterministic = True
                torch.cuda.manual_seed(self.random_seed)
                self.use_cuda = True
        else:
            self.use_cuda = False

        self.experiment_tag = self.config['checkpoint']['experiment_tag']
        self.report_frequency = self.config['checkpoint']['report_frequency']
        self.load_pretrained = self.config['checkpoint']['load_pretrained']
        self.load_from_tag = self.config['checkpoint']['load_from_tag']
        self.discount_gamma = self.config['training']['discount_gamma']

        # replay buffer and updates
        self.replay_batch_size = self.config['replay']['replay_batch_size']
        self.accumulate_reward_from_final = self.config['replay']['accumulate_reward_from_final']

        self.replay_memory = memory.ReplayMemory(self.config['replay']['replay_memory_capacity'],
                                                 priority_fraction=self.config['replay']['replay_memory_priority_fraction'],
                                                 discount_gamma=self.discount_gamma,
                                                 accumulate_reward_from_final=self.accumulate_reward_from_final)
        self.update_per_k_game_steps = self.config['replay']['update_per_k_game_steps']
        self.multi_step = self.config['replay']['multi_step']
        self.replay_sample_history_length = self.config['replay']['replay_sample_history_length']
        self.replay_sample_update_from = self.config['replay']['replay_sample_update_from']
        self.recurrent = self.config['model']['recurrent']

        # epsilon greedy
        self.epsilon_anneal_episodes = self.config['epsilon_greedy']['epsilon_anneal_episodes']
        self.epsilon_anneal_from = self.config['epsilon_greedy']['epsilon_anneal_from']
        self.epsilon_anneal_to = self.config['epsilon_greedy']['epsilon_anneal_to']
        self.epsilon = self.epsilon_anneal_from
        self.noisy_net = self.config['epsilon_greedy']['noisy_net']
        if self.noisy_net:
            # disable epsilon greedy
            self.epsilon_anneal_episodes = -1
            self.epsilon = 0.0

    def train(self):
        """
        Tell the agent that it's training phase.
        """
        self.mode = "train"
        self.online_net.train()

    def eval(self):
        """
        Tell the agent that it's evaluation phase.
        """
        self.mode = "eval"
        self.online_net.eval()

    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def reset_noise(self):
        if self.noisy_net:
            # Resets noisy weights in all linear layers (of online net only)
            self.online_net.reset_noise()

    def load_pretrained_model(self, load_from):
        """
        Load pretrained checkpoint from file.

        Arguments:
            load_from: File name of the pretrained model checkpoint.
        """
        print("loading model from %s\n" % (load_from))
        try:
            if self.use_cuda:
                pretrained_dict = torch.load(load_from)
            else:
                pretrained_dict = torch.load(load_from, map_location='cpu')

            model_dict = self.online_net.state_dict()
            pretrained_dict = {k: v for k,
                               v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.online_net.load_state_dict(model_dict)
            print("The loaded parameters are:")
            keys = [key for key in pretrained_dict]
            print(", ".join(keys))
            print("--------------------------")
        except:
            print("Failed to load checkpoint...")

    def save_model_to_path(self, save_to):
        torch.save(self.online_net.state_dict(), save_to)
        print("Saved checkpoint to %s..." % (save_to))

    def init(self, obs, infos):
        """
        Prepare the agent for the upcoming games.

        Arguments:
            obs: Previous command's feedback for each game.
            infos: Additional information for each game.
        """
        # reset agent, get vocabulary masks for verbs / adjectives / nouns
        self.scores = []
        self.dones = []
        self.still_running = np.ones((len(obs),), dtype="float32")
        self.kg.reset(node_capacity=self.node_capacity,
                      task_description=infos["objective"])

    def tokens_to_inputs(self, token_list):
        token_id_list = [_words_to_ids(tokens, self.word2id)
                         for tokens in token_list]
        res = to_pt(pad_sequences(token_id_list, maxlen=max(max_len(token_id_list) + 3, 7)),
                    self.use_cuda)  # 3 --> see layer.DepthwiseSeparableConv.padding
        return res

    def token_ids_to_inputs(self, token_id_list):
        res = to_pt(pad_sequences(token_id_list, maxlen=max(max_len(token_id_list) + 3, 7)),
                    self.use_cuda)  # 3 --> see layer.DepthwiseSeparableConv.padding
        return res

    def get_input_token_ids(self, input_list):
        token_list = [tokenize(sentence) for sentence in input_list]
        token_id_list = [_words_to_ids(tokens, self.word2id)
                         for tokens in token_list]
        return token_id_list

    def get_agent_inputs(self, input_list):
        token_list = [tokenize(sentence) for sentence in input_list]
        return self.tokens_to_inputs(token_list)

    def generate_commands(self, verb_indices, adj_indices, noun_indices):
        batch_size = verb_indices.size(0)
        res = []
        for i in range(batch_size):
            verb = self.word_vocab[verb_indices[i][0]]
            adj = self.word_vocab[adj_indices[i][0]]
            noun = self.word_vocab[noun_indices[i][0]]
            command = []
            if verb in ["where", "how"]:
                command += ["ask charlie"]
            command += [verb]
            if adj != "none":
                command += [adj]
            command += [noun]
            res.append(" ".join(command))
        return res

    def choose_random_command(self, action_rank, mask_word_ids=None):
        """
        Generate a command randomly, for epsilon greedy.
        """
        batch_size = action_rank.size(0)
        action_space_size = action_rank.size(-1)
        if mask_word_ids is None:
            indices = np.random.choice(action_space_size, batch_size)
        else:
            indices = []
            for j in range(batch_size):
                indices.append(np.random.choice(mask_word_ids[j]))
            indices = np.array(indices)
        action_indices = to_pt(
            indices, self.use_cuda).unsqueeze(-1)  # batch x 1
        return action_indices

    def choose_maxQ_command(self, action_rank, word_mask=None):
        """
        Generate a command by maximum q values, for epsilon greedy.
        """
        action_rank = action_rank - torch.min(action_rank, -1, keepdim=True)[
            0] + 1e-2  # minus the min value, so that all values are non-negative
        if word_mask is not None:
            assert word_mask.size() == action_rank.size(
            ), (word_mask.size().shape, action_rank.size())
            action_rank = action_rank * word_mask
        action_indices = torch.argmax(
            action_rank, -1, keepdim=True)  # batch x 1
        return action_indices

    def get_model(self, model_name):
        if model_name == "online":
            return self.online_net
        elif model_name == "target":
            return self.target_net
        else:
            raise NotImplementedError

    def get_node_features(self, node_token_ids, use_model):
        model = self.get_model(use_model)
        max_num_nodes = max(map(len, node_token_ids))
        node_representations = torch.zeros(
            len(node_token_ids), max_num_nodes, model.block_hidden_dim)
        if self.use_cuda:
            node_representations = node_representations.cuda()
        pos = []
        inputs = []
        for b in range(len(node_token_ids)):
            for i in range(len(node_token_ids[b])):
                inputs.append(node_token_ids[b][i])
                pos.append([b, i])

        fake_batch_size = 64
        num_batch = (len(inputs) + fake_batch_size - 1) // fake_batch_size
        rep = []
        for i in range(num_batch):
            ss = inputs[i * fake_batch_size: (i + 1) * fake_batch_size]
            input_words = self.tokens_to_inputs(ss)
            node_features_sequence, node_mask = model.representation_generator(
                input_words)  # fake_batch x sent_length x hid
            _mask = torch.sum(node_mask, -1)  # fake_batch
            tmp = torch.eq(_mask, 0).float()
            if node_features_sequence.is_cuda:
                tmp = tmp.cuda()
            _mask = _mask + tmp
            node_features = torch.sum(
                node_features_sequence, 1)  # fake_batch x hid
            node_features = node_features / _mask.unsqueeze(-1)

            rep += torch.unbind(node_features, 0)  # list of hid
        for r, p in zip(rep, pos):
            node_representations[p[0], p[1], :] = r

        return node_representations  # batch x n_node x hid

    def get_graph_representations(self, node_vocab, use_model):
        if self.setting in ["noquery", "query"]:
            return None, None
        node_features = self.get_node_features(
            node_vocab, use_model=use_model)  # batch x n_node x hid
        node_mask = torch.ne(torch.sum(node_features, -1),
                             0).float()  # batch x n_node
        return node_features, node_mask

    def get_match_representations(self, input_description, input_quest, node_representations, node_mask, use_model):
        model = self.get_model(use_model)
        description_representation_sequence, description_mask = model.representation_generator(
            input_description)
        quest_representation_sequence, quest_mask = model.representation_generator(
            input_quest)

        match_representation_sequence = model.get_match_representations(description_representation_sequence,
                                                                        description_mask,
                                                                        quest_representation_sequence,
                                                                        quest_mask,
                                                                        node_representations,
                                                                        node_mask)
        match_representation_sequence = match_representation_sequence * \
            description_mask.unsqueeze(-1)
        return match_representation_sequence, description_mask

    def get_ranks(self, input_description, input_quest, vocab_masks, node_representations, node_mask, previous_dynamics, use_model):
        """
        Given input description tensor, and previous hidden and cell states, call model forward, to get Q values of words.
        """
        model = self.get_model(use_model)
        match_representation_sequence, description_mask = self.get_match_representations(
            input_description, input_quest, node_representations, node_mask, use_model=use_model)
        if not self.recurrent:
            previous_dynamics = None
        verb_rank, adj_rank, noun_rank, current_dynamics = model.action_scorer(
            match_representation_sequence, description_mask, vocab_masks, previous_dynamics)
        if not self.recurrent:
            current_dynamics = None
        return verb_rank, adj_rank, noun_rank, current_dynamics

    def act_greedy(self, obs, infos, quest_token_ids, previous_dynamics):
        with torch.no_grad():
            batch_size = len(obs)
            # update kg
            kg_expanded = np.zeros((batch_size,), dtype="float32")
            for i in range(batch_size):
                if self.still_running[i] == 1.0:
                    kg_exp = self.kg.push_one(i, infos["extra.ans"][i])
                    kg_expanded[i] = kg_exp

            description_token_ids = self.get_input_token_ids(obs)
            input_description = self.token_ids_to_inputs(description_token_ids)
            input_quest = self.token_ids_to_inputs(quest_token_ids)

            # now verb/adj/noun share the same vocab mask
            vocab_masks, masked_ids = self.kg.get_vocabulary_mask(
                description_token_ids, quest_token_ids)
            vocab_masks_pt = to_pt(vocab_masks, True, "float")

            nodes = self.kg.get()
            node_representations, node_mask = self.get_graph_representations(
                nodes, use_model="online")  # batch x max_n_node x hid

            verb_rank, adj_rank, noun_rank, current_dynamics = self.get_ranks(
                input_description, input_quest, vocab_masks_pt, node_representations, node_mask, previous_dynamics, use_model="online")  # list of batch x vocab
            verb_indices = self.choose_maxQ_command(
                verb_rank, vocab_masks_pt[:, 0])
            adj_indices = self.choose_maxQ_command(
                adj_rank, vocab_masks_pt[:, 1])
            noun_indices = self.choose_maxQ_command(
                noun_rank, vocab_masks_pt[:, 2])
            chosen_strings = self.generate_commands(
                verb_indices, adj_indices, noun_indices)

            replay_info = [description_token_ids, quest_token_ids, verb_indices.cpu(), 
                adj_indices.cpu(), noun_indices.cpu(), nodes, masked_ids, kg_expanded]

            return chosen_strings, replay_info, current_dynamics

    def act_random(self, obs, infos, quest_token_ids, previous_dynamics):
        with torch.no_grad():
            batch_size = len(obs)
            # update kg
            kg_expanded = np.zeros((batch_size,), dtype="float32")
            for i in range(batch_size):
                if self.still_running[i] == 1.0:
                    kg_exp = self.kg.push_one(i, infos["extra.ans"][i])
                    kg_expanded[i] = kg_exp

            description_token_ids = self.get_input_token_ids(obs)
            input_description = self.token_ids_to_inputs(description_token_ids)
            input_quest = self.token_ids_to_inputs(quest_token_ids)

            # now verb/adj/noun share the same vocab mask
            vocab_masks, masked_ids = self.kg.get_vocabulary_mask(
                description_token_ids, quest_token_ids)
            vocab_masks_pt = to_pt(vocab_masks, True, "float")

            nodes = self.kg.get()
            node_representations, node_mask = self.get_graph_representations(
                nodes, use_model="online")  # batch x max_n_node x hid
            verb_rank, adj_rank, noun_rank, current_dynamics = self.get_ranks(
                input_description, input_quest, vocab_masks_pt, node_representations, node_mask, previous_dynamics, use_model="online")  # list of batch x vocab
            verb_indices = self.choose_random_command(verb_rank, masked_ids[0])
            adj_indices = self.choose_random_command(adj_rank, masked_ids[1])
            noun_indices = self.choose_random_command(noun_rank, masked_ids[2])
            chosen_strings = self.generate_commands(
                verb_indices, adj_indices, noun_indices)

            replay_info = [description_token_ids, quest_token_ids, verb_indices.cpu(), 
                adj_indices.cpu(), noun_indices.cpu(), nodes, masked_ids, kg_expanded]

            return chosen_strings, replay_info, current_dynamics

    def act(self, obs, infos, quest_token_ids, previous_dynamics, random=False):

        if self.mode == "eval":
            return self.act_greedy(obs, infos, quest_token_ids, previous_dynamics)
        if random:
            return self.act_random(obs, infos, quest_token_ids, previous_dynamics)

        with torch.no_grad():
            batch_size = len(obs)
            # update kg
            kg_expanded = np.zeros((batch_size,), dtype="float32")
            for i in range(batch_size):
                if self.still_running[i] == 1.0:
                    kg_exp = self.kg.push_one(i, infos["extra.ans"][i])
                    kg_expanded[i] = kg_exp

            description_token_ids = self.get_input_token_ids(obs)
            input_description = self.token_ids_to_inputs(description_token_ids)
            input_quest = self.token_ids_to_inputs(quest_token_ids)

            # now verb/adj/noun share the same vocab mask
            vocab_masks, masked_ids = self.kg.get_vocabulary_mask(
                description_token_ids, quest_token_ids)
            vocab_masks_pt = to_pt(vocab_masks, True, "float")

            nodes = self.kg.get()
            node_representations, node_mask = self.get_graph_representations(
                nodes, use_model="online")  # batch x max_n_node x hid

            verb_rank, adj_rank, noun_rank, current_dynamics = self.get_ranks(
                input_description, input_quest, vocab_masks_pt, node_representations, node_mask, previous_dynamics, use_model="online")  # list of batch x vocab

            verb_indices_random = self.choose_random_command(
                verb_rank, masked_ids[0])
            adj_indices_random = self.choose_random_command(
                adj_rank, masked_ids[1])
            noun_indices_random = self.choose_random_command(
                noun_rank, masked_ids[2])

            verb_indices_maxq = self.choose_maxQ_command(
                verb_rank, vocab_masks_pt[:, 0])
            adj_indices_maxq = self.choose_maxQ_command(
                adj_rank, vocab_masks_pt[:, 1])
            noun_indices_maxq = self.choose_maxQ_command(
                noun_rank, vocab_masks_pt[:, 2])

            # random number for epsilon greedy
            rand_num = np.random.uniform(
                low=0.0, high=1.0, size=(input_description.size(0), 1))
            less_than_epsilon = (rand_num < self.epsilon).astype(
                "float32")  # batch
            greater_than_epsilon = 1.0 - less_than_epsilon
            less_than_epsilon = to_pt(
                less_than_epsilon, self.use_cuda, type='long')
            greater_than_epsilon = to_pt(
                greater_than_epsilon, self.use_cuda, type='long')

            verb_indices = less_than_epsilon * verb_indices_random + \
                greater_than_epsilon * verb_indices_maxq
            adj_indices = less_than_epsilon * adj_indices_random + \
                greater_than_epsilon * adj_indices_maxq
            noun_indices = less_than_epsilon * noun_indices_random + \
                greater_than_epsilon * noun_indices_maxq
            chosen_strings = self.generate_commands(
                verb_indices, adj_indices, noun_indices)

            replay_info = [description_token_ids, quest_token_ids, verb_indices.cpu(), 
                adj_indices.cpu(), noun_indices.cpu(), nodes, masked_ids, kg_expanded]

            return chosen_strings, replay_info, current_dynamics

    def get_dqn_loss(self):
        """
        Update neural model in agent. In this example we follow algorithm
        of updating model in dqn with replay memory.
        """
        if len(self.replay_memory) < self.replay_batch_size:
            return None, None

        data = self.replay_memory.get_batch(
            self.replay_batch_size, self.multi_step)
        if data is None:
            return None, None

        obs_list, quest_list, verb_indices, adj_indices, noun_indices, nodes, vocab_mask_ids, rewards, next_obs_list, next_nodes, next_vocab_mask_ids, actual_ns = data
        batch_size = len(obs_list)
        vocab_masks = self.kg.masked_id_to_masks(batch_size, vocab_mask_ids)
        next_vocab_masks = self.kg.masked_id_to_masks(batch_size, next_vocab_mask_ids)

        vocab_masks = to_pt(vocab_masks, False, "float")
        next_vocab_masks = to_pt(next_vocab_masks, False, "float")
        if self.setting in ["noquery", "query"]:
            nodes, next_nodes = None, None
        if self.use_cuda:
            verb_indices, adj_indices, noun_indices, rewards = verb_indices.cuda(
            ), adj_indices.cuda(), noun_indices.cuda(), rewards.cuda()
            vocab_masks, next_vocab_masks = vocab_masks.cuda(), next_vocab_masks.cuda()

        input_description = self.token_ids_to_inputs(obs_list)
        input_quest = self.token_ids_to_inputs(quest_list)
        next_input_description = self.token_ids_to_inputs(next_obs_list)

        node_representations, node_mask = self.get_graph_representations(
            nodes, use_model="online")  # batch x max_n_node x hid
        verb_rank, adj_rank, noun_rank, _ = self.get_ranks(
            input_description, input_quest, vocab_masks, node_representations, node_mask, None, use_model="online")  # list of batch x vocab

        q_value_verb = ez_gather_dim_1(
            verb_rank, verb_indices).squeeze(1)  # batch
        q_value_adj = ez_gather_dim_1(
            adj_rank, adj_indices).squeeze(1)  # batch
        q_value_noun = ez_gather_dim_1(
            noun_rank, noun_indices).squeeze(1)  # batch
        q_value = torch.mean(torch.stack(
            [q_value_verb, q_value_adj, q_value_noun], -1), -1)  # average

        if self.verb_entropy_coefficient > 0.0:
            verb_prob = masked_softmax(verb_rank, vocab_masks[:, 0])  # batch x vocab
            verb_log_prob = masked_log_softmax(verb_rank, vocab_masks[:, 0])  # batch x vocab
            verb_entropy = -torch.mean(torch.sum(verb_prob * verb_log_prob * vocab_masks[:, 0], -1))  # 1
        else:
            verb_entropy = 0.0

        with torch.no_grad():
            if self.noisy_net:
                self.target_net.reset_noise()  # Sample new target net noise
            next_node_representations_online, next_node_mask = self.get_graph_representations(
                next_nodes, use_model="online")  # batch x max_n_node x hid
            next_node_representations_target, _ = self.get_graph_representations(
                next_nodes, use_model="target")  # batch x max_n_node x hid

            # pns Probabilities p(s_t+n, ·; θonline)
            next_verb_rank, next_adj_rank, next_noun_rank, _ = self.get_ranks(
                next_input_description, input_quest, next_vocab_masks, next_node_representations_online, next_node_mask, None, use_model="online")  # list of batch x vocab
            # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
            next_verb_indices = self.choose_maxQ_command(
                next_verb_rank, next_vocab_masks[:, 0])  # batch x 1
            next_adj_indices = self.choose_maxQ_command(
                next_adj_rank, next_vocab_masks[:, 1])  # batch x 1
            next_noun_indices = self.choose_maxQ_command(
                next_noun_rank, next_vocab_masks[:, 2])  # batch x 1
            # pns # Probabilities p(s_t+n, ·; θtarget)
            next_verb_rank, next_adj_rank, next_noun_rank, _ = self.get_ranks(
                next_input_description, input_quest, next_vocab_masks, next_node_representations_target, next_node_mask, None, use_model="target")  # batch x vocab
            # pns_a # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)
            next_q_value_verb = ez_gather_dim_1(
                next_verb_rank, next_verb_indices).squeeze(1)  # batch
            next_q_value_adj = ez_gather_dim_1(
                next_adj_rank, next_adj_indices).squeeze(1)  # batch
            next_q_value_noun = ez_gather_dim_1(
                next_noun_rank, next_noun_indices).squeeze(1)  # batch
            next_q_value = torch.mean(torch.stack(
                [next_q_value_verb, next_q_value_adj, next_q_value_noun], -1), -1)  # average

            discount = to_pt((np.ones_like(actual_ns) * self.discount_gamma)
                             ** actual_ns, self.use_cuda, type="float")

        rewards = rewards + next_q_value * discount  # batch
        loss = F.smooth_l1_loss(q_value, rewards) - self.verb_entropy_coefficient * verb_entropy
        return loss, q_value

    def get_drqn_loss(self):
        """
        Update neural model in agent. In this example we follow algorithm
        of updating model in dqn with replay memory.
        """
        if len(self.replay_memory) < self.replay_batch_size:
            return None, None
        data, contains_first_step = self.replay_memory.get_batch_of_sequences(
            self.replay_batch_size, sample_history_length=self.replay_sample_history_length)
        if data is None:
            return None, None

        seq_obs, quest, seq_verb_indices, seq_adj_indices, seq_noun_indices, seq_nodes, seq_vocab_mask_ids, seq_reward, seq_next_obs, seq_next_nodes, seq_next_vocab_mask_ids, seq_trajectory_mask = data
        sum_loss, sum_q_value, none_zero, sum_entropy = None, None, None, None
        previous_dynamics = None
        batch_size = len(quest)

        input_quest = self.token_ids_to_inputs(quest)
        for step_no in range(self.replay_sample_history_length):
            obs, verb_indices, adj_indices, noun_indices, nodes, vocab_mask_ids, rewards, next_obs_list, next_nodes, next_vocab_mask_ids, trajectory_mask = seq_obs[step_no], seq_verb_indices[step_no], seq_adj_indices[
                step_no], seq_noun_indices[step_no], seq_nodes[step_no], seq_vocab_mask_ids[step_no], seq_reward[step_no], seq_next_obs[step_no], seq_next_nodes[step_no], seq_next_vocab_mask_ids[step_no], seq_trajectory_mask[step_no]

            vocab_masks = self.kg.masked_id_to_masks(batch_size, vocab_mask_ids)
            next_vocab_masks = self.kg.masked_id_to_masks(batch_size, next_vocab_mask_ids)

            vocab_masks = to_pt(vocab_masks, False, "float")
            next_vocab_masks = to_pt(next_vocab_masks, False, "float")
            if self.setting in ["noquery", "query"]:
                nodes, next_nodes = None, None
            if self.use_cuda:
                verb_indices, adj_indices, noun_indices, rewards = verb_indices.cuda(
                ), adj_indices.cuda(), noun_indices.cuda(), rewards.cuda()
                vocab_masks, next_vocab_masks, trajectory_mask = vocab_masks.cuda(
                ), next_vocab_masks.cuda(), trajectory_mask.cuda()

            input_description = self.token_ids_to_inputs(obs)
            next_input_description = self.token_ids_to_inputs(next_obs_list)

            node_representations, node_mask = self.get_graph_representations(
                nodes, use_model="online")  # batch x max_n_node x hid
            verb_rank, adj_rank, noun_rank, current_dynamics = self.get_ranks(
                input_description, input_quest, vocab_masks, node_representations, node_mask, previous_dynamics, use_model="online")  # list of batch x vocab

            q_value_verb = ez_gather_dim_1(
                verb_rank, verb_indices).squeeze(1)  # batch
            q_value_adj = ez_gather_dim_1(
                adj_rank, adj_indices).squeeze(1)  # batch
            q_value_noun = ez_gather_dim_1(
                noun_rank, noun_indices).squeeze(1)  # batch
            q_value = torch.mean(torch.stack(
                [q_value_verb, q_value_adj, q_value_noun], -1), -1)  # average

            if self.verb_entropy_coefficient > 0.0:
                verb_prob = masked_softmax(verb_rank, vocab_masks[:, 0])  # batch x vocab
                verb_log_prob = masked_log_softmax(verb_rank, vocab_masks[:, 0])  # batch x vocab
                verb_entropy = -torch.sum(verb_prob * verb_log_prob * vocab_masks[:, 0], -1)  # batch
            else:
                verb_entropy = None

            previous_dynamics = current_dynamics
            if (not contains_first_step) and step_no < self.replay_sample_update_from:
                q_value = q_value.detach()
                previous_dynamics = previous_dynamics.detach()
                continue

            with torch.no_grad():
                if self.noisy_net:
                    self.target_net.reset_noise()  # Sample new target net noise
                # pns Probabilities p(s_t+n, ·; θonline)
                next_node_representations_online, next_node_mask = self.get_graph_representations(
                    next_nodes, use_model="online")  # batch x max_n_node x hid
                next_node_representations_target, _ = self.get_graph_representations(
                    next_nodes, use_model="target")  # batch x max_n_node x hid

                # pns Probabilities p(s_t+n, ·; θonline)
                next_verb_rank, next_adj_rank, next_noun_rank, _ = self.get_ranks(
                    next_input_description, input_quest, next_vocab_masks, next_node_representations_online, next_node_mask, previous_dynamics, use_model="online")  # list of batch x vocab
                # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
                next_verb_indices = self.choose_maxQ_command(
                    next_verb_rank, next_vocab_masks[:, 0])  # batch x 1
                next_adj_indices = self.choose_maxQ_command(
                    next_adj_rank, next_vocab_masks[:, 1])  # batch x 1
                next_noun_indices = self.choose_maxQ_command(
                    next_noun_rank, next_vocab_masks[:, 2])  # batch x 1
                # pns # Probabilities p(s_t+n, ·; θtarget)
                next_verb_rank, next_adj_rank, next_noun_rank, _ = self.get_ranks(
                    next_input_description, input_quest, next_vocab_masks, next_node_representations_target, next_node_mask, previous_dynamics, use_model="target")  # batch x vocab
                # pns_a # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)
                next_q_value_verb = ez_gather_dim_1(
                    next_verb_rank, next_verb_indices).squeeze(1)  # batch
                next_q_value_adj = ez_gather_dim_1(
                    next_adj_rank, next_adj_indices).squeeze(1)  # batch
                next_q_value_noun = ez_gather_dim_1(
                    next_noun_rank, next_noun_indices).squeeze(1)  # batch
                next_q_value = torch.mean(torch.stack(
                    [next_q_value_verb, next_q_value_adj, next_q_value_noun], -1), -1)  # average

            rewards = rewards + next_q_value * self.discount_gamma  # batch
            loss = F.smooth_l1_loss(
                q_value, rewards, reduction="none")  # batch
            loss = loss * trajectory_mask  # batch
            if verb_entropy is not None:
                verb_entropy = verb_entropy * trajectory_mask

            if sum_loss is None:
                sum_loss = torch.sum(loss)
                sum_q_value = torch.sum(q_value)
                none_zero = torch.sum(trajectory_mask)
                if verb_entropy is not None:
                    sum_entropy = torch.sum(verb_entropy)
            else:
                sum_loss = sum_loss + torch.sum(loss)
                none_zero = none_zero + torch.sum(trajectory_mask)
                sum_q_value = sum_q_value + torch.sum(q_value)
                if verb_entropy is not None:
                    sum_entropy = sum_entropy + torch.sum(verb_entropy)

        tmp = torch.eq(none_zero, 0).float()  # 1
        if sum_loss.is_cuda:
            tmp = tmp.cuda()
        none_zero = none_zero + tmp  # 1
        loss = sum_loss / none_zero
        q_value = sum_q_value / none_zero
        if self.verb_entropy_coefficient > 0.0:
            loss = loss - self.verb_entropy_coefficient * sum_entropy / none_zero 
        return loss, q_value

    def update_interaction(self):
        # update neural model by replaying snapshots in replay memory
        if self.recurrent:
            interaction_loss, q_value = self.get_drqn_loss()
        else:
            interaction_loss, q_value = self.get_dqn_loss()
        if interaction_loss is None:
            return None, None
        loss = interaction_loss
        # Backpropagate
        self.online_net.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(
            self.online_net.parameters(), self.clip_grad_norm)
        self.optimizer.step()  # apply gradients
        return to_np(torch.mean(interaction_loss)), to_np(torch.mean(q_value))

    def finish_of_episode(self, episode_no, batch_no, batch_size):
        # Update target network
        if (episode_no + batch_size) % self.target_net_update_frequency <= episode_no % self.target_net_update_frequency:
            self.update_target_net()

        # decay lambdas
        if episode_no < self.learn_start_from_this_episode:
            return
        if episode_no < self.epsilon_anneal_episodes + self.learn_start_from_this_episode:
            self.epsilon -= (self.epsilon_anneal_from -
                             self.epsilon_anneal_to) / float(self.epsilon_anneal_episodes)
            self.epsilon = max(self.epsilon, 0.0)

        # learning rate warmup
        if batch_no < self.learning_rate_warmup_until:
            cr = self.learning_rate / \
                math.log2(self.learning_rate_warmup_until)
            learning_rate = cr * math.log2(batch_no + 1)
        else:
            learning_rate = self.learning_rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate
