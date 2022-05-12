import os
import copy
import logging
import numpy as np

import torch
import torch.nn.functional as F
from layers import WordEmbedding, NoisyLinear
from layers import EncoderBlock, MultiHeadAttention

logger = logging.getLogger(__name__)


class LSTM_DQN(torch.nn.Module):
    model_name = 'lstm_dqn'

    def __init__(self, config, word_vocab):
        super(LSTM_DQN, self).__init__()
        self.config = config
        self.word_vocab = word_vocab
        self.word_vocab_size = len(self.word_vocab)
        self.read_config()
        self._def_layers()
        # self.print_parameters()

    def print_parameters(self):
        amount = 0
        for p in self.parameters():
            amount += np.prod(p.size())
        print("total number of parameters: %s" % (amount))
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        amount = 0
        for p in parameters:
            amount += np.prod(p.size())
        print("number of trainable parameters: %s" % (amount))

    def read_config(self):
        # model config
        model_config = self.config['model']

        self.word_embedding_trainable = model_config['word_embedding_trainable']
        self.embedding_dropout = model_config['embedding_dropout']

        self.encoder_layers = model_config['encoder_layers']
        self.encoder_conv_num = model_config['encoder_conv_num']
        self.block_hidden_dim = model_config['block_hidden_dim']
        self.n_heads = model_config['n_heads']
        self.block_dropout = model_config['block_dropout']
        self.attention_dropout = model_config['attention_dropout']
        self.recurrent = model_config['recurrent']

        self.noisy_net = self.config['epsilon_greedy']['noisy_net']

    def _def_layers(self):

        self.word_embedding = WordEmbedding(embedding_size=self.block_hidden_dim,
                                            vocab_size=self.word_vocab_size,
                                            trainable=self.word_embedding_trainable,
                                            dropout_rate=self.embedding_dropout)

        self.transformer_encoder = torch.nn.ModuleList([EncoderBlock(conv_num=self.encoder_conv_num, ch_num=self.block_hidden_dim, k=7,
                                                       block_hidden_dim=self.block_hidden_dim, n_head=self.n_heads, dropout=self.block_dropout) for _ in range(self.encoder_layers)])

        self.multi_head_attention_description_task = MultiHeadAttention(
            self.block_hidden_dim, self.n_heads, dropout=self.block_dropout)
        self.residual_connection_description_task = torch.nn.Linear(
            self.block_hidden_dim * 2, self.block_hidden_dim)

        self.multi_head_attention_description_graph = MultiHeadAttention(
            self.block_hidden_dim, self.n_heads, dropout=self.block_dropout)
        self.residual_connection_description_graph = torch.nn.Linear(
            self.block_hidden_dim * 2, self.block_hidden_dim)

        encoder_output_dim = self.block_hidden_dim

        if self.recurrent:
            self.rnncell = torch.nn.GRUCell(
                self.block_hidden_dim, self.block_hidden_dim)
            self.dynamics_aggregation = torch.nn.Linear(
                self.block_hidden_dim * 2, self.block_hidden_dim)
        else:
            self.rnncell, self.dynamics_aggregation = None, None

        linear_function = NoisyLinear if self.noisy_net else torch.nn.Linear

        self.action_scorer_shared_linear = linear_function(
            encoder_output_dim, self.block_hidden_dim)
        action_scorer_input_size = self.block_hidden_dim

        self.action_scorer_verb = linear_function(action_scorer_input_size, 1)
        self.action_scorer_adj = linear_function(action_scorer_input_size, 1)
        self.action_scorer_noun = linear_function(action_scorer_input_size, 1)
        self.action_scorer_verb_advantage = linear_function(
            action_scorer_input_size, self.word_vocab_size)
        self.action_scorer_adj_advantage = linear_function(
            action_scorer_input_size, self.word_vocab_size)
        self.action_scorer_noun_advantage = linear_function(
            action_scorer_input_size, self.word_vocab_size)

    def aggregate_two_inputs(self, input_1, mask_1, input_2, mask_2, aggregator, residual_aggregator=None):
        # input_1: batch x len_1 x hid
        # input_2: batch x len_2 x hid
        # mask_1: batch x len_1
        # mask_2: batch x len_2
        # aggregator: an aggregator module (attention)
        # residual_aggregator: an MLP

        # batch x len_1 x len_2
        squared_mask = torch.bmm(mask_1.unsqueeze(-1), mask_2.unsqueeze(1))
        aggregated_representations, _ = aggregator(
            input_1, squared_mask, input_2, input_2)  # batch x len_1 x hid
        if residual_aggregator is not None:
            aggregated_representations = torch.cat(
                [aggregated_representations, input_1], -1)  # batch x len_1 x hid*2
            aggregated_representations = torch.tanh(residual_aggregator(
                aggregated_representations))  # batch x len_1 x hid
            aggregated_representations = aggregated_representations * \
                mask_1.unsqueeze(-1)  # batch x len_1 x hid
        return aggregated_representations

    def get_match_representations(self, doc_encodings, doc_mask, q_encodings, q_mask, node_encodings, node_mask):
        # doc_encodings: batch x doc_len x hid
        # doc_mask: batch x doc_len
        # q_encodings: batch x q_len x hid
        # q_mask: batch x q_len
        # node_encodings: batch x num_node x hid
        # node_mask: batch x num_node
        description_representations_sequence = self.aggregate_two_inputs(doc_encodings, doc_mask,
                                                                         q_encodings, q_mask,
                                                                         self.multi_head_attention_description_task,
                                                                         self.residual_connection_description_task)  # batch x doc_len x hid

        if node_encodings is not None:
            description_representations_sequence = self.aggregate_two_inputs(description_representations_sequence, doc_mask,
                                                                             node_encodings, node_mask,
                                                                             self.multi_head_attention_description_task,
                                                                             self.residual_connection_description_task)  # batch x doc_len x hid
        return description_representations_sequence

    def get_embeddings(self, _input_word_ids):
        # _input_words: batch x time
        embeddings, masks = self.word_embedding(
            _input_word_ids)  # batch x time x emb
        embeddings = embeddings * masks.unsqueeze(-1)  # batch x time x emb
        return embeddings, masks

    def representation_generator(self, _input_word_ids):
        # _input_word_ids: batch x time
        embeddings, masks = self.get_embeddings(
            _input_word_ids)  # batch x time x emb, batch x time
        square_mask = torch.bmm(masks.unsqueeze(-1),
                                masks.unsqueeze(1))  # batch x time x time
        encoding_sequence = embeddings
        for i in range(self.encoder_layers):
            encoding_sequence = self.transformer_encoder[i](encoding_sequence, masks, square_mask, i * (
                self.encoder_conv_num + 2) + 1, self.encoder_layers)  # batch x time x enc
            encoding_sequence = encoding_sequence * masks.unsqueeze(-1)
        return encoding_sequence, masks

    def masked_mean(self, _input, _mask):
        # _input: batch x time x hid
        # _mask: batch x time
        _input = _input * _mask.unsqueeze(-1)
        # masked mean
        avg_input = torch.sum(_input, 1)  # batch x enc
        _m = torch.sum(_mask, -1)  # batch
        tmp = torch.eq(_m, 0).float()  # batch
        if avg_input.is_cuda:
            tmp = tmp.cuda()
        _m = _m + tmp
        avg_input = avg_input / _m.unsqueeze(-1)  # batch x enc
        return avg_input

    def action_scorer(self, state_representation_sequence, mask, vocab_masks, previous_dynamics):
        # state_representation: batch x time x enc_dim
        # mask: batch x time
        # vocab_masks: batch x 3 x vocab
        verb_mask, adj_mask, noun_mask = vocab_masks[:,
                                                     0], vocab_masks[:, 1], vocab_masks[:, 2]

        current_dynamics = self.masked_mean(
            state_representation_sequence, mask)
        if self.recurrent:
            current_dynamics = self.rnncell(
                current_dynamics, previous_dynamics) if previous_dynamics is not None else self.rnncell(current_dynamics)

        state_representation = self.action_scorer_shared_linear(
            current_dynamics)  # action scorer hidden dim
        state_representation = torch.relu(state_representation)

        verb_rank = self.action_scorer_verb(state_representation)  # batch x 1
        verb_rank_advantage = self.action_scorer_verb_advantage(
            state_representation)  # advantage stream  batch x n_vocab
        verb_rank = verb_rank + verb_rank_advantage - \
            verb_rank_advantage.mean(1, keepdim=True)  # combine streams
        verb_rank = verb_rank * verb_mask

        adj_rank = self.action_scorer_adj(state_representation)  # batch x 1
        adj_rank_advantage = self.action_scorer_adj_advantage(
            state_representation)  # advantage stream  batch x n_vocab
        adj_rank = adj_rank + adj_rank_advantage - \
            adj_rank_advantage.mean(1, keepdim=True)  # combine streams
        adj_rank = adj_rank * adj_mask

        noun_rank = self.action_scorer_noun(state_representation)  # batch x 1
        noun_rank_advantage = self.action_scorer_noun_advantage(
            state_representation)  # advantage stream  batch x n_vocab
        noun_rank = noun_rank + noun_rank_advantage - \
            noun_rank_advantage.mean(1, keepdim=True)  # combine streams
        noun_rank = noun_rank * noun_mask

        return verb_rank, adj_rank, noun_rank, current_dynamics

    def reset_noise(self):
        if self.noisy_net:
            self.action_scorer_shared_linear.reset_noise()
            self.action_scorer_verb.reset_noise()
            self.action_scorer_adj.reset_noise()
            self.action_scorer_noun.reset_noise()
            self.action_scorer_verb_advantage.reset_noise()
            self.action_scorer_adj_advantage.reset_noise()
            self.action_scorer_noun_advantage.reset_noise()
