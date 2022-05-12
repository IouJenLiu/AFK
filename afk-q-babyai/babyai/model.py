import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.categorical import Categorical
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import babyai.rl
from babyai.rl.utils.supervised_losses import required_heads
from babyai.utils.distribution import multi_categorical_maker
import gym
from babyai.layers.gcn import StackedGraphConvolution, masked_mean, masked_max
from babyai.layers.aggrerator import MultiHeadAttention
from babyai.utils.distribution import multi_categorical_maker


# From https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


# Inspired by FiLMedBlock from https://arxiv.org/abs/1709.07871
class FiLM(nn.Module):
    def __init__(self, in_features, out_features, in_channels, imm_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=imm_channels,
            kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(imm_channels)
        self.conv2 = nn.Conv2d(
            in_channels=imm_channels, out_channels=out_features,
            kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(out_features)

        self.weight = nn.Linear(in_features, out_features)
        self.bias = nn.Linear(in_features, out_features)

        self.apply(initialize_parameters)

    def forward(self, x, y):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        weight = self.weight(y).unsqueeze(2).unsqueeze(3)
        bias = self.bias(y).unsqueeze(2).unsqueeze(3)
        out = x * weight + bias
        return F.relu(self.bn2(out))


class ImageBOWEmbedding(nn.Module):
   def __init__(self, max_value, embedding_dim):
       super().__init__()
       self.max_value = max_value
       self.embedding_dim = embedding_dim
       self.embedding = nn.Embedding(3 * max_value, embedding_dim)
       self.apply(initialize_parameters)

   def forward(self, inputs):
       offsets = torch.Tensor([0, self.max_value, 2 * self.max_value, 3 * self.max_value]).to(inputs.device)
       #offsets = torch.Tensor([0, self.max_value, 2 * self.max_value]).to(inputs.device)
       inputs = (inputs + offsets[None, :, None, None]).long()
       return self.embedding(inputs).sum(1).permute(0, 3, 1, 2)


class ACModel(nn.Module, babyai.rl.RecurrentACModel):
    def __init__(self, obs_space, action_space,
                 image_dim=128, memory_dim=128, instr_dim=128,
                 use_instr=False, lang_model="gru", use_memory=False,
                 arch="bow_endpool_res", aux_info=None, query=False, flat_query=False, two_stage_query=False,
                 onehot_ans=False, no_qa_embedding=False, n_image_channel=4, kg_repr='one_hot', invariant_module='max',
                 gcn_adj_type='no_connect', controller_arch='film', query_arch='flat', no_kg='False', vocab=None):
        super().__init__()
        self.vocab = vocab
        self.no_kg = no_kg
        self.query_arch = query_arch
        self.hidden_size = 128
        self.gcn_adj_type = gcn_adj_type
        self.controller_arch = controller_arch
        endpool = 'endpool' in arch
        use_bow = 'bow' in arch
        pixel = 'pixel' in arch
        self.res = 'res' in arch
        self.query = query
        self.flat_query = flat_query
        self.two_stage_query = two_stage_query
        self.action_space = action_space
        self.kg_repr = kg_repr
        self.invariant_module = invariant_module
        # Decide which components are enabled
        self.use_instr = use_instr
        self.use_memory = use_memory
        self.arch = arch
        self.lang_model = lang_model
        self.aux_info = aux_info
        if self.res and image_dim != 128:
            raise ValueError(f"image_dim is {image_dim}, expected 128")
        self.image_dim = image_dim
        self.memory_dim = memory_dim
        self.instr_dim = instr_dim
        self.n_moving_action = 7
        self.n_wh = 2
        self.obs_space = obs_space
        self.onehot_ans = onehot_ans
        self.no_qa_embedding = no_qa_embedding


        for part in self.arch.split('_'):
            if part not in ['original', 'bow', 'pixels', 'endpool', 'res', 'att']:
                raise ValueError("Incorrect architecture name: {}".format(self.arch))

        # if not self.use_instr:
        #     raise ValueError("FiLM architecture can be used when instructions are enabled")

        self.image_conv = nn.Sequential(*[
            *([ImageBOWEmbedding(obs_space['image'], 128)] if use_bow else []),
            *([nn.Conv2d(
                in_channels=n_image_channel, out_channels=128, kernel_size=(8, 8),
                stride=8, padding=0)] if pixel else []),
            nn.Conv2d(
                in_channels=128 if use_bow or pixel else n_image_channel, out_channels=128,
                kernel_size=(3, 3) if endpool else (2, 2), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            *([] if endpool else [nn.MaxPool2d(kernel_size=(2, 2), stride=2)]),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            *([] if endpool else [nn.MaxPool2d(kernel_size=(2, 2), stride=2)])
        ])


        self.film_pool = nn.MaxPool2d(kernel_size=(7, 7) if endpool else (2, 2), stride=2)

        # Define instruction embedding
        if (self.use_instr or self.query) and not self.no_qa_embedding:
            if self.lang_model in ['gru', 'bigru', 'attgru']:
                self.word_embedding = nn.Embedding(obs_space["instr"], self.instr_dim)
                if self.lang_model in ['gru', 'bigru', 'attgru']:
                    gru_dim = self.instr_dim
                    if self.lang_model in ['bigru', 'attgru']:
                        gru_dim //= 2
                    self.instr_rnn = nn.GRU(
                        self.instr_dim, gru_dim, batch_first=True,
                        bidirectional=(self.lang_model in ['bigru', 'attgru']))
                    self.final_instr_dim = self.instr_dim
                else:
                    kernel_dim = 64
                    kernel_sizes = [3, 4]
                    self.instr_convs = nn.ModuleList([
                        nn.Conv2d(1, kernel_dim, (K, self.instr_dim)) for K in kernel_sizes])
                    self.final_instr_dim = kernel_dim * len(kernel_sizes)

            if self.lang_model == 'attgru':
                self.memory2key = nn.Linear(self.memory_size, self.final_instr_dim)

            if self.controller_arch == 'film':
                num_module = 2
                self.controllers = []
                for ni in range(num_module):
                    mod = FiLM(
                        #in_features=self.final_instr_dim * 2 if self.query else self.final_instr_dim,
                        in_features=self.final_instr_dim if self.query or self.flat_query else self.final_instr_dim,
                        out_features=128 if ni < num_module-1 else self.image_dim,
                        in_channels=128, imm_channels=128)
                    self.controllers.append(mod)
                    self.add_module('FiLM_' + str(ni), mod)
            elif self.controller_arch == 'att':
                self.aggregator_n_heads = 4
                self.controller = MultiHeadAttention(self.hidden_size, self.aggregator_n_heads, dropout=0.1)

        # Define memory and resize image embedding
        self.embedding_size = self.image_dim
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_dim, self.memory_dim)
            self.embedding_size = self.semi_memory_size

        # Define actor's model
        if self.query:
            self.actor = nn.Sequential(
                nn.Linear(self.embedding_size, 64),
                nn.Tanh(),
                nn.Linear(64, action_space.nvec[0])
            )
            self.actor_binary = nn.Sequential(
                nn.Linear(self.embedding_size, 64),
                nn.Tanh(),
                nn.Linear(64, 2)
            )
            self.actor_query = nn.Sequential(
                nn.Linear(self.embedding_size, 64),
                nn.Tanh(),
                nn.Linear(64, action_space.nvec[1])
            )
            self.merge_ans_instr = nn.Sequential(nn.Linear(self.embedding_size * 2, self.embedding_size), nn.Tanh())
        else:
            self.actor = nn.Sequential(
                nn.Linear(self.embedding_size, 64),
                nn.Tanh(),
                nn.Linear(64, action_space.n)
            )
        if self.query or self.flat_query:
            gru_dim = self.instr_dim
            if self.lang_model in ['bigru', 'attgru']:
                gru_dim //= 2
            self.ans_rnn = nn.GRU(
                self.instr_dim, gru_dim, batch_first=True,
                bidirectional=(self.lang_model in ['bigru', 'attgru']))
            self.ans_mlp = nn.Linear(5, self.instr_dim)


        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(initialize_parameters)

        # Define head for extra info
        if self.aux_info:
            self.extra_heads = None
            self.add_heads()

        if 'gcn' in self.invariant_module:
            self.gcn = StackedGraphConvolution(self.hidden_size, [self.hidden_size, self.hidden_size],
                                               dropout_rate=0.1, use_highway_connections=True)
        self.n_query_head = 5
        if 'pointer' in self.query_arch:
            from babyai.layers.pointer_network import Pointer
            if 'conditioned' in self.query_arch:
                adj_q_dim = self.hidden_size * 2
                embedding_size = self.embedding_size + 2 * self.hidden_size
            else:
                adj_q_dim = self.hidden_size
                embedding_size = self.embedding_size
            self.pointer_noun = Pointer(hidden_dim=self.hidden_size)
            self.pointer_adj = Pointer(hidden_dim=self.hidden_size, q_dim=adj_q_dim)
            self.actor_wh = nn.Sequential(
                nn.Linear(embedding_size, 64),
                nn.Tanh(),
                nn.Linear(64, self.n_wh)
            )
        elif self.query_arch == 'base':
            self.head_noun = nn.Sequential(
                nn.Linear(self.embedding_size, 64),
                nn.Tanh(),
                nn.Linear(64, vocab.max_size)
            )
            self.head_adj = nn.Sequential(
                nn.Linear(self.embedding_size, 64),
                nn.Tanh(),
                nn.Linear(64, vocab.max_size)
            )
            self.actor_wh = nn.Sequential(
                nn.Linear(self.embedding_size, 64),
                nn.Tanh(),
                nn.Linear(64, self.n_wh)
            )


    def add_heads(self):
        '''
        When using auxiliary tasks, the environment yields at each step some binary, continous, or multiclass
        information. The agent needs to predict those information. This function add extra heads to the model
        that output the predictions. There is a head per extra information (the head type depends on the extra
        information type).
        '''
        self.extra_heads = nn.ModuleDict()
        for info in self.aux_info:
            if required_heads[info] == 'binary':
                self.extra_heads[info] = nn.Linear(self.embedding_size, 1)
            elif required_heads[info].startswith('multiclass'):
                n_classes = int(required_heads[info].split('multiclass')[-1])
                self.extra_heads[info] = nn.Linear(self.embedding_size, n_classes)
            elif required_heads[info].startswith('continuous'):
                if required_heads[info].endswith('01'):
                    self.extra_heads[info] = nn.Sequential(nn.Linear(self.embedding_size, 1), nn.Sigmoid())
                else:
                    raise ValueError('Only continous01 is implemented')
            else:
                raise ValueError('Type not supported')
            # initializing these parameters independently is done in order to have consistency of results when using
            # supervised-loss-coef = 0 and when not using any extra binary information
            self.extra_heads[info].apply(initialize_parameters)

    def add_extra_heads_if_necessary(self, aux_info):
        '''
        This function allows using a pre-trained model without aux_info and add aux_info to it and still make
        it possible to finetune.
        '''
        try:
            if not hasattr(self, 'aux_info') or not set(self.aux_info) == set(aux_info):
                self.aux_info = aux_info
                self.add_heads()
        except Exception:
            raise ValueError('Could not add extra heads')

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.memory_dim

    def forward(self, obs, memory, instr_embedding=None, vocab=None):
        if not self.no_qa_embedding:
            if self.kg_repr == 'raw':
                kg_embedding, graph_mask, word_encoding, word_encoding_mask, adj_mask, noun_mask = self._get_kg_embedding(obs.kg_cc, vocab)
            if self.no_kg:
                ans_embedding = self._get_instr_embedding(obs.ans)
                instr_embedding = self._get_instr_embedding(obs.instr)
                kg_embedding = self.merge_ans_instr(torch.cat([ans_embedding, instr_embedding], dim=-1))

            if (self.use_instr or self.query) and self.lang_model == "attgru":
                # outputs: B x L x D
                # memory: B x M
                mask = (obs.instr != 0).float()
                # The mask tensor has the same length as obs.instr, and
                # thus can be both shorter and longer than instr_embedding.
                # It can be longer if instr_embedding is computed
                # for a subbatch of obs.instr.
                # It can be shorter if obs.instr is a subbatch of
                # the batch that instr_embeddings was computed for.
                # Here, we make sure that mask and instr_embeddings
                # have equal length along dimension 1.
                mask = mask[:, :instr_embedding.shape[1]]
                instr_embedding = instr_embedding[:, :mask.shape[1]]

                keys = self.memory2key(memory)
                pre_softmax = (keys[:, None, :] * instr_embedding).sum(2) + 1000 * mask
                attention = F.softmax(pre_softmax, dim=1)
                instr_embedding = (instr_embedding * attention[:, :, None]).sum(1)

        x = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3)

        if 'pixel' in self.arch:
            x /= 256.0
        x = self.image_conv(x)

        if (self.use_instr or self.query) and not self.no_qa_embedding:
            if self.controller_arch == 'film':
                for controller in self.controllers:
                    # TODO: add the following code for baselines with language
                    #if self.query or self.flat_query:
                    #    #out = controller(x, torch.cat([instr_embedding, ans_embedding], dim=-1))
                    #    out = controller(x, ans_embedding)
                    #else:
                    #if kg_embedding.size()[0] > 2:
                    #    print(kg_embedding.size())
                    out = controller(x, kg_embedding)
                    if self.res:
                        out += x
                    x = out
            elif self.controller_arch == 'att':
                assert self.invariant_module == 'gcn'
                x = x.view(x.size()[0], x.size()[1], -1).permute(0, 2, 1)
                dummy_img_mask = torch.ones(x.size()[0], x.size()[1], device=x.device)
                squared_mask = torch.bmm(dummy_img_mask.unsqueeze(-1), graph_mask.unsqueeze(1))
                x, _ = self.controller(x, squared_mask, kg_embedding, kg_embedding)
                x = x.permute(0, 2, 1)
                x = x.view(x.size()[0], x.size()[1], 7, 7)
            else:
                raise NotImplementedError

        x = F.relu(self.film_pool(x))
        x = x.reshape(x.shape[0], -1)

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if hasattr(self, 'aux_info') and self.aux_info:
            extra_predictions = {info: self.extra_heads[info](embedding) for info in self.extra_heads}
        else:
            extra_predictions = dict()

        x = self.actor(embedding)
        bin_dist = None
        if self.query:
            x_bin = self.actor_binary(embedding)
            log_prob_bin = F.log_softmax(x_bin, dim=1)
            try:
                bin_dist = Categorical(logits=log_prob_bin)
            except:
                print('')
            log_prob_a = F.log_softmax(x, dim=1) + log_prob_bin[:, 0:1]
            if 'pointer' in self.query_arch:
                bz, max_n_node, instr_len, embedding_size = word_encoding.size()
                dummy_mask = torch.ones(bz, 1, device=embedding.device)
                adj_word_encoding_mask = word_encoding_mask * adj_mask
                noun_word_encoding_mask = word_encoding_mask * noun_mask
                if 'conditioned' in self.query_arch:
                    prob_noun, _ = self.pointer_noun(embedding.unsqueeze(1), word_encoding.reshape(bz, max_n_node * instr_len, embedding_size),
                                             dummy_mask, noun_word_encoding_mask.view(bz, -1).float(), obs.kg_cc.view(bz, max_n_node * instr_len), vocab)
                    noun_idx = torch.argmax(prob_noun, dim=1)
                    noun_embedding = self.word_embedding(noun_idx)
                    prob_adj, _ = self.pointer_adj(torch.cat([embedding, noun_embedding], dim=1).unsqueeze(1), word_encoding.reshape(bz, max_n_node * instr_len, embedding_size),
                                            dummy_mask, adj_word_encoding_mask.view(bz, -1).float(), obs.kg_cc.view(bz, max_n_node * instr_len), vocab)
                    adj_idx = torch.argmax(prob_adj, dim=1)
                    adj_embedding = self.word_embedding(adj_idx)
                    x_wh = self.actor_wh(torch.cat([embedding, adj_embedding, noun_embedding], dim=1))
                    log_prob_wh = F.log_softmax(x_wh, dim=1)
                elif 'location' in self.query_arch:
                    prob_noun, att_noun = self.pointer_noun(embedding.unsqueeze(1), word_encoding.reshape(bz, max_n_node * instr_len, embedding_size),
                                                  dummy_mask, noun_word_encoding_mask.view(bz, -1).float(), obs.kg_cc.view(bz, max_n_node * instr_len), vocab)
                    prob_adj, _ = self._prev_adj(torch.argmax(att_noun, dim=-1), vocab, obs.kg_cc.view(bz, max_n_node * instr_len))
                    x_wh = self.actor_wh(embedding)
                    log_prob_wh = F.log_softmax(x_wh, dim=1)
                else:
                    prob_noun, _ = self.pointer_noun(embedding.unsqueeze(1), word_encoding.reshape(bz, max_n_node * instr_len, embedding_size),
                                             dummy_mask, noun_word_encoding_mask.view(bz, -1).float(), obs.kg_cc.view(bz, max_n_node * instr_len), vocab)
                    prob_adj, _ = self.pointer_adj(embedding.unsqueeze(1), word_encoding.reshape(bz, max_n_node * instr_len, embedding_size),
                                            dummy_mask, adj_word_encoding_mask.view(bz, -1).float(), obs.kg_cc.view(bz, max_n_node * instr_len), vocab)
                    x_wh = self.actor_wh(embedding)
                    log_prob_wh = F.log_softmax(x_wh, dim=1)
                MultiCat = multi_categorical_maker([2, self.n_moving_action, self.n_wh, vocab.max_size, vocab.max_size])
                dist = MultiCat(probs=torch.cat([torch.exp(log_prob_bin), torch.exp(log_prob_a), torch.exp(log_prob_wh), prob_adj, prob_noun], dim=1))
            elif self.query_arch == 'base':
                x_wh = self.actor_wh(embedding)
                x_adj = self.head_adj(embedding)
                x_noun = self.head_noun(embedding)
                x_noun[:, self.vocab.noun_idx:] = -100000000
                x_adj[:, self.vocab.noun_idx] = -100000000
                x_adj[:, self.vocab.adj_idx:] = -100000000
                MultiCat = multi_categorical_maker([2, self.n_moving_action, self.n_wh, vocab.max_size, vocab.max_size])
                dist = MultiCat(logits=torch.cat([log_prob_bin, log_prob_a, x_wh, x_adj, x_noun], dim=1))

            else:
                x_query = self.actor_query(embedding)
                log_prob_query = F.log_softmax(x_query, dim=1) + log_prob_bin[:, 1:2]
                log_prob = torch.cat([log_prob_a, log_prob_query], dim=-1)
                dist = Categorical(logits=log_prob)
        else:
            dist = Categorical(logits=F.log_softmax(x, dim=1))
        v = self.critic(embedding)
        value = v.squeeze(1)
        return {'dist': dist, 'value': value, 'memory': memory, 'extra_predictions': extra_predictions,
                'embedding': embedding, 'x': x, 'bin_dist': bin_dist}

    def _get_instr_embedding(self, instr, query=False):
        lengths = (instr != 0).sum(1).long()
        if self.lang_model == 'gru':
            if query:
                aa = self.word_embedding(instr)
                out, _ = self.ans_rnn(self.word_embedding(instr))
            else:
                out, _ = self.instr_rnn(self.word_embedding(instr))
            hidden = out[range(len(lengths)), lengths-1, :]
            #aa = torch.sum(out, dim=-1)
            return hidden

        elif self.lang_model in ['bigru', 'attgru']:
            masks = (instr != 0).float()

            if lengths.shape[0] > 1:
                seq_lengths, perm_idx = lengths.sort(0, descending=True)
                iperm_idx = torch.LongTensor(perm_idx.shape).fill_(0)
                if instr.is_cuda: iperm_idx = iperm_idx.cuda()
                for i, v in enumerate(perm_idx):
                    iperm_idx[v.data] = i

                inputs = self.word_embedding(instr)
                inputs = inputs[perm_idx]

                inputs = pack_padded_sequence(inputs, seq_lengths.data.cpu().numpy(), batch_first=True)

                outputs, final_states = self.instr_rnn(inputs)
            else:
                instr = instr[:, 0:lengths[0]]
                outputs, final_states = self.instr_rnn(self.word_embedding(instr))
                iperm_idx = None
            final_states = final_states.transpose(0, 1).contiguous()
            final_states = final_states.view(final_states.shape[0], -1)
            if iperm_idx is not None:
                outputs, _ = pad_packed_sequence(outputs, batch_first=True)
                outputs = outputs[iperm_idx]
                final_states = final_states[iperm_idx]

            return outputs if self.lang_model == 'attgru' else final_states

        else:
            ValueError("Undefined instruction architecture: {}".format(self.use_instr))

    def _get_ans_embedding(self, ans):
        embedding = self.ans_mlp(ans)
        return embedding

    def _get_kg_embedding(self, instr, vocab):
        lengths = (instr != 0).sum(2).long()
        n_nodes = (lengths != 0).sum(1).long()
        word_encoding_mask = (instr != 0).long()
        noun_mask = (instr <= vocab.noun_idx).long()
        adj_mask = (instr > vocab.noun_idx).long() * (instr <= vocab.adj_idx).long()
        # bz x max_n_node x instr_length
        if self.lang_model == 'gru':
            bz, max_n_node, max_instr_length = instr.size()

            out, _ = self.instr_rnn(self.word_embedding(instr).view(bz * max_n_node, max_instr_length, -1))
            word_encoding = out.clone().view(bz, max_n_node, max_instr_length, -1)
            out = out[range(len(lengths.view(-1))), lengths.view(-1) - 1] # Take the last word embedding as node embedding
            hidden = out.view(lengths.size()[0], lengths.size()[1], -1)
            if self.invariant_module == 'max':
                hidden = torch.max(hidden, dim=1)[0]
            elif 'gcn' in self.invariant_module:
                if self.gcn_adj_type == 'no_connect':
                    adj_mat = torch.eye(max_n_node, device=hidden.device)
                elif self.gcn_adj_type == 'fully_connect':
                    adj_mat = torch.ones(max_n_node, device=hidden.device)
                elif self.gcn_adj_type == 'ori_link' or self.gcn_adj_type == 'ori_link_indir':
                    # Assume linear, need update
                    adj_mat = torch.eye(max_n_node, device=hidden.device)
                    adj_mat[range(max_n_node - 1), range(1, max_n_node)] = 1
                    if self.gcn_adj_type == 'ori_link_indir':
                        adj_mat[range(1, max_n_node), range(max_n_node - 1)] = 1
                else:
                    raise NotImplementedError
                adj_mat = adj_mat.unsqueeze(0).repeat(bz, 1, 1)
                hidden = self.gcn(hidden, adj_mat)
                graph_mask = self._compute_graph_mask(n_nodes, max_n_node, hidden.device)
                if self.invariant_module == 'gcn_mean':
                    hidden = masked_mean(hidden, graph_mask)
                elif self.invariant_module == 'gcn_max':
                    hidden = masked_max(hidden, graph_mask)
                elif self.invariant_module == 'gcn':
                    # No pooling. Return embeddings of all nodes
                    pass
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
            if torch.sum(torch.isnan(hidden)):
                print('')
            return hidden, graph_mask, word_encoding, word_encoding_mask, adj_mask, noun_mask
        else:
            ValueError("Undefined instruction architecture: {}".format(self.use_instr))

    def _compute_graph_mask(self, n_nodes, max_n_node, device):
        """

        :param n_nodes: bz tensor
        :return: graph_mask bz x max_n_node tensor
        """

        bz = n_nodes.size()[0]
        """
        if torch.sum(n_nodes) > 2 * bz:
            print('')
        graph_mask = torch.zeros(bz, max_n_node, device=device)
        ii = [[i] * k for i, k in enumerate(n_nodes)]
        jj = [list(range(k)) for i, k in enumerate(n_nodes)]
        ii_ = sum(ii, [])
        jj_ = sum(jj, [])
        graph_mask[ii_, jj_] = 1
        """

        graph_mask2 = torch.zeros(bz, max_n_node, device=device)
        graph_mask2[torch.arange(bz), n_nodes.view(-1).long() - 1] = 1
        graph_mask2 = 1 - graph_mask2.cumsum(dim=-1)
        graph_mask2[torch.arange(bz), n_nodes.view(-1).long() - 1] = 1
        return graph_mask2

        #assert torch.equal(graph_mask, graph_mask2)


        #graph_mask = torch.ones(bz, max_n_node, device=device)
        #return graph_mask

    def _prev_adj(self, noun_inds, vocab, word_inds):
        """

        :param noun_inds:  bz x 1
        :param vocab:  Vocabulary
        :param word_inds: bz x # of node * # of word in each node
        :return: probability dist over vocab bz x vocab_size, None
        """
        prev_inds = word_inds[range(noun_inds.size()[0]), noun_inds.view(-1) - 1]
        prev_inds[prev_inds > vocab.adj_idx] = -1
        prev_inds[prev_inds <= vocab.noun_idx] = -1
        prev_inds[prev_inds == -1] = vocab.vocab['none']
        prev_inds = prev_inds.view(-1)
        dist = torch.zeros(word_inds.size()[0], vocab.max_size, device=word_inds.device)
        dist[range(prev_inds.size()[0]), prev_inds] = 1
        return dist, None


class TwoStageModel(nn.Module):
    def __init__(self, ac_model, n_action=8):
        super().__init__()
        self.ac_model = ac_model
        self.actor_query = nn.Sequential(
            nn.Linear(ac_model.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, n_action)
        )
        self.critic = nn.Sequential(
            nn.Linear(ac_model.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.n_action = n_action
        for param in self.ac_model.parameters():
            param.requires_grad = False

        self.action_space = gym.spaces.MultiDiscrete(
            [n_action, self.ac_model.action_space.n])
        self.memory_size = ac_model.memory_size

    def forward(self, obs, memory, instr_embedding=None):
        with torch.no_grad():
            model_results = self.ac_model(obs, memory)
        embedding = model_results['embedding']
        x = model_results['x']
        x_q = self.actor_query(embedding)
        multi_categorical = multi_categorical_maker([self.n_action, self.ac_model.action_space.n])
        dist = multi_categorical(logits=torch.cat([F.log_softmax(x_q, dim=1), F.log_softmax(x, dim=1)], dim=1))
        v = self.critic(embedding)
        value = v.squeeze(1)
        model_results['dist'] = dist
        model_results['value'] = value
        return model_results


