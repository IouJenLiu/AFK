import torch
import math
import numpy as np
import torch.nn.functional as F


def masked_log_softmax(vector: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    https://github.com/allenai/allennlp/blob/b6cc9d39651273e8ec2a7e334908ffa9de5c2026/allennlp/nn/util.py#L272
    ``torch.nn.functional.log_softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a log_softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular log_softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    In the case that the input vector is completely masked, the return value of this function is
    arbitrary, but not ``nan``.  You should be masking the result of whatever computation comes out
    of this in that case, anyway, so the specific values returned shouldn't matter.  Also, the way
    that we deal with this case relies on having single-precision floats; mixing half-precision
    floats with fully-masked vectors will likely give you ``nans``.
    If your logits are all extremely negative (i.e., the max value in your logit vector is -50 or
    lower), the way we handle masking here could mess you up.  But if you've got logit values that
    extreme, you've got bigger problems than this.
    """
    if mask is not None:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        # vector + mask.log() is an easy way to zero out masked elements in logspace, but it
        # results in nans when the whole vector is masked.  We need a very small value instead of a
        # zero in the mask for these cases.  log(1 + 1e-45) is still basically 0, so we can safely
        # just add 1e-45 before calling mask.log().  We use 1e-45 because 1e-46 is so small it
        # becomes 0 - this is just the smallest value we can actually use.
        vector = vector + (mask + 1e-45).log()
    return torch.nn.functional.log_softmax(vector, dim=dim)


def masked_softmax(vector: torch.Tensor,
                   mask: torch.Tensor,
                   dim: int = -1,
                   memory_efficient: bool = False,
                   mask_fill_value: float = -1e32) -> torch.Tensor:
    """
    https://github.com/allenai/allennlp/blob/b6cc9d39651273e8ec2a7e334908ffa9de5c2026/allennlp/nn/util.py#L231
    ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    If ``memory_efficient`` is set to true, we will simply use a very large negative number for those
    masked positions so that the probabilities of those positions would be approximately 0.
    This is not accurate in math, but works for most cases and consumes less memory.
    In the case that the input vector is completely masked and ``memory_efficient`` is false, this function
    returns an array of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of
    a model that uses categorical cross-entropy loss. Instead, if ``memory_efficient`` is true, this function
    will treat every element as equal, and do softmax over equal numbers.
    """
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = torch.nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = vector.masked_fill((1 - mask).byte(), mask_fill_value)
            result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result


def masked_mean(_input, _mask):
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


# Residual block
class ResidualBlock(torch.nn.Module):
    def __init__(self, block_hidden_dim, kernel_size):
        super(ResidualBlock, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=block_hidden_dim, out_channels=block_hidden_dim,
                                     kernel_size=kernel_size, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(block_hidden_dim)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(in_channels=block_hidden_dim, out_channels=block_hidden_dim,
                                     kernel_size=kernel_size, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(block_hidden_dim)

    def forward(self, x):
        residual = x  # batch x hid x w x h
        out = self.conv1(x)  # batch x hid x w x h
        out = self.bn1(out)  # batch x hid x w x h
        out = self.relu(out)  # batch x hid x w x h
        out = self.conv2(out)  # batch x hid x w x h
        out = self.bn2(out)  # batch x hid x w x h
        out += residual  # batch x hid x w x h
        out = self.relu(out)  # batch x hid x w x h
        return out


# From https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / \
            torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class ImageBOWEmbedding(torch.nn.Module):
    # Modified from https://github.com/mila-iqia/babyai/blob/863f3529371ba45ef0148a48b48f5ae6e61e06cc/babyai/model.py#L48
    def __init__(self, max_vocabulary_size, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_vocabulary_size = max_vocabulary_size
        self.embedding = torch.nn.Embedding(
            3 * max_vocabulary_size, embedding_dim)
        # self.mapping_channels = torch.nn.Linear(3 * embedding_dim, embedding_dim)
        self.apply(initialize_parameters)

    def forward(self, inputs):
        # input: batch x height x width x channel (3 in babyai)
        offsets = torch.Tensor(
            [0, self.max_vocabulary_size, 2 * self.max_vocabulary_size]).to(inputs.device)
        inputs = (inputs + offsets[None, None, None, :]).long()
        res = self.embedding(inputs)  # batch x height x width x channel x emb
        res = res.sum(3)  # batch x height x width x emb
        # res = res.view(res.size(0), res.size(1), res.size(2), -1)  # batch x h x w x emb*channel
        # res = torch.tanh(self.mapping_channels(res))  # batch x h x w x emb
        res = res.permute(0, 3, 1, 2)  # batch x emb x height x width
        return res


class WordEmbedding(torch.nn.Module):
    '''
    inputs: x:          batch x ...
    outputs:embedding:  batch x ... x emb
            mask:       batch x ...
    '''

    def __init__(self, embedding_size, vocab_size, dropout_rate=0.0, trainable=True, embedding_oov_init='random',
                 padding_idx=0):
        super(WordEmbedding, self).__init__()
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.dropout_rate = dropout_rate
        self.embedding_oov_init = embedding_oov_init
        self.trainable = trainable
        self.padding_idx = padding_idx
        self.embedding_layer = torch.nn.Embedding(
            self.vocab_size, self.embedding_size, padding_idx)
        self.init_weights()

    def init_weights(self):
        init_embedding_matrix = self.embedding_init()
        if self.embedding_layer.weight.is_cuda:
            init_embedding_matrix = init_embedding_matrix.cuda()
        self.embedding_layer.weight = torch.nn.Parameter(init_embedding_matrix)
        if not self.trainable:
            self.embedding_layer.weight.requires_grad = False

    def embedding_init(self):
        # Embeddings
        word_embedding_init = np.random.uniform(
            low=-0.05, high=0.05, size=(self.vocab_size, self.embedding_size))
        word_embedding_init[self.padding_idx, :] = 0
        word_embedding_init = torch.from_numpy(word_embedding_init).float()
        return word_embedding_init

    def compute_mask(self, x):
        mask = torch.ne(x, self.padding_idx).float()
        if x.is_cuda:
            mask = mask.cuda()
        return mask

    def forward(self, x):
        embeddings = self.embedding_layer(x)  # batch x time x emb
        embeddings = F.dropout(
            embeddings, p=self.dropout_rate, training=self.training)
        mask = self.compute_mask(x)  # batch x time
        return embeddings, mask


def PosEncoder(x, min_timescale=1.0, max_timescale=1.0e4):
    length = x.size(1)
    channels = x.size(2)
    signal = get_timing_signal(length, channels, min_timescale, max_timescale)
    if x.is_cuda:
        signal = signal.cuda()
    return x + signal


def get_timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    position = torch.arange(length).type(torch.float32)
    num_timescales = channels // 2
    log_timescale_increment = (math.log(
        float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
    inv_timescales = min_timescale * torch.exp(
        torch.arange(num_timescales).type(torch.float32) * -log_timescale_increment)
    scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
    m = torch.nn.ZeroPad2d((0, (channels % 2), 0, 0))
    signal = m(signal)
    signal = signal.view(1, length, channels)
    return signal


class DepthwiseSeparableConv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k, bias=True):
        super().__init__()
        self.depthwise_conv = torch.nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch,
                                              padding=k // 2, bias=False)
        self.pointwise_conv = torch.nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0,
                                              bias=bias)

    def forward(self, x, mask):
        # x: batch x length x h
        # mask: batch x length
        x = x.transpose(1, 2)
        res = self.depthwise_conv(x) * mask.unsqueeze(1)
        res = torch.relu(self.pointwise_conv(res)) * mask.unsqueeze(1)
        res = res.transpose(1, 2)
        return res


class SelfAttention(torch.nn.Module):
    def __init__(self, block_hidden_dim, n_head, dropout):
        super().__init__()
        self.block_hidden_dim = block_hidden_dim
        self.n_head = n_head
        self.dropout = dropout
        self.key_linear = torch.nn.Linear(
            block_hidden_dim, block_hidden_dim, bias=False)
        self.value_linear = torch.nn.Linear(
            block_hidden_dim, block_hidden_dim, bias=False)
        self.query_linear = torch.nn.Linear(
            block_hidden_dim, block_hidden_dim, bias=False)
        bias = torch.empty(1)
        torch.nn.init.constant_(bias, 0)
        self.bias = torch.nn.Parameter(bias)

    def forward(self, queries, query_mask, keys, values):

        query = self.query_linear(queries)
        key = self.key_linear(keys)
        value = self.value_linear(values)
        Q = self.split_last_dim(query, self.n_head)
        K = self.split_last_dim(key, self.n_head)
        V = self.split_last_dim(value, self.n_head)

        assert self.block_hidden_dim % self.n_head == 0
        key_depth_per_head = self.block_hidden_dim // self.n_head
        Q *= key_depth_per_head ** -0.5
        x = self.dot_product_attention(Q, K, V, mask=query_mask)
        return self.combine_last_two_dim(x.permute(0, 2, 1, 3))

    def dot_product_attention(self, q, k, v, bias=False, mask=None):
        """dot-product attention.
        Args:
        q: a Tensor with shape [batch, heads, length_q, depth_k]
        k: a Tensor with shape [batch, heads, length_kv, depth_k]
        v: a Tensor with shape [batch, heads, length_kv, depth_v]
        bias: bias Tensor (see attention_bias())
        is_training: a bool of training
        scope: an optional string
        Returns:
        A Tensor.
        """
        logits = torch.matmul(q, k.permute(0, 1, 3, 2))
        if bias:
            logits += self.bias
        if mask is not None:
            # shapes = [x if x != None else -1 for x in list(logits.size())]
            # mask = mask.view(shapes[0], 1, 1, shapes[-1])
            mask = mask.unsqueeze(1)
        weights = masked_softmax(logits, mask, -1)
        # dropping out the attention links for each of the heads
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        return torch.matmul(weights, v)

    def split_last_dim(self, x, n):
        """Reshape x so that the last dimension becomes two dimensions.
        The first of these two dimensions is n.
        Args:
        x: a Tensor with shape [..., m]
        n: an integer.
        Returns:
        a Tensor with shape [..., n, m/n]
        """
        old_shape = list(x.size())
        last = old_shape[-1]
        new_shape = old_shape[:-1] + [n] + [last // n if last else None]
        ret = x.view(new_shape)
        return ret.permute(0, 2, 1, 3)

    def combine_last_two_dim(self, x):
        """Reshape x so that the last two dimension become one.
        Args:
        x: a Tensor with shape [..., a, b]
        Returns:
        a Tensor with shape [..., ab]
        """
        old_shape = list(x.size())
        a, b = old_shape[-2:]
        new_shape = old_shape[:-2] + [a * b if a and b else None]
        ret = x.contiguous().view(new_shape)
        return ret


class EncoderBlock(torch.nn.Module):
    def __init__(self, conv_num, ch_num, k, block_hidden_dim, n_head, dropout):
        super().__init__()
        self.dropout = dropout
        self.convs = torch.nn.ModuleList(
            [DepthwiseSeparableConv(ch_num, ch_num, k) for _ in range(conv_num)])
        self.self_att = SelfAttention(block_hidden_dim, n_head, dropout)
        self.FFN_1 = torch.nn.Linear(ch_num, ch_num)
        self.FFN_2 = torch.nn.Linear(ch_num, ch_num)
        self.norm_C = torch.nn.ModuleList(
            [torch.nn.LayerNorm(block_hidden_dim) for _ in range(conv_num)])
        self.norm_1 = torch.nn.LayerNorm(block_hidden_dim)
        self.norm_2 = torch.nn.LayerNorm(block_hidden_dim)
        self.conv_num = conv_num

    def forward(self, x, mask, squared_mask, l, blks):
        total_layers = (self.conv_num + 2) * blks
        # conv layers
        out = PosEncoder(x)
        out = out * mask.unsqueeze(-1)
        for i, conv in enumerate(self.convs):
            res = out
            out = self.norm_C[i](out)
            if (i) % 2 == 0:
                out = F.dropout(out, p=self.dropout, training=self.training)
            out = conv(out, mask)
            out = self.layer_dropout(
                out, res, self.dropout * float(l) / total_layers)
            out = out * mask.unsqueeze(-1)
            l += 1
        res = out
        out = self.norm_1(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        # self attention
        out = self.self_att(out, squared_mask, out, out)
        out = self.layer_dropout(
            out, res, self.dropout * float(l) / total_layers)
        out = out * mask.unsqueeze(-1)
        l += 1
        res = out
        out = self.norm_2(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        # fully connected layers
        out = self.FFN_1(out)
        out = torch.relu(out)
        out = self.FFN_2(out)
        out = self.layer_dropout(
            out, res, self.dropout * float(l) / total_layers)
        out = out * mask.unsqueeze(-1)
        l += 1
        return out

    def layer_dropout(self, inputs, residual, dropout):
        if self.training == True:
            pred = torch.empty(1).uniform_(0, 1) < dropout
            if pred:
                return residual
            else:
                return F.dropout(inputs, dropout, training=self.training) + residual
        else:
            return inputs + residual


class ScaledDotProductAttention(torch.nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = torch.nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        attn = masked_softmax(attn, mask, 2)
        __attn = self.dropout(attn)
        output = torch.bmm(__attn, v)
        return output, attn


class MultiHeadAttention(torch.nn.Module):
    ''' From Multi-Head Attention module 
    https://github.com/jadore801120/attention-is-all-you-need-pytorch'''

    def __init__(self, block_hidden_dim, n_head, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.block_hidden_dim = block_hidden_dim
        self.w_qs = torch.nn.Linear(
            block_hidden_dim, n_head * block_hidden_dim, bias=False)
        self.w_ks = torch.nn.Linear(
            block_hidden_dim, n_head * block_hidden_dim, bias=False)
        self.w_vs = torch.nn.Linear(
            block_hidden_dim, n_head * block_hidden_dim, bias=False)
        torch.nn.init.normal_(self.w_qs.weight, mean=0,
                              std=np.sqrt(2.0 / (block_hidden_dim * 2)))
        torch.nn.init.normal_(self.w_ks.weight, mean=0,
                              std=np.sqrt(2.0 / (block_hidden_dim * 2)))
        torch.nn.init.normal_(self.w_vs.weight, mean=0,
                              std=np.sqrt(2.0 / (block_hidden_dim * 2)))
        self.attention = ScaledDotProductAttention(
            temperature=np.power(block_hidden_dim, 0.5))
        self.fc = torch.nn.Linear(n_head * block_hidden_dim, block_hidden_dim)
        self.layer_norm = torch.nn.LayerNorm(self.block_hidden_dim)
        torch.nn.init.xavier_normal_(self.fc.weight)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, q, mask, k, v):
        # q: batch x len_q x hid
        # k: batch x len_k x hid
        # v: batch x len_v x hid
        # mask: batch x len_q x len_k
        # output: batch x len_q x hid
        batch_size, len_q = q.size(0), q.size(1)
        len_k, len_v = k.size(1), v.size(1)
        assert mask.size(1) == len_q
        assert mask.size(2) == len_k
        residual = q

        q = self.w_qs(q).view(batch_size, len_q,
                              self.n_head, self.block_hidden_dim)
        k = self.w_ks(k).view(batch_size, len_k,
                              self.n_head, self.block_hidden_dim)
        v = self.w_vs(v).view(batch_size, len_v,
                              self.n_head, self.block_hidden_dim)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q,
                                                    self.block_hidden_dim)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k,
                                                    self.block_hidden_dim)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v,
                                                    self.block_hidden_dim)  # (n*b) x lv x dv

        mask = mask.repeat(self.n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)
        attn = attn.view(self.n_head, batch_size, len_q, -1)
        attn = torch.mean(attn, 0)  # batch x lq x lk

        output = output.view(self.n_head, batch_size,
                             len_q, self.block_hidden_dim)
        output = output.permute(1, 2, 0, 3).contiguous().view(
            batch_size, len_q, -1)  # b x lq x (n*dv)
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class NoisyLinear(torch.nn.Module):
    # Factorised NoisyLinear layer with bias
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = torch.nn.Parameter(
            torch.empty(out_features, in_features))
        self.weight_sigma = torch.nn.Parameter(
            torch.empty(out_features, in_features))
        self.register_buffer(
            'weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = torch.nn.Parameter(torch.empty(out_features))
        self.bias_sigma = torch.nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input):
        if self.training:
            return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)
