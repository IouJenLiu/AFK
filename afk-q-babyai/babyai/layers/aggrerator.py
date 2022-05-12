import torch
import numpy as np
import torch.nn.functional as F


def masked_softmax(x, m=None, axis=-1):
    '''
    x: batch x time x hid
    m: batch x time (optional)
    '''
    x = torch.clamp(x, min=-15.0, max=15.0)
    if m is not None:
        m = m.float()
        x = x * m
    e_x = torch.exp(x - torch.max(x, dim=axis, keepdim=True)[0])
    if m is not None:
        e_x = e_x * m
    softmax = e_x / (torch.sum(e_x, dim=axis, keepdim=True) + 1e-6)
    return softmax


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

    def __init__(self, block_hidden_dim, n_head, dropout=0.1, q_dim=128):
        super().__init__()
        self.q_dim = q_dim
        self.n_head = n_head
        self.block_hidden_dim = block_hidden_dim
        self.w_qs = torch.nn.Linear(q_dim, n_head * block_hidden_dim, bias=False)
        self.w_ks = torch.nn.Linear(block_hidden_dim, n_head * block_hidden_dim, bias=False)
        self.w_vs = torch.nn.Linear(block_hidden_dim, n_head * block_hidden_dim, bias=False)
        torch.nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (q_dim * 2)))
        torch.nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (block_hidden_dim * 2)))
        torch.nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (block_hidden_dim * 2)))
        self.attention = ScaledDotProductAttention(temperature=np.power(block_hidden_dim, 0.5))
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
        # attn: batch x len_q x len_k
        batch_size, len_q = q.size(0), q.size(1)
        len_k, len_v = k.size(1), v.size(1)
        assert mask.size(1) == len_q
        assert mask.size(2) == len_k
        residual = q

        q = self.w_qs(q).view(batch_size, len_q, self.n_head, self.block_hidden_dim)
        k = self.w_ks(k).view(batch_size, len_k, self.n_head, self.block_hidden_dim)
        v = self.w_vs(v).view(batch_size, len_v, self.n_head, self.block_hidden_dim)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, self.block_hidden_dim) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, self.block_hidden_dim) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, self.block_hidden_dim) # (n*b) x lv x dv

        mask = mask.repeat(self.n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)
        attn = attn.view(self.n_head, batch_size, len_q, -1)
        attn = torch.mean(attn, 0)  # batch x lq x lk
        output = None

        return output, attn