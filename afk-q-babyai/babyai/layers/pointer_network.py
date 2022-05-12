from babyai.layers.aggrerator import MultiHeadAttention
import torch
import numpy as np

class Pointer(torch.nn.Module):
    def __init__(self, hidden_dim=128, q_dim=128):
        super(Pointer, self).__init__()
        self.pointer = MultiHeadAttention(block_hidden_dim=hidden_dim, n_head=1, q_dim=q_dim)

    def att2dist(self, word_inds, vocab, attn):
        """
        :param word_inds: batch x len2
        :param vocab: a dictionary of vocab (e.g., {'go':1, 'to':2, 'mary':3,.....})
        :param attn: batch x len2 (e.g. batch x (max_n_node * instr_len))
        :return:
        """
        """
        dist = torch.zeros(word_inds.size()[0], vocab.max_size, device=attn.device)
        # TODO: Vectorize the following.
        for k, v in vocab.vocab.items():
            mask = (word_inds == v)
            prob = attn * mask
            prob = torch.sum(prob, dim=1, keepdim=True)
            dist[:, v - 1] = prob.view(-1) # vocab index start with 1
        """
        bz, max_sen_len = word_inds.size()
        a = torch.zeros(bz, vocab.max_size, max_sen_len, device=attn.device)
        a[list(np.repeat(range(bz), max_sen_len)) ,word_inds.view(-1) - 1, list(range(max_sen_len)) * bz] = 1
        dist2 = torch.bmm(a, attn.view(bz, -1, 1)).view(bz, -1)
        return dist2


        #return dist

    def forward(self, input_1, input_2, mask_1, mask_2, word_inds, vocab):
        """
        :param input_1: batch x len_1 x hid (e.g., an embedding can be size of batch x len_1 x hid)
        :param input_2: batch x len_2 x hid (e.g., a word encoding of the KG can be size of batch x (max_n_node * instr_len) x hid)
        :param mask_1: batch x len_1
        :param mask_2: batch x len_2
        :param word_inds:  batch x len_2
        :param vocab: a dictionary of vocab (e.g., {'go':1, 'to':2, 'mary':3,.....})
        :return:
        """
        squared_mask = torch.bmm(mask_1.unsqueeze(-1), mask_2.unsqueeze(1))
        _, attn = self.pointer(input_1, squared_mask, input_2, input_2)
        dist = self.att2dist(word_inds, vocab, attn.squeeze(1))
        return dist, attn


