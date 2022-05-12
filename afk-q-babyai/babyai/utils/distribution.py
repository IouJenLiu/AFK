from torch.distributions import Distribution
import torch
from torch.distributions.categorical import Categorical

class MultiCategorical(Distribution):

    def __init__(self, dists):
        super().__init__()
        self.dists = dists
        self.n_basic_action = 7
        self.probs = torch.cat([d.probs for d in dists], dim=1)

    def log_prob(self, value):
        ans = []
        # debug
        mask = torch.zeros(value.size()[0], 1)
        for i, (d, v) in enumerate(zip(self.dists, torch.split(value, 1, dim=-1))):
            log_prob = d.log_prob(v.squeeze(-1))
            #if i == 0:
            #    mask[v < self.n_basic_action] = 1
            #else:
            #    log_prob[mask.view(-1) == 0] = 0
            ans.append(log_prob)
            #ans.append(d.log_prob(v.squeeze(-1)))
        return torch.stack(ans, dim=-1).sum(dim=-1)

    def entropy(self):
        return torch.stack([d.entropy() for d in self.dists], dim=-1).sum(dim=-1)

    def sample(self, sample_shape=torch.Size()):
        return torch.stack([d.sample(sample_shape) for d in self.dists], dim=-1)

    def mode(self):
        ans = []
        for d in self.dists:
            ans.append(d.probs.argmax(1))
        return torch.stack(ans, dim=1)

    #def probs(self):
    #    ans = []
    #    for d in self.dists:
    #        ans.append(d.probs())
    #    return torch.stack(ans, dim=-1)


def multi_categorical_maker(nvec):
    def get_multi_categorical(logits=None, probs=None):
        start = 0
        ans = []
        if logits is not None:
            for n in nvec:
                ans.append(Categorical(logits=logits[:, start: start + n]))
                start += n
        else:
            for n in nvec:
                ans.append(Categorical(probs=probs[:, start: start + n]))
                start += n
        return MultiCategorical(ans)
    return get_multi_categorical