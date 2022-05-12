import os
import json
import numpy
import re
import torch
import babyai.rl

from .. import utils
import copy

def get_vocab_path(model_name):
    return os.path.join(utils.get_model_dir(model_name), "vocab.json")


class Vocabulary:
    def __init__(self, model_name, max_size=30, file_path='../babyai/vocab/vocab1.txt'):
        self.file_path = file_path
        self.path = get_vocab_path(model_name)

        if os.path.exists(self.path):
            self.vocab = json.load(open(self.path))
            # TODO: for debug, need to remove
            self.noun_idx, self.adj_idx = 24, 46
        else:
            self.vocab = {}
        if not self.vocab:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            for word in lines:
                token, part_of_speech = word.split()
                self.vocab[token] = len(self.vocab) + 1
                if part_of_speech == 'noun':
                    self.noun_idx = len(self.vocab)
                if part_of_speech == 'adj':
                    self.adj_idx = len(self.vocab)
        self.max_size = len(self.vocab) + 2
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def __getitem__(self, token):
        if not (token in self.vocab.keys()):
            print("UnSEEN!!!", token)
            raise NotImplementedError
            if len(self.vocab) >= self.max_size:
                print(token)
                raise ValueError("Maximum vocabulary capacity reached")
            old_vocab_len = len(self.vocab)
            self.vocab[token] = old_vocab_len + 1
            self.inverse_vocab[old_vocab_len + 1] = token

        return self.vocab[token]

    def save(self, path=None):
        if path is None:
            path = self.path
        utils.create_folders_if_necessary(path)
        json.dump(self.vocab, open(path, "w"))

    def copy_vocab_from(self, other):
        '''
        Copy the vocabulary of another Vocabulary object to the current object.
        '''
        self.vocab.update(other.vocab)
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}


class InstructionsPreprocessor(object):
    def __init__(self, model_name, load_vocab_from=None):
        self.model_name = model_name
        self.vocab = Vocabulary(model_name)
        self.colors = ['blue', 'green', 'grey']
        path = get_vocab_path(model_name)
        if not os.path.exists(path) and load_vocab_from is not None:
            # self.vocab.vocab should be an empty dict
            secondary_path = get_vocab_path(load_vocab_from)
            if os.path.exists(secondary_path):
                old_vocab = Vocabulary(load_vocab_from)
                self.vocab.copy_vocab_from(old_vocab)
            else:
                raise FileNotFoundError('No pre-trained model under the specified name')

    def __call__(self, obss, device=None, field="mission"):
        raw_instrs = []
        max_instr_len = 0

        for obs in obss:
            tokens = re.findall("([a-z0-9]+)", obs[field].lower())
            self.preprocess(tokens)
            instr = numpy.array([self.vocab[token] for token in tokens])
            raw_instrs.append(instr)
            max_instr_len = max(len(instr), max_instr_len)

        instrs = numpy.zeros((len(obss), max_instr_len))

        for i, instr in enumerate(raw_instrs):
            instrs[i, :len(instr)] = instr

        instrs = torch.tensor(instrs, device=device, dtype=torch.long)
        return instrs

    def preprocess(self, ans):
        #for i, a in enumerate(ans):
        #    if ans[i] in self.colors or ans[i] == 'favorite' or ans[i] == 'go':
        #        ans[i] = ans[i] + ' ' + ans[i + 1]
        #        ans.pop(i + 1)
        if 'is' in ans:
            ans.remove('is')
        if 'in' in ans:
            ans.remove('in')


class KGPreprocessor(object):
    def __init__(self, model_name, load_vocab_from=None):
        self.model_name = model_name
        self.vocab = Vocabulary(model_name)

        path = get_vocab_path(model_name)
        if not os.path.exists(path) and load_vocab_from is not None:
            # self.vocab.vocab should be an empty dict
            secondary_path = get_vocab_path(load_vocab_from)
            if os.path.exists(secondary_path):
                old_vocab = Vocabulary(load_vocab_from)
                self.vocab.copy_vocab_from(old_vocab)
            else:
                raise FileNotFoundError('No pre-trained model under the specified name')

    def __call__(self, obss, device=None):
        raw_instrs = [[] for _ in range(len(obss))]
        max_instr_len = 0
        max_n_node = 0
        for i, obs in enumerate(obss):
            max_n_node = max(max_n_node, len(obs['kg_cc']))
            for node_rep in obs['kg_cc']:
                tokens = re.findall("([a-z0-9]+)", node_rep.lower())
                instr = numpy.array([self.vocab[token] for token in tokens])
                raw_instrs[i].append(instr)
                max_instr_len = max(len(instr), max_instr_len)
        instrs = numpy.zeros((len(obss), max_n_node, max_instr_len))

        for i in range(len(raw_instrs)):
            for j in range(len(raw_instrs[i])):
                instrs[i, j, :len(raw_instrs[i][j])] = raw_instrs[i][j]
        instrs = torch.tensor(instrs, device=device, dtype=torch.long)
        return instrs


class RawImagePreprocessor(object):
    def __call__(self, obss, device=None):
        images = numpy.array([obs["image"] for obs in obss])
        images = torch.tensor(images, device=device, dtype=torch.float)
        return images


class IntImagePreprocessor(object):
    def __init__(self, num_channels, max_high=255):
        self.num_channels = num_channels
        self.max_high = max_high
        self.offsets = numpy.arange(num_channels) * max_high
        self.max_size = int(num_channels * max_high)

    def __call__(self, obss, device=None):
        images = numpy.array([obs["image"] for obs in obss])
        # The padding index is 0 for all the channels
        images = (images + self.offsets) * (images > 0)
        images = torch.tensor(images, device=device, dtype=torch.long)
        return images


class ObssPreprocessor:
    def __init__(self, model_name, obs_space=None, load_vocab_from=None, query=False, onehot_ans=False):
        self.image_preproc = RawImagePreprocessor()
        self.instr_preproc = InstructionsPreprocessor(model_name, load_vocab_from)
        self.kg_preproc = KGPreprocessor(model_name, load_vocab_from)
        self.vocab = self.instr_preproc.vocab
        self.obs_space = {
            "image": 147,
            "instr": self.vocab.max_size
        }
        self.ans_preproc = AnsOnehotPreprocessor(model_name)
        self.onehot_ans = onehot_ans
        if query:
            self.obs_space["ans"] = self.vocab.max_size


    def __call__(self, obss, device=None):
        obs_ = babyai.rl.DictList()

        if "image" in self.obs_space.keys():
            obs_.image = self.image_preproc(obss, device=device)

        if "instr" in self.obs_space.keys():
            obs_.instr = self.instr_preproc(obss, device=device)

        if "ans" in self.obs_space.keys():
            if self.onehot_ans:
                obs_.ans = self.ans_preproc(obss, device=device, field="ans")
            else:
                obs_.ans = self.instr_preproc(obss, device=device, field="ans")

        if "kg_cc" in obss[0]:
            obs_.kg_cc = self.kg_preproc(obss, device=device)


        return obs_


class IntObssPreprocessor(object):
    def __init__(self, model_name, obs_space, load_vocab_from=None):
        image_obs_space = obs_space.spaces["image"]
        self.image_preproc = IntImagePreprocessor(image_obs_space.shape[-1],
                                                  max_high=image_obs_space.high.max())
        self.instr_preproc = InstructionsPreprocessor(load_vocab_from or model_name)
        self.vocab = self.instr_preproc.vocab
        self.obs_space = {
            "image": self.image_preproc.max_size,
            "instr": self.vocab.max_size
        }

    def __call__(self, obss, device=None):
        obs_ = babyai.rl.DictList()

        if "image" in self.obs_space.keys():
            obs_.image = self.image_preproc(obss, device=device)

        if "instr" in self.obs_space.keys():
            obs_.instr = self.instr_preproc(obss, device=device)

        return obs_


class ActionPostprocessor(object):
    def __init__(self, restricted=False, res_actions=None):
        if restricted:
            self.actions = res_actions[0]
            self.action_colors = res_actions[1]
            self.action_objects = res_actions[2]
        else:
            self.actions = ['left', 'right', 'forward', 'pickup', 'drop', 'toggle', 'done', 'where', 'what']
            self.action_colors = ['None', 'red', 'green', 'blue', 'purple', 'yellow', 'gray']
            self.action_objects = ['box', 'key', 'ball', 'room0',
                                   'room1', 'room2', 'room3', 'room4',
                                   'room5', 'room6', 'room7', 'room8']

    def __call__(self, action, device=None):
        res = []
        # TODO: for loop could be vectorized.
        for i in range(action.size()[0]):
            if action[i][1] == 0:
                # None color
                res.append([self.actions[action[i][0]], self.action_objects[action[i][2]]])
            else:
                res.append([self.actions[action[i][0]], self.action_colors[action[i][1]], self.action_objects[action[i][2]]])
        return res


class FlatActionPostprocessor(object):
    def __init__(self, restricted=False, env=None):
        if env == 'BabyAI-GoToFavorite-v0':
            self.actions = [['left'], ['right'], ['forward'], ['pickup'], ['drop'], ['toggle'], ['done'],
                            ['where', 'blue', 'ball'], ['where', 'blue', 'key'], ['where', 'blue', 'box'],
                            ['where', 'green', 'ball'], ['where', 'green', 'key'], ['where', 'green', 'box'],
                            ['what', 'is', 'jack', 'favorite', 'toy'], ['what', 'is', 'mary', 'favorite', 'toy'], #['what', 'is', 'adam', 'favorite', 'toy'],
                            ['where', 'green', 'room'], ['what', 'ball'],
                            ['is', 'blue', 'box', 'open'], ['is', 'green', 'box', 'open'],
                            ['is', 'blue', 'key', 'around'], ['is', 'green', 'key', 'around'],
                            ['is', 'blue', 'ball', 'basketball'], ['is', 'green', 'ball', 'basketball'],
                            ['where', 'green', 'room'], ['what', 'ball'], ['where', 'green', 'key'], ['where', 'green', 'box'],
                            ['where', 'blue', 'room0'], ['where', 'blue', 'room1'], ['where', 'blue', 'key'],
                            ['what', 'ball'], ['what', 'room0'], ['what', 'room1'], ['what', 'key'],
                            ['what', 'blue', 'ball'], ['what', 'blue', 'room0'], ['what', 'blue', 'room1'], ['what', 'blue', 'key'],
                            ]
        elif 'ObjInLockedBox' in env or 'ObjInBox' in env:
            self.actions = [
                ['left'], ['right'], ['forward'], ['pickup'], ['drop'], ['toggle'], ['done'],
                ['where', 'blue', 'ball'], ['where', 'green', 'ball'], ['where', 'grey', 'ball'],
                ['which', 'key', 'to', 'blue', 'box'], ['which', 'key', 'to', 'green', 'box'], ['which', 'key', 'to', 'grey', 'box'],

            ]
        else:
            raise NotImplementedError

    def __call__(self, action, device=None):
        res = []
        # TODO: for loop could be vectorized.
        for i in range(action.size()[0]):
            res.append(self.actions[action[i]])
        return res


class TwoStageActionPostprocessor(object):
    def __init__(self):
        self.actions = [['left'], ['right'], ['forward'], ['pickup'], ['drop'], ['toggle'], ['done']]
        self.action_query = [['None'], ['where', 'ball'], ['where', 'room0'], ['where', 'room2'], ['where', 'key'],
                             ['where', 'blue', 'room2'], ['where', 'yellow', 'ball'], ['where', 'red', 'room2']]
        self.steps = [0] * 256

    def __call__(self, action, device=None):
        res = []
        # TODO: for loop could be vectorized.
        for i in range(action.size()[0]):
            #if action[i][0] == 0 or self.steps[i] % 30 != 0:
            if action[i][0] == 0:
                res.append(self.actions[action[i][1]])
            else:
                res.append(self.action_query[action[i][0]])
            self.steps[i] += 1
        return res


class AnsOnehotPreprocessor(object):
    def __init__(self, model_name):
        self.model_name = model_name
        self.vocab = Vocabulary(model_name)
        self.tokens = ['none', 'room0', 'room2', 'east', 'west']

    def __call__(self, obss, device=None, field="mission"):
        instrs = torch.zeros(len(obss), len(self.tokens), device=device)
        for i, obs in enumerate(obss):
            ans = re.findall("([a-z0-9]+)", obs[field].lower())
            for j, token in enumerate(self.tokens):
                if token in ans:
                    instrs[i, j] = 1
                    break

        return instrs


class PointerActionPostprocessor(object):
    def __init__(self, vocab=None, model_name=None):
        if not vocab:
            self.vocab = Vocabulary(model_name)
        else:
            self.vocab = vocab
        self.move_actions = [['left'], ['right'], ['forward'], ['pickup'], ['drop'], ['toggle'], ['done']]
        self.wh_actions = [['what', 'is'], ['where', 'is']]

    def __call__(self, action, device=None):
        """
        :param action: bz x n_head e.g. heads = [n_bin, n_moving_action, n_wh, n_adj, n_noun]
        :param device:
        :return:
        """
        res = []
        # TODO: for loop could be vectorized.
        """
        for i in range(action.size()[0]):
            a = action[i].item()
            if a < len(self.move_actions):
                res.append(self.move_actions[a])
            else:
                a = a - len(self.move_actions) + 1
                res.append(self.vocab.inverse_vocab[a])
        """

        for i in range(action.size()[0]):
            a = action[i]
            if a[0] == 0:
                res.append(self.move_actions[a[1]])
            else:
                wh = self.wh_actions[a[2]]
                adj = [self.vocab.inverse_vocab[a[3].item() + 1]]
                noun = [self.vocab.inverse_vocab[a[4].item() + 1]]
                if adj[0] == 'none':
                    res.append(wh + noun)
                else:
                    res.append(wh + adj + noun)

        #res = [['forward']] * action.size()[0]
        return res
