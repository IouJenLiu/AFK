import copy
import numpy as np
from generic import tokenize, _word_to_id, _words_to_ids


class SimpleGraph(object):

    def __init__(self, word2id, stopwords=set([]), setting="noquery", available_adjs=[], available_nouns=[]):
        self.node_capacity = 0
        self.word2id = word2id
        self.stopwords = stopwords
        self.setting = setting
        self.available_adjs = available_adjs
        self.available_nouns = available_nouns
        self.available_adj_set = set(available_adjs)
        self.available_noun_set = set(available_nouns)
        self.available_adj_ids = set(
            _words_to_ids(self.available_adjs, self.word2id))
        self.available_noun_ids = set(
            _words_to_ids(self.available_nouns, self.word2id))

    def push_batch(self, new_sentence_list):
        assert len(new_sentence_list) == len(self.nodes)  # batch size
        kg_expanded_list = []
        for b in range(len(new_sentence_list)):
            kg_expanded = self.push_one(b, new_sentence_list[b])
            kg_expanded_list.append(kg_expanded)
        return kg_expanded_list

    def has_common_token(self, new_token_set, list_of_token_sets):
        new_token_set = set(new_token_set)
        for item in list_of_token_sets:
            common_tokens = new_token_set & set(item)
            if len(common_tokens) > 0 and (len(common_tokens & self.available_adj_set) > 0 or len(common_tokens & self.available_noun_set) > 0):
                return True
        return False

    def push_one(self, b, new_sentence):
        if self.setting in ["noquery", "query"]:
            return 0.0

        if "cannot be cut nor cooked" in new_sentence or \
            "won't find any" in new_sentence or \
                "don't understand the question" in new_sentence:
            # queried something unknown
            return 0.0

        tokens = tokenize(new_sentence)
        tokens_wo_stopwords = sorted(
            list(set([item for item in tokens if item not in self.stopwords])))
        if tokens_wo_stopwords in self.nodes_wo_stopwords[b]:
            # duplicate
            return 0.0

        if not self.has_common_token(tokens_wo_stopwords, self.nodes_wo_stopwords[b]):
            # no common token, do not push new sent into the graph
            return 0.0

        self.nodes[b].append(tokens)
        self.nodes_wo_stopwords[b].append(tokens_wo_stopwords)
        if len(self.nodes[b]) > self.node_capacity:
            self.nodes[b] = self.nodes[b][-self.node_capacity:]
            self.nodes_wo_stopwords[b] = self.nodes_wo_stopwords[b][-self.node_capacity:]
        return 1.0

    def get_verb_masked_ids(self):
        batch_size = len(self.nodes_wo_stopwords)
        action_verbs = ["take", "open", "slice", "dice", "chop"]
        query_verbs = ["where", "how"]
        masked_ids = [[] for _ in range(batch_size)]

        if self.setting in ["noquery"]:
            for word in action_verbs:
                for i in range(batch_size):
                    masked_ids[i].append(_word_to_id(word, self.word2id))
        elif self.setting in ["query", "ours", "ours_note"]:
            for word in action_verbs + query_verbs:
                for i in range(batch_size):
                    masked_ids[i].append(_word_to_id(word, self.word2id))
        else:
            raise NotImplementedError
        return masked_ids

    def get_adj_masked_ids(self, surroundings_token_ids):
        batch_size = len(self.nodes_wo_stopwords)

        masked_ids = [[] for _ in range(batch_size)]
        for b in range(batch_size):
            available_ids = surroundings_token_ids[b]
            available_ids.add(self.word2id["none"])

            if self.setting in ["ours"]:
                _notebook_token_ids = []
                for sent in self.nodes_wo_stopwords[b]:
                    _notebook_token_ids += _words_to_ids(sent, self.word2id)
                available_ids = available_ids | set(_notebook_token_ids)
            elif self.setting in ["ours_note"]:
                _notebook_token_ids = []
                for sent in self.nodes_wo_stopwords[b]:
                    _notebook_token_ids += _words_to_ids(sent, self.word2id)
                available_ids = set(_notebook_token_ids)
                available_ids.add(self.word2id["none"])

            masked_ids[b] = list(available_ids & self.available_adj_ids)
        return masked_ids

    def get_noun_masked_ids(self, surroundings_token_ids):
        batch_size = len(self.nodes_wo_stopwords)

        masked_ids = [[] for _ in range(batch_size)]
        for b in range(batch_size):
            available_ids = surroundings_token_ids[b]

            if self.setting in ["ours"]:
                _notebook_token_ids = []
                for sent in self.nodes_wo_stopwords[b]:
                    _notebook_token_ids += _words_to_ids(sent, self.word2id)
                available_ids = available_ids | set(_notebook_token_ids)
            elif self.setting in ["ours_note"]:
                _notebook_token_ids = []
                for sent in self.nodes_wo_stopwords[b]:
                    _notebook_token_ids += _words_to_ids(sent, self.word2id)
                available_ids = set(_notebook_token_ids)

            masked_ids[b] = list(available_ids & self.available_noun_ids)

        return masked_ids

    def masked_id_to_masks(self, batch_size, masked_ids_list):
        verb_masked_ids, adj_masked_ids, noun_masked_ids = masked_ids_list
        vocab_mask = np.zeros((batch_size, 3, len(self.word2id)), dtype="float32")
        for b in range(batch_size):
            for w_id in verb_masked_ids[b]:
                vocab_mask[b][0][w_id] = 1.0
            for w_id in adj_masked_ids[b]:
                vocab_mask[b][1][w_id] = 1.0
            for w_id in noun_masked_ids[b]:
                vocab_mask[b][2][w_id] = 1.0
        return vocab_mask

    def get_vocabulary_mask(self, obs, objectives):
        surroundings_token_ids = [set(_obs + _obj)
                                  for _obs, _obj in zip(obs, objectives)]

        verb_masked_ids = self.get_verb_masked_ids()
        adj_masked_ids = self.get_adj_masked_ids(surroundings_token_ids)
        noun_masked_ids = self.get_noun_masked_ids(surroundings_token_ids)

        vocab_masks = self.masked_id_to_masks(len(self.nodes_wo_stopwords), [verb_masked_ids, adj_masked_ids, noun_masked_ids])
        return vocab_masks, [verb_masked_ids, adj_masked_ids, noun_masked_ids]

    def get(self):
        if self.setting in ["noquery", "query"]:
            return None
        else:
            return copy.deepcopy(self.nodes)

    def reset(self, node_capacity, task_description):
        # capacity is the max number of node an agent can store
        assert node_capacity >= 1
        self.node_capacity = node_capacity
        assert task_description is not None and len(task_description) > 0
        batch_size = len(task_description)
        self.nodes, self.nodes_wo_stopwords = [], []
        for i in range(batch_size):
            tokens = tokenize(task_description[i])
            tokens_wo_stopwords = set(
                [item for item in tokens if item not in self.stopwords])
            self.nodes.append([tokens])
            self.nodes_wo_stopwords.append([tokens_wo_stopwords])
