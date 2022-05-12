import re
from sent2vec.vectorizer import Vectorizer
from scipy import spatial


class KG(object):
    def __init__(self, mode='graph_overlap', n_gram=2):
        self.mode = mode
        self.graph = {}
        self.set = set()
        self.instr_node = None
        self.instr_CC_size = 0
        self.vectorizer = Vectorizer()
        if self.mode == 'graph_overlap':
            self.related_nodes_fn = self.get_all_related_nodes
        elif self.mode == 'graph_cosine':
            self.related_nodes_fn = self.get_all_related_nodes_cosine
        self.n_gram = n_gram

    def update(self, node):
        node1 = tuple(node)
        increased = False
        if self.mode == 'graph_overlap' or self.mode == 'graph_cosine':
            max_overlap = 0
            if node1 not in self.graph:
                related_nodes, max_overlap = self.related_nodes_fn(node1)
                self.graph[node1] = []
                for n in related_nodes:
                    self.graph[n].append(node1)
                    self.graph[node1].append(n)
            increased = False
            if len(self.getCC()) > self.instr_CC_size:
                self.instr_CC_size += 1
                increased = True
        elif self.mode == 'set':
            max_overlap = 0
            if node1 not in self.set:
                self.set.add(node1)
                increased = True
                max_overlap = 1
        return increased, max_overlap

    def is_related(self, node1, node2):
        # simple heuristic, need update
        for token in node1:
            if token in node2:
                return True
        return False

    def pre_process_n_gram(self, node1, n_gram):
        if n_gram <= 1:
            return node1
        p_node1 = []
        for i in range(len(node1) - n_gram + 1):
            n_gram_phrase = node1[i]
            for k in range(1, n_gram):
                if i + k >= len(node1):
                    break
                n_gram_phrase += " " + node1[i + k]
            p_node1.append(n_gram_phrase)
        return p_node1

    def n_overlap(self, node1, node2, n_gram=2):
        p_node1 = self.pre_process_n_gram(node1, n_gram)
        p_node2 = self.pre_process_n_gram(node2, n_gram)
        n = 0
        for token in p_node1:
            if token in p_node2:
                n += 1
        return n

    def get_all_related_nodes(self, node1):
        related_nodes = []
        max_overlap = 0
        for node2 in self.graph:
            if node2 == node1: continue
            n_overlap = self.n_overlap(node1, node2, n_gram=self.n_gram)
            if n_overlap > 0:
                related_nodes.append(node2)
                max_overlap = max(max_overlap, n_overlap)
        return related_nodes, max_overlap

    def get_all_related_nodes_cosine(self, node1):
        related_nodes = []
        max_overlap = 0
        n1 = node1[0]
        for s in node1[1:]:
            n1 = n1 + " " + s
        for node2 in self.graph:
            if node2 == node1: continue
            n2 = node2[0]
            for s in node2[1:]:
                n2 = n2 + " " + s
            self.vectorizer.bert([n1, n2])
            vectors = self.vectorizer.vectors
            n_overlap = spatial.distance.cosine(vectors[0], vectors[1])
            if n_overlap > 0.01:
                related_nodes.append(node2)
                max_overlap = max(max_overlap, n_overlap)
        return related_nodes, max_overlap


    def getCC(self):
        """
        :return: list of tuples which represents the connected component that contains the instr node.
        Ex. [('go', 'to', 'jack', 'favorite toy'), ('blue ball', 'room0')]
        """
        if self.mode == 'graph_overlap' or self.mode == 'graph_cosine':
            def bfs(node, graph):
                res = []
                visited = set()
                q = [node]
                visited.add(node)
                while q:
                    v = q.pop(0)
                    res.append(v)
                    for n in graph[v]:
                        if n not in visited:
                            q.append(n)
                            visited.add(n)
                return res
            return bfs(self.instr_node, self.graph)
        else:
            return list(self.set)

    def reset(self, node):
        # add none to instruction node of KG for no adj query
        # TODO: ugly, may need refactor
        self.instr_node = tuple(node)
        self.graph = {self.instr_node: []}
        self.set = set()
        self.set.add(self.instr_node)
        self.instr_CC_size = len(self.graph[self.instr_node]) + 1

    def __repr__(self):
        if self.mode == 'graph_overlap' or self.mode == 'graph_cosine':
            ret = ""
            for k, v in self.graph.items():
                ret += str(k) + ": " + str(v) + "\n"
            return ret
        else:
            return str(self.set)


