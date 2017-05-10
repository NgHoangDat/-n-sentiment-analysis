"""

"""
import math
import operator
import random
from time import time

from nltk.corpus import stopwords
import numpy as np

from RTNN.tree import Node, TreeNet
from six.moves import cPickle


class Model:

    # region init, save, load
    def __init__(self, dim, alpha):
        """
        
        :param dim: 
        :param alpha: 
        """
        self.dim = dim
        self.alpha = alpha

        # params
        self.word_map = dict()
        self.L = 0.01 * np.random.randn(self.dim,)

        self.V = 0.01 * np.random.randn(self.dim, 2 * self.dim, 2 * self.dim)
        self.W = 0.01 * np.random.randn(self.dim, 2 * self.dim)
        self.b = 0.01 * np.random.randn(self.dim,)

        self.Ws = 0.01 * np.random.randn(self.dim, self.dim)
        self.bs = 0.01 * np.random.randn(self.dim,)

        self.paramsShape = [self.V.shape, self.W.shape, self.b.shape, self.Ws.shape, self.bs.shape]

        # gradient
        self.dL, self.dV, self.dW, self.db, self.dWs, self.dbs = self.default_grad()

        # sum gradient ** 2
        self.sumdL2, self.sumdV2, self.sumdW2, self.sumdb2, self.sumdWs2, self.sumdbs2 = self.default_grad()

    def default_grad(self):
        return [dict()] + [np.zeros(p) for p in self.paramsShape]

    def save(self, file):
        """
        
        :param file: 
        :return: 
        """
        with open(file, 'wb') as f:
            cPickle.dump(self, f)
        print("save successfully")

    @staticmethod
    def load(file):
        """
        
        :param file: 
        :return: 
        """
        with open(file, 'rb') as f:
            return cPickle.load(f)

    # endregion

    # region activate function
    def activate_node(self, node, left, right):
        node.actf = Model.activate_function(self.V, self.W, self.b, left, right)
        node.prob = Model.softmax(self.Ws, self.bs, node.actf)

    @staticmethod
    def activate_function(tensor, weight, bias, left, right):
        lr = np.concatenate((left, right))
        return np.tanh(Model.tensordot(lr, tensor) + np.dot(weight, lr) + bias)

    @staticmethod
    def tensordot(a, b):
        left = np.asarray([np.dot(a.T, r).tolist() for r in b])
        return np.asarray([np.dot(r, a).tolist() for r in left])

    @staticmethod
    def softmax(weight, bias, x):
        r = np.dot(weight, x) + bias
        return np.exp(r) / np.sum(np.exp(r), axis=0)

    # endregion

    # region preproccess

    def preproccess(self, trainf, savef, word_limits):
        start = time()
        trees = TreeNet.load_trees(trainf)
        dictionary = set()
        count = [0] * 5
        words_count = dict()
        for tree in trees:
            label = int(tree.root.label)
            words = TreeNet.DFS_traverse(tree.root, (lambda node, l, r: l | r), (lambda node: {node.word}))
            dictionary |= {word for word in words if word not in stopwords.words("english")}
            count[label] += 1
            for word in words:
                try:
                    words_count[word][label] += 1
                except KeyError:
                    words_count[word] = [0] * 5
                    words_count[word][label] += 1
        word_IG = Model.compute_IG(words_count, count)
        dictionary = Model.extract_dict(word_IG, word_limits)
        self.word_map.update({dictionary[i]: i for i in range(len(dictionary))})
        for i in range(len(self.word_map)):
            self.L = np.vstack((self.L, 0.01 * np.random.randn(self.dim, )))
        with open(savef, 'wb') as f:
            for tree in trees:
                TreeNet.BFS_traverse(tree.root, self.process_node)
                cPickle.dump(tree, f)
        print('Finish proccess data in {t} seconds.'.format(t=time() - start))

    def process_node(self, node):
        if node.isLeaf:
            node.word = self.word_map.get(node.word, 0)
        node.label = int(node.label)

    @staticmethod
    def compute_IG(word_count, cc):
        word_IG = dict()
        dim = len(cc)
        esp = 1e-4

        def f(x):
            return math.log2(x + esp)

        N = sum(cc)
        for word, c in word_count.items():
            Nw = sum(c)
            word_IG[word] = sum([(-cc[i] / N) * f(cc[i] / N)
                                 + (Nw / N) * (c[i] / Nw) * f(c[i] / Nw)
                                 + (1 - Nw / N) * ((cc[i] - c[i]) / (N - Nw + esp)) * f((cc[i] - c[i]) / (N - Nw + esp))
                                 for i in range(dim)])
        return word_IG

    @staticmethod
    def extract_dict(word_IG, limit):
        sorted_words = sorted(word_IG.items(), key=operator.itemgetter(1))
        sorted_words.reverse()
        return sorted([word for word, _ in sorted_words[0:limit]])

    # endregion

    # region train
    def train(self, trainf, rpf, savef, save_fre, epoch, batch_size, step_size=0.01, fudge_factor=1e-8):
        with open(rpf, 'w') as report:
            trp = [time()]
            for i in range(epoch):
                report.write("=" * 80 + "\n")
                report.write("Starting epoch {n}...".format(n=i + 1))
                print("Starting epoch {n}...".format(n=i + 1), end='')
                self.sumdL2, self.sumdV2, self.sumdW2, self.sumdb2, self.sumdWs2, self.sumdbs2 = self.default_grad()
                with open(trainf, 'rb') as f:
                    res = np.zeros((4,))
                    data = list()
                    for j in range(2 * batch_size):
                        data.append(cPickle.load(f))
                    batch = list()
                    while True:
                        for j in range(batch_size):
                            try:
                                data.append(cPickle.load(f))
                            except EOFError:
                                break
                        if not len(data):
                            break
                        else:
                            random.shuffle(data)
                            for k in range(min(batch_size, len(data))):
                                batch.append(data.pop(0))
                            res += self.train_batch(batch, step_size, fudge_factor)
                            batch.clear()
                print("Cost: {c}. Correct node: {cor}. Total node: {t}. Correct tree {ct}. "
                      .format(c=res[0], cor=res[1], t=res[2], ct=res[3]))
                trp.append(time())
                report.write("Finish epoch {n} in {t}.\n".format(n=i+1, t=trp[-1]-trp[-2]))
                if not i % save_fre:
                    self.save(savef)

    def train_batch(self, batch, step_size, fudge_factor):

        res_batch = np.zeros((3,))
        correct_tree = 0
        self.dL, self.dV, self.dW, self.db, self.dWs, self.dbs = self.default_grad()

        for tree in batch:
            res_batch += self.forward_prob(tree.root)
            if np.argmax(tree.root.prob) == tree.root.label:
                correct_tree += 1
        res_batch[0] += self.alpha / 2 * (np.sum(self.V ** 2)
                                          + np.sum(self.W ** 2)
                                          + np.sum(self.Ws ** 2)
                                          + np.sum(self.b ** 2)
                                          + np.sum(self.bs ** 2))

        for tree in batch:
            self.backward_prob(tree.root)

        scale = 1.0 / len(batch)
        for val in self.dL.values():
            val *= scale

        self.dV = scale * (self.dV + self.alpha * self.V)
        self.dW = scale * (self.dW + self.alpha * self.W)
        self.db = scale * (self.db + self.alpha * self.b)

        self.dWs = scale * (self.dWs + self.alpha * self.Ws)
        self.dbs = scale * (self.dbs + self.alpha * self.bs)

        self.update(step_size, fudge_factor)
        return np.hstack((res_batch, correct_tree))

    def forward_prob(self, node: Node):
        err = np.zeros((3,))
        if node.isLeaf:
            node.actf = self.L[node.word]
            node.prob = Model.softmax(self.Ws, self.bs, node.actf)
        else:
            err += self.forward_prob(node.left)
            err += self.forward_prob(node.right)
            self.activate_node(node, node.left.actf, node.right.actf)
        return err \
            + np.asarray([-math.log(node.prob[node.label]), np.argmax(node.prob) == node.label, 1])

    def backward_prob(self, node: Node, err=None):
        delta = node.prob
        delta[node.label] -= 1.0

        # dWs
        self.dWs += np.outer(delta, node.actf)
        self.dbs += delta

        softmax_err = np.dot(self.Ws.T, delta) * (1 - node.actf ** 2)

        if err is None:
            com_err = softmax_err
        else:
            com_err = softmax_err + err * (1 - node.actf ** 2)

        if node.isLeaf:
            if node.word in self.dL.keys():
                self.dL[node.word] += com_err
            else:
                self.dL[node.word] = com_err
            return

        lr = np.concatenate((node.left.actf, node.right.actf))

        # dV
        self.dV += (np.outer(lr, lr)[..., None] * com_err).T
        # dW
        self.dW += np.outer(com_err, lr)
        # db
        self.db += com_err

        down_err = np.dot(self.W.T, com_err) \
            + np.tensordot(self.V.transpose((0, 2, 1)) + self.V, np.outer(com_err, lr).T, axes=([1, 0], [0, 1]))
        self.backward_prob(node.left, down_err[:self.dim])
        self.backward_prob(node.right, down_err[self.dim:])

    def update(self, step_size, fudge_factor):
        for key, val in self.dL.items():
            if key in self.sumdL2.keys():
                self.sumdL2[key] += val ** 2
            else:
                self.sumdL2[key] = val ** 2
            self.L[key] -= step_size / (fudge_factor + np.sqrt(self.sumdL2[key])) * val

        self.sumdV2 += self.dV ** 2
        self.V -= step_size / (fudge_factor + np.sqrt(self.sumdV2)) * self.dV

        self.sumdW2 += self.dW ** 2
        self.W -= step_size / (fudge_factor + np.sqrt(self.sumdW2)) * self.dW

        self.sumdWs2 += self.dWs ** 2
        self.Ws -= step_size / (fudge_factor + np.sqrt(self.sumdWs2)) * self.dWs

        self.sumdb2 += self.db ** 2
        self.b -= step_size / (fudge_factor + np.sqrt(self.sumdb2)) * self.db

        self.sumdbs2 += self.dbs ** 2
        self.bs -= step_size / (fudge_factor + np.sqrt(self.sumdbs2)) * self.dbs

    def validate(self, devf):
        confusion_matrix = np.zeros((self.dim, self.dim))
        while True:
            with open(devf, 'rb') as f:
                try:
                    tree = cPickle.load(f)
                    self.forward_prob(tree.root)
                except EOFError:
                    break

    # endregion

    # region test
    def test(self, testf, rpf):
        with open(rpf, 'w') as report:
            with open(testf, 'r') as f:
                result = np.zeros((4,))
                for line in f:
                    tree = TreeNet(line)
                    res = self.estimate_sentis(tree)
                    print("correct node: {cn} - total node: {ct} - correct tree: {ctr}- total tree: {ttr}."
                          .format(cn=res[0], ct=res[1], ctr=res[2], ttr=res[3]))
                    result += res
                print("=" * 80)
                report.write("correct node: {cn} - total node: {ct} - correct tree: {ctr}- total tree: {ttr}."
                             .format(cn=result[0], ct=result[1], ctr=result[2], ttr=result[3]))
                return result

    def estimate_sentis(self, tree):
        def traverse(node: Node):
            node.label = int(node.label)
            res = np.zeros((2,))
            if node.isLeaf:
                node.word = self.word_map.get(node.word.lower(), 0)
                node.actf = self.L[node.word]
                node.prob = Model.softmax(self.Ws, self.bs, node.actf)
                return np.asarray([np.argmax(node.prob) == node.label, 1])
            else:
                res += traverse(node.left)
                res += traverse(node.right)
                self.activate_node(node, node.left.actf, node.right.actf)
                # print("node.prob: {p} - node.label: {l}.".format(p=node.prob, l=node.label))
                return res + np.asarray([np.argmax(node.prob) == node.label, 1])
        result = traverse(tree.root)
        return np.concatenate((result, np.asarray([np.argmax(tree.root.prob) == tree.root.label, 1])))

    # endregion
