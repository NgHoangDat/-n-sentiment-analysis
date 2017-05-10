"""

"""
from RTNN.model import Model
from RTNN.tree import TreeNet
import numpy as np

model = Model(2, 1e-6)
model.L = np.asarray([0.4, 0.6])
model.L = np.vstack((model.L, np.asarray([0.2, 0.8])))
model.L = np.vstack((model.L, np.asarray([0.7, 0.3])))

tree = TreeNet("(0 (1 2)(1 (0 0)(1 1)))")


def p(node):
    if node.isLeaf:
        node.word = int(node.word)
    node.label = int(node.label)

TreeNet.BFS_traverse(tree.root, p)
res = model.forward_prob(tree.root)

