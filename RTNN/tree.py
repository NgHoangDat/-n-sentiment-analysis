"""

"""
from time import time


class Node:
    def __init__(self):
        self.label = None
        self.word = None
        self.left = None
        self.right = None
        self.isLeaf = False
        self.prob = None
        self.actf = None
        self.numWord = 0

    def __str__(self):
        if self.isLeaf:
            return str(self.label) + ' ' + str(self.word)
        else:
            return str(self.label) + '(' + str(self.left) + ')' + ' (' + str(self.right) + ')'


class TreeNet:
    def __init__(self, tree_str, open_char='(', close_char=')'):
        """

        :param tree_str:
        :param open_char: 
        :param close_char: 
        """
        self.open = open_char
        self.close = close_char
        self.root = self.parse(tree_str.strip())

    def parse(self, tree_str):
        """

        :param tree_str: 
        :return: 
        """
        assert tree_str[0] == self.open, "Wrong tree format"
        assert tree_str[-1] == self.close, "Wrong tree format"
        tree_str = tree_str[1:-1].strip()
        node = Node()

        split = 0
        while True:
            if tree_str[split] == ' ':
                node.label = tree_str[0:split]
                tree_str = tree_str[split + 1:]
                break
            else:
                split += 1

        if tree_str.find(self.open) == -1:
            node.word = tree_str.lower()
            node.isLeaf = True
            node.numWord = 1
            return node
        else:
            c_open, c_close, split = (1, 0, 1)

            while c_open != c_close:
                if tree_str[split] == self.open:
                    c_open += 1
                if tree_str[split] == self.close:
                    c_close += 1
                split += 1

            node.left = self.parse(tree_str[0:split].strip())
            node.right = self.parse(tree_str[split:].strip())
            node.numWord = node.left.numWord + node.right.numWord
            return node

    @staticmethod
    def BFS_traverse(node, fn=None):
        """

        :param node: 
        :param fn: 
        :param args: 
        :return: 
        """
        if fn is not None:
            fn(node)
        if node.left:
            TreeNet.BFS_traverse(node.left, fn)
        if node.right:
            TreeNet.BFS_traverse(node.right, fn)

    @staticmethod
    def DFS_traverse(node, gn=None, fn=None):
        """
        
        :param node: 
        :param gn:
        :param fn: 
        :return: 
        """
        if node.isLeaf:
            return fn(node) if fn is not None else None
        else:
            left = TreeNet.DFS_traverse(node.left, gn, fn)
            right = TreeNet.DFS_traverse(node.right, gn, fn)
            return gn(node, left, right) if gn is not None else None

    @staticmethod
    def load_trees(file):
        """
        
        :param file: 
        :return: 
        """
        start = time()
        with open(file, 'r') as f:
            trees = [TreeNet(line) for line in f.readlines()]
            print("Finish loading file in {t}.".format(t=time() - start))
            return trees

    def __str__(self):
        return '(' + str(self.root) + ')'
