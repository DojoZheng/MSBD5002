'''
==================================================
Author: ZHENG Dongjia
SID:    20546139
Date:   20/9/2018
Copyright 2018 ZHENG Dongjia. All rights reserved.
==================================================
Implement a Hash Tree for Apriori Algorithm
==================================================
Input:  3-itemsets candidates
Output: a hash tree of max leaf size 3 in the form of nested list
==================================================
'''
import sys


class Node:
    def __init__(self, max_leaf_size=3, height=0):
        # Tree Nodes
        self.max_leaf_size = max_leaf_size
        self.sub_nodes = [None] * self.max_leaf_size

        # Record the height of the node
        self.height = height

        # Bucket which stores a nested list of candidates
        self.bucket = []

    def insertToBucket(self, candidate=None):
        self.bucket.append(candidate)
        # Split the node if the bucket is full and the height smaller than 3
        if (len(self.bucket) > self.max_leaf_size) and (self.height < 3):
            for candidate in self.bucket:
                hashValue = self.hash(candidate[self.height])
                if self.sub_nodes[hashValue] is None:
                    self.sub_nodes[hashValue] = Node(self.max_leaf_size,
                                                     self.height+1)
                self.sub_nodes[hashValue].insertToBucket(candidate)
            self.bucket = []

    def isLeafNode(self):
        # Judge whether all of the subnodes are None
        for i in range(self.max_leaf_size):
            if self.sub_nodes[i] is not None:
                return False

        # Else this node is a leaf node
        return True

    def hash(self, value):
        ''' return the hash index '''
        hashValue = value % self.max_leaf_size
        index = 0
        if hashValue == 0:
            index = self.max_leaf_size - 1
        else:
            index = hashValue - 1
        return index


class HashTree:
    # Attributes
    max_leaf_size = 3

    def __init__(self, max_leaf_size=3):
        self.max_leaf_size = max_leaf_size
        self.root = Node(max_leaf_size=self.max_leaf_size)

    def hash(self, value):
        ''' return the bucket index '''
        hashValue = value % self.max_leaf_size
        index = 0
        if hashValue == 0:
            index = self.max_leaf_size - 1
        else:
            index = hashValue - 1
        return index

    def insertCandidate(self, height, currentNode, candidate):
        ''' Insert Candidate to the CurrentNode

        Args:
            height: represent the current layer of the hash tree
            currentNode: represent the current node
            candidate: represent the candidate to be inserted
        '''
        # n represent which subnode the candidate should be inserted to
        leaf_size = self.max_leaf_size
        n = self.hash(candidate[height])

        if currentNode.sub_nodes[n] is None:
            currentNode.sub_nodes[n] = Node(max_leaf_size=leaf_size,
                                            height=height+1)

        if currentNode.sub_nodes[n].isLeafNode():
            currentNode.sub_nodes[n].insertToBucket(candidate)

        else:
            height += 1
            currentNode = currentNode.sub_nodes[n]
            self.insertCandidate(height, currentNode, candidate)

    # Output the candidates in a nested list
    def traversal(self):
        if self is None:
            return None

        if self.root.isLeafNode():
            if len(self.root.bucket) == 1:
                return self.root.bucket[0]
            else:
                return self.root.bucket

        nested_list = []
        for i in range(self.max_leaf_size):
            if self.root.sub_nodes[i] is not None:
                tree = HashTree(self.max_leaf_size)
                tree.root = self.root.sub_nodes[i]
                nested_list.append(tree.traversal())
        return nested_list


# Get the candidates datasets
candidates_list = [[1, 2, 4], [1, 2, 9], [1, 3, 5], [1, 3, 9], [1, 4, 7],
                   [1, 5, 8], [1, 6, 7], [1, 7, 9], [1, 8, 9], [2, 3, 5],
                   [2, 4, 7], [2, 5, 6], [2, 5, 7], [2, 5, 8], [2, 6, 7],
                   [2, 6, 8], [2, 6, 9], [2, 7, 8], [3, 4, 5], [3, 4, 7],
                   [3, 5, 7], [3, 5, 8], [3, 6, 8], [3, 7, 9], [3, 8, 9],
                   [4, 5, 7], [4, 5, 8], [4, 6, 7], [4, 6, 9], [4, 7, 8],
                   [5, 6, 7], [5, 7, 9], [5, 8, 9], [6, 7, 8], [6, 7, 9]]


def main(argv):
    # Start the Test
    hashTree = HashTree()
    for candidate in candidates_list:
        hashTree.insertCandidate(height=0,
                                 currentNode=hashTree.root,
                                 candidate=candidate)

    nested_list = hashTree.traversal()
    print nested_list


if __name__ == '__main__':
    main(sys.argv)