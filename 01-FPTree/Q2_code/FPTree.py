'''
==================================================
Author: ZHENG Dongjia
SID:    20546139
Date:   23/9/2018
Copyright 2018 ZHENG Dongjia. All rights reserved.
==================================================
Implement a FP-Tree
==================================================
Input:
    - D, a transaction database from a grocery
        containing 9835 records
    - min_sup, the minimum support count threshold

Output: The complete set of frequent patterns.
==================================================
'''


import pandas as pd
import numpy as np
import sys
from itertools import combinations


class Node:
    ''' '''
    def __init__(self,
                 item_name,
                 sup_count=0,
                 parent=None,
                 link_node=None):
        self.item_name = item_name
        self.sup_count = sup_count
        self.parent = parent

        self.children = dict()
        self.link_node = link_node

    def PreOrderTraversal(self):
        if self is None:
            return None

        pre_order_output = []
        output_str = "{item_name} {count}".format(
                item_name=self.item_name,
                count=self.sup_count)
        pre_order_output.append(output_str)
        if len(self.children) != 0:
            for child_name in self.children.keys():
                child = self.children[child_name]
                child_output = child.PreOrderTraversal()
                pre_order_output.append(child_output)

        return pre_order_output


class FPTree:
    '''Construct a FP-Tree'''
    def __init__(self, transactions=None, min_sup=300):
        self.root = Node(item_name="Null Set", sup_count=1)
        self.min_sup = min_sup

        # Store the selected and sorted items in
        # items_dict: {item_name: sup_count, ...}
        self.items_dict = None

        # store the selected and sorted items in
        # order_dict: {item_name: item_index, ...}
        self.order_dict = None

        # Sort items_dict in the reverse order of
        # support count and store the result in
        # L: [(item_name, sup_count), ...]
        self.L = None

        # Store the header of the link-node in
        # header_table: [{"item_name": ..., 
        #                 "sup_count": ...,
        #                 "link_node": ...}, ...]
        self.header_table = []

        # Store the selected and sorted transations in
        # sorted_trans: [[item_A, item_B, ...], ...]
        self.sorted_trans = None

        # Store the frequent patterns in
        # fp_list = [[{item_A, item_B, ...}, sup_count], ...]
        self.fp_list = []

        # Start constructing the FP-Tree
        self.constructFPTree(transactions)

    def constructFPTree(self, transactions):
        # the first scan of the transactions
        self.FirstScan(transactions)

        # the second scan of the transactions
        self.SecondScan(transactions)

    def FirstScan(self, transactions):
        '''the first scan of the transactions

        Returns:
            L: the list of frequent items (1-itemsets) L in
               support count descending order
        '''
        # calculate the support count of each item
        self.items_dict = dict()
        for tran in transactions:
            for item in tran:
                # update the support count of the item
                if item in self.items_dict.keys():
                    self.items_dict[item] += 1
                else:
                    self.items_dict[item] = 1

        # Truncate the items with support count smaller than min_sup
        for item in self.items_dict.keys():
            if self.items_dict[item] < self.min_sup:
                self.items_dict.pop(item)

        # Sort items_dict in support count
        self.L = sorted(self.items_dict.items(),
                        key=lambda x: (-x[1], x[0]))
        self.order_dict = dict()
        self.header_table = []
        for i in range(len(self.L)):
            item_count_pair = self.L[i]
            item_name = item_count_pair[0]
            item_sup = item_count_pair[1]
            # store the sorted items in order_dict: (item_name: item_index)
            self.order_dict[item_name] = i
            # construct the header table for the frequent items' node link
            header = {
                "item_name": item_name,
                "sup_count": item_sup,
                "link_node": None
            }
            self.header_table.append(header)

    def SecondScan(self, transactions):
        # 1. Select and sort each transaction according to the order of L
        self.sorted_trans = []
        for tran in transactions:
            selected_tran = []
            for item in tran:
                # Select the item with support count smaller than min_sup
                if item in self.items_dict.keys():
                    selected_tran.append(item)

            if len(selected_tran) != 0:
                # Sort the item according to the ordered dictionary
                sorted_t = sorted(selected_tran,
                                  key=lambda x: self.order_dict[x])
                self.sorted_trans.append(sorted_t)

                # 2. insert the transaction into FP-Tree
                # p: the first item_name in sorted_trans
                # P: the list of the remaining item_name after p
                p = sorted_t[0]
                P = []
                if len(sorted_t) > 1:
                    P = sorted_t[1:]
                T = self.root
                self.InsertTree(p, P, T)

    def InsertTree(self, p, P, T):
        N = None
        if p in T.children.keys():
            # update the sup_count of the node
            N = T.children[p]
            N.sup_count += 1
        else:
            # insert the node into FP-Tree
            N = Node(item_name=p,
                     sup_count=1,
                     parent=T,
                     link_node=None)
            T.children[p] = N
            # link this node to the node-link
            for header in self.header_table:
                # locate to the correct header
                if header["item_name"] == N.item_name:
                    # find the last node of the header
                    if header["link_node"] is None:
                        header["link_node"] = N
                    else:
                        iter_node = header["link_node"]
                        while(iter_node.link_node is not None):
                            iter_node = iter_node.link_node
                        iter_node.link_node = N
        
        if len(P) != 0:
            p = P[0]
            P = P[1:]
            self.InsertTree(p, P, N)

    def SinglePath(self):
        ''' Judge whether the tree contains a single path'''
        iter_node = self.root
        if len(iter_node.children) == 0:
            return False, None

        path_list = []
        while len(iter_node.children) != 0:
            if len(iter_node.children) == 1:
                iter_node = iter_node.children.values()[0]
                path_list.append(iter_node)
            elif len(iter_node.children) > 1:
                return False, None
        return True, path_list

    def FPGrowth(self, alpha=None):
        '''FPGrowth(alpha) function will return all the
           frequent patterns in Tree containing alpha pattern

           Args:
                alpha: [{item_A, item_B, ...}, sup_count]
        '''
        if len(self.root.children) == 0:
            return None

        fp_result = []
        is_single_path, path_list = self.SinglePath()
        if is_single_path:
            patterns_result = self.GeneratePattern(path_list, alpha)
            return patterns_result
        else:
            for a_i in reversed(self.header_table):
                beta = [set(), 0]
                if alpha is None:
                    beta[0] = {a_i["item_name"]}
                else:
                    beta[0] = {a_i["item_name"]} | alpha[0]
                beta[1] = a_i["sup_count"]
                # Add the beta to fp_result
                fp_result.append(beta) 

                # get the transactions of the conditional pattern base
                beta_cpb = self.CondPatternBase(a_i)
                # get the conditional FP-Tree of beta
                beta_tree = FPTree(transactions=beta_cpb, min_sup=self.min_sup)
                if len(beta_tree.root.children) != 0:
                    beta_tree_fp = beta_tree.FPGrowth(beta)
                    for pattern in beta_tree_fp:
                        fp_result.append(pattern)

            return fp_result

    def CondPatternBase(self, a_i):
        ''' Return the transactions of beta's
            conditional pattern base '''
        # Get the link node of a_i
        iter_node = a_i["link_node"]
        if iter_node.parent is None:
            return None

        transactions = []
        while iter_node is not None:
            # Retrieve the original transaction
            # of the corresponding link node
            tran = []

            path_node = iter_node.parent
            while path_node.parent is not None:
                tran.append(path_node.item_name)
                path_node = path_node.parent

            tran = list(reversed(tran))
            path_sup = iter_node.sup_count
            for _ in range(path_sup):
                transactions.append(tran)

            # Go to the next link node
            iter_node = iter_node.link_node

        return transactions

    def GeneratePattern(self, path_list, alpha=None):
        '''Generate the frequent pattern from the single path and alpha'''
        node_combinations = []
        for subset_size in range(1, len(path_list) + 1):
            for subset in combinations(path_list, subset_size):
                node_combinations.append(subset)

        for combination in node_combinations:
            # beta: each frequent patters combination from the single path
            # the format of beta: [{item_A, item_B, ...}, sup_count]
            beta = [set(), 0]

            # Iterate the combination to find out the minimum support count
            beta[1] = combination[0].sup_count
            for node in combination:
                beta[0].add(node.item_name)
                if node.sup_count < beta[1]:
                    beta[1] = node.sup_count

            # The result pattern is the combination of alpha & beta while the
            # support count is the minimum support count of nodes in beta
            frequent_pattern = [set(), 0]
            if alpha is None:
                frequent_pattern = beta
            else:
                frequent_pattern = [alpha[0] | beta[0], beta[1]]

            self.fp_list.append(frequent_pattern)

        return self.fp_list

    def GenerateCondPattTree(self):
        cond_fp_trees = []
        corresponding_item = []

        for a_i in reversed(self.header_table):
            # get the transactions of the conditional pattern base
            cpb_trans = self.CondPatternBase(a_i)

            # get the conditional FP-Tree of beta
            cond_tree = FPTree(transactions=cpb_trans, min_sup=self.min_sup)
            if len(cond_tree.root.children) > 0:
                cond_fp_trees.append(cond_tree)
                corresponding_item.append(a_i["item_name"])
        return cond_fp_trees, corresponding_item


def preprocessData(database):
        ''' Transfer the database into an array without NaN value
        Returns:
            trans_list: the list of the transactions
        '''
        trans_list = []
        for tran in database:
            items = []
            for item in tran:
                if item is np.nan:
                    break
                items.append(item)
            trans_list.append(items)

        # return the list and the dictionary
        return trans_list


def WriteToFile(filename, fp_result):
    fp_list = []
    for pattern in fp_result:
        fp_str = str(pattern[0])
        fp_str = fp_str.replace('set([', '{')
        fp_str = fp_str.replace('])', '}')
        fp_list.append(fp_str)
        print fp_str

    fp_dataframe = pd.DataFrame(fp_list)
    fp_dataframe.to_csv("fp_results.csv", index=False, header=False)


def main(argv):
    if argv is None:
        print('world~!')
    else:
        groceries_data = pd.read_csv("./groceries.csv", header=None)
        transactions_db = preprocessData(groceries_data.values)
        fp_tree = FPTree(transactions=transactions_db, min_sup=300)

        # Write to csv file
        # Meanwhile, output the frequent patterns
        print "===== Frequent Patterns with Support Count >= 300 ====="
        fp_result = fp_tree.FPGrowth()
        WriteToFile('fp_results.csv', fp_result)
        print "Total number of frequent patterns: {}".format(len(fp_result))

        # Output the FP-Conditional Trees
        print "\n========== FP-Conditional Trees =========="
        cond_fp_trees, corresponding_item = fp_tree.GenerateCondPattTree()
        for i in range(len(cond_fp_trees)):
            print "FP-Conditional Tree {}".format(i)
            print "Corresponding item name: {}".format(corresponding_item[i])
            print cond_fp_trees[i].root.PreOrderTraversal()
            print "----------------------------------------"


if __name__ == '__main__':
    main(sys.argv)