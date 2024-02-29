import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from sklearn.model_selection import train_test_split
from collections import Counter

class Node:
    def __init__(self, nodetype=None, nodeattr=None, classID=None, children=[]):
        self.nodetype = nodetype
        self.nodeattr = nodeattr
        self.classID = classID
        self.children = []
        
class DecisionTree:
    def __init__(self, data):
        self.root = None
        self.attr_values = {}

    # Filter all the unique values
    def build_values(self, data, attr_list):
        for attr in attr_list:
            values = data[attr].unique()
            self.attr_values[attr] = sorted(values)

    # Function to construct tree
    def construct_tree(self, data, attr_list):
        node = Node()
        if len(attr_list) == 0 or data['target'].nunique() <= 1:
            leaf_value = data.iloc[0]['target'] if data['target'].nunique() <= 1 else data['target'].mode()[0]
            node = Node(nodetype='leaf', classID=leaf_value)
            return node

        best_attr = self.find_split(data, attr_list)
        node = Node(nodetype='decision', nodeattr=best_attr)
        attr_list.remove(best_attr)

        node.children = [
            self.construct_tree(split_data, attr_list) if not split_data.empty else Node(nodetype='leaf', classID=data['target'].mode()[0])
            for value in self.attr_values[best_attr]
            for split_data in [data[data[best_attr] == value]]
        ]
        return node

    # Function to calculate Gini index
    def gini_index(self, data):
        targets = data['target'].to_numpy()
        size = len(targets)
        if size == 0:
            return 0.0

        counts = np.array(list(Counter(targets).values()))
        probabilities = counts / size
        gini = 1 - np.sum(probabilities ** 2)
        return gini

    # Function to decide where to split using Gini index
    def find_split(self, data, attributes):
        total_gini = self.gini_index(data)
        gini_gains = {}
        for attr in attributes:
            unique_attr_values, count_of_attr_values = np.unique(data[attr].to_numpy(), return_counts=True)
            ginis = [count * self.gini_index(data[data[attr] == value]) for value, count in zip(unique_attr_values, count_of_attr_values)]
            gini_for_attr = sum(ginis) / sum(count_of_attr_values)
            gini_gains[attr] = total_gini - gini_for_attr

        best_attr = max(gini_gains, key=gini_gains.get, default=attributes[0])
        return best_attr

    # Function to determine the class
    def determine_class(self, current_node, datapoint):
        while current_node.nodetype != 'leaf':
            current_node = current_node.children[datapoint[current_node.nodeattr]]
        return current_node.classID

    # Function to calculate accuracy
    def accuracy(self, X, Y):
        preds = [self.determine_class(self.root, X.iloc[i]) == Y.iloc[i] for i in range(len(X))]
        return sum(preds) / len(preds)

data=pd.read_csv('DecisionTree/house_votes_84.csv')
train_accuracy=[]
test_accuracy=[]
data_without_target=data.drop(labels='target', axis=1)

for index in range(100):
    decision_tree=DecisionTree(data)
    attr_list = data_without_target.columns.tolist()
    decision_tree.build_values(data, attr_list)

    X_train, X_test, y_train, y_test=train_test_split(data_without_target, data.target, test_size=0.20, shuffle=True)
    decision_tree.root = decision_tree.construct_tree(X_train.join(y_train), attr_list.copy())

    train_accuracy.append(round(decision_tree.accuracy(X_train, y_train), 3))
    test_accuracy.append(round(decision_tree.accuracy(X_test, y_test), 3))
    
train_mean = np.mean(train_accuracy)
train_sd = np.std(train_accuracy)
print(f"Training Accuracy: Mean = {train_mean} Standard Deviation = {train_sd}")

test_mean = np.mean(test_accuracy)
test_sd = np.std(test_accuracy)
print(f"Testing Accuracy: Mean = {test_mean} Standard Deviation = {test_sd}")

plt.hist(train_accuracy, bins=5)
plt.xlabel('Accuracy')
plt.ylabel('Frequency')
plt.title("Training Accuracy")
plt.show()

plt.hist(test_accuracy, bins=5)
plt.xlabel('Accuracy')
plt.ylabel('Frequency')
plt.title("Testing Accuracy")
plt.show()
