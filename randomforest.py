# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 08:48:28 2019

@author: lucas
"""

# Random Forest Algorithm on Sonar Dataset
from random import seed
from random import randrange
from csv import reader
from math import sqrt

def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset


# Separa o dataset para o método de validação cruzada "k-fold"
def crossValidationSplit(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

# Calcula a porcentagem da acurácia/precisão
def accuracyMetric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

# Avalia o algorítmo através do método de validação cruzada acima.
def evaluateAlgorithm(dataset, algorithm, n_folds, *args):
    folds = crossValidationSplit(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        print("pred:",predicted)
        actual = [row[-1] for row in fold]
        accuracy = accuracyMetric(actual, predicted)
        scores.append(accuracy)
    return scores

# Separa o dataset de testes em "parâmetros" e "resultado".
def testSplit(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right

# Calcula o index para o split pelo método de gini.
def giniIndex(groups, classes):
	# count all samples at split point
	n_instances = float(sum([len(group) for group in groups]))
	# sum weighted Gini index for each group
	gini = 0.0
	for group in groups:
		size = float(len(group))
		# avoid divide by zero
		if size == 0:
			continue
		score = 0.0
		# score the group based on the score for each class
		for class_val in classes:
			p = [row[-1] for row in group].count(class_val) / size
			score += p * p
		# weight the group score by its relative size
		gini += (1.0 - score) * (size / n_instances)
	return gini

# Retorna as posições de split utilizando o método acima
def getSplit(dataset, n_features):
	class_values = list(set(row[-1] for row in dataset))
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	features = list()
	while len(features) < n_features:
		index = randrange(len(dataset[0])-1)
		if index not in features:
			features.append(index)
	for index in features:
		for row in dataset:
			groups = testSplit(index, row[index], dataset)
			gini = giniIndex(groups, class_values)
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}

# Retorna o nó terminal
def toTerminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)

def split(node, max_depth, min_size, n_features, depth):
	left, right = node['groups']
	del(node['groups'])
	# check for a no split
	if not left or not right:
		node['left'] = node['right'] = toTerminal(left + right)
		return
	# check for max depth
	if depth >= max_depth:
		node['left'], node['right'] = toTerminal(left), toTerminal(right)
		return
	# process left child
	if len(left) <= min_size:
		node['left'] = toTerminal(left)
	else:
		node['left'] = getSplit(left, n_features)
		split(node['left'], max_depth, min_size, n_features, depth+1)
	# process right child
	if len(right) <= min_size:
		node['right'] = toTerminal(right)
	else:
		node['right'] = getSplit(right, n_features)
		split(node['right'], max_depth, min_size, n_features, depth+1)

# Constrói a Arvore de Decisão
def buildTree(train, max_depth, min_size, n_features):
	root = getSplit(train, n_features)
	split(root, max_depth, min_size, n_features, 1)
	return root

# Realiza uma predição através da árvore e retorna o nó
def predict(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']


def subsample(dataset, ratio):
	sample = list()
	n_sample = round(len(dataset) * ratio)
	while len(sample) < n_sample:
		index = randrange(len(dataset))
		sample.append(dataset[index])
	return sample

# Faz uma predição pelo método de bagging e retona a mesma
def baggingPredict(trees, row):
	predictions = [predict(tree, row) for tree in trees]
	return max(set(predictions), key=predictions.count)

# Algoritmo da floresta, de acordo com as funções acima.
def randomForest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
	trees = list()
	for i in range(n_trees):
		sample = subsample(train, sample_size)
		tree = buildTree(sample, max_depth, min_size, n_features)
		trees.append(tree)
	predictions = [baggingPredict(trees, row) for row in test]
	return(predictions)

# Test the random forest algorithm
seed(2)
# load and prepare data
filename = 'dataset3_rf.csv'
dataset = load_csv(filename)
# convert string attributes to integers

# convert class n to integers
#str_column_to_int(dataset, len(dataset[0])-1)
best_mean=0
best_params = []

# evaluate algorithm
n_folds = 10
max_depth = 3
min_size = 1
sample_size = .5
n_features = int(sqrt(len(dataset[0])-1))
for sample_size in [1.25]:
    for min_size in [1]:
        for max_depth in [3]:
            for n_trees in [10]:
                scores = evaluateAlgorithm(dataset, randomForest, n_folds, max_depth, min_size, sample_size, n_trees, n_features)
                if sum(scores)/float(len(scores)) > best_mean:
                    best_params = [sample_size, min_size, max_depth, n_trees]
                    best_mean = sum(scores)/float(len(scores))
                print("Trees:",n_trees, "sample_size", sample_size, "min_size",min_size,"max_depth",max_depth)
                print('Scores: %s' % scores)
                print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
    
    print("Best Params: ", best_params) #1.25, 1, 3, 10
    print("Best Mean: ", best_mean)

    
