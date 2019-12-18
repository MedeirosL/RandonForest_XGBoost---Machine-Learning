# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 08:48:28 2019

@author: lucas


Algoritmo da Random Forest com o menor uso de blibliotecas possível
Trabalho final - Lucas Medeiros
"""

# Random Forest Algorithm on Sonar Dataset
from random import seed
from random import randrange
from csv import reader
from math import sqrt

def loadCsv(filename):
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
	datasplit = list()
	datacopy = list(dataset)
	fsize = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fsize:
			index = randrange(len(datacopy))
			fold.append(datacopy.pop(index))
		datasplit.append(fold)
	return datasplit

# Calcula a porcentagem da acurácia/precisão
def accuracyMetric(actual, predicted):
	x = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			x += 1
	return x / float(len(actual)) * 100.0

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
            rowc = list(row)
            test_set.append(rowc)
            rowc[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        print("pred:",predicted)
        actual = [row[-1] for row in fold]
        accuracy = accuracyMetric(actual, predicted)
        scores.append(accuracy)
    return scores

# Separa o dataset de testes em "parâmetros" e "resultado".
def testSplit(index, value, dataset):
	l, r = list(), list()
	for row in dataset:
		if row[index] < value:
			l.append(row)
		else:
			r.append(row)
	return l, r

# Calcula o index para o split pelo método de gini.
def giniIndex(groups, classes):
	instance = float(sum([len(group) for group in groups]))
	gini = 0.0
	for group in groups:
		size = float(len(group))
		if size == 0:
			continue
		score = 0.0
		for classV in classes:
			p = [row[-1] for row in group].count(classV) / size
			score += p * p
		gini += (1.0 - score) * (size / instance)
	return gini

# Retorna as posições de split utilizando o método acima
def getSplit(dataset, nfeat):
	class_values = list(set(row[-1] for row in dataset))
	bI, bV, bS, bG = 999, 999, 999, None
	features = list()
	while len(features) < nfeat:
		index = randrange(len(dataset[0])-1)
		if index not in features:
			features.append(index)
	for index in features:
		for row in dataset:
			groups = testSplit(index, row[index], dataset)
			gini = giniIndex(groups, class_values)
			if gini < bS:
				bI, bV, bS, bG = index, row[index], gini, groups
	return {'index':bI, 'value':bV, 'groups':bG}

# Retorna o nó terminal
def toTerminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)

def split(node, max_depth, min_size, nfeat, depth):
	l, r = node['groups']
	del(node['groups'])
	# check for a no split
	if not l or not r:
		node['l'] = node['r'] = toTerminal(l + r)
		return
	# check for max depth
	if depth >= max_depth:
		node['l'], node['r'] = toTerminal(l), toTerminal(r)
		return
	# process l child
	if len(l) <= min_size:
		node['l'] = toTerminal(l)
	else:
		node['l'] = getSplit(l, nfeat)
		split(node['l'], max_depth, min_size, nfeat, depth+1)
	# process r child
	if len(r) <= min_size:
		node['r'] = toTerminal(r)
	else:
		node['r'] = getSplit(r, nfeat)
		split(node['r'], max_depth, min_size, nfeat, depth+1)

# Constrói a Arvore de Decisão
def buildTree(train, max_depth, min_size, nfeat):
	r = getSplit(train, nfeat)
	split(r, max_depth, min_size, nfeat, 1)
	return r

# Realiza uma predição através da árvore e retorna o nó
def predict(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node['l'], dict):
			return predict(node['l'], row)
		else:
			return node['l']
	else:
		if isinstance(node['r'], dict):
			return predict(node['r'], row)
		else:
			return node['r']


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
def randomForest(train, test, max_depth, min_size, sample_size, ntrees, nfeat):
	trees = list()
	for i in range(ntrees):
		sample = subsample(train, sample_size)
		tree = buildTree(sample, max_depth, min_size, nfeat)
		trees.append(tree)
	predictions = [baggingPredict(trees, row) for row in test]
	return(predictions)

''' Teste '''
seed(2)

filename = 'dataset3_rf.csv'
dataset = loadCsv(filename)
best_mean=0
best_params = []

'''definiç~so dos parametros e plot dos resultados'''
n_folds = 10
max_depth = 3
min_size = 1
sample_size = .5
nfeat = int(sqrt(len(dataset[0])-1))
for sample_size in [1.25]:
    for min_size in [1]:
        for max_depth in [3]:
            for ntrees in [10]:
                scores = evaluateAlgorithm(dataset, randomForest, n_folds, max_depth, min_size, sample_size, ntrees, nfeat)
                if (sum(scores)/float(len(scores)) > best_mean):
                    best_params = [sample_size, min_size, max_depth, ntrees]
                    best_mean = sum(scores)/float(len(scores))
                print("Árvores:",ntrees, "Sample Size", sample_size, "Min Size",min_size,"Max Depth",max_depth)
                print('Scores: %s' % scores)
                print('Acurácia: %.3f%%' % (sum(scores)/float(len(scores))))
    
    print("Best Params: ", best_params) #1.25, 1, 3, 10
    print("Best Mean: ", best_mean)

    
