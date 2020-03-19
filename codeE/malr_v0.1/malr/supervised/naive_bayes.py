# -*- coding: utf-8 -*-
# Copyright (C) 2011, Filipe Rodrigues <fmpr@dei.uc.pt>

import sys
import numpy as np
from malr.core import * 
from math import log

class NaiveBayes(Classifier):
	"""NaiveBayes implementation"""
	
	"""Default smoothing factor is 1 (Laplacian regularization)"""
	def __init__(self, trainset, smoothing_factor=1):
		self.compatibility_check(trainset)
		self.trainset = trainset
		self.smoothing_factor = smoothing_factor
	
	def compatibility_check(self, dataset):
		# check if class if nominal of binary
		if dataset.class_dtype != str:
			raise Exception("Naive Bayes can only handle nominal and binary classes/labels.")
		for dtype in dataset.feature_dtypes:
			if dtype != str:
				raise Exception("Naive Bayes can only handle nominal and binary features.")
	
	def train(self):
		print "Training Naive Bayes model..."
		
		# compute class prior probabilities p(class)
		self.p_class = np.zeros(len(self.trainset.target_alphabet))
		for target in xrange(len(self.trainset.target_alphabet)):
			self.p_class[target] = len(np.where(self.trainset.targets == target)[0])
		self.p_class = self.p_class / self.p_class.sum(axis=0)
		print "p(class): " + str(self.p_class)
		print
		
		self.p_f_class = []
		for feature in xrange(self.trainset.num_features):
			self.p_f_class.append(np.zeros((len(self.trainset.feature_alphabets[feature]), len(self.trainset.target_alphabet))))
		
		# compute conditional probabilities p(feature|class)
		for i in xrange(self.trainset.num_instances):
			instance = self.trainset.data[i]
			label = self.trainset.targets[i]
			for feature in xrange(self.trainset.num_features):
				self.p_f_class[feature][instance[feature], label] += 1.0
				
		for feature in xrange(self.trainset.num_features): 
			print "p(" + self.trainset.feature_names[feature] + "|class):"
			feature_values = self.trainset.data[:,feature]
			self.p_f_class[feature] += self.smoothing_factor # add smoothing factor
			self.p_f_class[feature] /= self.p_f_class[feature].sum(axis=0)
		
			print self.p_f_class[feature]
			print self.p_f_class[feature].sum(axis=0)
			print 
			
		# show accuracy and confusion matrix
		accuracy = self.evaluate(self.trainset, confusion_matrix=True)
		print "Accuracy:", accuracy
			
	def loglikelihood(self):
		loglikelihood = .0
		for instance in self.trainset.data:
			for i in xrange(len(instance)):
				if i == self.trainset.class_index:
					loglikelihood += log(self.p_class[instance[i]])
				else:
					loglikelihood += log(self.p_f_class[i][instance[i], self.trainset.targets[i]])
		print loglikelihood
		return loglikelihood
	
	def apply(self, instance):
		logposterior = np.zeros(len(self.trainset.target_alphabet))
		for target in xrange(len(self.trainset.target_alphabet)):
			logposterior[target] = log(self.p_class[target])
			for feature in xrange(len(instance)):
				logposterior[target] += log(self.p_f_class[feature][instance[feature], target])
		pred_index = np.argmax(logposterior)
		pred = self.trainset.lookup_class_index(pred_index)
		posterior = np.exp(logposterior)
		posterior = posterior / posterior.sum()
		return pred, posterior[pred_index]
		
	def posterior(self, instance):
		posterior = np.zeros(len(self.trainset.target_alphabet))
		for target in xrange(len(self.trainset.target_alphabet)):
			posterior[target] = log(self.p_class[target])
			for feature in xrange(len(instance)):
				posterior[target] += log(self.p_f_class[feature][instance[feature], target])
		posterior = np.exp(posterior)
		return np.max(posterior / posterior.sum())

