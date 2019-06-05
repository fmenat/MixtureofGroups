# -*- coding: utf-8 -*-
# Copyright (C) 2011, Filipe Rodrigues <fmpr@dei.uc.pt>

import sys
import numpy as np
from malr.core import * 

class Classifier():
	"""Stub for a supervised classifier"""
	
	def __init__(self, trainset):
		self.compatibility_check(trainset)
		self.trainset = trainset
	
	def compatibility_check(self, dataset):
		raise Exception("compatibility_check method is not implemented!")
	
	def train(self):
		raise Exception("train method is not implemented!")
			
	def loglikelihood(self):
		raise Exception("loglikelihood method is not implemented!")
	
	def apply(self, instance):
		raise Exception("apply method is not implemented!")
	
	def evaluate(self, testset, confusion_matrix=False):
		self.compatibility_check(testset)
		accuracy = .0
		results = []
		for i in xrange(testset.num_instances):
			true = self.trainset.target_alphabet.lookup_index(testset.targets[i][0])
			pred, posterior = self.apply(testset.data[i])
			results.append((true, pred))
			if pred == true:
				accuracy += 1.0
			#print "true:", true, "\tpred:", pred, "\tprob:", posterior
		accuracy /= testset.num_instances
		
		if confusion_matrix:
			cm = ConfusionMatrix(results)
			cm.print_confusion_matrix()
		
		return accuracy
