# -*- coding: utf-8 -*-
# Copyright (C) 2011, Filipe Rodrigues <fmpr@dei.uc.pt>

import sys
import numpy as np

class ConfusionMatrix:
	"""Build and prints a confusion matrix from a list of (true, pred) pairs"""
	
	def __init__(self, results):
		self._results = results
		self._labels = np.unique(np.array(results)).tolist()
		self._confusion_matrix = np.zeros((len(self._labels), len(self._labels)))
		for true, pred in results:
			self._confusion_matrix[self._labels.index(true), self._labels.index(pred)] += 1
	
	def print_confusion_matrix(self):
		print "\nConfusion Matrix:"
		for i in xrange(len(self._labels)):
			if i == 0:
				s = " \t"
				for j in xrange(len(self._labels)):
					s += self._labels[j] + "\t"
				print s + " <- predicted"
			s = self._labels[i] + "\t"
			for j in xrange(len(self._labels)):
				s += str(self._confusion_matrix[i,j]) + "\t"
			print s
		print
		