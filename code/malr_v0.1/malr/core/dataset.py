# -*- coding: utf-8 -*-
# Copyright (C) 2011, Filipe Rodrigues <fmpr@dei.uc.pt>

import sys
import copy
import numpy as np

class Dataset:
	
	def __init__(self, name, feature_names, feature_dtypes, feature_alphabets, class_name, class_index, class_dtype, ids, targets, target_alphabet, annotators, annotators_alphabet, ma_feature, data):
		self.name = name
		self.num_instances = len(data)
		self.num_features = len(feature_names)
		self.feature_names = feature_names
		self.feature_dtypes = feature_dtypes
		self.feature_alphabets = feature_alphabets
		self.class_name = class_name
		self.class_index = class_index
		self.class_dtype = class_dtype
		self.ids = ids
		self.targets = targets
		self.target_alphabet = target_alphabet
		self.annotators = annotators
		self.annotators_alphabet = annotators_alphabet
		self.ma_feature = ma_feature
		self.data = data
		
	def __str__(self):
		description = "Dataset name: " + self.name + "\n"
		description += "Num instances: " + str(self.num_instances) + "\n"
		description += "Num features: " + str(self.num_features) + "\n"
		description += "Features: \n"
		for i in xrange(len(self.feature_names)):
			description += "\t" + str(i) + " " + str(self.feature_names[i]) + " " + str(self.feature_dtypes[i]) + "\n"
		description += "Class: " + self.class_name + " (" + str(self.class_dtype) + ")\n"
		description += "Class values: \n"
		for value in self.target_alphabet.get_objects():
			description += "\t" + str(value) + "\n"
		if self.annotators == None:
			description += "Num annotators: N/A"
		else:
			description += "Num annotators: " + str(len(self.annotators_alphabet))
		return description
		
	def percentage_split(self, percentage, keep_order=False):
		tmp_data = np.hstack((self.data, self.targets))
		if self.annotators != None:
			tmp_data = np.hstack((self.annotators, tmp_data))
		if not keep_order:
			np.random.shuffle(tmp_data)
		train, test = np.split(tmp_data, [self.num_instances*percentage], axis=0)
		if self.annotators != None:
			a,d,t = np.hsplit(train,[1,-1])
			trainset = Dataset(self.name + " (train/"+str(100*percentage)+"%)", self.feature_names, self.feature_dtypes, self.feature_alphabets, self.class_name, self.class_index, self.class_dtype, self.ids, t, self.target_alphabet, a, self.annotators_alphabet, self.ma_feature, d)
			a,d,t = np.hsplit(test,[1,-1])
			testset = Dataset(self.name + " (test/"+str(100*(1-percentage))+"%)", self.feature_names, self.feature_dtypes, self.feature_alphabets, self.class_name, self.class_index, self.class_dtype, self.ids, t, self.target_alphabet, a, self.annotators_alphabet, self.ma_feature, d)
		else:
			d,t = np.hsplit(train,[-1])
			trainset = Dataset(self.name + " (train/"+str(100*percentage)+"%)", self.feature_names, self.feature_dtypes, self.feature_alphabets, self.class_name, self.class_index, self.class_dtype, self.ids, t, self.target_alphabet, None, None, None, d)
			d,t = np.hsplit(test,[-1])
			testset = Dataset(self.name + " (test/"+str(100*(1-percentage))+"%)", self.feature_names, self.feature_dtypes, self.feature_alphabets, self.class_name, self.class_index, self.class_dtype, self.ids, t, self.target_alphabet, None, None, None, d)
		return trainset, testset

	def cross_validation_split(self, num_folds, keep_order=False):
		tmp_data = np.hstack((self.data, self.targets))
		if self.annotators != None:
			tmp_data = np.hstack((self.annotators, tmp_data))

		missing = num_folds - (float(len(tmp_data)) % float(num_folds))
		if float(len(tmp_data)) % float(num_folds) != 0:
			tmp_data = np.vstack((tmp_data, tmp_data[0:missing]))
		
		if not keep_order:
			np.random.shuffle(tmp_data)	
		folds = np.split(tmp_data, num_folds, axis=0)
		folds_datasets = []
		for fold in xrange(num_folds):
			train = None
			test = None
			for j in xrange(num_folds):
				if j == fold:
					test = folds[j]
				else:
					if train == None:
						train = folds[j].copy()
					else:
						train = np.vstack((train, folds[j]))

			if self.annotators != None:
				a,d,t = np.hsplit(train,[1,-1])
				trainset = Dataset(self.name + " (train/fold"+str(fold)+")", self.feature_names, self.feature_dtypes, self.feature_alphabets, self.class_name, self.class_index, self.class_dtype, self.ids, t, self.target_alphabet, a, self.annotators_alphabet, self.ma_feature, d)
				a,d,t = np.hsplit(test,[1,-1])
				testset = Dataset(self.name + " (test/fold"+str(fold)+")", self.feature_names, self.feature_dtypes, self.feature_alphabets, self.class_name, self.class_index, self.class_dtype, self.ids, t, self.target_alphabet, a, self.annotators_alphabet, self.ma_feature, d)
			else:
				d,t = np.hsplit(train,[-1])
				trainset = Dataset(self.name + " (train/fold"+str(fold)+")", self.feature_names, self.feature_dtypes, self.feature_alphabets, self.class_name, self.class_index, self.class_dtype, self.ids, t, self.target_alphabet, None, None, None, d)
				d,t = np.hsplit(test,[-1])
				testset = Dataset(self.name + " (test/fold"+str(fold)+")", self.feature_names, self.feature_dtypes, self.feature_alphabets, self.class_name, self.class_index, self.class_dtype, self.ids, t, self.target_alphabet, None, None, None, d)

			folds_datasets.append((trainset, testset))
		return folds_datasets

	def lookup_index(self, feature_index, index):
		return self.feature_alphabets[feature_index].lookup_index(index)
	
	def lookup_label(self, feature_index, obj):
		return self.feature_alphabets[feature_index].lookup_object(obj)
	
	def lookup_class_index(self, class_index):
		return self.target_alphabet.lookup_index(class_index)
	
	def lookup_class_label(self, class_value):
		return self.target_alphabet.lookup_object(class_value)

	def save_csv(self, filename):
		fw = open(filename, "w")

		# write header
		if self.annotators_alphabet != None:
			fw.write("annotator,")
		if self.ids != None:
			fw.write("id,")
		for feature in self.feature_names:
			fw.write(feature+",")
		fw.write(self.class_name+"\n")

		# write data
		for i in xrange(len(self.data)):
			if self.annotators_alphabet != None:
				fw.write(str(self.annotators_alphabet.lookup_index(self.annotators[i][0]))+",")
			if self.ids != None:
				fw.write(str(self.ids[i][0])+",")
			for feature in xrange(self.num_features):
				if feature >= len(self.feature_alphabets) or self.feature_alphabets[feature] == None:
					fw.write(str(self.data[i,feature])+",")
				else:
					fw.write(str(self.feature_alphabets[feature].lookup_index(self.data[i,feature]))+",")
			fw.write(str(self.target_alphabet.lookup_index(self.targets[i][0]))+"\n") 
		fw.close()
		
	def clone(self):
		return Dataset("copy of " + self.name, self.feature_names, self.feature_dtypes, self.feature_alphabets, self.class_name, self.class_index, self.class_dtype, self.ids.copy(), self.targets.copy(), self.target_alphabet, self.annotators.copy(), self.annotators_alphabet, self.ma_feature, self.data.copy())
		