# -*- coding: utf-8 -*-
# Copyright (C) 2011, Filipe Rodrigues <fmpr@dei.uc.pt>

import sys
import copy
import numpy as np
from dataset import *
from alphabet import *

class CSVDataset(Dataset):
	
	def __init__(self, filename, dataset_template=None, ma_feature=None, id_feature=None, delimiter=',', dtypes=None, class_index=None):
		name = filename.replace(".csv","")
		f = open(filename)
		header = f.readline().replace("\n","")
		
		# process provided ma_feature
		if ma_feature != None:
			new_feature_names = header.split(delimiter)
			if type(ma_feature) == str and ma_feature not in new_feature_names:
				raise Exception("Invalid multiple annotators feature: " + str(ma_feature))
			if type(ma_feature) == str:
				ma_feature = new_feature_names.index(ma_feature)
			elif type(ma_feature) != int:
				raise Exception("Invalid type for ma_feature")

		# process provided id_feature
		if id_feature != None:
			new_feature_names = header.split(delimiter)
			if type(id_feature) == str and id_feature not in new_feature_names:
				raise Exception("Invalid id feature: " + str(id_feature))
			if type(id_feature) == str:
				id_feature = new_feature_names.index(id_feature)
			elif type(id_feature) != int:
				raise Exception("Invalid type for id feature")
		
		if dataset_template != None:
			dtypes = dataset_template.feature_dtypes
			feature_names = dataset_template.feature_names
			class_name = dataset_template.class_name
			class_dtype = dataset_template.class_dtype
			class_index = dataset_template.class_index
			if dataset_template.annotators != None and dataset_template.class_index > dataset_template.ma_feature:
				class_index -= 1
			num_features = dataset_template.num_features
			feature_alphabets = dataset_template.feature_alphabets
			target_alphabet = dataset_template.target_alphabet
			annotators_alphabet = dataset_template.annotators_alphabet
		else:
			feature_names = header.split(delimiter)
			if dtypes != None:
				if type(dtypes) == list:
					assert len(feature_names) == len(self.dtypes)
				elif type(dtypes) == type:
					dtype = dtypes
					dtypes = []
					for i in xrange(len(feature_names)):
						dtypes.append(dtype)
				else:
					raise Exception("Invalid dtypes: " + str(dtypes))
			else:
				# infer dtypes from data
				dtypes = []
				for col in f.readline().split(delimiter):
					if col == ma_feature:
						continue 	# feature indicating the annotator is always treated as nominal
					dtype = str
					if col.lower() == "true" or col.lower() == "false":
						dtypes.append(bool)
						continue
					#try:
					#	value = int(col)
					#	dtypes.append(int)
					#	continue
					#except ValueError:
					#	pass
					try:
						value = float(col)
						dtypes.append(float)
						continue
					except ValueError:
						pass
					dtypes.append(str)
				# reset the file point to previous position (i.e. just after header)
				f.seek(0)
				f.readline()
			if class_index is None:
				class_index = len(feature_names) - 1 	# by default assume that last attribute is the class
			class_name = feature_names[class_index]
			class_dtype = dtypes[class_index]
		
			# remove ma_feature and class_index from feature_names and dtypes arrays
			if class_index > ma_feature:
				feature_names = feature_names[0:class_index] + feature_names[class_index+1:]
				dtypes = dtypes[0:class_index] + dtypes[class_index+1:]
				if ma_feature != None:
					feature_names = feature_names[0:ma_feature] + feature_names[ma_feature+1:]
					dtypes = dtypes[0:ma_feature] + dtypes[ma_feature+1:]
			else:
				feature_names = feature_names[0:ma_feature] + feature_names[ma_feature+1:]
				dtypes = dtypes[0:ma_feature] + dtypes[ma_feature+1:]
				if ma_feature != None:
					feature_names = feature_names[0:class_index] + feature_names[class_index+1:]
					dtypes = dtypes[0:class_index] + dtypes[class_index+1:]
			
			# remove id_feature from feature_names and dtypes arrays
			original_id_feature = id_feature
			if ma_feature != None and id_feature > ma_feature:
				id_feature -= 1
			if id_feature > class_index:
				id_feature -= 1
			if id_feature != None:
				feature_names = feature_names[0:id_feature] + feature_names[id_feature+1:]
				dtypes = dtypes[0:id_feature] + dtypes[id_feature+1:]
			id_feature = original_id_feature	
			num_features = len(feature_names)
			
			# initilize all necessary alphabets
			feature_alphabets = []
			for i in xrange(num_features):
				if dtypes[i] == str:
					feature_alphabets.append(Alphabet())
				else:
					feature_alphabets.append(None) # numeric features don't need an alphabet to map them
			if class_dtype == str:
				target_alphabet = Alphabet()
			else:
				target_alphabet = None
			if ma_feature != None:
				annotators_alphabet = Alphabet()
		
		line_no = 0
		data = []
		targets = []
		ids = []
		if ma_feature != None:
			annotators = []
		for line in f:
			line_no += 1
			line = line.replace("\n","")
			
			# parse instance and update alphabets along the way
			instance = []
			splited_line = line.split(delimiter)
			feature_index = 0
			for col in xrange(len(splited_line)):
				if col == ma_feature:
					annotators_alphabet.add(splited_line[col])
					annotators.append([annotators_alphabet.lookup_object(splited_line[col])])
				elif col == class_index:
					if class_dtype == str:
						target_alphabet.add(splited_line[col])
						targets.append([target_alphabet.lookup_object(splited_line[col])])
					elif class_dtype == int or class_dtype == bool:
						targets.append([int(splited_line[col])])
					else:
						targets.append([float(splited_line[col])])
				elif col == id_feature:
					ids.append([splited_line[col]])
				else:
					if dtypes[feature_index] == str:
						feature_alphabets[feature_index].add(splited_line[col])
						instance.append(feature_alphabets[feature_index].lookup_object(splited_line[col]))
					elif dtypes[feature_index] == int or dtypes[feature_index] == bool:
						instance.append(int(splited_line[col]))
					else:
						instance.append(float(splited_line[col]))
					feature_index += 1
			data.append(instance)
		f.close()
		num_instances = line_no
		data = np.array(data)
		targets = np.array(targets)
		if id_feature != None:
			ids = np.array(ids)
		else:
			ids = None
		if ma_feature != None:
			annotators = np.array(annotators)
		else:
			annotators = None
			annotators_alphabet = None
		class_values = np.unique(targets)

		Dataset.__init__(self, name, feature_names, dtypes, feature_alphabets, class_name, class_index, class_dtype, ids, targets, target_alphabet, annotators, annotators_alphabet, ma_feature, data)

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
