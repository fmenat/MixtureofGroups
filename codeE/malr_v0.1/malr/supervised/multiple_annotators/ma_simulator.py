import sys, os
import numpy as np
#print os.getcwd()[0:os.getcwd().rindex("/")]
sys.path = ['/Users/fmpr/Work/git/malr'] + sys.path
from malr.core import * 
from malr.supervised import *

class MASimulator():
	
	def simulate_annotators_by_random_flips(self, dataset, annotators_flip_probs):
		print "simulating annotators with random flips"
		print "annotators_flip_probs:", annotators_flip_probs
		original_ids = dataset.ids
		original_targets = dataset.targets
		num_classes = len(dataset.target_alphabet)
		num_annotators = len(annotators_flip_probs)
		new_ids = []
		annotators = []
		new_targets = []
		new_data = []
		annotators_alphabet = Alphabet()
		for annotator in xrange(num_annotators):
			annotators_alphabet.add("annotator"+str(annotator))
		for i in xrange(dataset.num_instances):
			label = dataset.targets[i][0]
			#print "true label:", label
			for annotator in xrange(num_annotators):
				annotator_label = label
				if np.random.rand() < annotators_flip_probs[annotator]:
					# pick a new random label that is different from the true label
					while annotator_label == label:
						annotator_label = int(np.random.rand()*num_classes) 
				#print "annotator " + str(annotator) + " label:", annotator_label
				new_ids.append([dataset.ids[i][0]])
				new_targets.append([annotator_label])
				new_data.append(dataset.data[i])
				annotators.append([annotator])
			#print
		new_data = np.array(new_data)	
		new_targets = np.array(new_targets)
		annotators = np.array(annotators)
		dataset.annotators = annotators
		dataset.data = new_data
		dataset.ids = np.array(new_ids)
		dataset.targets = new_targets
		dataset.num_instances = len(new_data)
		dataset.annotators_alphabet = annotators_alphabet
		
		# compute annotators accuracies
		accuracies = np.zeros(num_annotators)
		counts = np.zeros(num_annotators)
		for i in xrange(dataset.num_instances):
			_id = dataset.ids[i][0]
			annotator = dataset.annotators[i][0]
			annotator_label = dataset.targets[i][0]
			original_index = np.where(original_ids == _id)[0][0]
			true_label = original_targets[original_index][0]
			if annotator_label == true_label:
				accuracies[annotator] += 1.0
			counts[annotator] += 1.0
		accuracies /= counts
		print "annotators accuracies:", accuracies
		
		return dataset, accuracies
		
		
	def simulate_annotators_by_random_flips_less_rep(self, dataset, annotators_flip_probs, annotation_probs):
		print "simulating annotators with random flips (less-repeated labeling)"
		print "annotators_flip_probs:", annotators_flip_probs
		original_ids = dataset.ids
		original_targets = dataset.targets
		num_classes = len(dataset.target_alphabet)
		num_annotators = len(annotators_flip_probs)
		new_ids = []
		annotators = []
		new_targets = []
		new_data = []
		annotators_alphabet = Alphabet()
		for annotator in xrange(num_annotators):
			annotators_alphabet.add("annotator"+str(annotator))
		for i in xrange(dataset.num_instances):
			label = dataset.targets[i][0]
			#print "true label:", label
			for annotator in xrange(num_annotators):
				# decide whether or not he annotates this instance
				if np.random.rand() > annotation_probs[annotator]:
					continue
				annotator_label = label
				if np.random.rand() < annotators_flip_probs[annotator]:
					# pick a new random label that is different from the true label
					while annotator_label == label:
						annotator_label = int(np.random.rand()*num_classes) 
				#print "annotator " + str(annotator) + " label:", annotator_label
				new_ids.append([dataset.ids[i][0]])
				new_targets.append([annotator_label])
				new_data.append(dataset.data[i])
				annotators.append([annotator])
			#print
		new_data = np.array(new_data)	
		new_targets = np.array(new_targets)
		annotators = np.array(annotators)
		dataset.annotators = annotators
		dataset.data = new_data
		dataset.ids = np.array(new_ids)
		dataset.targets = new_targets
		dataset.num_instances = len(new_data)
		dataset.annotators_alphabet = annotators_alphabet

		# compute annotators accuracies
		accuracies = np.zeros(num_annotators)
		counts = np.zeros(num_annotators)
		for i in xrange(dataset.num_instances):
			_id = dataset.ids[i][0]
			annotator = dataset.annotators[i][0]
			annotator_label = dataset.targets[i][0]
			original_index = np.where(original_ids == _id)[0][0]
			true_label = original_targets[original_index][0]
			if annotator_label == true_label:
				accuracies[annotator] += 1.0
			counts[annotator] += 1.0
		accuracies /= counts
		print "annotators accuracies:", accuracies
		print "counts:", counts

		return dataset, accuracies
		
	def simulate_annotators_by_random_flips_non_rep(self, dataset, annotators_flip_probs):
		print "simulating annotators with random flips (non-repeated labeling)"
		print "annotators_flip_probs:", annotators_flip_probs
		original_ids = dataset.ids
		original_targets = dataset.targets
		num_classes = len(dataset.target_alphabet)
		num_annotators = len(annotators_flip_probs)
		new_ids = []
		annotators = []
		new_targets = []
		new_data = []
		annotators_alphabet = Alphabet()
		for annotator in xrange(num_annotators):
			annotators_alphabet.add("annotator"+str(annotator))
		annotator = 0
		annotator_instances = 0
		for i in xrange(dataset.num_instances):
			annotator_instances += 1
			if annotator_instances > (dataset.num_instances / num_annotators):
				print "next annotator at instance:", i
				annotator += 1
				annotator_instances = 0
			label = dataset.targets[i][0]
			
			annotator_label = label
			if np.random.rand() < annotators_flip_probs[annotator]:
				# pick a new random label that is different from the true label
				while annotator_label == label:
					annotator_label = int(np.random.rand()*num_classes) 
			#print "annotator " + str(annotator) + " label:", annotator_label
			new_ids.append([dataset.ids[i][0]])
			new_targets.append([annotator_label])
			new_data.append(dataset.data[i])
			annotators.append([annotator])
			#print
		new_data = np.array(new_data)	
		new_targets = np.array(new_targets)
		annotators = np.array(annotators)
		dataset.annotators = annotators
		dataset.data = new_data
		dataset.ids = np.array(new_ids)
		dataset.targets = new_targets
		dataset.num_instances = len(new_data)
		dataset.annotators_alphabet = annotators_alphabet

		# compute annotators accuracies
		accuracies = np.zeros(num_annotators)
		counts = np.zeros(num_annotators)
		for i in xrange(dataset.num_instances):
			_id = dataset.ids[i][0]
			annotator = dataset.annotators[i][0]
			annotator_label = dataset.targets[i][0]
			original_index = np.where(original_ids == _id)[0][0]
			true_label = original_targets[original_index][0]
			if annotator_label == true_label:
				accuracies[annotator] += 1.0
			counts[annotator] += 1.0
		accuracies /= counts
		print "annotators accuracies:", accuracies
		print "counts:", counts

		return dataset, accuracies		
	
	def simulate_annotators_by_lr(self, dataset, annotators_noise):
		print "simulating annotators with logistic regression"
		print "annotators_noise:", annotators_noise
		
		lr = LogisticRegression(dataset)
		lr.train()
		annotators_weights = []
		original_weights = lr.weights.copy()
		for annotator in xrange(len(annotators_noise)):
			new_weights = lr.weights.copy()
			for j in xrange(len(new_weights)):
				#new_weights[j] += np.random.normal(0, np.abs(10*new_weights[j]*annotators_noise[annotator]))
				#new_weights[j] += np.random.normal(0, np.abs(lr.weights).max() * annotators_noise[annotator])
				new_weights[j] += np.random.normal(0, annotators_noise[annotator])
			annotators_weights.append(new_weights)
		
		original_ids = dataset.ids
		original_targets = dataset.targets
		num_classes = len(dataset.target_alphabet)
		num_annotators = len(annotators_noise)
		new_ids = []
		annotators = []
		new_targets = []
		new_data = []
		annotators_alphabet = Alphabet()
		for annotator in xrange(num_annotators):
			annotators_alphabet.add("annotator"+str(annotator))
		for annotator in xrange(num_annotators):
			#print "annotator:", annotator
			lr.weights = annotators_weights[annotator]
			for i in xrange(dataset.num_instances):
				true_label = dataset.targets[i][0]
				annotator_label = np.argmax(lr.posterior(dataset.data[i]))
				#print "\t", i, true_label, annotator_label
				new_ids.append([dataset.ids[i][0]])
				new_targets.append([annotator_label])
				new_data.append(dataset.data[i][1:])
				annotators.append([annotator])
			#print
		new_data = np.array(new_data)	
		new_targets = np.array(new_targets)
		annotators = np.array(annotators)
		dataset.num_features -= 1
		dataset.annotators = annotators
		dataset.data = new_data
		dataset.ids = np.array(new_ids)
		dataset.targets = new_targets
		dataset.num_instances = len(new_data)
		dataset.annotators_alphabet = annotators_alphabet

		# compute annotators accuracies
		accuracies = np.zeros(num_annotators)
		counts = np.zeros(num_annotators)
		for i in xrange(dataset.num_instances):
			_id = dataset.ids[i][0]
			annotator = dataset.annotators[i][0]
			annotator_label = dataset.targets[i][0]
			original_index = np.where(original_ids == _id)[0][0]
			true_label = original_targets[original_index][0]
			if annotator_label == true_label:
				accuracies[annotator] += 1.0
			counts[annotator] += 1.0
		accuracies /= counts
		print "annotators accuracies:", accuracies

		return dataset, accuracies

	def simulate_annotators_by_lr_less_rep(self, dataset, annotators_noise, annotation_probs):
		print "simulating annotators with logistic regression"
		print "annotators_noise:", annotators_noise

		lr = LogisticRegression(dataset)
		lr.train()
		annotators_weights = []
		original_weights = lr.weights.copy()
		for annotator in xrange(len(annotators_noise)):
			new_weights = lr.weights.copy()
			new_weights = (new_weights - lr.weights.mean()) / lr.weights.std()
			for j in xrange(len(new_weights)):
				#new_weights[j] += np.random.normal(0, np.abs(10*new_weights[j]*annotators_noise[annotator]))
				#new_weights[j] += np.random.normal(0, np.abs(lr.weights).max() * annotators_noise[annotator])
				new_weights[j] += np.random.normal(0, annotators_noise[annotator])
			new_weights = lr.weights.std() * new_weights + lr.weights.mean()
			annotators_weights.append(new_weights)

		original_ids = dataset.ids
		original_targets = dataset.targets
		num_classes = len(dataset.target_alphabet)
		num_annotators = len(annotators_noise)
		new_ids = []
		annotators = []
		new_targets = []
		new_data = []
		annotators_alphabet = Alphabet()
		for annotator in xrange(num_annotators):
			annotators_alphabet.add("annotator"+str(annotator))
		for annotator in xrange(num_annotators):
			#print "annotator:", annotator
			lr.weights = annotators_weights[annotator]
			for i in xrange(dataset.num_instances):
				# decide whether or not he annotates this instance
				if np.random.rand() > annotation_probs[annotator]:
					continue
				true_label = dataset.targets[i][0]
				annotator_label = np.argmax(lr.posterior(dataset.data[i]))
				#print "\t", i, true_label, annotator_label
				new_ids.append([dataset.ids[i][0]])
				new_targets.append([annotator_label])
				new_data.append(dataset.data[i][1:])
				annotators.append([annotator])
			#print
		new_data = np.array(new_data)	
		new_targets = np.array(new_targets)
		annotators = np.array(annotators)
		dataset.num_features -= 1
		dataset.annotators = annotators
		dataset.data = new_data
		dataset.ids = np.array(new_ids)
		dataset.targets = new_targets
		dataset.num_instances = len(new_data)
		dataset.annotators_alphabet = annotators_alphabet

		# compute annotators accuracies
		accuracies = np.zeros(num_annotators)
		counts = np.zeros(num_annotators)
		for i in xrange(dataset.num_instances):
			_id = dataset.ids[i][0]
			annotator = dataset.annotators[i][0]
			annotator_label = dataset.targets[i][0]
			original_index = np.where(original_ids == _id)[0][0]
			true_label = original_targets[original_index][0]
			if annotator_label == true_label:
				accuracies[annotator] += 1.0
			counts[annotator] += 1.0
		accuracies /= counts
		print "annotators accuracies:", accuracies

		return dataset, accuracies
	
	def simulate_annotators_by_model_less_rep(self, dataset, annotators_model_probs, annotation_probs):
		print "simulating annotators with model (less-repeated labeling)"
		print "annotators_model_probs:", annotators_model_probs
		
		lr = LogisticRegression(dataset)
		lr.train()
		
		original_ids = dataset.ids
		original_targets = dataset.targets
		num_classes = len(dataset.target_alphabet)
		num_annotators = len(annotators_model_probs)
		new_ids = []
		annotators = []
		new_targets = []
		new_data = []
		annotators_alphabet = Alphabet()
		for annotator in xrange(num_annotators):
			annotators_alphabet.add("annotator"+str(annotator))
		for annotator in xrange(num_annotators):
			#print "annotator:", annotator
			for i in xrange(dataset.num_instances):
				# sample z_i from a bernoulli with parameter pi
				z_i = np.random.binomial(1, annotators_model_probs[annotator])
				if z_i == 1:
					# label according to lr model
					annotator_label = np.argmax(lr.posterior(dataset.data[i]))
				else:
					# label according to random model
					annotator_label = int(np.random.rand()*num_classes) 
				#print "\t", i, true_label, annotator_label
				new_ids.append([dataset.ids[i][0]])
				new_targets.append([annotator_label])
				new_data.append(dataset.data[i][1:])
				annotators.append([annotator])
			#print
		new_data = np.array(new_data)	
		new_targets = np.array(new_targets)
		annotators = np.array(annotators)
		dataset.num_features -= 1
		dataset.annotators = annotators
		dataset.data = new_data
		dataset.ids = np.array(new_ids)
		dataset.targets = new_targets
		dataset.num_instances = len(new_data)
		dataset.annotators_alphabet = annotators_alphabet

		# compute annotators accuracies
		accuracies = np.zeros(num_annotators)
		counts = np.zeros(num_annotators)
		for i in xrange(dataset.num_instances):
			_id = dataset.ids[i][0]
			annotator = dataset.annotators[i][0]
			annotator_label = dataset.targets[i][0]
			original_index = np.where(original_ids == _id)[0][0]
			true_label = original_targets[original_index][0]
			if annotator_label == true_label:
				accuracies[annotator] += 1.0
			counts[annotator] += 1.0
		accuracies /= counts
		print "annotators accuracies:", accuracies

		return dataset, accuracies
	
if __name__=="__main__":
	dataset = CSVDataset('../../../test/data/breast-cancer-wisconsin-train.csv', id_feature='id')
	print dataset.data.shape
	masimulator = MASimulator()
	#artificial_dataset = masimulator.simulate_annotators_by_random_flips(dataset, [0.1, 0.1, 0.5, 0.7, 0.7]) 
	artificial_dataset = masimulator.simulate_annotators_by_lr(dataset, [0.1, 0.1, 0.5, 0.7, 0.7])
	#artificial_dataset.save_csv('../../../test/data/breast-cancer-wisconsin-train-ma3.csv')
