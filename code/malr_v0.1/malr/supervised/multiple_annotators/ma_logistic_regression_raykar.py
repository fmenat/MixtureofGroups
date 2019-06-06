# -*- coding: utf-8 -*-
# Copyright (C) 2012, Filipe Rodrigues <fmpr@dei.uc.pt>

import sys
import numpy as np
from malr.core import * 
from scipy.sparse.linalg import spilu
from scipy.optimize.optimize import fmin_cg, fmin_ncg, fmin_bfgs

class MALogisticRegressionRaykar(Classifier):
	"""Multiple Annotator version of LogisticRegression"""
	
	"""Valid optimization methods are: 
		- Limited-memory BFGS (l-bfgs)
		- Newton's Conjugate Gradients (ncg)
		- Conjugate Gradients (cg)
		- Iterative Re-weighted Least Squares (irls)
		- Gradient Descent (gd)"""
	def __init__(self, trainset, ground_truth=None, testset=None, max_em_iter=10, max_iter=10, stoping_threshold=0.01, guassian_prior_sigma=1, optimization_method="l-bfgs"):
		self.compatibility_check(trainset)
		if trainset.annotators == None:
			raise Exception("Trainset for a MALogisticRegression model must have ma_feature set!")
		if trainset.ids == None:
			raise Exception("Trainset for a MALogisticRegression model must have id_feature set!")
		self.trainset = trainset
		self.ground_truth = ground_truth
		self.testset = testset
		self.num_classes = len(self.trainset.target_alphabet)
		self.max_em_iter = max_em_iter
		self.max_iter = max_iter
		self.stoping_threshold = stoping_threshold
		self.optimization_method = optimization_method
		self.guassian_prior_sigma = guassian_prior_sigma
	
	def compatibility_check(self, dataset):
		# check if class if nominal of binary
		if dataset.class_dtype != str:
			raise Exception("NaiveBayes can only handle nominal and binary classes/labels.")
			
	def majority_voting(self, dataset):
		print "Estimating ground truth using Majority Voting..."
		
		# vote
		votes = {}
		for i in xrange(dataset.num_instances):
			_id = dataset.ids[i][0]
			if not votes.has_key(_id):
				votes[_id] = np.zeros(self.num_classes)
			votes[_id][dataset.targets[i]] += 1
		estimated_ground_truth = []
		
		# compute most voted
		for i in xrange(dataset.num_instances):
			_id = dataset.ids[i][0]
			estimated_ground_truth.append(votes[_id] / votes[_id].sum())
		estimated_ground_truth = np.array(estimated_ground_truth)
		
		# compute majority voting accuracy
		if self.ground_truth != None:
			acc = 0.0
			count = 0.0
			for i in xrange(self.ground_truth.num_instances):
				_id = self.ground_truth.ids[i][0]
				if not votes.has_key(_id):
					continue
				if self.ground_truth.targets[i] == np.argmax(votes[_id]):
					acc += 1.0
				count += 1.0
			print "\tMajority Voting accuracy:", (acc / count)
		return estimated_ground_truth
	
	def train(self):
		print "Training Multiple Annotator Logistic Regression model..."
		
		self.trainset.targets = self.trainset.targets.astype(int)
		if self.ground_truth != None:
			self.ground_truth.targets = self.ground_truth.targets.astype(int)
		if self.testset != None:
			self.testset.targets = self.testset.targets.astype(int)
		
		# add feature for bias parameter (bias feature is always 1)
		self.trainset.data = np.hstack((np.ones(self.trainset.num_instances).reshape(-1,1), self.trainset.data) )
		self.trainset.num_features += 1
		self.ground_truth.data = np.hstack((np.ones(self.ground_truth.num_instances).reshape(-1,1), self.ground_truth.data) )
		self.ground_truth.num_features += 1
		
		# estimate ground truth using majority voting
		self.estimated_ground_truth = self.majority_voting(self.trainset)
		print self.estimated_ground_truth
		print self.estimated_ground_truth.shape
		
		# compute t_k (i.e. annotators targets in a 1-of-K coding scheme)
		self.t_k = np.zeros((self.trainset.num_instances, self.num_classes)) 
		self.t_k[np.arange(self.trainset.num_instances), self.trainset.targets.T] = 1.0  # slice indexes of interest and set them to 1
		
		print "\nStarting EM..."
		iteration = 0
		self.stats = []
		while(iteration < self.max_em_iter):
			iteration += 1
			print "\nEM iteration:", iteration
			self.stats.append({"iteration":iteration})
			self.mstep()
			self.estep()
		
		print "\nEM stopped after " + str(iteration) + " iterations"	
		
		# show stats
		print "\nEM statistics:"
		for d in self.stats:
			print d
		print
		
		# show accuracies
		if self.ground_truth != None:
			print "Ground truth accuracy:", self.evaluate(self.ground_truth, confusion_matrix=True)
		if self.testset != None:
			print "Testset accuracy:", self.evaluate(self.testset, confusion_matrix=True)
	
	def estep(self):
		print "\nE-Step"
		
		# re-estimate ground truth
		y_k = self.posteriors(self.trainset.data, self.weights)
		
		estimated_ground_truth = np.zeros((self.trainset.num_instances, self.num_classes))
		for i in xrange(self.trainset.num_instances):
			_id = self.trainset.ids[i]
			annotators = self.trainset.annotators[np.where(self.trainset.ids == _id)[0]]
			annotators_labels = self.t_k[np.where(self.trainset.ids == _id)[0]]
			for c in xrange(self.num_classes):
				a_ic = 1.0
				for r in xrange(len(annotators_labels)):
					annotator = annotators[r][0]	
					#print "annotator:", annotator			
					for k in xrange(self.num_classes):
						if self.alphas[annotator,c,k] == 0:
							pass
						a_ic *= (self.alphas[annotator,c,k]**annotators_labels[r,k])
						#print r,k,self.alphas[r,c,k],annotators_labels[r,k],a_ic
				#print i, c, a_ic
				estimated_ground_truth[i,c] = a_ic * y_k[i,c]
			estimated_ground_truth[i] /= estimated_ground_truth[i].sum()
			#print i, np.argmax(estimated_ground_truth[i]), estimated_ground_truth[i].sum()
		print self.estimated_ground_truth
		print estimated_ground_truth
		self.estimated_ground_truth = estimated_ground_truth
				
	
	def mstep(self):
		print "\nM-Step"
		
		# compute alphas
		# TODO: Optimize later - use numpy operations
		self.num_annotators = len(self.trainset.annotators_alphabet)
		print "Num Annotators:", self.num_annotators
		#self.alphas = np.zeros((self.num_annotators, self.num_classes, self.num_classes)) # assume all annotators are equally good
		self.alphas = 1.0 + np.zeros((self.num_annotators, self.num_classes, self.num_classes)) # assume all annotators are equally good
		for c in xrange(self.num_classes):
			#denom_c = np.zeros(self.num_annotators)
			denom_c = self.num_classes + np.zeros(self.num_annotators)
			for i in xrange(self.trainset.num_instances):
				annotator = self.trainset.annotators[i][0]
				#print "annotator", annotator
				denom_c[annotator] += self.estimated_ground_truth[i][c]
				for k in xrange(self.num_classes):
					self.alphas[annotator][c][k] += (self.estimated_ground_truth[i][c] * self.t_k[i][k])
			for annotator in xrange(self.num_annotators):
				self.alphas[annotator][c] /= denom_c[annotator]
		
		# initialize weights to zero
		#self.weights = np.zeros((self.trainset.num_features, self.num_classes-1))  # initialize with zeros
		self.weights = -.05 + 0.1 * np.random.rand(self.trainset.num_features * (self.num_classes-1))  # random initialization (close to zero)
				
		# learn model weights
		if self.optimization_method.lower() == "l-bfgs":
			self.weights = self._optimize_lbfgs()
		elif self.optimization_method.lower() == "cg":
			self.weights = self._optimize_cg()
		elif self.optimization_method.lower() == "ncg":
			self.weights = self._optimize_ncg()
		elif self.optimization_method.lower() == "irls":
			self.weights = self._optimize_irls()
		elif self.optimization_method.lower() == "gd":
			self.weights = self._optimize_gd(learning_rate=0.01)
		else:
			self.weights = self._optimize_lbfgs() # by default use L-BFGS

		# show current accuracies
		if self.ground_truth != None:
			ground_truth_accuracy = self.evaluate(self.ground_truth, confusion_matrix=False)
			self.stats[len(self.stats)-1]["Ground truth accuracy"] = ground_truth_accuracy
			print "Ground truth accuracy:", ground_truth_accuracy * 100
			print "Annotators RMSEs: ", self.annotators_rmse(self.ground_truth)
		if self.testset != None:
			testset_accuracy = self.evaluate(self.testset, confusion_matrix=False)
			self.stats[len(self.stats)-1]["Testset accuracy"] = testset_accuracy
			print "Testset accuracy:", testset_accuracy * 100
		
	def _optimize_lbfgs(self):
		# parameter optimization with quasi-Newton method L-BFGS
		print "Optimizing with L-BFGS"
		return fmin_bfgs(self.negative_loglikelihood, self.weights, fprime=self.negative_gradient, maxiter=200)
		
	def _optimize_ncg(self):
		# parameter optimization with Newton's conjugate gradients method
		print "Optimizing with Newton's Conjugate Gradients"
	
		# compute the dot product of dataset with its own transpose 
		# this necessary for computing the hessian (doing this here avoid unnecessary re-calculations)
		self._xx = np.dot(self.trainset.data.T, self.trainset.data)
		
		return fmin_ncg(self.negative_loglikelihood, self.weights, fprime=self.negative_gradient, fhess=self.negative_hessian)
		
	def _optimize_cg(self):
		# parameter optimization with conjugate gradients method
		print "Optimizing with Conjugate Gradients"
		return fmin_cg(self.negative_loglikelihood, self.weights, fprime=self.negative_gradient)
		
	def _optimize_irls(self):
		# parameter optimization with Newton-Raphson method IRLS
		print "Optimizing with IRLS"
	
		# compute the dot product of dataset with its own transpose 
		# this necessary for computing the hessian (doing this here avoid unnecessary re-calculations)
		self._xx = np.dot(self.trainset.data.T, self.trainset.data)
	
		loglikelihood = None
		old_loglikelihood = None
		iteration = 0
		while iteration < self.max_iter and (iteration <= 1 or (loglikelihood - old_loglikelihood) > self.stoping_threshold):
			iteration += 1
			print "\nIteration:", iteration
			
			# compute y_k for each possible class value (i.e. compute the posteriors)
			y_k = self.posteriors(self.trainset.data, self.weights)
		
			# compute loglikelihood
			old_loglikelihood = loglikelihood
			loglikelihood = self.parameters_loglikelihood(self.weights, y_k=y_k)
			
			# show current accuracy on trainset
			accuracy = self.evaluate(self.trainset)
			print "accuracy:", accuracy
			
			# if loglikelihood improvement falls below the defined threshold stop
			if iteration > 1 and (loglikelihood - old_loglikelihood) <= self.stoping_threshold:
				print "Optimization jump too small"
				self.weights = old_weights
				break
		
			# compute gradients
			gradients = self.gradient(self.weights, y_k=y_k)
		
			# compute Hessian (required for IRLS)
			H = self.hessian(self.weights, y_k=y_k)
			H_inv = np.linalg.inv(H)
			
			# update weights
			old_weights = self.weights
			self.weights = self.weights.flatten('F') + np.dot(H_inv, gradients) 	# Newton-Raphson update (IRLS)
		print "\nOptimization stoped after " + str(iteration) + " iterations"
		return old_weights
		
	def _optimize_gd(self, learning_rate=0.01):
		# parameter optimization with Gradient Descent
		print "Optimizing with Gradient Descent"
		loglikelihood = None
		old_loglikelihood = None
		iteration = 0
		while iteration < self.max_iter and (iteration <= 5 or (loglikelihood - old_loglikelihood) > self.stoping_threshold):
			iteration += 1
			print "\nIteration:", iteration
			
			# compute y_k for each possible class value (i.e. compute the posteriors)
			y_k = self.posteriors(self.trainset.data, self.weights)
		
			# compute loglikelihood
			old_loglikelihood = loglikelihood
			loglikelihood = self.parameters_loglikelihood(self.weights, y_k=y_k)
			
			# show current accuracy on trainset
			accuracy = self.evaluate(self.trainset)
			print "accuracy:", accuracy
			
			# if loglikelihood improvement falls below the defined threshold stop (but never do less than 5 iterations...)
			if iteration > 5 and (loglikelihood - old_loglikelihood) <= self.stoping_threshold:
				print "Optimization jump too small"
				self.weights = old_weights
				break
		
			# compute gradients
			gradients = self.gradient(self.weights, y_k=y_k)
			
			# update weights
			old_weights = self.weights
			self.weights = self.weights.flatten('F') + learning_rate * gradients # Gradient Descent update
		print "\nOptimization stoped after " + str(iteration) + " iterations"
		return self.weights
			
	def loglikelihood(self):
		return self.parameters_loglikelihood(self.weights)
		
	def parameters_loglikelihood(self, w, y_k=None):
		# compute class posteriors (if not provided)
		if y_k == None:
			y_k = self.posteriors(self.trainset.data, w) 
			
		y_Ci = y_k[np.arange(self.trainset.num_instances), self.trainset.targets.T].T # slice indexes of interest
		
		loglikelihood = np.sum(self.estimated_ground_truth * np.log(y_k)) # multiple annotators version
		#loglikelihood = np.sum(np.log(y_Ci)) # old
		
		l2_regularization = np.sum(w**2 / (2*self.guassian_prior_sigma**2))
		loglikelihood -= l2_regularization
		print "loglikelihood:", loglikelihood
		return loglikelihood
	
	def negative_loglikelihood(self, w):
		return -1.0*self.parameters_loglikelihood(w)
		
	def gradient(self, w, y_k=None):
		# compute class posteriors (if not provided)
		if y_k == None:
			y_k = self.posteriors(self.trainset.data, w)
		#print y_k
		
		gradients = np.dot(np.transpose(self.estimated_ground_truth - y_k), self.trainset.data).T # multiple annotators version
		#gradients_old = np.dot(np.transpose(self.t_k - y_k), self.trainset.data).T # old
		
		# flatten matrix into an array
		gradients = gradients[:,:-1].flatten('C')
		
		l2_regularization = w / (self.guassian_prior_sigma**2)
		gradients -= l2_regularization.flatten()
		return gradients
		
	def negative_gradient(self, w, y_k=None):
		return -1.0*self.gradient(w, y_k=y_k)
	
	def hessian(self, w, y_k=None):
		# compute class posteriors (if not provided)
		if y_k == None:
			y_k = self.posteriors(self.trainset.data, w)
		
		H = np.zeros((self.trainset.num_features * (self.num_classes-1), 
						self.trainset.num_features * (self.num_classes-1)))
		for j in xrange(self.num_classes-1):
			for k in xrange(self.num_classes-1):
				H_k_j = np.dot(y_k[:,k], (float(j==k) - y_k[:,j])) * self._xx  # _xx is the dot product of dataset with its own transpose
				for m1 in xrange(self.trainset.num_features):
					for m2 in xrange(self.trainset.num_features):
						H[j*self.trainset.num_features+m1, k*self.trainset.num_features+m2] = H_k_j[m1, m2]
		return H
		
	def negative_hessian(self, w, y_k=None):
		return -1.0 * self.hessian(w, y_k=y_k)
	
	def apply(self, instance):
		y_k = self.posterior(instance)
		best_index = np.argmax(y_k)
		best_label = self.trainset.target_alphabet.lookup_index(best_index)
		return best_label, y_k[best_index]
		
	def posterior(self, instance):
		a_k = np.dot(instance, self.weights.reshape(self.trainset.num_features, -1))
		a_k = np.hstack((a_k, np.zeros(1)))
		y_k = np.zeros(self.num_classes)
		for k in xrange(self.num_classes):
			exp = np.exp(a_k - a_k[k])
			_sum = np.nan_to_num(exp.sum())
			y_k[k] = 1.0 / _sum
		return y_k
	
	def posteriors(self, instances, weights):
		a_k = np.dot(instances, weights.reshape(self.trainset.num_features, -1))
		a_k = np.hstack((a_k, np.zeros(len(instances)).reshape(-1,1)))
		y_k = np.zeros((len(instances), self.num_classes))
		for k in xrange(self.num_classes):
			exp = np.transpose(np.exp(a_k.T - a_k[:,k]))
			_sum = np.nan_to_num(exp.sum(axis=1))
			y_k[:,k] = 1.0 / _sum
		return y_k
		
	def evaluate(self, testset, confusion_matrix=False):
		if testset.data.shape[1] == self.trainset.data.shape[1]-1:
			# need to add bias feature (always set to 1)
			testset.data = np.hstack((np.ones(testset.num_instances).reshape(-1,1), testset.data))
			testset.num_features += 1
		
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
			#print self.trainset.target_alphabet.lookup_index(0), self.testset.target_alphabet.lookup_index(0)
			#print self.trainset.target_alphabet.lookup_index(1), self.testset.target_alphabet.lookup_index(1)
			#print self.trainset.target_alphabet.lookup_index(2), self.testset.target_alphabet.lookup_index(2)
			#print
		accuracy /= testset.num_instances

		if confusion_matrix:
			cm = ConfusionMatrix(results)
			cm.print_confusion_matrix()

		return accuracy

	def annotators_rmse(self, ground_truth):
		assert self.trainset.annotators != None

		num_annotators = len(self.trainset.annotators_alphabet)

		# compute true annotators accuracies
		true_accuracies = np.zeros(num_annotators)
		normalizing_counts = np.zeros(num_annotators)
		for i in xrange(self.trainset.num_instances):
			annotator = self.trainset.annotators[i]
			annotator_label = self.trainset.targets[i][0]
			true_label = ground_truth.targets[np.where(ground_truth.ids == self.trainset.ids[i])[0][0]][0]
			if annotator_label == true_label:
				true_accuracies[annotator] += 1.0
			normalizing_counts[annotator] += 1.0
		true_accuracies /= normalizing_counts
		#print "true_accuracies:", true_accuracies

		# compute estimated annotators accuracies
		y_k = np.argmax(self.posteriors(self.trainset.data, self.weights), axis=1)
		pred_accuracies = np.zeros(num_annotators)
		normalizing_counts = np.zeros(num_annotators)
		for i in xrange(self.trainset.num_instances):
			annotator = self.trainset.annotators[i]
			annotator_label = self.trainset.targets[i][0]
			pred_label = y_k[i]
			if annotator_label == pred_label:
				pred_accuracies[annotator] += 1.0
			normalizing_counts[annotator] += 1.0
		pred_accuracies /= normalizing_counts
		#print "pred_accuracies:", pred_accuracies

		true_accuracies *= 100.0
		pred_accuracies *= 100.0
		diff = true_accuracies - pred_accuracies
		diff = diff*diff
		rmse = np.sqrt(diff.mean())
		#print "rmse:", rmse

		return rmse, pred_accuracies, true_accuracies

	def annotators_rmse2(self, ground_truth):
		assert self.trainset.annotators != None

		num_annotators = len(self.trainset.annotators_alphabet)

		# compute true annotators accuracies
		true_accuracies = np.zeros((num_annotators, self.num_classes))
		normalizing_counts = np.zeros((num_annotators, self.num_classes))
		for i in xrange(self.trainset.num_instances):
			annotator = self.trainset.annotators[i]
			annotator_label = self.trainset.targets[i][0]
			true_label = ground_truth.targets[np.where(ground_truth.ids == self.trainset.ids[i])[0][0]][0]
			if annotator_label == true_label:
				true_accuracies[annotator,true_label] += 1.0
			normalizing_counts[annotator,true_label] += 1.0
		true_accuracies /= normalizing_counts
		#print "true_accuracies:", true_accuracies

		# compute estimated annotators accuracies
		y_k = np.argmax(self.posteriors(self.trainset.data, self.weights), axis=1)
		pred_accuracies = np.zeros((num_annotators, self.num_classes))
		normalizing_counts = np.zeros((num_annotators, self.num_classes))
		for i in xrange(self.trainset.num_instances):
			annotator = self.trainset.annotators[i]
			annotator_label = self.trainset.targets[i][0]
			pred_label = y_k[i]
			if annotator_label == pred_label:
				pred_accuracies[annotator,pred_label] += 1.0
			normalizing_counts[annotator,pred_label] += 1.0
		pred_accuracies /= normalizing_counts
		#print "pred_accuracies:", pred_accuracies

		true_accuracies *= 100.0
		pred_accuracies *= 100.0
		diff = true_accuracies - pred_accuracies
		diff = diff*diff
		rmse = np.sqrt(np.ma.masked_invalid(diff).mean())
		#print "rmse:", rmse

		return rmse