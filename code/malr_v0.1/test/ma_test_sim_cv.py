import sys, os, time
sys.path = [os.getcwd()[0:os.getcwd().rindex("/")]] + sys.path
#print sys.path
import numpy as np
import malr
from malr.core import *
from malr.supervised import *
from malr.supervised.multiple_annotators import *
import sys

#Quick, turn off printing!
class dummyStream:
	''' dummyStream behaves like a stream but does nothing. '''
	def __init__(self): pass
	def write(self,data): pass
	def read(self,data): pass
	def flush(self): pass
	def close(self): pass

if __name__=="__main__":
	num_runs = 30
	prior = 2
	
	dataset_name = sys.argv[1] + "-std"
	
	#sim_method = "flips"
	#sim_method = "noise"
	#sim_method = "model"
	sim_method = sys.argv[2]
	
	flip_probs = [0.1, 0.3, 0.5, 0.6, 0.7]
	model_probs = [0.9, 0.7, 0.4, 0.3, 0.1]
	#annotators_noise = [0.2, 1.0, 3.0, 5.0, 7.0]
	annotators_noise = [0.2, 0.5, 3.0, 4.0, 5.0]
	#annotators_noise = [0.2, 1.0, 3.0, 5.0, 7.0]
	#annotators_noise = [0.2, 0.5, 1.0, 2.0, 3.0]
	#annotators_noise = [0.5, 1.0, 5.0, 6.0, 7.0]
	#annotators_noise = [0.5, 0.5, 5.0, 5.0, 5.0]
	
	num_annotators = 5
	annotations_prob = 0.7
	annotations_prob = float(sys.argv[3])
	annotations_probs = np.ones(num_annotators) * annotations_prob
	
	old_printerators = [sys.stdout,sys.stderr,sys.stdin,sys.__stdout__,sys.__stderr__,sys.__stdin__][:]
	buf = "Time: " + str(time.asctime()) + "\n"
	buf = "Sim method: " + str(sim_method) + "\n"
	if sim_method == "flips":
		output_filename = "results/cv_"+dataset_name+"_"+sim_method+"_p"+str(prior)+"_"+str(flip_probs).replace(" ","")+"_a"+str(annotations_prob)+".txt"
		buf += "Flip probs: " + str(flip_probs) + "\n"
	elif sim_method == "model":
		output_filename = "results/cv_"+dataset_name+"_"+sim_method+"_p"+str(prior)+"_"+str(model_probs).replace(" ","")+"_a"+str(annotations_prob)+".txt"
		buf += "Model probs (pi's): " + str(model_probs) + "\n"
	elif sim_method == "noise":
		output_filename = "results/cv_"+dataset_name+"_"+sim_method+"_p"+str(prior)+"_"+str(annotators_noise).replace(" ","")+"_a"+str(annotations_prob)+".txt"
		buf += "Annotators noise: " + str(annotators_noise) + "\n"
	else:
		raise Exception()
	buf += "Annotation prob: " + str(annotations_prob) + "\n"
	buf += "Num. runs: " + str(num_runs) + "\n"
	buf += "   \t    A.RMSE\t        \t    \t  \tGT \t     \t     \t    \t  \tTEST\n"
	buf += "RUN\tMV\tMVhard\tMA\tRAYKAR\tMV\tMVhard\tMA\tRAYKAR\tMV\tMVhard\tMA1b\tRAYKAR\n"
	print "Output filename:", output_filename
	if sim_method == "flips":
		print "Flip probs: " + str(flip_probs)
	elif sim_method == "model":
		print "Model probs: " + str(model_probs)
	elif sim_method == "noise":
		print "Annotators noise: " + str(annotators_noise)
	print "Annotation prob: " + str(annotations_prob)
	print "Num. runs: " + str(num_runs)
	print "   \t    A.RMSE\t        \t    \t  \tGT \t     \t     \t    \t  \tTEST"
	print "RUN\tMV\tMVhard\tMA\tRAYKAR\tMV\tMVhard\tMA\tRAYKAR\tMV\tMVhard\tMA1b\tRAYKAR"
	
	write_file = True
	
	num_folds = 10
	num_methods = 4
	mv = True
	mv_h = True
	malr = True
	raykar = True
	
	results = np.zeros((num_runs, 3*num_methods))
	avg_annotator_accuracies = np.zeros(num_annotators)
	
	for run in xrange(num_runs):
		# redirect all print deals
		sys.stdout,sys.stderr,sys.stdin,sys.__stdout__,sys.__stderr__,sys.__stdin__=dummyStream(),dummyStream(),dummyStream(),dummyStream(),dummyStream(),dummyStream()
	
		original_dataset = "data/"+dataset_name+".csv"
		dataset = CSVDataset(original_dataset, dtypes=str)
		folds = dataset.cross_validation_split(num_folds)
		groundtruth_files = []
		train_files = []
		test_files = []
		for fold in xrange(num_folds):
			trainset, testset = folds[fold]
			groundtruth_files.append(original_dataset.replace(".csv","") + "-fold"+str(fold)+"-gt.csv")
			train_files.append(original_dataset.replace(".csv","") + "-fold"+str(fold)+"-train.csv")
			test_files.append(original_dataset.replace(".csv","") + "-fold"+str(fold)+"-test.csv")
			trainset.save_csv(groundtruth_files[fold])
			testset.save_csv(test_files[fold])
	
		# simulate multiple annotators
		for fold in xrange(num_folds):
			ma_simulator = MASimulator()
			dataset = CSVDataset(groundtruth_files[fold], id_feature='id')
			if sim_method == "noise":
				artificial_dataset, accuracies = ma_simulator.simulate_annotators_by_lr_less_rep(dataset, annotators_noise, annotations_probs)
			elif sim_method == "flips":
				artificial_dataset, accuracies = ma_simulator.simulate_annotators_by_random_flips_less_rep(dataset, flip_probs, annotations_probs)
				#artificial_dataset, accuracies = ma_simulator.simulate_annotators_by_random_flips_non_rep(dataset, flip_probs) 
			elif sim_method == "model":
				artificial_dataset, accuracies = ma_simulator.simulate_annotators_by_model_less_rep(dataset, model_probs, annotations_probs)
			else:
				raise Exception()
			artificial_dataset.save_csv(train_files[fold])
			avg_annotator_accuracies += accuracies
		
		#buf += "annotators accuracies:" + str(accuracies) + "\n"
		sys.stdout,sys.stderr,sys.stdin,sys.__stdout__,sys.__stderr__,sys.__stdin__=old_printerators
		print "\nRun:", run 
		print "annotators accuracies:" + str(accuracies)
		sys.stdout,sys.stderr,sys.stdin,sys.__stdout__,sys.__stderr__,sys.__stdin__=dummyStream(),dummyStream(),dummyStream(),dummyStream(),dummyStream(),dummyStream()
	
		if mv:
			for fold in xrange(num_folds):
				train = CSVDataset(train_files[fold], id_feature='id', ma_feature='annotator')
				ground_truth = CSVDataset(groundtruth_files[fold], id_feature='id', dataset_template=train)
				test = CSVDataset(test_files[fold], id_feature='id', dataset_template=train)
				lr = MALogisticRegressionMV(train, ground_truth=ground_truth, testset=test, guassian_prior_sigma=prior)
				lr.train()
				results[run,0] += lr.annotators_rmse(ground_truth)
				results[run,0+num_methods] += lr.evaluate(ground_truth)*100
				results[run,0+num_methods*2] += lr.evaluate(test)*100
			results[run,0] /= num_folds
			results[run,0+num_methods] /= num_folds
			results[run,0+num_methods*2] /= num_folds
			sys.stdout,sys.stderr,sys.stdin,sys.__stdout__,sys.__stderr__,sys.__stdin__=old_printerators
			print "MV:\t", results[run,0], results[run,0+num_methods], results[run,0+num_methods*2]
			sys.stdout,sys.stderr,sys.stdin,sys.__stdout__,sys.__stderr__,sys.__stdin__=dummyStream(),dummyStream(),dummyStream(),dummyStream(),dummyStream(),dummyStream()

		if mv_h:
			for fold in xrange(num_folds):
				train = CSVDataset(train_files[fold], id_feature='id', ma_feature='annotator')
				ground_truth = CSVDataset(groundtruth_files[fold], id_feature='id', dataset_template=train)
				test = CSVDataset(test_files[fold], id_feature='id', dataset_template=train)
				lr = MALogisticRegressionMVHard(train, ground_truth=ground_truth, testset=test, guassian_prior_sigma=prior)
				lr.train()
				results[run,1] += lr.annotators_rmse(ground_truth)
				results[run,1+num_methods] += lr.evaluate(ground_truth)*100
				results[run,1+num_methods*2] += lr.evaluate(test)*100
			results[run,1] /= num_folds
			results[run,1+num_methods] /= num_folds
			results[run,1+num_methods*2] /= num_folds
			sys.stdout,sys.stderr,sys.stdin,sys.__stdout__,sys.__stderr__,sys.__stdin__=old_printerators
			print "MVhard:\t", results[run,1], results[run,1+num_methods], results[run,1+num_methods*2]
			sys.stdout,sys.stderr,sys.stdin,sys.__stdout__,sys.__stderr__,sys.__stdin__=dummyStream(),dummyStream(),dummyStream(),dummyStream(),dummyStream(),dummyStream()

		if malr:
			for fold in xrange(num_folds):
				train = CSVDataset(train_files[fold], id_feature='id', ma_feature='annotator')
				ground_truth = CSVDataset(groundtruth_files[fold], id_feature='id', dataset_template=train)
				test = CSVDataset(test_files[fold], id_feature='id', dataset_template=train)
				lr = MALogisticRegression(train, ground_truth=ground_truth, testset=test, guassian_prior_sigma=prior)
				lr.train()
				results[run,2] += lr.annotators_rmse(ground_truth)
				results[run,2+num_methods] += lr.evaluate(ground_truth)*100
				results[run,2+num_methods*2] += lr.evaluate(test)*100
			results[run,2] /= num_folds
			results[run,2+num_methods] /= num_folds
			results[run,2+num_methods*2] /= num_folds
			sys.stdout,sys.stderr,sys.stdin,sys.__stdout__,sys.__stderr__,sys.__stdin__=old_printerators
			print "MA:\t", results[run,2], results[run,2+num_methods], results[run,2+num_methods*2]
			sys.stdout,sys.stderr,sys.stdin,sys.__stdout__,sys.__stderr__,sys.__stdin__=dummyStream(),dummyStream(),dummyStream(),dummyStream(),dummyStream(),dummyStream()
			
		if raykar:
			for fold in xrange(num_folds):
				train = CSVDataset(train_files[fold], id_feature='id', ma_feature='annotator')
				ground_truth = CSVDataset(groundtruth_files[fold], id_feature='id', dataset_template=train)
				test = CSVDataset(test_files[fold], id_feature='id', dataset_template=train)
				lr = MALogisticRegressionRaykar(train, ground_truth=ground_truth, testset=test, guassian_prior_sigma=prior)
				lr.train()
				results[run,3] += lr.annotators_rmse(ground_truth)
				results[run,3+num_methods] += lr.evaluate(ground_truth)*100
				results[run,3+num_methods*2] += lr.evaluate(test)*100
			results[run,3] /= num_folds
			results[run,3+num_methods] /= num_folds
			results[run,3+num_methods*2] /= num_folds
			sys.stdout,sys.stderr,sys.stdin,sys.__stdout__,sys.__stderr__,sys.__stdin__=old_printerators
			print "RA:\t", results[run,3], results[run,3+num_methods], results[run,3+num_methods*2]
			sys.stdout,sys.stderr,sys.stdin,sys.__stdout__,sys.__stderr__,sys.__stdin__=dummyStream(),dummyStream(),dummyStream(),dummyStream(),dummyStream(),dummyStream()
		
		#Turn printing back on!
		sys.stdout,sys.stderr,sys.stdin,sys.__stdout__,sys.__stderr__,sys.__stdin__=old_printerators
	
		results_line = str(run+1)
		for i in xrange(num_methods*3):
			results_line += "\t%.2f" % (results[run,i])
		print results_line
		buf += results_line + "\n"
	
	avg_annotator_accuracies /= num_runs
	line = ""
	for i in xrange(num_annotators):
		line += str(avg_annotator_accuracies[i]) + "\t"
	print "Average annotator accuracies: " + line
	buf += "\nAverage annotator accuracies: " + line + "\n"
	
	average = results.mean(axis=0)
	results_line = "AVG"
	for i in xrange(num_methods*3):
		results_line += "\t%.2f" % (average[i])
	print results_line
	buf += results_line + "\n"
	stddev = results.std(axis=0)
	results_line = "STD"
	for i in xrange(num_methods*3):
		results_line += "\t%.2f" % (stddev[i])
	print results_line
	buf += results_line + "\n"
	print 
	
	print "Writting file:", output_filename
	if write_file:
		fw = open(output_filename, "w")
		fw.write(buf + "\n")
		fw.flush()
		fw.close()
	
