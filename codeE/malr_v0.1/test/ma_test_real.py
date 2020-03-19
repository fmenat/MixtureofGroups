import sys, os, time
sys.path = [os.getcwd()[0:os.getcwd().rindex("/")]] + sys.path
#print sys.path
import numpy as np
from matplotlib import pyplot as plt
import malr
from malr.core import *
from malr.supervised import *
from malr.supervised.multiple_annotators import *

#Quick, turn off printing!
class dummyStream:
	''' dummyStream behaves like a stream but does nothing. '''
	def __init__(self): pass
	def write(self,data): pass
	def read(self,data): pass
	def flush(self): pass
	def close(self): pass

if __name__=="__main__":
	
	#ground_truth_file = 'data/mturk_polarity_gold_lsa_topics.csv'
	#test_file = 'data/mturk_polarity_test_lsa_topics.csv'
	#train_file = 'data/mturk_polarity_workers_lsa_topics.csv'
	
	ground_truth_file = sys.argv[1]
	train_file = sys.argv[2]
	test_file = sys.argv[3]
	
	old_printerators = [sys.stdout,sys.stderr,sys.stdin,sys.__stdout__,sys.__stderr__,sys.__stdin__][:]
	output_filename = train_file.replace("data/","results/").replace(".csv"," "+time.asctime()+".txt")
	buf = "Time: " + str(time.asctime()) + "\n"
	buf += "   \t    \t  \tA.RMSE\t  \t   \t     \t      \t    \t  \tGT \t     \t   \t     \t      \t    \t  \tTEST\n"
	buf += "RUN\tMV\tMVhard\tMA\tRAYKAR\tMV\tMVhard\tMA\tRAYKAR\tMV\tMVhard\tMA\tRAYKAR\n"
	print "Output filename:", output_filename
	
	write_file = True
	if write_file:
		fw = open(output_filename, "w")
	
	prior = 2
	num_methods = 4
	mv = True
	mv_h = True
	malr = True
	malrr = True
	
	debug = True
	
	results = np.zeros(3*num_methods)
	
	true_accuracies = None
	pred_accuracies_mv = None
	pred_accuracies_mv_h = None
	pred_accuracies_ma = None
	pred_accuracies_raykar = None
	
	# redirect all print deals
	if not debug:
		sys.stdout,sys.stderr,sys.stdin,sys.__stdout__,sys.__stderr__,sys.__stdin__=dummyStream(),dummyStream(),dummyStream(),dummyStream(),dummyStream(),dummyStream()
	
	if mv:
		train = CSVDataset(train_file, id_feature='id', ma_feature='annotator')
		ground_truth = CSVDataset(ground_truth_file, id_feature='id', dataset_template=train)
		test = CSVDataset(test_file, id_feature='id', dataset_template=train)
		lr = MALogisticRegressionMV(train, ground_truth=ground_truth, testset=test, guassian_prior_sigma=prior)
		lr.train()
		results[0], pred_accuracies_mv, true_accuracies = lr.annotators_rmse(ground_truth)
		results[0+num_methods] = lr.evaluate(ground_truth)*100
		results[0+num_methods*2] = lr.evaluate(test)*100
		sys.stdout,sys.stderr,sys.stdin,sys.__stdout__,sys.__stderr__,sys.__stdin__=old_printerators
		print "MV:\t", results[0], results[0+num_methods], results[0+num_methods*2]
		if write_file:
			fw.write("MV:\t"+str(results[0])+"\t"+str(results[0+num_methods])+"\t"+str(results[0+num_methods*2])+"\n")
			fw.flush()
		if not debug:
			sys.stdout,sys.stderr,sys.stdin,sys.__stdout__,sys.__stderr__,sys.__stdin__=dummyStream(),dummyStream(),dummyStream(),dummyStream(),dummyStream(),dummyStream()
	
	if mv_h:
		train = CSVDataset(train_file, id_feature='id', ma_feature='annotator')
		ground_truth = CSVDataset(ground_truth_file, id_feature='id', dataset_template=train)
		test = CSVDataset(test_file, id_feature='id', dataset_template=train)
		lr = MALogisticRegressionMVHard(train, ground_truth=ground_truth, testset=test, guassian_prior_sigma=prior)
		lr.train()
		results[1], pred_accuracies_mv_h, true_accuracies = lr.annotators_rmse(ground_truth)
		results[1+num_methods] = lr.evaluate(ground_truth)*100
		results[1+num_methods*2] = lr.evaluate(test)*100
		sys.stdout,sys.stderr,sys.stdin,sys.__stdout__,sys.__stderr__,sys.__stdin__=old_printerators
		print "MVhard:\t", results[1], results[1+num_methods], results[1+num_methods*2]
		if write_file:
			fw.write("MVhard:\t"+str(results[1])+"\t"+str(results[1+num_methods])+"\t"+str(results[1+num_methods*2])+"\n")
			fw.flush()
		if not debug:
			sys.stdout,sys.stderr,sys.stdin,sys.__stdout__,sys.__stderr__,sys.__stdin__=dummyStream(),dummyStream(),dummyStream(),dummyStream(),dummyStream(),dummyStream()

	if malr:
		train = CSVDataset(train_file, id_feature='id', ma_feature='annotator')
		ground_truth = CSVDataset(ground_truth_file, id_feature='id', dataset_template=train)
		test = CSVDataset(test_file, id_feature='id', dataset_template=train)
		lr = MALogisticRegression(train, ground_truth=ground_truth, testset=test, guassian_prior_sigma=prior)
		lr.train()
		results[2], pred_accuracies_ma, true_accuracies = lr.annotators_rmse(ground_truth)
		results[2+num_methods] = lr.evaluate(ground_truth)*100
		results[2+num_methods*2] = lr.evaluate(test)*100
		sys.stdout,sys.stderr,sys.stdin,sys.__stdout__,sys.__stderr__,sys.__stdin__=old_printerators
		print "MA:\t", results[2], results[2+num_methods], results[2+num_methods*2]
		if write_file:
			fw.write("MA:\t"+str(results[2])+"\t"+str(results[2+num_methods])+"\t"+str(results[2+num_methods*2])+"\n")
			fw.flush()
		if not debug:
			sys.stdout,sys.stderr,sys.stdin,sys.__stdout__,sys.__stderr__,sys.__stdin__=dummyStream(),dummyStream(),dummyStream(),dummyStream(),dummyStream(),dummyStream()
		
	if malrr:
		train = CSVDataset(train_file, id_feature='id', ma_feature='annotator')
		ground_truth = CSVDataset(ground_truth_file, id_feature='id', dataset_template=train)
		test = CSVDataset(test_file, id_feature='id', dataset_template=train)
		lr = MALogisticRegressionRaykar(train, ground_truth=ground_truth, testset=test, guassian_prior_sigma=prior)
		lr.train()
		results[3], pred_accuracies_raykar, true_accuracies = lr.annotators_rmse(ground_truth)
		results[3+num_methods] = lr.evaluate(ground_truth)*100
		results[3+num_methods*2] = lr.evaluate(test)*100
		sys.stdout,sys.stderr,sys.stdin,sys.__stdout__,sys.__stderr__,sys.__stdin__=old_printerators
		print "RA:\t", results[3], results[3+num_methods], results[3+num_methods*2]
		if write_file:
			fw.write("RA:\t"+str(results[3])+"\t"+str(results[3+num_methods])+"\t"+str(results[3+num_methods*2])+"\n")
			fw.flush()
		if not debug:
			sys.stdout,sys.stderr,sys.stdin,sys.__stdout__,sys.__stderr__,sys.__stdin__=dummyStream(),dummyStream(),dummyStream(),dummyStream(),dummyStream(),dummyStream()

	#Turn printing back on!
	sys.stdout,sys.stderr,sys.stdin,sys.__stdout__,sys.__stderr__,sys.__stdin__=old_printerators

	l = zip(true_accuracies, pred_accuracies_mv, pred_accuracies_mv_h, pred_accuracies_ma, pred_accuracies_raykar)
	l = sorted(l, reverse=True)
	true_accuracies = np.array(zip(*l)[0])
	pred_accuracies_mv = np.array(zip(*l)[1])
	pred_accuracies_mv_h = np.array(zip(*l)[2])
	pred_accuracies_ma = np.array(zip(*l)[3])
	pred_accuracies_raykar = np.array(zip(*l)[4])
	print true_accuracies
	print pred_accuracies_mv
	print pred_accuracies_mv_h
	print pred_accuracies_ma
	print pred_accuracies_raykar
	plt.plot(range(len(true_accuracies)), true_accuracies, 'k-')
	#plt.plot(range(len(true_accuracies)), pred_accuracies_mv, 'r--')
	#plt.plot(range(len(true_accuracies)), pred_accuracies_mv_h, 'r-.')
	plt.plot(range(len(true_accuracies)), pred_accuracies_ma, 'bx')
	plt.plot(range(len(true_accuracies)), pred_accuracies_raykar, 'g+')
	plt.xlabel("annotator")
	plt.ylabel("accuracy")
	plt.show()
	print
	print "MV mean absolute error=", np.mean(np.abs(true_accuracies-pred_accuracies_mv))
	print "MV_h mean absolute error=", np.mean(np.abs(true_accuracies-pred_accuracies_mv_h))
	print "MA mean absolute error=", np.mean(np.abs(true_accuracies-pred_accuracies_ma))
	print "RA mean absolute error=", np.mean(np.abs(true_accuracies-pred_accuracies_raykar))
	print
	print "MV mediam absolute error=", np.median(np.abs(true_accuracies-pred_accuracies_mv))
	print "MV_h mediam absolute error=", np.median(np.abs(true_accuracies-pred_accuracies_mv_h))
	print "MA mediam absolute error=", np.median(np.abs(true_accuracies-pred_accuracies_ma))
	print "RA mediam absolute error=", np.median(np.abs(true_accuracies-pred_accuracies_raykar))
	print

	results_line = ""
	for i in xrange(num_methods*3):
		results_line += "\t%.2f" % (results[i])
	print results_line
	buf += results_line + "\n"
	
	if write_file:
		fw.write("\n\n")
		fw.write(buf + "\n")
		fw.flush()
		fw.close()
	
