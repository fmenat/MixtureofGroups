import os, sys

dataset = sys.argv[1]
sim_method = sys.argv[2]

plabel = 1.0
while plabel >= 0.1:
	print "python ma_test_sim_cv.py", dataset, sim_method, plabel
	os.system("python ma_test_sim_cv.py "+dataset+" "+sim_method+" "+str(plabel))
	plabel -= 0.1
