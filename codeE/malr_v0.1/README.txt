Requirements: numpy, scipy

For running an experiment with simulated annotators, run:
python ma_test_sim_cv.py <dataset_csv_file> <sim_method (flips or noise)> <plabel (e.g. 0.7)>
The annotators are simulated and different methods are applied to the generated data.

For running an experiment with real data, run:
python ma_test_real.py <ground_truth_data> <multiple_annotator_data> <test_data> 
Example:
python ma_test_real.py polarity_data_gold.csv polarity_data_mturk.csv polarity_data_test.csv 