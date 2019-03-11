#!/bin/bash
cd simple_scenario1
python ../../run_simple.py -M 3 -p ../../ -s 1 > result_scenario1.txt
cd ..
cd simple_scenario2
python ../../run_simple.py -M 3 -p ../../ -s 2 > result_scenario2.txt
cd ..
cd simple_scenario3
python ../../run_simple.py -M 3 -p ../../ -s 3 > result_scenario3.txt
cd ..
cd simple_scenario4
python ../../run_simple.py -M 3 -p ../../ -s 4 > result_scenario4.txt
cd ..
cd simple_scenario5
python ../../run_simple.py -M 3 -p ../../ -s 5 > result_scenario5.txt
cd ..
cd simple_scenario6
python ../../run_simple.py -M 3 -p ../../ -s 6 > result_scenario6.txt
cd ..
cd simple_scenario7
python ../../run_simple.py -M 3 -p ../../ -s 7 > result_scenario7.txt
cd ..


