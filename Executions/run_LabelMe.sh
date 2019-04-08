#!/bin/bash
cd LabelMe_noflatten
python ../../run_LabelMe.py -M 3 -p ../../LabelMe -v 1 > result.txt
cd ..
cd LabelMe_v2_noflatten
python ../../run_LabelMe.py -M 3 -p ../../LabelMe -v 2 > result.txt
cd ..
cd LabelMe_v3_noflatten
python ../../run_LabelMe.py -M 3 -p ../../LabelMe -v 3 > result.txt
cd ..
