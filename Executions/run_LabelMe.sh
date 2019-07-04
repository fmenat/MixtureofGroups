#!/bin/bash
cd LabelMe
python ../../run_LabelMe.py -M 3 -p ../../LabelMe -v 1 -e raykar/ds/oursglobal > result.txt
python ../../run_LabelMe.py -M 2 -p ../../LabelMe -v 1 -e mv/oursindividual > result.txt
cd ..
cd LabelMe_v2
python ../../run_LabelMe.py -M 3 -p ../../LabelMe -v 2 -e raykar/ds/oursglobal > result.txt
python ../../run_LabelMe.py -M 3 -p ../../LabelMe -v 2 -e  mv/oursindividual > result.txt
cd ..
