#!/bin/bash
cd LabelMe_norm
python ../../run_LabelMe.py -M 3 -p ../../LabelMe > result.txt
cd ..
