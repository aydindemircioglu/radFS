#!/bin/bash

# clear
rm results/*.png
rm paper/*

# eval
python3 ./evaluate_training_time.py
python3 ./evaluate_stability.py
python3 ./evaluate_ranking.py
python3 ./evaluate_ranking_by_type.py
python3 ./evaluate_predictive_performance.py
python3 ./evaluate_tradeoffs.py


#
