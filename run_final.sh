#!/bin/bash
cd "$(dirname "$0"
P="python3 -W ignore run_eval.py"
$P mcts 8000 results/tune2_g0.35.pkl seed0=100000 gamma=0.35
$P mcts 40000 results/final_mcts.pkl
$P thresh 40000 results/final_thresh_tuned.pkl t0=0.3 t1=0.15 t2=0.03
$P mcts 20000 results/abl_mcts_flat.pkl flat=1
echo FINALDONE
