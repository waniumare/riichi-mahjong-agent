#!/bin/bash
cd "$(dirname "$0"
P="python3 -W ignore run_eval.py"
# A: final evaluation, fresh seeds 0..39999
for a in pushall ruleopp; do $P $a 40000 results/final_$a.pkl; done
$P thresh 40000 results/final_thresh.pkl
$P mcts 40000 results/final_mcts.pkl
# C: opponent-model ablations (same fresh seeds, 20000)
$P thresh 20000 results/abl_thresh_flat.pkl flat=1
$P mcts 20000 results/abl_mcts_flat.pkl flat=1
# B: threshold sweeps on tuning seeds
for v in 0 0.05 0.1 0.2 0.3 0.5 1.0; do $P thresh 20000 results/sw_t0_$v.pkl seed0=100000 t0=$v; done
for v in 0 0.02 0.05 0.1 0.15 0.25 1.0; do $P thresh 20000 results/sw_t1_$v.pkl seed0=100000 t1=$v; done
for v in 0 0.01 0.02 0.04 0.08 0.15 1.0; do $P thresh 20000 results/sw_t2_$v.pkl seed0=100000 t2=$v; done
echo ALLDONE
