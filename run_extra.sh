#!/bin/bash
cd "$(dirname "$0"
while ! grep -q ALLDONE results/run_all.log; do sleep 20; done
P="python3 -W ignore run_eval.py"
for n in 0.25 0.5 1.0 1.5; do $P thresh 20000 results/noise_$n.pkl seed0=100000 noise=$n; done
# re-optimised t1 under high noise (sweep)
for v in 0 0.02 0.05 0.15; do $P thresh 20000 results/noise1_t1_$v.pkl seed0=100000 noise=1.0 t1=$v; done
echo EXTRADONE
