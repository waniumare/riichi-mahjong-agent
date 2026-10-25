"""Held-out reliability check of the danger model (fresh seeds 2,000,000+)."""
import random, pickle, numpy as np
from engine import *
from agents import *
from calibrate import LoggingOpp
m = DangerModel.load("model.pkl")
ev = []
for s in range(2_000_000, 2_006_000):
    rnd = random.Random(s)
    ags = [LoggingOpp(style=rnd.choice("RD")) for _ in range(4)]
    random.seed(s)
    g = Game(s, ags, log={"danger": [], "tenpai": [], "uk": []})
    g.play()
    ev += [(m.q(k), h) for k, h, _ in g.log["danger"]]
    tp = g.log["tenpai"]
pred = np.array([e[0] for e in ev]); hit = np.array([e[1] for e in ev], float)
# flat baseline
flat = hit.mean()
edges = [0, .01, .03, .05, .08, .12, .2, 1]
rows = []
for lo, hi in zip(edges[:-1], edges[1:]):
    mk = (pred >= lo) & (pred < hi)
    if mk.sum():
        rows.append((lo, hi, int(mk.sum()), float(pred[mk].mean()), float(hit[mk].mean())))
brier = float(np.mean((pred - hit) ** 2)); brier_flat = float(np.mean((flat - hit) ** 2))
# AUC
from itertools import islice
order = np.argsort(pred); r = np.empty(len(pred)); r[order] = np.arange(len(pred))
npos = hit.sum(); nneg = len(hit) - npos
auc = (r[hit == 1].sum() - npos * (npos - 1) / 2) / (npos * nneg)
pickle.dump(dict(rows=rows, brier=brier, brier_flat=brier_flat, auc=float(auc), n=len(pred), base=float(flat)), open("results/heldout_calib.pkl", "wb"))
print("n", len(pred), "base rate", round(flat, 4), "brier", round(brier, 5), "flat", round(brier_flat, 5), "AUC", round(auc, 3))
for r_ in rows: print([round(x, 3) if isinstance(x, float) else x for x in r_])
