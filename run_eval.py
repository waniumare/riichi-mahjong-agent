"""Evaluate an agent in seat 0 vs three RuleOpp bots (styles random per seed). Paired seeds across agents.
usage: python run_eval.py NAME N OUT [key=val ...]    NAME in pushall|ruleopp|thresh|mcts"""
import sys, random, pickle, time
import numpy as np
from multiprocessing import Pool
from engine import *
from agents import *

def make_agent(name, kw, model):
    if name == "pushall": return PushAll()
    if name == "ruleopp": return RuleOpp()
    if name == "thresh":
        return ThresholdAgent(model, tau=(kw.get("t0", .2), kw.get("t1", .1), kw.get("t2", .04)), noise=kw.get("noise", 0.0))
    if name == "mcts":
        return MCTSAgent(model, W=int(kw.get("W", 150)), W_final=int(kw.get("Wf", 1200)), gamma=kw.get("gamma", 0.35), pten=kw.get("pten", 1500.0), pnot=kw.get("pnot", -1250.0))
    raise ValueError(name)

def work(args):
    name, kw, seeds = args
    model = DangerModel.load("model.pkl")
    model.stats["dfold"] = 0.005
    if kw.get("flat"): model.set_flat()
    out = []
    for s in seeds:
        rnd = random.Random(s)
        opp = [RuleOpp(style=rnd.choice("RD")) for _ in range(3)]
        random.seed(s)
        me = make_agent(name, kw, model)
        random.seed(s + 7)
        g = Game(s, [me] + opp)
        r = g.play()
        rec = dict(seed=s, delta=int(g.delta[0]), kind=r["kind"], win=r["winner"] == 0,
                   deal=r["kind"] == "ron" and r["loser"] == 0, dealer=g.dealer == 0,
                   riichi=g.riichi[0] >= 0, opp_riichi=sum(x >= 0 for x in g.riichi[1:]),
                   value=r["value"] if r["winner"] == 0 else 0, nd=g.nd,
                   tenpai=(r.get("tenpai") or [False] * 4)[0])
        if name == "thresh":
            rec.update(nfold=me.nfold, last_fold=me.last_fold, npush=me.npush, last_push_s=me.last_push_s)
        if name == "mcts":
            rec["decisions"] = me.decisions
        out.append(rec)
    return out

if __name__ == "__main__":
    name, N, outfn = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    kw = {k: float(v) for k, v in (a.split("=") for a in sys.argv[4:])}
    seed0 = int(kw.pop("seed0", 0))
    seeds = list(range(seed0, seed0 + N))
    chunks = [(name, kw, seeds[i::40]) for i in range(40)]
    t0 = time.time()
    with Pool(2) as pool:
        parts = pool.map(work, chunks)
    recs = sorted([r for p in parts for r in p], key=lambda r: r["seed"])
    pickle.dump(recs, open(outfn, "wb"))
    d = np.array([r["delta"] for r in recs])
    print(f"{name} {kw} N={N}  mean delta {d.mean():8.1f} ± {d.std()/np.sqrt(N):6.1f}  win {np.mean([r['win'] for r in recs]):.3f} "
          f"deal-in {np.mean([r['deal'] for r in recs]):.3f}  time {time.time()-t0:.0f}s")
