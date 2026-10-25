"""Fit the opponent/danger model from simulated games (RuleOpp x4, calibration seeds 10^6+)."""
import sys, random, pickle, time, math
import numpy as np
from multiprocessing import Pool
from engine import *
from agents import *

N = int(sys.argv[1]) if len(sys.argv) > 1 else 6000

class LoggingOpp(RuleOpp):
    """RuleOpp that also logs ukeire/shanten stats."""
    def choose(self, g, p):
        h, sh, uk, mn = candidates(g, p)
        g.last_s = mn
        g.log["uk"].append((mn, int(uk[sh == mn].max())))
        return RuleOpp.choose(self, g, p)

def work(seeds):
    out = dict(danger=[], tenpai=[], uk=[], ron=[], riichi=[], nw=[], rfit=[])
    for s in seeds:
        rnd = random.Random(s)
        ags = [LoggingOpp(style=rnd.choice("RD")) for _ in range(4)]
        random.seed(s)
        g = Game(s, ags, log={"danger": [], "tenpai": [], "uk": []})
        r = g.play()
        for k in ("danger", "tenpai", "uk"):
            out[k] += g.log[k]
        if r["kind"] == "ron" and g.riichi[r["winner"]] >= 0:
            out["ron"].append((r["winner"] == g.dealer, r["value"]))
        if r["kind"] == "tsumo" and g.riichi[r["winner"]] >= 0:
            out["ron"].append((r["winner"] == g.dealer, None))
        # riichi players: (rounds left at declaration unknown here) -> use outcome rate in terms of waits
        for q in range(4):
            if g.riichi[q] >= 0:
                n_after = None
        out["riichi"].append((r["kind"], r["winner"], [x for x in g.riichi], g.nd))
        for q in range(4):
            if g.rinfo[q] is not None:
                wl, u, U = g.rinfo[q]
                oc = "win" if r["winner"] == q else ("other" if r["kind"] != "draw" else "draw")
                out["rfit"].append((wl // 4, u, U, oc))
        for q in range(4):
            if g.riichi[q] >= 0 and g.wm[q] is not None:
                live = int((g.rem_for(q) * g.wm[q]).sum())
                out["nw"].append(live)
    return out

if __name__ == "__main__":
    t0 = time.time()
    seeds = list(range(1_000_000, 1_000_000 + N))
    chunks = [seeds[i::24] for i in range(24)]
    with Pool(2) as pool:
        parts = pool.map(work, chunks)
    D = dict(danger=[], tenpai=[], uk=[], ron=[], riichi=[], nw=[], rfit=[])
    for p in parts:
        for k in D:
            D[k] += p[k]
    m = DangerModel()
    m.fit(D["danger"], D["tenpai"])
    # --- scalar stats
    dpush = {}
    for key, hit, s in D["danger"]:
        if s >= 0 and s < 99:
            a = dpush.setdefault(min(s, 4), [0, 0]); a[0] += hit; a[1] += 1
    ukt = {}
    for s, u in D["uk"]:
        a = ukt.setdefault(min(max(s, 0), 4), []); a.append(u)
    rd = [v for d, v in D["ron"] if d and v]; rn = [v for d, v in D["ron"] if (not d) and v]
    # riichi-player win rate per game: used to fit rho later; store outcomes
    nw = [x for x in D["nw"] if x > 0]
    safe_hit = [(h) for k, h, s in D["danger"] if s == -9]
    m.stats = dict(
        dpush={s: dpush[s][0] / max(1, dpush[s][1]) if s in dpush else 0.02 for s in range(5)},
        utab={s: float(np.mean(ukt[s])) if s in ukt else 6.0 for s in range(5)},
        L_d=float(np.mean(rd)) if rd else 9000, L_nd=float(np.mean(rn)) if rn else 6000,
        nw=nw, rho=2.0, eta=0.02, dfold=0.005)
    # --- fit rho (ron multiplier) and eta (others-win hazard) by multinomial MLE on riichi players
    best = (-1e18, 2.0, 0.02)
    R = D["rfit"]
    for rho in np.arange(0.0, 4.01, 0.25):
        for eta in np.arange(0.0, 0.3001, 0.01):
            ll = 0.0
            for T, u, U, oc in R[::3]:
                h = min(0.95, (1 + rho) * u / U)
                stay = (1 - h) * (1 - eta)
                Tn = max(T, 1)
                pw = h * (1 - eta) * (1 - stay ** Tn) / (1 - stay + 1e-12)
                po = eta * (1 - stay ** Tn) / (1 - stay + 1e-12)
                pd = stay ** Tn
                p_ = pw if oc == "win" else (po if oc == "other" else pd)
                ll += math.log(max(p_, 1e-12))
            if ll > best[0]:
                best = (ll, rho, eta)
    m.stats["rho"], m.stats["eta"] = float(best[1]), float(best[2])
    m.stats["riichi_n"] = len(R)
    m.stats["riichi_win_rate"] = float(np.mean([oc == "win" for *_, oc in R]))
    m.stats["dpush_n"] = {s: dpush[s][1] for s in dpush}
    m.save("model.pkl")
    pickle.dump(D["riichi"], open("calib_outcomes.pkl", "wb"))
    print("games", N, "sec", round(time.time() - t0))
    print("danger events", len(D["danger"]), "tenpai obs", len(D["tenpai"]))
    print({k: round(v, 3) for k, v in m.stats["dpush"].items()}, m.stats["dpush_n"])
    print({k: round(v, 1) for k, v in m.stats["utab"].items()})
    print("L nd/d", m.stats["L_nd"], m.stats["L_d"], "mean waits", np.mean(nw))
    print("rho,eta", m.stats["rho"], m.stats["eta"], "riichi win rate", m.stats["riichi_win_rate"])
    for b in range(4):
        print("tp bin", b, round(m.tp(b), 3))
