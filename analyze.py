import pickle, glob, os, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results/"
ld = lambda f: pickle.load(open(R + f, "rb"))
C = dict(pushall="#8c8c8c", ruleopp="#4c78a8", thresh="#f58518", thresh_tuned="#e45756", mcts_v1="#b6d7a8", mcts="#54a24b")
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})
out = {}

def arr(rs, k): return np.array([r[k] for r in rs], float)

# ---------------- 1. agent comparison (40k fresh seeds)
names = ["pushall", "ruleopp", "thresh", "thresh_tuned", "mcts_v1", "mcts"]
F = {n: ld(f"final_{n}.pkl") for n in names}
ref = arr(F["ruleopp"], "delta")
rows = []
for n in names:
    d = arr(F[n], "delta"); diff = d - ref; N = len(d)
    opp = arr(F[n], "opp_riichi") > 0
    rows.append(dict(agent=n, N=N, mean=d.mean(), se=d.std() / np.sqrt(N),
                     paired=diff.mean(), paired_se=diff.std() / np.sqrt(N),
                     win=arr(F[n], "win").mean(), deal=arr(F[n], "deal").mean(),
                     riichi=arr(F[n], "riichi").mean(),
                     avg_win=arr(F[n], "value")[arr(F[n], "win") == 1].mean(),
                     contested_mean=d[opp].mean(), contested_se=d[opp].std() / np.sqrt(opp.sum()),
                     uncontested_mean=d[~opp].mean(), contested_share=opp.mean()))
out["compare"] = rows
for r in rows:
    print(f"{r['agent']:8s} mean {r['mean']:7.1f}±{r['se']:5.1f} paired-vs-ruleopp {r['paired']:7.1f}±{r['paired_se']:5.1f}  "
          f"win {r['win']:.3f} deal {r['deal']:.3f} avgwin {r['avg_win']:.0f} | contested {r['contested_mean']:7.1f} uncontested {r['uncontested_mean']:7.1f}")
fig, ax = plt.subplots(figsize=(7.2, 3.5))
x = np.arange(len(names))
ax.bar(x, [r["paired"] for r in rows], yerr=[1.96 * r["paired_se"] for r in rows], color=[C[n] for n in names], capsize=4)
ax.axhline(0, color="k", lw=.7)
ax.set_xticks(x); ax.set_xticklabels(["PushAll", "RuleOpp\n(reference)", "Threshold\n(default)", "Threshold\n(tuned)", "MC search\nv1", "MC search\nv2"], fontsize=8)
ax.set_ylabel("points / hand vs RuleOpp (paired)"); ax.set_title("40,000 hands per agent, identical deals (95% CI)")
plt.tight_layout(); plt.savefig(R + "fig1_agents.png"); plt.close()

# ---------------- 2. threshold sweeps
def sweep(prefix, vals):
    pts = []
    for v in vals:
        f = f"{R}sw_{prefix}_{v}.pkl"
        if os.path.exists(f):
            d = arr(pickle.load(open(f, "rb")), "delta"); pts.append((float(v), d.mean(), d.std() / np.sqrt(len(d))))
    return pts
SW = dict(t0=sweep("t0", ["0", "0.05", "0.1", "0.2", "0.3", "0.5", "1.0"]),
          t1=sweep("t1", ["0", "0.02", "0.05", "0.1", "0.15", "0.25", "1.0"]),
          t2=sweep("t2", ["0", "0.01", "0.02", "0.04", "0.08", "0.15", "1.0"]))
out["sweeps"] = SW
fig, axs = plt.subplots(1, 3, figsize=(9.5, 3.1), sharey=True)
for ax, (k, lab) in zip(axs, [("t0", "tenpai (s=0)"), ("t1", "1-shanten"), ("t2", "2+-shanten")]):
    p = np.array(SW[k])
    if len(p):
        xs = np.arange(len(p))
        ax.errorbar(xs, p[:, 1], yerr=1.96 * p[:, 2], marker="o", color=C["thresh"], capsize=3)
        ax.set_xlabel(f"push threshold τ on P(deal-in), {lab}")
        ax.set_xticks(xs); ax.set_xticklabels([("never fold" if v >= 1 else f"{v:g}") for v in p[:, 0]], fontsize=7, rotation=45)
axs[0].set_ylabel("mean points / hand")
plt.tight_layout(); plt.savefig(R + "fig2_sweeps.png"); plt.close()
for k in SW:
    print(k, [(round(a, 2), round(b, 1)) for a, b, c in SW[k]])

# ---------------- 3. MCTS decision thresholds (what the search actually does)
D = [d for r in F["mcts"] for d in r["decisions"]]
out["n_decisions"] = len(D)
def logit_fit(X, y, it=40):
    X = np.c_[np.ones(len(X)), X]; b = np.zeros(X.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30))); W = p * (1 - p) + 1e-9
        b += np.linalg.solve(X.T @ (X * W[:, None]) + 1e-6 * np.eye(len(b)), X.T @ (y - p))
    return b
for d in D:
    d["dev"] = d["chosen_s"] > d["s"]            # gave up hand progress (shanten got worse) for safety
    d["swap"] = (not d["is_eff"])                 # discarded something other than the efficient tile
tab = {}
dbins = [(0, .01), (.01, .03), (.03, .06), (.06, .10), (.10, .20), (.20, 1.01)]
Tb = [(1, 3), (3, 6), (6, 10), (10, 40)]
def cell(x, key):
    return (float(np.mean([d[key] for d in x])), len(x)) if len(x) >= 30 else (None, len(x))
for s in (0, 1, 2):
    xs = [d for d in D if d["s"] == s]
    tab[s] = dict(n=len(xs), dev=cell(xs, "dev"), swap=cell(xs, "swap"), fold=cell(xs, "chose_fold"),
                  grid_dev={f"T{lo}-{hi-1}": [cell([d for d in xs if lo <= d["T"] < hi and a_ <= d["d_eff"] < b_], "dev") for a_, b_ in dbins] for lo, hi in Tb},
                  grid_swap={f"T{lo}-{hi-1}": [cell([d for d in xs if lo <= d["T"] < hi and a_ <= d["d_eff"] < b_], "swap") for a_, b_ in dbins] for lo, hi in Tb})
    dd = np.array([d["d_eff"] for d in xs]); y = np.array([d["dev"] for d in xs], float)
    X = np.c_[dd * 10, [d["V"] / d["L"] for d in xs], [d["T"] / 5 for d in xs], [d["uk"] / 10 for d in xs], [d["n_riichi"] for d in xs]]
    if y.sum() > 50:
        b = logit_fit(X, y); tab[s]["logit_dev"] = dict(zip(["const", "danger(+0.1)", "V/L", "turns_left(+5)", "ukeire(+10)", "n_riichi"], map(float, b)))
        b1 = logit_fit(dd[:, None] * 10, y)
        tab[s]["d50_dev"] = float(-b1[0] / b1[1] / 10) if b1[1] > 0 else None
out["mcts_table"] = tab
fig, axs = plt.subplots(1, 2, figsize=(9, 3.3), sharey=True)
for ax, s in zip(axs, (1, 2)):
    xs = [d for d in D if d["s"] == s]
    for (lo, hi), col in zip(Tb, ["#c7e9c0", "#74c476", "#31a354", "#006d2c"]):
        mids, rates = [], []
        for a_, b_ in dbins:
            x = [d for d in xs if lo <= d["T"] < hi and a_ <= d["d_eff"] < b_]
            if len(x) >= 30:
                mids.append(np.mean([d["d_eff"] for d in x])); rates.append(np.mean([d["dev"] for d in x]))
        ax.plot(mids, rates, "o-", color=col, label=f"{lo}-{hi-1} rounds left")
    ax.set_xscale("symlog", linthresh=.01); ax.set_xlabel("P(deal-in) of the most efficient discard")
    ax.set_title(f"{s}-shanten (n={len(xs):,})")
axs[0].set_ylabel("P(search sacrifices hand progress for safety)"); axs[0].legend(fontsize=7, frameon=False)
plt.tight_layout(); plt.savefig(R + "fig3_mcts_threshold.png"); plt.close()
print(json.dumps(tab, indent=1, default=float))

# ---------------- 4. opponent-model ablation
ab = {}
for n in ("thresh", "mcts"):
    full = F[n][:20000]; fl = ld(f"abl_{n}_flat.pkl")
    a = arr(full, "delta"); b = arr(fl, "delta"); diff = a - b
    ab[n] = dict(full=a.mean(), flat=b.mean(), diff=diff.mean(), se=diff.std() / np.sqrt(len(a)),
                 deal_full=arr(full, "deal").mean(), deal_flat=arr(fl, "deal").mean())
out["ablation"] = ab; print(ab)

# ---------------- 5. noise robustness
nz = []
base = arr(ld("final_thresh.pkl"), "delta")
tune_ref = None
for n in ("0.25", "0.5", "1.0", "1.5"):
    f = f"{R}noise_{n}.pkl"
    if os.path.exists(f):
        d = arr(pickle.load(open(f, "rb")), "delta"); nz.append((float(n), d.mean(), d.std() / np.sqrt(len(d))))
t1base = [p for p in SW["t1"] if abs(p[0] - 0.1) < 1e-9]
if t1base: nz.insert(0, (0.0, t1base[0][1], t1base[0][2]))
out["noise"] = nz
nz1 = []
for v in ("0", "0.02", "0.05", "0.15"):
    f = f"{R}noise1_t1_{v}.pkl"
    if os.path.exists(f):
        d = arr(pickle.load(open(f, "rb")), "delta"); nz1.append((float(v), d.mean(), d.std() / np.sqrt(len(d))))
out["noise_t1"] = nz1
if nz:
    fig, ax = plt.subplots(figsize=(5, 3.2)); p = np.array(nz)
    ax.errorbar(p[:, 0], p[:, 1], yerr=1.96 * p[:, 2], marker="o", color=C["thresh"], capsize=3)
    ax.set_xlabel("σ of log-noise on estimated danger"); ax.set_ylabel("mean points / hand"); ax.set_title("Threshold rule under misestimated danger")
    plt.tight_layout(); plt.savefig(R + "fig4_noise.png"); plt.close()
print("noise", nz, nz1)

# ---------------- 6. model calibration (held-out)
H = ld("heldout_calib.pkl"); out["heldout"] = H
fig, ax = plt.subplots(figsize=(3.8, 3.6)); rr = np.array(H["rows"])
ax.plot([0, .15], [0, .15], "k:", lw=1); ax.scatter(rr[:, 3], rr[:, 4], s=rr[:, 2] / 150, color=C["ruleopp"])
ax.set_xlabel("predicted P(tile is a wait)"); ax.set_ylabel("observed frequency"); ax.set_title(f"Held-out reliability (AUC {H['auc']:.2f})")
plt.tight_layout(); plt.savefig(R + "fig5_reliability.png"); plt.close()
json.dump(out, open(R + "analysis.json", "w"), indent=1, default=float)
