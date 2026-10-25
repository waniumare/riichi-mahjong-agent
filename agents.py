"""Agents: RuleOpp / PushAll (baselines), ThresholdAgent (rule-based push/fold), MCTSAgent (determinized UCB search
over discards with a learned opponent model)."""
import math, pickle, random
import numpy as np
from numba import njit
from engine import (TH, eval_discards, shanten, waits_of, feat_key, turn_bin, hand_han, ron_value,
                    tile_class)

# ------------------------------------------------------------------ opponent model
class DangerModel:
    """Learned from simulated logs.
       q(key)  : P(tile is a winning tile of a riichi opponent | tile not genbutsu, features)
       tp[bin] : P(non-riichi opponent is (dama-)tenpai | discards-so-far bin)
       plus scalar stats used by the search (ukeire by shanten, push deal-in rate by shanten, ron loss, waits dist)"""
    def __init__(self):
        self.q_cnt = {}
        self.tp_cnt = {}
        self.stats = {}
        self.prior = 0.05

    def fit(self, danger, tenpai):
        coarse = {}
        for key, hit, _s in danger:
            k = self.q_cnt.setdefault(key, [0, 0]); k[0] += hit; k[1] += 1
            c = coarse.setdefault(key[:2], [0, 0]); c[0] += hit; c[1] += 1
        self.coarse = coarse
        for b, t in tenpai:
            k = self.tp_cnt.setdefault(b, [0, 0]); k[0] += t; k[1] += 1

    def q(self, key):
        if getattr(self, 'flat', False):
            return self.flat_q
        c = self.coarse.get(key[:2], [0, 1])
        pc = (c[0] + 1) / (c[1] + 20)
        k = self.q_cnt.get(key, [0, 0])
        return (k[0] + 30 * pc) / (k[1] + 30)

    def set_flat(self):
        self.flat = True
        self.flat_q = sum(v[0] for v in self.q_cnt.values()) / sum(v[1] for v in self.q_cnt.values())

    def tp(self, b):
        k = self.tp_cnt.get(b, [0, 0])
        return (k[0] + 1) / (k[1] + 30)

    def save(self, fn):
        pickle.dump(self, open(fn, "wb"))

    @staticmethod
    def load(fn):
        return pickle.load(open(fn, "rb"))


def tile_danger(g, p, t, q, model):
    """P(tile t is a winning tile of opponent q) from viewer p's perspective, incl. tenpai uncertainty."""
    if g.riichi[q] >= 0:
        if g.safe[q][t]:
            return 0.0
        return model.q(feat_key(t, g.vis[t] + g.h[p][t], g.disc_cnt[q], g.riichi[q]))
    if g.disc_cnt[q][t] > 0:
        return 0.0
    pi = model.tp(turn_bin(len(g.disc[q])))
    return pi * model.q(feat_key(t, g.vis[t] + g.h[p][t], g.disc_cnt[q], 8))


def combined_danger(g, p, t, model):
    s = 1.0
    for q in range(4):
        if q != p:
            s *= 1 - tile_danger(g, p, t, q, model)
    return 1 - s


# ------------------------------------------------------------------ helpers
def candidates(g, p):
    h = g.h[p].astype(np.int8).copy()
    rem = g.rem_for(p)
    sh, uk = eval_discards(h, rem)
    mn = int(sh.min())
    return h, sh, uk, mn


def riichi_choice(g, p, style, s_after):
    """returns declare flag when closed tenpai after discard."""
    if s_after != 0 or g.wall_left() < 4:
        return False
    return True


def yaku_all_waits(g, p, d):
    """Would a dama win be possible on every live wait after discarding d? (hand 14 -> 13 by d)"""
    h = g.h[p].copy(); h[d] -= 1
    w = waits_of(h.copy())
    ok = True; any_w = False
    for t in np.nonzero(w)[0]:
        any_w = True
        c = h.copy(); c[t] += 1
        if hand_han(c, False, False, g.seat_wind[p], np.zeros(34, np.int64)) == 0:
            ok = False
    return ok and any_w


def decide_riichi(g, p, d, s_after, style):
    if s_after != 0 or g.wall_left() < 4:
        return False
    if style == "D" and yaku_all_waits(g, p, d):
        return False
    return True


class RuleOpp:
    """Efficiency-driven bot with optional naive defense: when someone is in riichi and the hand is >=2 shanten
    (after best discard) it discards a genbutsu tile if it has one."""
    def __init__(self, style="R", defense=True):
        self.style = style
        self.defense = defense

    def choose(self, g, p):
        h, sh, uk, mn = candidates(g, p)
        g.last_s = mn
        riichi_opps = g.any_riichi_other(p)
        if self.defense and riichi_opps and mn >= 2:
            safe = [t for t in range(34) if h[t] > 0 and all(g.safe[q][t] for q in riichi_opps)]
            if safe:
                best = min(safe, key=lambda t: (sh[t], -uk[t]))
                # recompute shanten for non-min discards already in sh
                return best, False
        cands = [t for t in range(34) if sh[t] == mn]
        d = max(cands, key=lambda t: (uk[t], TH[t], -tile_class(t) if False else 0, random.random()))
        return d, decide_riichi(g, p, d, mn, self.style)


class PushAll(RuleOpp):
    def __init__(self, style="R"):
        super().__init__(style, defense=False)


# ------------------------------------------------------------------ rule-based threshold agent
class ThresholdAgent:
    """Push/fold by explicit danger thresholds tau[s] on the combined deal-in probability of the efficient discard,
    s = shanten after the efficient discard (s>=2 uses tau[2])."""
    def __init__(self, model, tau=(0.20, 0.10, 0.04), style="R", noise=0.0):
        self.model = model; self.tau = tau; self.style = style; self.noise = noise
        self.log = []
        self.nfold = 0; self.npush = {}; self.last_fold = False; self.last_push_s = None

    def choose(self, g, p):
        h, sh, uk, mn = candidates(g, p)
        g.last_s = mn
        opps = g.any_riichi_other(p)
        cands = [t for t in range(34) if sh[t] == mn]
        self.last_fold = False; self.last_push_s = None
        if not opps:
            d = min(cands, key=lambda t: (-uk[t], combined_danger(g, p, t, self.model)))
            d = max(cands, key=lambda t: (uk[t] - 1000 * combined_danger(g, p, t, self.model)))
            return d, decide_riichi(g, p, d, mn, self.style)
        # best efficient tile = highest ukeire, tie-> safer
        d_eff = max(cands, key=lambda t: (uk[t] - 40 * combined_danger(g, p, t, self.model)))
        dd = combined_danger(g, p, d_eff, self.model)
        if self.noise > 0 and dd > 0:
            dd = min(1.0, dd * math.exp(self.noise * random.gauss(0, 1) - self.noise ** 2 / 2))
        thr = self.tau[min(max(mn, 0), 2)] if mn >= 0 else self.tau[0]
        if dd <= thr:
            self.last_push_s = mn; self.npush[min(mn, 4)] = self.npush.get(min(mn, 4), 0) + 1
            return d_eff, decide_riichi(g, p, d_eff, mn, self.style)
        # fold: minimal danger tile; tie-break by efficiency
        self.last_fold = True; self.nfold += 1
        tiles = [t for t in range(34) if h[t] > 0]
        d = min(tiles, key=lambda t: (round(combined_danger(g, p, t, self.model), 4), sh[t], -uk[t]))
        return d, False


# ------------------------------------------------------------------ MCTS search (numba)
@njit(cache=True)
def _rollout(arm, s_after, u0, bail, d0, act, nw, u_root, u_ot, u_eta, u_my, u_ron, u_dp, dpush, utab, dfold, L, Lt, V, U, T, rho, eta, no, gamma, pten, pnot):
    """One rollout of one arm in one sampled world (all randomness pre-drawn -> common random numbers across arms)."""
    for o in range(no):
        if act[o] and u_root[o] < d0[arm, o]:
            return -L[o]
    s = s_after
    u = u0
    dead = (bail < 0) or (s > bail)
    for k in range(T):
        if u_eta[k] < eta:
            return 0.0
        for o in range(no):
            if act[o] and u_ot[k, o] < nw[o] / U:
                return -Lt[o]
        prog = gamma if dead else 1.0          # a folding hand still progresses at a reduced rate
        if u_my[k] < prog * u / U:
            if s == 0:
                return V
            s -= 1
            u = utab[s]
        elif s == 0 and u_ron[k] < prog * rho * u / U:
            return V
        if s > bail:
            dead = True
        for o in range(no):
            if act[o]:
                dd = dfold if dead else dpush[min(max(s, 0), 4)]
                if u_dp[k, o] < dd:
                    return -L[o]
    # exhaustive draw: noten payments
    if s == 0 and not dead:
        return pten
    return pnot


@njit(cache=True)
def _eval_arms(arm_ids, seed0, W, s_after, u0, bail, d0, pi, nw_dist, dpush, utab, dfold, L, Lt, V, U, T, rho, eta, no, gamma, pten, pnot):
    """Paired (common-random-number) evaluation of arms on W sampled worlds (determinizations of opponents' hidden
    state: who is really threatening, how many waits they have, whether the tile I discard hits)."""
    n = arm_ids.shape[0]
    tot = np.zeros(n)
    tot2 = np.zeros(n)
    act = np.zeros(no, np.bool_)
    nw = np.zeros(no)
    u_root = np.zeros(no)
    u_ot = np.zeros((T, no)); u_dp = np.zeros((T, no))
    u_eta = np.zeros(T); u_my = np.zeros(T); u_ron = np.zeros(T)
    for w in range(W):
        np.random.seed(seed0 + w)
        for o in range(no):
            act[o] = np.random.random() < pi[o]
            nw[o] = nw_dist[np.random.randint(nw_dist.shape[0])]
            u_root[o] = np.random.random()
        for k in range(T):
            u_eta[k] = np.random.random(); u_my[k] = np.random.random(); u_ron[k] = np.random.random()
            for o in range(no):
                u_ot[k, o] = np.random.random(); u_dp[k, o] = np.random.random()
        for j in range(n):
            a = arm_ids[j]
            r = _rollout(a, s_after[a], u0[a], bail[a], d0, act, nw, u_root, u_ot, u_eta, u_my, u_ron, u_dp,
                         dpush, utab, dfold, L, Lt, V, U, T, rho, eta, no, gamma, pten, pnot)
            tot[j] += r
            tot2[j] += r * r
    return tot, tot2


class MCTSAgent:
    def __init__(self, model, n_iter=1500, style="R", c=0.6, k_eff=4, k_safe=3, record=True, W=150, W_final=1200, gamma=0.35, pten=1500.0, pnot=-1250.0):
        self.gamma, self.pten, self.pnot = gamma, pten, pnot
        self.model = model
        self.n_iter = n_iter
        self.W, self.W_final = int(W), int(W_final)
        self.style = style
        self.c = c
        self.k_eff, self.k_safe = k_eff, k_safe
        self.decisions = []
        self.record = record
        st = model.stats
        self.dpush = np.array([st["dpush"][min(s, 4)] for s in range(5)], np.float64)
        self.utab = np.array([st["utab"][s] for s in range(5)], np.float64)
        self.nw = np.array(st["nw"], np.float64)
        self.rho = st["rho"]
        self.eta = st["eta"]
        self.dfold = st["dfold"]
        self.L = (st["L_nd"], st["L_d"])
        self.seed = random.randrange(1 << 30)

    # hand value estimate (points on ron) if I win from the current shape
    def est_value(self, g, p, h14):
        han = 1  # riichi
        dc = g.dora_cnt
        han += int((h14.astype(np.int64) * dc).sum())
        if not any(h14[t] > 0 and TH[t] for t in range(34)):
            han += 1
        for d in (31, 32, 33, g.seat_wind[p], 27):
            if h14[d] >= 3: han += 1
            elif h14[d] == 2: han += 0.4
        han += 0.4  # ura expectation
        return ron_value(int(round(han)), 30, p == g.dealer) + 1000 * (g.sticks + 1)

    def choose(self, g, p):
        h, sh, uk, mn = candidates(g, p)
        g.last_s = mn
        r_opps = g.any_riichi_other(p)
        model = self.model
        tiles = [t for t in range(34) if h[t] > 0]
        dang = {t: combined_danger(g, p, t, model) for t in tiles}
        cands = [t for t in range(34) if sh[t] == mn]
        if not r_opps:
            d = max(cands, key=lambda t: (uk[t] - 1000 * dang[t] - 0.01 * random.random()))
            return d, decide_riichi(g, p, d, mn, self.style)

        # ---- build arms: (tile, bail) -------------------------------------------------
        eff = sorted(tiles, key=lambda t: (sh[t], -uk[t], dang[t]))[: self.k_eff]
        safe = sorted(tiles, key=lambda t: (dang[t], sh[t], -uk[t]))[: self.k_safe]
        arm_tiles = []
        arm_bail = []
        for t in eff:
            for b in (4, 1, -1):
                arm_tiles.append(t); arm_bail.append(b)
        for t in safe:
            if t not in eff:
                for b in (1, -1):
                    arm_tiles.append(t); arm_bail.append(b)
        n_arms = len(arm_tiles)
        # threats: riichi opps (pi=1) + dama-tenpai opps (pi=tp)
        thr = []
        for q in range(4):
            if q == p:
                continue
            pi = 1.0 if g.riichi[q] >= 0 else model.tp(turn_bin(len(g.disc[q])))
            if pi > 0.02:
                thr.append((q, pi))
        no = len(thr)
        d0 = np.zeros((n_arms, no))
        L = np.zeros(no); Lt = np.zeros(no); pis = np.zeros(no)
        for j, (q, pi) in enumerate(thr):
            pis[j] = pi
            L[j] = self.L[1] if q == g.dealer else self.L[0]
            share = (1 / 3) if q == g.dealer else (0.5 if p == g.dealer else 0.25)
            Lt[j] = L[j] * share
            for a, t in enumerate(arm_tiles):
                d0[a, j] = tile_danger(g, p, t, q, model) / pi if g.riichi[q] < 0 else tile_danger(g, p, t, q, model)
        s_after = np.array([sh[t] for t in arm_tiles], np.int64)
        u0 = np.array([uk[t] if sh[t] == mn else self.utab[min(sh[t], 4)] for t in arm_tiles], np.float64)
        bail = np.array(arm_bail, np.int64)
        U = float(max(1, (g.rem_for(p)).sum()))
        T = max(1, g.wall_left() // 4)
        V = float(self.est_value(g, p, h))
        scale = max(V, float(L.max()), 1000.0)
        n_r = len(r_opps)
        eta_eff = max(0.0, self.eta - n_r * float(self.nw.mean()) / U)
        args = (s_after, u0, bail, d0, pis, self.nw, self.dpush, self.utab, self.dfold, L, Lt, V, U, T, self.rho, eta_eff, no, self.gamma, self.pten, self.pnot)
        base = self.seed
        self.seed = (self.seed * 1103515245 + 12345) & 0x7FFFFFFF
        # level-2 bandit: sequential halving over tiles inside each group (push policies / fold policies)
        finalists = []
        for grp in ([a for a in range(n_arms) if arm_bail[a] >= 0], [a for a in range(n_arms) if arm_bail[a] < 0]):
            alive = np.array(grp, np.int64)
            cum = {int(x): 0.0 for x in alive}
            off = 0
            rounds = max(1, int(np.ceil(np.log2(max(2, len(alive))))))
            while True:
                tot, _ = _eval_arms(alive, base + off, self.W, *args)
                off += self.W
                for x, v in zip(alive, tot):
                    cum[int(x)] += v
                if len(alive) == 1:
                    break
                keep = max(1, (len(alive) + 1) // 2)
                alive = np.array(sorted(alive, key=lambda x: -cum[int(x)])[:keep], np.int64)
            finalists.append(int(alive[0]))
        # root decision push vs fold on a fresh shared batch of worlds (paired)
        fin = np.array(finalists, np.int64)
        tot, tot2 = _eval_arms(fin, base + 100000, self.W_final, *args)
        mean = tot / self.W_final
        ev_push, ev_fold = float(mean[0]), float(mean[1])
        a_best = finalists[0] if ev_push >= ev_fold else finalists[1]
        d = arm_tiles[a_best]
        if self.record:
            push_arms = [a for a in range(n_arms) if arm_bail[a] >= 0]
            fold_arms = [a for a in range(n_arms) if arm_bail[a] < 0]
            d_eff = max(cands, key=lambda t: uk[t] - 40 * dang[t])
            self.decisions.append(dict(
                turn=len(g.disc[p]), s=mn, uk=int(uk[d_eff]), n_riichi=len(r_opps), d_eff=dang[d_eff],
                V=V, L=float(L.max()), U=U, T=T,
                ev_push=ev_push, ev_fold=ev_fold,
                chose_fold=arm_bail[a_best] < 0, ev_best=max(ev_push, ev_fold),
                chosen_s=int(s_after[a_best]), chosen_d=dang[d], d_min=min(dang.values()), is_eff=bool(d == d_eff),
                dealer=p == g.dealer, ))
        return d, (decide_riichi(g, p, d, int(s_after[a_best]), self.style) if arm_bail[a_best] >= 0 else False)
