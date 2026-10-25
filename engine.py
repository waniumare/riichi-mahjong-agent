"""Simplified closed-hand Riichi Mahjong engine (single hand per game).

Scope (deliberate simplifications, see README/report):
  * 4 players, 136 tiles, no red fives, no calls (chi/pon/kan) -> all hands are closed (menzen)
  * Yaku: riichi, menzen-tsumo, tanyao, yakuhai, chiitoitsu, honitsu, chinitsu, kokushi (as 13 han)
  * Dora + ura-dora, fixed fu (30 regular / 25 chiitoi), standard mangan+ caps
  * Furiten (own-discard), noten payments at exhaustive draw, riichi sticks
"""
import random
import numpy as np
from numba import njit

# tiles: 0-8 man, 9-17 pin, 18-26 sou, 27-30 winds E S W N, 31-33 dragons
TH = np.zeros(34, np.bool_)           # terminal or honor
for _i in range(34):
    if _i >= 27 or _i % 9 in (0, 8):
        TH[_i] = True
KOKUSHI = np.array([0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33], np.int64)


import os

@njit(cache=True)
def _enum(c, i, m, t, pair, seq, out):
    """Enumerate decompositions of one suit; out[pair, m] = max number of taatsu."""
    n = c.shape[0]
    while i < n and c[i] == 0:
        i += 1
    if i >= n:
        if t > out[pair, m]:
            out[pair, m] = t
        return 0
    r = i
    if c[i] >= 3:
        c[i] -= 3
        _enum(c, i, m + 1, t, pair, seq, out)
        c[i] += 3
    if seq and r <= n - 3 and c[i + 1] > 0 and c[i + 2] > 0:
        c[i] -= 1; c[i + 1] -= 1; c[i + 2] -= 1
        _enum(c, i, m + 1, t, pair, seq, out)
        c[i] += 1; c[i + 1] += 1; c[i + 2] += 1
    if c[i] >= 2:
        c[i] -= 2
        if pair == 0:
            _enum(c, i, m, t, 1, seq, out)
        _enum(c, i, m, t + 1, pair, seq, out)
        c[i] += 2
    if seq:
        if r <= n - 2 and c[i + 1] > 0:
            c[i] -= 1; c[i + 1] -= 1
            _enum(c, i, m, t + 1, pair, seq, out)
            c[i] += 1; c[i + 1] += 1
        if r <= n - 3 and c[i + 2] > 0:
            c[i] -= 1; c[i + 2] -= 1
            _enum(c, i, m, t + 1, pair, seq, out)
            c[i] += 1; c[i + 2] += 1
    c[i] -= 1
    _enum(c, i, m, t, pair, seq, out)
    c[i] += 1
    return 0


@njit(cache=True)
def _build(n, seq, maxsum):
    size = 5 ** n
    T = np.full((size, 2, 5), -1, np.int8)
    c = np.zeros(n, np.int64)
    out = np.full((2, 5), -1, np.int64)
    for idx in range(size):
        x = idx
        s = 0
        for k in range(n):
            c[k] = x % 5
            x //= 5
            s += c[k]
        if s > maxsum:
            continue
        out[:, :] = -1
        _enum(c, 0, 0, 0, 0, seq, out)
        for p in range(2):
            for m in range(5):
                T[idx, p, m] = out[p, m]
    return T


def _load_tables():
    fn = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shanten_tables.npz")
    if os.path.exists(fn):
        z = np.load(fn)
        return z["tn"], z["th"]
    tn = _build(9, True, 14)
    th = _build(7, False, 14)
    np.savez(fn, tn=tn, th=th)
    return tn, th


TN, THT = _load_tables()
P5 = np.array([1, 5, 25, 125, 625, 3125, 15625, 78125, 390625], np.int64)


@njit(cache=True)
def shanten_nb(c, TN, THT, P5):
    dp = np.full((2, 5), -1, np.int64)
    dp[0, 0] = 0
    for s in range(4):
        idx = 0
        if s < 3:
            for k in range(9):
                idx += c[9 * s + k] * P5[k]
        else:
            for k in range(7):
                idx += c[27 + k] * P5[k]
        new = np.full((2, 5), -1, np.int64)
        for p1 in range(2):
            for m1 in range(5):
                a = dp[p1, m1]
                if a < 0:
                    continue
                for p2 in range(2 - p1):
                    for m2 in range(5 - m1):
                        v = TN[idx, p2, m2] if s < 3 else THT[idx, p2, m2]
                        if v < 0:
                            continue
                        if a + v > new[p1 + p2, m1 + m2]:
                            new[p1 + p2, m1 + m2] = a + v
        dp = new
    best = 100
    for p in range(2):
        for m in range(5):
            t = dp[p, m]
            if t < 0:
                continue
            if t > 4 - m:
                t = 4 - m
            v = 8 - 2 * m - t - p
            if v < best:
                best = v
    # chiitoitsu / kokushi
    pairs = 0
    kinds = 0
    for i in range(34):
        if c[i] >= 1:
            kinds += 1
        if c[i] >= 2:
            pairs += 1
    ch = 6 - pairs
    if kinds < 7:
        ch += 7 - kinds
    if ch < best:
        best = ch
    kn = 0
    kp = 0
    for j in range(13):
        k = KOKUSHI[j]
        if c[k] >= 1:
            kn += 1
        if c[k] >= 2:
            kp = 1
    ks = 13 - kn - kp
    if ks < best:
        best = ks
    return best


@njit(cache=True)
def shanten_reg_nb(c, TN, THT, P5):
    # regular-hand only (used for yaku checks); mask chiitoi/kokushi by recomputation
    dp = np.full((2, 5), -1, np.int64)
    dp[0, 0] = 0
    for s in range(4):
        idx = 0
        if s < 3:
            for k in range(9):
                idx += c[9 * s + k] * P5[k]
        else:
            for k in range(7):
                idx += c[27 + k] * P5[k]
        new = np.full((2, 5), -1, np.int64)
        for p1 in range(2):
            for m1 in range(5):
                a = dp[p1, m1]
                if a < 0:
                    continue
                for p2 in range(2 - p1):
                    for m2 in range(5 - m1):
                        v = TN[idx, p2, m2] if s < 3 else THT[idx, p2, m2]
                        if v < 0:
                            continue
                        if a + v > new[p1 + p2, m1 + m2]:
                            new[p1 + p2, m1 + m2] = a + v
        dp = new
    best = 100
    for p in range(2):
        for m in range(5):
            t = dp[p, m]
            if t < 0:
                continue
            if t > 4 - m:
                t = 4 - m
            v = 8 - 2 * m - t - p
            if v < best:
                best = v
    return best


def shanten(c):
    return shanten_nb(c, TN, THT, P5)


def shanten_reg(c):
    return shanten_reg_nb(c, TN, THT, P5)


@njit(cache=True)
def waits_nb(c, TN, THT, P5):
    w = np.zeros(34, np.bool_)
    for t in range(34):
        if c[t] >= 4:
            continue
        c[t] += 1
        if shanten_nb(c, TN, THT, P5) == -1:
            w[t] = True
        c[t] -= 1
    return w


def waits_of(c):
    return waits_nb(c, TN, THT, P5)


@njit(cache=True)
def eval_nb(c, rem, TN, THT, P5):
    sh = np.full(34, 99, np.int64)
    uk = np.zeros(34, np.int64)
    mn = 99
    for d in range(34):
        if c[d] > 0:
            c[d] -= 1
            s = shanten_nb(c, TN, THT, P5)
            c[d] += 1
            sh[d] = s
            if s < mn:
                mn = s
    for d in range(34):
        if sh[d] == mn:
            c[d] -= 1
            tot = 0
            for t in range(34):
                if rem[t] <= 0:
                    continue
                rel = c[t] > 0
                if not rel and t < 27:
                    r = t % 9
                    if (r >= 1 and c[t - 1] > 0) or (r >= 2 and c[t - 2] > 0) or (r <= 7 and c[t + 1] > 0) or (r <= 6 and c[t + 2] > 0):
                        rel = True
                if not rel:
                    continue
                c[t] += 1
                if shanten_nb(c, TN, THT, P5) < mn:
                    tot += rem[t]
                c[t] -= 1
            c[d] += 1
            uk[d] = tot
    return sh, uk


def eval_discards(c, rem):
    return eval_nb(c, rem, TN, THT, P5)


# ------------------------------------------------------------------ scoring
def dora_of(ind):
    if ind < 27:
        return ind - ind % 9 + (ind % 9 + 1) % 9
    if ind < 31:
        return 27 + (ind - 27 + 1) % 4
    return 31 + (ind - 31 + 1) % 3


def hand_han(c, riichi, tsumo, seat_wind, dora_cnt):
    """c: complete 14-tile hand (np int8[34]). Returns han (0 = no yaku). seat_wind: tile id 27..30."""
    cc = c.astype(np.int8).copy()
    kok = all(cc[k] >= 1 for k in KOKUSHI)
    if kok and shanten(cc) == -1:
        return 13
    yaku = 0
    if riichi:
        yaku += 1
    if tsumo:
        yaku += 1
    reg = shanten_reg(cc.copy()) == -1
    nz = [i for i in range(34) if cc[i] > 0]
    chii = (len(nz) == 7 and all(cc[i] == 2 for i in nz))
    if not reg and not chii:
        return 0
    if all(not TH[i] for i in nz):
        yaku += 1
    suits = {i // 9 for i in nz if i < 27}
    hon = any(i >= 27 for i in nz)
    if len(suits) == 1:
        yaku += 3 if hon else 6
        if not hon:
            yaku += 0
    if reg:
        for d in (31, 32, 33):
            if cc[d] >= 3:
                yaku += 1
        if cc[seat_wind] >= 3:
            yaku += 1
        if cc[27] >= 3:
            yaku += 1
    if chii and not reg:
        yaku += 2
    elif chii and reg:
        yaku += 2  # shape counted as chiitoi (cannot be both unless 4-of-a-kind; never here)
    if yaku == 0:
        return 0
    return yaku + int((cc.astype(np.int64) * dora_cnt).sum())


def basic_points(han, fu):
    if han >= 13: return 8000
    if han >= 11: return 6000
    if han >= 8: return 4000
    if han >= 6: return 3000
    return min(2000, fu * 2 ** (han + 2))


def _ceil100(x):
    return int(-(-x // 100) * 100)


def ron_value(han, fu, dealer):
    b = basic_points(han, fu)
    return _ceil100(b * (6 if dealer else 4))


def tsumo_payments(han, fu, winner_is_dealer):
    """returns (dealer_pays, nondealer_pays); if winner dealer both equal."""
    b = basic_points(han, fu)
    if winner_is_dealer:
        x = _ceil100(2 * b)
        return x, x
    return _ceil100(2 * b), _ceil100(b)


# ------------------------------------------------------------------ game
def tile_class(t):
    if t >= 27: return 0
    r = t % 9
    if r in (0, 8): return 1
    if r in (1, 7): return 2
    return 3


def suji_state(t, disc_cnt_q):
    """0 none / 1 half / 2 full (or only relevant side) for numeral tiles; honors -> 0."""
    if t >= 27:
        return 0
    r = t % 9
    lo = r >= 3 and disc_cnt_q[t - 3] > 0
    hi = r <= 5 and disc_cnt_q[t + 3] > 0
    if r <= 2:    # 1,2,3 : only t+3 matters
        return 2 if hi else 0
    if r >= 6:    # 7,8,9 : only t-3 matters
        return 2 if lo else 0
    return 2 if (lo and hi) else (1 if (lo or hi) else 0)


def feat_key(t, seen, disc_cnt_q, ndisc_at_riichi):
    late = 0 if ndisc_at_riichi <= 6 else (1 if ndisc_at_riichi <= 12 else 2)
    return (tile_class(t), suji_state(t, disc_cnt_q), min(3, int(seen)), late)


def turn_bin(n):
    return 0 if n <= 4 else (1 if n <= 8 else (2 if n <= 12 else 3))


class Game:
    def __init__(self, seed, agents, dealer=None, log=None):
        self.rng = random.Random(seed)
        self.agents = agents
        self.log = log                      # None or dict of lists for calibration
        w = [t for t in range(34) for _ in range(4)]
        self.rng.shuffle(w)
        self.dead = w[-14:]
        self.wall = w[:-14]
        self.pos = 52
        self.dealer = self.rng.randrange(4) if dealer is None else dealer
        self.h = np.zeros((4, 34), np.int8)
        for p in range(4):
            for t in w[13 * p: 13 * p + 13]:
                self.h[p][t] += 1
        self.dora_ind = [self.dead[4]]
        self.ura_ind = [self.dead[5]]
        self.vis = np.zeros(34, np.int8)
        self.vis[self.dora_ind[0]] += 1
        self.dora_cnt = np.zeros(34, np.int64)
        self.dora_cnt[dora_of(self.dora_ind[0])] += 1
        self.ura_cnt = np.zeros(34, np.int64)
        self.ura_cnt[dora_of(self.ura_ind[0])] += 1
        self.disc = [[] for _ in range(4)]
        self.disc_cnt = np.zeros((4, 34), np.int8)
        self.safe = np.zeros((4, 34), np.bool_)
        self.riichi = [-1] * 4              # number of own discards at declaration
        self.sticks = 0
        self.delta = np.zeros(4, np.int64)
        self.nd = 0                         # total discards so far
        self.result = None
        self.seat_wind = [27 + (p - self.dealer) % 4 for p in range(4)]
        self.n_riichi = 0
        self.rinfo = [None] * 4

    # helpers for agents
    def wall_left(self):
        return len(self.wall) - self.pos

    def rem_for(self, p):
        return (4 - self.vis.astype(np.int64) - self.h[p]).clip(0)

    def any_riichi_other(self, p):
        return [q for q in range(4) if q != p and self.riichi[q] >= 0]

    def _han(self, p, tsumo, ron_tile=None):
        c = self.h[p].copy()
        if ron_tile is not None:
            c[ron_tile] += 1
        r = self.riichi[p] >= 0
        dc = self.dora_cnt + (self.ura_cnt if r else 0)
        return hand_han(c, r, tsumo, self.seat_wind[p], dc), c

    def _is_furiten(self, q):
        return bool((self.wm[q] & (self.disc_cnt[q] > 0)).any())

    def _finish_win(self, p, han, c, tsumo, loser=None):
        fu = 25 if (c.max() <= 2 and (c > 0).sum() == 7) else 30
        pd = p == self.dealer
        d = np.zeros(4, np.int64)
        if tsumo:
            dp, nd = tsumo_payments(han, fu, pd)
            for q in range(4):
                if q != p:
                    pay = dp if q == self.dealer else nd
                    d[q] -= pay; d[p] += pay
            kind = "tsumo"
        else:
            v = ron_value(han, fu, pd)
            d[loser] -= v; d[p] += v
            kind = "ron"
        d[p] += 1000 * self.sticks
        self.sticks = 0
        self.delta += d
        self.result = dict(kind=kind, winner=p, loser=loser, han=han, value=int(d[p]),
                           turns=self.nd, riichi=[x >= 0 for x in self.riichi])

    def _log_discard(self, p, d):
        """Calibration logging, called before the discard is applied."""
        L = self.log
        s_after = getattr(self, "last_s", 9) if self.riichi[p] < 0 else -9
        for q in range(4):
            if q == p:
                continue
            if self.riichi[q] >= 0:
                if self.safe[q][d]:
                    continue
                seen = self.vis[d] + self.h[p][d]
                hit = bool(self.wm[q][d]) and not self._is_furiten(q)
                L["danger"].append((feat_key(d, seen, self.disc_cnt[q], self.riichi[q]), hit, s_after))
            else:
                L["tenpai"].append((turn_bin(len(self.disc[q])), self.wm[q] is not None))

    def play(self):
        self.wm = [None] * 4
        for q in range(4):
            if shanten(self.h[q].copy()) == 0:
                self.wm[q] = waits_of(self.h[q].copy())
        p = self.dealer
        while True:
            if self.pos >= len(self.wall):
                return self._draw_end()
            t = self.wall[self.pos]; self.pos += 1
            self.h[p][t] += 1
            if self.wm[p] is not None and self.wm[p][t]:
                han, c = self._han(p, True)
                if han > 0:
                    self._finish_win(p, han, c, True)
                    return self.result
            if self.riichi[p] >= 0:
                d, decl = t, False
            else:
                d, decl = self.agents[p].choose(self, p)
            if self.log is not None:
                self._log_discard(p, d)
            self.h[p][d] -= 1
            self.disc[p].append(d)
            self.disc_cnt[p][d] += 1
            self.vis[d] += 1
            self.nd += 1
            self.safe[p][d] = True
            for q in range(4):
                if q != p and self.riichi[q] >= 0:
                    self.safe[q][d] = True
            hc = self.h[p].copy()
            self.wm[p] = waits_of(hc) if shanten(hc) == 0 else None
            if decl and self.wm[p] is None:
                raise RuntimeError("riichi without tenpai")
            # ron check (head bump: first player in turn order)
            ronned = False
            for k in range(1, 4):
                q = (p + k) % 4
                if self.wm[q] is not None and self.wm[q][d]:
                    han, c = self._han(q, False, ron_tile=d)
                    if han > 0 and not self._is_furiten(q):
                        self._finish_win(q, han, c, False, loser=p)
                        return self.result
            if decl:
                self.riichi[p] = len(self.disc[p]) - 1
                self.n_riichi += 1
                rm = self.rem_for(p)
                self.rinfo[p] = (self.wall_left(), int((rm * self.wm[p]).sum()), int(rm.sum()))
                self.delta[p] -= 1000
                self.sticks += 1
            p = (p + 1) % 4

    def _draw_end(self):
        ten = [self.wm[q] is not None for q in range(4)]
        n = sum(ten)
        d = np.zeros(4, np.int64)
        if 0 < n < 4:
            for q in range(4):
                d[q] = 3000 // n if ten[q] else -3000 // (4 - n)
        self.delta += d
        self.result = dict(kind="draw", winner=None, loser=None, han=0, value=0, turns=self.nd,
                           riichi=[x >= 0 for x in self.riichi], tenpai=ten)
        return self.result
