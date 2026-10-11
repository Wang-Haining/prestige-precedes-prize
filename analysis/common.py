"""Shared data loading and estimators.

All rates and differences are new lead awards per 1,000 index papers. Intervals come from a two-way bootstrap that
weights each paper by an exponential weight for its scientist times one for its journal.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("DATA_DIR", ROOT / "data"))
RESULTS = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))
B = int(os.environ.get("B", 1000))  # bootstrap replicates
BH = 200  # replicates for the breakpoint fits
GAP = 24  # months between the two papers of a pair, so that their after windows are disjoint
NBIN = 20
YS = ("y_pre", "y_post")  # before window (months -24 to <0) and after window (months +12 to <36)
TOP = 0.95


def load():
    """Index papers of last authors with derived columns, sorted by scientist and publication month."""
    d = pd.read_parquet(DATA / "index_papers.parquet")
    pub = pd.to_datetime(d.publication_date)
    d["mo"] = pub.dt.year * 12 + pub.dt.month - 1 + (pub.dt.day - 1) / 31
    yrs24 = ((pub - pd.DateOffset(months=24)) - pd.to_datetime(d.first_last_date)).dt.days / 365.25
    d["L"] = np.select([yrs24 >= 10, yrs24 >= 5], ["ge10", "5to10"], "lt5")  # years as last author at month -24
    d["bin"] = np.minimum((d.pct.to_numpy(float) * NBIN).astype(int), NBIN - 1)
    d["s"] = pd.factorize(d.choice_set_id)[0]
    d["a"] = pd.factorize(d.author_id)[0]
    d["j"] = pd.factorize(d.journal_id)[0]
    return d.sort_values(["a", "mo"], kind="mergesort").reset_index(drop=True)


def cuts(d):
    """Definitions of a top journal."""
    return {
        "top journal in set": d.pct.eq(1.0).to_numpy(),
        "top 5%": (d.pct >= TOP).to_numpy(),
        "top decile": d.dec.eq(10).to_numpy(),
        "top quartile": d.quartile.eq(4).to_numpy(),
    }


def boot_w(d, rng):
    return rng.exponential(1.0, d.a.max() + 1)[d.a.to_numpy()] * rng.exponential(1.0, d.j.max() + 1)[d.j.to_numpy()]


def std_gap(s, A, y, w):
    """Directly standardized difference (per 1,000) and ratio, top vs other, weighting sets by index papers."""
    k = s.max() + 1
    A = A.astype(float)
    n1, n0 = np.bincount(s, w * A, k), np.bincount(s, w * (1 - A), k)
    e1, e0 = np.bincount(s, w * A * y, k), np.bincount(s, w * (1 - A) * y, k)
    ok = (n1 > 0) & (n0 > 0)
    W = (n1 + n0)[ok]
    p1, p0 = (W * e1[ok] / n1[ok]).sum() / W.sum(), (W * e0[ok] / n0[ok]).sum() / W.sum()
    return 1000 * (p1 - p0), p1 / p0, 1000 * p1, 1000 * p0


def pairs(d, A, min_gap=GAP, max_gap=None):
    """One pair per scientist: the earliest index paper followed, at least min_gap months later, by an index paper at
    the other level, paired with the first such later paper. d must be sorted by scientist and month."""
    a, mo = d.a.to_numpy(), d.mo.to_numpy()
    st = np.flatnonzero(np.r_[True, a[1:] != a[:-1]])
    en = np.r_[st[1:], len(a)]
    p1, p2 = [], []
    for s_, e_ in zip(st, en, strict=True):
        if e_ - s_ < 2:
            continue
        AA, tt = A[s_:e_], mo[s_:e_]
        if AA.min() == AA.max():
            continue
        for i in range(e_ - s_ - 1):
            g = tt[i + 1 :] - tt[i]
            ok = (AA[i + 1 :] != AA[i]) & (g >= min_gap)
            if max_gap is not None:
                ok &= g < max_gap
            c = np.flatnonzero(ok)
            if len(c):
                p1.append(s_ + i)
                p2.append(s_ + i + 1 + c[0])
                break
    p1, p2 = np.array(p1, int), np.array(p2, int)
    first = A[p1] == 1
    top = np.where(first, p1, p2)
    non = np.where(first, p2, p1)
    return pd.DataFrame(
        {
            "p1": p1,
            "top": top,
            "non": non,
            "first": first,
            "gap": mo[p2] - mo[p1],
            "L": d.L.to_numpy()[p1],
            "F": d.f60.to_numpy()[p1],
            "year1": d.publication_year.to_numpy()[p1],
            "year2": d.publication_year.to_numpy()[p2],
            "dom": d.dom_name.to_numpy()[p1],
        }
    )


def pair_est(P, d, y, wa=None, wj=None):
    """Order-balanced difference (per 1,000) and ratio, top-journal vs other paper of the same scientist."""
    Y = d[y].to_numpy(float)
    a, j = d.a.to_numpy(), d.j.to_numpy()
    out = {}
    for o, m in (("first", P["first"].to_numpy()), ("second", ~P["first"].to_numpy())):
        t, n = P.top.to_numpy()[m], P.non.to_numpy()[m]
        if len(t) == 0:
            return None
        w_a = np.ones(len(t)) if wa is None else wa[a[t]]
        wt = w_a if wj is None else w_a * wj[j[t]]
        wn = w_a if wj is None else w_a * wj[j[n]]
        mt, mn = (wt * Y[t]).sum() / wt.sum(), (wn * Y[n]).sum() / wn.sum()
        out[o] = (1000 * (mt - mn), np.log(mt / mn) if mt > 0 and mn > 0 else np.nan, 1000 * mt, 1000 * mn, len(t))
    rd = 0.5 * (out["first"][0] + out["second"][0])
    lrr = 0.5 * (out["first"][1] + out["second"][1])
    return {
        "rd_bal": rd,
        "rr_bal": np.exp(lrr),
        "rd_first": out["first"][0],
        "rd_second": out["second"][0],
        "rr_first": np.exp(out["first"][1]),
        "rr_second": np.exp(out["second"][1]),
        "rate_top": 0.5 * (out["first"][2] + out["second"][2]),
        "rate_non": 0.5 * (out["first"][3] + out["second"][3]),
        "n_first": out["first"][4],
        "n_second": out["second"][4],
    }


def thin(d, min_gap=GAP):
    """Index papers kept forward from each scientist's earliest, at least min_gap months apart (scientists with >= 2)."""
    a, mo = d.a.to_numpy(), d.mo.to_numpy()
    keep = np.zeros(len(d), bool)
    last_a, last_t = -1, -1e9
    for i in range(len(d)):
        if a[i] != last_a or mo[i] - last_t >= min_gap:
            keep[i] = True
            last_a, last_t = a[i], mo[i]
    t = d[keep].reset_index(drop=True)
    n = t.groupby("a").a.transform("size")
    return t[n >= 2].reset_index(drop=True)


def demean(g, M, w):
    k = g.max() + 1
    sw = np.bincount(g, w, k)
    sw[sw == 0] = 1
    out = np.empty_like(M)
    for c in range(M.shape[1]):
        out[:, c] = M[:, c] - (np.bincount(g, w * M[:, c], k) / sw)[g]
    return out


def wls(X, y, w):
    Xw = X * w[:, None]
    return np.linalg.lstsq(X.T @ Xw, Xw.T @ y, rcond=None)[0]


def att(c, t, y, w):
    """Weighted matched difference: mean over first top-journal papers of (y - mean of comparison papers in the same
    cell); cells without both kinds of paper are dropped. Returns the difference per 1,000 and the share kept."""
    k = c.max() + 1
    wt, wc = w * t, w * (1 - t)
    nt, nc = np.bincount(c, wt, k), np.bincount(c, wc, k)
    st, sc = np.bincount(c, wt * y, k), np.bincount(c, wc * y, k)
    ok = (nt > 0) & (nc > 0)
    return 1000 * (st[ok] - nt[ok] * sc[ok] / nc[ok]).sum() / nt[ok].sum(), nt[ok].sum() / nt.sum()


def cells(d, keys):
    return pd.factorize(pd.MultiIndex.from_frame(d[keys].astype(str)))[0]


def save(rows, name):
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS / name, index=False)
