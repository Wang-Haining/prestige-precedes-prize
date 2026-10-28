"""First top-journal paper as a time-to-event comparison (last authors; United States 2014-2021, Japan 2014-2018).

Each trial is a scientist-month in which a last author with no earlier first- or last-authored top-journal paper (from
2009) and an article or review in the previous 36 months publishes a ranked paper: a treated trial when it is the first
top-journal paper, a comparison trial when it is a paper in another journal (near-top arm: percentile 0.85 to <0.95).
Comparison trials are matched to treated trials of the same month in exact cells on the prior record (A: field, years
as last author, an award before month -24, career age, productivity, institution band, an award in months -60 to -24,
highest earlier percentile; B: A and any new award in months -24 to -1), capped at 20 per treated trial, and weighted
to the treated trials of their cell. Quarter cells replace the month by the calendar quarter.

After time zero (the trial month) a trial is active without a new award, funded (a new lead award started), or exited
(its last article or review is followed by more than 24 months without one, dated at the last paper). Discrete monthly
Aalen-Johansen estimates per arm, with comparison trials weighted by the stabilized inverse probability of not yet
being censored by their own first top-journal paper (see censoring_weights) and administrative censoring at the end of
complete award coverage. Measures at 12 to 84 months: cumulative incidence of a new award (exit competing, and exit not
competing), P(active and funded), exit, NIH competing renewal among trials holding a renewable NIH project at time zero,
mean cumulative award amounts started, and the incidence of cumulative amounts reaching $500,000 and $1,000,000 (Japan:
5 and 20 million yen). Plain windows (no exit, no censoring) give the share with any new award in months -24 to -1, 0
to 23 and 12 to 35.

Paired change: for each after window (months 12-35, 36-59, 60-83), trials observed through its end, with the before
window (months -24 to -1) on the same trials: difference after, difference before, their change, the ratio of the risk
ratios, and the change over the comparison rate in the after window, uncorrected and with each arm-window rate divided
by its detection share (acknowledged own awards; binomial draws per replicate, as in bounds.py). Restriction: trials of
2017-2020 with the before window and months 12-35, beside first_paper.py.

Intervals: 1,000 two-way bootstrap replicates (scientist x journal exponential weights); matching weights recomputed in
each replicate; cells and censoring weights held fixed.
-> results/event_curves.csv (new award, with and without exit competing), event_horizons.csv, event_before.csv (plain
windows), event_sizes.csv, event_change.csv, event_detection.csv, event_validation.csv
"""

import zlib

import numpy as np
import pandas as pd

from analysis.common import DATA, RESULTS, B, save

K = 85  # months 0..84 after time zero
HOR = [12, 24, 36, 48, 60, 72, 84]
SEED_CIF, SEED_CHANGE = 20261027, 20261028
THR = {"US": [500_000, 1_000_000], "JP": [5_000_000, 20_000_000]}
AFTER = {"months 12..35": (12, 36), "months 36..59": (36, 60), "months 60..83": (60, 84)}
BEFORE = (-24, 0)
REF = {"months -24..-1": "ref_m24_m1", "months 12..35": "ref_12_35", "months 36..59": "ref_36_59", "months 60..83": "ref_60_83"}
COHORT = {  # cells, k6 suffix (seed), comparator
    "month": ("month", "", "all other journals"),
    "quarter": ("quarter", "_q", "all other journals"),
    "neartop": ("month", "_nt", "near-top journals"),
}
MAIN = {"US": "US 2014-2021, last authors", "JP": "JP 2014-2018, last authors"}
CIF_SAMPLES = {  # label: (index years, administrative end)
    "US": {"US 2014-2021, last authors": (2021, "T_admin")},
    "JP": {
        "JP 2014-2018, last authors": (2018, "T_admin"),
        "JP 2014-2021, last authors": (2021, "T_admin"),
        "JP 2014-2018, coverage to 2024-05": (2018, "T_admin_alt"),
    },
}


def q(x):
    return np.quantile(x, 0.025), np.quantile(x, 0.975)


def load(cc, cohort):
    """Trials of one cohort, the censoring weights of its comparison trials (months 0..84), and award events."""
    pre = "" if cc == "US" else "jp_"
    u = pd.read_parquet(DATA / f"{pre}event_trials.parquet")
    d = u[u[f"in_{cohort}"]].reset_index(drop=True)
    d["cellA"], d["cellB"] = d[f"cellA_{cohort}"], d[f"cellB_{cohort}"]
    d["trial"] = np.arange(len(d))
    w = pd.concat([pd.read_parquet(f) for f in sorted(DATA.glob(f"{pre}event_ipcw_{cohort}*.parquet"))])
    c = d.loc[d.treated.eq(0), ["author_id", "mo"]].merge(w, on=["author_id", "mo"], how="left")
    wq = np.nan_to_num(c[[f"q{i:02d}" for i in range(29)]].to_numpy(np.float32), nan=1.0)  # NaN: after censoring
    W = np.repeat(wq, 3, axis=1)[:, :K]
    d["ctrl_row"] = -1
    d.loc[d.treated.eq(0), "ctrl_row"] = np.arange(int((d.treated == 0).sum()))
    ev = pd.read_parquet(DATA / f"{pre}event_awards.parquet").merge(d[["author_id", "mo", "trial"]], on=["author_id", "mo"])
    return d, W, ev


def censoring_weights(c, am, x_base, seed=SEED_CIF, n_fit=400_000):
    """Stabilized inverse probability of not yet being censored by a first top-journal paper (comparison trials c).

    Pooled logistic regression over person-quarters on time since time zero (nine bands), the baseline prior record
    (x_base, dummy-coded) and, since time zero, a new award, exit, log ranked papers, a paper at percentile >= 0.85, and
    log works in the last 12 months; fitted on n_fit sampled comparison trials of the first- and last-author cohort;
    numerator on time only; truncated at the 0.5th and 99.5th percentiles. The author-month panel am (author_id, mo,
    kind = work or ranked, pct) stays on the cluster; data/*event_ipcw_*.parquet hold the resulting weights of the
    last-author comparison trials. Returns an (n x 85) array."""
    from sklearn.linear_model import LogisticRegression

    nq = K // 3 + 1
    cadm = np.minimum(c.T_admin.to_numpy(), K)
    ttop = c.T_top.fillna(1e9).to_numpy()
    codes = pd.Index(pd.unique(am.author_id))
    ac = codes.get_indexer(c.author_id)

    def keys(x):
        x = x.assign(ac=codes.get_indexer(x.author_id)).sort_values(["ac", "mo"])
        return x.ac.to_numpy(np.int64) * 100000 + x.mo.to_numpy(np.int64)

    rk = am[am.kind.eq("ranked")]
    kw, kr, kn = keys(am[am.kind.eq("work")]), keys(rk), keys(rk[rk.pct >= 0.85])
    b0 = x_base.to_numpy(np.float32)
    mo = c.mo.to_numpy(np.int64)
    ta, te = c.T_award.fillna(1e9).to_numpy(), c.T_exit.fillna(1e9).to_numpy()
    qbins = np.array([0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 6, 6] + [7] * 8 + [8] * (nq - 20))[:nq]

    def feats(rows, qq):
        s = 3 * qq
        ft = np.tile(np.eye(9, dtype=np.float32)[qbins[qq]][1:], (len(rows), 1))
        a = ac[rows]

        def cnt(kk, lo, hi):
            return np.maximum(np.searchsorted(kk, a * 100000 + hi) - np.searchsorted(kk, a * 100000 + lo), 0)

        tv = np.column_stack(
            [
                ta[rows] < s,
                te[rows] < s,
                np.log1p(cnt(kr, mo[rows] + 1, mo[rows] + s)),
                cnt(kn, mo[rows] + 1, mo[rows] + s) > 0,
                np.log1p(cnt(kw, mo[rows] + s - 12, mo[rows] + s)),
            ]
        ).astype(np.float32)
        return np.hstack([ft, b0[rows], tv]), ft

    rng = np.random.default_rng(seed)
    sub = rng.choice(len(c), size=min(len(c), n_fit), replace=False)
    X, Y, XT = [], [], []
    for qq in range(nq):
        r = sub[(3 * qq < cadm[sub]) & (ttop[sub] >= 3 * qq)]
        if len(r):
            f, ft = feats(r, qq)
            X.append(f)
            XT.append(ft)
            Y.append((ttop[r] < 3 * qq + 3).astype(np.int8))
    X, Y, XT = np.vstack(X), np.concatenate(Y), np.vstack(XT)
    den = LogisticRegression(max_iter=500, C=1e4).fit(X, Y)
    num = LogisticRegression(max_iter=500, C=1e4).fit(XT, Y)
    W = np.ones((len(c), K), np.float32)
    cum = np.ones(len(c))
    for qq in range(nq):
        lo = 3 * qq
        if lo >= K:
            break
        W[:, lo : min(lo + 3, K)] = cum[:, None]
        r = np.flatnonzero((lo < cadm) & (ttop >= lo))
        if len(r):
            f, ft = feats(r, qq)
            cum[r] = cum[r] * (1 - num.predict_proba(ft)[:, 1]) / (1 - den.predict_proba(f)[:, 1])
    lo, hi = np.quantile(W[W != 1], [0.005, 0.995])
    return np.clip(W, lo, hi)


class Arm:
    """At-risk and event structures of one arm; replicate weights enter in run()."""

    def __init__(self, s, ip, tev, dol, cens, exit_competes=True):
        n = len(s)
        ip = np.ones((n, K), np.float32) if ip is None else ip
        kk = np.arange(K)[None, :]
        ta = np.nan_to_num(tev, nan=1e9)
        te = np.nan_to_num(s.T_exit.to_numpy(float), nan=1e9) if exit_competes else np.full(n, 1e9)
        s0 = np.minimum(ta, te)
        is01 = ta <= te
        self.R0 = (ip * ((kk <= s0[:, None]) & (kk < cens[:, None]))).astype(np.float32)
        te1 = np.where(te <= ta, ta + 1, te)
        r1 = is01[:, None] & (kk > ta[:, None]) & (kk <= te1[:, None]) & (kk < cens[:, None])
        self.R1 = (ip * r1).astype(np.float32)
        self.Ralive = (ip * ((kk <= te[:, None]) & (kk < cens[:, None]))).astype(np.float32)

        def ev(t, flag):
            idx = np.flatnonzero(flag & (t < cens) & (t < K))
            tt = t[idx].astype(int)
            return idx, tt, ip[idx, tt]

        self.e01, self.e02 = ev(s0, is01), ev(s0, ~is01)
        self.e12 = ev(te1, is01 & (te1 > ta))
        self.eE = ev(te, np.ones(n, bool))
        self.dol = None
        if dol is not None:
            r, kd, amt = dol
            ok = (kd < cens[r]) & (kd <= te[r]) & (kd < K)
            self.dol = (r[ok], kd[ok].astype(int), amt[ok] * ip[r[ok], kd[ok].astype(int)])

    def run(self, w):
        w = w.astype(np.float32)

        def bc(e):
            return np.bincount(e[1], weights=(w[e[0]] * e[2]).astype(float), minlength=K)

        r0, r1 = (w @ self.R0).astype(float), (w @ self.R1).astype(float)
        d01, d02, d12 = bc(self.e01), bc(self.e02), bc(self.e12)
        with np.errstate(divide="ignore", invalid="ignore"):
            h01 = np.where(r0 > 0, d01 / r0, 0)
            h02 = np.where(r0 > 0, d02 / r0, 0)
            h12 = np.where(r1 > 0, d12 / r1, 0)
        p0, p1, f = np.empty(K + 1), np.empty(K + 1), np.zeros(K + 1)
        p0[0], p1[0] = 1.0, 0.0
        for t in range(K):
            f[t + 1] = f[t] + p0[t] * h01[t]
            p0[t + 1] = p0[t] * (1 - h01[t] - h02[t])
            p1[t + 1] = p1[t] * (1 - h12[t]) + p0[t] * h01[t]
        out = {"cif": f, "active_funded": p1, "exit": 1 - p0 - p1}
        if self.dol is not None:
            ra = (w @ self.Ralive).astype(float)
            de = bc(self.eE)
            r, kd, a = self.dol
            dd = np.bincount(kd, weights=(w[r] * a).astype(float), minlength=K)
            with np.errstate(divide="ignore", invalid="ignore"):
                he, md = np.where(ra > 0, de / ra, 0), np.where(ra > 0, dd / ra, 0)
            surv = np.concatenate([[1.0], np.cumprod(1 - he)])
            out["dollars"] = np.concatenate([[0.0], np.cumsum(surv[:-1] * md)])
        return out


def cell_weights(cell, t, b):
    """Treated trials keep b if their cell has a comparison trial; comparison trials take b times the cell ratio."""
    kc = cell.max() + 1
    st, sc = np.bincount(cell, b * t, kc), np.bincount(cell, b * (1 - t), kc)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(sc > 0, st / sc, 0)
    return np.where(t == 1, b * (sc[cell] > 0), b * r[cell])


def matched(s, cellcol):
    """Trials in cells with both treated and comparison trials."""
    cell = pd.factorize(s[cellcol])[0]
    g = pd.DataFrame({"c": cell, "t": s.treated.to_numpy()}).groupby("c").t.agg(["sum", "size"])
    return np.isin(cell, g.index[(g["sum"] > 0) & (g["sum"] < g["size"])])


def threshold_times(dd, n, thr):
    """Month at which cumulative amounts started since time zero first reach thr (NaN if not within 84 months)."""
    x = dd.sort_values(["r", "k"]).copy()
    x["cum"] = x.groupby("r").amount.cumsum()
    return x[x.cum >= thr].groupby("r").k.min().reindex(range(n)).to_numpy(float)


def any_in(e, n, lo, hi):
    return np.isin(np.arange(n), e.r[(e.k >= lo) & (e.k < hi)].unique()).astype(float)


def cif(cc, d, W, ev, label, variant, admin_col, cohort):
    """Aalen-Johansen measures, plain windows and sizes for one sample, cell variant and cohort."""
    n_treat = int(d.treated.sum())
    cellcol = {"A": "cellA", "B": "cellB"}[variant]
    s = d[matched(d, cellcol)].reset_index(drop=True)
    ctrl = s.ctrl_row.to_numpy()
    cell, t = pd.factorize(s[cellcol])[0], s.treated.to_numpy()
    cens = np.minimum(np.where(t == 1, s[admin_col], np.minimum(s[admin_col], s.T_top.fillna(1e9))), K).astype(float)
    pos = pd.Series(np.arange(len(s)), index=s.trial.to_numpy())
    dd = ev[ev.trial.isin(pos.index) & ev.k.between(0, 84)].copy()
    dd["r"] = pos.reindex(dd.trial).to_numpy()
    dd["amount"] = dd.amount.fillna(0).astype(float)
    a_, j_ = pd.factorize(s.author_id)[0], pd.factorize(s.journal_id.fillna("none"))[0]
    rng = np.random.default_rng(SEED_CIF + zlib.crc32((label + variant).encode()) % 1000)
    one_ = np.ones(len(s), bool)
    specs = {"award": (one_, s.T_award.to_numpy(float), True), "award_noexit": (one_, s.T_award.to_numpy(float), False)}
    if cc == "US":
        specs["award_pub"] = (one_, s.T_award_pub.to_numpy(float), False)
        specs["renewal"] = (s.ren_elig.to_numpy(bool), s.T_ren.to_numpy(float), False)
    for x in THR[cc]:
        specs[f"dollars_ge_{x}"] = (one_, threshold_times(dd, len(s), x), False)
    arms = {}
    for o, (m, tev, withdol) in specs.items():
        for arm in (1, 0):
            rows = np.flatnonzero(m & (t == arm))
            dl = None
            if withdol:
                sel = dd[dd.r.isin(rows)]
                loc = pd.Series(np.arange(len(rows)), index=rows)
                dl = (loc.reindex(sel.r).to_numpy(), sel.k.to_numpy(), sel.amount.to_numpy())
            ip = None if arm == 1 else W[ctrl[rows]]
            arms[(o, arm)] = (rows, Arm(s.iloc[rows], ip, tev[rows], dl, cens[rows], exit_competes=(o != "award_noexit")))
    adm = s[admin_col].to_numpy()
    win = {
        "months -24..-1": (s.pre_award_m24.to_numpy(float), np.ones(len(s))),
        "months 0..23": (any_in(dd, len(s), 0, 24), (adm >= 24).astype(float)),
        "months 12..35": (any_in(dd, len(s), 12, 36), (adm >= 36).astype(float)),
    }

    def one(b):
        res = {}
        for o, (m, _, _) in specs.items():
            ww = cell_weights(cell[m] if m.all() else pd.factorize(cell[m])[0], t[m], b[m])
            wfull = np.zeros(len(s))
            wfull[np.flatnonzero(m)] = ww
            for arm in (1, 0):
                rows, A = arms[(o, arm)]
                res[(o, arm)] = A.run(wfull[rows])
        for wn, (y, ob) in win.items():
            wo = cell_weights(cell, t, b * ob)
            res[(wn, 1)] = (wo * y * t).sum() / (wo * t).sum()
            res[(wn, 0)] = (wo * y * (1 - t)).sum() / (wo * (1 - t)).sum()
        return res

    est = one(np.ones(len(s)))
    reps = [one(rng.exponential(1.0, a_.max() + 1)[a_] * rng.exponential(1.0, j_.max() + 1)[j_]) for _ in range(B)]
    tag = {"cc": cc, "sample": label, "variant": variant, "cells": COHORT[cohort][0], "comparator": COHORT[cohort][2]}
    curves, hors = [], []
    for o in specs:
        for kx in ["cif", "active_funded", "exit"] + (["dollars"] if "dollars" in est[(o, 1)] else []):
            curve = o in ("award", "award_noexit")
            e1, e0 = est[(o, 1)][kx], est[(o, 0)][kx]
            r1 = np.array([r[(o, 1)][kx] for r in reps])
            r0 = np.array([r[(o, 0)][kx] for r in reps])
            for mth in range(K + 1) if curve else ():
                (l1, h1), (l0, h0), (ld, hd) = q(r1[:, mth]), q(r0[:, mth]), q(r1[:, mth] - r0[:, mth])
                curves.append(
                    {
                        **tag,
                        "outcome": o,
                        "measure": kx,
                        "month": mth,
                        "treated": e1[mth],
                        "treated_lo": l1,
                        "treated_hi": h1,
                        "control": e0[mth],
                        "control_lo": l0,
                        "control_hi": h0,
                        "diff": e1[mth] - e0[mth],
                        "diff_lo": ld,
                        "diff_hi": hd,
                    }
                )
            for h in HOR:
                with np.errstate(divide="ignore", invalid="ignore"):
                    rel = r1[:, h] / r0[:, h] - 1
                ld, hd = q(r1[:, h] - r0[:, h])
                lr, hr = q(rel[np.isfinite(rel)]) if np.isfinite(rel).any() else (np.nan, np.nan)
                hors.append(
                    {
                        **tag,
                        "outcome": o,
                        "measure": kx,
                        "horizon": h,
                        "treated": e1[h],
                        "control": e0[h],
                        "diff": e1[h] - e0[h],
                        "diff_lo": ld,
                        "diff_hi": hd,
                        "rel": e1[h] / e0[h] - 1 if e0[h] > 0 else np.nan,
                        "rel_lo": lr,
                        "rel_hi": hr,
                    }
                )
    windows = []
    for wn in win:
        b1, b0 = np.array([r[(wn, 1)] for r in reps]), np.array([r[(wn, 0)] for r in reps])
        ld, hd = q(b1 - b0)
        e1, e0 = est[(wn, 1)], est[(wn, 0)]
        windows.append(
            {
                **tag,
                "window": wn,
                "treated": e1,
                "control": e0,
                "diff": e1 - e0,
                "diff_lo": ld,
                "diff_hi": hd,
                "rel": e1 / e0 - 1 if e0 > 0 else np.nan,
            }
        )
    sizes = []
    tex = s.T_exit.to_numpy(float)
    for o, (m, tev, _) in specs.items():
        for arm in (1, 0):
            rows = np.flatnonzero(m & (t == arm))
            ta, te, cr = tev[rows], np.nan_to_num(tex[rows], nan=1e9), cens[rows]
            top = np.nan_to_num(s.T_top.to_numpy(float)[rows], nan=1e9)
            sel = dd[dd.r.isin(rows)]
            rr = sel.r.to_numpy()
            after_exit = (sel.k.to_numpy() > np.nan_to_num(tex[rr], nan=1e9)) & (sel.k.to_numpy() < cens[rr])
            sizes.append(
                {
                    **tag,
                    "outcome": o,
                    "arm": "treated" if arm else "control",
                    "treated_in_sample": n_treat,
                    "treated_matched": int(t.sum()),
                    "trials_with_award_after_exit_date": int(sel.r[after_exit].nunique()),
                    "trials": len(rows),
                    "scientists": int(s.author_id.iloc[rows].nunique()),
                    "events_0_84": int(np.sum((ta < cr) & (ta <= te))),
                    "exits_0_84": int(np.sum(te < cr)),
                    "censored_top_paper": int(np.sum(top < np.minimum(adm[rows], K))),
                    "followed_to_36": int(np.sum(cr > 36)),
                    "followed_to_60": int(np.sum(cr > 60)),
                    "followed_to_84": int(np.sum(cr >= 85)),
                }
            )
    print(cc, label, cohort, variant, flush=True)
    return curves, hors, windows, sizes


def change(cc, d, ev, label, cohort, after=AFTER):
    """Paired change on the absolute and relative scales, before window against each after window, A cells."""
    s0 = d.reset_index(drop=True)
    n = len(s0)
    pos = pd.Series(np.arange(n), index=s0.trial.to_numpy())
    e = ev[ev.trial.isin(pos.index) & ev.k.between(-24, 83)].assign(r=lambda x: pos.reindex(x.trial).to_numpy())
    yb = any_in(e, n, *BEFORE)
    rows, det = [], []
    for wn, (lo, hi) in after.items():
        idx = np.flatnonzero(s0.T_admin.to_numpy() >= hi)
        s = s0.iloc[idx].reset_index(drop=True)
        ya, yb_ = any_in(e, n, lo, hi)[idx], yb[idx]
        keep = matched(s, "cellA")
        s, ya, yb_ = s[keep].reset_index(drop=True), ya[keep], yb_[keep]
        cell, t = pd.factorize(s.cellA)[0], s.treated.to_numpy()
        shares = {}
        if "ref_pop" in s:
            pop = s.ref_pop.to_numpy(bool)
            for arm in (1, 0):
                for w_, rr, yy in (("before", s[REF["months -24..-1"]].to_numpy(), yb_), ("after", s[REF[wn]].to_numpy(), ya)):
                    sel = pop & (t == arm) & (rr == 1)
                    shares[(arm, w_)] = (int(sel.sum()), int(yy[sel].sum()))
                    det.append(
                        {
                            "cc": cc,
                            "sample": label,
                            "cells": COHORT[cohort][0] + " A",
                            "comparator": COHORT[cohort][2],
                            "after_window": wn,
                            "arm": "treated" if arm else "control",
                            "window": w_,
                            "trial_windows_with_reference": int(sel.sum()),
                            "detected": int(yy[sel].sum()),
                            "share": float(yy[sel].mean()) if sel.any() else np.nan,
                        }
                    )
        a_, j_ = pd.factorize(s.author_id)[0], pd.factorize(s.journal_id.fillna("none"))[0]
        rng = np.random.default_rng(SEED_CHANGE + zlib.crc32((label + wn + COHORT[cohort][1]).encode()) % 1000)
        rngd = np.random.default_rng(SEED_CHANGE + 1)

        def rates(b):
            w = cell_weights(cell, t, b)
            return np.array([(w * y * (t == a)).sum() / (w * (t == a)).sum() for y, a in ((yb_, 1), (yb_, 0), (ya, 1), (ya, 0))])

        est = rates(np.ones(len(s)))
        R = np.array([rates(rng.exponential(1.0, a_.max() + 1)[a_] * rng.exponential(1.0, j_.max() + 1)[j_]) for _ in range(B)])
        corrs = {"none": (np.ones(4), np.ones((B, 4)))}
        if shares:
            order = [(1, "before"), (0, "before"), (1, "after"), (0, "after")]
            pt = np.array([shares[o][1] / shares[o][0] for o in order])
            dr = np.column_stack([rngd.binomial(shares[o][0], shares[o][1] / shares[o][0], B) / shares[o][0] for o in order])
            corrs["detection"] = (pt, dr)
        for corr, (pe, pr) in corrs.items():
            x, r = est / pe, R / pr
            ch, chr_ = (x[2] - x[3]) - (x[0] - x[1]), (r[:, 2] - r[:, 3]) - (r[:, 0] - r[:, 1])
            meas = {
                "rate treated, before": (x[0], r[:, 0]),
                "rate control, before": (x[1], r[:, 1]),
                "rate treated, after": (x[2], r[:, 2]),
                "rate control, after": (x[3], r[:, 3]),
                "difference before": (x[0] - x[1], r[:, 0] - r[:, 1]),
                "difference after": (x[2] - x[3], r[:, 2] - r[:, 3]),
                "change (after minus before)": (ch, chr_),
                "risk ratio before": (x[0] / x[1], r[:, 0] / r[:, 1]),
                "risk ratio after": (x[2] / x[3], r[:, 2] / r[:, 3]),
                "ratio of risk ratios (after / before)": ((x[2] / x[3]) / (x[0] / x[1]), (r[:, 2] / r[:, 3]) / (r[:, 0] / r[:, 1])),
                "change / control after-window rate": (ch / x[3], chr_ / r[:, 3]),
            }
            for mz, (pv, rv) in meas.items():
                lo_, hi_ = q(rv)
                rows.append(
                    {
                        "cc": cc,
                        "sample": label,
                        "cells": COHORT[cohort][0] + " A",
                        "comparator": COHORT[cohort][2],
                        "after_window": wn,
                        "correction": corr,
                        "measure": mz,
                        "estimate": pv,
                        "lo": lo_,
                        "hi": hi_,
                        "treated_trials": int(t.sum()),
                        "control_trials": int((1 - t).sum()),
                        "scientists": int(s.author_id.nunique()),
                    }
                )
    print(cc, label, cohort, "change", flush=True)
    return rows, det


def restriction():
    """US trials of 2017-2020: before window and months 12-35 (month A cells), beside first_paper.csv."""
    d, _, ev = load("US", "month")
    s = d[d.publication_year.between(2017, 2020)].reset_index(drop=True)
    rows, _ = change(
        "US", s.drop(columns=[c for c in s if c.startswith("ref_")]), ev, "US 2017-2020, last authors", "month", {"months 12..35": (12, 36)}
    )
    fp = pd.read_csv(RESULTS / "first_paper.csv").iloc[0]
    for x, y in (("difference after", "att_after"), ("difference before", "att_before"), ("change (after minus before)", "did")):
        rows.append(
            {
                "cc": "US",
                "sample": "first_paper.csv, index papers 2017-2020, no earlier top-journal paper",
                "cells": "index year",
                "comparator": "all other journals",
                "after_window": "months 12..35",
                "correction": "none",
                "measure": x,
                "estimate": fp[y] / 1000,
                "lo": fp[f"{y}_lo"] / 1000,
                "hi": fp[f"{y}_hi"] / 1000,
                "treated_trials": int(fp.treated),
                "control_trials": int(fp.controls),
                "scientists": int(fp.authors),
            }
        )
    save(rows, "event_validation.csv")


def main():
    out = {k: [] for k in ("curves", "horizons", "before", "sizes", "change", "detection")}
    for cc in ("US", "JP"):
        for cohort in COHORT:
            d, W, ev = load(cc, cohort)
            for label, (y1, adm) in CIF_SAMPLES[cc].items():
                if cohort == "neartop" and label != MAIN[cc]:
                    continue
                variants = ("A",) if cohort == "neartop" else (("B", "A") if label == MAIN[cc] else ("B",))
                m = d.publication_year.le(y1).to_numpy()
                for v in variants:
                    c_, h_, w_, s_ = cif(cc, d[m].reset_index(drop=True), W, ev, label, v, adm, cohort)
                    out["curves"] += c_
                    out["horizons"] += h_
                    out["before"] += w_
                    out["sizes"] += s_
            m = d.publication_year.le(CIF_SAMPLES[cc][MAIN[cc]][0]).to_numpy()
            ch, dt = change(cc, d[m].reset_index(drop=True), ev, MAIN[cc], cohort)
            out["change"] += ch
            out["detection"] += dt
            for k_, v_ in out.items():
                save(v_, f"event_{k_}.csv")
    restriction()


if __name__ == "__main__":
    main()
