"""Award rates over the percentile of citation standing (between and within scientists), straight-line and breakpoint
fits, and same-scientist pairs: all index papers, by field, with a scientist-only bootstrap, and against near-top
journals. -> results/curves.csv, hinge.csv, pairs.csv, hinge_<field>.csv, pairs_<field>.csv, pairs_scientist_boot.csv,
pairs_neartop.csv"""

import numpy as np
import pandas as pd

from analysis.common import BH, NBIN, YS, B, cuts, demean, load, pair_est, pairs, save, thin, wls

FIELDS = ("Health Sciences", "Life Sciences", "Physical Sciences", "Social Sciences")
QS = ("rd_bal", "rr_bal", "rd_first", "rd_second")


def merged_bins(t):
    """Merge adjacent low bins until each has >= 2,000 scientists with a paper in the bin and in another bin."""
    nb = t.groupby("a").bin.nunique()
    multi = t[t.a.isin(nb.index[nb >= 2])]
    cnt = multi.groupby("bin").a.nunique().reindex(range(NBIN), fill_value=0).to_numpy()
    lab, cur, acc = np.zeros(NBIN, int), 0, 0
    for b in range(NBIN):
        lab[b] = cur
        acc += cnt[b]
        if acc >= 2000 or b == NBIN - 1:
            cur, acc = cur + 1, 0
    return lab


def design(t, lab, years=True):
    g = lab[t.bin.to_numpy()]
    top = g.max()
    X = np.column_stack([(g == k).astype(float) for k in range(top)])
    if years:
        yr = pd.get_dummies(t.publication_year.astype(str), drop_first=True, dtype=float).to_numpy()
        X = np.hstack([X, yr])
    return X, top


def curve(t, y, grp, lab, w, years):
    X, top = design(t, lab, years)
    g = t[grp].to_numpy()
    Xd = demean(g, X, w)
    yd = demean(g, t[y].to_numpy(float)[:, None], w)[:, 0]
    return 1000 * wls(Xd, yd, w)[:top]


def hinge(t, y, grp, w, years, grid):
    """Straight-line slope and a breakpoint fit with the knot chosen by least squares (slopes per 1,000 per 0.1)."""
    x = t.pct.to_numpy(float)
    g = t[grp].to_numpy()
    yd = demean(g, t[y].to_numpy(float)[:, None], w)[:, 0]
    base = [x]
    if years:
        base += list(pd.get_dummies(t.publication_year.astype(str), drop_first=True, dtype=float).to_numpy().T)
    Bd = demean(g, np.column_stack(base), w)
    best = None
    for k in grid:
        h = demean(g, np.maximum(x - k, 0)[:, None], w)
        X = np.column_stack([Bd[:, :1], h, Bd[:, 1:]])
        b = wls(X, yd, w)
        ssr = (w * (yd - X @ b) ** 2).sum()
        if best is None or ssr < best[0]:
            best = (ssr, k, b[0], b[0] + b[1])
    lin = wls(Bd, yd, w)[0]
    return {"knot": best[1], "slope_below": 100 * best[2], "slope_above": 100 * best[3], "slope_linear": 100 * lin}


def run(d, dom=None):
    if dom is not None:
        d = d[d.dom_name.eq(dom)].reset_index(drop=True)
        for c in ("a", "j", "s"):
            d[c] = pd.factorize(d[c])[0]
    tag = dom or "all"
    sfx = "" if dom is None else "_" + dom.split()[0].lower()
    rng = np.random.default_rng(20261011)
    rows_c, rows_h, rows_p = [], [], []
    # between scientists: topic-year set fixed effects, all papers; same scientist: scientist fixed effects and index
    # year, papers at least 24 months apart
    t = thin(d)
    lab = merged_bins(t)
    grid = np.round(np.arange(0.5, 0.976, 0.025), 3)
    for k_, (frame, grp, yrs) in {"between": (d, "s", False), "within": (t, "a", True)}.items():
        for y in YS if k_ == "between" else ("y_post",):
            pt = curve(frame, y, grp, lab, np.ones(len(frame)), yrs)
            hp = hinge(frame, y, grp, np.ones(len(frame)), yrs, grid)
            rc, rh = [], []
            aa, jj = frame.a.to_numpy(), frame.j.to_numpy()
            for b in range(B):
                w = rng.exponential(1.0, aa.max() + 1)[aa] * rng.exponential(1.0, jj.max() + 1)[jj]
                rc.append(curve(frame, y, grp, lab, w, yrs))
                if b < BH:
                    rh.append(hinge(frame, y, grp, w, yrs, grid))
            rc = np.array(rc)
            for i in range(len(pt)):
                bins = np.flatnonzero(lab == i)
                rows_c.append(
                    {
                        "field": tag,
                        "comparison": k_,
                        "window": y,
                        "bin_group": i,
                        "pct_lo": bins.min() / NBIN,
                        "pct_hi": (bins.max() + 1) / NBIN,
                        "rd_vs_top": pt[i],
                        "lo": np.quantile(rc[:, i], 0.025),
                        "hi": np.quantile(rc[:, i], 0.975),
                        "papers": int(np.isin(frame.bin, bins).sum()),
                        "authors": int(frame.a[np.isin(frame.bin, bins)].nunique()),
                    }
                )
            rh = pd.DataFrame(rh)
            rows_h.append(
                {
                    "field": tag,
                    "comparison": k_,
                    "window": y,
                    **hp,
                    **{f"{c}_lo": rh[c].quantile(0.025) for c in rh},
                    **{f"{c}_hi": rh[c].quantile(0.975) for c in rh},
                    "papers": len(frame),
                    "authors": int(frame.a.nunique()),
                }
            )
    if dom is None:
        save(rows_c, "curves.csv")
    save(rows_h, f"hinge{sfx}.csv")
    # same-scientist pairs for each definition of a top journal
    tt = t if dom is not None else thin(d)
    aa, jj = d.a.to_numpy(), d.j.to_numpy()
    W = [(rng.exponential(1.0, aa.max() + 1), rng.exponential(1.0, jj.max() + 1)) for _ in range(B)]
    for c, A in cuts(d).items():
        A = A.astype(np.int8)
        P = pairs(d, A)
        Pall = pairs(d, A, min_gap=0)
        sets = {"pairs >=24 m": (P, "y_post", np.ones(len(P), bool))}
        if dom is None:
            sets.update(
                {
                    f"pairs >=24 m, {k}": (P, "y_post", m)
                    for k, m in {
                        **{f"L={v}": P.L.eq(v).to_numpy() for v in ("lt5", "5to10", "ge10")},
                        **{f"F={v}": P.F.eq(v).to_numpy() for v in (0, 1)},
                        "index years <=2020": (P.year2 <= 2020).to_numpy(),
                    }.items()
                }
            )
            sets["before window, pairs >=60 m"] = (P, "y_pre", (P.gap >= 60).to_numpy())
            sets["pairs <24 m (windows overlap)"] = (Pall, "y_post", (Pall.gap < 24).to_numpy())
        res = {}
        for k, (PP, y, m) in sets.items():
            Q = PP[m].reset_index(drop=True)
            e = pair_est(Q, d, y)
            if e is None:
                continue
            reps = pd.DataFrame([r for r in (pair_est(Q, d, y, wa, wj) for wa, wj in W) if r is not None])
            res[k] = (e, reps)
            row = {"field": tag, "definition": c, "set": k, "window": y, **e}
            for q in QS:
                row[f"{q}_lo"], row[f"{q}_hi"] = reps[q].quantile(0.025), reps[q].quantile(0.975)
            rows_p.append(row)
        if dom is None:
            e1, r1 = res["pairs >=24 m, L=lt5"]
            e2, r2 = res["pairs >=24 m, L=ge10"]
            n = min(len(r1), len(r2))
            diff = r1.rd_bal.to_numpy()[:n] - r2.rd_bal.to_numpy()[:n]
            rows_p.append(
                {
                    "field": tag,
                    "definition": c,
                    "set": "difference: L<5 minus L>=10",
                    "window": "y_post",
                    "rd_bal": e1["rd_bal"] - e2["rd_bal"],
                    "rd_bal_lo": np.quantile(diff, 0.025),
                    "rd_bal_hi": np.quantile(diff, 0.975),
                }
            )
            # scientist fixed effects and index year on papers at least 24 months apart, binary definition
            At = (cuts(tt)[c]).astype(float)
            X = np.column_stack([At, pd.get_dummies(tt.publication_year.astype(str), drop_first=True, dtype=float).to_numpy()])
            g = tt.a.to_numpy()

            def fe(w):
                return 1000 * wls(demean(g, X, w), demean(g, tt["y_post"].to_numpy(float)[:, None], w)[:, 0], w)[0]

            b0 = fe(np.ones(len(tt)))
            ta, tj = tt.a.to_numpy(), tt.j.to_numpy()
            rb = [fe(rng.exponential(1.0, ta.max() + 1)[ta] * rng.exponential(1.0, tj.max() + 1)[tj]) for _ in range(min(B, 200))]
            rows_p.append(
                {
                    "field": tag,
                    "definition": c,
                    "set": "fixed effects, papers >=24 m apart",
                    "window": "y_post",
                    "rd_bal": b0,
                    "rd_bal_lo": np.quantile(rb, 0.025),
                    "rd_bal_hi": np.quantile(rb, 0.975),
                    "n_first": len(tt),
                    "n_second": int(tt.a.nunique()),
                }
            )
    save(rows_p, f"pairs{sfx}.csv")


def scientist_boot(d):
    """Same-scientist pairs with the bootstrap resampling scientists only."""
    rng = np.random.default_rng(20261012)
    rows = []
    for c, A in cuts(d).items():
        P = pairs(d, A.astype(np.int8))
        e = pair_est(P, d, "y_post")
        reps = pd.DataFrame([pair_est(P, d, "y_post", rng.exponential(1.0, d.a.max() + 1), None) for _ in range(B)])
        row = {"definition": c, "set": "pairs >=24 m, scientist-only bootstrap", "window": "y_post", **e}
        for q in ("rd_bal", "rr_bal"):
            row[f"{q}_lo"], row[f"{q}_hi"] = reps[q].quantile(0.025), reps[q].quantile(0.975)
        rows.append(row)
    save(rows, "pairs_scientist_boot.csv")


def neartop(d):
    """Same-scientist pairs, top 5% against near-top journals."""
    rng = np.random.default_rng(20261015)
    rows = []
    for lab, lo in {"near-top 0.85-0.95": 0.85, "near-top 0.90-0.95": 0.90}.items():
        s = d[d.pct >= lo].reset_index(drop=True)
        for c in ("a", "j"):
            s[c] = pd.factorize(s[c])[0]
        A = (s.pct >= 0.95).to_numpy().astype(np.int8)
        P = pairs(s, A)
        W = [(rng.exponential(1.0, s.a.max() + 1), rng.exponential(1.0, s.j.max() + 1)) for _ in range(B)]
        sets = {
            "pairs >=24 m": ("y_post", np.ones(len(P), bool)),
            **{f"pairs >=24 m, L={v}": ("y_post", P.L.eq(v).to_numpy()) for v in ("lt5", "5to10", "ge10")},
            "pairs >=24 m, F=0": ("y_post", P.F.eq(0).to_numpy()),
            "before window, pairs >=60 m": ("y_pre", (P.gap >= 60).to_numpy()),
        }
        for k, (y, m) in sets.items():
            Q = P[m].reset_index(drop=True)
            e = pair_est(Q, s, y)
            if e is None:
                continue
            reps = pd.DataFrame([r for r in (pair_est(Q, s, y, wa, wj) for wa, wj in W) if r is not None])
            row = {"definition": lab, "set": k, "window": y, **e}
            for q in QS:
                row[f"{q}_lo"], row[f"{q}_hi"] = reps[q].quantile(0.025), reps[q].quantile(0.975)
            rows.append(row)
    save(rows, "pairs_neartop.csv")


if __name__ == "__main__":
    import sys

    d = load()
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    if part in ("all", "main"):
        run(d)
    if part in ("all", "fields"):
        for dom in FIELDS:
            run(d, dom)
    if part in ("all", "boot"):
        scientist_boot(d)
    if part in ("all", "neartop"):
        neartop(d)
