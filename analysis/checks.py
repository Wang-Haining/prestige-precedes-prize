"""Further checks, top 5% of journals: other windows, months 0 to <36, pairs by months apart and the profile of the two
order groups, pairs standardized over cells, linkage recall and detection of the binary outcome by group.
-> results/timing.csv, cumulative.csv, pair_checks.csv, standardized_order.csv, recall_by_group.csv, detection.csv"""

import numpy as np
import pandas as pd

from analysis.common import B, load, pair_est, pairs, save, std_gap

COMPARATORS = (("vs other journals", 0.0), ("vs near-top journals", 0.85))


def subset(d, lo):
    s = d[d.pct >= lo].reset_index(drop=True)
    A = (s.pct >= 0.95).to_numpy()
    si = pd.factorize(s.s)[0]
    aa, jj = pd.factorize(s.a)[0], pd.factorize(s.j)[0]
    return s, A, si, aa, jj


def timing(d):
    rng = np.random.default_rng(20261016)
    rows = []
    ys = {
        "months -24 to <-12": "w2412",
        "months -12 to <0": "w1200",
        "months -24 to <0": "y_pre",
        "months 0 to <12": "w0012",
        "months +12 to <36": "y_post",
    }
    for comp, lo in COMPARATORS:
        s, A, si, aa, jj = subset(d, lo)
        est = {k: std_gap(si, A, s[y].to_numpy(float), np.ones(len(s))) for k, y in ys.items()}
        reps = {k: [] for k in ys}
        for _ in range(B):
            w = rng.exponential(1.0, aa.max() + 1)[aa] * rng.exponential(1.0, jj.max() + 1)[jj]
            for k, y in ys.items():
                reps[k].append(std_gap(si, A, s[y].to_numpy(float), w)[0])
        for k in ys:
            rows.append(
                {
                    "comparison": "between scientists",
                    "contrast": comp,
                    "window": k,
                    "estimate": est[k][0],
                    "lo": np.quantile(reps[k], 0.025),
                    "hi": np.quantile(reps[k], 0.975),
                    "rate_top": est[k][2],
                    "rate_other": est[k][3],
                }
            )
        # same scientist: the window of the later paper must not overlap the after window of the earlier paper
        for c in ("a", "j"):
            s[c] = pd.factorize(s[c])[0]
        P = pairs(s, A.astype(np.int8))
        W = [(rng.exponential(1.0, s.a.max() + 1), rng.exponential(1.0, s.j.max() + 1)) for _ in range(B)]
        for k, y, gap in (
            ("months -24 to <-12", "w2412", 60),
            ("months -12 to <0", "w1200", 48),
            ("months -24 to <0", "y_pre", 60),
            ("months 0 to <12", "w0012", 36),
        ):
            Q = P[P.gap >= gap].reset_index(drop=True)
            e = pair_est(Q, s, y)
            r = pd.DataFrame([x for x in (pair_est(Q, s, y, wa, wj) for wa, wj in W) if x is not None])
            rows.append(
                {
                    "comparison": f"same scientist, pairs >= {gap} months apart",
                    "contrast": comp,
                    "window": k,
                    "estimate": e["rd_bal"],
                    "lo": r.rd_bal.quantile(0.025),
                    "hi": r.rd_bal.quantile(0.975),
                    "rate_top": e["rate_top"],
                    "rate_other": e["rate_non"],
                    "pairs": len(Q),
                }
            )
    save(rows, "timing.csv")


def cumulative(d):
    """Any new lead award in months 0 to <36: between scientists and same scientist (pairs >= 36 months apart)."""
    rng = np.random.default_rng(20261019)
    rows = []
    for comp, lo in COMPARATORS:
        s, A, si, aa, jj = subset(d, lo)
        e = std_gap(si, A, s["w0036"].to_numpy(float), np.ones(len(s)))
        reps = [
            std_gap(si, A, s["w0036"].to_numpy(float), rng.exponential(1.0, aa.max() + 1)[aa] * rng.exponential(1.0, jj.max() + 1)[jj])[0]
            for _ in range(B)
        ]
        rows.append(
            {
                "comparison": "between scientists",
                "contrast": comp,
                "window": "months 0 to <36",
                "estimate": e[0],
                "lo": np.quantile(reps, 0.025),
                "hi": np.quantile(reps, 0.975),
                "rate_top": e[2],
                "rate_other": e[3],
            }
        )
        for c in ("a", "j"):
            s[c] = pd.factorize(s[c])[0]
        P = pairs(s, A.astype(np.int8), min_gap=36)
        W = [(rng.exponential(1.0, s.a.max() + 1), rng.exponential(1.0, s.j.max() + 1)) for _ in range(B)]
        e = pair_est(P, s, "w0036")
        r = pd.DataFrame([x for x in (pair_est(P, s, "w0036", wa, wj) for wa, wj in W) if x is not None])
        rows.append(
            {
                "comparison": "same scientist, pairs >= 36 months apart",
                "contrast": comp,
                "window": "months 0 to <36",
                "estimate": e["rd_bal"],
                "lo": r.rd_bal.quantile(0.025),
                "hi": r.rd_bal.quantile(0.975),
                "rate_top": e["rate_top"],
                "rate_other": e["rate_non"],
                "pairs": len(P),
            }
        )
    save(rows, "cumulative.csv")


def pair_checks(d):
    rng = np.random.default_rng(20261017)
    rows = []
    for comp, lo in COMPARATORS:
        s = d[d.pct >= lo].reset_index(drop=True)
        for c in ("a", "j"):
            s[c] = pd.factorize(s[c])[0]
        A = (s.pct >= 0.95).to_numpy().astype(np.int8)
        P = pairs(s, A)
        W = [(rng.exponential(1.0, s.a.max() + 1), rng.exponential(1.0, s.j.max() + 1)) for _ in range(B)]
        for k, m in {"months apart 24 to <48": (P.gap < 48).to_numpy(), "months apart >=48": (P.gap >= 48).to_numpy()}.items():
            Q = P[m].reset_index(drop=True)
            e = pair_est(Q, s, "y_post")
            r = pd.DataFrame([x for x in (pair_est(Q, s, "y_post", wa, wj) for wa, wj in W) if x is not None])
            rows.append(
                {
                    "part": "stratum",
                    "contrast": comp,
                    "stratum": k,
                    "estimate": e["rd_bal"],
                    "lo": r.rd_bal.quantile(0.025),
                    "hi": r.rd_bal.quantile(0.975),
                    "n_first": e["n_first"],
                    "n_second": e["n_second"],
                }
            )
        for o, m in (("top journal first", P["first"].to_numpy()), ("top journal second", ~P["first"].to_numpy())):
            q = P[m]
            rows.append(
                {
                    "part": "profile",
                    "contrast": comp,
                    "stratum": o,
                    "pairs": int(m.sum()),
                    "share_lt5": float(q.L.eq("lt5").mean()),
                    "share_ge10": float(q.L.eq("ge10").mean()),
                    "share_award_60m": float(q.F.mean()),
                    "months_apart_median": float(q.gap.median()),
                    "first_year_mean": float(q.year1.mean()),
                }
            )
    save(rows, "pair_checks.csv")


def standardized_order(d):
    """Order-specific differences within cells of earlier-paper year x later-paper year x field x years as last author
    x prior award, averaged over the two orders within each cell, standardized to the pooled distribution of pairs
    over cells with both orders."""
    rng = np.random.default_rng(20261020)
    rows = []
    for comp, lo in COMPARATORS:
        s = d[d.pct >= lo].reset_index(drop=True)
        for c in ("a", "j"):
            s[c] = pd.factorize(s[c])[0]
        A = (s.pct >= 0.95).to_numpy().astype(np.int8)
        P = pairs(s, A)
        cell = pd.factorize(pd.MultiIndex.from_frame(P[["year1", "year2", "dom", "L", "F"]].astype(str)))[0]
        Y = s["y_post"].to_numpy(float)
        a_, j_ = s.a.to_numpy(), s.j.to_numpy()
        first = P["first"].to_numpy()
        t, n = P.top.to_numpy(), P.non.to_numpy()

        def est(wa=None, wj=None):
            wt = np.ones(len(P)) if wa is None else wa[a_[t]] * wj[j_[t]]
            wn = np.ones(len(P)) if wa is None else wa[a_[n]] * wj[j_[n]]
            k = cell.max() + 1
            out = []
            for o in (first, ~first):
                st, sn = np.bincount(cell, wt * Y[t] * o, k), np.bincount(cell, wn * Y[n] * o, k)
                nt, nn = np.bincount(cell, wt * o, k), np.bincount(cell, wn * o, k)
                out.append((st, sn, nt, nn))
            ok = (out[0][2] > 0) & (out[1][2] > 0)
            rd = 0.5 * sum(1000 * (st[ok] / nt[ok] - sn[ok] / nn[ok]) for st, sn, nt, nn in out)
            w = np.bincount(cell, minlength=k)[ok].astype(float)
            return (w * rd).sum() / w.sum(), w.sum() / len(P)

        e, kept = est()
        reps = [est(rng.exponential(1.0, a_.max() + 1), rng.exponential(1.0, j_.max() + 1))[0] for _ in range(B)]
        rows.append(
            {
                "contrast": comp,
                "estimate": e,
                "lo": np.quantile(reps, 0.025),
                "hi": np.quantile(reps, 0.975),
                "pairs": len(P),
                "share_of_pairs_in_cells_with_both_orders": kept,
            }
        )
    save(rows, "standardized_order.csv")


def groups(d):
    """Analysis groups for recall and detection."""
    top = d.pct.ge(0.95).to_numpy()
    near = d.pct.between(0.85, 0.95, inclusive="left").to_numpy()
    ft = pd.to_datetime(d.first_top_date)
    ordinary = (ft.isna() | (ft >= pd.to_datetime(d.publication_date))).to_numpy()
    coh = d.publication_year.between(2017, 2020).to_numpy()
    P = pairs(d, top.astype(np.int8))
    return top, near, ordinary, coh, P


def recall(d):
    """Share of acknowledged own awards (months -24 to <36) that the linkage attached to the index paper."""
    top, near, ordinary, coh, P = groups(d)
    rows = []

    def add(label, m):
        x = d[m]
        rows.append(
            {"group": label, "reference_awards": int(x.ref_awards.sum()), "recall": float(x.ref_linked.sum() / max(x.ref_awards.sum(), 1))}
        )

    add("all papers: top 5%", top)
    add("all papers: other journals", ~top)
    add("all papers: near-top journals", near)
    for v, nm in (("lt5", "<5"), ("5to10", "5 to <10"), ("ge10", ">=10")):
        add(f"years as last author {nm}: top 5%", top & d.L.eq(v).to_numpy())
        add(f"years as last author {nm}: other journals", ~top & d.L.eq(v).to_numpy())
    for o, m in (("top journal first", P["first"].to_numpy()), ("top journal second", ~P["first"].to_numpy())):
        for role, col in (("top-journal paper", "top"), ("other paper", "non")):
            mm = np.zeros(len(d), bool)
            mm[P[col].to_numpy()[m]] = True
            add(f"same-scientist pairs, {o}: {role}", mm)
    add("first top-journal paper (2017-2020, no earlier top-journal paper)", coh & ordinary & top)
    add("comparison papers, other journals (2017-2020, no earlier top-journal paper)", coh & ordinary & ~top)
    add("comparison papers, near-top journals (2017-2020, no earlier top-journal paper)", coh & ordinary & near)
    save(rows, "recall_by_group.csv")


def detection(d):
    """Among index paper-windows with at least one acknowledged own award, the share in which the linkage records a new
    lead award."""
    top, near, ordinary, coh, P = groups(d)
    inpair_top, inpair_non = np.zeros(len(d), bool), np.zeros(len(d), bool)
    inpair_top[P.top.to_numpy()] = True
    inpair_non[P.non.to_numpy()] = True
    gs = {
        "all papers: top 5%": top,
        "all papers: other journals": ~top,
        "all papers: near-top journals": near,
        "same-scientist pairs: top-journal paper": inpair_top,
        "same-scientist pairs: other paper": inpair_non,
        "first top-journal paper (2017-2020)": coh & ordinary & top,
        "comparison papers, other journals (2017-2020)": coh & ordinary & ~top,
        "comparison papers, near-top journals (2017-2020)": coh & ordinary & near,
    }
    rows = []
    for g, m in gs.items():
        for w, rc, yc in (("before window", "ref_pre", "y_pre"), ("after window", "ref_post", "y_post")):
            r = m & (d[rc].to_numpy() == 1)
            rows.append(
                {
                    "group": g,
                    "window": w,
                    "paper_windows_with_reference": int(r.sum()),
                    "detected": float(d[yc].to_numpy()[r].mean()) if r.any() else np.nan,
                }
            )
    save(rows, "detection.csv")


if __name__ == "__main__":
    d = load()
    timing(d)
    cumulative(d)
    pair_checks(d)
    standardized_order(d)
    recall(d)
    detection(d)
